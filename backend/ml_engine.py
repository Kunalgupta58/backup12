import os
import io
import sys
import tempfile
import torch
import torchaudio
import numpy as np
import time
import random
import logging
from typing import Callable, Any
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

# Monkeypatch for torchaudio >= 2.1.0 compatibility with SpeechBrain
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']
if not hasattr(torchaudio, 'get_audio_backend'):
    torchaudio.get_audio_backend = lambda: 'soundfile'
if not hasattr(torchaudio, 'set_audio_backend'):
    torchaudio.set_audio_backend = lambda x: None


def with_retry(max_retries: int = 5, base_delay: float = 1.0, jitter: bool = True):
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e).upper()
                    if "503" in error_str or "CAPACITY_EXHAUSTED" in error_str:
                        retries += 1
                        if retries >= max_retries:
                            logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}. Error: {e}")
                            raise
                        delay = base_delay * (2 ** (retries - 1))
                        if jitter:
                            delay += random.uniform(0, 0.1 * delay)
                        logger.warning(f"Retry {retries}/{max_retries} for {func.__name__} after {delay:.2f}s.")
                        time.sleep(delay)
                    else:
                        raise
            return None
        return wrapper
    return decorator


from huggingface_hub import snapshot_download
try:
    from speechbrain.inference import EncoderClassifier
except ImportError:
    from speechbrain.pretrained import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy
from backend.config import BASE_DIR
from backend.models import User

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss-cpu not installed. Falling back to Torch vector search.")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MLEngine:
    def __init__(self):
        self.classifier = None
        self.embedding_dim = 192
        self.index = None
        self.user_ids = []
        self.username_map = {}
        self.torch_embeddings_list = []   # always initialized — fixes AttributeError on fresh deploy
        self.torch_embeddings = None

        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.embedding_dim)

        logger.info("MLEngine initialized (model not yet loaded).")

    def is_ready(self) -> bool:
        return self.classifier is not None

    def initialize(self):
        if self.classifier:
            return
        logger.info("Starting ML model initialization...")
        try:
            self._load_model()
            logger.info("ML model successfully initialized.")
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            self.classifier = None

    @with_retry(max_retries=5)
    def _load_model(self):
        model_dir = Path(BASE_DIR).parent / "pretrained_models" / "spkrec-ecapa-voxceleb"
        os.makedirs(model_dir, exist_ok=True)

        run_opts = {"device": DEVICE}
        required_files = {
            "hyperparams.yaml",
            "embedding_model.ckpt",
            "mean_var_norm_emb.ckpt",
            "classifier.ckpt",
            "label_encoder.txt",
        }

        try:
            existing = {p.name for p in model_dir.iterdir()}
        except StopIteration:
            existing = set()

        has_local_model = required_files.issubset(existing)

        if not has_local_model:
            logger.info("Downloading SpeechBrain model from Hugging Face...")
            snapshot_download(
                repo_id="speechbrain/spkrec-ecapa-voxceleb",
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
            )
        else:
            logger.info("Using cached local SpeechBrain model.")

        hyperparams_path = model_dir / "hyperparams.yaml"
        local_hyperparams = hyperparams_path.read_text(encoding="utf-8").replace(
            "pretrained_path: speechbrain/spkrec-ecapa-voxceleb",
            f'pretrained_path: "{model_dir.as_posix()}"',
        )

        temp_hparams_file = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", encoding="utf-8", delete=False
            ) as tmp:
                tmp.write(local_hyperparams)
                temp_hparams_file = tmp.name

            self.classifier = EncoderClassifier.from_hparams(
                source=str(model_dir),
                savedir=str(model_dir),
                hparams_file=temp_hparams_file,
                run_opts=run_opts,
                local_strategy=LocalStrategy.COPY,
            )
        finally:
            if temp_hparams_file and os.path.exists(temp_hparams_file):
                os.unlink(temp_hparams_file)

        self.classifier.eval()

    def _ensure_initialized(self):
        if not self.classifier:
            logger.info("Lazy-loading ML model on first request...")
            self.initialize()
            if not self.classifier:
                raise RuntimeError("ML model failed to initialize.")

    def extract_embedding(self, audio_path: str) -> torch.Tensor:
        self._ensure_initialized()
        from scipy.io import wavfile
        fs, y = wavfile.read(audio_path)
        return self._compute_embedding(fs, y)

    def extract_embedding_bytes(self, wav_bytes: bytes) -> torch.Tensor:
        self._ensure_initialized()
        from scipy.io import wavfile
        stream = io.BytesIO(wav_bytes)
        fs, y = wavfile.read(stream)
        return self._compute_embedding(fs, y)

    def _compute_embedding(self, fs, y) -> torch.Tensor:
        self._ensure_initialized()
        if y.dtype == np.int16:
            y = y.astype(np.float32) / 32768.0
        signal = torch.from_numpy(y).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            embeddings = self.classifier.encode_batch(signal)
            embeddings = embeddings.squeeze()
        return embeddings

    def compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        emb1 = torch.nn.functional.normalize(emb1.squeeze(), p=2, dim=0)
        emb2 = torch.nn.functional.normalize(emb2.squeeze(), p=2, dim=0)
        return float(torch.dot(emb1, emb2).item())

    def load_all_embeddings(self, db_session):
        users = db_session.query(User).all()

        if FAISS_AVAILABLE and self.index is not None:
            self.index.reset()

        self.user_ids = []
        self.username_map = {}
        self.torch_embeddings_list = []
        self.torch_embeddings = None

        if not users:
            logger.info("No users in DB — FAISS index is empty.")
            return

        embeddings_list = []
        for i, u in enumerate(users):
            emb = np.frombuffer(u.embedding, dtype=np.float32)
            if len(emb) != self.embedding_dim:
                logger.warning(f"Skipping {u.username}: embedding dim {len(emb)} != {self.embedding_dim}")
                continue
            emb_norm = emb / np.linalg.norm(emb)
            embeddings_list.append(emb_norm)
            self.user_ids.append(u.id)
            self.username_map[len(self.user_ids) - 1] = u.username
            self.torch_embeddings_list.append(torch.from_numpy(emb_norm))

        if not embeddings_list:
            return

        self.torch_embeddings = torch.stack(self.torch_embeddings_list).to(DEVICE)

        if FAISS_AVAILABLE and self.index is not None:
            matrix = np.vstack(embeddings_list).astype('float32')
            self.index.add(matrix)
            logger.info(f"Loaded {len(embeddings_list)} embeddings into FAISS.")
        else:
            logger.info(f"Loaded {len(embeddings_list)} embeddings into Torch memory.")

    def add_user_to_index(self, user_id: int, username: str, embedding: torch.Tensor):
        emb_np = embedding.cpu().numpy()
        emb_norm = emb_np / np.linalg.norm(emb_np)
        idx = len(self.user_ids)

        if FAISS_AVAILABLE and self.index is not None:
            self.index.add(np.array([emb_norm], dtype='float32'))

        self.torch_embeddings_list.append(torch.from_numpy(emb_norm))
        self.torch_embeddings = torch.stack(self.torch_embeddings_list).to(DEVICE)
        self.user_ids.append(user_id)
        self.username_map[idx] = username

    def search_index(self, query_emb: torch.Tensor, top_k: int = 1):
        if FAISS_AVAILABLE and self.index is not None and self.index.ntotal > 0:
            emb_np = query_emb.cpu().numpy()
            emb_norm = emb_np / np.linalg.norm(emb_np)
            distances, indices = self.index.search(np.array([emb_norm], dtype='float32'), top_k)
            best_idx = indices[0][0]
            best_score = distances[0][0]
            if best_idx != -1:
                return self.username_map[best_idx], float(best_score)

        if self.torch_embeddings is not None and len(self.torch_embeddings) > 0:
            q_norm = torch.nn.functional.normalize(query_emb.squeeze(), p=2, dim=0)
            similarities = torch.matmul(self.torch_embeddings, q_norm)
            best_idx = torch.argmax(similarities).item()
            return self.username_map[best_idx], float(similarities[best_idx].item())

        return None, -1.0


ml_engine = MLEngine()

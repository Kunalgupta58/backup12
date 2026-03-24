"""
Run this script once before building the Docker image to pre-download
the SpeechBrain ECAPA-TDNN model into pretrained_models/.

Usage:
    python download_model.py

The model files will be saved to:
    pretrained_models/spkrec-ecapa-voxceleb/
"""
import os
from huggingface_hub import snapshot_download

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "pretrained_models",
    "spkrec-ecapa-voxceleb",
)

print(f"Downloading SpeechBrain ECAPA-TDNN model to: {MODEL_DIR}")
snapshot_download(
    repo_id="speechbrain/spkrec-ecapa-voxceleb",
    local_dir=MODEL_DIR,
    local_dir_use_symlinks=False,
)
print("Download complete. You can now build the Docker image.")

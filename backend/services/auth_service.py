import asyncio
import logging
import re
import time
from typing import Optional

import numpy as np
import torch
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from ..auth import create_access_token
from ..audio_utils import convert_audio_bytes_to_wav_bytes, check_liveness_heuristic_bytes
from ..ml_engine import ml_engine
from ..models import User

logger = logging.getLogger(__name__)

# Confidence threshold
# Cosine similarity is 0.0-1.0, multiplied by 100 for display.
# Users must score >= 80.00% to receive a JWT token.
MATCH_THRESHOLD = 0.80
CONFIDENCE_THRESHOLD_PCT = MATCH_THRESHOLD * 100


def get_risk_level(is_live: bool, similarity: float) -> str:
    if not is_live:
        return "High (Spoofing Detected)"
    if similarity >= 0.93:
        return "Low"
    if similarity >= MATCH_THRESHOLD:
        return "Medium"
    return "High (Low confidence)"


async def register_user_service(
    username: str, audio_samples: list[bytes], db: Session
):
    """
    Accepts multiple audio samples, extracts embeddings from valid samples,
    and stores the normalized average as the user's voiceprint.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]{3,32}", username):
        raise HTTPException(
            status_code=400,
            detail="Invalid username. Use 3-32 characters: letters, numbers, underscore, or hyphen.",
        )

    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already taken",
        )

    if ml_engine.classifier is None:
        raise HTTPException(
            status_code=500, detail="ML model is not loaded properly."
        )

    start_time = time.time()
    collected_embeddings: list[torch.Tensor] = []
    best_liveness_score = 0.0

    for idx, audio_bytes in enumerate(audio_samples):
        sample_label = f"[{username}] Sample {idx + 1}"
        if not audio_bytes:
            logger.warning("%s Empty audio payload. Skipping sample.", sample_label)
            continue
        t1 = time.time()

        wav_bytes = await asyncio.to_thread(
            convert_audio_bytes_to_wav_bytes, audio_bytes
        )
        if not wav_bytes:
            logger.warning("%s Audio conversion failed. Skipping sample.", sample_label)
            continue
        logger.info(
            "%s Audio conversion took: %.3fs", sample_label, time.time() - t1
        )
        t2 = time.time()

        liveness_data = check_liveness_heuristic_bytes(wav_bytes)
        if not liveness_data["is_live"]:
            logger.warning(
                "%s Liveness failed: %s. Skipping.",
                sample_label,
                liveness_data["reason"],
            )
            continue
        logger.info(
            "%s Liveness check took: %.3fs", sample_label, time.time() - t2
        )
        best_liveness_score = max(best_liveness_score, liveness_data["score"])
        t3 = time.time()

        emb = await asyncio.to_thread(ml_engine.extract_embedding_bytes, wav_bytes)
        logger.info(
            "%s Embedding extraction took: %.3fs", sample_label, time.time() - t3
        )
        collected_embeddings.append(emb)

    if not collected_embeddings:
        raise HTTPException(
            status_code=400,
            detail=(
                "All audio samples failed liveness or conversion. "
                "Please re-record in a quiet environment."
            ),
        )

    stacked = torch.stack(collected_embeddings, dim=0)
    mean_emb = stacked.mean(dim=0)
    mean_emb = torch.nn.functional.normalize(mean_emb, p=2, dim=0)
    emb_bytes_array = mean_emb.numpy().tobytes()

    logger.info(
        "[%s] Built voiceprint from %s/%s samples. Total time: %.3fs",
        username,
        len(collected_embeddings),
        len(audio_samples),
        time.time() - start_time,
    )

    new_user = User(
        username=username,
        embedding=emb_bytes_array,
        registration_score=best_liveness_score,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    if hasattr(ml_engine, "add_user_to_index"):
        ml_engine.add_user_to_index(new_user.id, new_user.username, mean_emb)

    return {
        "status": "success",
        "message": f"Voice profile created from {len(collected_embeddings)} samples",
        "username": username,
    }


async def login_user_service(
    audio_bytes: bytes, username: Optional[str], db: Session
):
    start_time = time.time()
    t1 = time.time()

    wav_bytes = await asyncio.to_thread(convert_audio_bytes_to_wav_bytes, audio_bytes)
    if not wav_bytes:
        raise HTTPException(status_code=500, detail="Audio conversion failed")

    t2 = time.time()
    logger.info("[LOGIN] Audio conversion (In-Memory) took: %.3fs", t2 - t1)

    liveness_data = check_liveness_heuristic_bytes(wav_bytes)
    if not liveness_data["is_live"]:
        logger.warning("Login liveness failed. Reason: %s", liveness_data["reason"])
        raise HTTPException(
            status_code=401,
            detail=f"Anti-spoofing alert: {liveness_data['reason']}",
        )

    t3 = time.time()
    logger.info("[LOGIN] Liveness check (In-Memory) took: %.3fs", t3 - t2)

    best_similarity: float
    target_username: str

    if username:
        user = db.query(User).filter(User.username == username).first()
        if not user:
            login_emb = await asyncio.to_thread(
                ml_engine.extract_embedding_bytes, wav_bytes
            )
            if hasattr(ml_engine, "search_index"):
                _ = ml_engine.search_index(login_emb, top_k=1)
            time.sleep(np.random.uniform(0.01, 0.05))
            raise HTTPException(status_code=404, detail="User not found")

        login_emb = await asyncio.to_thread(ml_engine.extract_embedding_bytes, wav_bytes)
        stored_emb = torch.from_numpy(np.frombuffer(user.embedding, dtype=np.float32))
        best_similarity = ml_engine.compute_similarity(login_emb, stored_emb)
        target_username = username
    else:
        login_emb = await asyncio.to_thread(ml_engine.extract_embedding_bytes, wav_bytes)

        if not hasattr(ml_engine, "search_index"):
            raise HTTPException(
                status_code=500, detail="ML search index not configured."
            )

        faiss_username, faiss_score = ml_engine.search_index(login_emb, top_k=1)
        if not faiss_username:
            raise HTTPException(status_code=404, detail="No matching user found")

        best_similarity = faiss_score
        target_username = faiss_username

    t4 = time.time()
    logger.info("[LOGIN] Database/Similarity check took: %.3fs", t4 - t3)

    confidence_pct = round(best_similarity * 100, 2)

    if best_similarity >= MATCH_THRESHOLD:
        logger.info(
            "Login granted for '%s'. Confidence: %.2f%% (threshold: %.0f%%). Total time: %.3fs",
            target_username,
            confidence_pct,
            CONFIDENCE_THRESHOLD_PCT,
            t4 - start_time,
        )
        access_token = create_access_token(data={"sub": target_username})
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "confidence": confidence_pct,
            "risk_level": get_risk_level(liveness_data["is_live"], best_similarity),
            "username": target_username,
        }

    logger.warning(
        "Login denied for '%s'. Confidence: %.2f%% (need >= %.0f%%).",
        target_username,
        confidence_pct,
        CONFIDENCE_THRESHOLD_PCT,
    )
    raise HTTPException(
        status_code=401,
        detail=(
            f"Voice authentication failed. Confidence score {confidence_pct:.2f}% "
            f"is below the {CONFIDENCE_THRESHOLD_PCT:.0f}% security threshold. "
            "Please try again with a clearer recording."
        ),
    )

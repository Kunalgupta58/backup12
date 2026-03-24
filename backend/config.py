import os
from dotenv import load_dotenv

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load .env from project root (only applies locally — Render injects env vars directly)
load_dotenv(os.path.join(BASE_DIR, "..", ".env"))


def _normalize_database_url(url: str) -> str:
    """Render emits postgres:// — SQLAlchemy 1.4+ requires postgresql://"""
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://"):]
    return url


# Database — priority: DATABASE_URL > SUPABASE_DB_URL > local SQLite fallback
DATABASE_URL = _normalize_database_url(
    os.getenv("DATABASE_URL")
    or os.getenv("SUPABASE_DB_URL")
    or "sqlite:///./voice_auth.db"
)

# JWT
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    import secrets
    import logging
    logging.getLogger(__name__).warning(
        "SECRET_KEY env var not set — generating a random key. "
        "All existing JWTs will be invalidated on restart."
    )
    SECRET_KEY = secrets.token_hex(32)

ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# HuggingFace / SpeechBrain — disable symlinks (required on Windows, harmless on Linux)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# Numba cache to a writable tmp dir
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp")

# Temporary audio processing folder
TEMP_AUDIO_DIR = os.path.join(BASE_DIR, "temp_audio")
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

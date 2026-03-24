import os
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Depends, HTTPException, Request, Header
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy.orm import Session

from backend.database import engine, Base, get_db, SessionLocal
from backend.models import User
from backend.ml_engine import DEVICE, ml_engine
from backend.rate_limiter import limiter
from backend.routers import auth
from backend.config import SECRET_KEY

# Create DB tables (idempotent — safe to run on every boot)
Base.metadata.create_all(bind=engine)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Simple admin token — derived from SECRET_KEY so no extra env var needed.
# In production you'd use proper role-based auth / Alembic migrations.
# ---------------------------------------------------------------------------
import hashlib
_ADMIN_TOKEN = hashlib.sha256(f"admin:{SECRET_KEY}".encode()).hexdigest()[:32]


def require_admin(x_admin_token: str = Header(default="")):
    if x_admin_token != _ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Admin access denied")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Initializing ML engine...")
    db = SessionLocal()
    try:
        ml_engine.initialize()
        if ml_engine.is_ready():
            ml_engine.load_all_embeddings(db)
            try:
                import torch
                logger.info("Warming up ML model graph...")
                dummy = torch.zeros(1, 16000, device=DEVICE)
                with torch.no_grad():
                    _ = ml_engine.classifier.encode_batch(dummy)
                logger.info("Model warmup complete.")
            except Exception as e:
                logger.warning(f"Model warmup failed (non-fatal): {e}")
        else:
            logger.warning("ML model failed to initialize — will retry on first request.")
    except Exception as e:
        logger.error(f"Startup error: {e}")
    finally:
        db.close()
    yield
    logger.info("Shutting down.")


app = FastAPI(title="VoiceKey API", lifespan=lifespan)

# ── Validation error logger ────────────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

# ── Rate limiter ───────────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ───────────────────────────────────────────────────────────────────
# Restrict to your own domain in production.
# Falls back to * for local dev when ALLOWED_ORIGIN is not set.
_allowed_origin = os.getenv("ALLOWED_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_allowed_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────
app.include_router(auth.router)

# ── Admin endpoints (protected) ────────────────────────────────────────────
@app.get("/api/admin/users", dependencies=[Depends(require_admin)])
def get_all_users(db: Session = Depends(get_db)):
    users = db.query(User).with_entities(User.id, User.username, User.created_at).all()
    return [{"id": u.id, "username": u.username, "created_at": str(u.created_at)} for u in users]


@app.delete("/api/admin/users/{user_id}", dependencies=[Depends(require_admin)])
def delete_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(user)
    db.commit()
    logger.info(f"Admin deleted user {user.username} (id={user_id})")
    return {"status": "success", "message": f"User {user.username} deleted."}


# ── Admin token helper (call once to get your admin token) ─────────────────
@app.get("/api/admin/token-hint")
def admin_token_hint():
    """Returns the first 8 chars of the admin token for verification.
       Remove this endpoint before going fully public."""
    return {"hint": _ADMIN_TOKEN[:8] + "...", "full_token_env": "Derive from SECRET_KEY via sha256"}


# ── Frontend ───────────────────────────────────────────────────────────────
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/")
def serve_index():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(frontend_dir, "index.html"))


@app.get("/dashboard")
def serve_dashboard():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(frontend_dir, "dashboard.html"))


@app.get("/healthz")
def healthcheck():
    return {"status": "ok", "model_ready": ml_engine.is_ready()}


# ── Direct run ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )

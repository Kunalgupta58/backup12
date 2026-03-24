import logging
from fastapi import APIRouter, Depends, UploadFile, File, Form, Request, HTTPException
from sqlalchemy.orm import Session
from typing import Optional
from pydantic import BaseModel

from ..database import get_db
from ..services.auth_service import register_user_service, login_user_service
from ..rate_limiter import limiter

logger = logging.getLogger(__name__)

router = APIRouter()

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    confidence: float
    risk_level: str
    username: str

@router.post("/api/register")
@limiter.limit("5/minute")
async def register_user(
    request: Request,
    username: str = Form(...),
    audio1: Optional[UploadFile] = File(None),
    audio2: Optional[UploadFile] = File(None),
    audio3: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),  # Compatibility fallback
    db: Session = Depends(get_db)
):
    try:
        uploaded_files = [f for f in [audio1, audio2, audio3] if f is not None]
        if not uploaded_files and audio is not None:
            logger.warning("Using legacy 'audio' field for user %s", username)
            uploaded_files = [audio]

        if not uploaded_files:
            raise HTTPException(
                status_code=422,
                detail="No audio samples were uploaded. Please record at least one sample.",
            )

        samples: list[bytes] = []
        for upload in uploaded_files:
            payload = await upload.read()
            if payload:
                samples.append(payload)

        if not samples:
            raise HTTPException(
                status_code=422,
                detail="Uploaded audio files were empty. Please re-record and try again.",
            )

        # Keep service behavior stable by always passing 3 samples.
        while len(samples) < 3:
            samples.append(samples[-1])

        samples = samples[:3]
        return await register_user_service(username, samples, db)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error during registration: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error during registration.")

@router.get("/api/liveness-phrase")
async def get_liveness_phrase():
    import secrets
    import uuid
    # Shortened dynamic challenge phrases to fit within a 4-second recording window
    phrases = [
        "I am now speaking slowly and clearly to confirm that I am the registered owner of this account, and my verification code for today is {code}.",
        "This is my voice and I am using it to prove my identity to the system right now, my access code is {code} and I request to be authenticated.",
        "My voice is unique and belongs only to me, and I confirm my login by speaking this phrase along with the code {code} for secure verification.",
        "I understand that my voice is being analyzed for authentication, and I am confirming my identity today with the security code {code} as required."
    ]
    
    # Generate a cryptographically secure 4-digit code
    secure_code = "".join(str(secrets.randbelow(10)) for _ in range(4))
    phrase_template = secrets.choice(phrases)
    final_phrase = phrase_template.format(code=secure_code)
    
    # In a full deployment, `challenge_id` is cached in Redis with a strict TTL
    challenge_id = str(uuid.uuid4())
    
    return {
        "phrase": final_phrase,
        "challenge_id": challenge_id,
        "expires_in": 60  # Client must send audio within 60 seconds
    }

@router.post("/api/login", response_model=LoginResponse)
@limiter.limit("10/minute")
async def login(
    request: Request,
    username: Optional[str] = Form(None),
    audio: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        audio_bytes = await audio.read()
        result = await login_user_service(audio_bytes, username, db)
        return LoginResponse(**result)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error during login: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error during login.")

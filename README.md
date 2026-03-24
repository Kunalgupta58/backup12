# VoiceKey

VoiceKey is a full-stack biometric voice authentication app with a FastAPI backend, a vanilla HTML/CSS/JavaScript frontend, and a SpeechBrain ECAPA-TDNN speaker-embedding model. Users enroll with three voice recordings, then authenticate with a fresh recording that is checked for basic liveness before speaker matching.

## What The Repository Contains

- `frontend/`: browser UI for registration, login, and the dashboard.
- `backend/`: API routes, auth logic, database access, audio processing, and ML inference.
- `pretrained_models/spkrec-ecapa-voxceleb/`: local SpeechBrain model assets used for speaker embedding extraction.
- `start_localhost.ps1` and `start_localhost.bat`: local startup helpers.
- `Dockerfile`: containerized deployment option.

## How The App Works

1. The frontend records microphone audio with the `MediaRecorder` API.
2. Registration collects three 15-second samples and posts them to `POST /api/register`.
3. Login records one 10-second sample and posts it to `POST /api/login`.
4. The backend converts uploaded audio to 16 kHz mono WAV.
5. A heuristic liveness check rejects clips that are too short, too quiet, or have abnormal ZCR/SNR patterns.
6. SpeechBrain extracts a 192-dimensional embedding from each accepted sample.
7. Registration averages and normalizes the three embeddings, then stores the result as raw bytes in the database.
8. Login either:
   - compares the live embedding against a specific username, or
   - searches all known users when no username is supplied.
9. If cosine similarity is at least `0.80`, the backend returns a JWT plus confidence and risk metadata.

## Main Components

### Frontend

- `frontend/index.html`: login and enrollment UI.
- `frontend/dashboard.html`: post-login view plus a basic admin user list.
- `frontend/js/app.js`: recording flow, API calls, localStorage handling, and dashboard actions.

### Backend

- `backend/main.py`: FastAPI app bootstrap, startup lifecycle, static hosting, and admin endpoints.
- `backend/routers/auth.py`: `/api/register`, `/api/login`, and `/api/liveness-phrase`.
- `backend/services/auth_service.py`: registration and login business logic.
- `backend/audio_utils.py`: audio conversion and heuristic liveness checks.
- `backend/ml_engine.py`: model loading, embedding extraction, similarity scoring, and FAISS search.
- `backend/database.py`: SQLAlchemy engine and session management.
- `backend/models.py`: `users` table definition.
- `backend/auth.py`: JWT helper functions.
- `backend/config.py`: environment loading and runtime configuration.

## Data And Storage

- Default database: local SQLite at `voice_auth.db`.
- Optional hosted database: set `SUPABASE_DB_URL` or `DATABASE_URL` to use Postgres.
- Stored user data:
  - `id`
  - `username`
  - `embedding` as raw float32 bytes
  - `registration_score`
- Raw audio is processed in memory and is not persisted as the biometric record.

## Environment Variables

The app reads `.env` from the project root. Example values are in `.env.example`.

```env
SUPABASE_DB_URL=postgresql://postgres:YOUR_PASSWORD@db.YOUR_PROJECT_REF.supabase.co:5432/postgres?sslmode=require
SECRET_KEY=change-me-before-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

Notes:

- If no Postgres URL is set, the app falls back to SQLite automatically.
- `postgres://` URLs are normalized to `postgresql://`.
- Hugging Face symlink warnings are disabled for Windows compatibility in code.

## GitHub And Deploy Checklist

Before pushing this project:

- keep `.env` local and never commit it
- use `.env.example` as the public config template
- commit source files, startup scripts, Dockerfile, and README
- do not commit `venv/`, `__pycache__/`, logs, or `backend/temp_audio/`

This repository now includes `.gitignore` and `.dockerignore` files to enforce that structure.

## Local Setup

### Requirements

- Python 3.10+ recommended
- FFmpeg installed and available on `PATH`
- A working virtual environment

### Install

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Then update `.env` with your real secrets and optional database connection.

### Run Locally

Recommended:

```powershell
.\start_localhost.ps1
```

Alternative:

```powershell
.\venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Or:

```powershell
python backend/main.py
```

Open `http://127.0.0.1:8000`.

Important:

- Do not run with `--reload` on Windows for this project.
- The startup path loads the ML model, warms it up, and loads saved embeddings into memory.

## Docker

Build:

```bash
docker build -t voicekey .
```

Run:

```bash
docker run -p 8000:8000 --env-file .env voicekey
```

The container installs `ffmpeg` and `libsndfile1`, then starts Gunicorn with Uvicorn workers.

## Render Deployment

This project is set up to deploy well on Render as a single Docker web service.

Recommended approach:

1. Push this repository to GitHub.
2. In Render, create a new Web Service from the repo.
3. Let Render detect the `Dockerfile`, or deploy via the included `render.yaml` blueprint.
4. Add a Postgres database in Render and connect its `DATABASE_URL` to the web service.
5. Set a strong `SECRET_KEY` in Render.
6. Deploy and wait for the initial model load to finish.

Important for Render:

- Do not rely on the SQLite fallback in production because Render's local filesystem is ephemeral.
- The service now binds to Render's dynamic `PORT`.
- Default worker count is `1` to avoid loading the SpeechBrain model multiple times and exhausting memory.
- Health check path: `/healthz`

Minimum required environment variables on Render:

```env
DATABASE_URL=<your Render Postgres Internal Database URL>
SECRET_KEY=<a long random secret>
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
WEB_CONCURRENCY=1
```

Notes:

- `DATABASE_URL` is preferred on Render. The app also accepts `SUPABASE_DB_URL`.
- If your Render instance is memory-constrained, keep `WEB_CONCURRENCY=1`.
- First boot can be slower because the ML engine warms up on startup.

## API Summary

- `POST /api/register`
  - form fields: `username`, `audio1`, `audio2`, `audio3`
  - accepts one legacy `audio` field as fallback
- `POST /api/login`
  - form fields: `audio`, optional `username`
- `GET /api/liveness-phrase`
  - returns a challenge phrase, challenge ID, and expiry
- `GET /api/admin/users`
  - lists registered users
- `DELETE /api/admin/users/{user_id}`
  - deletes a user from the database

Swagger docs are available at `http://127.0.0.1:8000/docs`.

## Matching Behavior

- Enrollment uses three samples and averages their embeddings.
- Login threshold is `80%` cosine similarity.
- Providing a username performs closed-set verification.
- Omitting the username performs open-set identification across all known users.
- FAISS is used for fast search when available; otherwise a Torch-based fallback is used.

## Important Operational Notes

- Audio conversion prefers in-memory decoding and falls back to `pydub` when needed.
- The pretrained model is reused from `pretrained_models/spkrec-ecapa-voxceleb/` if present; otherwise it is downloaded from Hugging Face.
- The backend serves the frontend directly, so you only need one server process for local use.
- `start_localhost.ps1` checks whether port `8000` is already occupied before starting.

## Current Limitations

- The liveness check is heuristic only and should not be treated as production-grade anti-spoofing.
- Admin endpoints are exposed without JWT enforcement in the current code.
- The dashboard trusts `localStorage` for session display.
- Deleting a user removes the database row, but in-memory search state is only fully rebuilt on restart.
- The repository contains generated folders such as `venv/`, `__pycache__/`, and temporary audio artifacts that are not part of the core application logic.

## Troubleshooting

- If audio conversion fails, verify that FFmpeg is installed and reachable from the shell.
- If model loading fails on first run, confirm internet access for Hugging Face download or ensure the pretrained files already exist locally.
- If the wrong Python interpreter is being used, point your IDE to `venv\Scripts\python.exe`.
- If port `8000` is busy, stop the existing process or use the provided startup script to detect the conflict early.

## Tech Stack

- FastAPI
- SQLAlchemy
- SQLite or Postgres/Supabase
- SpeechBrain
- PyTorch and Torchaudio
- FAISS
- Pydub, SoundFile, and Librosa
- Vanilla HTML, CSS, and JavaScript

# backup

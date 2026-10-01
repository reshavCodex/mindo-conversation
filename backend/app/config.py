from pathlib import Path
import os

from dotenv import load_dotenv


# ============================================================
# PATH CONFIGURATION
# ============================================================

# Current directory:
# D:\MINDO\conversation\backend\app

APP_DIR = Path(__file__).resolve().parent

# Backend directory:
# D:\MINDO\conversation\backend

BACKEND_DIR = APP_DIR.parent

# .env file:
# D:\MINDO\conversation\backend\.env

ENV_FILE = BACKEND_DIR / ".env"


# ============================================================
# LOAD .ENV
# ============================================================

load_dotenv(dotenv_path=ENV_FILE)


# ============================================================
# GEMINI API KEY
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


if not GEMINI_API_KEY:

    raise RuntimeError(
        "GEMINI_API_KEY is not set.\n"
        f"Expected .env file at:\n{ENV_FILE}"
    )


# ============================================================
# GEMINI LIVE MODEL
# ============================================================

GEMINI_LIVE_MODEL = "gemini-3.8-live"


# ============================================================
# APPLICATION
# ============================================================

APP_NAME = "MINDO AI Realtime Backend"

APP_VERSION = "1.0.0"


# ============================================================
# AUDIO
# ============================================================

AUDIO_SAMPLE_RATE = 16000

AUDIO_CHANNELS = 1

AUDIO_FORMAT = "audio/pcm"


# ============================================================
# VIDEO
# ============================================================

VIDEO_FORMAT = "image/jpeg"

VIDEO_FPS = 1


# ============================================================
# SERVER
# ============================================================

# Render can provide these through environment variables.
# Local development keeps the existing behavior.

HOST = os.getenv(
    "HOST",
    "127.0.0.1",
)

PORT = int(
    os.getenv(
        "PORT",
        "8000",
    )
)


# ============================================================
# CORS
# ============================================================

# Local development origins are retained by default.
#
# Production frontend origins can be supplied through:
#
# ALLOWED_ORIGINS=https://your-frontend.pages.dev
#
# Multiple origins can be separated with commas:
#
# ALLOWED_ORIGINS=https://example.pages.dev,https://mindo.example.com

_default_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5500",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5500",
]

_allowed_origins_env = os.getenv("ALLOWED_ORIGINS")

if _allowed_origins_env:
    ALLOWED_ORIGINS = [
        origin.strip()
        for origin in _allowed_origins_env.split(",")
        if origin.strip()
    ]
else:
    ALLOWED_ORIGINS = _default_origins


# ============================================================
# DEBUG INFORMATION
# ============================================================

print("=" * 60)
print("MINDO AI REALTIME BACKEND")
print("=" * 60)

print(f"App directory  : {APP_DIR}")
print(f"Backend        : {BACKEND_DIR}")
print(f".env path      : {ENV_FILE}")
print(f".env exists    : {ENV_FILE.exists()}")
print(f"API key loaded : {bool(GEMINI_API_KEY)}")
print(f"Gemini model   : {GEMINI_LIVE_MODEL}")
print(f"Host           : {HOST}")
print(f"Port           : {PORT}")
print(f"Allowed origins: {ALLOWED_ORIGINS}")

print("=" * 60)
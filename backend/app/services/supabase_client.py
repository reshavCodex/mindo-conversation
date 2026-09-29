import os

from dotenv import load_dotenv
from supabase import Client, create_client


# ================================================================
# ENVIRONMENT
# ================================================================

load_dotenv()


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")


# ================================================================
# VALIDATION
# ================================================================

if not SUPABASE_URL:
    raise RuntimeError(
        "SUPABASE_URL is not configured in the backend .env file."
    )

if not SUPABASE_SECRET_KEY:
    raise RuntimeError(
        "SUPABASE_SECRET_KEY is not configured in the backend .env file."
    )


# ================================================================
# CLIENT
# ================================================================

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_SECRET_KEY,
)
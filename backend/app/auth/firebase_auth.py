import json
import os
from pathlib import Path

import firebase_admin
from dotenv import load_dotenv
from firebase_admin import auth, credentials


# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

# Backend directory:
# D:\MINDO\conversation\backend

BACKEND_DIR = Path(__file__).resolve().parents[2]

load_dotenv(BACKEND_DIR / ".env")


# ============================================================
# FIREBASE SERVICE ACCOUNT
# ============================================================

# Local development can continue using the existing
# firebase-service-account.json file.
#
# Production can provide the complete service-account JSON
# through the FIREBASE_SERVICE_ACCOUNT_JSON environment
# variable instead of storing the credential file in Git.

SERVICE_ACCOUNT_PATH = os.getenv(
    "FIREBASE_SERVICE_ACCOUNT_PATH",
    str(BACKEND_DIR / "firebase-service-account.json"),
)

SERVICE_ACCOUNT_JSON = os.getenv(
    "FIREBASE_SERVICE_ACCOUNT_JSON"
)


# ============================================================
# FIREBASE INITIALIZATION
# ============================================================

def initialize_firebase() -> None:
    """Initialize Firebase Admin SDK once."""

    if firebase_admin._apps:
        return

    # --------------------------------------------------------
    # PRODUCTION: SERVICE ACCOUNT FROM ENVIRONMENT VARIABLE
    # --------------------------------------------------------

    if SERVICE_ACCOUNT_JSON:

        try:
            service_account_info = json.loads(
                SERVICE_ACCOUNT_JSON
            )

        except json.JSONDecodeError as exc:

            raise RuntimeError(
                "FIREBASE_SERVICE_ACCOUNT_JSON contains invalid JSON."
            ) from exc

        credential = credentials.Certificate(
            service_account_info
        )

        firebase_admin.initialize_app(
            credential
        )

        return

    # --------------------------------------------------------
    # LOCAL DEVELOPMENT: SERVICE ACCOUNT FILE
    # --------------------------------------------------------

    service_account = Path(
        SERVICE_ACCOUNT_PATH
    )

    if not service_account.exists():

        raise FileNotFoundError(
            "Firebase service-account credentials were not found.\n"
            f"Expected local file at: {service_account}\n"
            "Or set FIREBASE_SERVICE_ACCOUNT_JSON "
            "for environment-based credentials."
        )

    credential = credentials.Certificate(
        str(service_account)
    )

    firebase_admin.initialize_app(
        credential
    )


# ============================================================
# FIREBASE TOKEN VERIFICATION
# ============================================================

def verify_firebase_token(id_token: str) -> dict:
    """
    Verify a Firebase ID token and return its decoded claims.

    Raises Firebase Admin SDK exceptions if the token is invalid.
    """

    initialize_firebase()

    return auth.verify_id_token(
        id_token
    )
from datetime import datetime, timezone
from typing import Any

from app.services.supabase_client import supabase


# ================================================================
# HELPERS
# ================================================================

def utc_now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


# ================================================================
# USERS
# ================================================================

def get_or_create_user(firebase_uid: str) -> dict:
    """
    Find the MINDO user associated with a Firebase UID.

    If the user does not exist yet, create them.

    Firebase UID is the external identity.
    Supabase users.id is the internal database identity.
    """

    if not firebase_uid:
        raise ValueError(
            "Firebase UID is required."
        )

    # ------------------------------------------------------------
    # Try to find existing user.
    # ------------------------------------------------------------

    result = (
        supabase
        .table("users")
        .select("*")
        .eq("firebase_uid", firebase_uid)
        .limit(1)
        .execute()
    )

    if result.data:
        return result.data[0]

    # ------------------------------------------------------------
    # Create new user.
    # ------------------------------------------------------------

    result = (
        supabase
        .table("users")
        .insert({
            "firebase_uid": firebase_uid,
        })
        .execute()
    )

    if not result.data:
        raise RuntimeError(
            "Failed to create MINDO user in Supabase."
        )

    return result.data[0]


# ================================================================
# SESSIONS
# ================================================================

def create_session(
    firebase_uid: str,
    session_id: str,
    started_at: str | None = None,
) -> dict:
    """
    Create a new database session owned by the authenticated
    Firebase user.

    The ConversationSession-generated session_id is preserved.
    """

    if not firebase_uid:
        raise ValueError(
            "Firebase UID is required."
        )

    if not session_id:
        raise ValueError(
            "Session ID is required."
        )

    user = get_or_create_user(
        firebase_uid
    )

    result = (
        supabase
        .table("sessions")
        .insert({
            "session_id": session_id,
            "user_id": user["id"],
            "started_at": started_at or utc_now_iso(),
            "status": "active",
        })
        .execute()
    )

    if not result.data:
        raise RuntimeError(
            "Failed to create MINDO session in Supabase."
        )

    return result.data[0]


def mark_session_processing(
    session_id: str,
    ended_at: str | None = None,
) -> dict:
    """
    Mark a session as processing after the live conversation ends.

    At this point JSON generation and RAG/report generation are
    taking place.
    """

    result = (
        supabase
        .table("sessions")
        .update({
            "status": "processing",
            "ended_at": ended_at or utc_now_iso(),
        })
        .eq("session_id", session_id)
        .execute()
    )

    if not result.data:
        raise RuntimeError(
            f"Session not found while marking processing: "
            f"{session_id}"
        )

    return result.data[0]


def mark_session_completed(
    session_id: str,
    rag_result: dict[str, Any],
) -> dict:
    """
    Mark a session as completed and store the RAG assessment
    metadata.

    The full PDF/JSON files are NOT stored in the database.
    Those live in Supabase Storage.

    Recommendations are stored separately as JSONB so the
    frontend can display the actual RAG-generated next steps
    directly on the Summary screen.
    """

    assessment = (
        rag_result.get("assessment")
        or {}
    )

    category = assessment.get(
        "category"
    )

    confidence = assessment.get(
        "confidence"
    )

    summary = assessment.get(
        "summary"
    )

    recommendations = assessment.get(
        "recommendations",
        [],
    )

    # ------------------------------------------------------------
    # Ensure recommendations are stored as a JSON array.
    # ------------------------------------------------------------

    if not isinstance(
        recommendations,
        list,
    ):
        recommendations = []

    # ------------------------------------------------------------
    # Store completed session metadata.
    # ------------------------------------------------------------

    result = (
        supabase
        .table("sessions")
        .update({
            "status": "completed",
            "assessment_category": category,
            "confidence": confidence,
            "summary": summary,
            "recommendations": recommendations,
        })
        .eq("session_id", session_id)
        .execute()
    )

    if not result.data:
        raise RuntimeError(
            f"Session not found while marking completed: "
            f"{session_id}"
        )

    return result.data[0]


def mark_session_failed(
    session_id: str,
) -> dict:
    """
    Mark a session as failed.

    Used when report generation or another post-session
    processing step fails.
    """

    result = (
        supabase
        .table("sessions")
        .update({
            "status": "failed",
        })
        .eq("session_id", session_id)
        .execute()
    )

    if not result.data:
        raise RuntimeError(
            f"Session not found while marking failed: "
            f"{session_id}"
        )

    return result.data[0]


# ================================================================
# SESSION LOOKUP
# ================================================================

def get_session_for_user(
    firebase_uid: str,
    session_id: str,
) -> dict | None:
    """
    Retrieve a session only if it belongs to the specified
    Firebase user.

    This ownership check is intentionally performed by the
    backend and never trusts a frontend-supplied user ID.
    """

    if not firebase_uid:
        raise ValueError(
            "Firebase UID is required."
        )

    user = get_or_create_user(
        firebase_uid
    )

    result = (
        supabase
        .table("sessions")
        .select("*")
        .eq("session_id", session_id)
        .eq("user_id", user["id"])
        .limit(1)
        .execute()
    )

    if not result.data:
        return None

    return result.data[0]


def get_user_sessions(
    firebase_uid: str,
    limit: int = 20,
) -> list[dict]:
    """
    Retrieve recent sessions belonging only to the authenticated
    Firebase user.
    """

    if not firebase_uid:
        raise ValueError(
            "Firebase UID is required."
        )

    if limit < 1:
        limit = 1

    if limit > 100:
        limit = 100

    user = get_or_create_user(
        firebase_uid
    )

    result = (
        supabase
        .table("sessions")
        .select("*")
        .eq("user_id", user["id"])
        .order(
            "created_at",
            desc=True,
        )
        .limit(limit)
        .execute()
    )

    return result.data or []


# ================================================================
# ARTIFACTS
# ================================================================

def register_artifact(
    session_id: str,
    artifact_type: str,
    storage_path: str,
) -> dict:
    """
    Register a JSON/PDF artifact belonging to a session.

    Actual files live in Supabase Storage.
    This table only stores their metadata/path.
    """

    allowed_types = {
        "session_context",
        "semantic_context",
        "report_pdf",
    }

    if artifact_type not in allowed_types:
        raise ValueError(
            f"Invalid artifact type: {artifact_type}"
        )

    if not storage_path:
        raise ValueError(
            "Storage path is required."
        )

    # ------------------------------------------------------------
    # Resolve the internal Supabase session ID.
    # ------------------------------------------------------------

    session_result = (
        supabase
        .table("sessions")
        .select("id")
        .eq("session_id", session_id)
        .limit(1)
        .execute()
    )

    if not session_result.data:
        raise RuntimeError(
            f"Session not found while registering artifact: "
            f"{session_id}"
        )

    database_session_id = (
        session_result.data[0]["id"]
    )

    # ------------------------------------------------------------
    # Upsert artifact metadata.
    #
    # The database has:
    #
    # unique (session_id, artifact_type)
    #
    # so each session can have exactly one current artifact
    # of each type.
    # ------------------------------------------------------------

    result = (
        supabase
        .table("session_artifacts")
        .upsert(
            {
                "session_id": database_session_id,
                "artifact_type": artifact_type,
                "storage_path": storage_path,
            },
            on_conflict="session_id,artifact_type",
        )
        .execute()
    )

    if not result.data:
        raise RuntimeError(
            f"Failed to register artifact: "
            f"{artifact_type}"
        )

    return result.data[0]


def get_session_artifacts(
    firebase_uid: str,
    session_id: str,
) -> list[dict]:
    """
    Retrieve artifact metadata only when the session belongs
    to the authenticated Firebase user.
    """

    session = get_session_for_user(
        firebase_uid,
        session_id,
    )

    if not session:
        return []

    result = (
        supabase
        .table("session_artifacts")
        .select("*")
        .eq("session_id", session["id"])
        .order(
            "created_at",
            desc=True,
        )
        .execute()
    )

    return result.data or []
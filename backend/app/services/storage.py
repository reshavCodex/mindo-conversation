from app.services.supabase_client import supabase

BUCKET_NAME = "mindo-sessions"


def build_storage_path(
    firebase_uid: str,
    session_id: str,
    artifact_type: str,
) -> str:
    filenames = {
        "session_context": "session_context.json",
        "semantic_context": "semantic_context.json",
        "report_pdf": "report.pdf",
    }

    if artifact_type not in filenames:
        raise ValueError(f"Invalid artifact type: {artifact_type}")

    if not firebase_uid:
        raise ValueError("Firebase UID is required.")

    if not session_id:
        raise ValueError("Session ID is required.")

    return f"{firebase_uid}/{session_id}/{filenames[artifact_type]}"


def upload_artifact(
    firebase_uid: str,
    session_id: str,
    artifact_type: str,
    file_bytes: bytes,
    content_type: str,
) -> str:
    if not file_bytes:
        raise ValueError(f"No data provided for {artifact_type}.")

    storage_path = build_storage_path(
        firebase_uid=firebase_uid,
        session_id=session_id,
        artifact_type=artifact_type,
    )

    supabase.storage.from_(BUCKET_NAME).upload(
        storage_path,
        file_bytes,
        {
            "content-type": content_type,
            "upsert": "true",
        },
    )

    return storage_path
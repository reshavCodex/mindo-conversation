from app.services.storage import upload_artifact


FIREBASE_UID = "storage-test-user"
SESSION_ID = "storage-test-session"

test_json = b'{"test": true, "source": "MINDO storage test"}'

path = upload_artifact(
    firebase_uid=FIREBASE_UID,
    session_id=SESSION_ID,
    artifact_type="session_context",
    file_bytes=test_json,
    content_type="application/json",
)

print("STORAGE UPLOAD SUCCESS")
print(f"Path: mindo-sessions/{path}")
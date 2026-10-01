import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.firebase_auth import verify_firebase_token
from app.services.session_store import (
    get_session_for_user,
    get_session_artifacts,
    get_user_sessions,
)
from app.services.supabase_client import supabase


router = APIRouter(
    prefix="/api/v1/sessions",
    tags=["sessions"],
)


security = HTTPBearer()


# ================================================================
# FIREBASE AUTHENTICATION
# ================================================================

def get_firebase_uid(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """
    Verify the Firebase ID token supplied in the Authorization header.

    Expected header:

        Authorization: Bearer <firebase-id-token>
    """

    try:
        decoded_token = verify_firebase_token(
            credentials.credentials
        )

        firebase_uid = decoded_token.get("uid")

        if not firebase_uid:
            raise HTTPException(
                status_code=401,
                detail="Invalid Firebase authentication token.",
            )

        return firebase_uid

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Authentication failed.",
        )


# ================================================================
# GET USER SESSION HISTORY
# ================================================================

@router.get("")
async def get_user_session_history(
    limit: int = 100,
    firebase_uid: str = Depends(get_firebase_uid),
):
    """
    Return recent sessions belonging only to the authenticated
    Firebase user.

    The Firebase UID is obtained from the verified Firebase ID
    token and is never supplied directly by the frontend.
    """

    try:
        sessions = get_user_sessions(
            firebase_uid=firebase_uid,
            limit=limit,
        )

        return {
            "sessions": sessions,
            "count": len(sessions),
        }

    except Exception as error:
        print(
            "[SESSIONS] Failed to retrieve user sessions:",
            repr(error),
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve your session history.",
        )


# ================================================================
# GET SESSION STATUS
# ================================================================

@router.get("/{session_id}")
async def get_session_status(
    session_id: str,
    firebase_uid: str = Depends(get_firebase_uid),
):
    """
    Return the authenticated user's session status and result.

    The session must belong to the authenticated Firebase user.
    """

    session = get_session_for_user(
        firebase_uid=firebase_uid,
        session_id=session_id,
    )

    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found.",
        )

    recommendations = session.get(
        "recommendations",
        [],
    )

    if not isinstance(
        recommendations,
        list,
    ):
        recommendations = []

    return {
        "session_id": session["session_id"],
        "status": session["status"],
        "started_at": session.get("started_at"),
        "ended_at": session.get("ended_at"),
        "assessment_category": session.get(
            "assessment_category"
        ),
        "confidence": session.get("confidence"),
        "summary": session.get("summary"),
        "recommendations": recommendations,
    }


# ================================================================
# GET SESSION FER / EMOTION DATA
# ================================================================

@router.get("/{session_id}/emotion-data")
async def get_session_emotion_data(
    session_id: str,
    firebase_uid: str = Depends(get_firebase_uid),
):
    """
    Retrieve FER data for a completed session.

    The raw session_context.json remains private in Supabase
    Storage. This endpoint verifies session ownership, downloads
    the existing session_context artifact, and returns only the
    FER-related information required by the frontend charts.

    No RAG data or conversation text is returned here.
    """

    # ------------------------------------------------------------
    # VERIFY SESSION OWNERSHIP
    # ------------------------------------------------------------

    session = get_session_for_user(
        firebase_uid=firebase_uid,
        session_id=session_id,
    )

    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found.",
        )

    # ------------------------------------------------------------
    # SESSION MUST BE COMPLETED
    # ------------------------------------------------------------

    if session.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail="Emotion data is not ready yet.",
        )

    # ------------------------------------------------------------
    # FIND SESSION ARTIFACTS
    # ------------------------------------------------------------

    artifacts = get_session_artifacts(
        firebase_uid=firebase_uid,
        session_id=session_id,
    )

    session_context_artifact = next(
        (
            artifact
            for artifact in artifacts
            if artifact.get("artifact_type")
            == "session_context"
        ),
        None,
    )

    if not session_context_artifact:
        raise HTTPException(
            status_code=404,
            detail="Session context artifact not found.",
        )

    storage_path = session_context_artifact.get(
        "storage_path"
    )

    if not storage_path:
        raise HTTPException(
            status_code=404,
            detail="Session context storage path not found.",
        )

    # ------------------------------------------------------------
    # DOWNLOAD SESSION CONTEXT FROM PRIVATE SUPABASE STORAGE
    # ------------------------------------------------------------

    try:
        session_context_bytes = (
            supabase.storage
            .from_("mindo-sessions")
            .download(storage_path)
        )

    except Exception as error:
        print(
            "[EMOTION DATA] Failed to download "
            "session context:",
            repr(error),
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve session emotion data.",
        )

    if not session_context_bytes:
        raise HTTPException(
            status_code=404,
            detail="Session context file is empty or unavailable.",
        )

    # ------------------------------------------------------------
    # PARSE SESSION CONTEXT JSON
    # ------------------------------------------------------------

    try:
        session_context = json.loads(
            session_context_bytes.decode("utf-8")
        )

    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        print(
            "[EMOTION DATA] Invalid session context JSON:",
            repr(error),
        )

        raise HTTPException(
            status_code=500,
            detail="The stored session emotion data is invalid.",
        )

    # ------------------------------------------------------------
    # EXTRACT FER TIMELINE
    # ------------------------------------------------------------

    conversation = session_context.get(
        "conversation",
        {}
    )

    turns = conversation.get(
        "turns",
        []
    )

    if not isinstance(turns, list):
        turns = []

    timeline = []

    for turn in turns:

        if not isinstance(turn, dict):
            continue

        turn_number = turn.get(
            "turn_number"
        )

        emotional_observations = turn.get(
            "emotional_observations",
            {}
        )

        if not isinstance(
            emotional_observations,
            dict,
        ):
            continue

        turn_timeline = emotional_observations.get(
            "timeline",
            []
        )

        if not isinstance(
            turn_timeline,
            list,
        ):
            continue

        for observation in turn_timeline:

            if not isinstance(
                observation,
                dict,
            ):
                continue

            probabilities = observation.get(
                "probabilities",
                {}
            )

            if not isinstance(
                probabilities,
                dict,
            ):
                continue

            timeline.append({
                "timestamp": observation.get(
                    "timestamp"
                ),
                "dominant_emotion": observation.get(
                    "dominant_emotion"
                ),
                "probabilities": probabilities,
                "assignment": observation.get(
                    "assignment"
                ),
                "turn_number": turn_number,
            })

    # ------------------------------------------------------------
    # ENSURE CHRONOLOGICAL ORDER
    # ------------------------------------------------------------

    timeline.sort(
        key=lambda item: (
            item.get("timestamp")
            if isinstance(
                item.get("timestamp"),
                (int, float),
            )
            else 0
        )
    )

    # ------------------------------------------------------------
    # OVERALL FER SUMMARY
    # ------------------------------------------------------------

    emotion_summary = session_context.get(
        "emotion_summary",
        {}
    )

    if not isinstance(
        emotion_summary,
        dict,
    ):
        emotion_summary = {}

    # ------------------------------------------------------------
    # RETURN FER DATA ONLY
    # ------------------------------------------------------------

    return JSONResponse(
        content={
            "session_id": session_id,
            "emotion_summary": emotion_summary,
            "timeline": timeline,
            "observation_count": len(timeline),
        }
    )


# ================================================================
# DOWNLOAD SESSION REPORT
# ================================================================

@router.get("/{session_id}/report")
async def download_session_report(
    session_id: str,
    firebase_uid: str = Depends(get_firebase_uid),
):
    """
    Download the authenticated user's generated PDF report.

    The PDF remains private in Supabase Storage.
    The backend verifies ownership before downloading it.
    """

    # ------------------------------------------------------------
    # VERIFY SESSION OWNERSHIP
    # ------------------------------------------------------------

    session = get_session_for_user(
        firebase_uid=firebase_uid,
        session_id=session_id,
    )

    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found.",
        )

    # ------------------------------------------------------------
    # SESSION MUST BE COMPLETED
    # ------------------------------------------------------------

    if session.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail="The report is not ready yet.",
        )

    # ------------------------------------------------------------
    # FIND SESSION ARTIFACTS
    # ------------------------------------------------------------

    artifacts = get_session_artifacts(
        firebase_uid=firebase_uid,
        session_id=session_id,
    )

    report_artifact = next(
        (
            artifact
            for artifact in artifacts
            if artifact.get("artifact_type") == "report_pdf"
        ),
        None,
    )

    if not report_artifact:
        raise HTTPException(
            status_code=404,
            detail="Report artifact not found.",
        )

    storage_path = report_artifact.get(
        "storage_path"
    )

    if not storage_path:
        raise HTTPException(
            status_code=404,
            detail="Report storage path not found.",
        )

    # ------------------------------------------------------------
    # DOWNLOAD FROM PRIVATE SUPABASE STORAGE
    # ------------------------------------------------------------

    try:
        report_bytes = (
            supabase.storage
            .from_("mindo-sessions")
            .download(storage_path)
        )

    except Exception as error:
        print(
            "[REPORT] Failed to download report:",
            repr(error),
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve the report.",
        )

    if not report_bytes:
        raise HTTPException(
            status_code=404,
            detail="Report file is empty or unavailable.",
        )

    # ------------------------------------------------------------
    # RETURN PDF TO BROWSER
    # ------------------------------------------------------------

    filename = (
        f"mindo_assessment_{session_id}.pdf"
    )

    return Response(
        content=report_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{filename}"'
            )
        },
    )
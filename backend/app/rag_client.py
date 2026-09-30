import asyncio
import os
from typing import Any
from urllib.parse import urlsplit

import httpx
from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# RAG SERVICE URL
# ============================================================

RAG_REPORT_URL = os.getenv(
    "RAG_REPORT_URL",
    "http://127.0.0.1:8001/api/v1/reports/generate",
)


# ============================================================
# RAG WAKE / RETRY CONFIGURATION
# ============================================================

# RAG may be asleep on Render Free. We first send a lightweight
# request to the RAG service root so Render has an opportunity
# to wake the service before we send the actual report payload.

WAKE_MAX_ATTEMPTS = 8

WAKE_RETRY_DELAYS = [
    5,
    10,
    15,
    20,
    25,
    30,
    30,
]


# ============================================================
# REPORT RETRY CONFIGURATION
# ============================================================

# These retries are kept as a safety net in case the RAG service
# becomes available between the wake check and the report POST.

REPORT_MAX_ATTEMPTS = 8

REPORT_RETRY_DELAYS = [
    5,
    10,
    15,
    20,
    25,
    30,
    30,
]


# ============================================================
# HTTP TIMEOUTS
# ============================================================

WAKE_TIMEOUT = httpx.Timeout(
    connect=30.0,
    read=30.0,
    write=30.0,
    pool=30.0,
)

REPORT_TIMEOUT = httpx.Timeout(
    connect=30.0,
    read=180.0,
    write=30.0,
    pool=30.0,
)


# ============================================================
# RETRYABLE HTTP STATUS CODES
# ============================================================

RETRYABLE_STATUS_CODES = {
    502,
    503,
    504,
}


# ============================================================
# RAG ROOT URL
# ============================================================

def _get_rag_root_url() -> str:
    """
    Derive the RAG service root URL from RAG_REPORT_URL.

    Example:

        https://mindo-rag.onrender.com/api/v1/reports/generate

    becomes:

        https://mindo-rag.onrender.com/
    """

    parsed = urlsplit(RAG_REPORT_URL)

    if not parsed.scheme or not parsed.netloc:
        raise ValueError(
            f"Invalid RAG_REPORT_URL: {RAG_REPORT_URL}"
        )

    return f"{parsed.scheme}://{parsed.netloc}/"


RAG_ROOT_URL = _get_rag_root_url()


# ============================================================
# WAKE / HEALTH CHECK
# ============================================================

async def _ensure_rag_available(
    client: httpx.AsyncClient,
) -> None:
    """
    Make sure the RAG FastAPI service is reachable before
    sending the actual report-generation request.

    The root endpoint does not need to return 200.

    A 404 is acceptable because it proves that the FastAPI
    application is alive and handling the request.

    Render gateway errors and connection failures are retried
    because they may indicate that the service is still waking.
    """

    print(
        f"[RAG] Checking/waking RAG service → "
        f"{RAG_ROOT_URL}"
    )

    last_error: Exception | None = None

    for attempt in range(1, WAKE_MAX_ATTEMPTS + 1):

        print(
            f"[RAG] Wake attempt "
            f"{attempt}/{WAKE_MAX_ATTEMPTS}"
        )

        try:
            response = await client.get(
                RAG_ROOT_URL,
                timeout=WAKE_TIMEOUT,
            )

            # ------------------------------------------------
            # Any non-gateway response proves that the
            # application is reachable.
            #
            # This includes 404 because the RAG root route
            # does not need to exist.
            # ------------------------------------------------

            if response.status_code not in RETRYABLE_STATUS_CODES:
                print(
                    f"[RAG] RAG service is reachable "
                    f"(HTTP {response.status_code})."
                )

                return

            # ------------------------------------------------
            # Render may still be waking the service.
            # ------------------------------------------------

            last_error = httpx.HTTPStatusError(
                f"RAG wake returned HTTP "
                f"{response.status_code}",
                request=response.request,
                response=response,
            )

            print(
                f"[RAG] Wake request returned HTTP "
                f"{response.status_code}. "
                f"RAG may still be starting."
            )

        except (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
            httpx.RemoteProtocolError,
        ) as exc:

            last_error = exc

            print(
                f"[RAG] Wake connection error: "
                f"{type(exc).__name__}: {exc}"
            )

            print(
                "[RAG] RAG may still be waking."
            )

        # ----------------------------------------------------
        # Retry wake request
        # ----------------------------------------------------

        if attempt < WAKE_MAX_ATTEMPTS:

            delay = WAKE_RETRY_DELAYS[
                min(
                    attempt - 1,
                    len(WAKE_RETRY_DELAYS) - 1,
                )
            ]

            print(
                f"[RAG] Waiting {delay} seconds "
                f"before wake retry..."
            )

            await asyncio.sleep(delay)

    # ========================================================
    # Wake attempts exhausted
    # ========================================================

    print(
        "[RAG] Could not confirm that the RAG service "
        f"is reachable after {WAKE_MAX_ATTEMPTS} attempts."
    )

    if last_error is not None:
        raise last_error

    raise RuntimeError(
        "RAG service could not be reached."
    )


# ============================================================
# REPORT GENERATION
# ============================================================

async def generate_report(
    semantic_context: dict[str, Any],
) -> dict[str, Any]:
    """
    Send semantic session context to the RAG backend
    and request PDF report generation.

    The RAG backend returns the generated PDF as base64
    data so the Conversation backend does not depend on
    the RAG service's local filesystem.

    Before generating the report, the RAG service is explicitly
    checked/woken. This allows the Render Free instance to start
    when it has previously been spun down due to inactivity.
    """

    if not semantic_context:
        raise ValueError(
            "Semantic context cannot be empty."
        )

    async with httpx.AsyncClient() as client:

        # ====================================================
        # PHASE 1: ENSURE RAG IS AWAKE / AVAILABLE
        # ====================================================

        await _ensure_rag_available(client)

        # ====================================================
        # PHASE 2: GENERATE REPORT
        # ====================================================

        last_error: Exception | None = None

        for attempt in range(
            1,
            REPORT_MAX_ATTEMPTS + 1,
        ):

            print(
                f"[RAG] Report generation attempt "
                f"{attempt}/{REPORT_MAX_ATTEMPTS} → "
                f"{RAG_REPORT_URL}"
            )

            try:
                response = await client.post(
                    RAG_REPORT_URL,
                    json=semantic_context,
                    timeout=REPORT_TIMEOUT,
                )

                # ------------------------------------------------
                # Successful response
                # ------------------------------------------------

                if response.is_success:

                    print(
                        "[RAG] Report generation successful."
                    )

                    return response.json()

                # ------------------------------------------------
                # Retryable HTTP errors
                # ------------------------------------------------

                if response.status_code in RETRYABLE_STATUS_CODES:

                    last_error = httpx.HTTPStatusError(
                        f"RAG returned retryable HTTP "
                        f"status {response.status_code}",
                        request=response.request,
                        response=response,
                    )

                    print(
                        f"[RAG] Report request returned HTTP "
                        f"{response.status_code}. "
                        f"Retrying..."
                    )

                else:
                    # Non-transient HTTP errors should fail
                    # immediately.
                    response.raise_for_status()

                    # This line should normally never be reached.
                    return response.json()

            except (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadTimeout,
                httpx.WriteTimeout,
                httpx.PoolTimeout,
                httpx.RemoteProtocolError,
            ) as exc:

                last_error = exc

                print(
                    f"[RAG] Report connection error: "
                    f"{type(exc).__name__}: {exc}"
                )

                print(
                    "[RAG] Retrying report generation..."
                )

            # ----------------------------------------------------
            # Retry delay
            # ----------------------------------------------------

            if attempt < REPORT_MAX_ATTEMPTS:

                delay = REPORT_RETRY_DELAYS[
                    min(
                        attempt - 1,
                        len(REPORT_RETRY_DELAYS) - 1,
                    )
                ]

                print(
                    f"[RAG] Waiting {delay} seconds "
                    f"before report retry..."
                )

                await asyncio.sleep(delay)

        # ========================================================
        # All report attempts exhausted
        # ========================================================

        print(
            "[RAG] Report generation failed after "
            f"{REPORT_MAX_ATTEMPTS} attempts."
        )

        if last_error is not None:
            raise last_error

        raise RuntimeError(
            "RAG report generation failed without "
            "a captured exception."
        )
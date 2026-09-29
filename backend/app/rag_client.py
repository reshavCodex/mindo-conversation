import asyncio
import os
from typing import Any

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
# RETRY CONFIGURATION
# ============================================================

# RAG may be asleep on Render Free and need time to wake up,
# start FastAPI, load the knowledge base, and initialize the
# RAG pipeline before it can handle the report request.

MAX_ATTEMPTS = 8

RETRY_DELAYS = [
    5,
    10,
    15,
    20,
    25,
    30,
    30,
]

REQUEST_TIMEOUT = httpx.Timeout(
    connect=30.0,
    read=180.0,
    write=30.0,
    pool=30.0,
)

RETRYABLE_STATUS_CODES = {
    502,
    503,
    504,
}


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

    Because the RAG service may be sleeping on Render Free,
    transient connection failures and gateway errors are
    retried automatically until the RAG service becomes ready.
    """

    if not semantic_context:
        raise ValueError(
            "Semantic context cannot be empty."
        )

    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT
    ) as client:

        last_error: Exception | None = None

        for attempt in range(1, MAX_ATTEMPTS + 1):

            print(
                f"[RAG] Report generation attempt "
                f"{attempt}/{MAX_ATTEMPTS} → {RAG_REPORT_URL}"
            )

            try:
                response = await client.post(
                    RAG_REPORT_URL,
                    json=semantic_context,
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
                        f"[RAG] Received HTTP "
                        f"{response.status_code}. "
                        f"RAG may still be starting. "
                        f"Retrying..."
                    )

                else:
                    # Non-transient HTTP errors should fail
                    # immediately instead of being retried.
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
                    f"[RAG] Transient connection error: "
                    f"{type(exc).__name__}: {exc}"
                )

                print(
                    "[RAG] RAG may be waking up. "
                    "Retrying..."
                )

            # ----------------------------------------------------
            # Retry delay
            # ----------------------------------------------------

            if attempt < MAX_ATTEMPTS:

                delay = RETRY_DELAYS[
                    min(
                        attempt - 1,
                        len(RETRY_DELAYS) - 1,
                    )
                ]

                print(
                    f"[RAG] Waiting {delay} seconds "
                    f"before retry..."
                )

                await asyncio.sleep(delay)

        # ========================================================
        # All retry attempts exhausted
        # ========================================================

        print(
            "[RAG] Report generation failed after "
            f"{MAX_ATTEMPTS} attempts."
        )

        if last_error is not None:
            raise last_error

        raise RuntimeError(
            "RAG report generation failed without "
            "a captured exception."
        )
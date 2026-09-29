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
    """

    if not semantic_context:
        raise ValueError(
            "Semantic context cannot be empty."
        )

    async with httpx.AsyncClient(
        timeout=180.0
    ) as client:

        response = await client.post(
            RAG_REPORT_URL,
            json=semantic_context,
        )

        response.raise_for_status()

        return response.json()
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    APP_NAME,
    APP_VERSION,
    ALLOWED_ORIGINS,
)

from app.realtime.websocket import (
    realtime_conversation,
)

from app.api.session import (
    router as session_router,
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# API ROUTERS
# ============================================================

app.include_router(
    session_router,
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": APP_NAME,
        "version": APP_VERSION,
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():
    return {
        "status": "healthy",
    }


# ============================================================
# REAL-TIME WEBSOCKET
# ============================================================

@app.websocket("/ws/realtime")
async def realtime_websocket(
    websocket: WebSocket,
):
    print()
    print("=" * 60)
    print("INCOMING WEBSOCKET CONNECTION")
    print("=" * 60)

    print("Client:", websocket.client)
    print("Headers:", dict(websocket.headers))

    await realtime_conversation(
        websocket
    )
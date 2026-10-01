# MINDO Conversation Backend

> **Using the live app from a phone?** MINDO is not optimised for mobile screens yet, and visuals may look broken. Please use your browser's **Desktop site** option.

Realtime AI conversation backend for [MINDO](https://github.com/reshavCodex/mindo-frontend), an AI-assisted counselling and wellness platform for students and young adults.

**Public service:** https://mindo-conversation.onrender.com (Render free tier, may sleep when idle)

> To try the full app, open this URL once to wake the service (along with the [RAG](https://mindo-rag.onrender.com) and [chatbot](https://mindo-chat-backend.onrender.com) services), then visit https://mindo-frontend.vercel.app. This is only because of free-tier hosting.

---

## Purpose

This service runs MINDO's live counselling check-ins and owns everything around a session: authentication, the realtime conversation with Gemini Live, session records, report generation hand-off, and storage.

## Responsibilities

- FastAPI backend served with Uvicorn
- WebSocket endpoint for the live session
- Firebase ID token verification (Firebase Admin SDK)
- Gemini Live conversation (audio in, audio and transcriptions out)
- Receiving facial-emotion signals from the browser and keeping them separate from Gemini
- Building `session_context` and `semantic_context` after a session
- Calling the RAG Backend to generate the report
- Uploading session files and the PDF to Supabase and recording session status
- REST endpoints for session history, emotion chart data and report download

## Where It Sits in MINDO

```text
Frontend → Conversation Backend → Gemini Live
Conversation Backend → RAG Backend → PDF report (base64)
Conversation Backend → Supabase (database + storage)
Frontend → Conversation Backend (session history, emotion data, report)
```

The frontend talks to this service over the WebSocket and REST endpoints below. The RAG Backend is called only from here.

## Endpoints

| Type | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/` | none | Service status |
| GET | `/health` | none | Liveness check |
| WebSocket | `/ws/realtime` | Firebase token in first message | Live check-in |
| GET | `/api/v1/sessions` | Bearer token | Authenticated user's recent sessions (`limit` query, default 100) |
| GET | `/api/v1/sessions/{session_id}` | Bearer token | Session status and result (category, confidence, summary, recommendations) |
| GET | `/api/v1/sessions/{session_id}/emotion-data` | Bearer token | FER timeline for a completed session (no conversation text or RAG data) |
| GET | `/api/v1/sessions/{session_id}/report` | Bearer token | Download the session's PDF report |

REST calls expect `Authorization: Bearer <firebase-id-token>`. The Firebase UID comes from the verified token, never from the request. Sessions are looked up per user, so a user cannot read another user's session.

## WebSocket Protocol

The first message must authenticate:

```json
{ "type": "auth", "token": "<firebase-id-token>" }
```

| Direction | Message `type` | Meaning |
|---|---|---|
| Client → server | `auth` | Firebase ID token (must be first) |
| Client → server | `emotion` | `timestamp`, `dominant_emotion`, `probabilities` from browser FER |
| Client → server | `speech_start`, `speech_end` | Turn boundaries |
| Client → server | `end_session` | Finish the check-in and start report processing |
| Server → client | `auth_ok` | Authentication accepted |
| Server → client | `session_started` | Includes the new `session_id` |
| Server → client | `transcription`, `transcription_interim` | Speech transcripts |
| Server → client | `error` | Error details |

Audio is forwarded to Gemini Live (configured for 16 kHz mono PCM).

## How Gemini Is Used

`GeminiLiveSession` (`app/realtime/gemini_live.py`) streams the conversation through the Gemini Live API. The model name is set in `app/config.py` (`GEMINI_LIVE_MODEL`). The API key comes from the environment.

## How FER Is Integrated

Facial emotion recognition runs **in the browser** (MediaPipe face detection plus an ONNX model). The backend receives only the resulting labels and probabilities as `emotion` messages and stores them as vision signals on the session. As the code states, FER data is **never sent to Gemini**; it is kept for the session context, the analysis, and the dashboard emotion charts.

The `fer/` folder contains the model development code: EfficientNet-B0 definition, config, face detection, temporal smoothing, ONNX export, tests, and model weights.

## Session Lifecycle

1. Client authenticates; a `sessions` row is created with status `active`.
2. Live conversation runs; emotion signals are collected per turn.
3. On `end_session`, status becomes `processing`.
4. The backend builds `session_context` and `semantic_context` (`app/context/`). The semantic builder uses spaCy; embedding-based enrichment (`all-MiniLM-L6-v2`) is opt-in via `MINDO_ENABLE_SEMANTIC_EMBEDDINGS`.
5. The semantic context is sent to the RAG Backend.
6. The returned base64 PDF is decoded and uploaded as `report.pdf`, alongside `session_context.json` and `semantic_context.json`.
7. The session is marked `completed` with assessment category, confidence, summary and recommendations, or `failed` on error.

## Communication With RAG

`app/rag_client.py` sends `POST` to `RAG_REPORT_URL` (default `http://127.0.0.1:8001/api/v1/reports/generate`) with the **semantic context JSON as the request body**. The RAG service requires `session.session_id` to be present.

Because the RAG service may be asleep on the free tier, the client first pings the RAG root URL (up to 8 attempts with increasing delays), then requests the report (up to 8 attempts), retrying on timeouts and retryable HTTP statuses. The RAG response contains `report_base64`, which this service decodes.

## Communication With Supabase

| Item | Value |
|---|---|
| Tables | `users`, `sessions`, `session_artifacts` |
| Storage bucket | `mindo-sessions` (private) |
| Object path | `<firebase_uid>/<session_id>/<filename>` |
| Files | `session_context.json`, `semantic_context.json`, `report.pdf` |

Artifacts are uploaded with upsert and registered in `session_artifacts` (types `session_context`, `semantic_context`, `report_pdf`). Downloads happen server-side after an ownership check.

## Environment Variables

Names only. Never commit values. Locally the app loads `backend/.env`.

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Gemini access (required; startup fails without it) |
| `FIREBASE_SERVICE_ACCOUNT_JSON` | Firebase Admin credentials as JSON text (recommended for deployment) |
| `FIREBASE_SERVICE_ACCOUNT_PATH` | Path to a service-account file (local alternative; defaults to `backend/firebase-service-account.json`) |
| `RAG_REPORT_URL` | Full URL of the RAG report endpoint |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins (defaults to common localhost ports) |
| `HOST`, `PORT` | Bind address and port for local runs (defaults `127.0.0.1`, `8000`) |
| `MINDO_ENABLE_SEMANTIC_EMBEDDINGS` | Set to `true` to enable embedding-based enrichment (needs more memory) |
| Supabase connection variables | Read in `backend/app/services/supabase_client.py` |

## Local Setup

```bash
git clone https://github.com/reshavCodex/mindo-conversation.git
cd mindo-conversation
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
python -m spacy download en_core_web_sm
```

Create `backend/.env` with the variables above. For local Firebase credentials, place the service-account file at `backend/firebase-service-account.json` (git-ignored) or set `FIREBASE_SERVICE_ACCOUNT_JSON`. The semantic builder looks for a spaCy English model (`en_core_web_sm` or `en_core_web_md`).

## Running

```bash
uvicorn app.main:app --reload --app-dir backend
```

The API is served at `http://127.0.0.1:8000`. Run the RAG Backend on port 8001 to generate reports locally.

## Deployment

Deployed as a Render web service at https://mindo-conversation.onrender.com. Set the environment variables above in the Render dashboard (use `FIREBASE_SERVICE_ACCOUNT_JSON` instead of a file), and set `ALLOWED_ORIGINS` to the frontend origin. On the free tier the service sleeps when idle.

## Security Considerations

- Firebase ID tokens are verified on the backend for every REST call and for the WebSocket.
- Reports and session data live in a private bucket and are served only after an ownership check.
- Session files contain conversation text and emotion signals; treat the storage as sensitive.
- `.env` files and `firebase-service-account.json` are git-ignored. Never commit credentials.
- Restrict `ALLOWED_ORIGINS` to the deployed frontend.

## Part of the MINDO Project

| Repository | Role |
|---|---|
| [mindo-frontend](https://github.com/reshavCodex/mindo-frontend) | **Main project** and web app |
| [mindo-conversation](https://github.com/reshavCodex/mindo-conversation) | This repo |
| [mindo-rag](https://github.com/reshavCodex/mindo-rag) | Retrieval and report generation, called from here |
| [mindo-chat-backend](https://github.com/reshavCodex/mindo-chat-backend) | Separate chatbot service |

Live app: https://mindo-frontend.vercel.app. For the full overview, read the [main MINDO README](https://github.com/reshavCodex/mindo-frontend).

## Disclaimer

MINDO is an AI-assisted wellness and academic project, not a replacement for professional mental-health care. Facial emotion recognition is not clinically validated.

## Built By

**Reshav Pradhan**: [LinkedIn](https://www.linkedin.com/in/reshavpradhan/) · [GitHub](https://github.com/reshavCodex)

## License

No explicit open-source license has been selected. Until one is added, default copyright applies.

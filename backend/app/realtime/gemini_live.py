from google import genai
from google.genai import types

from app.config import (
    GEMINI_API_KEY,
    GEMINI_LIVE_MODEL,
)

from app.ai.prompts import SYSTEM_INSTRUCTION


class GeminiLiveSession:
    """
    Manages one real-time Gemini Live API session.

    MINDO sends only microphone audio to Gemini.
    Camera frames are handled separately by the local FER pipeline.
    """

    def __init__(self):

        # --------------------------------------------------
        # Create Gemini client
        # --------------------------------------------------

        self.client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        # Gemini Live session
        self.session = None

        # Async context manager returned by
        # client.aio.live.connect()
        self._context_manager = None

    # ======================================================
    # CONNECT TO GEMINI LIVE
    # ======================================================

    async def connect(self):

        print("Connecting to Gemini Live...")

        # --------------------------------------------------
        # Gemini Live configuration
        # --------------------------------------------------

        config = types.LiveConnectConfig(

            # Gemini returns audio responses
            response_modalities=[
                "AUDIO"
            ],

            # Instructions for MINDO
            system_instruction=SYSTEM_INSTRUCTION,

            # Lower thinking = lower latency
            thinking_config=types.ThinkingConfig(
                thinking_level="low"
            ),

            # --------------------------------------------------
            # Context Window Compression
            #
            # Important for longer-running audio sessions.
            # This allows Gemini to automatically compress older
            # conversation context instead of allowing the
            # context window to grow indefinitely.
            # --------------------------------------------------

            context_window_compression=(
                types.ContextWindowCompressionConfig(
                    sliding_window=types.SlidingWindow(),
                )
            ),

            # --------------------------------------------------
            # User speech → text
            # --------------------------------------------------

            input_audio_transcription=(
                types.AudioTranscriptionConfig()
            ),

            # --------------------------------------------------
            # Gemini speech → text
            # --------------------------------------------------

            output_audio_transcription=(
                types.AudioTranscriptionConfig()
            ),
        )

        # --------------------------------------------------
        # Create Live connection
        # --------------------------------------------------

        self._context_manager = (
            self.client.aio.live.connect(
                model=GEMINI_LIVE_MODEL,
                config=config,
            )
        )

        # Enter the async context
        self.session = await (
            self._context_manager.__aenter__()
        )

        print("Gemini Live connected successfully.")

        return self.session

    # ======================================================
    # SEND AUDIO
    # ======================================================

    async def send_audio(
        self,
        audio_bytes: bytes,
    ):

        if self.session is None:

            raise RuntimeError(
                "Gemini Live session is not connected."
            )

        await self.session.send_realtime_input(

            audio=types.Blob(
                data=audio_bytes,
                mime_type="audio/pcm;rate=16000",
            )
        )

    # ======================================================
    # CLOSE SESSION
    # ======================================================

    async def close(self):

        if self._context_manager is None:

            return

        print("Closing Gemini Live session...")

        try:

            await self._context_manager.__aexit__(
                None,
                None,
                None,
            )

        except Exception as error:

            print(
                "Error closing Gemini session:",
                error,
            )

        finally:

            self.session = None

            self._context_manager = None

            print(
                "Gemini Live session closed."
            )
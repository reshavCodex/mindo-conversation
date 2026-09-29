from datetime import datetime, timezone
from uuid import uuid4


class ConversationSession:
    """
    Stores one complete MINDO conversation.

    The conversation is organized into turns:

        Turn 1
            USER
            ASSISTANT

        Turn 2
            USER
            ASSISTANT

    Streaming transcription chunks are accumulated
    separately for the current user and assistant turn.

    Browser speech activity intervals are stored separately
    from transcription because speech timing and transcription
    timing represent different events.
    """

    def __init__(self):
        self.session_id = str(uuid4())

        self.start_time = datetime.now(timezone.utc)
        self.end_time = None

        # Final messages
        self.messages = []

        # Current streaming buffers
        self._user_transcription_buffer = ""
        self._assistant_transcription_buffer = ""

        # Turn state
        self._current_turn_number = 0
        self._user_turn_active = False
        self._assistant_turn_active = False

        # =====================================================
        # USER SPEECH ACTIVITY
        # =====================================================

        # Currently active browser-detected speech interval.
        self._active_user_speech_start = None

        # Completed browser-detected speech intervals.
        self.user_speech_intervals = []

        # Future analysis signals
        self.conversation_signals = []
        self.vision_signals = []

        # Future final analysis
        self.summary = None
        self.final_assessment = None

    # =========================================================
    # CHUNK HANDLING
    # =========================================================

    @staticmethod
    def _append_chunk(
        existing: str,
        chunk: str,
    ) -> str:
        """
        Safely append a streaming transcription chunk.
        """

        if not chunk:
            return existing

        chunk = chunk.strip()

        if not chunk:
            return existing

        if not existing:
            return chunk

        no_space_before = (
            ".",
            ",",
            "!",
            "?",
            ";",
            ":",
            "%",
            ")",
            "]",
            "}",
            "'",
            '"',
            "’",
            "”",
        )

        if chunk.startswith(no_space_before):
            return existing + chunk

        if existing.endswith(
            (" ", "\n", "\t", "(", "[", "{")
        ):
            return existing + chunk

        return existing + " " + chunk

    # =========================================================
    # USER SPEECH ACTIVITY
    # =========================================================

    @staticmethod
    def _normalize_speech_timestamp(timestamp):
        """
        Normalize a browser speech timestamp.

        The frontend sends Date.now(), which is Unix time
        in milliseconds.

        Returns:
            int | None
        """

        if timestamp is None:
            return None

        try:
            timestamp = int(timestamp)
        except (TypeError, ValueError):
            return None

        if timestamp <= 0:
            return None

        return timestamp

    def start_user_speech(self, timestamp):
        """
        Start a browser-detected user speech interval.

        The browser VAD sends this when speech begins.
        """

        timestamp = self._normalize_speech_timestamp(
            timestamp
        )

        if timestamp is None:
            return

        # Ignore duplicate speech_start events while
        # an interval is already active.
        if self._active_user_speech_start is not None:
            return

        self._active_user_speech_start = timestamp

    def end_user_speech(self, timestamp):
        """
        End the current browser-detected user speech interval.

        The browser VAD sends this when speech ends.
        """

        timestamp = self._normalize_speech_timestamp(
            timestamp
        )

        if timestamp is None:
            return

        # Ignore speech_end when there is no active interval.
        if self._active_user_speech_start is None:
            return

        start_timestamp = self._active_user_speech_start

        # Protect against invalid or out-of-order timestamps.
        if timestamp < start_timestamp:
            return

        duration_seconds = (
            timestamp - start_timestamp
        ) / 1000.0

        self.user_speech_intervals.append({
            "speech_start": start_timestamp,
            "speech_end": timestamp,
            "duration_seconds": round(
                duration_seconds,
                3,
            ),
        })

        self._active_user_speech_start = None

    def _close_active_user_speech(self, timestamp):
        """
        Close an unfinished speech interval.

        Used during session shutdown if the browser disconnects
        while the user is still speaking.
        """

        if self._active_user_speech_start is None:
            return

        self.end_user_speech(timestamp)

    # =========================================================
    # USER TRANSCRIPTION
    # =========================================================

    def add_user_transcription(
        self,
        text: str,
    ):
        """
        Add a user transcription chunk.

        A new user transcription after the previous AI turn
        automatically starts a new conversation turn.
        """

        if not text:
            return

        # If the previous turn is complete, start a new turn.
        if not self._user_turn_active:
            self._current_turn_number += 1
            self._user_turn_active = True
            self._user_transcription_buffer = ""

        self._user_transcription_buffer = (
            self._append_chunk(
                self._user_transcription_buffer,
                text,
            )
        )

    # =========================================================
    # ASSISTANT TRANSCRIPTION
    # =========================================================

    def add_assistant_transcription(
        self,
        text: str,
    ):
        """
        Add an assistant transcription chunk.
        """

        if not text:
            return

        if not self._assistant_turn_active:
            self._assistant_turn_active = True
            self._assistant_transcription_buffer = ""

        self._assistant_transcription_buffer = (
            self._append_chunk(
                self._assistant_transcription_buffer,
                text,
            )
        )

    # =========================================================
    # FINALIZE USER TURN
    # =========================================================

    def finalize_user_turn(self):
        """
        Finalize the current user transcription.
        """

        text = self._user_transcription_buffer.strip()

        if not text:
            return

        self.add_message(
            role="user",
            text=text,
        )

        self._user_transcription_buffer = ""
        self._user_turn_active = False

    # =========================================================
    # FINALIZE ASSISTANT TURN
    # =========================================================

    def finalize_assistant_turn(self):
        """
        Finalize the current assistant transcription.
        """

        text = self._assistant_transcription_buffer.strip()

        if not text:
            return

        self.add_message(
            role="assistant",
            text=text,
        )

        self._assistant_transcription_buffer = ""
        self._assistant_turn_active = False

    # =========================================================
    # TURN COMPLETE
    # =========================================================

    def complete_turn(self):
        """
        Finalize the current conversation turn.

        Called when Gemini reports turn_complete.
        """

        self.finalize_user_turn()
        self.finalize_assistant_turn()

    # =========================================================
    # BASIC MESSAGE
    # =========================================================

    def add_message(
        self,
        role: str,
        text: str,
    ):
        """
        Add one completed conversation message.
        """

        if not text:
            return

        text = text.strip()

        if not text:
            return

        message = {
            "role": role,
            "text": text,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        self.messages.append(message)

    # =========================================================
    # FLUSH
    # =========================================================

    def flush_transcription_buffers(self):
        """
        Save any unfinished transcription when
        the session is closed.
        """

        self.finalize_user_turn()
        self.finalize_assistant_turn()

    # =========================================================
    # CONVERSATION SIGNALS
    # =========================================================

    def add_conversation_signal(
        self,
        signal: dict,
    ):
        if not signal:
            return

        self.conversation_signals.append({
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "data": signal,
        })

    # =========================================================
    # VISION SIGNALS
    # =========================================================

    def add_vision_signal(
        self,
        signal: dict,
    ):
        if not signal:
            return

        self.vision_signals.append({
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "data": signal,
        })

    # =========================================================
    # END SESSION
    # =========================================================

    def end_session(self):
        """
        Finish the conversation session.

        Any active browser speech interval is closed using
        the session end timestamp.
        """

        self.flush_transcription_buffers()

        self.end_time = datetime.now(
            timezone.utc
        )

        # Convert the session end time to Unix milliseconds
        # so it matches the browser Date.now() timestamps.
        session_end_timestamp = int(
            self.end_time.timestamp() * 1000
        )

        self._close_active_user_speech(
            session_end_timestamp
        )

    # =========================================================
    # SESSION DATA
    # =========================================================

    def get_session_data(self) -> dict:
        """
        Return complete session data.

        This will later be passed to the Context Builder.
        """

        return {
            "session_id": self.session_id,

            "start_time": (
                self.start_time.isoformat()
            ),

            "end_time": (
                self.end_time.isoformat()
                if self.end_time
                else None
            ),

            "messages": self.messages,

            "user_speech_intervals": (
                self.user_speech_intervals
            ),

            "conversation_signals": (
                self.conversation_signals
            ),

            "vision_signals": (
                self.vision_signals
            ),

            "summary": self.summary,

            "final_assessment": (
                self.final_assessment
            ),
        }
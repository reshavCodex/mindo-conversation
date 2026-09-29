from datetime import datetime, timezone
from typing import Any


EMOTION_CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


FER_ASSOCIATION_WINDOW_SECONDS = 5.0


def _parse_timestamp(timestamp) -> datetime | None:
    """
    Convert supported timestamp formats into a timezone-aware
    UTC datetime object.

    Supported formats:
        - ISO 8601 strings
        - Unix timestamps in seconds
        - Unix timestamps in milliseconds
    """

    if timestamp is None:
        return None

    if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool):
        try:
            if abs(timestamp) >= 100_000_000_000:
                timestamp_seconds = timestamp / 1000.0
            else:
                timestamp_seconds = timestamp

            return datetime.fromtimestamp(
                timestamp_seconds,
                tz=timezone.utc,
            )

        except (ValueError, OSError, OverflowError):
            return None

    if isinstance(timestamp, str):
        timestamp = timestamp.strip()

        if not timestamp:
            return None

        try:
            parsed = datetime.fromisoformat(
                timestamp.replace("Z", "+00:00")
            )

            if parsed.tzinfo is None:
                parsed = parsed.replace(
                    tzinfo=timezone.utc
                )

            return parsed.astimezone(timezone.utc)

        except (ValueError, TypeError):
            return None

    return None


def _calculate_duration(
    start_time: str | int | float | None,
    end_time: str | int | float | None,
) -> float | None:
    """Calculate session duration in seconds."""

    start = _parse_timestamp(start_time)
    end = _parse_timestamp(end_time)

    if not start or not end:
        return None

    return max(
        0.0,
        (end - start).total_seconds(),
    )


def _get_signal_data(signal: dict) -> dict:
    """
    Extract the actual vision signal from the
    session's wrapped vision-signal structure.
    """

    data = signal.get("data")

    if isinstance(data, dict):
        return data

    return signal


def _normalize_probabilities(
    probabilities: dict | None,
) -> dict[str, float]:
    """
    Return a complete probability dictionary using
    the authoritative FER emotion class order.
    """

    probabilities = (
        probabilities
        if isinstance(probabilities, dict)
        else {}
    )

    normalized = {}

    for emotion in EMOTION_CLASSES:
        try:
            normalized[emotion] = float(
                probabilities.get(
                    emotion,
                    0.0,
                )
            )
        except (TypeError, ValueError):
            normalized[emotion] = 0.0

    return normalized


def _dominant_emotion(
    probabilities: dict[str, float],
    fallback: str | None = None,
) -> str | None:
    """Determine the highest-probability emotion."""

    if not probabilities:
        return fallback

    return max(
        probabilities,
        key=probabilities.get,
    )


def _average_probabilities(
    observations: list[dict],
) -> dict[str, float]:
    """Calculate average emotion probabilities."""

    if not observations:
        return {
            emotion: 0.0
            for emotion in EMOTION_CLASSES
        }

    totals = {
        emotion: 0.0
        for emotion in EMOTION_CLASSES
    }

    valid_count = 0

    for observation in observations:
        probabilities = observation.get(
            "probabilities",
            {},
        )

        if not isinstance(probabilities, dict):
            continue

        valid_count += 1

        for emotion in EMOTION_CLASSES:
            try:
                value = float(
                    probabilities.get(
                        emotion,
                        0.0,
                    )
                )
            except (TypeError, ValueError):
                value = 0.0

            totals[emotion] += value

    if valid_count == 0:
        return {
            emotion: 0.0
            for emotion in EMOTION_CLASSES
        }

    return {
        emotion: round(
            totals[emotion] / valid_count,
            6,
        )
        for emotion in EMOTION_CLASSES
    }


def _prepare_vision_signals(
    vision_signals: list[dict],
) -> list[dict]:
    """
    Normalize raw FER observations into a consistent
    structure while preserving their timestamps.
    """

    prepared = []

    for signal in vision_signals:
        if not isinstance(signal, dict):
            continue

        data = _get_signal_data(signal)

        probabilities = _normalize_probabilities(
            data.get("probabilities")
        )

        timestamp = data.get(
            "timestamp",
            signal.get("timestamp"),
        )

        dominant = data.get(
            "dominant_emotion"
        )

        if not dominant:
            dominant = _dominant_emotion(
                probabilities
            )

        prepared.append({
            "timestamp": timestamp,
            "dominant_emotion": dominant,
            "probabilities": probabilities,
        })

    prepared.sort(
        key=lambda item: (
            _parse_timestamp(
                item.get("timestamp")
            )
            or datetime.min.replace(
                tzinfo=timezone.utc
            )
        )
    )

    return prepared


def _prepare_speech_intervals(
    speech_intervals: list[dict],
) -> list[dict]:
    """
    Normalize browser-detected user speech intervals.

    Each interval represents one continuous period in which
    the browser VAD considered the user to be speaking.

    The internal datetime fields are used only for comparison
    and are removed before the final context is returned.
    """

    prepared = []

    for interval in speech_intervals:
        if not isinstance(interval, dict):
            continue

        start = interval.get("speech_start")
        end = interval.get("speech_end")

        start_datetime = _parse_timestamp(start)
        end_datetime = _parse_timestamp(end)

        if not start_datetime or not end_datetime:
            continue

        if end_datetime < start_datetime:
            continue

        duration = (
            end_datetime - start_datetime
        ).total_seconds()

        prepared.append({
            "speech_start": start,
            "speech_end": end,
            "duration_seconds": round(
                max(0.0, duration),
                3,
            ),
            "_start_datetime": start_datetime,
            "_end_datetime": end_datetime,
        })

    prepared.sort(
        key=lambda item: item["_start_datetime"]
    )

    return prepared


def _build_emotion_summary(
    vision_signals: list[dict],
) -> dict[str, Any]:
    """Build session-level emotional observations."""

    if not vision_signals:
        return {
            "observation_count": 0,
            "overall_average_probabilities": {
                emotion: 0.0
                for emotion in EMOTION_CLASSES
            },
            "overall_dominant_emotion": None,
            "dominant_emotions": [],
        }

    average_probabilities = _average_probabilities(
        vision_signals
    )

    dominant_emotions = []

    for observation in vision_signals:
        emotion = observation.get(
            "dominant_emotion"
        )

        if (
            emotion
            and emotion not in dominant_emotions
        ):
            dominant_emotions.append(emotion)

    return {
        "observation_count": len(
            vision_signals
        ),
        "overall_average_probabilities": (
            average_probabilities
        ),
        "overall_dominant_emotion": (
            _dominant_emotion(
                average_probabilities
            )
        ),
        "dominant_emotions": dominant_emotions,
    }


def _create_empty_emotional_observations() -> dict:
    """Create the default emotional observation structure."""

    return {
        "observation_count": 0,

        "average_probabilities": {
            emotion: 0.0
            for emotion in EMOTION_CLASSES
        },

        "dominant_emotion": None,

        "timeline": [],
    }


def _create_turn(
    turn_number: int,
    user_message: dict | None = None,
) -> dict:
    """Create a standardized conversation turn."""

    return {
        "turn_number": turn_number,

        "user": user_message,

        "assistant": None,

        "emotional_observations": (
            _create_empty_emotional_observations()
        ),
    }


def _get_user_turn_indexes(
    turns: list[dict],
) -> list[int]:
    """
    Return indexes of turns that contain a USER message.

    These indexes are used to align speech intervals with
    user turns by chronological order rather than by
    transcription timestamp.
    """

    indexes = []

    for index, turn in enumerate(turns):
        user = turn.get("user")

        if isinstance(user, dict):
            indexes.append(index)

    return indexes


def _assign_speech_intervals_to_turns(
    turns: list[dict],
    speech_intervals: list[dict],
) -> None:
    """
    Associate browser-detected speech intervals with USER turns.

    IMPORTANT:

    Speech intervals are aligned to user turns by chronological
    sequence.

        speech interval #1 -> user turn #1
        speech interval #2 -> user turn #2
        speech interval #3 -> user turn #3
        ...

    We intentionally DO NOT use the transcription timestamp
    here because the transcription timestamp represents when
    the transcript was finalized, not when the user physically
    started speaking.
    """

    if not turns or not speech_intervals:
        return

    prepared_intervals = _prepare_speech_intervals(
        speech_intervals
    )

    if not prepared_intervals:
        return

    user_turn_indexes = _get_user_turn_indexes(
        turns
    )

    if not user_turn_indexes:
        return

    pair_count = min(
        len(user_turn_indexes),
        len(prepared_intervals),
    )

    for position in range(pair_count):
        turn_index = user_turn_indexes[position]
        interval = prepared_intervals[position]

        turns[turn_index]["user"][
            "speech_interval"
        ] = {
            "speech_start": interval[
                "speech_start"
            ],
            "speech_end": interval[
                "speech_end"
            ],
            "duration_seconds": interval[
                "duration_seconds"
            ],
            "association": {
                "method": "sequential_user_turn_alignment",
                "speech_interval_index": position + 1,
            },
        }


def _assign_vision_signals_to_turns(
    turns: list[dict],
    vision_signals: list[dict],
    speech_intervals: list[dict] | None = None,
) -> None:
    """
    Assign FER observations to user turns.

    A FER observation is assigned to a user turn ONLY when
    its timestamp falls inside that turn's actual browser-
    detected speech interval.

    This is intentionally different from nearest-timestamp
    matching.

    FER observations captured before, after, or between user
    speech intervals remain available in raw_evidence but are
    not falsely attributed to a user turn.

    Each FER observation can be assigned to AT MOST one turn.
    """

    if not turns or not vision_signals:
        return

    if not speech_intervals:
        return

    prepared_intervals = _prepare_speech_intervals(
        speech_intervals
    )

    if not prepared_intervals:
        return

    user_turn_indexes = _get_user_turn_indexes(
        turns
    )

    if not user_turn_indexes:
        return

    assigned_by_turn = {
        index: []
        for index in user_turn_indexes
    }

    pair_count = min(
        len(user_turn_indexes),
        len(prepared_intervals),
    )

    assigned_vision_indexes = set()

    for vision_index, vision_signal in enumerate(
        vision_signals
    ):
        signal_timestamp = _parse_timestamp(
            vision_signal.get("timestamp")
        )

        if not signal_timestamp:
            continue

        matched_turn_index = None
        matched_interval = None

        for position in range(pair_count):
            turn_index = user_turn_indexes[position]
            interval = prepared_intervals[position]

            start = interval["_start_datetime"]
            end = interval["_end_datetime"]

            if start <= signal_timestamp <= end:
                matched_turn_index = turn_index
                matched_interval = interval
                break

        if (
            matched_turn_index is None
            or matched_interval is None
        ):
            continue

        signal_with_assignment = {
            **vision_signal,
            "assignment": {
                "turn_number": turns[
                    matched_turn_index
                ].get("turn_number"),
                "method": "inside_user_speech_interval",
                "speech_interval": {
                    "speech_start": matched_interval[
                        "speech_start"
                    ],
                    "speech_end": matched_interval[
                        "speech_end"
                    ],
                },
            },
        }

        assigned_by_turn[
            matched_turn_index
        ].append(
            signal_with_assignment
        )

        assigned_vision_indexes.add(
            vision_index
        )

    for turn_index, assigned_signals in (
        assigned_by_turn.items()
    ):
        if not assigned_signals:
            continue

        probabilities = _average_probabilities(
            assigned_signals
        )

        turns[turn_index][
            "emotional_observations"
        ] = {
            "observation_count": len(
                assigned_signals
            ),

            "average_probabilities": (
                probabilities
            ),

            "dominant_emotion": (
                _dominant_emotion(
                    probabilities
                )
            ),

            "timeline": assigned_signals,
        }


def _build_conversation_turns(
    messages: list[dict],
    vision_signals: list[dict],
    speech_intervals: list[dict] | None = None,
) -> list[dict]:
    """
    Organize messages into user/assistant turns.

    Speech intervals are aligned to USER turns first using
    chronological sequence.

    FER observations are then associated only with the
    speech interval belonging to that user turn.
    """

    turns = []
    current_turn = None

    for message in messages:

        if not isinstance(message, dict):
            continue

        role = message.get("role")
        text = message.get("text")

        if role not in ("user", "assistant"):
            continue

        if not text:
            continue

        timestamp = message.get("timestamp")

        if role == "user":

            if current_turn is not None:
                turns.append(
                    current_turn
                )

            current_turn = _create_turn(
                turn_number=len(turns) + 1,
                user_message={
                    "text": text,
                    "timestamp": timestamp,
                },
            )

        elif role == "assistant":

            if current_turn is None:
                current_turn = _create_turn(
                    turn_number=len(turns) + 1
                )

            current_turn["assistant"] = {
                "text": text,
                "timestamp": timestamp,
            }

    if current_turn is not None:
        turns.append(current_turn)

    # ---------------------------------------------------------
    # Attach actual browser speech timing to USER messages.
    #
    # This uses sequential alignment, NOT transcription
    # timestamps.
    # ---------------------------------------------------------

    _assign_speech_intervals_to_turns(
        turns,
        speech_intervals or [],
    )

    # ---------------------------------------------------------
    # Associate FER only with moments where the user was
    # actually speaking.
    # ---------------------------------------------------------

    _assign_vision_signals_to_turns(
        turns,
        vision_signals,
        speech_intervals=speech_intervals,
    )

    return turns


def build_session_context(
    session_data: dict,
) -> dict:
    """
    Build a standardized context representation
    from completed ConversationSession data.

    This function does NOT perform:
        - RAG
        - LLM generation
        - diagnosis
        - counselling
        - report generation

    It only merges and organizes the session evidence.
    """

    if not isinstance(session_data, dict):
        raise TypeError(
            "session_data must be a dictionary."
        )

    session_id = session_data.get(
        "session_id"
    )

    start_time = session_data.get(
        "start_time"
    )

    end_time = session_data.get(
        "end_time"
    )

    messages = session_data.get(
        "messages",
        [],
    )

    vision_signals = session_data.get(
        "vision_signals",
        [],
    )

    speech_intervals = session_data.get(
        "user_speech_intervals",
        [],
    )

    prepared_vision_signals = (
        _prepare_vision_signals(
            vision_signals
        )
    )

    turns = _build_conversation_turns(
        messages,
        prepared_vision_signals,
        speech_intervals=speech_intervals,
    )

    return {
        "schema_version": "1.0",

        "session": {
            "session_id": session_id,

            "start_time": start_time,

            "end_time": end_time,

            "duration_seconds": (
                _calculate_duration(
                    start_time,
                    end_time,
                )
            ),
        },

        "conversation": {
            "turn_count": len(turns),

            "turns": turns,
        },

        "emotion_summary": _build_emotion_summary(
            prepared_vision_signals
        ),

        "raw_evidence": {
            "messages": messages,

            "vision_signals": (
                prepared_vision_signals
            ),

            "user_speech_intervals": (
                speech_intervals
            ),
        },
    }
import json

from app.context.builder import build_session_context


def main():
    # ---------------------------------------------------------
    # Sample session data matching ConversationSession output
    # ---------------------------------------------------------

    session_data = {
        "session_id": "test-session-001",
        "start_time": "2026-09-07T12:00:00+00:00",
        "end_time": "2026-09-07T12:04:00+00:00",

        "messages": [
            {
                "role": "user",
                "text": "I've been really stressed about my exams.",
                "timestamp": "2026-09-07T12:00:10+00:00",
            },
            {
                "role": "assistant",
                "text": "That sounds difficult. Tell me more about what has been stressing you.",
                "timestamp": "2026-09-07T12:00:15+00:00",
            },
            {
                "role": "user",
                "text": "I feel like I have too much to prepare.",
                "timestamp": "2026-09-07T12:01:00+00:00",
            },
            {
                "role": "assistant",
                "text": "It can feel overwhelming when there is a lot to prepare.",
                "timestamp": "2026-09-07T12:01:05+00:00",
            },
        ],

        "conversation_signals": [],

        "vision_signals": [
            {
                "timestamp": "2026-09-07T12:00:08+00:00",
                "data": {
                    "timestamp": "2026-09-07T12:00:08+00:00",
                    "dominant_emotion": "fear",
                    "probabilities": {
                        "angry": 0.02,
                        "disgust": 0.01,
                        "fear": 0.60,
                        "happy": 0.02,
                        "neutral": 0.25,
                        "sad": 0.08,
                        "surprise": 0.02,
                    },
                },
            },
            {
                "timestamp": "2026-09-07T12:00:12+00:00",
                "data": {
                    "timestamp": "2026-09-07T12:00:12+00:00",
                    "dominant_emotion": "fear",
                    "probabilities": {
                        "angry": 0.01,
                        "disgust": 0.01,
                        "fear": 0.55,
                        "happy": 0.03,
                        "neutral": 0.30,
                        "sad": 0.08,
                        "surprise": 0.02,
                    },
                },
            },
            {
                "timestamp": "2026-09-07T12:00:58+00:00",
                "data": {
                    "timestamp": "2026-09-07T12:00:58+00:00",
                    "dominant_emotion": "sad",
                    "probabilities": {
                        "angry": 0.02,
                        "disgust": 0.01,
                        "fear": 0.35,
                        "happy": 0.02,
                        "neutral": 0.30,
                        "sad": 0.28,
                        "surprise": 0.02,
                    },
                },
            },
        ],

        "summary": None,
        "final_assessment": None,
    }

    # ---------------------------------------------------------
    # Build context
    # ---------------------------------------------------------

    context = build_session_context(session_data)

    # ---------------------------------------------------------
    # Print formatted JSON
    # ---------------------------------------------------------

    print(
        json.dumps(
            context,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
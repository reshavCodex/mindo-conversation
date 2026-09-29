from __future__ import annotations

import re
from collections import Counter
from typing import Any

# Heavy NLP dependencies are intentionally imported lazily.
# This prevents spaCy / sentence-transformers / PyTorch from being
# loaded during FastAPI startup on memory-constrained deployments.
spacy = None
SentenceTransformer = None


EMOTION_CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


# =========================================================
# NLP MODEL CONFIGURATION
# =========================================================

_SPACY_MODEL = None
_EMBEDDING_MODEL = None


def _get_spacy_model():
    global _SPACY_MODEL
    global spacy

    if _SPACY_MODEL is not None:
        return _SPACY_MODEL

    if spacy is None:
        try:
            import spacy as _spacy
            spacy = _spacy
        except ImportError:
            return None

    for model_name in (
        "en_core_web_sm",
        "en_core_web_md",
    ):
        try:
            _SPACY_MODEL = spacy.load(model_name)
            return _SPACY_MODEL
        except Exception:
            continue

    return None


def _get_embedding_model():
    global _EMBEDDING_MODEL
    global SentenceTransformer

    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL

    if SentenceTransformer is None:
        try:
            from sentence_transformers import (
                SentenceTransformer as _SentenceTransformer,
            )
            SentenceTransformer = _SentenceTransformer
        except ImportError:
            return None

    try:
        _EMBEDDING_MODEL = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )
        return _EMBEDDING_MODEL
    except Exception:
        return None


# =========================================================
# GENERAL HELPERS
# =========================================================

def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""

    return " ".join(value.strip().split())


def _lower_text(value: Any) -> str:
    return _clean_text(value).lower()


def _unique_strings(values: list[str]) -> list[str]:
    result = []
    seen = set()

    for value in values:
        value = _clean_text(value)

        if not value:
            continue

        key = value.lower()

        if key in seen:
            continue

        seen.add(key)
        result.append(value)

    return result


def _unique_dicts(
    values: list[dict],
    key_fields: tuple[str, ...],
) -> list[dict]:
    result = []
    seen = set()

    for value in values:
        if not isinstance(value, dict):
            continue

        key = tuple(
            _clean_text(value.get(field)).lower()
            for field in key_fields
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(value)

    return result


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_turn_id(turn: dict) -> Any:
    return turn.get("turn_number")


# =========================================================
# CONVERSATION ACCESS
# =========================================================

def _get_user_text(turn: dict) -> str:
    user = turn.get("user")

    if not isinstance(user, dict):
        return ""

    return _clean_text(user.get("text"))


def _get_assistant_text(turn: dict) -> str:
    """
    Return MINDO's actual response.

    Assistant language is preserved for traceability only.
    It is never treated as user semantic evidence.
    """

    assistant = turn.get("assistant")

    if not isinstance(assistant, dict):
        return ""

    return _clean_text(assistant.get("text"))


# =========================================================
# TEXT PROCESSING
# =========================================================

def _process_text(text: str):
    nlp = _get_spacy_model()

    if nlp is None:
        return None

    text = _clean_text(text)

    if not text:
        return None

    try:
        return nlp(text)
    except Exception:
        return None


def _extract_sentences(text: str) -> list[str]:
    text = _clean_text(text)

    if not text:
        return []

    doc = _process_text(text)

    if doc is None:
        return _unique_strings(
            [
                sentence.strip()
                for sentence in re.split(
                    r"(?<=[.!?])\s+",
                    text,
                )
                if sentence.strip()
            ]
        )

    return _unique_strings(
        [
            sentence.text
            for sentence in doc.sents
            if _clean_text(sentence.text)
        ]
    )


# =========================================================
# LINGUISTIC CONTENT ANALYSIS
# =========================================================

_CONTENT_POS = {
    "NOUN",
    "PROPN",
    "ADJ",
    "VERB",
    "ADV",
}


_GENERIC_LEMMAS = {
    "be",
    "have",
    "do",
    "get",
    "go",
    "make",
    "know",
    "say",
    "tell",
    "think",
    "want",
    "need",
    "like",
    "let",
    "see",
    "look",
}


def _content_tokens(doc) -> list:
    if doc is None:
        return []

    return [
        token
        for token in doc
        if (
            not token.is_space
            and not token.is_punct
            and not token.is_stop
            and len(token.text) >= 2
            and token.pos_ in _CONTENT_POS
        )
    ]


def _meaningful_lexical_tokens(doc) -> list:
    if doc is None:
        return []

    return [
        token
        for token in doc
        if (
            not token.is_space
            and not token.is_punct
            and not token.is_stop
            and len(token.text) >= 2
        )
    ]


def _has_first_person_reference(doc) -> bool:
    if doc is None:
        return False

    return any(
        token.lower_
        in {
            "i",
            "me",
            "my",
            "mine",
            "myself",
        }
        for token in doc
    )


def _is_negated(token) -> bool:
    """
    Dependency-based negation check for a token.

    Looks at the token itself and its head for a 'neg' child, or
    an 'n't'/'not' adverbial modifier, so that contracted forms
    ("can't", "isn't") and full forms ("can not") are both caught.
    """

    if token is None:
        return False

    candidates = [token]

    head = getattr(token, "head", None)

    if head is not None and head is not token:
        candidates.append(head)

    for candidate in candidates:
        for child in candidate.children:
            if child.dep_ == "neg":
                return True

            if (
                child.dep_ == "advmod"
                and child.lower_ in {"n't", "not"}
            ):
                return True

    return False


def _has_meaningful_state_predicate(doc) -> bool:
    """
    Canonical test for a meaningful state/experience predicate.

    This consolidates what used to be two independently drifting
    heuristics: a predicate-only check (VERB/ADJ) and a separate
    object-only check (NOUN/PROPN) that lived in
    _is_personal_state_sentence. Both are evaluated here so that
    every caller agrees on what counts as a meaningful state.

    Generic verbs ("have", "get", etc.) are still ignored UNLESS
    they carry a concrete NOUN/PROPN complement -- e.g. "I've got
    insomnia" / "I have anxiety" -- since this is one of the most
    common colloquial ways people disclose a symptom or condition
    and must not be blanket-excluded.
    """

    if doc is None:
        return False

    for token in doc:
        lemma = token.lemma_.lower()

        if token.pos_ in {
            "VERB",
            "ADJ",
        }:
            if token.dep_ not in {
                "ROOT",
                "acomp",
                "xcomp",
                "ccomp",
                "attr",
            }:
                continue

            if lemma == "thank":
                continue

            if lemma in _GENERIC_LEMMAS:
                if lemma not in {"have", "get"}:
                    continue

                has_concrete_complement = any(
                    child.pos_ in {"NOUN", "PROPN"}
                    for child in token.children
                )

                if not has_concrete_complement:
                    continue

            return True

        if token.pos_ in {
            "NOUN",
            "PROPN",
        }:
            if token.dep_ not in {
                "ROOT",
                "attr",
                "obj",
                "dobj",
            }:
                continue

            if lemma in _GENERIC_LEMMAS or lemma == "thank":
                continue

            return True

    return False


def _has_meaningful_predicate(doc) -> bool:
    return _has_meaningful_state_predicate(doc)


def _has_concrete_information(doc) -> bool:
    if doc is None:
        return False

    if getattr(doc, "ents", []):
        return True

    for token in doc:
        if token.pos_ in {
            "NOUN",
            "PROPN",
        }:
            return True

    return False


# =========================================================
# CONVERSATIONAL FUNCTION
# =========================================================

_DISCOURSE_PROTOTYPES = [
    "The speaker is simply acknowledging what was said.",
    "The speaker is confirming or agreeing with the previous statement.",
    "The speaker is giving a brief conversational response.",
    "The speaker is expressing gratitude or politeness.",
    "The speaker is politely closing the conversation.",
    "The speaker is using a conversational filler.",
    "The speaker is preparing to say something without giving the actual information yet.",
]


def _embedding_similarity(
    text: str,
    prototypes: list[str],
) -> float | None:
    model = _get_embedding_model()

    if model is None:
        return None

    try:
        embeddings = model.encode(
            [text] + prototypes,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        query = embeddings[0]

        similarities = [
            float(query @ vector)
            for vector in embeddings[1:]
        ]

        if not similarities:
            return None

        return max(similarities)

    except Exception:
        return None


def _looks_like_gratitude(text: str, doc=None) -> bool:
    """
    Structural gratitude detection.

    This is intentionally narrow. It prevents common gratitude
    utterances from becoming semantic facts without turning the
    whole semantic layer into a giant keyword blacklist.
    """

    text = _lower_text(text)

    if not text:
        return False

    if doc is None:
        doc = _process_text(text)

    if doc is None:
        normalized = re.sub(
            r"[^a-z0-9']+",
            " ",
            text,
        ).strip()

        return bool(
            re.fullmatch(
                r"(thank|thanks)( you)?"
                r"(\s+for\s+.+)?",
                normalized,
            )
        )

    roots = [
        token
        for token in doc
        if token.dep_ == "ROOT"
    ]

    if not roots:
        return False

    root = roots[0]

    if root.lemma_.lower() not in {
        "thank",
    }:
        return False

    return True


def _looks_like_filler(text: str, doc=None) -> bool:
    """
    Detect very short conversational filler without using a
    large hardcoded semantic vocabulary.
    """

    text = _clean_text(text)

    if not text:
        return True

    doc = doc or _process_text(text)

    if doc is None:
        words = re.findall(
            r"\b[\w']+\b",
            text,
        )

        return len(words) <= 2

    lexical = _meaningful_lexical_tokens(doc)

    if len(lexical) > 2:
        return False

    if len(lexical) == 0:
        return True

    # INTJ-only utterances such as "um", "ah", etc.
    if all(
        token.pos_ == "INTJ"
        for token in lexical
    ):
        return True

    # Very short non-propositional utterances.
    if len(lexical) == 1:
        token = lexical[0]

        if token.pos_ in {
            "INTJ",
            "PRON",
            "DET",
            "PART",
        }:
            return True

    return False


def _looks_like_conversational_setup(
    text: str,
    doc=None,
) -> bool:
    """
    Detect statements that announce future conversation without
    actually providing the information.

    Example:
        "I want to tell you something."

    This should remain in conversation_flow but not become
    persistent semantic context.
    """

    text = _clean_text(text)

    if not text:
        return True

    doc = doc or _process_text(text)

    if doc is None:
        normalized = _lower_text(text)

        return bool(
            re.fullmatch(
                r"i\s+want\s+to\s+(tell|say)\s+you\s+something\.?",
                normalized,
            )
        )

    if not _has_first_person_reference(doc):
        return False

    meaningful = [
        token
        for token in doc
        if (
            not token.is_stop
            and not token.is_punct
            and token.pos_
            in {
                "NOUN",
                "PROPN",
                "ADJ",
                "VERB",
            }
        )
    ]

    # "I want to tell you something."
    # "I want to say something."
    if len(meaningful) <= 3:
        lemmas = {
            token.lemma_.lower()
            for token in meaningful
        }

        if (
            "want" in lemmas
            and (
                "tell" in lemmas
                or "say" in lemmas
            )
        ):
            return True

    return False


def _looks_like_empty_response(
    text: str,
    doc=None,
) -> bool:
    """
    Detect responses whose semantic contribution is effectively
    empty, such as "Nothing."
    """

    text = _clean_text(text)

    if not text:
        return True

    doc = doc or _process_text(text)

    if doc is None:
        return len(
            re.findall(
                r"\b[\w']+\b",
                text,
            )
        ) <= 2

    lexical = _meaningful_lexical_tokens(doc)

    if len(lexical) == 0:
        return True

    # A single bare noun/pronoun/adverb is not enough to establish
    # durable context when it functions as a conversational reply.
    if len(lexical) == 1:
        token = lexical[0]

        if token.pos_ in {
            "PRON",
            "ADV",
            "INTJ",
            "DET",
        }:
            return True

    return False


def _is_discourse_heavy(
    text: str,
    doc=None,
) -> bool:
    """
    Determine whether an utterance primarily performs a
    conversational function rather than adding semantic context.
    """

    text = _clean_text(text)

    if not text:
        return True

    doc = doc or _process_text(text)

    if _looks_like_gratitude(
        text,
        doc,
    ):
        return True

    if _looks_like_filler(
        text,
        doc,
    ):
        return True

    if _looks_like_conversational_setup(
        text,
        doc,
    ):
        return True

    if _looks_like_empty_response(
        text,
        doc,
    ):
        return True

    if doc is None:
        return False

    lexical = _meaningful_lexical_tokens(doc)
    content = _content_tokens(doc)

    if not lexical:
        return True

    # Short confirmation/acknowledgement structures.
    if len(lexical) <= 2:
        if not _has_meaningful_predicate(doc):
            return True

    # Mostly functional language without an actual proposition.
    content_ratio = (
        len(content) / len(lexical)
        if lexical
        else 0.0
    )

    if (
        content_ratio < 0.34
        and not _has_meaningful_predicate(doc)
        and not _has_concrete_information(doc)
    ):
        return True

    similarity = _embedding_similarity(
        text,
        _DISCOURSE_PROTOTYPES,
    )

    if similarity is not None:
        if similarity >= 0.72:
            # Do not discard information-rich sentences merely
            # because they contain conversational wording.
            if not _has_concrete_information(doc):
                return True

            if len(content) <= 1:
                return True

    return False


# =========================================================
# SEMANTIC INFORMATION TEST
# =========================================================

def _has_semantic_information(
    text: str,
) -> bool:
    """
    Decide whether a USER utterance contributes meaningful
    semantic context.

    This is deliberately conservative.

    Conversation management stays in conversation_flow.
    """

    text = _clean_text(text)

    if not text:
        return False

    doc = _process_text(text)

    if _looks_like_gratitude(
        text,
        doc,
    ):
        return False

    if _looks_like_filler(
        text,
        doc,
    ):
        return False

    if _looks_like_conversational_setup(
        text,
        doc,
    ):
        return False

    if _looks_like_empty_response(
        text,
        doc,
    ):
        return False

    if doc is None:
        words = re.findall(
            r"\b[\w']+\b",
            text,
        )

        return len(words) >= 3

    if _is_discourse_heavy(
        text,
        doc,
    ):
        return False

    lexical = _meaningful_lexical_tokens(doc)
    content = _content_tokens(doc)

    if not lexical:
        return False

    # Questions carry semantic information.
    if "?" in text:
        return True

    # Explicit first-person states/experiences.
    if (
        _has_first_person_reference(doc)
        and _has_meaningful_predicate(doc)
    ):
        return True

    # Named entities or concrete nouns.
    if _has_concrete_information(doc):
        return True

    # Content-rich statement.
    content_ratio = (
        len(content) / len(lexical)
    )

    return (
        len(content) >= 2
        and content_ratio >= 0.45
    )


# =========================================================
# USER STATEMENTS
# =========================================================

def _build_user_statements(
    turns: list[dict],
) -> list[dict]:
    """
    Build meaningful USER statements.

    Entire turns are retained as the semantic unit.
    Sentence-level splitting is avoided here because a single
    user turn can express one coherent concern across multiple
    sentences.
    """

    statements = []

    for turn in turns:
        if not isinstance(turn, dict):
            continue

        text = _get_user_text(turn)

        if not text:
            continue

        if not _has_semantic_information(
            text
        ):
            continue

        statements.append({
            "text": text,
            "turn_ids": [
                _get_turn_id(turn)
            ],
        })

    return statements


# =========================================================
# CONVERSATION FLOW
# =========================================================

def _build_conversation_pairs(
    turns: list[dict],
) -> list[dict]:
    """
    Preserve the complete user ↔ MINDO interaction.

    This includes conversational-only utterances.
    """

    pairs = []

    for turn in turns:
        user_text = _get_user_text(turn)
        assistant_text = _get_assistant_text(turn)

        if not user_text and not assistant_text:
            continue

        item = {
            "turn_id": _get_turn_id(turn),
        }

        if user_text:
            item["user"] = user_text

        if assistant_text:
            item["mindo"] = assistant_text

        pairs.append(item)

    return pairs


# =========================================================
# PHRASE NORMALIZATION
# =========================================================

def _normalize_phrase(
    phrase: str,
) -> str:
    phrase = _clean_text(phrase)

    if not phrase:
        return ""

    doc = _process_text(phrase)

    if doc is None:
        return phrase.lower()

    tokens = []

    for token in doc:
        if token.is_space or token.is_punct:
            continue

        if token.is_stop:
            if token.lower_ not in {
                "not",
                "no",
                "never",
                "n't",
            }:
                continue

        lemma = _clean_text(
            token.lemma_
        )

        if lemma:
            tokens.append(
                lemma.lower()
            )

    return " ".join(tokens)


# =========================================================
# CONCEPT QUALITY
# =========================================================

def _concept_is_useful(
    phrase: str,
) -> bool:
    phrase = _clean_text(phrase)

    if not phrase:
        return False

    doc = _process_text(phrase)

    if doc is None:
        return len(
            phrase.split()
        ) >= 2

    # Never permit gratitude to become a concept.
    if _looks_like_gratitude(
        phrase,
        doc,
    ):
        return False

    if _looks_like_filler(
        phrase,
        doc,
    ):
        return False

    content = _content_tokens(doc)

    if not content:
        return False

    if all(
        token.pos_ in {
            "PRON",
            "DET",
        }
        for token in content
    ):
        return False

    # Meaningful nouns/adjectives/proper nouns.
    if any(
        token.pos_
        in {
            "ADJ",
            "NOUN",
            "PROPN",
        }
        for token in content
    ):
        return True

    # Meaningful verbs.
    for token in content:
        if token.pos_ == "VERB":
            if (
                token.lemma_.lower()
                not in _GENERIC_LEMMAS
                and token.lemma_.lower()
                != "thank"
            ):
                return True

    return False


# =========================================================
# CONCEPT EXTRACTION
# =========================================================

def _extract_meaningful_phrases(
    text: str,
) -> list[str]:
    text = _clean_text(text)

    if not text:
        return []

    if not _has_semantic_information(
        text
    ):
        return []

    doc = _process_text(text)

    if doc is None:
        return []

    candidates = []

    # -----------------------------------------------------
    # Noun phrases
    # -----------------------------------------------------

    for chunk in doc.noun_chunks:
        phrase = _clean_text(
            chunk.text
        )

        if not phrase:
            continue

        if not _concept_is_useful(
            phrase
        ):
            continue

        normalized = _normalize_phrase(
            phrase
        )

        if normalized:
            candidates.append({
                "phrase": phrase,
                "normalized": normalized,
                "score": 3.0,
            })

    # -----------------------------------------------------
    # Meaningful adjectives / verbs
    # -----------------------------------------------------

    for token in doc:
        if token.is_stop:
            continue

        if token.is_punct or token.is_space:
            continue

        if len(token.text) < 3:
            continue

        if token.pos_ not in {
            "ADJ",
            "VERB",
        }:
            continue

        lemma = _clean_text(
            token.lemma_
        ).lower()

        if not lemma:
            continue

        if lemma in _GENERIC_LEMMAS:
            continue

        if lemma == "thank":
            continue

        if not _concept_is_useful(
            lemma
        ):
            continue

        if _is_negated(token):
            phrase = f"not {lemma}"
        else:
            phrase = lemma

        candidates.append({
            "phrase": phrase,
            "normalized": phrase,
            "score": 1.5,
        })

    # -----------------------------------------------------
    # Verb + complement
    # -----------------------------------------------------

    for token in doc:
        if token.pos_ != "VERB":
            continue

        if token.lemma_.lower() in (
            _GENERIC_LEMMAS
            | {"thank"}
        ):
            continue

        children = [
            child
            for child in token.children
            if (
                not child.is_stop
                and not child.is_punct
                and child.pos_
                in {
                    "NOUN",
                    "PROPN",
                    "PRON",
                    "ADJ",
                    "VERB",
                }
            )
        ]

        if not children:
            continue

        selected = sorted(
            children,
            key=lambda child: child.i,
        )

        phrase_tokens = [
            token.text
        ] + [
            child.text
            for child in selected[:4]
        ]

        if _is_negated(token):
            phrase_tokens = ["not"] + phrase_tokens

        phrase = _clean_text(
            " ".join(phrase_tokens)
        )

        if len(phrase.split()) < 2:
            continue

        if not _concept_is_useful(
            phrase
        ):
            continue

        normalized = _normalize_phrase(
            phrase
        )

        if normalized:
            candidates.append({
                "phrase": phrase,
                "normalized": normalized,
                "score": 2.5,
            })

    # -----------------------------------------------------
    # Deduplicate
    # -----------------------------------------------------

    grouped = {}

    for candidate in candidates:
        normalized = candidate["normalized"]

        if not normalized:
            continue

        existing = grouped.get(
            normalized
        )

        if existing is None:
            grouped[normalized] = candidate
            continue

        if candidate["score"] > existing["score"]:
            grouped[normalized] = candidate

    ranked = sorted(
        grouped.values(),
        key=lambda item: (
            -item["score"],
            len(
                item["normalized"].split()
            ),
        ),
    )

    return [
        item["phrase"]
        for item in ranked[:8]
    ]


def _build_key_concepts(
    statements: list[dict],
    max_items: int = 8,
) -> list[str]:
    if not statements:
        return []

    scores = Counter()
    display_forms = {}

    for statement in statements:
        text = _clean_text(
            statement.get("text")
        )

        if not text:
            continue

        phrases = _extract_meaningful_phrases(
            text
        )

        for phrase in phrases:
            normalized = _normalize_phrase(
                phrase
            )

            if not normalized:
                continue

            if not _concept_is_useful(
                phrase
            ):
                continue

            scores[normalized] += 1.0

            if normalized not in display_forms:
                display_forms[
                    normalized
                ] = phrase

    ranked = sorted(
        scores.items(),
        key=lambda item: (
            -item[1],
            len(
                item[0].split()
            ),
        ),
    )

    return [
        display_forms[key]
        for key, _ in ranked[:max_items]
    ]


# =========================================================
# SEMANTIC GROUPING
# =========================================================

def _semantic_similarity_groups(
    statements: list[dict],
    similarity_threshold: float = 0.38,
) -> list[dict]:
    model = _get_embedding_model()

    if (
        model is None
        or len(statements) < 2
    ):
        return []

    valid_statements = [
        statement
        for statement in statements
        if _has_semantic_information(
            statement.get("text", "")
        )
    ]

    if len(valid_statements) < 2:
        return []

    texts = [
        _clean_text(
            statement["text"]
        )
        for statement in valid_statements
    ]

    try:
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception:
        return []

    groups = []
    assigned = set()

    for index in range(len(texts)):
        if index in assigned:
            continue

        related = [index]

        for other_index in range(
            index + 1,
            len(texts),
        ):
            if other_index in assigned:
                continue

            similarity = float(
                embeddings[index]
                @ embeddings[other_index]
            )

            if similarity >= similarity_threshold:
                related.append(
                    other_index
                )

        if len(related) < 2:
            continue

        for item_index in related:
            assigned.add(item_index)

        related_turn_ids = []

        for item_index in related:
            related_turn_ids.extend(
                valid_statements[
                    item_index
                ].get(
                    "turn_ids",
                    [],
                )
            )

        groups.append({
            "representative": texts[
                related[0]
            ],
            "related_statements": [
                texts[item_index]
                for item_index in related
            ],
            "turn_ids": _unique_strings(
                [
                    str(turn_id)
                    for turn_id in related_turn_ids
                ]
            ),
        })

    return groups


# =========================================================
# ENTITY EXTRACTION
# =========================================================

def _extract_entities(
    statements: list[dict],
) -> list[dict]:
    entities = []

    for statement in statements:
        text = _clean_text(
            statement.get("text")
        )

        if not text:
            continue

        doc = _process_text(text)

        if doc is None:
            continue

        for entity in doc.ents:
            entity_text = _clean_text(
                entity.text
            )

            if not entity_text:
                continue

            entities.append({
                "text": entity_text,
                "type": entity.label_,
                "turn_ids": statement.get(
                    "turn_ids",
                    [],
                ),
            })

    return _unique_dicts(
        entities,
        ("text", "type"),
    )


# =========================================================
# TEMPORAL CONTEXT
# =========================================================

def _extract_temporal_context(
    statements: list[dict],
) -> list[dict]:
    temporal = []

    fallback_patterns = [
        r"\btoday\b",
        r"\btomorrow\b",
        r"\byesterday\b",
        r"\btonight\b",
        r"\brecently\b",
        r"\blately\b",
        r"\bthis\s+(?:morning|afternoon|evening|week|month|year)\b",
        r"\blast\s+(?:night|week|month|year)\b",
        r"\bnext\s+(?:week|month|year)\b",
        r"\b\d+\s+(?:days?|weeks?|months?|years?)\s+ago\b",
    ]

    for statement in statements:
        text = _clean_text(
            statement.get("text")
        )

        if not text:
            continue

        doc = _process_text(text)

        if doc is not None:
            for entity in doc.ents:
                if entity.label_ in {
                    "DATE",
                    "TIME",
                }:
                    value = _clean_text(
                        entity.text
                    )

                    if value:
                        temporal.append({
                            "text": value,
                            "turn_ids": statement.get(
                                "turn_ids",
                                [],
                            ),
                        })

        normalized = _lower_text(text)

        for pattern in fallback_patterns:
            matches = re.findall(
                pattern,
                normalized,
                flags=re.IGNORECASE,
            )

            for match in matches:
                value = _clean_text(
                    match
                )

                if value:
                    temporal.append({
                        "text": value,
                        "turn_ids": statement.get(
                            "turn_ids",
                            [],
                        ),
                    })

    return _unique_dicts(
        temporal,
        ("text",),
    )


# =========================================================
# USER INTENT
# =========================================================

def _contains_question(
    sentence,
) -> bool:
    return "?" in _clean_text(
        sentence.text
    )


def _is_request_sentence(
    sentence,
) -> bool:
    """
    Conservative structural request detection.

    Gratitude and conversational acknowledgements are explicitly
    excluded before request detection.
    """

    text = _clean_text(
        sentence.text
    )

    if not text:
        return False

    if _looks_like_gratitude(
        text,
        sentence.doc,
    ):
        return False

    if _contains_question(sentence):
        return False

    if _looks_like_conversational_setup(
        text,
        sentence.doc,
    ):
        return False

    if _looks_like_filler(
        text,
        sentence.doc,
    ):
        return False

    first = sentence[0]

    # Imperative structure.
    if first.pos_ == "VERB":
        has_subject = any(
            token.dep_
            in {
                "nsubj",
                "nsubjpass",
            }
            for token in sentence
        )

        if not has_subject:
            return True

    # Modal request structures.
    for token in sentence:
        if token.pos_ == "AUX":
            for child in token.children:
                if child.dep_ in {
                    "xcomp",
                    "ccomp",
                }:
                    return True

    return False


def _is_personal_state_sentence(
    sentence,
) -> bool:
    if not _has_first_person_reference(sentence):
        return False

    return _has_meaningful_state_predicate(sentence)


def _infer_intent_for_sentence(
    sentence,
) -> str:
    text = _clean_text(
        sentence.text
    )

    if _looks_like_gratitude(
        text,
        sentence.doc,
    ):
        return "acknowledgment_or_conversational_response"

    if _looks_like_conversational_setup(
        text,
        sentence.doc,
    ):
        return "acknowledgment_or_conversational_response"

    if _contains_question(sentence):
        return "question_or_information_seeking"

    if _is_request_sentence(sentence):
        return "request_or_instruction"

    if _is_personal_state_sentence(sentence):
        return "personal_experience_or_state"

    if _is_discourse_heavy(
        text,
        sentence.doc,
    ):
        return "acknowledgment_or_conversational_response"

    return "statement_or_information_sharing"


def _infer_user_intents(
    statements: list[dict],
) -> list[dict]:
    """
    Infer broad user intent only from meaningful user turns.

    Conversational functions are represented separately and never
    allowed to masquerade as requests or facts.
    """

    results = []

    for statement in statements:
        text = _clean_text(
            statement.get("text")
        )

        if not text:
            continue

        doc = _process_text(text)

        if doc is None:
            continue

        for sentence in doc.sents:
            sentence_text = _clean_text(
                sentence.text
            )

            if not sentence_text:
                continue

            # Pure gratitude/filler/setup.
            if (
                _looks_like_gratitude(
                    sentence_text,
                    sentence.doc,
                )
                or _looks_like_filler(
                    sentence_text,
                    sentence.doc,
                )
                or _looks_like_conversational_setup(
                    sentence_text,
                    sentence.doc,
                )
                or _looks_like_empty_response(
                    sentence_text,
                    sentence.doc,
                )
            ):
                continue

            if not _has_semantic_information(
                sentence_text
            ):
                continue

            intent = _infer_intent_for_sentence(
                sentence
            )

            results.append({
                "intent": intent,
                "evidence": sentence_text,
                "turn_ids": statement.get(
                    "turn_ids",
                    [],
                ),
            })

    return _unique_dicts(
        results,
        ("intent", "evidence"),
    )[:20]


# =========================================================
# IMPORTANT FACTS
# =========================================================

def _sentence_contains_factual_signal(
    sentence,
) -> bool:
    text = _clean_text(
        sentence.text
    )

    if not text:
        return False

    if "?" in text:
        return False

    if _looks_like_gratitude(
        text,
        sentence.doc,
    ):
        return False

    if _looks_like_filler(
        text,
        sentence.doc,
    ):
        return False

    if _looks_like_conversational_setup(
        text,
        sentence.doc,
    ):
        return False

    if _looks_like_empty_response(
        text,
        sentence.doc,
    ):
        return False

    if _is_personal_state_sentence(
        sentence
    ):
        return True

    doc = sentence.doc

    if _has_concrete_information(
        doc
    ):
        if any(
            token.pos_ == "VERB"
            and token.lemma_.lower()
            not in (
                _GENERIC_LEMMAS
                | {"thank"}
            )
            for token in sentence
        ):
            return True

    if any(
        entity.label_
        in {
            "DATE",
            "TIME",
        }
        for entity in getattr(
            doc,
            "ents",
            [],
        )
    ):
        return True

    return False


def _extract_important_facts(
    statements: list[dict],
) -> list[dict]:
    """
    Extract explicit information-bearing USER turns.

    A complete user turn is retained whenever it contains useful
    information. This avoids turning one coherent utterance into
    arbitrary sentence fragments.
    """

    facts = []

    for statement in statements:
        text = _clean_text(
            statement.get("text")
        )

        if not text:
            continue

        if _looks_like_gratitude(
            text
        ):
            continue

        if _looks_like_conversational_setup(
            text
        ):
            continue

        if _looks_like_filler(
            text
        ):
            continue

        if _looks_like_empty_response(
            text
        ):
            continue

        doc = _process_text(text)

        if doc is None:
            continue

        useful_sentence_found = False

        for sentence in doc.sents:
            if _sentence_contains_factual_signal(
                sentence
            ):
                useful_sentence_found = True
                break

        if useful_sentence_found:
            facts.append({
                "text": text,
                "turn_ids": statement.get(
                    "turn_ids",
                    [],
                ),
            })

    return _unique_dicts(
        facts,
        ("text",),
    )[:20]


# =========================================================
# STATED CONCERNS
# =========================================================

_CONCERN_PROTOTYPES = [
    "The speaker is describing a personal difficulty.",
    "The speaker is describing something that worries them.",
    "The speaker is describing something that scares them.",
    "The speaker is describing a personal emotional difficulty.",
    "The speaker is describing a problem they are experiencing.",
]


def _contains_explicit_state_or_concern(
    text: str,
) -> bool:
    text = _clean_text(text)

    if not text:
        return False

    doc = _process_text(text)

    if doc is None:
        return False

    if _looks_like_gratitude(
        text,
        doc,
    ):
        return False

    if _looks_like_conversational_setup(
        text,
        doc,
    ):
        return False

    if _looks_like_filler(
        text,
        doc,
    ):
        return False

    if _looks_like_empty_response(
        text,
        doc,
    ):
        return False

    # Explicit first-person state.
    if (
        _has_first_person_reference(doc)
        and _has_meaningful_predicate(doc)
    ):
        return True

    # Semantic similarity is only supporting evidence.
    similarity = _embedding_similarity(
        text,
        _CONCERN_PROTOTYPES,
    )

    if similarity is not None:
        if similarity >= 0.68:
            return True

    return False


def _build_stated_concerns(
    statements: list[dict],
) -> list[dict]:
    """
    Preserve the complete meaningful user utterance.

    We do not split one coherent concern into sentence fragments.
    """

    concerns = []

    for statement in statements:
        text = _clean_text(
            statement.get("text")
        )

        if not text:
            continue

        if not _contains_explicit_state_or_concern(
            text
        ):
            continue

        concerns.append({
            "text": text,
            "turn_ids": statement.get(
                "turn_ids",
                [],
            ),
        })

    return _unique_dicts(
        concerns,
        ("text",),
    )


# =========================================================
# SITUATIONAL FACTORS
# =========================================================

def _has_event_structure(doc) -> bool:
    if doc is None:
        return False

    event_verbs = 0

    for token in doc:
        if token.pos_ != "VERB":
            continue

        if token.lemma_.lower() in {
            "be",
            "have",
            "do",
            "want",
            "know",
            "tell",
        }:
            continue

        event_verbs += 1

    return event_verbs > 0


def _has_setting_or_context(doc) -> bool:
    if doc is None:
        return False

    if any(
        entity.label_
        in {
            "GPE",
            "LOC",
            "FAC",
            "ORG",
        }
        for entity in getattr(
            doc,
            "ents",
            [],
        )
    ):
        return True

    for token in doc:
        if token.dep_ == "prep":
            children = list(
                token.children
            )

            if any(
                child.pos_
                in {
                    "NOUN",
                    "PROPN",
                }
                for child in children
            ):
                return True

    return False


def _has_temporal_context(doc) -> bool:
    if doc is None:
        return False

    return any(
        entity.label_
        in {
            "DATE",
            "TIME",
        }
        for entity in getattr(
            doc,
            "ents",
            [],
        )
    )


def _has_situational_structure(
    text: str,
    doc,
) -> bool:
    """
    Require an actual event/circumstance.

    A statement about wanting advice is not automatically a
    situation. A concrete event such as being somewhere yesterday
    is.
    """

    if doc is None:
        return False

    if not _has_event_structure(
        doc
    ):
        return False

    if not _has_first_person_reference(
        doc
    ):
        return False

    has_setting = _has_setting_or_context(
        doc
    )

    has_time = _has_temporal_context(
        doc
    )

    content_count = len(
        _content_tokens(doc)
    )

    if (
        has_setting
        and content_count >= 2
    ):
        return True

    if (
        has_time
        and content_count >= 2
    ):
        return True

    # First-person event with richer structure.
    has_relation = any(
        token.dep_
        in {
            "advcl",
            "ccomp",
            "xcomp",
        }
        for token in doc
    )

    if (
        has_relation
        and content_count >= 3
    ):
        return True

    # Short real-time voice turns rarely reach the thresholds
    # above even when something situational was actually said.
    # A first-person event verb with a concrete NOUN/PROPN object
    # (e.g. "I skipped dinner", "I missed my train") is still a
    # situational statement on its own.
    for token in doc:
        if token.pos_ != "VERB":
            continue

        if token.lemma_.lower() in {
            "be",
            "have",
            "do",
            "want",
            "know",
            "tell",
        }:
            continue

        has_object = any(
            child.pos_ in {"NOUN", "PROPN"}
            for child in token.children
        )

        if has_object and content_count >= 2:
            return True

    return False


def _build_situational_factors(
    statements: list[dict],
) -> list[dict]:
    situations = []

    for statement in statements:
        text = _clean_text(
            statement.get("text")
        )

        if not text:
            continue

        doc = _process_text(text)

        if not _has_situational_structure(
            text,
            doc,
        ):
            continue

        situations.append({
            "text": text,
            "turn_ids": statement.get(
                "turn_ids",
                [],
            ),
        })

    return _unique_dicts(
        situations,
        ("text",),
    )


# =========================================================
# QUESTIONS
# =========================================================

def _extract_questions(
    statements: list[dict],
) -> list[dict]:
    questions = []

    for statement in statements:
        text = _clean_text(
            statement.get("text")
        )

        if not text:
            continue

        for sentence in _extract_sentences(
            text
        ):
            if "?" not in sentence:
                continue

            questions.append({
                "text": sentence,
                "turn_ids": statement.get(
                    "turn_ids",
                    [],
                ),
            })

    return _unique_dicts(
        questions,
        ("text",),
    )


# =========================================================
# OBSERVED BEHAVIORAL PATTERNS
# =========================================================

def _meaningful_short_turn(
    text: str,
) -> bool:
    """
    A short response only counts as a behavioral candidate when
    it actually contains semantic information.

    Greetings, gratitude, fillers and empty responses do not count.
    """

    if not _has_semantic_information(
        text
    ):
        return False

    doc = _process_text(text)

    if doc is None:
        return len(
            re.findall(
                r"\b[\w']+\b",
                text,
            )
        ) <= 3

    lexical = _meaningful_lexical_tokens(
        doc
    )

    return len(lexical) <= 3


def _build_observed_behavioral_patterns(
    turns: list[dict],
) -> list[dict]:
    """
    Behavioral patterns require repetition.

    A single short utterance is not considered a meaningful
    behavioral pattern.
    """

    short_turns = []

    for turn in turns:
        user_text = _get_user_text(
            turn
        )

        if not user_text:
            continue

        if _meaningful_short_turn(
            user_text
        ):
            short_turns.append(
                _get_turn_id(turn)
            )

    patterns = []

    # Require repeated meaningful short responses.
    if len(short_turns) >= 2:
        patterns.append({
            "signal": "repeated_short_user_responses",
            "description": (
                "User produced multiple meaningful "
                "responses containing three or fewer "
                "lexical tokens."
            ),
            "turns": short_turns,
            "confidence": "observed",
        })

    return patterns


# =========================================================
# EMOTION PROCESSING
# =========================================================

def _get_emotional_observations(
    turn: dict,
) -> dict:
    observations = turn.get(
        "emotional_observations"
    )

    if not isinstance(
        observations,
        dict,
    ):
        return {
            "observation_count": 0,
            "average_probabilities": {},
            "dominant_emotion": None,
            "timeline": [],
        }

    return observations


def _get_observation_count(
    turn: dict,
) -> int:
    observations = _get_emotional_observations(
        turn
    )

    value = observations.get(
        "observation_count",
        0,
    )

    try:
        return max(
            0,
            int(value),
        )
    except (
        TypeError,
        ValueError,
    ):
        return 0


def _get_dominant_emotion(
    turn: dict,
) -> str | None:
    """
    FER emotion is returned only when actual observations exist.
    """

    observations = _get_emotional_observations(
        turn
    )

    observation_count = _get_observation_count(
        turn
    )

    if observation_count <= 0:
        return None

    dominant = observations.get(
        "dominant_emotion"
    )

    if (
        isinstance(dominant, str)
        and dominant in EMOTION_CLASSES
    ):
        return dominant

    probabilities = observations.get(
        "average_probabilities",
        {},
    )

    if not isinstance(
        probabilities,
        dict,
    ):
        return None

    valid = {}

    for emotion, value in probabilities.items():
        if emotion not in EMOTION_CLASSES:
            continue

        try:
            valid[emotion] = float(value)
        except (
            TypeError,
            ValueError,
        ):
            continue

    if not valid:
        return None

    return max(
        valid,
        key=valid.get,
    )


def _get_supporting_emotions(
    turn: dict,
) -> list[str]:
    if _get_observation_count(
        turn
    ) <= 0:
        return []

    observations = _get_emotional_observations(
        turn
    )

    probabilities = observations.get(
        "average_probabilities",
        {},
    )

    if not isinstance(
        probabilities,
        dict,
    ):
        return []

    dominant = _get_dominant_emotion(
        turn
    )

    ranked = []

    for emotion in EMOTION_CLASSES:
        if emotion == dominant:
            continue

        value = _safe_float(
            probabilities.get(
                emotion,
                0.0,
            )
        )

        if value > 0:
            ranked.append(
                (emotion, value)
            )

    ranked.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        emotion
        for emotion, _ in ranked[:2]
    ]


# =========================================================
# BEHAVIORAL SIGNALS
# =========================================================

def _build_behavioral_signals(
    turn: dict,
) -> list[dict]:
    signals = []

    emotion = _get_dominant_emotion(
        turn
    )

    observation_count = _get_observation_count(
        turn
    )

    if (
        emotion
        and observation_count > 0
    ):
        signals.append({
            "signal": (
                f"facial_emotion:{emotion}"
            ),
            "source": "FER",
            "evidence": (
                f"{observation_count} FER "
                "observation(s) associated "
                "with this user turn."
            ),
            "confidence": "model_output",
        })

    user = turn.get("user")

    if not isinstance(
        user,
        dict,
    ):
        user = {}

    speech_interval = user.get(
        "speech_interval"
    )

    if isinstance(
        speech_interval,
        dict,
    ):
        duration = speech_interval.get(
            "duration_seconds"
        )

        if isinstance(
            duration,
            (int, float),
        ):
            signals.append({
                "signal": "speech_duration",
                "source": "browser_vad",
                "evidence": (
                    "User speech duration: "
                    f"{round(float(duration), 3)} "
                    "seconds."
                ),
                "confidence": "observed",
            })

    return signals


# =========================================================
# TURN KEY POINTS
# =========================================================

def _build_turn_key_points(
    text: str,
) -> list[str]:
    if not text:
        return []

    if not _has_semantic_information(
        text
    ):
        return []

    points = []

    phrases = _extract_meaningful_phrases(
        text
    )

    for phrase in phrases[:5]:
        if _concept_is_useful(
            phrase
        ):
            points.append(
                f"concept:{phrase}"
            )

    doc = _process_text(text)

    if doc is not None:
        for entity in doc.ents:
            entity_text = _clean_text(
                entity.text
            )

            if entity_text:
                points.append(
                    f"entity:{entity_text}"
                )

    for question in _extract_questions([
        {
            "text": text,
            "turn_ids": [],
        }
    ]):
        points.append(
            f"question:{question['text']}"
        )

    for temporal in _extract_temporal_context([
        {
            "text": text,
            "turn_ids": [],
        }
    ]):
        points.append(
            f"temporal:{temporal['text']}"
        )

    return _unique_strings(
        points
    )[:8]


# =========================================================
# DIAGNOSTICS
# =========================================================

def _build_diagnostics(
    *,
    total_turns: int,
    user_statements: list[dict],
    stated_concerns: list[dict],
    situational_factors: list[dict],
    dynamic_analysis: dict,
) -> dict:
    """
    Surface model-load status and per-stage extraction counts.

    _get_spacy_model()/_get_embedding_model() intentionally
    degrade quietly to weaker fallbacks on failure (P7), which
    makes pipeline regressions invisible from the JSON output
    alone. This block makes that state explicit and cheap to
    check without re-deriving it from source.
    """

    spacy_model = _get_spacy_model()
    embedding_model = _get_embedding_model()

    return {
        "spacy_model_loaded": spacy_model is not None,
        "embedding_model_loaded": embedding_model is not None,
        "total_turns": total_turns,
        "user_statements_extracted": len(
            user_statements
        ),
        "stated_concerns_extracted": len(
            stated_concerns
        ),
        "situational_factors_extracted": len(
            situational_factors
        ),
        "important_facts_extracted": len(
            dynamic_analysis.get(
                "important_facts", []
            )
        ),
        "user_intents_extracted": len(
            dynamic_analysis.get(
                "user_intents", []
            )
        ),
        "key_concepts_extracted": len(
            dynamic_analysis.get(
                "key_concepts", []
            )
        ),
    }


# =========================================================
# CONVERSATION TURN REPRESENTATION
# =========================================================

def _build_conversation_turn(
    turn: dict,
) -> dict:
    user_text = _get_user_text(
        turn
    )

    assistant_text = _get_assistant_text(
        turn
    )

    dominant_emotion = _get_dominant_emotion(
        turn
    )

    supporting_signals = (
        _get_supporting_emotions(
            turn
        )
    )

    return {
        "turn_id": _get_turn_id(turn),

        "user": {
            "text": user_text,
            "key_points": (
                _build_turn_key_points(
                    user_text
                )
            ),
        },

        "mindo": {
            "response": assistant_text,
        },

        "emotion": {
            "dominant": dominant_emotion,
            "supporting_signals": (
                supporting_signals
            ),
            "source": "FER",
        },
    }


# =========================================================
# DYNAMIC SEMANTIC ANALYSIS
# =========================================================

def _build_dynamic_analysis(
    turns: list[dict],
) -> dict:
    user_statements = _build_user_statements(
        turns
    )

    key_concepts = _build_key_concepts(
        user_statements,
        max_items=8,
    )

    semantic_groups = (
        _semantic_similarity_groups(
            user_statements
        )
    )

    user_intents = _infer_user_intents(
        user_statements
    )

    questions = _extract_questions(
        user_statements
    )

    important_facts = (
        _extract_important_facts(
            user_statements
        )
    )

    conversation_flow = (
        _build_conversation_pairs(
            turns
        )
    )

    return {
        "key_concepts": key_concepts,

        "semantic_groups": semantic_groups,

        "user_intents": user_intents,

        "questions": questions,

        "important_facts": important_facts,

        "conversation_flow": conversation_flow,
    }


# =========================================================
# MAIN BUILDER
# =========================================================

def build_semantic_context(
    session_context: dict,
) -> dict:
    """
    Build semantic_context.json from the existing
    session_context.json.

    The existing session_context.json remains untouched.

    Design principles:

        - USER language is the semantic source.
        - MINDO responses are traceability only.
        - Conversational filler is not semantic context.
        - Gratitude is not a fact, concept, concern or request.
        - Complete user turns are preferred over fragments.
        - Semantic relevance is conservative.
        - Behavioral patterns require repeated evidence.
        - FER remains independent evidence.
        - VAD remains independent observational evidence.
        - No diagnosis is inferred from FER or language.
        - No LLM is used.
    """

    if not isinstance(
        session_context,
        dict,
    ):
        raise TypeError(
            "session_context must be a dictionary."
        )

    # -----------------------------------------------------
    # Session
    # -----------------------------------------------------

    session = session_context.get(
        "session",
        {},
    )

    if not isinstance(
        session,
        dict,
    ):
        session = {}

    # -----------------------------------------------------
    # Conversation
    # -----------------------------------------------------

    conversation = session_context.get(
        "conversation",
        {},
    )

    if not isinstance(
        conversation,
        dict,
    ):
        conversation = {}

    turns = conversation.get(
        "turns",
        [],
    )

    if not isinstance(
        turns,
        list,
    ):
        turns = []

    valid_turns = [
        turn
        for turn in turns
        if isinstance(
            turn,
            dict,
        )
    ]

    # -----------------------------------------------------
    # Conversation turns
    # -----------------------------------------------------

    semantic_turns = [
        _build_conversation_turn(
            turn
        )
        for turn in valid_turns
    ]

    # -----------------------------------------------------
    # Semantic extraction
    # -----------------------------------------------------

    user_statements = _build_user_statements(
        valid_turns
    )

    stated_concerns = (
        _build_stated_concerns(
            user_statements
        )
    )

    situational_factors = (
        _build_situational_factors(
            user_statements
        )
    )

    dynamic_analysis = (
        _build_dynamic_analysis(
            valid_turns
        )
    )

    # -----------------------------------------------------
    # Final schema
    # -----------------------------------------------------

    return {
        "schema_version": "2.0",

        "session": {
            "session_id": session.get(
                "session_id"
            ),
            "start_time": session.get(
                "start_time"
            ),
            "end_time": session.get(
                "end_time"
            ),
            "duration_seconds": session.get(
                "duration_seconds"
            ),
            "total_turns": len(
                semantic_turns
            ),
        },

        "conversation": {
            "turns": semantic_turns,
        },


        "context": {
            "stated_concerns": (
                stated_concerns
            ),

            "situational_factors": (
                situational_factors
            ),

            "dynamic_analysis": (
                dynamic_analysis
            ),

        },

        "safety": {
            "assessment_status": "not_assessed",
        },
    }
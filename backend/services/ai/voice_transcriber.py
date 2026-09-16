import os
import time
import logging

from openai import OpenAI


logger = logging.getLogger(__name__)


GROQ_MODEL = os.getenv(
    "WHATSAPP_STT_GROQ_MODEL",
    "whisper-large-v3-turbo",
)

OPENAI_MODEL = os.getenv(
    "WHATSAPP_STT_OPENAI_MODEL",
    "whisper-1",
)

MAX_ATTEMPTS = 2
RETRY_DELAY_SECONDS = 1.5


def _clean_transcript(value):
    """Return a normalized non-empty transcript or an empty string."""
    if value is None:
        return ""

    text = str(value).strip()

    # Avoid treating obvious provider placeholders as customer speech.
    if text.lower() in {
        "null",
        "none",
        "undefined",
    }:
        return ""

    return text


def _build_stt_prompt(menu_items=None, language=None):
    """
    Build a conservative Whisper context prompt.

    The prompt provides vocabulary context only. It does not tell
    the model what the customer said.
    """

    parts = []

    if language:
        language = str(language).strip().lower()

        if language.startswith("fr"):
            parts.append(
                "Customer language is likely French."
            )
        elif language.startswith("en"):
            parts.append(
                "Customer language is likely English."
            )

    if menu_items:
        names = []

        for item in menu_items:
            if isinstance(item, dict):
                name = item.get("name")
                description = item.get("description")
                category = item.get("category")
            else:
                name = getattr(item, "name", None)
                description = getattr(item, "description", None)
                category = getattr(item, "category", None)

            if name:
                names.append(str(name).strip())

            # Descriptions/categories are useful context, but
            # don't make the prompt unnecessarily large.
            if description:
                names.append(str(description).strip()[:100])

            if category:
                names.append(str(category).strip())

        # Deduplicate while preserving order.
        seen = set()
        cleaned = []

        for value in names:
            value = " ".join(value.split())

            if not value:
                continue

            key = value.lower()

            if key in seen:
                continue

            seen.add(key)
            cleaned.append(value)

        if cleaned:
            parts.append(
                "Restaurant vocabulary and menu terms: "
                + ", ".join(cleaned[:80])
            )

    if not parts:
        return None

    return " ".join(parts)[:3500]


def _transcribe_with_groq(
    audio_file_path,
    language=None,
    menu_items=None,
):
    """Transcribe using Groq Whisper with restaurant context."""

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured."
        )

    from groq import Groq

    client = Groq(
        api_key=api_key,
        timeout=60.0,
    )

    request = {
        "file": None,
        "model": GROQ_MODEL,
        "response_format": "json",
    }

    if language:
        request["language"] = language

    prompt = _build_stt_prompt(
        menu_items=menu_items,
        language=language,
    )

    if prompt:
        request["prompt"] = prompt

    with open(audio_file_path, "rb") as audio_file:
        request["file"] = audio_file

        response = client.audio.transcriptions.create(
            **request
        )

    text = _clean_transcript(
        getattr(response, "text", "")
    )

    if not text:
        raise RuntimeError(
            "Groq returned an empty transcription."
        )

    return text


def _transcribe_with_openai(
    audio_file_path,
    language=None,
    menu_items=None,
):
    """Transcribe using OpenAI Whisper with restaurant context."""

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    client = OpenAI(
        api_key=api_key,
        timeout=60.0,
    )

    request = {
        "model": OPENAI_MODEL,
    }

    if language:
        request["language"] = language

    prompt = _build_stt_prompt(
        menu_items=menu_items,
        language=language,
    )

    if prompt:
        request["prompt"] = prompt

    with open(audio_file_path, "rb") as audio_file:
        request["file"] = audio_file

    response = client.audio.transcriptions.create(
        **request
    )

    text = _clean_transcript(
        getattr(response, "text", "")
    )

    if not text:
        raise RuntimeError(
            "OpenAI returned an empty transcription."
        )

    return text


def _menu_context_score(text, menu_items=None):
    """
    Measure whether a transcript contains vocabulary that exists
    in this restaurant's actual menu.

    This is contextual evidence, not permission to invent a menu item.
    """

    if not text or not menu_items:
        return 0.0

    normalized = _normalize_for_quality(text)

    if not normalized:
        return 0.0

    score = 0.0

    for item in menu_items:
        if isinstance(item, dict):
            name = item.get("name")
            description = item.get("description")
            category = item.get("category")
        else:
            name = getattr(item, "name", None)
            description = getattr(item, "description", None)
            category = getattr(item, "category", None)

        candidates = (
            name,
            description,
            category,
        )

        for candidate in candidates:
            candidate = _normalize_for_quality(candidate)

            if not candidate or len(candidate) < 3:
                continue

            if candidate in normalized:
                if candidate == _normalize_for_quality(name):
                    score += 1.0
                else:
                    score += 0.25

                break

    return min(score, 3.0)


def _candidate_score(
    text,
    menu_items=None,
    language=None,
):
    """
    Rank a transcription candidate conservatively.

    Menu matches are useful evidence, but a transcript never becomes
    an order merely because it contains a menu word.
    """

    quality = _transcript_quality_score(text)

    menu_score = _menu_context_score(
        text,
        menu_items=menu_items,
    )

    score = float(quality)

    # A direct menu-name match is meaningful evidence.
    score += min(menu_score, 2.0) * 12.0

    # Preserve short but valid utterances.
    if len((text or "").split()) == 1 and menu_score > 0:
        score += 8.0

    return score


def _deduplicate_candidates(candidates):
    seen = set()
    result = []

    for candidate in candidates:
        candidate = _clean_transcript(candidate)

        if not candidate:
            continue

        key = _normalize_for_quality(candidate)

        if key in seen:
            continue

        seen.add(key)
        result.append(candidate)

    return result



def transcribe_audio(
    audio_file_path,
    language=None,
    menu_items=None,
):
    """
    Production restaurant speech-to-text pipeline.

    Provider strategy:
        1. Primary provider attempt
        2. Retry
        3. Alternate provider
        4. Context-aware candidate ranking

    Restaurant menu context is supplied to the speech model as
    vocabulary context and used for conservative candidate ranking.

    The returned text is still passed to the normal deterministic
    restaurant agent, which remains authoritative.
    """

    if not audio_file_path:
        raise RuntimeError(
            "Audio file path is missing."
        )

    if not os.path.isfile(audio_file_path):
        raise RuntimeError(
            "Downloaded audio file does not exist."
        )

    file_size = os.path.getsize(
        audio_file_path
    )

    if file_size < 100:
        raise RuntimeError(
            "Downloaded audio file is empty or invalid."
        )

    providers = []

    if os.getenv("GROQ_API_KEY"):
        providers.append(
            ("groq", _transcribe_with_groq)
        )

    if os.getenv("OPENAI_API_KEY"):
        providers.append(
            ("openai", _transcribe_with_openai)
        )

    if not providers:
        raise RuntimeError(
            "No speech-to-text provider is configured."
        )

    candidates = []
    failures = []

    # --------------------------------------------------------
    # First pass: collect a candidate from each provider.
    # We don't immediately trust the first transcript.
    # --------------------------------------------------------
    for provider_name, provider in providers:

        for attempt in range(1, MAX_ATTEMPTS + 1):

            started = time.monotonic()

            try:
                logger.info(
                    "[VOICE] STT provider=%s attempt=%s",
                    provider_name,
                    attempt,
                )

                text = provider(
                    audio_file_path,
                    language=language,
                    menu_items=menu_items,
                )

                elapsed = time.monotonic() - started

                quality = _transcript_quality_score(
                    text
                )

                contextual = _menu_context_score(
                    text,
                    menu_items=menu_items,
                )

                score = _candidate_score(
                    text,
                    menu_items=menu_items,
                    language=language,
                )

                logger.info(
                    "[VOICE] STT candidate provider=%s "
                    "attempt=%s quality=%s menu_score=%.2f "
                    "candidate_score=%.2f elapsed=%.3fs",
                    provider_name,
                    attempt,
                    quality,
                    contextual,
                    score,
                    elapsed,
                )

                if text:
                    candidates.append({
                        "text": text,
                        "provider": provider_name,
                        "attempt": attempt,
                        "score": score,
                    })

                # A strong menu-aware candidate is good enough
                # to avoid wasting another provider call.
                if (
                    contextual >= 1.0
                    and quality >= 70
                ):
                    break

                # Otherwise try this provider once more.
                if attempt < MAX_ATTEMPTS:
                    time.sleep(
                        RETRY_DELAY_SECONDS
                    )

            except Exception as exc:

                elapsed = time.monotonic() - started

                failures.append(
                    f"{provider_name} attempt {attempt}: "
                    f"{type(exc).__name__}: {exc}"
                )

                logger.warning(
                    "[VOICE] STT failed provider=%s "
                    "attempt=%s elapsed=%.3fs",
                    provider_name,
                    attempt,
                    elapsed,
                )

                if attempt < MAX_ATTEMPTS:
                    time.sleep(
                        RETRY_DELAY_SECONDS
                    )

    # --------------------------------------------------------
    # If we have candidates, choose the strongest one.
    # --------------------------------------------------------
    if candidates:

        candidates = sorted(
            candidates,
            key=lambda item: item["score"],
            reverse=True,
        )

        best = candidates[0]

        logger.info(
            "[VOICE] STT selected provider=%s score=%.2f "
            "candidates=%s",
            best["provider"],
            best["score"],
            len(candidates),
        )

        return best["text"]

    raise RuntimeError(
        "All configured speech-to-text providers failed: "
        + " | ".join(failures)
    )

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


def _transcribe_with_groq(audio_file_path, language=None):
    """Transcribe using Groq Whisper."""
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


def _transcribe_with_openai(audio_file_path, language=None):
    """Transcribe using OpenAI Whisper as a provider fallback."""
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


def transcribe_audio(
    audio_file_path,
    language=None,
):
    """
    Production WhatsApp speech-to-text pipeline.

    Provider order:
        1. Groq Whisper
        2. OpenAI Whisper fallback

    Each configured provider receives up to two attempts.
    Language is intentionally optional so Whisper can automatically
    detect French, English, or another supported language.
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

    failures = []

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
                )

                elapsed = time.monotonic() - started

                logger.info(
                    "[VOICE] STT success provider=%s "
                    "attempt=%s elapsed=%.3fs",
                    provider_name,
                    attempt,
                    elapsed,
                )

                return text

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

    raise RuntimeError(
        "All configured speech-to-text providers failed: "
        + " | ".join(failures)
    )

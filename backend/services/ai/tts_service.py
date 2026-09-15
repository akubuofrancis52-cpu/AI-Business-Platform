"""
Botify AI — Provider-independent text-to-speech service.

Provider order:
    1. Piper (local, when configured)
    2. OpenAI TTS fallback

The rest of the application should only call synthesize_speech().
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

VOICE_MODEL_DIR = Path(
    os.getenv(
        "WHATSAPP_TTS_PIPER_MODEL_DIR",
        str(
            Path(__file__).resolve().parents[2]
            / "voice_models"
            / "piper"
        ),
    )
).resolve()

PIPER_MODEL = os.getenv(
    "WHATSAPP_TTS_PIPER_MODEL",
    "",
).strip()
PIPER_EXECUTABLE = os.getenv(
    "WHATSAPP_TTS_PIPER_EXECUTABLE",
    "piper",
).strip()

OPENAI_TTS_MODEL = os.getenv(
    "WHATSAPP_TTS_OPENAI_MODEL",
    "gpt-4o-mini-tts",
).strip()

OPENAI_TTS_VOICE = os.getenv(
    "WHATSAPP_TTS_OPENAI_VOICE",
    "alloy",
).strip()

MAX_TEXT_LENGTH = int(
    os.getenv("WHATSAPP_TTS_MAX_TEXT_LENGTH", "1200")
)


def _clean_text(text: str) -> str:
    if text is None:
        return ""

    value = " ".join(str(text).split())

    if not value:
        return ""

    return value[:MAX_TEXT_LENGTH]


def _language_to_piper_voice(language: str | None) -> str | None:
    """
    Optional language-specific Piper model override.

    This lets production configure separate French/English models
    without changing application code.
    """
    language = (language or "").lower().strip()

    if language.startswith("fr"):
        configured = os.getenv(
            "WHATSAPP_TTS_PIPER_MODEL_FR",
            "",
        ).strip()

        if configured:
            return configured

        return str(
            VOICE_MODEL_DIR
            / "fr_FR-siwis-medium.onnx"
        )

    if language.startswith("en"):
        configured = os.getenv(
            "WHATSAPP_TTS_PIPER_MODEL_EN",
            "",
        ).strip()

        if configured:
            return configured

        return str(
            VOICE_MODEL_DIR
            / "en_US-lessac-medium.onnx"
        )

    if PIPER_MODEL:
        return PIPER_MODEL

    # Default to English when the caller has not supplied a
    # language. The restaurant agent normally supplies one.
    return str(
        VOICE_MODEL_DIR
        / "en_US-lessac-medium.onnx"
    )


def _synthesize_with_piper(text: str, output_path: str, language=None):
    """
    Generate WAV audio with Piper.

    Piper is optional. If no model is configured, this provider is skipped.
    """
    model = _language_to_piper_voice(language)

    if not model:
        raise RuntimeError("Piper model is not configured.")

    model_path = Path(model)

    if not model_path.is_file():
        raise RuntimeError(
            f"Piper model does not exist: {model}"
        )

    config_path = Path(
        str(model_path) + ".json"
    )

    if not config_path.is_file():
        raise RuntimeError(
            f"Piper model config does not exist: {config_path}"
        )

    executable = PIPER_EXECUTABLE

    command = [
        executable,
        "--model",
        model,
        "--output_file",
        output_path,
    ]

    started = time.monotonic()

    result = subprocess.run(
        command,
        input=text,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )

    elapsed = time.monotonic() - started

    if result.returncode != 0:
        raise RuntimeError(
            "Piper failed: "
            + (result.stderr or result.stdout or "unknown error")[-1000:]
        )

    if not os.path.isfile(output_path):
        raise RuntimeError("Piper produced no audio file.")

    if os.path.getsize(output_path) < 100:
        raise RuntimeError("Piper produced an empty audio file.")

    logger.info(
        "[VOICE] TTS success provider=piper elapsed=%.3fs",
        elapsed,
    )

    return output_path


def _synthesize_with_openai(text: str, output_path: str, language=None):
    """Generate MP3 audio using OpenAI TTS as a cloud fallback."""
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        timeout=60.0,
    )

    started = time.monotonic()

    with client.audio.speech.with_streaming_response.create(
        model=OPENAI_TTS_MODEL,
        voice=OPENAI_TTS_VOICE,
        input=text,
        response_format="mp3",
    ) as response:
        response.stream_to_file(output_path)

    elapsed = time.monotonic() - started

    if not os.path.isfile(output_path):
        raise RuntimeError("OpenAI TTS produced no audio file.")

    if os.path.getsize(output_path) < 100:
        raise RuntimeError("OpenAI TTS produced an empty audio file.")

    logger.info(
        "[VOICE] TTS success provider=openai elapsed=%.3fs",
        elapsed,
    )

    return output_path


def synthesize_speech(text: str, language=None) -> str:
    """
    Synthesize speech and return a temporary audio file path.

    Piper is preferred when a valid model is configured.
    OpenAI TTS is used as a reliable fallback.

    Caller owns the returned temporary file and must delete it.
    """
    text = _clean_text(text)

    if not text:
        raise RuntimeError("TTS received empty text.")

    failures = []

    providers = []

    if _language_to_piper_voice(language):
        providers.append(("piper", _synthesize_with_piper))

    if os.getenv("OPENAI_API_KEY"):
        providers.append(("openai", _synthesize_with_openai))

    if not providers:
        raise RuntimeError(
            "No text-to-speech provider is configured."
        )

    for provider_name, provider in providers:
        suffix = ".wav" if provider_name == "piper" else ".mp3"

        fd, output_path = tempfile.mkstemp(
            prefix="botify_tts_",
            suffix=suffix,
        )
        os.close(fd)

        try:
            provider(
                text,
                output_path,
                language=language,
            )
            return output_path

        except Exception as exc:
            failures.append(
                f"{provider_name}: {type(exc).__name__}: {exc}"
            )

            logger.warning(
                "[VOICE] TTS provider failed provider=%s error=%s",
                provider_name,
                type(exc).__name__,
            )

            try:
                os.remove(output_path)
            except OSError:
                pass

    raise RuntimeError(
        "All configured TTS providers failed: "
        + " | ".join(failures)
    )

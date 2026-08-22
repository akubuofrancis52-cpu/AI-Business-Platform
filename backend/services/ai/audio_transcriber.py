import os
import base64
import logging
import requests

from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger(__name__)


OPENROUTER_TRANSCRIPTION_URL = (
    "https://openrouter.ai/api/v1/audio/transcriptions"
)

TRANSCRIPTION_MODEL = os.getenv(
    "OPENROUTER_TRANSCRIPTION_MODEL",
    "openai/whisper-1",
)


def transcribe_audio(
    audio_bytes,
    audio_format="ogg",
    language=None,
):
    """
    Transcribe audio bytes using OpenRouter speech-to-text.

    Returns:
        str: Transcribed text.

    Raises:
        RuntimeError: If transcription fails.
    """

    if not audio_bytes:
        raise ValueError(
            "No audio data was provided."
        )

    api_key = os.getenv(
        "OPENROUTER_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing."
        )

    audio_base64 = base64.b64encode(
        audio_bytes
    ).decode("utf-8")

    payload = {
        "model": TRANSCRIPTION_MODEL,
        "input_audio": {
            "data": audio_base64,
            "format": audio_format,
        },
    }

    if language:
        payload["language"] = language

    try:

        response = requests.post(
            OPENROUTER_TRANSCRIPTION_URL,

            headers={
                "Authorization": (
                    f"Bearer {api_key}"
                ),
                "Content-Type": (
                    "application/json"
                ),
            },

            json=payload,

            timeout=60,
        )

        if not response.ok:

            logger.error(
                "OpenRouter transcription error "
                "%s: %s",
                response.status_code,
                response.text[:1000],
            )

            raise RuntimeError(
                "Audio transcription failed."
            )

        result = response.json()

        transcript = (
            result.get("text")
            or ""
        ).strip()

        if not transcript:

            raise RuntimeError(
                "Audio transcription returned "
                "no text."
            )

        return transcript

    except requests.RequestException as exc:

        logger.exception(
            "OpenRouter transcription request failed."
        )

        raise RuntimeError(
            "Could not connect to the "
            "transcription service."
        ) from exc
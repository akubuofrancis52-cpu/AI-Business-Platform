import os
import logging

from openai import OpenAI


logger = logging.getLogger(__name__)


def transcribe_audio(
    audio_file_path,
    language=None
):
    """
    Transcribe a downloaded WhatsApp audio file.

    Returns:
        str: Transcribed customer message.
    """

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    client = OpenAI(
        api_key=api_key
    )

    try:

        with open(
            audio_file_path,
            "rb"
        ) as audio_file:

            request = {
                "model": "whisper-1",
                "file": audio_file,
            }

            if language:
                request["language"] = language

            response = (
                client.audio.transcriptions.create(
                    **request
                )
            )

        text = (
            getattr(
                response,
                "text",
                ""
            )
            or ""
        ).strip()

        if not text:
            raise RuntimeError(
                "Audio transcription returned empty text."
            )

        return text

    except Exception as exc:

        logger.exception(
            "Audio transcription failed."
        )

        raise RuntimeError(
            f"Audio transcription failed: {exc}"
        ) from exc
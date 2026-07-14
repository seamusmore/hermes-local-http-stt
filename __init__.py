"""
Local HTTP STT Plugin
=====================

A TranscriptionProvider that POSTs audio files to a self-hosted
HTTP STT service (e.g. sensevoice / whisper) at ``stt.http_stt.service_url``.

This replaces the native ``http_stt`` handler that was inlined in
``tools/transcription_tools.py`` and keeps the core clean — the STT
client lives entirely in this plugin.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.transcription_provider import TranscriptionProvider

logger = logging.getLogger(__name__)


def _load_http_stt_config() -> dict:
    """Read the ``stt.http_stt`` section from config.yaml."""
    try:
        from hermes_cli.config import load_config
        return load_config().get("stt", {}).get("http_stt", {})
    except Exception:
        return {}


class LocalHttpSTTProvider(TranscriptionProvider):
    """Speech-to-text via a local HTTP /transcribe endpoint."""

    @property
    def name(self) -> str:
        return "http_stt"

    @property
    def display_name(self) -> str:
        return "http_stt"

    def is_available(self) -> bool:
        cfg = _load_http_stt_config()
        url = cfg.get("service_url", "").strip()
        return bool(url)

    def list_models(self) -> List[Dict[str, Any]]:
        return []

    def default_model(self) -> Optional[str]:
        cfg = _load_http_stt_config()
        return cfg.get("model", "base")

    def transcribe(
        self,
        file_path: str,
        *,
        model: Optional[str] = None,
        language: Optional[str] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        cfg = _load_http_stt_config()
        service_url = str(cfg.get("service_url", "")).strip().rstrip("/")
        if not service_url:
            return {
                "success": False,
                "transcript": "",
                "error": "http_stt STT is not configured (stt.http_stt.service_url)",
                "provider": "http_stt",
            }
        lang = language or cfg.get("language", "auto")
        model_name = model or cfg.get("model", "base")

        try:
            import requests

            params: Dict[str, str] = {"language": lang, "model": model_name}
            with open(file_path, "rb") as audio_file:
                response = requests.post(
                    f"{service_url}/transcribe",
                    params=params,
                    files={"file": (Path(file_path).name, audio_file)},
                    timeout=300,
                )

            if response.status_code != 200:
                return {
                    "success": False,
                    "transcript": "",
                    "error": (
                        f"Local HTTP STT error (HTTP {response.status_code}): "
                        f"{response.text[:300]}"
                    ),
                    "provider": "http_stt",
                }

            result = response.json()
            transcript = (
                result.get("transcript") or result.get("text") or ""
            ).strip()
            if not transcript:
                return {
                    "success": False,
                    "transcript": "",
                    "error": "Local HTTP STT returned empty transcript",
                    "provider": "http_stt",
                }

            logger.info(
                "Transcribed %s via http_stt (%s, %d chars)",
                Path(file_path).name,
                model_name,
                len(transcript),
            )
            return {
                "success": True,
                "transcript": transcript,
                "provider": "http_stt",
            }

        except Exception as exc:
            logger.error("Local HTTP STT failed: %s", exc, exc_info=True)
            return {
                "success": False,
                "transcript": "",
                "error": f"Local HTTP STT failed: {exc}",
                "provider": "http_stt",
            }


def register(ctx) -> None:
    """Register the http_stt STT provider."""
    ctx.register_transcription_provider(LocalHttpSTTProvider())
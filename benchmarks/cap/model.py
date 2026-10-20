"""Single-call multimodal model adapter used by every CaP condition."""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from PIL import Image


# The GTA agents load repository configuration through config.py, which invokes
# load_dotenv(). The CaP CLI is standalone and may never import config.py before
# constructing its model client, so it must load the same environment itself.
load_dotenv()

GEMINI_REQUEST_TIMEOUT_MS = 500_000


class CapModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelResponse:
    text: str
    metadata: Dict[str, Any]


def _load_rgb_image(path: Path) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:
        raise CapModelError(f"Unable to read frozen scene image {path}: {exc}") from exc


def _pil_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


class CapModelClient:
    """Provider-neutral stateless policy generator.

    CAP_PROVIDER/CAP_MODEL can explicitly pin a benchmark model. If omitted,
    the Skill Generator provider/model are used so all CaP conditions share the
    same model family already configured for GTA-2. Provider request settings
    match BaseAgent except that no conversation/chat history is attached.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        openai_client=None,
        gemini_client=None,
    ):
        self.provider = (
            provider
            or os.getenv("CAP_PROVIDER")
            or os.getenv("SKILL_GENERATOR_PROVIDER")
            or ""
        ).lower()
        self.model = model or os.getenv("CAP_MODEL") or os.getenv("SKILL_GENERATOR_MODEL")
        if self.provider not in {"openai", "gemini"}:
            raise CapModelError("CAP_PROVIDER must be 'openai' or 'gemini'")
        if not self.model:
            raise CapModelError("Set CAP_MODEL or SKILL_GENERATOR_MODEL")
        self.api_key = api_key or os.getenv(
            "OPENAI_API_KEY" if self.provider == "openai" else "GEMINI_API_KEY"
        )
        if not self.api_key and openai_client is None and gemini_client is None:
            raise CapModelError(f"API key is missing for provider {self.provider}")
        self._openai_client = openai_client
        self._gemini_client = gemini_client

    def generate(self, system_prompt: str, user_prompt: str, image_path: Path) -> ModelResponse:
        image = _load_rgb_image(Path(image_path))
        if self.provider == "openai":
            return self._generate_openai(system_prompt, user_prompt, image)
        return self._generate_gemini(system_prompt, user_prompt, image)

    def _generate_openai(self, system_prompt, user_prompt, image):
        if self._openai_client is None:
            from openai import OpenAI

            self._openai_client = OpenAI(api_key=self.api_key)
        try:
            response = self._openai_client.responses.create(
                model=self.model,
                instructions=system_prompt,
                reasoning={"effort": "high"},
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": user_prompt},
                            {"type": "input_image", "image_url": _pil_data_url(image)},
                        ],
                    }
                ],
            )
        except Exception as exc:
            raise CapModelError(f"OpenAI policy generation failed: {exc}") from exc

        text = getattr(response, "output_text", None)
        if not text:
            pieces = []
            for item in getattr(response, "output", []) or []:
                for part in getattr(item, "content", []) or []:
                    value = getattr(part, "text", None)
                    if value:
                        pieces.append(value)
            text = "\n".join(pieces)
        if not text:
            raise CapModelError("OpenAI returned no policy source")
        usage = getattr(response, "usage", None)
        metadata = {
            "provider": "openai",
            "requested_model": self.model,
            "served_model": getattr(response, "model", None) or self.model,
            "response_id": getattr(response, "id", None),
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "reasoning_effort": "high",
            "stateful": False,
        }
        return ModelResponse(text=text, metadata=metadata)

    def _generate_gemini(self, system_prompt, user_prompt, image):
        from google.genai import types

        if self._gemini_client is None:
            from google import genai

            self._gemini_client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(timeout=GEMINI_REQUEST_TIMEOUT_MS),
            )
        try:
            print(
                f"[CaP Model] Image loaded; waiting for Gemini policy response "
                f"({self.model}, timeout=500s)...",
                flush=True,
            )
            response = self._gemini_client.models.generate_content(
                model=self.model,
                contents=[user_prompt, image],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                ),
            )
            print("[CaP Model] Gemini policy response received.", flush=True)
        except Exception as exc:
            raise CapModelError(f"Gemini policy generation failed: {exc}") from exc
        text = getattr(response, "text", None)
        if not text:
            raise CapModelError("Gemini returned no policy source")
        usage = getattr(response, "usage_metadata", None)
        metadata = {
            "provider": "gemini",
            "requested_model": self.model,
            "served_model": getattr(response, "model_version", None) or self.model,
            "response_id": getattr(response, "response_id", None),
            "input_tokens": getattr(usage, "prompt_token_count", None),
            "output_tokens": getattr(usage, "candidates_token_count", None),
            "stateful": False,
        }
        return ModelResponse(text=text, metadata=metadata)

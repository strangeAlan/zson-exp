"""Thin client for the existing local Qwen multimodal generation service."""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass

import numpy as np
import requests
from PIL import Image


class QwenServiceError(RuntimeError):
    """Raised when the local Qwen service violates its runtime contract."""


@dataclass(frozen=True)
class QwenEndpoint:
    base_url: str = "http://127.0.0.1:18080"
    connect_timeout: float = 3.0
    read_timeout: float = 180.0
    max_new_tokens: int = 512
    api_style: str = "native"
    model: str = "qwen3-vl-8b"

    @classmethod
    def from_environment(cls) -> "QwenEndpoint":
        return cls(
            base_url=os.environ.get(
                "ZSON3_QWEN_BASE_URL", "http://127.0.0.1:18080"
            ).rstrip("/"),
            connect_timeout=float(os.environ.get("ZSON3_QWEN_CONNECT_TIMEOUT", "3")),
            read_timeout=float(os.environ.get("ZSON3_QWEN_READ_TIMEOUT", "180")),
            max_new_tokens=int(os.environ.get("ZSON3_QWEN_MAX_NEW_TOKENS", "512")),
            api_style=os.environ.get("ZSON3_QWEN_API_STYLE", "native"),
            model=os.environ.get("ZSON3_QWEN_MODEL", "qwen3-vl-8b"),
        )


class QwenClient:
    """Protocol adapter; it does not own or load the model process."""

    def __init__(self, endpoint: QwenEndpoint | None = None) -> None:
        self.endpoint = endpoint or QwenEndpoint.from_environment()
        self.session = requests.Session()
        # Local model traffic must not be routed through machine-wide proxies.
        self.session.trust_env = False

    @staticmethod
    def _image_data_url(image: np.ndarray | Image.Image) -> str:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8, copy=False))
        if not isinstance(image, Image.Image):
            raise TypeError(f"Unsupported Qwen image type: {type(image)!r}")
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def health(self) -> dict:
        path = "/v1/models" if self.endpoint.api_style == "openai" else "/health"
        try:
            response = self.session.get(
                f"{self.endpoint.base_url}{path}",
                timeout=(self.endpoint.connect_timeout, self.endpoint.connect_timeout),
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as error:
            raise QwenServiceError(
                f"Qwen health check failed at {self.endpoint.base_url}: {error}"
            ) from error
        if self.endpoint.api_style == "openai":
            models = [item.get("id") for item in payload.get("data", [])]
            if self.endpoint.model not in models:
                raise QwenServiceError(
                    f"Qwen model {self.endpoint.model!r} not served: {models!r}"
                )
            return {
                "ok": True,
                "backend": "vllm-openai",
                "model": self.endpoint.model,
            }
        if payload.get("ok") is not True:
            raise QwenServiceError(f"Unexpected Qwen health payload: {payload!r}")
        return payload

    def generate(self, *, prompt: str, image: np.ndarray | Image.Image) -> str:
        image_url = self._image_data_url(image)
        if self.endpoint.api_style == "openai":
            payload = {
                "model": self.endpoint.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                "max_tokens": self.endpoint.max_new_tokens,
                "temperature": 0.0,
            }
            path = "/v1/chat/completions"
        else:
            payload = {
                "prompt": prompt,
                "images": [image_url],
                "max_new_tokens": self.endpoint.max_new_tokens,
            }
            path = "/generate"
        try:
            response = self.session.post(
                f"{self.endpoint.base_url}{path}",
                json=payload,
                timeout=(self.endpoint.connect_timeout, self.endpoint.read_timeout),
            )
            response.raise_for_status()
            result = response.json()
        except (requests.RequestException, ValueError) as error:
            raise QwenServiceError(
                f"Qwen generation failed at {self.endpoint.base_url}: {error}"
            ) from error
        if self.endpoint.api_style == "openai":
            choices = result.get("choices", [])
            text = choices[0].get("message", {}).get("content") if choices else None
        else:
            text = result.get("text")
        if not isinstance(text, str) or not text.strip():
            raise QwenServiceError(f"Unexpected Qwen generation payload: {result!r}")
        return text

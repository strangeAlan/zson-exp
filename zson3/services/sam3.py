"""Client for the out-of-process OpenFrontier SAM3 segmentation service."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import requests

from utils.sam3_transport import decode_masks, encode_image


class Sam3ServiceError(RuntimeError):
    """Raised when SAM3 is unavailable or violates its response contract."""


@dataclass(frozen=True)
class Sam3Endpoint:
    base_url: str = "http://127.0.0.1:12186"
    connect_timeout: float = 3.0
    read_timeout: float = 180.0

    @classmethod
    def from_environment(cls) -> "Sam3Endpoint":
        return cls(
            base_url=os.environ.get(
                "ZSON3_SAM3_BASE_URL", "http://127.0.0.1:12186"
            ).rstrip("/"),
            connect_timeout=float(os.environ.get("ZSON3_SAM3_CONNECT_TIMEOUT", "3")),
            read_timeout=float(os.environ.get("ZSON3_SAM3_READ_TIMEOUT", "180")),
        )


class Sam3Client:
    def __init__(self, endpoint: Sam3Endpoint | None = None) -> None:
        self.endpoint = endpoint or Sam3Endpoint.from_environment()
        self.session = requests.Session()
        self.session.trust_env = False

    def health(self) -> dict:
        try:
            response = self.session.get(
                f"{self.endpoint.base_url}/health",
                timeout=(self.endpoint.connect_timeout, self.endpoint.connect_timeout),
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as error:
            raise Sam3ServiceError(
                f"SAM3 health check failed at {self.endpoint.base_url}: {error}"
            ) from error
        if payload.get("ok") is not True or payload.get("service") != "sam3":
            raise Sam3ServiceError(f"Unexpected SAM3 health payload: {payload!r}")
        return payload

    def segment(self, *, image: np.ndarray, prompt: str) -> dict:
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError(f"Expected HxWx3/4 image, got {image.shape}")
        try:
            response = self.session.post(
                f"{self.endpoint.base_url}/sam3",
                json={
                    **encode_image(image),
                    "prompt": prompt,
                    "response_format": "packed-v1",
                },
                timeout=(self.endpoint.connect_timeout, self.endpoint.read_timeout),
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as error:
            raise Sam3ServiceError(
                f"SAM3 inference failed at {self.endpoint.base_url}: {error}"
            ) from error
        if payload.get("result") != "success":
            raise Sam3ServiceError(
                f"SAM3 inference error: {payload.get('message', payload)!r}"
            )
        try:
            masks = decode_masks(payload)
        except (KeyError, TypeError, ValueError) as error:
            raise Sam3ServiceError(
                f"Unexpected SAM3 mask payload: {error}"
            ) from error
        boxes, scores = payload.get("boxes"), payload.get("scores")
        if not isinstance(boxes, list) or not isinstance(scores, list):
            raise Sam3ServiceError(f"Unexpected SAM3 inference payload: {payload!r}")
        if not (len(masks) == len(boxes) == len(scores)):
            raise Sam3ServiceError(
                f"SAM3 output lengths differ: masks={len(masks)}, boxes={len(boxes)}, scores={len(scores)}"
            )
        payload["masks"] = masks
        return payload

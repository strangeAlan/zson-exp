"""HTTP adapters for the already-running VLFM detector services.

The wire format matches the evaluated T1 clients while keeping all model
packages and checkpoints outside the ZSON3 runtime environment.
"""

from __future__ import annotations

import base64
import os
import socket
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
import requests


class ApexTargetServiceError(RuntimeError):
    pass


@dataclass(frozen=True)
class Detection:
    box_xyxy_normalized: np.ndarray
    confidence: float
    phrase: str


class ApexTargetServiceClient:
    def __init__(self) -> None:
        self.gdino_url = os.environ.get("ZSON3_GDINO_URL", "http://127.0.0.1:12181/gdino")
        self.yolo_url = os.environ.get("ZSON3_YOLOV7_URL", "http://127.0.0.1:12184/yolov7")
        self.mobile_sam_url = os.environ.get("ZSON3_MOBILE_SAM_URL", "http://127.0.0.1:12183/mobile_sam")
        self.connect_timeout = float(os.environ.get("ZSON3_APEX_CONNECT_TIMEOUT", "3"))
        self.read_timeout = float(os.environ.get("ZSON3_APEX_READ_TIMEOUT", "180"))
        self.session = requests.Session()
        self.session.trust_env = False

    def health(self) -> dict:
        endpoints = {
            "grounding_dino": self.gdino_url,
            "mobile_sam": self.mobile_sam_url,
            "yolov7": self.yolo_url,
        }
        status = {}
        for name, url in endpoints.items():
            authority = url.split("//", 1)[-1].split("/", 1)[0]
            host, port_text = authority.rsplit(":", 1)
            try:
                with socket.create_connection(
                    (host, int(port_text)), timeout=self.connect_timeout
                ):
                    status[name] = {"ok": True, "url": url}
            except OSError as error:
                raise ApexTargetServiceError(
                    f"{name} unavailable at {url}: {error}"
                ) from error
        return {"ok": True, "services": status}

    def detect(self, *, image: np.ndarray, backend: str, caption: str) -> List[Detection]:
        payload = {"image": _encode_jpeg(image)}
        if backend == "grounding_dino":
            url = self.gdino_url
            payload["caption"] = caption
        elif backend == "yolov7":
            url = self.yolo_url
        else:
            raise ValueError(f"Unsupported Apex target detector: {backend}")
        result = self._post(url, payload)
        boxes, logits, phrases = result.get("boxes"), result.get("logits"), result.get("phrases")
        if not isinstance(boxes, list) or not isinstance(logits, list) or not isinstance(phrases, list):
            raise ApexTargetServiceError(f"Malformed detector response from {url}: {result!r}")
        if not (len(boxes) == len(logits) == len(phrases)):
            raise ApexTargetServiceError(f"Detector response lengths differ from {url}")
        return [
            Detection(np.asarray(box, dtype=np.float32), float(score), str(phrase).strip().lower())
            for box, score, phrase in zip(boxes, logits, phrases)
        ]

    def segment_bbox(self, *, image: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray:
        result = self._post(
            self.mobile_sam_url,
            {"image": _encode_jpeg(image), "bbox": np.asarray(bbox_xyxy, dtype=float).tolist()},
        )
        encoded = result.get("cropped_mask")
        if not isinstance(encoded, str):
            raise ApexTargetServiceError(f"Malformed MobileSAM response: {result!r}")
        try:
            mask = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
            return mask.reshape(image.shape[:2])
        except (ValueError, TypeError) as error:
            raise ApexTargetServiceError(f"Invalid MobileSAM mask: {error}") from error

    def _post(self, url: str, payload: dict) -> dict:
        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=(self.connect_timeout, self.read_timeout),
            )
            response.raise_for_status()
            result = response.json()
        except (requests.RequestException, ValueError) as error:
            raise ApexTargetServiceError(f"Apex target service failed at {url}: {error}") from error
        if not isinstance(result, dict):
            raise ApexTargetServiceError(f"Unexpected response from {url}: {result!r}")
        return result


def _encode_jpeg(image: np.ndarray) -> str:
    # Deliberately matches the T1 service transport, including JPEG quality 90.
    ok, buffer = cv2.imencode(".jpg", np.asarray(image), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise ApexTargetServiceError("Failed to encode detector input image")
    return base64.b64encode(buffer).decode("utf-8")

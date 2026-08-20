"""Lossless wire encoding for the local SAM3 service.

This module deliberately depends only on NumPy and the Python standard
library so the isolated SAM3 environment does not inherit runtime-client
dependencies such as requests.
"""

from __future__ import annotations

import base64

import numpy as np


IMAGE_ENCODING = "raw-uint8-base64-v1"
MASK_ENCODING = "packbits-base64-v1"


def encode_image(image: np.ndarray) -> dict:
    array = np.ascontiguousarray(image, dtype=np.uint8)
    return {
        "image_encoding": IMAGE_ENCODING,
        "image_shape": list(array.shape),
        "image_data": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def decode_image(payload: dict) -> np.ndarray:
    if payload.get("image_encoding") == IMAGE_ENCODING:
        shape = tuple(int(value) for value in payload["image_shape"])
        raw = base64.b64decode(payload["image_data"], validate=True)
        expected = int(np.prod(shape, dtype=np.int64))
        if len(raw) != expected:
            raise ValueError(
                f"SAM3 image byte count mismatch: got {len(raw)}, expected {expected}"
            )
        return np.frombuffer(raw, dtype=np.uint8).reshape(shape)
    if "image" in payload:
        return np.asarray(payload["image"], dtype=np.uint8)
    raise ValueError("SAM3 request contains no supported image encoding")


def encode_masks(masks: np.ndarray) -> dict:
    array = np.ascontiguousarray(masks, dtype=np.bool_)
    packed = np.packbits(array.reshape(-1), bitorder="little")
    return {
        "mask_encoding": MASK_ENCODING,
        "mask_shape": list(array.shape),
        "mask_data": base64.b64encode(packed.tobytes()).decode("ascii"),
    }


def decode_masks(payload: dict) -> np.ndarray:
    if payload.get("mask_encoding") == MASK_ENCODING:
        shape = tuple(int(value) for value in payload["mask_shape"])
        count = int(np.prod(shape, dtype=np.int64))
        raw = base64.b64decode(payload["mask_data"], validate=True)
        unpacked = np.unpackbits(
            np.frombuffer(raw, dtype=np.uint8), count=count, bitorder="little"
        )
        return unpacked.astype(np.bool_, copy=False).reshape(shape)
    masks = payload.get("masks")
    if not isinstance(masks, list):
        raise ValueError("SAM3 response contains no supported mask encoding")
    return np.asarray(masks, dtype=np.bool_)

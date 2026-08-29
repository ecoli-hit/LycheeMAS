"""Framework-neutral helpers for lossless multimodal task inputs."""

from __future__ import annotations

import base64
import binascii
import re
from io import BytesIO
from typing import Any

_DATA_IMAGE_URI = re.compile(
    r"^data:(image/[A-Za-z0-9.+-]+);base64,([A-Za-z0-9+/=\r\n]+)$"
)


def pil_image_from_data_uri(value: Any):
    """Decode any valid base64 image data URI into a Pillow image.

    AutoGen's ``Image.from_uri`` currently accepts PNG and JPEG prefixes only,
    while benchmark datasets can legally contain GIF or WebP data URIs. Decode
    through Pillow so every framework receives the same pixels.
    """

    from PIL import Image as PILImage

    uri = str(value or "")
    match = _DATA_IMAGE_URI.fullmatch(uri)
    if match is None:
        raise ValueError("image content must be a base64-encoded image data URI")
    try:
        payload = base64.b64decode(match.group(2), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image data URI contains invalid base64 data") from exc
    try:
        image = PILImage.open(BytesIO(payload))
        image.load()
    except Exception as exc:
        raise ValueError("image data URI does not contain a supported image") from exc
    return image


__all__ = ["pil_image_from_data_uri"]

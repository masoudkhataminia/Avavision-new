"""Reading the pack's header-card code (QR or 1D barcode such as Code 128) with the station camera."""

from __future__ import annotations

import cv2
import numpy as np
import zxingcpp


def read_codes(image: np.ndarray, max_side: int = 2400) -> list[str]:
    """Texts of every QR code or barcode in the frame, sorted; empty when none is readable."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    scale = min(1.0, max_side / max(gray.shape[:2]))
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return sorted({r.text for r in zxingcpp.read_barcodes(gray) if r.valid and r.text})


def code_image(text: str, scale: int = 4) -> np.ndarray:
    """A printable QR code (used for the demo pack and for printing header-card labels)."""
    barcode = zxingcpp.create_barcode(text, zxingcpp.BarcodeFormat.QRCode)
    return np.array(zxingcpp.write_barcode_to_image(barcode, scale=scale))

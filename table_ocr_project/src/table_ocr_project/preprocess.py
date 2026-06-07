from __future__ import annotations

import cv2
import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def upscale(image: np.ndarray, scale: float = 2.0) -> np.ndarray:
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def enhance_document(image: np.ndarray, scale: float = 2.0, clahe_clip: float = 2.0) -> np.ndarray:
    gray = to_gray(image)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = upscale(gray, scale=scale)
    return gray


def preprocess_region_for_ocr(region_img: np.ndarray, scale: float = 2.2) -> np.ndarray:
    gray = enhance_document(region_img, scale=scale, clahe_clip=2.0)
    return gray

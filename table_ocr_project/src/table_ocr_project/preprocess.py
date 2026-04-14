from __future__ import annotations

from typing import Tuple

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


def binarize(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )


def remove_lines_in_small_region(image: np.ndarray) -> np.ndarray:
    gray = to_gray(image)
    inv = 255 - gray
    h, w = inv.shape

    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, w // 3), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, h // 2)))

    horiz = cv2.morphologyEx(inv, cv2.MORPH_OPEN, hk)
    vert = cv2.morphologyEx(inv, cv2.MORPH_OPEN, vk)
    mask = cv2.bitwise_or(horiz, vert)
    cleaned = cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)
    return cleaned


def preprocess_cell_for_ocr(cell_img: np.ndarray, remove_lines: bool = True) -> np.ndarray:
    gray = enhance_document(cell_img, scale=2.5, clahe_clip=2.2)
    if remove_lines:
        gray = remove_lines_in_small_region(gray)
    bw = binarize(gray)
    return bw


def preprocess_region_for_ocr(region_img: np.ndarray, scale: float = 2.2) -> np.ndarray:
    gray = enhance_document(region_img, scale=scale, clahe_clip=2.0)
    return gray

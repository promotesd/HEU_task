from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]


@dataclass
class LayoutResult:
    image_size: Tuple[int, int]
    outer_table: BBox
    title: BBox
    main_table: BBox
    remark: BBox
    bottom: BBox

    def to_dict(self) -> Dict[str, List[int]]:
        return {
            'outer_table': list(self.outer_table),
            'title': list(self.title),
            'main_table': list(self.main_table),
            'remark': list(self.remark),
            'bottom': list(self.bottom),
        }


def _clip_box(box: BBox, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(x1 + 1, min(width, int(x2)))
    y2 = max(y1 + 1, min(height, int(y2)))
    return x1, y1, x2, y2


def detect_outer_table_bbox(image: np.ndarray, min_area_ratio: float = 0.08) -> BBox:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    h, w = gray.shape
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 15)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = h * w * min_area_ratio
    best = None
    best_area = 0
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        area = ww * hh
        if area < min_area:
            continue
        aspect = ww / max(hh, 1)
        if area > best_area and 1.5 < aspect < 5.0:
            best = (x, y, x + ww, y + hh)
            best_area = area
    if best is None:
        return 0, 0, w, h
    return _clip_box(best, w, h)


def _extract_vertical_lines(roi_gray: np.ndarray) -> np.ndarray:
    bw = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 15)
    kernel_h = max(30, roi_gray.shape[0] // 8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))
    vertical = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    return vertical


def detect_remark_split_x(image: np.ndarray, outer_table: BBox) -> int:
    x1, y1, x2, y2 = outer_table
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
    vertical = _extract_vertical_lines(gray)
    proj = vertical.sum(axis=0)
    if proj.max() <= 0:
        return x1 + int(0.72 * (x2 - x1))
    left = int(0.60 * len(proj))
    right = int(0.90 * len(proj))
    band = proj[left:right]
    peak = int(np.argmax(band)) + left
    return x1 + peak


def build_layout_from_template(image: np.ndarray) -> LayoutResult:
    h, w = image.shape[:2]
    outer = detect_outer_table_bbox(image)
    ox1, oy1, ox2, oy2 = outer
    split_x = detect_remark_split_x(image, outer)

    title_h = max(120, int(0.17 * (oy1 + 40)))
    title = _clip_box((ox1, max(0, oy1 - title_h), ox2, oy1), w, h)
    main_table = _clip_box((ox1, oy1, split_x, oy2), w, h)
    remark = _clip_box((split_x, oy1, ox2, oy2), w, h)

    outer_w = ox2 - ox1
    bottom = _clip_box(
        (ox1 + int(0.30 * outer_w), oy2 + 10, ox1 + int(0.88 * outer_w), min(h, oy2 + 220)),
        w,
        h,
    )

    return LayoutResult(
        image_size=(w, h),
        outer_table=outer,
        title=title,
        main_table=main_table,
        remark=remark,
        bottom=bottom,
    )


def crop_by_box(image: np.ndarray, box: BBox) -> np.ndarray:
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2].copy()


def draw_layout_debug(image: np.ndarray, layout: LayoutResult) -> np.ndarray:
    vis = image.copy()
    colors = {
        'outer_table': (0, 255, 0),
        'title': (255, 0, 0),
        'main_table': (0, 0, 255),
        'remark': (255, 255, 0),
        'bottom': (255, 0, 255),
    }
    for name, box in layout.to_dict().items():
        x1, y1, x2, y2 = box
        cv2.rectangle(vis, (x1, y1), (x2, y2), colors[name], 2)
        cv2.putText(vis, name, (x1 + 5, max(25, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[name], 2)
    return vis

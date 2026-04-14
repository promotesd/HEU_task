from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import cv2
import numpy as np


@dataclass
class GridResult:
    x_lines: List[int]
    y_lines: List[int]

    def to_dict(self) -> Dict[str, List[int]]:
        return {
            'x_lines': [int(v) for v in self.x_lines],
            'y_lines': [int(v) for v in self.y_lines],
        }


def _cluster_positions(vals: Sequence[int], tol: int = 3) -> List[int]:
    vals = sorted(int(v) for v in vals)
    if not vals:
        return []
    groups: List[List[int]] = [[vals[0]]]
    for v in vals[1:]:
        if abs(v - groups[-1][-1]) <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [int(round(sum(g) / len(g))) for g in groups]


def _line_positions_from_projection(line_map: np.ndarray, axis: int, proj_ratio: float) -> List[int]:
    proj = line_map.sum(axis=axis).astype(np.float32)
    if proj.max() <= 0:
        return []
    thr = proj.max() * proj_ratio
    idx = np.where(proj >= thr)[0]
    return _cluster_positions(idx.tolist(), tol=3)


def _merge_close_lines(lines: Sequence[int], min_gap: int = 5) -> List[int]:
    lines = sorted(int(v) for v in lines)
    if not lines:
        return []
    merged = [lines[0]]
    for v in lines[1:]:
        if v - merged[-1] < min_gap:
            merged[-1] = int(round((merged[-1] + v) / 2))
        else:
            merged.append(v)
    return merged


def extract_grid_lines(
    main_table_img: np.ndarray,
    vertical_kernel_scale: float = 0.15,
    horizontal_kernel_scale: float = 0.08,
    projection_ratio: float = 0.28,
) -> GridResult:
    gray = cv2.cvtColor(main_table_img, cv2.COLOR_BGR2GRAY) if main_table_img.ndim == 3 else main_table_img.copy()
    h, w = gray.shape

    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 15)
    v_kernel_h = max(20, int(h * vertical_kernel_scale))
    h_kernel_w = max(20, int(w * horizontal_kernel_scale))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_h))
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_w, 1))
    vertical = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vert_kernel)
    horizontal = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hori_kernel)

    x_lines = _line_positions_from_projection(vertical, axis=0, proj_ratio=projection_ratio)
    y_lines = _line_positions_from_projection(horizontal, axis=1, proj_ratio=projection_ratio)

    if not x_lines or x_lines[0] > 3:
        x_lines = [0] + x_lines
    if x_lines[-1] < w - 3:
        x_lines = x_lines + [w - 1]
    if not y_lines or y_lines[0] > 3:
        y_lines = [0] + y_lines
    if y_lines[-1] < h - 3:
        y_lines = y_lines + [h - 1]

    x_lines = _merge_close_lines(x_lines, min_gap=5)
    y_lines = _merge_close_lines(y_lines, min_gap=5)
    return GridResult(x_lines=x_lines, y_lines=y_lines)


def build_cells_from_grid(
    x_lines: Sequence[int],
    y_lines: Sequence[int],
    min_cell_w: int = 6,
    min_cell_h: int = 6,
) -> List[Dict[str, object]]:
    cells: List[Dict[str, object]] = []
    for r in range(len(y_lines) - 1):
        for c in range(len(x_lines) - 1):
            x1, x2 = int(x_lines[c]), int(x_lines[c + 1])
            y1, y2 = int(y_lines[r]), int(y_lines[r + 1])
            if x2 - x1 < min_cell_w or y2 - y1 < min_cell_h:
                continue
            cells.append({'row': r, 'col': c, 'bbox': [x1, y1, x2, y2]})
    return cells


def draw_grid_debug(main_table_img: np.ndarray, x_lines: Sequence[int], y_lines: Sequence[int]) -> np.ndarray:
    vis = main_table_img.copy()
    for x in x_lines:
        cv2.line(vis, (int(x), 0), (int(x), vis.shape[0] - 1), (0, 0, 255), 1)
    for y in y_lines:
        cv2.line(vis, (0, int(y)), (vis.shape[1] - 1, int(y)), (255, 0, 0), 1)
    return vis

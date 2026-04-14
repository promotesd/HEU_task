from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

from .alignment import align_image_to_template
from .config_utils import dump_json, ensure_dir, load_json
from .grid import build_cells_from_grid, draw_grid_debug, extract_grid_lines
from .layout import build_layout_from_template, crop_by_box, draw_layout_debug
from .narrative import render_report
from .ocr_engine import PaddleOCREngine
from .semantic_extractors import (
    extract_bottom_fields,
    extract_main_table,
    extract_remark_fields,
    extract_title_fields,
)


BBox = Tuple[int, int, int, int]


def read_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    return img


def save_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f'Failed to write image: {path}')


def _clip_box(box: Sequence[int], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return [x1, y1, x2, y2]


def _box_from_relative(parent_box: Sequence[int], rel: Sequence[float], width: int, height: int) -> List[int]:
    px1, py1, px2, py2 = [int(v) for v in parent_box]
    pw = px2 - px1
    ph = py2 - py1
    x1 = px1 + int(round(rel[0] * pw))
    y1 = py1 + int(round(rel[1] * ph))
    x2 = px1 + int(round(rel[2] * pw))
    y2 = py1 + int(round(rel[3] * ph))
    return _clip_box([x1, y1, x2, y2], width, height)


def _build_default_semantic_config(layout: Any, grid: Any, image_shape: Tuple[int, int, int]) -> Dict[str, Any]:
    h, w = image_shape[:2]
    title = layout.title
    remark = layout.remark
    bottom = layout.bottom
    x_lines = grid.x_lines

    semantic = {
        'title_fields': {
            'confidentiality': _box_from_relative(title, [0.00, 0.00, 0.16, 0.68], w, h),
            'approved': _box_from_relative(title, [0.00, 0.45, 0.25, 1.00], w, h),
            'date': _box_from_relative(title, [0.22, 0.00, 0.50, 0.50], w, h),
            'title_text': _box_from_relative(title, [0.20, 0.00, 0.67, 0.60], w, h),
            'astronomical_times': _box_from_relative(title, [0.72, 0.00, 1.00, 0.95], w, h),
        },
        'remark_fields': {
            'title': _box_from_relative(remark, [0.35, 0.00, 0.65, 0.08], w, h),
            'occupancy_time': _box_from_relative(remark, [0.02, 0.27, 0.42, 0.39], w, h),
            'total': _box_from_relative(remark, [0.18, 0.20, 0.52, 0.33], w, h),
            'training_columns': ['序号', '参训机型', '数量', '架次', '核算架次', '时间'],
            'training_rows': [],
        },
        'bottom_fields': {
            'line1': {
                '队长': _box_from_relative(bottom, [0.00, 0.00, 0.37, 0.46], w, h),
                '政治委员': _box_from_relative(bottom, [0.40, 0.00, 0.82, 0.46], w, h),
            },
            'line2': {
                '队长': _box_from_relative(bottom, [0.00, 0.46, 0.37, 0.95], w, h),
                '政治委员': _box_from_relative(bottom, [0.40, 0.46, 0.82, 0.95], w, h),
            },
        },
        'main_table_schema': {
            'left_cols': [0, 1, 2],
            'time_cols': [3, max(3, len(x_lines) - 4)],
            'right_cols': [max(0, len(x_lines) - 3), max(0, len(x_lines) - 2)],
            'hour_row': 0,
            'minute_row': 1,
            'top_label_rows': [2, 3, 4, 5, 6],
            'subheader_row': 7,
            'body_start_row': 8,
        },
    }

    # Build 6 default remark training rows from a manually chosen top table region.
    remark_table = _box_from_relative(remark, [0.10, 0.10, 0.98, 0.25], w, h)
    rx1, ry1, rx2, ry2 = remark_table
    rw = rx2 - rx1
    rh = ry2 - ry1
    col_edges = [0.00, 0.12, 0.34, 0.50, 0.66, 0.86, 1.00]
    header_h = int(0.25 * rh)
    body_y1 = ry1 + header_h
    body_h = max(1, ry2 - body_y1)
    num_rows = 6
    row_h = max(1, body_h // num_rows)
    training_rows = []
    for i in range(num_rows):
        yy1 = body_y1 + i * row_h
        yy2 = body_y1 + (i + 1) * row_h if i < num_rows - 1 else ry2
        row_boxes = []
        for a, b in zip(col_edges[:-1], col_edges[1:]):
            row_boxes.append(_clip_box([rx1 + int(a * rw), yy1, rx1 + int(b * rw), yy2], w, h))
        training_rows.append(row_boxes)
    semantic['remark_fields']['training_rows'] = training_rows
    return semantic


def bootstrap_template_config(
    template_path: str | Path,
    output_config_path: str | Path,
    output_debug_dir: str | Path,
) -> Dict[str, Any]:
    template = read_image(template_path)
    layout = build_layout_from_template(template)
    main_table = crop_by_box(template, layout.main_table)
    grid = extract_grid_lines(main_table)
    semantic = _build_default_semantic_config(layout, grid, template.shape)

    config = {
        'template': {
            'image_path': str(Path(template_path).resolve()),
            'image_size': [template.shape[1], template.shape[0]],
        },
        'alignment': {
            'method': 'orb_homography',
            'max_features': 8000,
            'keep_top_k_matches': 400,
            'ransac_thresh': 5.0,
        },
        'regions': layout.to_dict(),
        'grid': grid.to_dict(),
        'semantic': semantic,
    }
    dump_json(config, output_config_path)

    out_dir = ensure_dir(output_debug_dir)
    save_image(out_dir / 'template_layout_debug.png', draw_layout_debug(template, layout))
    save_image(out_dir / 'template_main_table_grid_debug.png', draw_grid_debug(main_table, grid.x_lines, grid.y_lines))
    return config


def process_image_with_fixed_template(
    image_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> Dict[str, Any]:
    config = load_json(config_path)
    output_dir = ensure_dir(output_dir)
    template = read_image(config['template']['image_path'])
    image = read_image(image_path)

    align_cfg = config.get('alignment', {})
    align_res = align_image_to_template(
        image,
        template,
        max_features=int(align_cfg.get('max_features', 8000)),
        keep_top_k_matches=int(align_cfg.get('keep_top_k_matches', 400)),
        ransac_thresh=float(align_cfg.get('ransac_thresh', 5.0)),
    )
    aligned = align_res.aligned
    save_image(output_dir / 'aligned.png', aligned)

    regions = config['regions']
    crops: Dict[str, np.ndarray] = {}
    for name in ['title', 'main_table', 'remark', 'bottom']:
        crop = crop_by_box(aligned, tuple(regions[name]))
        crops[name] = crop
        save_image(output_dir / f'{name}.png', crop)

    x_lines = config['grid']['x_lines']
    y_lines = config['grid']['y_lines']
    save_image(output_dir / 'main_table_grid_debug.png', draw_grid_debug(crops['main_table'], x_lines, y_lines))

    cells = build_cells_from_grid(x_lines, y_lines)
    cells_dir = ensure_dir(output_dir / 'cells')
    cell_meta: List[Dict[str, Any]] = []
    for cell in cells:
        row = int(cell['row'])
        col = int(cell['col'])
        x1, y1, x2, y2 = cell['bbox']
        crop = crops['main_table'][y1:y2, x1:x2]
        filename = f'r{row:03d}_c{col:03d}.png'
        save_image(cells_dir / filename, crop)
        cell_meta.append({
            'row': row,
            'col': col,
            'bbox_in_main_table': [x1, y1, x2, y2],
            'file': str((Path('cells') / filename).as_posix()),
        })

    meta = {
        'input_image': str(Path(image_path).resolve()),
        'template_image': config['template']['image_path'],
        'alignment': {
            'num_matches': align_res.num_matches,
            'inliers': align_res.inliers,
            'homography': align_res.homography.tolist(),
        },
        'regions': regions,
        'grid': {
            'x_lines': x_lines,
            'y_lines': y_lines,
            'num_rows': max(0, len(y_lines) - 1),
            'num_cols': max(0, len(x_lines) - 1),
        },
        'cells': cell_meta,
    }
    dump_json(meta, output_dir / 'metadata.json')
    return meta


def _load_lexicon(lexicon_path: str | Path | None) -> Dict[str, List[str]]:
    if not lexicon_path:
        return {}
    return load_json(lexicon_path)


def run_full_pipeline(
    image_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    lexicon_path: str | Path | None = None,
    lang: str = 'ch',
) -> Dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    meta = process_image_with_fixed_template(image_path=image_path, config_path=config_path, output_dir=output_dir)
    config = load_json(config_path)
    aligned = read_image(output_dir / 'aligned.png')
    lexicon = _load_lexicon(lexicon_path)
    engine = PaddleOCREngine(lang=lang)

    title = extract_title_fields(aligned, config, engine, lexicon)
    remark = extract_remark_fields(aligned, config, engine, lexicon)
    bottom = extract_bottom_fields(aligned, config, engine, lexicon)
    main_table = extract_main_table(aligned, config, engine, Path(output_dir), lexicon)

    result = {
        'input_image': str(Path(image_path).resolve()),
        'title': title,
        'remark': remark,
        'bottom': bottom,
        'main_table': main_table,
        'metadata': meta,
    }
    result['text_report'] = render_report(result)

    dump_json(result, Path(output_dir) / 'ocr_result.json')
    with open(Path(output_dir) / 'report.txt', 'w', encoding='utf-8') as f:
        f.write(result['text_report'])
    return result

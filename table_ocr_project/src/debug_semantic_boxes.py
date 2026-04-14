from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from table_ocr_project.alignment import align_image_to_template
from table_ocr_project.config_utils import load_json
from table_ocr_project.layout import crop_by_box
from table_ocr_project.ocr_engine import PaddleOCREngine
from table_ocr_project.pipeline import read_image


def save_img(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"Failed to save image: {path}")


def crop(img, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return img[0:0, 0:0].copy()
    return img[y1:y2, x1:x2].copy()


def dump_one(engine, aligned, name, box, out_dir):
    img = crop(aligned, box)
    save_img(out_dir / f"{name}.png", img)
    try:
        text_raw = engine.ocr_region_text(img, preprocess=False)
    except Exception as e:
        text_raw = f"[OCR ERROR] {repr(e)}"
    try:
        text_pre = engine.ocr_region_text(img, preprocess=True)
    except Exception as e:
        text_pre = f"[OCR ERROR] {repr(e)}"
    return {
        "name": name,
        "box": list(box),
        "text_preprocess_false": text_raw,
        "text_preprocess_true": text_pre,
    }


def is_box_like(v):
    return isinstance(v, list) and len(v) == 4 and all(isinstance(x, (int, float)) for x in v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_json(args.config)

    image = read_image(args.input)
    template = read_image(cfg["template"]["image_path"])

    align_cfg = cfg.get("alignment", {})
    align_res = align_image_to_template(
        image,
        template,
        max_features=int(align_cfg.get("max_features", 8000)),
        keep_top_k_matches=int(align_cfg.get("keep_top_k_matches", 400)),
        ransac_thresh=float(align_cfg.get("ransac_thresh", 5.0)),
    )
    aligned = align_res.aligned
    save_img(out_dir / "aligned.png", aligned)

    engine = PaddleOCREngine(lang="ch")

    debug = {
        "alignment": {
            "num_matches": int(getattr(align_res, "num_matches", 0)),
            "inliers": int(getattr(align_res, "inliers", 0)),
        },
        "regions": {},
        "semantic": {},
    }

    # 大区域
    for k, box in cfg["regions"].items():
        region_img = crop_by_box(aligned, tuple(box))
        save_img(out_dir / f"region_{k}.png", region_img)
        try:
            region_text_raw = engine.ocr_region_text(region_img, preprocess=False)
        except Exception as e:
            region_text_raw = f"[OCR ERROR] {repr(e)}"
        try:
            region_text_pre = engine.ocr_region_text(region_img, preprocess=True)
        except Exception as e:
            region_text_pre = f"[OCR ERROR] {repr(e)}"

        debug["regions"][k] = {
            "box": list(box),
            "text_preprocess_false": region_text_raw,
            "text_preprocess_true": region_text_pre,
        }

    sem = cfg.get("semantic", {})

    # title_fields
    for k, box in sem.get("title_fields", {}).items():
        if is_box_like(box):
            debug["semantic"][f"title.{k}"] = dump_one(engine, aligned, f"title_{k}", box, out_dir)

    # remark_fields
    remark_fields = sem.get("remark_fields", {})
    for k, v in remark_fields.items():
        if k == "training_rows":
            for i, row in enumerate(v):
                for j, box in enumerate(row):
                    name = f"remark_training_r{i}_c{j}"
                    debug["semantic"][name] = dump_one(engine, aligned, name, box, out_dir)
        elif is_box_like(v):
            debug["semantic"][f"remark.{k}"] = dump_one(engine, aligned, f"remark_{k}", v, out_dir)

    # bottom_fields
    for line_name, line_cfg in sem.get("bottom_fields", {}).items():
        for role, box in line_cfg.items():
            if is_box_like(box):
                name = f"bottom_{line_name}_{role}"
                debug["semantic"][name] = dump_one(engine, aligned, name, box, out_dir)

    with open(out_dir / "debug_semantic_boxes.json", "w", encoding="utf-8") as f:
        json.dump(debug, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
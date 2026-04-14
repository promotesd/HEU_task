from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import cv2
import numpy as np

from .preprocess import preprocess_cell_for_ocr, preprocess_region_for_ocr
from .text_utils import normalize_text


@dataclass
class OCRLine:
    text: str
    score: float
    box: List[List[float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "score": float(self.score),
            "box": self.box,
        }


class PaddleOCREngine:
    """
    PaddleOCR 3.x wrapper
    关键修复：
    1. 兼容 result.json / result.res / dict
    2. 自动展开 payload["res"]
    3. 自动兼容 rec_polys / rec_boxes
    4. 不使用 pytesseract
    """

    def __init__(self, lang: str = "ch") -> None:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

        from paddleocr import PaddleOCR

        self.backend = "paddleocr"
        self.lang = lang
        self.ocr = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    @staticmethod
    def _ensure_bgr_uint8(image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError("Input image is None")

        img = image

        if img.dtype == np.bool_:
            img = img.astype(np.uint8) * 255

        if np.issubdtype(img.dtype, np.floating):
            img = np.clip(img, 0, 255).astype(np.uint8)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.ndim == 3 and img.shape[2] == 3:
            pass
        else:
            raise ValueError(f"Unsupported image shape for OCR: {img.shape}")

        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        return img

    def _to_plain_payload(self, first: Any) -> Dict[str, Any]:
        """
        兼容 PaddleOCR 3.x 的多种结果对象，并展开到真正 OCR 字段所在层
        官方文档里的 predict() 结果是 {'res': {... rec_texts ...}}。
        """
        payload: Any = {}

        if hasattr(first, "json"):
            payload = first.json
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    payload = {}
        elif hasattr(first, "res"):
            payload = first.res
        elif isinstance(first, dict):
            payload = first
        else:
            try:
                payload = dict(first)
            except Exception:
                payload = {}

        if not isinstance(payload, dict):
            return {}

        # 关键修复：真正字段在 payload["res"] 里
        if "res" in payload and isinstance(payload["res"], dict):
            payload = payload["res"]

        # 有些服务端/包装可能叫 prunedResult
        if "prunedResult" in payload and isinstance(payload["prunedResult"], dict):
            payload = payload["prunedResult"]

        return payload

    def _predict_one(self, image: np.ndarray) -> Dict[str, Any]:
        image = self._ensure_bgr_uint8(image)
        result = self.ocr.predict(image)

        if result is None:
            return {}

        if not isinstance(result, (list, tuple)):
            result = list(result)

        if len(result) == 0:
            return {}

        first = result[0]
        payload = self._to_plain_payload(first)
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _safe_list(x: Any) -> List[Any]:
        if x is None:
            return []
        if isinstance(x, list):
            return x
        if isinstance(x, tuple):
            return list(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        return [x]

    @staticmethod
    def _to_box4(box: Any) -> List[List[float]]:
        if box is None:
            return [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

        arr = np.array(box)

        # polygon -> 4x2
        if arr.ndim == 2 and arr.shape[0] == 4 and arr.shape[1] == 2:
            return [[float(x), float(y)] for x, y in arr.tolist()]

        # rect -> [x1,y1,x2,y2]
        if arr.ndim == 1 and arr.shape[0] == 4:
            x1, y1, x2, y2 = [float(v) for v in arr.tolist()]
            return [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ]

        flat = arr.flatten().tolist()
        if len(flat) >= 4:
            x1, y1, x2, y2 = [float(v) for v in flat[:4]]
            return [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ]

        return [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

    def ocr_region(self, image: np.ndarray, preprocess: bool = True) -> List[OCRLine]:
        img = preprocess_region_for_ocr(image) if preprocess else image
        img = self._ensure_bgr_uint8(img)

        payload = self._predict_one(img)

        rec_texts = self._safe_list(payload.get("rec_texts", []))
        rec_scores = self._safe_list(payload.get("rec_scores", []))

        # 优先 polygon，其次 rect
        raw_boxes = payload.get("rec_polys", None)
        if raw_boxes is None or len(self._safe_list(raw_boxes)) == 0:
            raw_boxes = payload.get("rec_boxes", [])

        raw_boxes = self._safe_list(raw_boxes)

        n = max(len(rec_texts), len(rec_scores), len(raw_boxes))
        lines: List[OCRLine] = []

        for i in range(n):
            text = ""
            score = 0.0
            box = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

            if i < len(rec_texts):
                text = normalize_text(str(rec_texts[i]))
            if i < len(rec_scores):
                try:
                    score = float(rec_scores[i])
                except Exception:
                    score = 0.0
            if i < len(raw_boxes):
                box = self._to_box4(raw_boxes[i])

            if text:
                lines.append(OCRLine(text=text, score=score, box=box))

        return lines

    def ocr_region_text(self, image: np.ndarray, preprocess: bool = True) -> str:
        lines = self.ocr_region(image, preprocess=preprocess)
        if not lines:
            return ""
        lines = sorted(
            lines,
            key=lambda line: (
                sum(p[1] for p in line.box) / 4.0,
                sum(p[0] for p in line.box) / 4.0,
            ),
        )
        return " ".join([line.text for line in lines]).strip()

    def ocr_cell(self, image: np.ndarray, remove_lines: bool = True) -> Dict[str, Any]:
        img = preprocess_cell_for_ocr(image, remove_lines=remove_lines)
        img = self._ensure_bgr_uint8(img)

        payload = self._predict_one(img)

        rec_texts = self._safe_list(payload.get("rec_texts", []))
        rec_scores = self._safe_list(payload.get("rec_scores", []))

        texts = [normalize_text(str(t)) for t in rec_texts if normalize_text(str(t))]
        text = " ".join(texts).strip()

        vals = []
        for s in rec_scores:
            try:
                vals.append(float(s))
            except Exception:
                pass
        score = float(sum(vals) / len(vals)) if vals else 0.0

        return {
            "text": text,
            "score": score,
        }
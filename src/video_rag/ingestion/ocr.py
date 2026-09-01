from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from video_rag.schemas import Keyframe, OCRText


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    payload = getattr(value, "json", None)
    if callable(payload):
        payload = payload()
    return payload if isinstance(payload, dict) else {}


def _parse_prediction_result(result: Any) -> list[tuple[str, float, Any]]:
    """Normalize PaddleOCR 2.x and 3.x result shapes."""
    parsed: list[tuple[str, float, Any]] = []
    for item in result or ():
        payload = _as_dict(item)
        values = payload.get("res", payload)
        texts = values.get("rec_texts") if isinstance(values, dict) else None
        scores = values.get("rec_scores") if isinstance(values, dict) else None
        polygons = values.get("dt_polys") if isinstance(values, dict) else None
        if texts is not None and scores is not None:
            polygons = polygons or [()] * len(texts)
            parsed.extend(zip(texts, scores, polygons, strict=False))
            continue
        rows = item if isinstance(item, list) else ()
        if rows and len(rows) == 2 and isinstance(rows[1], tuple):
            rows = [rows]
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            recognition = row[1]
            if not isinstance(recognition, (list, tuple)) or len(recognition) < 2:
                continue
            parsed.append((str(recognition[0]), float(recognition[1]), row[0]))
    return parsed


class PaddleOCRExtractor:
    """Timestamped OCR over selected keyframes with a lazy PaddleOCR backend."""

    def __init__(
        self,
        *,
        language: str = "ch",
        minimum_confidence: float = 0.55,
        engine_factory: Any = None,
    ) -> None:
        if not 0 <= minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be between 0 and 1")
        self.language = language
        self.minimum_confidence = minimum_confidence
        self._engine_factory = engine_factory
        self._engine: Any = None

    def _load(self) -> Any:
        if self._engine is None:
            if self._engine_factory is None:
                try:
                    from paddleocr import PaddleOCR
                except ImportError as exc:
                    raise RuntimeError(
                        "OCR extraction requires: pip install -e '.[ocr]'"
                    ) from exc
                self._engine_factory = PaddleOCR
            try:
                self._engine = self._engine_factory(
                    lang=self.language,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
            except TypeError:
                self._engine = self._engine_factory(lang=self.language, use_angle_cls=True)
        return self._engine

    def extract(self, keyframes: Iterable[Keyframe]) -> list[OCRText]:
        engine = self._load()
        extracted: list[OCRText] = []
        for frame in keyframes:
            if not Path(frame.path).is_file():
                continue
            raw = (
                engine.predict(frame.path)
                if hasattr(engine, "predict")
                else engine.ocr(frame.path, cls=True)
            )
            for text, confidence, polygon in _parse_prediction_result(raw):
                normalized = str(text).strip()
                score = float(confidence)
                if not normalized or score < self.minimum_confidence:
                    continue
                bbox = tuple(
                    (float(point[0]), float(point[1]))
                    for point in (polygon or ())
                    if isinstance(point, (list, tuple)) and len(point) >= 2
                )
                extracted.append(
                    OCRText(
                        timestamp=frame.timestamp,
                        text=normalized,
                        confidence=score,
                        bbox=bbox,
                    )
                )
        return deduplicate_ocr(extracted)


def deduplicate_ocr(
    items: Iterable[OCRText], *, temporal_window: float = 3.0
) -> list[OCRText]:
    """Suppress repeated overlays while preserving their strongest observation."""
    selected: list[OCRText] = []
    for item in sorted(items, key=lambda value: (value.timestamp, value.text)):
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(selected)
                if existing.text.casefold() == item.text.casefold()
                and abs(existing.timestamp - item.timestamp) <= temporal_window
            ),
            None,
        )
        if duplicate_index is None:
            selected.append(item)
        elif item.confidence > selected[duplicate_index].confidence:
            selected[duplicate_index] = item
    return sorted(selected, key=lambda value: (value.timestamp, value.text))

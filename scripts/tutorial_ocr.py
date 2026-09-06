"""Run in the isolated .runtime-ocr target, leaving main model dependencies intact."""

import json
from dataclasses import asdict, replace
from pathlib import Path

from video_rag.ingestion.ocr import PaddleOCRExtractor
from video_rag.storage import load_segments, save_segments

segments = load_segments("artifacts/tutorialvqa/segments.whisper.jsonl")
frames = {}
for s in segments:
    for f in s.keyframes:
        # Sample clear screen frames at regular 30-second intervals, independent of QA.
        if f.timestamp >= 10 and (f.timestamp - 10) % 30 == 0:
            frames[(s.video_id, f.timestamp)] = f
engine = PaddleOCRExtractor(language="en")
by_video = {}
log = []
for (vid, t), frame in frames.items():
    items = engine.extract([frame])
    by_video.setdefault(vid, []).extend(items)
    log.append(
        {"video_id": vid, "timestamp": t, "path": frame.path, "items": [asdict(i) for i in items]}
    )
    print("OCR", vid, t, len(items), flush=True)
result = []
for s in segments:
    items = tuple(
        i for i in by_video.get(s.video_id, []) if s.start_time <= i.timestamp < s.end_time
    )
    result.append(
        replace(s, ocr_items=items, ocr_text=" ".join(dict.fromkeys(i.text for i in items)))
    )
assert any(s.ocr_text for s in result), "OCR returned no recognized text"
save_segments("artifacts/tutorialvqa/segments.ocr.jsonl", result)
Path("artifacts/tutorialvqa/ocr-run.json").write_text(json.dumps(log, ensure_ascii=False, indent=2))

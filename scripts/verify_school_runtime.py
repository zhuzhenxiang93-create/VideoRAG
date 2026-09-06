"""Real-model regression on existing news; does NOT count as TutorialVQA acceptance."""

import json
import logging
from dataclasses import asdict
from pathlib import Path

from run_server import build_real_pipeline

from video_rag.api import create_app
from video_rag.ingestion.media import WhisperTranscriber
from video_rag.ingestion.ocr import PaddleOCRExtractor
from video_rag.storage import load_segments

out = Path("artifacts/tutorialvqa/runtime-regression")
out.mkdir(parents=True, exist_ok=True)
segments = load_segments("artifacts/segments.ocr.jsonl")
chosen = next(s for s in segments if s.video_id.startswith("012_"))
report = {"dataset": "existing news, NOT TutorialVQA", "video_id": chosen.video_id}


def save():
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


try:
    asr = WhisperTranscriber()
    rows = asr.transcribe(chosen.source_path)
    assert rows, "Whisper returned no timed text"
    (out / "whisper.json").write_text(
        json.dumps([asdict(r) for r in rows], ensure_ascii=False, indent=2)
    )
    report["whisper"] = {"status": "passed", "segments": len(rows), "input": chosen.source_path}
    asr.unload()
except Exception as e:
    logging.getLogger(__name__).exception("Ingestion verification failed")
    report["whisper"] = {"status": "failed", "error": repr(e)}
save()
try:
    # A known legible conference title frame, using the existing Chinese OCR model.
    frame = chosen.keyframes[0]
    rows = PaddleOCRExtractor(language="ch").extract([frame])
    assert rows, "OCR returned no text"
    (out / "ocr.json").write_text(
        json.dumps([asdict(r) for r in rows], ensure_ascii=False, indent=2)
    )
    report["ocr"] = {"status": "passed", "items": len(rows), "frame": frame.path}
except Exception as e:
    logging.getLogger(__name__).exception("Ingestion verification failed")
    report["ocr"] = {"status": "failed", "error": repr(e)}
save()
p = build_real_pipeline(
    segments_path=Path("artifacts/segments.ocr.jsonl"),
    index_dir=Path("artifacts/indexes-ocr"),
    config_path=Path("config.toml"),
)
p.warmup()
report["retrieval_warmup"] = "passed"
client = create_app(p).test_client()
report["questions"] = []
questions = [
    "发言者参加的会议叫什么名称？",
    "What does the speaker say about innovation in tourism?",
    "讲台上显示了哪些英文字母？",
    "How do I configure a Kubernetes cluster on Mars using this video?",
    "发言者认为旅游行业为什么需要创新？",
]
for i, q in enumerate(questions):
    response = client.post("/api/ask", json={"question": q, "video_ids": [chosen.video_id]})
    data = response.get_json()
    (out / f"answer-{i + 1}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))
    evidence = data.get("evidence", [])
    checks = []
    for e in evidence:
        assert e["video_id"] == chosen.video_id
        r = client.get(e["video_url"], headers={"Range": "bytes=0-1023"})
        checks.append(r.status_code == 206)
    row = {
        "question": q,
        "http_status": response.status_code,
        "answer": data.get("answer"),
        "abstained": data.get("abstained"),
        "citations": data.get("citations"),
        "video_ranges_readable": checks,
    }
    report["questions"].append(row)
    save()
    print(json.dumps(row, ensure_ascii=False), flush=True)
search = client.post("/api/search", json={"question": "旅游", "video_ids": [chosen.video_id]})
assert search.status_code == 200 and search.json["evidence"]
assert all(e["video_id"] == chosen.video_id for e in search.json["evidence"])
report["scope_search"] = "passed"
report["page_http"] = client.get("/").status_code
report["browser_interaction"] = "not run: no browser installed on server"
save()
print("REGRESSION COMPLETE", flush=True)

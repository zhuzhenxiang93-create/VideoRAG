"""Real API acceptance. Gold labels are used only after generation for reporting."""

import json
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:5000"
OUT = Path("artifacts/tutorialvqa/acceptance")
OUT.mkdir(exist_ok=True)


def request(path, payload=None, headers=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path, data=data, headers=headers or {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        return response.status, response.read()


for attempt in range(120):
    try:
        request("/api/health")
        break
    except OSError:
        time.sleep(1)
else:
    raise RuntimeError("Server did not become ready")
report = {
    "dataset": "TutorialVQA",
    "subtitle_source": "real Whisper, bounded audio",
    "questions": [],
}
rows = [
    json.loads(s)
    for s in Path("artifacts/tutorialvqa/evaluation/dev.jsonl").read_text().splitlines()
]
for qid in ["dev:1168", "dev:501", "dev:1131"]:
    gold = next(q for q in rows if q["question_id"] == qid)
    status, body = request(
        "/api/ask", {"question": gold["question"], "video_ids": [gold["video_id"]]}
    )
    response = json.loads(body)
    (OUT / f"{qid.replace(':', '-')}.json").write_text(
        json.dumps(response, ensure_ascii=False, indent=2)
    )
    evidence = response.get("evidence", [])
    scoped = bool(evidence) and all(e["video_id"] == gold["video_id"] for e in evidence)
    readable = bool(evidence) and all(
        request(e["video_url"], headers={"Range": "bytes=0-1023"})[0] == 206 for e in evidence
    )
    overlap = any(
        min(e["end_time"], g["end_time"]) > max(e["start_time"], g["start_time"])
        for e in evidence
        for g in gold["reference_intervals"]
    )
    row = {
        "question_id": qid,
        "question": gold["question"],
        "status": status,
        "answer": response.get("answer"),
        "abstained": response.get("abstained"),
        "citations": response.get("citations"),
        "scoped": scoped,
        "media_range_206": readable,
        "overlap_with_reference": overlap,
        "reference_intervals": gold["reference_intervals"],
    }
    report["questions"].append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    (OUT / "api-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
status, body = request(
    "/api/ask",
    {
        "question": "What dosage of antibiotics should I give a sick penguin on Mars?",
        "video_ids": ["14643"],
    },
)
response = json.loads(body)
(OUT / "unanswerable.json").write_text(json.dumps(response, ensure_ascii=False, indent=2))
report["unanswerable"] = response.get("abstained") is True and not response.get("citations")
_, ocr_body = request(
    "/api/search", {"question": "Which button is labelled Save?", "video_ids": ["14643"]}
)
ocr_results = json.loads(ocr_body)["evidence"]
report["ocr_in_search_evidence"] = any(e.get("ocr_items") for e in ocr_results)
report["scope_search"] = {}
for vid in ["14643", "14644", "14645", "14646", "14647"]:
    _, body = request("/api/search", {"question": "layer", "video_ids": [vid]})
    results = json.loads(body)["evidence"]
    report["scope_search"][vid] = bool(results) and all(e["video_id"] == vid for e in results)
report["passed"] = (
    all(
        q["answer"] and not q["abstained"] and q["scoped"] and q["media_range_206"]
        for q in report["questions"]
    )
    and report["unanswerable"]
    and all(report["scope_search"].values())
)
(OUT / "api-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
print("API ACCEPTANCE", report["passed"], flush=True)

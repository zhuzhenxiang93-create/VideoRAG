"""Export evaluation labels only; this file is never read by index preparation."""

import json
from pathlib import Path

root = Path("data/tutorialvqa/original/repository")
out = Path("artifacts/tutorialvqa/evaluation")
out.mkdir(parents=True, exist_ok=True)
selected = {"14643", "14644", "14645", "14646", "14647"}
videos = {v["video_id"]: v for v in json.loads((root / "videos.json").read_text())}
audit = {
    "total": 0,
    "exact_annotation_matches": 0,
    "single_sentence_spans": 0,
    "index_origin": "zero-based",
    "end_boundary": "inclusive",
    "selected": sorted(selected),
    "splits": {},
}
for split in ["train", "dev", "test"]:
    exported = []
    for idx, q in enumerate(json.loads((root / f"{split}.json").read_text())):
        v = videos[q["video_id"]]
        a, b = q["answer_start"], q["answer_end"]
        assert 0 <= a <= b < len(v["transcript"])
        matches = [s for s in v["segments"] if s["sentence_indexes"] == {"start": a, "end": b}]
        assert matches, f"No annotation match: {split}:{idx}"
        audit["total"] += 1
        audit["exact_annotation_matches"] += 1
        audit["single_sentence_spans"] += int(a == b)
        if q["video_id"] in selected:
            exported.append(
                {
                    **q,
                    "question_id": f"{split}:{idx}",
                    "split": split,
                    "reference_answer": " ".join(v["transcript"][a : b + 1]),
                    "reference_intervals": [
                        {
                            "start_time": float(s["timestamps"]["start"]),
                            "end_time": float(s["timestamps"]["end"]),
                        }
                        for s in matches
                    ],
                }
            )
    (out / f"{split}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in exported)
    )
    audit["splits"][split] = len(exported)
(out / "schema-audit.json").write_text(json.dumps(audit, indent=2))
print(audit)

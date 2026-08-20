from __future__ import annotations

import json
from pathlib import Path


source = Path("data/evaluation/questions.zh.seed.jsonl")
backup = source.with_suffix(".pre_metadata.jsonl")
if not backup.exists():
    backup.write_bytes(source.read_bytes())

records = []
for line in source.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    item = json.loads(line)
    first_segment = item["relevant_segment_ids"][0]
    item["video_id"] = first_segment.rsplit("_", 1)[0]
    item["answerable"] = True
    item["verification_status"] = "generated_candidate"
    item["annotation_source"] = "existing 20-question seed set; pending independent human verification"
    records.append(item)
source.write_text(
    "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
    encoding="utf-8",
)

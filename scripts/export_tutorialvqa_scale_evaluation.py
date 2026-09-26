"""Export official labels for scoring only; output is never used for indexing."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,default=Path("artifacts/tutorialvqa/scale-76")); parser.add_argument("--repository",type=Path,default=Path("data/tutorialvqa/original/repository")); args=parser.parse_args()
    selected=set(json.loads((args.root/"selected-videos.json").read_text())); videos={str(v["video_id"]):v for v in json.loads((args.repository/"videos.json").read_text())}; out=args.root/"evaluation"; out.mkdir(exist_ok=True)
    audit={"selected_video_count":len(selected),"index_origin":"zero-based","end_boundary":"inclusive","splits":{}}
    for split in ("train","dev","test"):
        exported=[]
        for index,q in enumerate(json.loads((args.repository/f"{split}.json").read_text())):
            vid=str(q["video_id"])
            if vid not in selected: continue
            video=videos[vid]; a,b=int(q["answer_start"]),int(q["answer_end"]); matches=[s for s in video["segments"] if s["sentence_indexes"]=={"start":a,"end":b}]
            if not matches: raise ValueError(f"No official interval for {split}:{index}")
            exported.append({**q,"video_id":vid,"question_id":f"{split}:{index}","split":split,"reference_answer":" ".join(video["transcript"][a:b+1]),"reference_intervals":[{"start_time":float(s["timestamps"]["start"]),"end_time":float(s["timestamps"]["end"])} for s in matches]})
        (out/f"{split}.jsonl").write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in exported)); audit["splits"][split]=len(exported)
    (out/"schema-audit.json").write_text(json.dumps(audit,indent=2)); print(json.dumps(audit,indent=2))

if __name__=="__main__": main()

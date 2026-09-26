"""Controlled Chinese-CLIP vs OpenAI CLIP comparison on TutorialVQA."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from video_rag.evaluation import evaluate_retrieval
from video_rag.evaluation.tutorialvqa import latency_summary, relevant_segment_ids
from video_rag.retrieval import ClipVisionRetriever
from video_rag.storage import load_segments


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--segments",type=Path,default=Path("artifacts/tutorialvqa/bounded-v2/segments.ocr.jsonl"))
    parser.add_argument("--zh-index",type=Path,default=Path("artifacts/tutorialvqa/bounded-v2/indexes"))
    parser.add_argument("--en-index",type=Path,default=Path("artifacts/tutorialvqa/bounded-v2/indexes-en-clip"))
    parser.add_argument("--output",type=Path,default=Path("artifacts/tutorialvqa/clip-ablation/report.json"))
    parser.add_argument("--top-k",type=int,default=10)
    args=parser.parse_args()
    segments=load_segments(args.segments)
    by_video={}
    for s in segments: by_video.setdefault(s.video_id,set()).add(s.segment_id)
    retrievers={
      "chinese_clip":ClipVisionRetriever("OFA-Sys/chinese-clip-vit-base-patch16",device="cuda",index_dir=args.zh_index),
      "openai_clip":ClipVisionRetriever("openai/clip-vit-large-patch14",device="cuda",index_dir=args.en_index),
    }
    for r in retrievers.values():
        r.build(segments); r.search("visible interface and tutorial action",1)
    report={"protocol":{"top_k":args.top_k,"scoped_by_video_id":True,"retrieval_route":"vision_only","same_93_segments":True},"splits":{}}
    for split in ("dev","test"):
        questions=read_jsonl(Path(f"artifacts/tutorialvqa/evaluation/{split}.jsonl"))
        truth={q["question_id"]:relevant_segment_ids(segments,q["video_id"],q["reference_intervals"]) for q in questions}
        predictions={name:{} for name in retrievers}; timings={name:[] for name in retrievers}
        for q in questions:
            for name,r in retrievers.items():
                start=perf_counter(); hits=r.search(q["question"],args.top_k,allowed_ids=by_video[q["video_id"]]); timings[name].append((perf_counter()-start)*1000)
                predictions[name][q["question_id"]]=[h.segment_id for h in hits]
        metrics={name:{**evaluate_retrieval(predictions[name],truth),"latency_ms":latency_summary(timings[name])} for name in retrievers}
        zh=predictions["chinese_clip"]; en=predictions["openai_clip"]
        metrics["comparison"]={
          "top1_changed":sum(zh[q][:1]!=en[q][:1] for q in truth),
          "openai_top1_wins":sum(not(set(zh[q][:1])&set(truth[q])) and bool(set(en[q][:1])&set(truth[q])) for q in truth),
          "chinese_top1_wins":sum(not(set(en[q][:1])&set(truth[q])) and bool(set(zh[q][:1])&set(truth[q])) for q in truth),
          "same_top1_relevance":sum(bool(set(en[q][:1])&set(truth[q]))==bool(set(zh[q][:1])&set(truth[q])) for q in truth),
        }
        report["splits"][split]=metrics
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__": main()

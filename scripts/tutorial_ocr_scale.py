"""Resume-safe English OCR for a prepared TutorialVQA scale corpus."""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict,replace
from pathlib import Path
from video_rag.ingestion.ocr import PaddleOCRExtractor
from video_rag.schemas import OCRText
from video_rag.storage import load_segments,save_segments


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,default=Path("artifacts/tutorialvqa/scale-76")); args=parser.parse_args()
    segments=load_segments(args.root/"segments.whisper.jsonl"); cache_path=args.root/"ocr-cache.jsonl"; cached={}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            row=json.loads(line); cached[(row["video_id"],float(row["timestamp"]))]=tuple(OCRText(**item) for item in row["items"])
    frames={}
    for segment in segments:
        for frame in segment.keyframes:
            if frame.timestamp>=10 and (frame.timestamp-10)%30==0: frames[(segment.video_id,frame.timestamp)]=frame
    engine=PaddleOCRExtractor(language="en")
    for position,(key,frame) in enumerate(sorted(frames.items()),1):
        if key not in cached:
            items=tuple(engine.extract([frame])); cached[key]=items
            with cache_path.open("a") as stream: stream.write(json.dumps({"video_id":key[0],"timestamp":key[1],"path":frame.path,"items":[asdict(item) for item in items]},ensure_ascii=False)+"\n")
        print(f"[{position}/{len(frames)}] OCR {key[0]} {key[1]:.1f}s items={len(cached[key])}",flush=True)
    result=[]
    for segment in segments:
        items=tuple(item for (vid,_),values in cached.items() if vid==segment.video_id for item in values if segment.start_time<=item.timestamp<segment.end_time)
        result.append(replace(segment,ocr_items=items,ocr_text=" ".join(dict.fromkeys(item.text for item in items))))
    if not any(segment.ocr_text for segment in result): raise ValueError("OCR returned no text")
    save_segments(args.root/"segments.ocr.jsonl",result)
    print(f"SAVED segments={len(result)} sampled_frames={len(frames)} ocr_items={sum(len(v) for v in cached.values())}",flush=True)

if __name__=="__main__": main()

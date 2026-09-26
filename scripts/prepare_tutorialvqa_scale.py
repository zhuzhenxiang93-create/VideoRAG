"""Prepare a deterministic TutorialVQA video subset without using QA labels."""
from __future__ import annotations
import argparse
import json
import shutil
import zipfile
from dataclasses import asdict
from pathlib import Path
import cv2
from video_rag.ingestion import WhisperTranscriber, materialize_segments, probe_video
from video_rag.schemas import Keyframe, TimedText
from video_rag.storage import save_segments


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--video-count",type=int,default=76)
    parser.add_argument("--output-root",type=Path,default=Path("artifacts/tutorialvqa/scale-76"))
    parser.add_argument("--video-dir",type=Path,default=Path("data/tutorialvqa/videos"))
    parser.add_argument("--archive",type=Path,default=Path("data/tutorialvqa/original/videos.zip"))
    parser.add_argument("--metadata",type=Path,default=Path("data/tutorialvqa/original/repository/videos.json"))
    args=parser.parse_args()
    metadata=json.loads(args.metadata.read_text())
    ids=sorted(str(item["video_id"]) for item in metadata)[:args.video_count]
    if len(ids)<args.video_count: raise ValueError("Requested more videos than metadata contains")
    args.output_root.mkdir(parents=True,exist_ok=True); args.video_dir.mkdir(parents=True,exist_ok=True)
    (args.output_root/"selected-videos.json").write_text(json.dumps(ids,indent=2))
    with zipfile.ZipFile(args.archive) as archive:
        for vid in ids:
            destination=args.video_dir/f"{vid}.webm"; info=archive.getinfo(f"videos/{vid}.webm")
            if not destination.exists():
                with archive.open(info) as source,destination.open("wb") as target: shutil.copyfileobj(source,target)
            if destination.stat().st_size!=info.file_size: raise ValueError(f"Size mismatch for {vid}")
    whisper=WhisperTranscriber(); all_segments=[]; catalog=[]
    for position,vid in enumerate(ids,1):
        path=(args.video_dir/f"{vid}.webm").resolve(); info=probe_video(path)
        cache_dir=args.output_root/"whisper"; cache_dir.mkdir(exist_ok=True)
        cache=cache_dir/f"{vid}.json"; existing=Path(f"artifacts/tutorialvqa/bounded-v2/{vid}.whisper.json")
        if cache.exists(): source=cache
        elif existing.exists(): shutil.copy2(existing,cache); source=cache
        else:
            print(f"[{position}/{len(ids)}] WHISPER {vid}",flush=True)
            transcript=whisper.transcribe_bounded(path,language="en",seconds=20)
            if not transcript: raise ValueError(f"Empty Whisper output for {vid}")
            cache.write_text(json.dumps([asdict(item) for item in transcript],ensure_ascii=False,indent=2)); source=cache
        transcript=[TimedText(**item) for item in json.loads(source.read_text())]
        frame_dir=args.output_root/"frames"/vid; frame_dir.mkdir(parents=True,exist_ok=True)
        capture=cv2.VideoCapture(str(path)); frames=[]; timestamp=0.0
        while timestamp<info.duration:
            frame_path=frame_dir/f"frame-{timestamp:08.3f}.jpg"
            if not frame_path.exists():
                capture.set(cv2.CAP_PROP_POS_MSEC,timestamp*1000); ok,image=capture.read()
                if not ok: timestamp+=5; continue
                if not cv2.imwrite(str(frame_path),image): raise RuntimeError(f"Cannot write {frame_path}")
            frames.append(Keyframe(timestamp,str(frame_path.resolve()),selection_source="uniform_5s",described_by_vlm=False)); timestamp+=5
        capture.release()
        if not frames: raise ValueError(f"No frames for {vid}")
        video_segments=materialize_segments(video_id=vid,source_path=str(path),duration=info.duration,transcript=transcript,keyframes=frames,strategy="fixed",window_seconds=20,overlap_seconds=5)
        all_segments.extend(video_segments)
        catalog.append({"video_id":vid,"duration":info.duration,"frames":len(frames),"asr_chunks":len(transcript),"segments":len(video_segments),"transcript_source":"Whisper small, independent 20-second audio blocks; no official transcript or QA labels used"})
        save_segments(args.output_root/"segments.whisper.jsonl",all_segments)
        (args.output_root/"input-manifest.json").write_text(json.dumps(catalog,indent=2))
        print(f"[{position}/{len(ids)}] PREPARED {vid} duration={info.duration:.1f}s segments={len(video_segments)}",flush=True)
    whisper.unload(); print(f"SAVED videos={len(ids)} segments={len(all_segments)}",flush=True)

if __name__=="__main__": main()

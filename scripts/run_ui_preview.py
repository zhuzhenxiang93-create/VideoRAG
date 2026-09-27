"""Explicit CPU preview using indexed ASR/OCR data, never generated answers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from video_rag.adapters import EvidenceGenerator, TokenOverlapReranker
from video_rag.api import create_app
from video_rag.pipeline import VideoRAGPipeline
from video_rag.retrieval import BM25Retriever
from video_rag.storage import load_segments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=Path,
                        default=Path("artifacts/tutorialvqa/scale-76/segments.ocr.jsonl"))
    parser.add_argument("--video-metadata", type=Path,
                        default=Path("data/tutorialvqa/original/repository/videos.json"))
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()
    pipeline = VideoRAGPipeline(
        retrievers=[BM25Retriever()], reranker=TokenOverlapReranker(),
        generator=EvidenceGenerator(), fusion_top_k=5, dedupe_overlap_ratio=0.2,
    )
    pipeline.build(load_segments(args.segments))
    titles = {}
    if args.video_metadata.is_file():
        titles = {str(v["video_id"]): v["video_title"]
                  for v in json.loads(args.video_metadata.read_text()) if v.get("video_title")}
    print("PREVIEW ONLY: BM25 retrieval and playback; /api/ask returns 503.", flush=True)
    create_app(pipeline, video_titles=titles, preview=True).run(
        host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from video_rag.config import load_config
from video_rag.retrieval import ClipVisionRetriever
from video_rag.retrieval.faiss_dense import normalize_rows, unwrap_model_features
from video_rag.storage import load_segments


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def cuda_memory() -> dict[str, float | None]:
    try:
        import torch
    except ImportError:
        return {"peak_allocated_mib": None, "peak_reserved_mib": None}
    if not torch.cuda.is_available():
        return {"peak_allocated_mib": None, "peak_reserved_mib": None}
    return {
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
    }


def reset_cuda_peak() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def component_benchmark(retriever: ClipVisionRetriever, questions: list[str]) -> dict:
    """Separate CPU preprocessing, GPU text encoding and FAISS search timing."""
    import torch

    model, processor = retriever._load()
    preprocessing, encoding, faiss_search = [], [], []
    for question in questions:
        started = perf_counter()
        inputs = processor(text=[question], return_tensors="pt", padding=True).to(retriever.device)
        preprocessing.append((perf_counter() - started) * 1000)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = perf_counter()
        with torch.inference_mode():
            values = unwrap_model_features(model.get_text_features(**inputs)).float().cpu().numpy()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        encoding.append((perf_counter() - started) * 1000)
        vector = normalize_rows(np.asarray(values, dtype=np.float32))
        started = perf_counter()
        retriever._index.search(vector, min(20, len(retriever._segment_ids)))
        faiss_search.append((perf_counter() - started) * 1000)

    def stats(values: list[float]) -> dict:
        return {
            "mean": mean(values), "p50": median(values), "p95": percentile(values, .95),
            "p99": percentile(values, .99), "max": max(values),
        }

    return {
        "samples": len(questions),
        "cpu_processor_ms": stats(preprocessing),
        "gpu_text_encoder_and_transfer_ms": stats(encoding),
        "faiss_indexflatip_ms": stats(faiss_search),
        "clock": "time.perf_counter; CUDA synchronized around GPU component",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the existing segment-mean Chinese-CLIP index.")
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--segments", type=Path, default=Path("artifacts/segments.jsonl"))
    parser.add_argument("--index-dir", type=Path, default=Path("artifacts/indexes"))
    parser.add_argument("--questions", type=Path, default=Path("data/evaluation/questions.zh.seed.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation/visual_index_baseline.json"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--measure-build", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")

    config = load_config(args.config)
    segments = load_segments(args.segments)
    questions = [
        json.loads(line)["question"]
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not questions:
        raise ValueError("question file is empty")

    retriever = ClipVisionRetriever(config.models.clip, device="cuda", index_dir=args.index_dir)
    retriever.build(segments)
    reset_cuda_peak()
    started = perf_counter()
    retriever.search(questions[0], args.top_k)
    cold_ms = (perf_counter() - started) * 1000
    cold_memory = cuda_memory()

    timings = []
    reset_cuda_peak()
    for _ in range(args.repeats):
        for question in questions:
            started = perf_counter()
            retriever.search(question, args.top_k)
            timings.append((perf_counter() - started) * 1000)
    warm_memory = cuda_memory()
    components = component_benchmark(retriever, questions)

    index_path = args.index_dir / "vision_dense_zh.faiss"
    metadata_path = args.index_dir / "vision_dense_zh.json"
    import faiss

    index = faiss.read_index(str(index_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    physical_paths = {
        str(Path(frame.path).resolve())
        for segment in segments
        for frame in segment.keyframes
    }
    memberships = sum(len(segment.keyframes) for segment in segments)
    report = {
        "status": "measured",
        "model": config.models.clip,
        "device": retriever.device,
        "index": {
            "name": "vision_dense_zh",
            "aggregation": "arithmetic mean of all keyframe embeddings in each segment, followed by L2 normalization",
            "empty_segment_behavior": "segments without keyframes are skipped",
            "entries": int(index.ntotal),
            "metadata_entries": len(metadata["segment_ids"]),
            "dimension": int(index.d),
            "faiss_bytes": index_path.stat().st_size,
            "metadata_bytes": metadata_path.stat().st_size,
            "unique_physical_frames": len(physical_paths),
            "frame_memberships_encoded_by_current_builder": memberships,
            "duplicate_encoding_requests_due_to_overlap": memberships - len(physical_paths),
        },
        "query_benchmark": {
            "question_source": str(args.questions),
            "question_status": "generated_candidate; latency workload only, not accuracy evidence",
            "question_count": len(questions),
            "repeats": args.repeats,
            "warmup_queries_before_measurement": 1,
            "samples": len(timings),
            "top_k": args.top_k,
            "cold_first_query_ms_including_model_load": cold_ms,
            "warm_ms": {
                "mean": mean(timings),
                "median": median(timings),
                "p95": percentile(timings, 0.95),
                "p99": percentile(timings, 0.99),
                "max": max(timings),
            },
            "component_breakdown_single_pass": components,
            "clock": "time.perf_counter",
            "cold_peak_gpu": cold_memory,
            "warm_peak_gpu": warm_memory,
        },
        "build_benchmark": {"status": "not_requested"},
    }

    if args.measure_build:
        with TemporaryDirectory(prefix="visual-index-baseline-", dir=str(args.output.parent)) as temporary:
            builder = ClipVisionRetriever(
                config.models.clip, device="cuda", index_dir=temporary, force_rebuild=True
            )
            # Reuse the already loaded model and processor so this measures encoding/indexing,
            # while cold model startup remains separately reported above.
            builder._model = retriever._model
            builder._processor = retriever._processor
            reset_cuda_peak()
            started = perf_counter()
            builder.build(segments)
            elapsed = perf_counter() - started
            report["build_benchmark"] = {
                "status": "measured_warm_model",
                "seconds": elapsed,
                "physical_images_opened_and_encoded": memberships,
                "unique_physical_frames": len(physical_paths),
                "avoidable_overlap_reencodes": memberships - len(physical_paths),
                "peak_gpu": cuda_memory(),
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

# TutorialVQA scoped end-to-end evaluation

Run: Slurm job 18398, completed with exit code 0 on 2026-09-20 in 7m03s.

## Protocol

- Data: 5 selected TutorialVQA videos (14643-14647).
- Splits: official question splits, dev=61 and test=55. The same videos occur across question splits, so this is not a video-disjoint generalization test.
- Input: bounded Whisper-small transcripts, extracted frames, and PaddleOCR evidence.
- Retrieval: current cascade configuration, scoped to each question's `video_id`; no RRF.
- Relevance: an indexed segment is relevant when its time interval overlaps an official TutorialVQA answer interval.
- Generation: Qwen2.5-VL with citations. Reference answers and intervals are used only after inference for scoring.
- All TutorialVQA items in these splits are answerable. Abstention quality therefore requires a separate negative set.

## Results

| Metric | Dev (61) | Test (55) |
|---|---:|---:|
| Recall@1 | 0.5574 | 0.5636 |
| Recall@5 | 0.8852 | 0.8909 |
| Recall@10 | 0.9672 | 1.0000 |
| MRR | 0.7005 | 0.7073 |
| nDCG@5 | 0.4130 | 0.3863 |
| nDCG@10 | 0.4449 | 0.4276 |
| Mean best temporal IoU@1 | 0.2467 | 0.2500 |
| Mean best temporal IoU@5 | 0.4110 | 0.4013 |
| Mean best temporal IoU@10 | 0.4552 | 0.4706 |
| Token F1 | 0.6156 | 0.6048 |
| Exact match | 0.0000 | 0.0000 |
| Answered rate | 0.9180 | 0.9818 |
| Citation validity among answered | 1.0000 | 1.0000 |
| Citation in selected video among answered | 1.0000 | 1.0000 |
| Evidence hit among answered | 0.8393 | 0.8519 |
| Mean evidence precision among answered | 0.5909 | 0.5864 |
| Grounded answer rate | 0.7705 | 0.8364 |
| Scope violations | 0 | 0 |

## Latency

| Metric | Dev | Test |
|---|---:|---:|
| Retrieval mean | 1.904 ms | 1.258 ms |
| Retrieval median | 0.408 ms | 0.414 ms |
| Retrieval p95 | 0.615 ms | 0.521 ms |
| End-to-end answer mean | 3.505 s | 3.477 s |
| End-to-end answer median | 3.291 s | 3.229 s |
| End-to-end answer p95 | 5.783 s | 5.723 s |

Exact match is expected to be low for free-form procedural answers and is reported only as a strict auxiliary metric. Token F1, evidence hit, grounded answer rate, and temporal retrieval metrics are more informative for this product.

Machine-readable reports and per-question predictions are stored under `artifacts/tutorialvqa/metrics/` on the school server.

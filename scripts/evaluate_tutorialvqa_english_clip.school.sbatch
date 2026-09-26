#!/bin/bash
#SBATCH --job-name=tutorial-enclip
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/data/zzhu126/VideoRAG/logs/tutorial-enclip-%j.out
#SBATCH --error=/data/zzhu126/VideoRAG/logs/tutorial-enclip-%j.err
set -euo pipefail
cd /data/zzhu126/VideoRAG
source env.school.sh
mkdir -p "$TMPDIR" logs artifacts/tutorialvqa/metrics-en-clip
python -u scripts/build_indexes.py \
  --segments artifacts/tutorialvqa/bounded-v2/segments.ocr.jsonl \
  --index-dir artifacts/tutorialvqa/bounded-v2/indexes-en-clip \
  --config config.tutorialvqa.toml \
  --device cuda
for split in dev test; do
  python -u scripts/evaluate_tutorialvqa.py \
    --questions "artifacts/tutorialvqa/evaluation/${split}.jsonl" \
    --output-dir artifacts/tutorialvqa/metrics-en-clip \
    --segments artifacts/tutorialvqa/bounded-v2/segments.ocr.jsonl \
    --index-dir artifacts/tutorialvqa/bounded-v2/indexes-en-clip \
    --config config.tutorialvqa.toml \
    --device cuda \
    --top-k 10
done

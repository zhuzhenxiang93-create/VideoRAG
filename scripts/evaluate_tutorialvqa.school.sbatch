#!/bin/bash
#SBATCH --job-name=tutorial-eval
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/data/zzhu126/VideoRAG/logs/tutorial-eval-%j.out
#SBATCH --error=/data/zzhu126/VideoRAG/logs/tutorial-eval-%j.err
set -euo pipefail
cd /data/zzhu126/VideoRAG
source env.school.sh
mkdir -p "$TMPDIR" logs artifacts/tutorialvqa/metrics
for split in dev test; do
  python -u scripts/evaluate_tutorialvqa.py \
    --questions "artifacts/tutorialvqa/evaluation/${split}.jsonl" \
    --output-dir artifacts/tutorialvqa/metrics \
    --segments artifacts/tutorialvqa/bounded-v2/segments.ocr.jsonl \
    --index-dir artifacts/tutorialvqa/bounded-v2/indexes \
    --config config.tutorialvqa.toml \
    --device cuda \
    --top-k 10 \
    --generate
 done

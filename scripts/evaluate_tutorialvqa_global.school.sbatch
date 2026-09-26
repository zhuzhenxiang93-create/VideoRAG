#!/bin/bash
#SBATCH --job-name=tutorial-global
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=/data/zzhu126/VideoRAG/logs/tutorial-global-%j.out
#SBATCH --error=/data/zzhu126/VideoRAG/logs/tutorial-global-%j.err
set -euo pipefail
cd /data/zzhu126/VideoRAG
source env.school.sh
mkdir -p "$TMPDIR" logs
python -u scripts/evaluate_tutorialvqa_global.py \
  --questions artifacts/tutorialvqa/scale-76/evaluation/test.jsonl \
  --output-dir artifacts/tutorialvqa/scale-76/metrics-global-smoke \
  --limit 10 --generate-sample 3
python -u scripts/evaluate_tutorialvqa_global.py \
  --questions artifacts/tutorialvqa/scale-76/evaluation/test.jsonl \
  --output-dir artifacts/tutorialvqa/scale-76/metrics-global \
  --generate-sample 30

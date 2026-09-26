#!/bin/bash
#SBATCH --job-name=tutorial-demo
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/data/zzhu126/VideoRAG/logs/tutorial-demo-%j.out
#SBATCH --error=/data/zzhu126/VideoRAG/logs/tutorial-demo-%j.err
set -euo pipefail
cd /data/zzhu126/VideoRAG
source env.school.sh
mkdir -p "$TMPDIR" logs
python -u scripts/run_server.py \
  --segments artifacts/tutorialvqa/scale-76/segments.ocr.jsonl \
  --index-dir artifacts/tutorialvqa/scale-76/indexes-en-clip \
  --config config.tutorialvqa.full-demo.toml \
  --host 127.0.0.1 --port 5000 --device cuda

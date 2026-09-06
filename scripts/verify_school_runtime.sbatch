#!/bin/bash
#SBATCH --job-name=videorag-check
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/data/zzhu126/VideoRAG/logs/videorag-%j.out
#SBATCH --error=/data/zzhu126/VideoRAG/logs/videorag-%j.err

set -euo pipefail

cd /data/zzhu126/VideoRAG
source env.school.sh
mkdir -p "$TMPDIR"

python -u scripts/verify_school_runtime.py
python scripts/run_server.py \
  --segments "${VIDEORAG_SEGMENTS:-artifacts/segments.ocr.jsonl}" \
  --index-dir "${VIDEORAG_INDEX_DIR:-artifacts/indexes-ocr}" \
  --config "${VIDEORAG_CONFIG:-config.toml}" \
  --host 127.0.0.1 \
  --port "${VIDEORAG_PORT:-5000}" \
  --device cuda

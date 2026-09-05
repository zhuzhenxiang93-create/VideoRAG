#!/bin/bash
#SBATCH --job-name=videorag-api
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

python scripts/run_server.py \
  --segments artifacts/segments.ocr.jsonl \
  --index-dir artifacts/indexes-ocr \
  --config config.toml \
  --host 127.0.0.1 \
  --port 5000 \
  --device cuda

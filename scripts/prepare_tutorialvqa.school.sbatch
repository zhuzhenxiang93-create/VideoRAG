#!/bin/bash
#SBATCH --job-name=tutorial-prepare
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/data/zzhu126/VideoRAG/logs/tutorial-prepare-%j.out
#SBATCH --error=/data/zzhu126/VideoRAG/logs/tutorial-prepare-%j.err
set -euo pipefail
cd /data/zzhu126/VideoRAG
source env.school.sh
source env.school-network.sh
python -u scripts/prepare_tutorialvqa.py
PYTHONPATH="$PWD/.runtime-ocr:$PWD/src" HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -u scripts/tutorial_ocr.py
python -u scripts/build_indexes.py --segments artifacts/tutorialvqa/segments.ocr.jsonl --index-dir artifacts/tutorialvqa/indexes --config config.tutorialvqa.toml --device cuda

#!/bin/bash
#SBATCH --job-name=tutorial-scale76
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=/data/zzhu126/VideoRAG/logs/tutorial-scale76-%j.out
#SBATCH --error=/data/zzhu126/VideoRAG/logs/tutorial-scale76-%j.err
set -euo pipefail
cd /data/zzhu126/VideoRAG
source env.school.sh
mkdir -p "$TMPDIR" logs artifacts/tutorialvqa/scale-76/metrics
python -u scripts/prepare_tutorialvqa_scale.py --video-count 76 --output-root artifacts/tutorialvqa/scale-76
OMP_NUM_THREADS=12 PYTHONPATH="$PWD/.runtime-ocr:$PWD/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -u scripts/tutorial_ocr_scale.py --root artifacts/tutorialvqa/scale-76
python -u scripts/build_indexes.py --segments artifacts/tutorialvqa/scale-76/segments.ocr.jsonl --index-dir artifacts/tutorialvqa/scale-76/indexes-en-clip --config config.tutorialvqa.toml --device cuda
python -u scripts/export_tutorialvqa_scale_evaluation.py --root artifacts/tutorialvqa/scale-76
for split in dev test; do
  python -u scripts/evaluate_tutorialvqa.py --questions "artifacts/tutorialvqa/scale-76/evaluation/${split}.jsonl" --output-dir artifacts/tutorialvqa/scale-76/metrics --segments artifacts/tutorialvqa/scale-76/segments.ocr.jsonl --index-dir artifacts/tutorialvqa/scale-76/indexes-en-clip --config config.tutorialvqa.toml --device cuda --top-k 10
done

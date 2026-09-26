#!/bin/bash
#SBATCH --job-name=tutorial-clipcmp
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/data/zzhu126/VideoRAG/logs/tutorial-clipcmp-%j.out
#SBATCH --error=/data/zzhu126/VideoRAG/logs/tutorial-clipcmp-%j.err
set -euo pipefail
cd /data/zzhu126/VideoRAG
source env.school.sh
mkdir -p "$TMPDIR" logs artifacts/tutorialvqa/clip-ablation
python -u scripts/compare_tutorial_clip.py

#!/bin/bash
#SBATCH --job-name=tutorial-diagnose
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/data/zzhu126/VideoRAG/logs/tutorial-diagnose-%j.out
#SBATCH --error=/data/zzhu126/VideoRAG/logs/tutorial-diagnose-%j.err
set -euo pipefail
cd /data/zzhu126/VideoRAG
source env.school.sh
mkdir -p "$TMPDIR" logs artifacts/tutorialvqa/diagnostics
python -u scripts/diagnose_tutorial_generation.py

#!/bin/bash
#SBATCH --job-name=tutorial-final
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

python -u scripts/repair_tutorial_timestamps.py
python -u scripts/build_indexes.py --segments artifacts/tutorialvqa/bounded-v2/segments.ocr.jsonl --index-dir artifacts/tutorialvqa/bounded-v2/indexes --config config.tutorialvqa.toml --device cuda
python -u scripts/run_server.py --segments artifacts/tutorialvqa/bounded-v2/segments.ocr.jsonl --index-dir artifacts/tutorialvqa/bounded-v2/indexes --config config.tutorialvqa.toml --host 127.0.0.1 --port 5000 --device cuda &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT
python -u scripts/verify_tutorial_api.py
PYTHONPATH="$PWD/.runtime-browser:$PWD/src" PLAYWRIGHT_BROWSERS_PATH="$PWD/.runtime-browser/browsers" python -u scripts/verify_tutorial_browser.py || echo "BROWSER_CHECK_FAILED; see error log"
wait "$server_pid"

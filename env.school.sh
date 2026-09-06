export VIDEORAG_ROOT=/data/zzhu126/VideoRAG
export HF_HOME=/data/zzhu126/cache/videorag/huggingface
export HUGGINGFACE_HUB_CACHE=/data/zzhu126/cache/videorag/huggingface/hub
export TORCH_HOME=/data/zzhu126/cache/videorag/torch
export PIP_CACHE_DIR=/data/zzhu126/cache/videorag/pip
export XDG_CACHE_HOME=/data/zzhu126/cache/videorag
export TMPDIR=/data/zzhu126/tmp/videorag
export PADDLE_PDX_CACHE_HOME=/data/zzhu126/cache/videorag/paddlex
export PATH=/data/zzhu126/environments/videorag/bin:$PATH
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1

# Reuse the school-installed ffmpeg binary without changing other environments.
export PATH="$PATH:/data/zzhu126/environments/vbench-tpami/bin"
export PYTHONPATH="$VIDEORAG_ROOT/.runtime-ocr${PYTHONPATH:+:$PYTHONPATH}"

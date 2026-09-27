# 启动与复现

## 学校服务器
主机 zzhu126@foscsmlprd01.its.auckland.ac.nz，项目 /data/zzhu126/VideoRAG。2026-09-27 核对的 Slurm 节点也是 foscsmlprd01；未来换节点时应调整 SSH 跳板转发，不沿用回环地址假设。

```bash
cd /data/zzhu126/VideoRAG
source env.school.sh
git branch --show-current # demo/tutorial-video-qa
squeue -j 19048
scontrol show job 19048
sacct -j 19048 --format=JobID,State,ExitCode,Elapsed,NodeList
```

19048 已提交，当前 PENDING / JobHeldUser / PENDING_APPROVAL，**不要重复提交或取消**。Slurm 保存提交时的 sbatch 副本；调用的 Python 文件启动时读取，本次评测修复会生效。先跑 10 问检索/3 问生成，再跑 1,239 问检索/30 问生成样本。

```bash
tail -f logs/tutorial-global-19048.out logs/tutorial-global-19048.err
ls artifacts/tutorialvqa/scale-76/metrics-global/
ls artifacts/tutorialvqa/scale-76/metrics-global-smoke/
```

需要交互 GPU 服务时再提交服务作业（本次没有额外提交）：

```bash
sbatch scripts/run_tutorial_full_demo.school.sbatch
squeue -u "$USER"
# 用返回 ID 替换 JOB_ID
tail -f logs/tutorial-demo-JOB_ID.out logs/tutorial-demo-JOB_ID.err
```

在已获批 GPU 分配内，可直接启动：

```bash
python scripts/run_server.py \
  --segments artifacts/tutorialvqa/scale-76/segments.ocr.jsonl \
  --index-dir artifacts/tutorialvqa/scale-76/indexes-en-clip \
  --config config.tutorialvqa.full-demo.toml \
  --video-metadata data/tutorialvqa/original/repository/videos.json \
  --host 127.0.0.1 --port 5000 --device cuda
```

个人电脑转发（不把认证信息发给任何人）：

```bash
ssh -S /tmp/videorag-ssh -O forward \
  -L 15000:127.0.0.1:5000 zzhu126@foscsmlprd01.its.auckland.ac.nz
# 无共享会话时：
ssh -N -L 15000:127.0.0.1:5000 zzhu126@foscsmlprd01.its.auckland.ac.nz
```

访问 http://127.0.0.1:15000 。首次请求可能加载精排/生成模型；页面请求耗时含等待，预热评测不含首次加载。服务单线程避免并行请求抢占 GPU。

## CPU 预览
需要已经处理好的片段和可读原视频，不随仓库分发数据。

```bash
cd /data/zzhu126/VideoRAG
source env.school.sh
python scripts/run_ui_preview.py --port 5001
```

本机转发：

```bash
ssh -S /tmp/videorag-ssh -O forward \
  -L 15001:127.0.0.1:5001 zzhu126@foscsmlprd01.its.auckland.ac.nz
```

访问 http://127.0.0.1:15001 。真实 BM25 与媒体播放可用，黄色提示明确没有 Embedding、Qwen 精排或生成；/api/ask 返回 503。本次服务日志 logs/ui-preview.log。

## 其他 GPU 环境
建议 Linux、Python 3.10+、FFmpeg、64GB RAM、100GB 磁盘和 24GB+ NVIDIA 显存，优先 A100；显存下限不是本次测得的保证。按驱动安装兼容 PyTorch，不要 source 学校硬编码环境脚本。

```bash
git clone --branch demo/tutorial-video-qa https://github.com/zhuzhenxiang93-create/VideoRAG.git
cd VideoRAG
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[models,video,ocr,dev]"
ffmpeg -version
python scripts/fetch_tutorialvqa.py --download-videos
```

联网阶段准备 Qwen/Qwen3-Embedding-0.6B、Qwen/Qwen3-Reranker-0.6B、Qwen/Qwen2.5-VL-7B-Instruct、openai/clip-vit-large-patch14、Whisper small 和英文 PaddleOCR，各按模型自己的许可使用。计算节点离线时预先缓存；PaddlePaddle 固定 3.2.2。

```bash
python scripts/prepare_tutorialvqa_scale.py --video-count 76 --output-root artifacts/tutorialvqa/scale-76
python scripts/tutorial_ocr_scale.py --root artifacts/tutorialvqa/scale-76
python scripts/build_indexes.py \
  --segments artifacts/tutorialvqa/scale-76/segments.ocr.jsonl \
  --index-dir artifacts/tutorialvqa/scale-76/indexes-en-clip \
  --config config.tutorialvqa.full-demo.toml --device cuda
python scripts/export_tutorialvqa_scale_evaluation.py --root artifacts/tutorialvqa/scale-76
```

随后运行上面的 run_server.py 命令，24GB 环境可加 --low-vram，反复卸载会增加延迟。换模型须重建索引；跨主机复制片段时绝对视频/关键帧路径须重定位并重建索引/manifest。

```bash
python scripts/evaluate_tutorialvqa_global.py \
  --questions artifacts/tutorialvqa/scale-76/evaluation/test.jsonl \
  --output-dir artifacts/tutorialvqa/scale-76/metrics-global \
  --generate-sample 30
```

输出 run-config.json（配置、哈希、提交）、predictions.jsonl（逐题排序、引用时间、耗时）、report.json。时间判定为同视频且正长度重叠；精排复用同一冻结候选池。P95 为排序后 round((n-1)*0.95) 位置。问答 F1 为辅助自动指标，不能替代人工事实检查。

## 验证命令
```bash
python -m pytest -q
python -m ruff check src scripts tests
# 浏览器额外依赖：
python -m pip install playwright
python -m playwright install chromium
# 先启动 5001 预览
python scripts/verify_portfolio_browser.py
```

学校已装运行时：
```bash
PYTHONPATH="$PWD/.runtime-browser:$PYTHONPATH" \
PLAYWRIGHT_BROWSERS_PATH="$PWD/.runtime-browser/browsers" \
python scripts/verify_portfolio_browser.py
```

浏览器脚本区分真实检索/播放与 UI 响应夹具，夹具不算模型验收。GitHub Pages 只能显示静态文档，不能运行 Flask、GPU 模型或受 SSH 保护的视频。

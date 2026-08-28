# VideoRAG：片段级中文多模态视频问答

一个可复现的片段级 VideoRAG 系统：将视频转成带时间戳的 ASR、关键帧和视觉描述，使用稀疏、文本向量和图文向量三路召回，经 RRF 融合与 Qwen3-Reranker 精排后，由 Qwen2.5-VL 基于 Top-K 证据生成答案。API 同时返回证据片段、时间范围和阶段延迟，前端可以直接跳转到对应视频位置。

> 当前状态：核心流水线、索引校验、帧级检索实验、评测工具和人工复核系统均已实现；仓库中的问题集仍是自动生成候选集，尚不能把诊断指标当作正式人工测试集结果。

## 系统架构

```text
视频
 ├─ Whisper 时间戳 ASR ─────────────────────────────┐
 └─ 场景变化/清晰度关键帧 ─ Qwen2.5-VL 视觉描述 ────┤
                                                     ▼
                                  20 秒窗口、5 秒重叠的视频片段
                                                     │
             ┌───────────────────┬───────────────────┴───────────────────┐
             ▼                   ▼                                       ▼
      BM25-like 稀疏召回   Qwen3-Embedding + FAISS        Chinese-CLIP + FAISS
             └───────────────────┴───────────────────┬───────────────────┘
                                                     ▼
                                               RRF 排名融合
                                                     ▼
                                           Qwen3-Reranker 精排
                                                     ▼
                                      Qwen2.5-VL 证据约束生成
                                                     ▼
                                  答案 + 证据 + 时间戳 + 阶段延迟
```

## 核心能力

- `VideoSegment` 统一保存视频 ID、源路径、起止时间、ASR、视觉描述和关键帧。
- Whisper 保留 ASR 片段时间戳，并与 20 秒滑动窗口自动对齐。
- 每秒采样视频，结合场景变化、Laplacian 清晰度和时间去重选择关键帧。
- Qwen2.5-VL 生成客观关键帧描述，并基于最终证据完成一次性多模态生成。
- BM25-like、Qwen3-Embedding、Chinese-CLIP 三路召回，使用 RRF 避免异构分数直接标定。
- Qwen3-Reranker 对融合候选精排，低于置信度阈值时拒答。
- 所有向量转为 L2 归一化 `float32`，使用 FAISS `IndexFlatIP` 实现余弦相似度精确检索。
- 索引 manifest 记录模型、维度、条目数、相似度定义和文件 SHA-256，启动时拒绝不一致索引。
- Flask API 返回结构化证据，网页播放器支持跳转到证据 `start_time`。
- Recall@K、MRR、nDCG@K、Exact Match、字符级 Token F1 和分阶段延迟评测。
- append-only 人工复核事件、revision 冲突检测、事件备份和 video-disjoint split 校验。

## 默认模型

| 阶段 | 模型 |
|---|---|
| ASR | `openai/whisper-small` |
| 文本向量 | `Qwen/Qwen3-Embedding-0.6B` |
| 中文图文向量 | `OFA-Sys/chinese-clip-vit-base-patch16` |
| 候选精排 | `Qwen/Qwen3-Reranker-0.6B` |
| 视觉描述与答案生成 | `Qwen/Qwen2.5-VL-7B-Instruct` |

配置集中在 [`config.toml`](config.toml)。

## 项目结构

```text
src/video_rag/
  adapters/       # CPU 演示与 Qwen 模型适配器
  api/            # Flask API 与视频证据页面
  evaluation/     # 检索、答案、数据集和复核事件校验
  ingestion/      # 视频探测、Whisper ASR、关键帧和片段化
  retrieval/      # 稀疏、FAISS、物理帧检索与 RRF
  pipeline.py     # Recall -> Fusion -> Rerank -> Generate 编排
scripts/          # 数据下载、预处理、建库、评测和人工标注工具
tests/            # 不加载大模型的单元与工作流测试
data/evaluation/  # 候选问题、复核队列和数据说明
docs/             # 人工标注规范
```

## 快速验证（CPU，无需下载大模型）

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/run_demo.py
```

示例请求：

```bash
curl -X POST http://127.0.0.1:5000/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"CLIP有什么作用？"}'
```

返回值包含：

```json
{
  "answer": "...",
  "abstained": false,
  "evidence": [
    {
      "segment_id": "demo_0001",
      "video_id": "demo",
      "start_time": 15.0,
      "end_time": 35.0,
      "transcript": "...",
      "visual_caption": "...",
      "fused_score": 0.016,
      "rerank_score": 0.5,
      "video_url": "/api/videos/demo"
    }
  ],
  "latency_ms": {
    "recall": 0.2,
    "fusion": 0.1,
    "rerank": 0.1,
    "generation": 0.1,
    "total": 0.5
  }
}
```

## GPU 完整复现

建议使用 Ubuntu 22.04、Python 3.11+、FFmpeg、单张 24 GB 以上 NVIDIA GPU、64 GB RAM 和至少 100 GB 磁盘。

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[models,video]"
```

### 1. 获取开放许可视频（可选）

```bash
python scripts/download_open_news.py \
  --output data/raw/open_news_zh \
  --count 24
```

下载脚本从 Wikimedia Commons 筛选 Public Domain、CC0 或 CC BY 视频，并在 manifest 中保留来源、许可证和作者信息。原始视频不提交到 Git。

### 2. 预处理视频

```bash
python scripts/prepare_videos.py \
  --input data/raw/open_news_zh \
  --output artifacts/segments.jsonl \
  --frames-dir artifacts/frames \
  --language zh
```

如只验证 ASR 和切片，可增加 `--skip-captions`。

### 3. 构建文本和视觉索引

```bash
python scripts/build_indexes.py \
  --segments artifacts/segments.jsonl \
  --index-dir artifacts/indexes
```

### 4. 启动真实服务

```bash
# 默认模式
python scripts/run_server.py --host 127.0.0.1 --port 5000

# 24 GB 显存：精排后卸载 Reranker，再加载 Qwen-VL
python scripts/run_server.py --host 127.0.0.1 --port 5000 --low-vram
```

开发服务默认只绑定本机。远程使用建议通过 SSH 隧道访问；如需公网部署，应增加反向代理、认证、限流和生产级 WSGI 服务。

## 物理帧检索实验

重叠窗口会使同一关键帧出现在多个片段中。`FrameClipVisionRetriever` 以唯一物理帧建库，保存所有片段成员关系，并在查询时使用 `max` 或 `top2_mean` 聚合回片段，避免重复图片编码。

```bash
python scripts/build_frame_index.py \
  --segments artifacts/segments.jsonl \
  --index-dir artifacts/indexes

python scripts/evaluate_visual_granularity.py --help
python scripts/audit_keyframes.py --help
```

该实现已经用于受控实验；默认在线服务仍使用稳定的片段均值视觉索引，二者应在正式人工测试集上进一步比较后再切换。

## 检索消融

```bash
python scripts/evaluate_retrieval.py \
  --questions data/evaluation/questions.zh.seed.jsonl \
  --segments artifacts/segments.jsonl \
  --index-dir artifacts/indexes \
  --output artifacts/evaluation/retrieval_ablation.json \
  --csv-output artifacts/evaluation/retrieval_ablation.csv \
  --with-reranker
```

脚本输出单路、双路、三路 RRF 和可选 Reranker 的 Recall@1/5/10、MRR、nDCG 及延迟。

**评测边界：** 仓库中的 seed、candidate 和 review queue 都是自动生成候选数据，不是人工金标准。它们可用于流程验证和诊断实验，但不能用于对外宣称正式准确率。详细字段和限制见 [`data/evaluation/README.md`](data/evaluation/README.md)。

## 人工复核

候选数据经过实际视频播放和证据核验后，才能标记为 `verified`：

```bash
python scripts/annotation_server.py \
  --candidates data/evaluation/questions.zh.review_queue.v1.jsonl \
  --segments artifacts/segments.supplement_a.jsonl \
  --events artifacts/annotations/review_events.jsonl \
  --reviewer-id YOUR_REVIEWER_ID
```

标注服务强制绑定 `127.0.0.1`。完整要求见 [`docs/ANNOTATION_GUIDE_ZH.md`](docs/ANNOTATION_GUIDE_ZH.md)。正式指标至少应满足：

- 问题经过逐条人工视频核验；
- 保存 append-only 复核事件与候选文件 SHA-256；
- 按视频冻结 development/validation/test，避免相邻片段泄漏；
- 单独报告 audio、visual、OCR、multimodal 和 unanswerable；
- 同时报告检索、答案质量、P50/P95 延迟和显存峰值。

## 测试与质量约束

```bash
python -m pytest -q
python -m ruff check src scripts tests
```

测试不会下载或加载大模型，覆盖片段化、RRF、Pipeline、FAISS 数值规范、帧级聚合、索引 manifest、API、数据集验证、split 防泄漏和复核事件生命周期。真实 GPU 验收仍需单独执行完整预处理、建库和问答流程。

# VideoRAG：片段级中文多模态视频问答

一个可复现的片段级 VideoRAG 系统：将视频转成带时间戳的 ASR、OCR、关键帧和视觉描述，通过多标签查询路由选择主检索器，仅在低置信度时级联回退；多模态问题对相关通道候选取并集并统一精排。候选经过重叠去重和上下文扩展后，由 Qwen-VL 基于可验证引用生成答案。API 同时返回证据片段、时间范围、路由标签、置信度和阶段延迟。

> 当前状态：核心流水线、索引校验、帧级检索实验、评测工具和人工复核系统均已实现；仓库中的问题集仍是自动生成候选集，尚不能把诊断指标当作正式人工测试集结果。

## 系统架构

```text
视频
 ├─ Whisper 时间戳 ASR ─────────────────────────────┐
 ├─ 关键帧 ─ PaddleOCR 时间戳文字 ──────────────────┤
 └─ 场景变化/清晰度关键帧 ─ Qwen-VL 视觉描述 ───────┤
                                                     ▼
                              ASR/场景边界感知的语义视频片段
                                                     │
                                         查询意图路由
                           语音/视觉/OCR/多模态/时序
                                           │
             ┌───────────────────┬─────────┴─────────┬───────────────────┐
             ▼                   ▼                                       ▼
       Okapi BM25      独立 OCR BM25       Qwen3-Embedding / Qwen3-VL
             └───────────────────┴───────────────────┬───────────────────┘
                                                     ▼
                        主检索优先 / 低置信度回退 / 多模态候选并集
                                                     ▼
                           可选多模态精排 + 时序邻居上下文扩展
                                                     ▼
                                  Qwen-VL 结构化证据约束生成
                                                     ▼
                         答案 + 已验证引用 + 时间戳 + 置信度 + 延迟
```

## 核心能力

- `VideoSegment` 统一保存视频 ID、源路径、起止时间、ASR、OCR、视觉描述和关键帧。
- Whisper 保留 ASR 片段时间戳，并与语义窗口自动对齐。
- PaddleOCR 在关键帧上提取文字、置信度、坐标和时间戳，连续覆盖文字自动去重，并使用独立 `ocr_bm25` 召回。
- 语义切片优先在 ASR 句末或场景边界结束，同时保留固定窗口兼容模式。
- 每秒采样视频，结合场景变化、Laplacian 清晰度和时间去重选择关键帧。
- Qwen2.5-VL 生成客观关键帧描述，并基于最终证据完成一次性多模态生成。
- 可配置 Okapi BM25、OCR BM25、Qwen3-Embedding、Chinese-CLIP/Qwen3-VL 四路召回；默认 `b=0`，避免片段文本长度差异造成不稳定惩罚。
- 四路召回分别使用独立 Top-K，便于调参与消融。
- 多标签路由识别 text、semantic、visual、ocr、multimodal 和 temporal；事实题默认 BM25、概括题默认文本向量、视觉题默认视觉向量、文字读取题默认 OCR。
- 单模态主检索未达到来源独立阈值时才执行回退；多模态问题轮询合并相关通道候选，避免弱检索器通过固定 RRF 稀释强结果。
- 重叠候选按时间交并去重；普通问题扩展相邻片段，时序问题使用更宽的前后上下文。
- 生成端严格返回 `answerable/answer/confidence/citations`；未知引用、无引用、低置信度或证据不足都会触发拒答。
- 实测文本 Qwen3-Reranker 在当前诊断集产生负收益，因此默认保留融合排序；Qwen3-VL 多模态精排作为可选升级。
- 可选 Qwen3-VL-Embedding、Qwen3-VL-Reranker 和 Qwen3-VL 生成端到端升级路径；旧模型配置仍可直接运行。
- 所有向量转为 L2 归一化 `float32`，使用 FAISS `IndexFlatIP` 实现余弦相似度精确检索。
- 索引 manifest 记录模型、维度、条目数、相似度定义和文件 SHA-256，启动时拒绝不一致索引。
- Flask API 返回结构化证据，网页播放器支持跳转到证据 `start_time`。
- Recall@K、MRR、nDCG@K、Exact Match、字符级 Token F1 和分阶段延迟评测。
- append-only 人工复核事件、revision 冲突检测、事件备份和 video-disjoint split 校验。

## 默认模型

| 阶段 | 模型 |
|---|---|
| ASR | `openai/whisper-small` |
| OCR | PaddleOCR（中文） |
| 文本向量 | `Qwen/Qwen3-Embedding-0.6B` |
| 中文图文向量 | `OFA-Sys/chinese-clip-vit-base-patch16` |
| 候选精排 | 默认保持融合排序；可选 `Qwen/Qwen3-Reranker-0.6B` |
| 视觉描述与答案生成 | `Qwen/Qwen2.5-VL-7B-Instruct` |

配置集中在 [`config.toml`](config.toml)。

仓库提供两套可切换配置：

- `config.toml`：稳定模式，采用查询路由级联 + BM25/OCR/文本向量/Chinese-CLIP + Qwen2.5-VL；RRF 仅作为可切换消融基线。
- `config.qwen3-vl.toml`：升级模式，事实题保留低延迟 BM25 路径，OCR题启用独立OCR召回，视觉题才用片段文本与有序关键帧进行 Qwen3-VL 召回和精排。

四项升级的字段、路由和拒答规则见 [`docs/GROUNDED_MULTIMODAL_PIPELINE_ZH.md`](docs/GROUNDED_MULTIMODAL_PIPELINE_ZH.md)。

## 项目结构

```text
src/video_rag/
  adapters/       # CPU 演示与 Qwen 模型适配器
  api/            # Flask API 与视频证据页面
  evaluation/     # 检索、答案、数据集和复核事件校验
  ingestion/      # 视频探测、Whisper ASR、关键帧和片段化
  retrieval/      # 稀疏、FAISS、物理帧、级联选择与 RRF 基线
  pipeline.py     # Route -> Cascade/Union -> Rerank -> Generate 编排
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

完整视频预处理需要安装模型、视频和OCR依赖；旧 JSONL 仍可读取，但只有重新预处理后才会包含OCR和语义切片字段：

```bash
python -m pip install -e ".[models,video,ocr]"
python scripts/prepare_videos.py --input data/raw --output artifacts/segments.semantic.jsonl
```

OCR 依赖固定使用 PaddlePaddle 3.2.2；3.3.x 的 CPU oneDNN/PIR 推理路径存在上游兼容问题。

已有片段可只补OCR而不重复运行ASR和视觉描述：

```bash
python scripts/enrich_ocr.py \
  --segments artifacts/segments.jsonl \
  --output artifacts/segments.ocr.jsonl
```

默认配置的既有向量不包含 OCR，因而向量值无需重算；复制索引并更新片段哈希后即可启用独立 OCR 召回：

```bash
cp -a artifacts/indexes artifacts/indexes-ocr
python scripts/refresh_index_manifest.py \
  --segments artifacts/segments.ocr.jsonl \
  --index-dir artifacts/indexes-ocr
python scripts/run_server.py \
  --segments artifacts/segments.ocr.jsonl \
  --index-dir artifacts/indexes-ocr \
  --low-vram
```

若使用 `config.qwen3-vl.toml`，多模态向量会读取 OCR 证据文本，应改用 `build_indexes.py` 为 `segments.ocr.jsonl` 完整重建新索引。

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

如果使用 Qwen3-VL 升级配置，安装额外依赖并克隆官方实现：

```bash
python -m pip install -e ".[models,video,qwen3]"
git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git \
  /root/autodl-tmp/Qwen3-VL-Embedding
python -m pip install -e /root/autodl-tmp/Qwen3-VL-Embedding
```

`config.qwen3-vl.toml` 中的 `qwen3_vl_repository` 必须指向这个官方仓库。适配器只动态调用官方的 `Qwen3VLEmbedder` 和 `Qwen3VLReranker`，本项目不复制或修改其源码。

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

升级模式需要用独立配置重建索引；构建脚本会生成 schema v3 manifest，记录文本与多模态索引各自的模型和哈希：

```bash
python scripts/build_indexes.py \
  --segments artifacts/segments.jsonl \
  --index-dir artifacts/indexes-qwen3-vl \
  --config config.qwen3-vl.toml
```

### 4. 启动真实服务

```bash
# 默认模式
python scripts/run_server.py --host 127.0.0.1 --port 5000

# 24 GB 显存：精排后卸载 Reranker，再加载 Qwen-VL
python scripts/run_server.py --host 127.0.0.1 --port 5000 --low-vram

# Qwen3-VL 升级模式
python scripts/run_server.py \
  --segments artifacts/segments.jsonl \
  --index-dir artifacts/indexes-qwen3-vl \
  --config config.qwen3-vl.toml \
  --host 127.0.0.1 --port 5000 --low-vram
```

升级模式的 `frame_sequence` 会按证据片段顺序把最多 16 张关键帧作为一个视频帧序列交给 Qwen3-VL，同时附上 Top-3 片段的 ASR、视觉描述、片段 ID、起止时间和原始问题。它不是把整段原视频无裁剪地塞进模型，因此能控制视觉 token 和显存开销，也不会丢失证据时间戳。

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

脚本输出单路、路由级联、双路/三路 RRF 和可选 Reranker 的 Recall@1/5/10、MRR、nDCG 及延迟。

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

测试不会下载或加载大模型，覆盖片段化、查询路由、级联回退、候选并集、RRF 基线、Pipeline、FAISS 数值规范、帧级聚合、索引 manifest、API、数据集验证、split 防泄漏和复核事件生命周期。真实 GPU 验收仍需单独执行完整预处理、建库和问答流程。

# Improved Video RAG

这是原多模态视频问答原型的重构版本。它把检索单位从“完整视频”改为“带时间戳的视频片段”，并明确采用：

```text
多路召回 -> RRF 融合 -> 候选证据精排 -> 基于证据生成一次答案
```

项目同时提供 CPU-only 演示适配器和真实 GPU 模型适配器。没有 GPU 时可以验证架构、数据格式、评测和 API；云端环境可直接执行视频预处理、FAISS 建库、真实问答及消融实验。

## 已完成

- 带起止时间的 `VideoSegment` 数据模型；
- 滑动窗口切片；
- ASR 时间段与关键帧自动对齐；
- Retriever/Reranker/Generator 可替换接口；
- CPU BM25-like 开发召回器；
- Reciprocal Rank Fusion，多路内部自动去重；
- Recall -> Fusion -> Rerank -> Generate 编排；
- 低置信度拒答；
- 结构化答案与证据 API；
- Recall@K、MRR、nDCG@K；
- 无 GPU 单元测试和演示服务。
- Whisper 带时间戳 ASR；
- 基于采样、场景变化和清晰度的关键帧提取；
- Qwen2.5-VL 关键帧客观描述；
- Qwen3-Embedding + FAISS 文本语义召回；
- OFA Chinese-CLIP + FAISS 中文图文召回；
- Qwen3-Reranker 候选证据精排；
- Qwen2.5-VL 基于 Top-3 证据单次生成；
- 点击证据跳转至视频对应时间；
- 索引 manifest、答案评测和批量预测脚本。

## 本地运行

```powershell
cd improved_video_rag
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python scripts\run_demo.py
```

请求示例：

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:5000/api/ask `
  -ContentType application/json `
  -Body '{"question":"CLIP有什么作用？"}'
```

## 返回格式

```json
{
  "answer": "...",
  "abstained": false,
  "evidence": [
    {
      "segment_id": "demo_0001",
      "video_id": "demo",
      "source_path": "data/demo/demo.mp4",
      "start_time": 15,
      "end_time": 35,
      "transcript": "...",
      "visual_caption": "...",
      "fused_score": 0.016,
      "rerank_score": 0.5
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

## 云端模型接入顺序

## 云端完整复现

推荐 Ubuntu 22.04、Python 3.11、单张 24GB 以上 NVIDIA GPU、64GB RAM、至少 100GB 磁盘。首次运行需要提前安装系统 FFmpeg。

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[models,video]"
```

### 1. 准备视频片段

```bash
python scripts/prepare_videos.py \
  --input data/raw \
  --output artifacts/segments.jsonl \
  --frames-dir artifacts/frames \
  --language zh
```

该命令依次执行：

1. Whisper ASR，并保留每段转写的起止时间；
2. 每秒采样、场景变化检测、清晰度排序和时间去重；
3. Qwen2.5-VL 为关键帧生成客观描述；
4. 将 ASR、关键帧和描述对齐到 20 秒窗口（5 秒重叠）。

如果只想验证 ASR 和切片，可以增加 `--skip-captions`，以后再补视觉描述。

### 2. 构建两个 FAISS 索引

```bash
python scripts/build_indexes.py \
  --segments artifacts/segments.jsonl \
  --index-dir artifacts/indexes
```

文本向量和 CLIP 向量都会转换为归一化 `float32`，并使用 `IndexFlatIP`。索引目录同时保存模型名称、数据 SHA-256、片段数量和相似度定义。

### 3. 启动真实系统

48GB 显存：

```bash
python scripts/run_server.py --host 0.0.0.0 --port 5000
```

24GB 显存：

```bash
python scripts/run_server.py --host 0.0.0.0 --port 5000 --low-vram
```

`--low-vram` 会在候选精排结束后卸载 0.6B Reranker，再加载 Qwen2.5-VL。速度更慢，但降低多个模型同时驻留造成的显存峰值。

浏览器访问 `http://服务器地址:5000`。生产或公网使用时应放在反向代理和认证之后；当前开发服务建议绑定 `127.0.0.1` 并通过 SSH 隧道访问。点击任意证据，播放器会加载对应源视频并跳转到 `start_time`。

## 标注与评测

测试集使用 JSONL，每行格式：

```json
{
  "question_id": "q001",
  "question": "飞机在哪里降落？",
  "answer": "飞机在机场附近降落。",
  "relevant_segment_ids": ["video_001_0003"],
  "question_type": "audio"
}
```

检索消融：

```bash
python scripts/evaluate_retrieval.py \
  --questions data/questions.jsonl \
  --output artifacts/retrieval_metrics.json
```

输出单路、双路、三路RRF以及可选Reranker的消融结果：

- BM25-like 关键词召回；
- Qwen3-Embedding；
- OFA Chinese-CLIP；
- 三路 RRF。

指标包括 Recall@1/5/10、MRR 和 nDCG@1/5/10。

生成答案并计算自动指标：

```bash
python scripts/generate_predictions.py \
  --questions data/questions.jsonl \
  --output artifacts/answer_predictions.jsonl \
  --low-vram

python scripts/evaluate_answers.py \
  --questions data/questions.jsonl \
  --predictions artifacts/answer_predictions.jsonl
```

自动输出 Exact Match 和字符级 Token F1。开放式答案还应进行人工核验或加入有固定评分规范的 LLM Judge；不要将 LLM Judge 单独当作最终正确率。

## 模型与索引约束

默认模型：

- `openai/whisper-small`
- `Qwen/Qwen3-Embedding-0.6B`
- `OFA-Sys/chinese-clip-vit-base-patch16`
- `Qwen/Qwen3-Reranker-0.6B`
- `Qwen/Qwen2.5-VL-7B-Instruct`

所有向量必须转为 `float32` 并进行 L2 归一化。使用 `IndexFlatIP` 后，返回值可直接作为余弦相似度，避免旧代码将平方 L2 距离再次平方的问题。

配置集中在 `config.toml`，不再使用 `E:`、`F:` 等硬编码模型和数据路径。

## 项目验收标准

- 5～10 个短视频可以一键预处理和建库；
- API 返回真实视频时间段；
- 点击证据后前端跳转至 `start_time`；
- 至少 100 条标注问题；
- 输出 BM25、Embedding、CLIP、融合、+Reranker 的消融结果；
- 记录 Recall@5、MRR、nDCG@10、答案正确率、P50/P95 延迟和显存峰值。

当前本地测试不会下载或加载任何大模型。真实 GPU 验收需要在云端依次运行上述三个阶段，并保留 `manifest.json`、评测 JSON 和运行日志，作为简历量化结果的证据。

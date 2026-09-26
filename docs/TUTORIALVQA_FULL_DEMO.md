# TutorialVQA 全库问答 Demo

这个演示面向软件教程和操作视频：用户直接输入问题，系统在 76 个视频中找相关片段，再生成有时间证据的答案。页面默认搜索全部视频；也可以勾选一个或多个视频限定范围。点击证据会切换视频并跳到对应时间。

## 真实链路

```text
问题 → 全库 BM25 + Qwen3-Embedding 召回 → 轮询合并至最多 20 个候选
     → Qwen3-Reranker-0.6B 对 ASR/OCR/可用视觉描述文本精排至 Top 5
     → 时间邻居与关键帧预算 → Qwen2.5-VL-7B 回答
     → 引用 ID 校验 → 视频/时间播放跳转
```

界面文字问题还会启用 OCR BM25；画面问题还会启用 English CLIP。模型精排读取文本证据，不直接读取图片像素；Qwen-VL 在生成阶段读取关键帧。候选合并按各检索器名次轮询，不相加不同检索器的原始分数，不使用 RRF。全库提问不会预先提供正确视频 ID；可选视频范围在各检索器取 Top-K 之前生效。证据不足时返回拒答。

`config.tutorialvqa.full-demo.toml` 仅用于这个全库演示，保留原 `config.toml` 的新闻/中文运行方式和 `config.tutorialvqa.toml` 的旧评测口径。精排模型使用服务器已缓存的 Qwen3-Reranker，而非尚未验证的 BGE 模型。

## 数据准备和评测

TutorialVQA 原始数据来自 [官方仓库](https://github.com/acolas1/TutorialVQAData)，视频来自其 [Archive.org 备用镜像](https://archive.org/details/videos_202604)。遵守 CC BY-NC 4.0 研究用途条件；本仓库不提交原视频、模型权重、索引或官方问答 JSON。训练、开发、测试划分保持原样，标准答案和时间标签只用于推理后的评分，不进入索引和模型提示。

在学校服务器 `/data/zzhu126/VideoRAG` 中，先准备 Python 3.10+、ffmpeg、模型依赖、Whisper、PaddleOCR 和 Hugging Face 模型缓存，再执行。其他 GPU 主机可按 `pyproject.toml` 安装 `.[models,video,ocr]`，并把脚本中的路径与 Slurm 资源配置改成自己的环境：

```bash
source env.school.sh
python scripts/fetch_tutorialvqa.py --download-videos
sbatch scripts/prepare_tutorialvqa_scale.school.sbatch
```

预处理作业提取 76 个视频、Whisper-small 语音转写、5 秒间隔关键帧、英文 OCR、20 秒片段，并建立 Qwen3 文本与 English CLIP 的独立 FAISS 索引。所有大文件保存在 `data/tutorialvqa/` 和 `artifacts/tutorialvqa/scale-76/`，由 `.gitignore` 排除。

完成数据准备后提交真实模型评测：

```bash
sbatch scripts/evaluate_tutorialvqa_global.school.sbatch
```

作业先跑 10 条跨视频检索、3 条完整问答，随后评测官方 Test 的 1,239 条跨视频检索，并等间隔抽取 30 条问题做完整问答。报告与逐题结果在 `artifacts/tutorialvqa/scale-76/metrics-global/`。报告对比精排前后：视频 Recall@1/5、正确时间片段 Recall@1/5、候选池 Recall@20、检索与精排耗时；问答样本另报答案 F1、有效引用、证据命中与从进入 Pipeline 到返回答案的总延迟。模型已预热；该总延迟不包含首次模型加载、网页渲染和网络传输。样本量和评测口径必须连同指标一起引用。

## 启动和访问

```bash
sbatch scripts/run_tutorial_full_demo.school.sbatch
squeue -u "$USER"
tail -f logs/tutorial-demo-JOB_ID.out
```

服务绑定学校服务器的 `127.0.0.1:5000`。在自己电脑上建立 SSH 转发（按学校要求输入一次性 Token）：

```bash
ssh -L 5000:127.0.0.1:5000 zzhu126@foscsmlprd01.its.auckland.ac.nz
```

打开 <http://127.0.0.1:5000/>。API 也可复现：

```bash
curl -sS http://127.0.0.1:5000/api/videos
curl -sS -X POST http://127.0.0.1:5000/api/search -H 'Content-Type: application/json' -d '{"question":"How do I create a new action?","rerank":true}'
curl -sS -X POST http://127.0.0.1:5000/api/ask -H 'Content-Type: application/json' -d '{"question":"How do I create a new action?"}'
```

加入 `"video_ids":["4237"]` 可限定视频；不传该字段就是全库搜索。`/api/search` 不调用答案生成，`/api/ask` 返回答案、引用视频与时间、分阶段延迟和证据不足状态。视频文件通过 `/api/videos/<video_id>` 在认证后的 SSH 隧道内播放。

## 当前报告口径

既有 `artifacts/tutorialvqa/scale-76/metrics/test-report.json` 的 Recall@5=92.17% 是**已提供正确视频 ID 的视频内片段检索**。它不是全库视频召回，也未经过模型精排。全库和端到端结果应读取 `metrics-global/report.json`，仅在对应 GPU 作业成功完成后使用。

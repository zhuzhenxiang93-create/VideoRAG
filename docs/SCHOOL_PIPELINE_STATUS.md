> 最新更新：已找到并验证学校官方代理，数据及 OCR 依赖已准备；见 [网络与数据发现记录](SCHOOL_NETWORK_DISCOVERY.md)。下文网络阻塞结论为此前历史记录。

# 学校服务器启动与验收记录（2026-09-06）

项目 `/data/zzhu126/VideoRAG`，主机 `foscsmlprd01.its.auckland.ac.nz`。
基线 `e2954e0`，分支 `codex/modernize-videorag`。开始时只有未跟踪 logs/，没有运行作业。
所有修改、下载尝试、测试与模型运行均在服务器完成；没有推送仓库或部署到外部平台。

## 当前结论

**TutorialVQA 完整 Pipeline 尚未通过验收。** GitHub、GitHub API、codeload、archive.org 和 PyPI 均出现连接重置/TLS 错误；没有取得可核对的官方 JSON 或视频。未编造转换器字段、句子区间边界或教程准确率。

可用的学校代理或原始文件在服务器上的路径，是继续数据接入所需的外部条件。不能通过本地下载绕过。

## 已完成代码

- `/api/videos` 列出视频；`/api/ask` 和 `/api/search` 接收 `video_ids`。
- 指定范围时 BM25 在评分前限制候选；FAISS 根据范围重建对应向量集合，精确计算范围内余弦 Top-K。没有全库小 Top-K 后过滤，也不共享/修改请求范围状态。
- 空列表、未知视频返回 400；引用继续按本次允许证据验证；空答案拒答。生成 RuntimeError 返回 503。
- 保持 cascade，无新增 RRF。回退与主路轮流取候选，保留主路预算，不把不同通道原始分数相加。
- 增加英语 OCR/视觉/时序等规则；独立 `config.tutorialvqa.toml` 设置 OCR 英语，原中文配置不变。继续使用现有 Qwen 文本向量、Chinese-CLIP；英语检索质量尚未用教程评测。
- 图片按规范路径及视频/时间去重；预算内跨片段轮流选帧，附带 segment_id、video_id、秒数，再按时间排列。邻居扩展保留已命中锚点，时序问题按时间组织。
- 页面提供多选视频、提问、独立找片段、答案/拒答、OCR 文字、证据时间及播放器 seek。
- 启动脚本可通过环境变量选择数据、索引、配置和端口。
- ffmpeg 复用服务器已有 `/data/zzhu126/environments/vbench-tpami/bin/ffmpeg`，未修改该环境。

## 实际运行证据

- Slurm 16004：基线 Chinese-CLIP 预热成功，真实 Qwen2.5-VL 回答成功。
- Slurm 16006：修改后真实模型回归、范围检索和服务运行。
- Slurm 16007：加载最终代码的原新闻服务；原 16004/16006 均为本轮自行启动并结束的作业。
- `logs/tutorial-tests.log`：99 项测试通过，包括原有测试和 7 项新增关键回归。
- `artifacts/tutorialvqa/runtime-regression/report.json`：模型与输入验证结果。
- `answer-1.json` 至 `answer-5.json`：逐次模型响应，保留拒答。
- 3 条中文问题返回非空答案和引用；所有引用均属于所选 `012_mandarin_news_e7d178b6`，证据媒体 HTTP Range 返回 206。
- 无关问题（在火星上配置 Kubernetes）返回 `insufficient_evidence`、空引用。
- 英语 innovation 问题在此中文新闻上拒答；不能据此声称英语教程效果已通过。
- 实际查看该视频 4 秒画面，讲台显示“2023 世界旅游合作与发展大会”，与第一个答案一致；位置属于引用 0000 的 0–20 秒。
- Whisper 实际重转写了整个原新闻视频，输出 `whisper.json`（2 个带时间转写块）。这不是教程 ASR 验收。
- 页面只验证 HTTP/代码及 API；服务器无浏览器，**没有实际执行浏览器选择/播放 seek 验收**。

本轮上述输入均为已有新闻视频与原先的字幕/视觉/OCR 索引。新 Whisper 输出仅为路径验证，未覆盖已有数据。没有使用 TutorialVQA 官方字幕，也未声称教程端到端完成。

## 尚未完成

- 至少 5 个 TutorialVQA 视频、官方字幕字段和句子索引/秒数/结束边界核对、独立教程索引。
- 教程视频 Whisper 路径、清晰教程帧 OCR 实跑及进入证据链。
- 3 条教程真实问答与官方人工问题抽查。
- 浏览器实际选择与时间跳转。
- PaddleOCR/PaddlePaddle Python 包缺失。安装被 PyPI 连接重置阻塞；现有索引可检索已有 OCR，但本轮重新 OCR 未成功。中英文 OCR 均明确记录失败，未静默回退。

## 启动原数据服务（现在可用）

在服务器执行：

```bash
cd /data/zzhu126/VideoRAG
source env.school.sh
sbatch scripts/run_server.school.sbatch
squeue -u zzhu126
# 将 JOBID 替换为 sbatch 返回的编号
cat logs/videorag-JOBID.out
cat logs/videorag-JOBID.err
curl --fail http://127.0.0.1:5000/api/health
```

先检查已有任务和端口；不要重复启动同端口服务。只结束自己确认的本项目作业。

在个人电脑终端建立转发（文档命令，本轮未替用户执行）：

```bash
ssh -S /tmp/videorag-ssh -N -L 15000:127.0.0.1:5000 zzhu126@foscsmlprd01.its.auckland.ac.nz
```

浏览器访问 `http://127.0.0.1:15000`。学校当前作业节点与登录节点相同；若未来 Slurm 分配到其他节点，需使用该节点地址作为转发目标，并按学校网络策略设置监听地址。

## 网络恢复后的数据准备

使用学校允许的标准 HTTPS_PROXY 环境配置；不保存认证 Token。下载脚本保留原始 split 和 README/LICENSE，不生成可检索的 QA 内容：

```bash
cd /data/zzhu126/VideoRAG
source env.school.sh
python scripts/fetch_tutorialvqa.py --download-videos
python -m pip install --target .runtime-ocr 'paddleocr>=3.0,<4' 'paddlepaddle==3.2.2'
```

来源：https://github.com/acolas1/TutorialVQAData
官方视频镜像：https://archive.org/details/videos_202604
按 CC BY-NC 4.0 非商业研究用途保留归属；正式下载后须一并核对官方 README/LICENSE。仅处理 5–10 个视频；允许下载整个 videos.zip。原始文件放 `data/tutorialvqa/original/`，派生结果放 `artifacts/tutorialvqa/`，不得覆盖新闻数据。

**转换尚未实现，不能跳过以下步骤：**读取真实字段；确认答案是否句子索引及结束边界是否包含；仅用字幕文本/时间、帧和 OCR 构建片段；问题、答案和标准区间保留在独立评测文件；保留 train/dev/test 原始 split。官方字幕必须明确标记来源，不能标成 Whisper 结果。

转换和 OCR 完成、生成 `artifacts/tutorialvqa/segments.ocr.jsonl` 后，建索引必须提交 Slurm（沿用学校资源配置）：

```bash
sbatch --job-name=tutorial-index --partition=slurmpartition --gres=gpu:a100:1 \
  --cpus-per-task=8 --mem=64G --time=03:00:00 \
  --output=/data/zzhu126/VideoRAG/logs/tutorial-index-%j.out \
  --wrap='cd /data/zzhu126/VideoRAG && source env.school.sh && python scripts/build_indexes.py --segments artifacts/tutorialvqa/segments.ocr.jsonl --index-dir artifacts/tutorialvqa/indexes --config config.tutorialvqa.toml --device cuda'

sbatch --export=ALL,VIDEORAG_SEGMENTS=artifacts/tutorialvqa/segments.ocr.jsonl,VIDEORAG_INDEX_DIR=artifacts/tutorialvqa/indexes,VIDEORAG_CONFIG=config.tutorialvqa.toml \
  scripts/run_server.school.sbatch
```

以上教程命令依赖尚未准备好的数据，**本轮未执行建索引或教程启动**。

## 重跑验证

```bash
source env.school.sh
python -m pytest -q
# 真实模型回归后保持原新闻服务运行，需确保 5000 端口未被占用
sbatch scripts/verify_school_runtime.sbatch
```

`verify_school_ingestion.py` 可在已有本项目 GPU allocation 中用 `srun --jobid=JOBID --overlap --ntasks=1 python -u scripts/verify_school_ingestion.py` 单独重验输入。它明确记录失败，不覆盖新闻片段或索引。

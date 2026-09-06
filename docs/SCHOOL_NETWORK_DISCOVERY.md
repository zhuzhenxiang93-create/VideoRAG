# 学校网络与 TutorialVQA 文件查找结果（2026-09-06 更新）

本记录替代 `SCHOOL_PIPELINE_STATUS.md` 中“未找到网络入口/数据”的旧结论。此前结论不够充分：进一步查询学校公开手册后，找到了官方代理。

## 已验证的官方代理

来源：https://uoa-eresearch.github.io/vmhandbook/doc/linux-proxy.html
发布方：University of Auckland Centre for eResearch。

```bash
cd /data/zzhu126/VideoRAG
source env.school.sh
source env.school-network.sh
```

`env.school-network.sh` 仅为当前项目 shell 设置：

```bash
export http_proxy=http://squid.auckland.ac.nz:3128
export https_proxy=http://squid.auckland.ac.nz:3128
export no_proxy=localhost,127.0.0.1,localaddress,.auckland.ac.nz,169.254.169.254
```

服务器实测 GitHub API、PyPI、archive.org 均 HTTP 200。无账号密码、无 Token，未修改系统或用户全局网络配置。直连失败日志保留在 `logs/network-discovery.json`。

## 文件与来源

账户 `/data/zzhu126` 和 `/home/zzhu126` 下未找到此前已存在的 TutorialVQA 数据，也检查了此前迁移的两个 videorag 离线包。它们不含 TutorialVQA 或 PaddleOCR wheel。

通过上述学校代理，现已在服务器下载官方原始文件：

- `/data/zzhu126/VideoRAG/data/tutorialvqa/original/repository/train.json`
- `/data/zzhu126/VideoRAG/data/tutorialvqa/original/repository/dev.json`
- `/data/zzhu126/VideoRAG/data/tutorialvqa/original/repository/test.json`
- `/data/zzhu126/VideoRAG/data/tutorialvqa/original/repository/videos.json`
- 同目录保存官方 README.md、LICENSE.md。许可为 CC BY-NC 4.0，保留原始 split，仅限授权的非商业研究用途。

官方视频镜像：https://archive.org/details/videos_202604

`videos.zip` 大小 3,477,052,732 字节，后台下载日志 `logs/tutorial-fetch-proxy.log`。下载期间文件名为 `videos.zip.part`；仅完成后改为 `videos.zip`。不要将 .part 当成完整归档。

已从包中完整提取并验证 ZIP CRC 的 5 个视频，位于 `/data/zzhu126/VideoRAG/data/tutorialvqa/videos/`：

| ID / 文件 | 标题 | 字节数 |
|---|---|---:|
| 14643.webm | Export and save the design | 50,231,660 |
| 14644.webm | Add text and effects | 64,587,528 |
| 14645.webm | Get to know layers | 53,808,797 |
| 14646.webm | Include vector graphics | 62,481,166 |
| 14647.webm | Combine images using layer masks | 64,201,536 |

校验记录：`artifacts/tutorialvqa/selected-files.json`。完整包就绪后可运行 `python scripts/extract_tutorialvqa_subset.py` 重验；只处理此子集，不覆盖新闻数据。

## 字段核对与防止标签泄漏

`videos.json` 共 76 条视频。`transcript` 是句子字符串列表；时间戳只在 `segments` 人工操作区间中，并不是逐句字幕。视频包只有视频，无 SRT/VTT。

6195 条 QA 的 `answer_start/answer_end` 都精确匹配对应 `sentence_indexes.start/end`。这些是从 0 开始、含结束句的索引，不能当秒数。示例 14643 的 [4,6] 包含“Choose File, Save As”、改名、选择 Photoshop 并保存三句，对应人工区间 32.96–48.78 秒。

因此准备脚本不使用官方 transcript、操作标题、QA 或人工时间边界生成可检索内容；5 个视频均计划使用真实 Whisper（small，英语），再独立按 20 秒窗口、5 秒重叠分段，每 5 秒取帧。不能再将本轮描述为“使用官方带时间字幕”。

评测单独导出至 `artifacts/tutorialvqa/evaluation/{train,dev,test}.jsonl`，各 160/61/55 条；保留原 split 和原行号。`schema-audit.json` 保存完整 6195 条匹配检查。索引准备脚本不读取这些评测文件。

## OCR 与视频依赖

OCR 隔离目录：`/data/zzhu126/VideoRAG/.runtime-ocr`。
实际安装 PaddleOCR 3.7.0、PaddlePaddle 3.2.2；huggingface-hub 限制 <1，避免与现有 Transformers 不兼容。OCR 单独进程使用其 PYTHONPATH，主 Qwen 进程不加载这个依赖目录。

```bash
source env.school-network.sh
python -m pip install --target .runtime-ocr 'paddleocr==3.7.0' 'paddlepaddle==3.2.2'
python -m pip install --target .runtime-ocr --upgrade 'huggingface-hub>=0.34,<1'
python -m pip install --target .runtime-video --no-deps 'opencv-python-headless==4.10.0.84'
```

**实际通过**：英语 OCR 在 `14643` 的 40 秒帧上识别出 64 条文字。输出 `artifacts/tutorialvqa/ocr-prewarm.json`，日志 `logs/tutorial-ocr-prewarm.log`。这次独立 OCR 已运行，但还未进入待构建的教程检索索引。

浏览器在 `.runtime-browser` 中安装，仅用于服务器端验收；Chromium 已实际打开现有新闻页面。教程页面的完整问答/seek 验收尚未执行。

## GPU 任务状态及继续步骤

已提交 Slurm 作业 **16012**：`scripts/prepare_tutorialvqa.school.sbatch`，资源沿用学校 a100/8 CPU/64 GB 配置。记录时状态 `PENDING_APPROVAL`（管理员审批），未绕过审批或改动其他作业。

批准后脚本依次执行：
1. `prepare_tutorialvqa.py`：5 个视频 Whisper、取帧和独立切段；
2. `tutorial_ocr.py`：英语 OCR 写入时间证据；
3. `build_indexes.py`：独立 `artifacts/tutorialvqa/indexes`。

```bash
squeue -u zzhu126
# 不要重复提交尚在审批的 16012
cat logs/tutorial-prepare-16012.out
cat logs/tutorial-prepare-16012.err
```

作业完成后再按旧启动说明使用 `config.tutorialvqa.toml` 和教程索引启动服务、执行真实问答及浏览器验收。当前 **不能声称教程 Pipeline 已完成**。

依赖修复后原项目测试实跑 `99 passed`，见 `logs/tutorial-tests-network.log`。

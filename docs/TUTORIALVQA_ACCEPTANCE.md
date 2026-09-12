> 2026-09-12 更新：作业 16017 已实际执行，但三条教程问题均拒答，浏览器验收在答案断言处中止，完整验收未通过。详见 [实际结果](../reports/tutorialvqa/2026-09-07/README.md)。下文“待运行/等待审批”是提交时的历史状态。

# TutorialVQA 验收进度

项目：`/data/zzhu126/VideoRAG`；本文件记录最新状态。较早新闻验收见 SCHOOL_PIPELINE_STATUS.md，网络发现见 SCHOOL_NETWORK_DISCOVERY.md。

## 已实际完成

- 16012 已完成（13 分 23 秒、退出码 0）：5 个视频真实 Whisper、英语 PaddleOCR、文本向量与 Chinese-CLIP 索引。
- 视频：14643 / Export and save the design；14644 / Add text and effects；14645 / Get to know layers；14646 / Include vector graphics；14647 / Combine images using layer masks。
- 93 个片段、270 张唯一帧；44 个片段具有实际 OCR 文字。英语 OCR 结果位于 `artifacts/tutorialvqa/ocr-run.json`。
- 本轮没有使用官方字幕作为转写。官方 `videos.json` 的时间字段属于人工操作标注，不能用于创建可检索的输入区间。所有可检索转写来自 Whisper。
- 原始视频完整包已下载，保留官方 README/LICENSE 和 train/dev/test；校验结果见 `artifacts/tutorialvqa/archive-verified.json`。
- 新旧所有单元测试 101 passed，日志 `logs/tutorial-final-tests.log`。

## 发现并修复的具体问题

旧 Whisper Transformers 长音频拼接输出包含 0–287 秒等超长转写区间，导致整段文字重复进入多个 20 秒片段。虽然作业成功退出，但不能据此声称时间证据可靠。

代码新增 `WhisperTranscriber.transcribe_bounded`：ffmpeg 解码后独立识别每个 20 秒音频块，将局部时间加上实际块偏移，并限制在音频块内。两个关键测试覆盖时间偏移、尾块裁剪和无效配置。

旧结果保留于 `artifacts/tutorialvqa/segments.ocr.jsonl` 和 `indexes/`，仅作回退和对照，不作为最终教程服务输入。修复后的输出位于 `artifacts/tutorialvqa/bounded-v2/`。已运行的 OCR/关键帧直接复用，不重复识别或借用人工答案边界。

## 待运行的完整验证

已提交作业 **16017**（`scripts/run_tutorial_pipeline.school.sbatch`）。记录时等待学校审批。为避免运行旧时间索引，取消了本轮自行提交、尚未启动的服务作业 16016；未中断其他项目的作业。

批准后依次执行：

1. `repair_tutorial_timestamps.py`：全部 5 视频的有界 Whisper 转写。
2. `build_indexes.py`：独立 bounded-v2 索引及 manifest。
3. `run_server.py`：教程 API 服务，127.0.0.1:5000。
4. `verify_tutorial_api.py`：官方 dev:1168 / dev:501 / dev:1131 三条问题；非空答案、视频范围、媒体 Range 读取、与标准区间的重叠记录；无关问题拒答；5 个视频各自范围检索；OCR 证据进入搜索结果。
5. `verify_tutorial_browser.py`：服务器 Chromium 实际选择视频、提交问题、展示答案、点击证据、验证 currentTime 与媒体解码状态，再切换视频执行搜索。
6. 保持服务运行至作业时限，不推送、不部署外部平台。

输出：`artifacts/tutorialvqa/acceptance/` 中的 API 响应、api-report.json、browser-report.json、browser-answer.png。这些输出存在且内容验证通过之前，不能声称教程问答或浏览器验收已完成。人工抽查至少一个真实答案及视频画面仍需在模型输出后执行。

## 继续及访问

```bash
cd /data/zzhu126/VideoRAG
squeue -j 16017
cat logs/videorag-16017.out
cat logs/videorag-16017.err
```

16017 仍在等待或运行时不要重复提交。完成修复以后，后续仅启动服务：

```bash
sbatch --export=ALL,VIDEORAG_SEGMENTS=artifacts/tutorialvqa/bounded-v2/segments.ocr.jsonl,VIDEORAG_INDEX_DIR=artifacts/tutorialvqa/bounded-v2/indexes,VIDEORAG_CONFIG=config.tutorialvqa.toml scripts/run_server.school.sbatch
```

个人电脑终端转发：

```bash
ssh -S /tmp/videorag-ssh -N -L 15000:127.0.0.1:5000 zzhu126@foscsmlprd01.its.auckland.ac.nz
```

浏览器访问 http://127.0.0.1:15000 。此命令只建立转发，服务仍须实际启动成功。

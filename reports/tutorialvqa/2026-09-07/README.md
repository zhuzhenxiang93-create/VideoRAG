# TutorialVQA 实际运行结果（2026-09-07）

代码基线：`e773590`；Slurm 作业：16017。2026-09-12 从服务器原始输出整理，未重新运行 GPU 模型。

**完整 Pipeline 验收未通过。** 三条官方 dev 问题全部拒答，不能将 HTTP 200 或进程启动成功视为问答成功。

## 已实际完成

- 5 个视频读取、Whisper small 英语转写、OCR、关键帧和独立向量索引。
- 修复后的 ASR 共 340 个片段，最长 9 秒；没有使用官方字幕或人工答案区间作为检索输入。
- 每个视频的范围检索均通过；OCR 文字进入搜索证据。
- 明显无关的问题返回证据不足。
- 2026-09-12 重新运行单元测试：101 passed。

## 尚未通过

- dev:1168、dev:501、dev:1131 均为 abstained=true、空引用，没有得到三条非空教程答案。
- 浏览器已打开页面并提交请求，但因答案拒答，在 `assert not answer["abstained"] and answer["answer"]` 处中止；未完成媒体 seek 检查，也没有 browser-report.json。
- 尚未完成人工核对正确教程答案和对应视频位置。
- 服务随后达到 Slurm 三小时时限而停止；报告不是当前在线服务承诺。

`api-report.json` 中 scoped/media_range_206=false 是由于没有返回证据，**不表示已观察到跨视频泄漏**。独立 scope_search 字段才是范围检索检查结果。

## 文件

- api-report.json：三条问题与范围/拒答/OCR 检查。
- dev-*.json、unanswerable.json：原始 API 响应。
- input-manifest.json：五个视频的 ASR、帧、OCR 统计。
- archive-verified.json：原始视频包与官方 MD5 校验。
- schema-audit.json：官方字段和 split 核对结果。

只保存小型结果记录；原始视频、字幕全集、模型、索引、Python 依赖和完整运行日志仍保留在服务器。

## 数据归属

TutorialVQA，Colas et al., LREC 2020。数据许可 CC BY-NC 4.0，以上包含少量评测问题及标准时间区间，须保留来源并遵循其非商业使用条件。

- 官方数据：https://github.com/acolas1/TutorialVQAData
- 官方许可：https://github.com/acolas1/TutorialVQAData/blob/master/LICENSE.md
- 论文：https://aclanthology.org/2020.lrec-1.670/
- 视频镜像：https://archive.org/details/videos_202604

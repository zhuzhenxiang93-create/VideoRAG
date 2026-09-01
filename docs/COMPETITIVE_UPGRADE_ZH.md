# VideoRAG 竞品调研与远程实测记录

## 可借鉴项目

| 项目 | 值得借鉴的设计 | 本项目的落地方式 |
|---|---|---|
| [HKUDS/VideoRAG](https://github.com/HKUDS/VideoRAG) | 图结构文本知识、多模态双通道、分层上下文编码、跨视频理解 | 保留片段级时间证据；图结构只作为长视频/跨视频阶段的后续能力，不在当前 24 个短视频上强行引入 |
| [AdaVideoRAG](https://github.com/xzc-zju/AdaVideoRAG) | 根据问题复杂度和意图动态分配检索路径 | `AdaptiveFusionPolicy` 识别视觉意图，事实题跳过无收益的向量与视觉分支 |
| [MomentSearch](https://github.com/traversaal-ai/momentsearch) | 视觉/文本双分支、RRF、跨模态同一时刻合并与增强 | 加权 RRF 和 `agreement_bonus` 奖励多路命中的同一证据片段 |
| [MARQUIS](https://github.com/debashishc/marquis) | 查询分解、加权融合、视频重排、带来源和时间戳的结构化证据包 | 已落地加权融合与证据时间戳；查询分解只计划用于多跳问题，避免给简单事实题增加延迟 |
| [Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding) | 文本、图片、视频帧序列统一向量空间，并提供配套多模态 Reranker | 提供可选 Qwen3-VL 检索/精排适配器及独立索引 manifest |

## 远程环境

- GPU：NVIDIA GeForce RTX 4090 D，24GB。
- 数据：24 个中文开放许可新闻视频，20 条自动生成诊断问题。
- 片段：20 秒窗口、5 秒重叠。
- 指标文件：`artifacts/evaluation/retrieval_routed_final.json`。

## 检索消融结果

| 方法 | Recall@1 | Recall@5 | nDCG@5 | MRR | 平均召回延迟 |
|---|---:|---:|---:|---:|---:|
| BM25（`b=0`） | 0.7000 | 0.9500 | 0.8474 | 0.8196 | 0.56ms |
| Qwen3 文本向量 | 0.5000 | 0.9000 | 0.7267 | 0.6775 | 43.11ms |
| Chinese-CLIP | 0.3500 | 0.4500 | 0.4122 | 0.4286 | 7.71ms |
| 三路等权 RRF | 0.6500 | 0.9500 | 0.8243 | 0.7821 | 51.38ms |
| 查询自适应路由 + 加权 RRF | **0.7000** | **0.9500** | **0.8508** | **0.8196** | **1.34ms** |

相对三路等权 RRF，自适应方案保持 Recall@5，Recall@1 提升 0.05，nDCG@5 提升 0.0266，平均召回延迟减少约 97.4%（约 38 倍）。

文本 Qwen3-Reranker 的实测结果为 Recall@1 0.55、nDCG@5 0.8109、平均检索加重排延迟 338.77ms，属于明确负收益，因此不再作为默认在线路径。

## Qwen3-VL 跨模态检索实验

在同一服务器上用 `Qwen/Qwen3-VL-Embedding-2B` 对片段文本和有序关键帧重新建库。Qwen3-VL 单路检索的总体 Recall@1 为 0.55、Recall@5 为 0.85、nDCG@5 为 0.6677、MRR 为 0.6447；只看 2 条视觉题时，Recall@1 和 MRR 均为 1.0。相比之下，Chinese-CLIP 单路检索的总体 Recall@1 为 0.35，视觉题 Recall@1 为 0.5。

Qwen3-VL 对视觉题有明显的探索性优势，但如果让它参与所有问题，三路自适应结果只有 Recall@1 0.60，平均召回延迟约 102ms。因此 `config.qwen3-vl.toml` 也采用按需路由：事实题只跑 BM25，明确视觉题才融合 Qwen3-VL。端到端复测后，该策略的总体 Recall@1 为 0.75、Recall@5 为 0.95、nDCG@5 为 0.8622、MRR 为 0.8446；平均在线召回延迟取决于视觉问题占比，在当前 10% 视觉题分布下为 9.51ms。

结果保存在 `artifacts/evaluation/retrieval_qwen3_vl_routed.json`。视觉题只有 2 条，不能据此宣称模型在正式测试集达到 100% 视觉召回。

## 解释边界

这些数字只能作为工程诊断结果，不能写成正式模型准确率：问题集只有 20 条，而且由模型自动生成；其中视觉题只有 2 条。重叠片段还存在相关性标注不完整，例如三个检索器都把包含相同场景的相邻片段排在首位，但候选标签没有把它列为相关证据。

正式简历结果应在人工复核、video-disjoint 的测试集上重新报告，并至少分别统计 audio、visual、OCR、multimodal 和 unanswerable。

## 下一阶段建议

1. 完成至少 100 条人工复核问题，优先补充视觉、OCR、多跳和拒答样本。
2. 在验证集上校准视觉意图路由词表和拒答阈值，测试集只运行一次。
3. 对 Qwen3-VL-Embedding 与 Chinese-CLIP 做同一问题集、同一候选规模的视觉检索对照。
4. 只有当视频规模扩展到数十小时且出现跨视频问题时，再增加实体—事件—时间图谱和查询分解。

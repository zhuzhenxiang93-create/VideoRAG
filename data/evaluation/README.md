# 中文视频问答评测

本目录使用人工核对的中文新闻问题评估“召回 → 融合 → 重排”链路。种子集包含
20 个问题，主要用于验证实验代码和发现系统缺陷；简历中的最终数字应在扩充至至少
100 个问题并完成第二人复核后更新。

## 数据格式

每行是一个 JSON 对象，主要字段如下：

- `question_id`: 唯一问题编号。
- `question`: 中文问题。
- `answer`: 标准答案。
- `answer_aliases`: 可接受的同义答案。
- `relevant_segment_ids`: 人工标注的相关视频片段。
- `question_type`: `audio`、`visual` 或 `multimodal`。
- `evidence_start` / `evidence_end`: 证据时间范围。

## 运行实验

```bash
source env.sh
python scripts/evaluate_retrieval.py \
  --questions data/evaluation/questions.zh.seed.jsonl \
  --with-reranker \
  --device cuda
```

结果写入：

- `artifacts/evaluation/retrieval_ablation.json`
- `artifacts/evaluation/retrieval_ablation.csv`

脚本比较 BM25、Qwen Embedding、CLIP、三种双路 RRF、三路 RRF 和
`三路 RRF + Qwen3-Reranker`，并报告 Recall@1/5/10、MRR、NDCG、平均延迟、
P95 延迟和按问题类型分组的指标。

## 当前种子集观察

- BM25 是新闻事实题上最强的单路基线。
- 英文 `openai/clip-vit-large-patch14` 对中文查询表现很差：Recall@1/5 为 5%/5%。
- `OFA-Sys/chinese-clip-vit-base-patch16` 将视觉单路 Recall@1/5 提升到
  35%/45%，并将三路 RRF Recall@1/5 从 60%/90% 提升到 65%/95%。
- Qwen3-Reranker 能改善 Recall@5，但当前未改善 Recall@1。

这些结论只适用于当前 20 题种子集。下一步应增加纯视觉、OCR、跨模态和无答案问题，
并继续测试动态路由、加权 RRF 或将中文查询翻译成英文后再执行图文检索。

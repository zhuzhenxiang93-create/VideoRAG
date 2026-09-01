# 可信多模态流水线

## OCR

`PaddleOCRExtractor`只处理已经选出的关键帧，保存文字、置信度、四边形坐标和视频时间戳。相同文字在3秒内重复出现时保留置信度最高的一次。`ocr_bm25`只索引`VideoSegment.ocr_text`，因此可以单独评估OCR贡献，不会把ASR命中误算为OCR召回。

安装与重新预处理：

```bash
python -m pip install -e ".[models,video,ocr]"
python scripts/prepare_videos.py \
  --input data/raw \
  --output artifacts/segments.semantic.jsonl
```

`--skip-ocr`可用于不安装PaddleOCR的兼容运行。旧JSONL没有`ocr_text`和`ocr_items`时仍可加载，但OCR检索结果为空。

已有片段可以运行`python scripts/enrich_ocr.py`复用全部物理关键帧增量补充OCR；脚本强制写入新文件，不覆盖源数据。切换到新片段文件后，应为该文件重新构建或刷新匹配的索引manifest。

## 多标签路由

路由器可以同时返回`text`、`visual`、`ocr`、`multimodal`和`temporal`。运行时仅调用权重大于0的检索器；OCR问题优先`ocr_bm25`，视觉问题优先视觉向量，显式跨模态问题取各模态权重的最大值。`temporal`标签不额外启动模型，而是扩大前后邻居范围。

API响应中的`route_labels`用于诊断实际走过的逻辑路径。

## 语义切片与上下文

`segmentation.strategy="semantic"`在最短和最长窗口约束内，选择最接近目标时长的ASR句末或场景边界。固定窗口仍可通过`strategy="fixed"`使用。

检索融合后先按同一视频的时间重叠比例去重，只保留排名更高的锚点；随后为锚点扩展相邻片段。普通问题使用`neighbor_hops`，时序问题使用`temporal_neighbor_hops`，最终上下文受`max_generation_segments`限制。

## 拒答与引用验证

`FusionOrderReranker`只表达排序，不再被当作可校准置信度。只有声明`supports_confidence=true`的真实相关性模型才会应用`minimum_rerank_score`。

生成器必须返回：

```json
{
  "answerable": true,
  "answer": "答案",
  "confidence": 0.83,
  "citations": ["video_0003"]
}
```

以下任一条件在开启拒答时都会返回“根据当前视频内容无法确定”：

- `answerable=false`；
- 引用不在本次生成上下文中；
- `require_citations=true`但没有引用；
- 生成置信度低于`minimum_generator_confidence`；
- 校准后的重排分数低于`minimum_rerank_score`。

API只返回验证通过的引用及其证据片段，邻居片段可以被引用，但未被答案采用的上下文不会冒充最终证据。

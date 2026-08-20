# VideoRAG 中文评测集标注规范

## 1. 真实性边界

- 自动生成、模型改写或从字幕模板抽取的问题一律标为 `generated_candidate`。
- 只有项目所有者实际播放证据时间段、核对问题与答案并点击“接受”后，才能导出为 `verified`。
- `verified` 必须指向追加写入的 `review_event_id`、`reviewer_id` 和带时区的 `reviewed_at`，不能只手改一个状态字段。
- 当前20题种子集只用于开发诊断，不进入最终 held-out test。

## 2. 题型定义

| 类型 | 判定标准 | 不能算作该类型的情况 |
|---|---|---|
| `audio` | 仅靠ASR/可听语音即可直接回答 | 答案只出现在字幕条但没有被说出 |
| `visual` | 必须观察人物、物体、动作、颜色、位置或场景 | 只检索视觉描述文本但没有核对原帧 |
| `ocr` | 答案来自画面中文字、标题、字幕条、标志或数字 | Qwen-VL概括中出现文字但未核对原图 |
| `multimodal` | 必须联合至少两种来源才能唯一回答，如语音实体+画面动作 | 任一单路证据已经足够 |
| `unknown_route` | 在线路由器无法可靠判断所需模态 | 不能用它代替“视频中无答案” |

`question_type`与`answerable`相互独立。无答案题使用`answerable=false`、空答案和空相关segment，并记录`unanswerable_reason`及人工实际检查的`checked_time_ranges`；在线路由的`unknown_route`不能代替无答案标签。

verified题必须保存结构化`modality_evidence`：audio需人工确认ASR原文；visual需帧时间戳和人工画面观察；OCR需帧时间戳和人工抄录文字；multimodal需至少两种人工确认来源，并确认任一单路不足以完整回答。

## 3. 证据与相关segment

- 证据区间使用秒，满足 `0 <= evidence_start < evidence_end <= 视频时长`。
- 标注者必须播放完整证据区间，并至少向前、向后各检查5秒，避免截断上下文。
- 相关segment必须与证据区间存在严格时间重叠。
- 因20秒窗口、5秒重叠，同一证据可对应多个segment；所有真实覆盖证据的相邻窗口都可以列为相关。
- 正式评测同时报告segment命中和时间证据命中，避免返回语义等价相邻窗口却被误判失败。
- 帧级视觉证据记录最佳帧时间戳，不能只给视频ID。

## 4. 答案与别名

- `answer`使用视频证据中最简洁、规范的中文答案。
- `answer_aliases`必须包含标准答案，可加入繁简体、人名常见译法、合理数字格式；不能加入语义更宽泛的答案来放宽评分。
- 不确定读音、专名或OCR字符时拒绝该候选或在备注中标记待二次复核，不可猜测。

## 5. 人工复核步骤

1. 播放证据区间及前后上下文。
2. 分别核对ASR、原始画面、视觉描述；OCR通道上线后再核对OCR文本。
3. 判断问题是否自然、是否只有一个明确答案、是否泄漏文件名或segment ID。
4. 修正题型、答案、别名、证据时间和全部相关segment。
5. 接受或拒绝；拒绝必须选择原因。
6. 接受事件追加写入审计日志，禁止覆盖历史事件。
7. 如后续发现错误，追加`reopen`并重新决策；导出只认日志顺序中的最新有效决策。

## 6. 拒绝原因

- `ambiguous_answer`：存在多个合理答案。
- `evidence_missing`：证据时间内不能直接支持答案。
- `wrong_type`：无法修正到明确定义的题型。
- `asr_or_ocr_uncertain`：关键文字识别无法人工确认。
- `external_knowledge_required`：必须依赖视频外知识。
- `duplicate_question`：与已接受题目语义重复。
- `unnatural_or_leaky`：问题不自然或泄漏内部ID/文件名。

## 7. Split冻结

- 达到至少100道 `verified` 后再冻结split。
- 按视频分组，单个视频只能属于 development、validation、test 中一个，防止重叠画面和字幕泄漏。
- 在视频分组约束下尽量保持题型分层，目标约70%/15%/15%。
- development用于错误分析，validation用于选择路由、融合、depth和拒答阈值；test冻结SHA后只做最终评估。
- 每次导出保存问题文件SHA、split manifest、代码commit、模型/index manifest和硬件信息。
- 同一`paraphrase_group_id`不得跨split；split按视频及近重复关联形成的连通组整体分配。

## 8. 二次QA

- 所有带`review_flags`的高风险题必须二审。
- 无flag题确定性抽样至少20%做盲二审。
- 二审使用`reopen`事件，不修改或删除初审日志；报告初审接受率、二审修改率和最终拒绝率。

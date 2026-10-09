# 阶段记录：Stage 10 公司简介快照 / 版本 / 语义时序（Refinement R1）

> 本文件由 `scripts/ch3_data/10_build_company_text_semantics.py` 生成，全部数字来自真实运行结果。

## 1. 与岗位层的关键区别

- 公司简介时序**以公司实体为单位**，不是以岗位ID为单位；
- 先在「公司实体 + 观测时间」上折叠岗位重复，避免岗位数量多的公司在企业语义时序中被重复加权；
- 新增**公司简介快照层**：同一公司同一时间存在多个非空简介时标记 AMBIGUOUS，禁止用多数表决生成正式版本；
- 一次企业简介变化最多记作一条事件，不因该公司发布多个岗位而重复计数。

## 2. 快照层状态

- 公司简介快照总数：165031；
- CONSISTENT：154704；
- AMBIGUOUS：0；
- MISSING：10327；
- 同时间多简介涉及公司数：0。

> 只有 CONSISTENT 快照允许进入正式公司简介版本历史；AMBIGUOUS / MISSING 一律排除，
且即使主候选支持率很高也不自动进入正式时序。

## 3. 规模

- 进入正式公司时序的公司实体数：3664；
- 公司简介版本总数：3899；
- 公司简介语义变化事件数：235；
- 有简介版本变化的公司数：16；
- 跨歧义快照的变化事件数：0；
- 未能映射到正式实体的观测行数：6324。

## 4. 相似度计算方式与阈值

- 深层语义：`BAAI/bge-small-zh-v1.5`（状态 NOT_RUN，维度 NOT_RUN）；
- TF-IDF 词表规模（词面基线）：7522；
- 向量缓存状态：（键 = 文本类型 + 语料 SHA256 + 模型配置）；
- Embedding 语义距离 = 1 − 余弦相似度；TF-IDF 余弦仅为词面基线。

| 指标 | 阈值 | 样本数 | 来源 |
| --- | --- | --- | --- |
| 公司简介语义距离P50 | — | 0 | 全体相邻版本语义距离的真实分位数统计 |
| 公司简介语义距离P75 | — | 0 | 全体相邻版本语义距离的真实分位数统计 |
| 公司简介语义距离P90 | — | 0 | 全体相邻版本语义距离的真实分位数统计 |
| 公司简介语义距离P95 | — | 0 | 全体相邻版本语义距离的真实分位数统计 |

## 5. 变化候选类型

| 公司简介变化候选类型 | 事件数 | 占比 |
| --- | --- | --- |
| 品牌宣传变化候选 | 170 | 72.34% |
| 业务范围扩展候选 | 65 | 27.66% |

> 企业级变化同样全部保留「候选」字样，需人工复核后方可作为结论。

## 6. 跨歧义快照事件（前 10 条）

| company_entity_id | 旧版本 | 新版本 | 变化时间 | 跨越歧义快照数量 |
| --- | --- | --- | --- | --- |

> A → B 之间若存在 AMBIGUOUS 快照，事件必须记录跨越歧义快照数量，
避免把变化时间（新版本首次有效观测时间）误读为精确发生时刻。

## 7. 重构前后对比

| 指标 | 旧值 | 新值 | 绝对差 | 相对差 | 是否预期变化 | 变化原因 |
| --- | --- | --- | --- | --- | --- | --- |
| 公司简介版本数 | 3899 | 3899 | 0.0 | 0.0 | 1 | 只允许 CONSISTENT 快照进入正式版本历史，同时间多简介不再串成时间版本 |
| 正式公司实体数 | 4377 | 3664 | -713.0 | -0.162897 | 1 | 跨地域不再自动拆分，MULTI_LOCATION_AMBIGUOUS 等映射不再进入正式公司时序 |
| 公司简介变化事件数 | 235 | 235 | 0.0 | 0.0 | 1 | 随版本层重算，并新增跨越歧义快照说明字段 |
| 有变化公司数 | 16 | 16 | 0.0 | 0.0 | 1 | 随版本层与事件层重算 |

## 8. 业务 / AI 关键词变化（Top 15）

| 新增业务关键词 | 出现事件数 |
| --- | --- |
| 社交 | 18 |
| 大数据 | 17 |
| 保险 | 9 |
| 营销 | 9 |
| 机器人 | 5 |
| 游戏 | 4 |
| 广告 | 1 |
| 直播 | 1 |
| 电商 | 1 |

- 新增 AI 关键词事件数：39；
- 新增大模型关键词事件数：0。

## 9. 句向量存储与复现

- 向量文件：`NOT_RUN`；索引表：`NOT_RUN`；
- 复现记录：`NOT_RUN`；
- 向量数组：{}；
- 高维向量不写入 `company_profile_history.parquet`。

## 10. 运行门禁

| 门禁项 | 状态 |
| --- | --- |
| COMPANY_PROFILE_SNAPSHOT_BUILD | PASS |
| COMPANY_PROFILE_SNAPSHOT_UNIQUE | PASS |
| COMPANY_SAME_TIME_CONFLICT_AUDIT | PASS |
| AMBIGUOUS_SNAPSHOT_EXCLUSION | PASS |
| COMPANY_VERSION_REBUILD | PASS |
| COMPANY_PROFILE_VERSION_BUILD | PASS |
| COMPANY_EVENT_REBUILD | PASS |
| COMPANY_TEXT_EVENT_BUILD | PASS |
| COMPANY_TEXT_AUDIT_EXPORT | PASS |

## 11. 产物清单

- 公司简介快照：`data\interim\company_profile_snapshots.parquet`（165031 行）；
- 公司简介版本：`data\processed\company_profile_history.parquet`（3899 行）；
- 公司简介语义事件：`data\processed\company_text_change_events.parquet`（235 行）；
- 审计表：`outputs\tables\ch3\13_company_text_semantic_audit.xlsx`；
- 总修复审计表：`outputs\tables\ch3\14_text_semantic_refinement_audit.xlsx`；
- 封版记录：`docs\records\13_text_semantic_refinement_record.md`。

> 公司简介语义时序属于**公司实体层增强分析**，其结果不与岗位层语义事件混用计数。

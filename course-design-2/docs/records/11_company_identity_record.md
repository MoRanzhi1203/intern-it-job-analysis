# 阶段记录：Stage 09 公司实体高置信度识别（Refinement R1）

> 本文件由 `scripts/ch3_data/09_resolve_company_entities.py` 生成。

## 1. 为什么必须先做公司实体识别

一个公司可能发布多个岗位并重复包含**同一份公司简介**；若按「实习岗位ID」直接统计公司简介变化，岗位多的公司会被重复加权，因此公司简介语义时序必须建立在**公司实体层**之上。

## 2. 映射方式（保守分级，本轮收紧）

| 映射方式 | 判定依据 | 是否进入正式公司时序 | 默认置信度 |
| --- | --- | --- | --- |
| EXACT_NORMALIZED_NAME | 规范化名称完全一致 | 是 | 1.00 |
| APPROVED_ALIAS | 人工确认别名（`config/company_aliases.yml`） | 是 | 0.95 |
| MULTI_LOCATION_AMBIGUOUS | 同名跨地域，所在地不足以自动拆分 | **否**（人工复核） | 0.70 |
| HIGH_CONFIDENCE_MATCH | 名称不同但公司简介指纹完全一致 | 否（人工确认前不合并） | 0.90 |
| UNRESOLVED | 公司名称为空 | 否 | 0.00 |

> **所在地不再作为自动拆分正式公司实体的充分条件**：北京/上海/深圳可能只是同一家企业不同办公地点，旧统计范围 `EXACT_NAME_LOCATION`（名称@所在地）只保留为历史诊断字段，不再作为正式进入条件。

## 3. 识别结果

- 原始公司名称数：4422；
- 公司实体总数（含未解析）：4421；
- 进入正式公司时序的实体数：4377（覆盖岗位 16558 个）；
- 同名跨地域行数：77（规范名称 33 个）；
- 需人工复核映射行数：88；
- 无法解析公司行数：0。

## 4. 映射方式分布

| 映射方式 | 名称数 | 实体数 | 岗位数 | 是否进入正式公司时序 |
| --- | --- | --- | --- | --- |
| APPROVED_ALIAS | 2 | 1 | 39 | 是 |
| EXACT_NORMALIZED_NAME | 4376 | 4376 | 16519 | 是 |
| HIGH_CONFIDENCE_MATCH | 11 | 11 | 84 | 否 |
| MULTI_LOCATION_AMBIGUOUS | 33 | 33 | 547 | 否 |

## 5. 正式时序排除原因

| 映射方式 | 排除原因 | 行数 | 名称数 | 岗位数 |
| --- | --- | --- | --- | --- |
| MULTI_LOCATION_AMBIGUOUS | 同名跨地域，所在地不足以作为自动拆分正式实体的充分条件，需人工确认 | 77 | 33 | 547 |
| HIGH_CONFIDENCE_MATCH | 不同名称但公司简介指纹一致，未经人工确认不得合并 | 11 | 11 | 84 |

## 6. 同名跨地域（前 15 条）

| 公司规范名称 | 所在地数量 | 所在地集合 | 岗位数 | 映射方式 |
| --- | --- | --- | --- | --- |
| 小米 | 2 | 北京市/北京、广东/深圳 | 173 | MULTI_LOCATION_AMBIGUOUS |
| 科大讯飞 | 2 | 北京市/北京、安徽/合肥 | 122 | MULTI_LOCATION_AMBIGUOUS |
| 腾讯 | 2 | 北京、深圳 | 26 | MULTI_LOCATION_AMBIGUOUS |
| 新东方 | 6 | 北京、四川/成都、广东/广州、成都、江苏/南京、重庆市 | 25 | MULTI_LOCATION_AMBIGUOUS |
| 科锐国际 | 3 | 北京市、江苏/南京、苏州 | 23 | MULTI_LOCATION_AMBIGUOUS |
| 联想 | 3 | 上海、北京、北京市 | 19 | MULTI_LOCATION_AMBIGUOUS |
| 老虎国际 | 2 | 北京市、北京市/北京 | 16 | MULTI_LOCATION_AMBIGUOUS |
| 亚信科技 | 3 | 北京、北京市/北京、浙江/杭州 | 15 | MULTI_LOCATION_AMBIGUOUS |
| 脉脉 | 2 | 北京、北京市/北京 | 12 | MULTI_LOCATION_AMBIGUOUS |
| 华晨宝马 | 2 | 北京市/北京、辽宁/沈阳 | 11 | MULTI_LOCATION_AMBIGUOUS |
| 奇绩创坛 | 2 | 北京市、北京市/北京 | 10 | MULTI_LOCATION_AMBIGUOUS |
| 西门子 | 3 | 北京、广东、辽宁/大连 | 10 | MULTI_LOCATION_AMBIGUOUS |
| 法雷奥 | 2 | 上海市、江苏/南京 | 10 | MULTI_LOCATION_AMBIGUOUS |
| Lionbridge | 2 | 北京、山东/济南 | 9 | MULTI_LOCATION_AMBIGUOUS |
| 龙湖集团 | 2 | 上海市、北京市 | 6 | MULTI_LOCATION_AMBIGUOUS |

## 7. 别名与人工复核

| 规范名称 | 别名列表 | 状态 | 确认依据 |
| --- | --- | --- | --- |
| 智驾新程 | 智驾大陆、智驾新程 | approved | 12_manual_review_queue.xlsx，公司名称冲突岗位经人工复核判定为同源改名 |
| 谷川集团 | 谷川集团、谷川联行 | pending | 公司名称冲突岗位，简介一致但主体关系未确认 |
| 岩山科技 | 岩山科技、美团 | pending | 公司名称冲突岗位，疑似招聘主体更换 |

- 一对多（同一实体由多个名称合并而来）记录数：1；
- 多对一（同一名称拆分为多个实体）记录数：0；
- 简介指纹共享候选名称数：12。

## 8. 运行门禁

| 门禁项 | 状态 |
| --- | --- |
| COMPANY_IDENTITY_BUILD | PASS |
| COMPANY_IDENTITY_AMBIGUITY_CHECK | PASS |
| MULTI_LOCATION_COMPANY_CHECK | PASS |
| FORMAL_COMPANY_IDENTITY_FILTER | PASS |
| COMPANY_IDENTITY_AUDIT_EXPORT | PASS |

## 9. 产物清单

- 公司实体映射：`data\processed\company_entity_map.parquet`（4495 行）；
- 审计表：`outputs\tables\ch3\12_company_identity_audit.xlsx`；
- 指标 JSON：`outputs\logs\metrics\stage_09.json`。

# course-design-2：互联网 IT 实习岗位薪资影响因素分析与预测

> 本 README 只描述**当前有效架构**；历史返工过程与旧方案已随本轮清理删除，仅存于 Git 历史
> （提交历史即旧实现的长期存档）。原方法论文档 `docs/methodology/data_preprocessing_pipeline.md`
> 当前未随仓库保留（见第 8 节）。

## 1. 项目结构

> 本节描述**磁盘上实际存在的结构**。本 README 其余章节提到的
> `tests/`、`notebooks/`、`docs/methodology`、`docs/records`、`docs/prompts`、
> `outputs/logs/` 当前**尚未建立**，属待补充项（见第 8 节「已知结构问题」）。

```text
course-design-2/
├─ config/                 配置唯一权威来源（结构化字段 / 技能词表 / 公司别名 / 停用词）
├─ data/
│  ├─ raw/                 Stage 00 MySQL 只读导出（永久不可修改）
│  ├─ interim/             Stage 01~06 中间层
│  ├─ processed/           Stage 04~11 正式产物
│  └─ features/            高维向量（npz 不入库，附 meta.json 复现记录）
├─ src/                    共享模块（一次性实现，禁止各阶段重复造轮子）
├─ scripts/                Stage 00~11 独立脚本 + Stage 12~17 专项脚本 + 流水线总执行器
│  └─ _rework_history/     已归档的返工 / 被取代脚本（说明见该目录 README，不参与现行流程）
├─ docs/
│  └─ paper/               论文终稿（.docx / .pdf，只读保护区）
└─ outputs/
   ├─ tables/              审计表（全项目统一编号，含编号冲突，见第 8 节）
   ├─ figures/             仅保留论文 docx 中出现的 26 张图（见第 5.1 节）
   │  ├─ (根)              fig_4_7 / 5_7 / 7_8 / 8_9 / 8_10 / 8_11
   │  ├─ eda/              05_tech_skill_top20
   │  ├─ supplementary/    图S09/S16/S44/S50/S55/S59/S60/S62/S63/S64/S66/S68/S69/S70/S71
   │  └─ time/             01 / 02 / 03
   ├─ models/salary_model/ 正式薪资模型产物（pipeline + manifest + schema）
   └─ results/             「补全任务」交付物（补充 CSV / 图件）
```

## 2. 正式执行命令

```bash
# 全流程（复用已存在 raw，不访问 MySQL）
python scripts/run_data_pipeline.py --reuse-existing-raw

# 指定阶段区间
python scripts/run_data_pipeline.py --from-stage 1 --to-stage 11 --reuse-existing-raw

# 强制重新从 MySQL 导出 raw（仅在确需刷新原始快照时使用）
python scripts/run_data_pipeline.py --force-export-raw

# 单阶段独立运行（每个脚本均可独立执行）
python scripts/01_localize_raw_columns.py

# 静态检查（tests/ 当前未建立，见第 8 节）
python -m compileall src scripts
```

## 3. 阶段路线（Stage 00~11，编号固定不再变更）

| Stage | 职责 | 实现脚本 |
| --- | --- | --- |
| 00 | 原始数据导出（MySQL 只读，raw 永久不可变） | `scripts/00_export_raw.py` |
| 01 | 源记录标准化与公司属性语义槽位异常修复 | `scripts/01_localize_raw_columns.py` |
| 02 | 岗位身份确认（intern_id 实体键 / URL 标准化 / 1:1 校验） | `scripts/02_audit_job_identity.py` |
| 03 | 搜索来源关系与观测快照（含去重前公司认证标签映射） | `scripts/03_build_job_observations.py` |
| 04 | 岗位版本时序（核心 / 完整页面双签名 + 变化事件） | `scripts/04_build_job_versions.py` |
| 05 | 最终岗位实体（最终核心版本优先） | `scripts/05_build_unique_jobs.py` |
| 06 | 岗位版本文本语料 | `scripts/06_prepare_text_corpus.py` |
| 07 | 技能与文本特征 | `scripts/07_extract_job_text_features.py` |
| 08 | 岗位描述语义时序（完整 / 去薪资双口径） | `scripts/08_build_job_text_semantics.py` |
| 09 | 公司实体识别 | `scripts/09_resolve_company_entities.py` |
| 10 | 公司简介快照 / 版本 / 语义时序 | `scripts/10_build_company_text_semantics.py` |
| 11 | 结构化字段清洗与薪资目标解析 | `scripts/11_clean_structured_fields.py` |

正式问题的专项审计表：`20_text_semantic_refinement_audit.xlsx`（文本语义封版）、
`22_company_attribute_semantic_anomaly_audit.xlsx`（源记录公司属性语义槽位异常）、
`25_preprocessing_refactor_cleanup_audit.xlsx`（预处理重构与冗余清理）。

## 3.1 独立脚本层（Stage 12~17，不进入 Stage 00~11 流水线）

| 脚本 | 职责 | 审计表 |
| --- | --- | --- |
| `scripts/12_build_modeling_dataset.py` | 建模宽表与泄漏审计 | `28_modeling_dataset_audit.xlsx` |
| `scripts/13_run_eda.py` | 正式 EDA 与统计检验 | `29_eda_statistical_analysis.xlsx` |
| `scripts/14_train_salary_model.py` | 模型对比与验证集选模 | `30_model_comparison.xlsx` |
| `scripts/15_ablation_robustness_shap.py` | 消融 / Company Group Split / TreeSHAP | `31_ablation_robustness_shap.xlsx` |
| `scripts/16_company_field_semantic_audit.py` | 公司字段语义只读取证（认证 vs 标签） | `32_company_field_semantic_audit.xlsx` |
| `scripts/17_final_interpretation_audit.py` | 最终解释审计与封版 | `33_final_interpretation_audit.xlsx` |

术语纪律：**公司认证**（公司认证标签，最佳雇主 / 行业认证）与**公司标签（福利标签）**
（公司标签列表，免费健身设施等）必须严格区分；技能 SHAP 方向以 `presence_direction`
（技能存在时的平均 SHAP 贡献方向）为准，`global_mean_SHAP_direction` 仅作对比，两者均非因果。

统计推断口径（封版）：公司认证 = 有限四类 Kruskal–Wallis + epsilon² + 成对 Mann–Whitney /
Cliff's delta / BH-FDR；公司标签（福利标签）= 描述性统计 + 单标签 present vs absent 二元比较
（两侧样本 ≥ 50，统一 BH-FDR）；**369 组整体 KW 已标记 `DEPRECATED_INFERENCE`（已废止推断）**，
不再进入正式因素比较与论文主结论（追溯见 29 号表 16_已废止推断、32/33 号审计表）。

实验状态：`EXPERIMENT_FREEZE = TRUE`（模型实验冻结）+ `ANALYSIS_FREEZE = TRUE`（EDA/统计推断口径冻结）。

## 4. 正式数据产物

| 层 | 路径 | 规模 |
| --- | --- | --- |
| raw | `data/raw/shixiseng_job_details.parquet` | 172,063 × 30（永久不可修改） |
| 工作副本 | `data/interim/shixiseng_job_details_cn.parquet` | 172,063 × 30 |
| 观测快照 / 分类关系 | `data/interim/job_observation_snapshots.parquet`、`data/processed/job_category_membership.parquet` | — |
| 版本时序 | `data/processed/job_version_history.parquet`、`job_change_events.parquet` | — |
| 最终岗位实体 | `data/processed/job_details_unique.parquet` | 17,144 行 |
| 文本特征 | `data/processed/job_text_features.parquet`、`job_text_change_events.parquet` | 17,144 行 |
| 技能关系（long-format） | `data/features/job_skill_membership.parquet` | 一岗位 × 一规范技能 |
| 公司层 | `data/processed/company_entity_map.parquet`、`company_profile_history.parquet`、`company_text_change_events.parquet` | — |
| 建模目标层 | `data/processed/job_salary_targets.parquet`、`job_structured_features.parquet` | 17,144 行 |
| 建模数据集（Stage 12） | `data/processed/job_analysis_dataset.parquet`（17,144 × 65，全量分析集）、`job_salary_model_dataset.parquet`（14,883 × 69，正式薪资建模集） | 一岗一行 |

核心锚点（回归必须保持）：raw 172,063 × 30；unique intern_id = 17,144；
normalized URL unique = 17,144；ID ↔ normalized URL 严格 1:1；最终岗位实体 = 17,144。

## 5. 正式审计

- `outputs/tables/`：每个正式阶段 / 正式问题一张最终审计表（编号 00~74；存在重复编号，见第 8 节）；
- 各阶段门禁 / 指标 JSON（`outputs/logs/gates|metrics/`）与流水线日志
  （`outputs/logs/data_pipeline_latest.log`）由流水线运行时生成，当前未随仓库保留（见第 8 节）；
- 专项审计：`22_company_attribute_semantic_anomaly_audit.xlsx`（源记录公司属性语义槽位异常，
  含 MySQL↔raw 跨源证据）、`25_preprocessing_refactor_cleanup_audit.xlsx`（预处理重构与冗余清理）、
  `26_job_skill_extraction_audit.xlsx`（Stage 07 技能需求提取）、
  `27_skill_eda_scope_audit.xlsx`（Stage 13 技能 EDA 双口径与分层榜单，含双口径稳健性）、
   `28_modeling_dataset_audit.xlsx`（Stage 12 建模数据集、Feature Manifest 与目标泄漏审计）、
   `29_eda_statistical_analysis.xlsx`（Stage 13 正式 EDA 与统计检验，15 张子表）、
   `30_model_comparison.xlsx`（Stage 14 薪资预测模型对比、验证集选模与一次性 test 结果）、
  `31_ablation_robustness_shap.xlsx`（Stage 15 特征组消融、公司 Group Split、极端值/目标稳健性、
   TreeSHAP 整体与技能排名，11 张子表）、
  `32_company_field_semantic_audit.xlsx`（Stage 16 公司字段语义核查：公司认证 369 组溯源与问题分类）、
  `33_final_interpretation_audit.xlsx`（Stage 17 最终解释审计：字段语义修正前后 + 技能 SHAP
  presence 口径 + 核心结果回归保护 + 门禁与测试）。
- 正式 EDA 图件：`outputs/figures/eda/` 现仅保留 `05_tech_skill_top20`（其余已按第 5.1 节留存口径删除）。
- 建模图件与模型产出：`figures/modeling/` 目录已按第 5.1 节留存口径**整体移除**；
  模型产出仍保留于 `outputs/models/salary_model/`（完整 Pipeline + Feature Manifest + 技能列 +
  类别编码 schema + 文本降维 + 参数）。

### 5.1 图件目录说明（`outputs/figures/`）

`outputs/figures/` 现**只保留论文 docx 中实际出现的 26 张图**（严格逐文件口径）：

| 位置 | 图件 |
| --- | --- |
| `figures/` 根 | `fig_4_7` / `fig_5_7` / `fig_7_8` / `fig_8_9` / `fig_8_10` / `fig_8_11` |
| `figures/eda/` | `05_tech_skill_top20` |
| `figures/supplementary/` | `图S09`/`S16`/`S44`/`S50`/`S55`/`S59`/`S60`/`S62`/`S63`/`S64`/`S66`/`S68`/`S69`/`S70`/`S71` |
| `figures/time/` | `01` / `02` / `03`（发布时间队列与技能时间结构） |
| `results/figures/` | `fig_7_actual_vs_predicted_salary` |

**留存口径**：以论文 `docs/paper/*.docx` 内嵌图片为准（docx 即 zip，图在 `word/media/`），
逐文件校验「SHA256 完全一致 或 32×32 灰度归一化互相关 NCC ≥ 0.95」；不满足者一律删除。
据此共删除 181 个 png/pdf（含全部 PDF 与 `*_display` 变体，及 `sci/`、`modeling/` 两个目录）；
论文图表实际来自 `figures/supplementary/`、`figures/` 根目录与 `figures/time/`。图件可由相应脚本重新生成。

## 6. Notebook

`notebooks/` 目录当前未建立（见第 8 节）。后续 EDA / 建模 / 论文出图 Notebook 按统一规范新增；
Stage 00~11 不再保留 Notebook 双实现。

## 7. 运行约定

- 数据工程（Stage 00~11）只用 `scripts/` + `src/` 的 Python 流水线，禁止在 Notebook 中做数据加工；
- 公司属性语义槽位异常只在 Stage 01 确定性修复一次，下游只验证不重复修复；
- 薪资主目标固定为「薪资中点」，面议岗位正式目标为缺失（不做永久填补）；
- 删除任何文件前必须做引用检查（代码 imports / README / docs / Notebook）；
- 定时机制与运行环境由本机 conda 环境提供，命令见第 2 节。

## 8. 已知结构问题（待后续处理）

以下问题本轮**只登记不改动**（均涉及被脚本引用的文件，改动需同步修改引用并回归验证）：

1. **审计表编号重复**：`outputs/tables/` 实际编号已延伸至 74，且存在同名编号冲突——
   两组 `33_*`（`33_final_interpretation_audit.xlsx` 与 `33_business_time_dimension_analysis.xlsx`）、
   两组 `34_*`（`34_stage25_factor_revision.xlsx` 与 `34_visual_evidence_registry.xlsx`），
   与第 5 节「全项目统一编号」的约定冲突；两组编号均被脚本引用，需统一规划后再调整。
   （原 `35_*` 冲突已随 `35_visual_evidence_native_layout_registry.xlsx` 删除而消解。）
2. **证据图集已整体移除**：`outputs/figures/evidence_native/`（Stage27.0A 单图）与
   `outputs/figures/evidence/`（Stage27.0 组合图）两个目录、其生成脚本（`36`/`40`/`43`/`44`）
   及布局登记表 `35_visual_evidence_native_layout_registry.xlsx` 已按要求删除；
   归档脚本（`41`/`42`）中的 `evidence_native/*` 图片清单引用随之失效（仅存档留痕）。
   论文 `.docx` 内已嵌入这些截图，正文不受影响。
3. **顶层 `results/` 与 `outputs/` 功能重叠**：`results/` 为「补全任务」交付物
   （含 `补全任务_完成报告.md` 自述映射到论文段落），未纳入统一结构约定。
4. **待补充目录**：`tests/`、`notebooks/`、`docs/methodology|records|prompts`、`outputs/logs/`
   当前不存在；指向这些路径的失效引用已在本轮清理，相关叙述保留为待补充目标。
5. **仓库根目录遗留**：`_s24/prot_before.json` 为历史遗留、零引用，已于本轮删除。
6. **返工脚本已归档**：18 个"零引用且已被取代 / 目标目录已不存在"的返工脚本
   （Stage23.1、Stage26.x 中途稿、Stage27.0 / 27.0A 论文装配链、Word 装配与一次性补丁等）
   已移至 `scripts/_rework_history/`，附索引说明；现行流程不依赖该目录。

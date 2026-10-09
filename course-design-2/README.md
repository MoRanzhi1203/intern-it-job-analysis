# course-design-2：互联网 IT 实习岗位薪资影响因素分析与预测

> 本 README 只描述**当前有效架构**，历史返工过程仅存于 Git 历史。原方法论文档
> `docs/methodology/data_preprocessing_pipeline.md` 未随仓库保留（见第 8 节）。

## 1. 项目结构

> 本节描述**磁盘上实际存在的结构**；`docs/methodology`、`docs/records`、`outputs/logs/`
> 中的阶段记录 / 方法论文档 / 门禁与指标 JSON 在流水线运行时生成。

```text
course-design-2/
├─ config/                 配置唯一权威来源（结构化字段 / 技能词表 / 公司别名 / 停用词）
├─ data/
│  ├─ raw/                 Stage 00 MySQL 只读导出（永久不可修改）
│  ├─ interim/             Stage 01~06 中间层（含 skill_annotation_sample.xlsx 人工标注表）
│  ├─ processed/           Stage 04~11 正式产物
│  └─ features/            高维向量（npz 不入库，附 meta.json 复现记录）
├─ src/                    共享模块（扁平结构，一次性实现，禁止各阶段重复造轮子）
├─ scripts/                全部脚本（按「角色 + 论文章节」分目录）
│  ├─ pipeline/            数据流水线 Stage 00~12 + 总执行器（14）
│  ├─ ch4_lifecycle/       第 4 章 岗位特征与招聘生命周期分析 + Stage26 递进链（9）
│  ├─ ch5_factors/         第 5 章 薪资影响因素分析（3）
│  ├─ ch6_skills/          第 6 章 技能特征与薪资关联分析（1）
│  ├─ ch7_model/           第 7 章 薪资预测模型训练（1）
│  ├─ ch8_robust/          第 8 章 稳健性 / 泛化 / 解释分析（1）
│  ├─ figures/             全部出图脚本（base 出图基座 + 按章 ch4~ch8）（16）
│  ├─ experiments/         补充实验 E1~E8（4）
│  └─ supplementary/       补全任务交付（1）
├─ docs/
│  ├─ paper/               论文终稿（.docx / .pdf，只读保护区）
│  ├─ methodology/         方法论文档目录（流水线运行时生成）
│  └─ records/             阶段记录目录（流水线运行时生成）
└─ outputs/
   ├─ tables/              审计表（编号 00~59 连续；按论文章节分目录 ch3~ch9；第9章仅汇总表）
   ├─ figures/             论文 docx 实际出现的图件，按论文章节分目录、以论文图号命名
   │  ├─ ch3/ ~ ch8/       fig_<章>_<序>_<slug>.png（如 ch4/fig_4_6_planned_demand_by_job_category.png）
   ├─ models/salary_model/ 正式薪资模型产物（pipeline + manifest + schema）
   ├─ logs/                门禁 / 指标 JSON 与流水线日志（运行时生成）
   ├─ registries/          图件登记元数据（*_registry.json，与图件本体分离）
   ├─ results/             实验数值产物（按实验分组：E1_E3_E4_E5 / E2_E7 / E3_E4_E5_bootstrap / E8）
   └─ deliverables/        「补全任务」交付物（报告 / 派生表 / 派生图）
```


## 2. 正式执行命令

```bash
# 全流程（复用已存在 raw，不访问 MySQL）
python scripts/pipeline/run_data_pipeline.py --reuse-existing-raw

# 指定阶段区间
python scripts/pipeline/run_data_pipeline.py --from-stage 1 --to-stage 11 --reuse-existing-raw

# 强制重新从 MySQL 导出 raw（仅在确需刷新原始快照时使用）
python scripts/pipeline/run_data_pipeline.py --force-export-raw

# 单阶段独立运行（每个脚本均可独立执行）
python scripts/pipeline/01_localize_raw_columns.py

# 静态检查（tests/ 当前未建立，见第 8 节）
python -m compileall src scripts
```

## 3. 阶段路线（Stage 00~11，编号固定不再变更）

| Stage | 职责 | 实现脚本 |
| --- | --- | --- |
| Stage 00 | 原始数据导出（MySQL 只读，raw 永久不可变） | `scripts/pipeline/00_export_raw.py` |
| Stage 01 | 源记录标准化与公司属性语义槽位异常修复 | `scripts/pipeline/01_localize_raw_columns.py` |
| Stage 02 | 岗位身份确认（intern_id 实体键 / URL 标准化 / 1:1 校验） | `scripts/pipeline/02_audit_job_identity.py` |
| Stage 03 | 搜索来源关系与观测快照（含去重前公司认证标签映射） | `scripts/pipeline/03_build_job_observations.py` |
| Stage 04 | 岗位版本时序（核心 / 完整页面双签名 + 变化事件） | `scripts/pipeline/04_build_job_versions.py` |
| Stage 05 | 最终岗位实体（最终核心版本优先） | `scripts/pipeline/05_build_unique_jobs.py` |
| Stage 06 | 岗位版本文本语料 | `scripts/pipeline/06_prepare_text_corpus.py` |
| Stage 07 | 技能与文本特征 | `scripts/pipeline/07_extract_job_text_features.py` |
| Stage 08 | 岗位描述语义时序（完整 / 去薪资两种统计范围） | `scripts/pipeline/08_build_job_text_semantics.py` |
| Stage 09 | 公司实体识别 | `scripts/pipeline/09_resolve_company_entities.py` |
| Stage 10 | 公司简介快照 / 版本 / 语义时序 | `scripts/pipeline/10_build_company_text_semantics.py` |
| Stage 11 | 结构化字段清洗与薪资目标解析 | `scripts/pipeline/11_clean_structured_fields.py` |

正式问题的专项审计表：`ch3/14_text_semantic_refinement_audit.xlsx`（文本语义封版）、
`ch3/16_company_attribute_semantic_anomaly_audit.xlsx`（源记录公司属性语义槽位异常）、
`ch3/17_preprocessing_refactor_cleanup_audit.xlsx`（预处理重构与冗余清理）。

## 3.1 独立脚本层（Stage 12~17，不进入 Stage 00~11 流水线）

| 脚本 | 职责 | 审计表 |
| --- | --- | --- |
| `scripts/pipeline/12_build_modeling_dataset.py` | 建模宽表与泄漏审计 | `ch3/20_modeling_dataset_audit.xlsx` |
| `scripts/ch4_lifecycle/01_run_eda.py` | 正式 EDA 与统计检验 | `ch4/21_eda_statistical_analysis.xlsx` |
| `scripts/ch6_skills/01_skill_eda_scope_audit.py` | 技能 EDA 两种统计范围与分层榜单 | `ch6/19_skill_eda_scope_audit.xlsx` |
| `scripts/ch7_model/01_train_salary_model.py` | 模型对比与验证集选模 | `ch7/22_model_comparison.xlsx` |
| `scripts/ch8_robust/01_ablation_robustness_shap.py` | 消融 / Company Group Split / TreeSHAP | `ch8/23_ablation_robustness_shap.xlsx` |
| `scripts/ch5_factors/01_company_field_semantic_audit.py` | 公司字段语义只读取证（认证 vs 标签） | `ch5/24_company_field_semantic_audit.xlsx` |
| `scripts/ch5_factors/02_final_interpretation_audit.py` | 最终解释审计与封版 | `ch5/25_final_interpretation_audit.xlsx` |

术语纪律：**公司认证**（公司认证标签，最佳雇主 / 行业认证）与**公司标签（福利标签）**
（公司标签列表，免费健身设施等）必须严格区分；技能 SHAP 方向以 `presence_direction`
（技能存在时的平均 SHAP 贡献方向）为准，`global_mean_SHAP_direction` 仅作对比，两者均非因果。

统计推断方法（封版）：公司认证 = 有限四类 Kruskal–Wallis + epsilon² + 成对 Mann–Whitney /
Cliff's delta / BH-FDR；公司标签（福利标签）= 描述性统计 + 单标签 present vs absent 二元比较
（两侧样本 ≥ 50，统一 BH-FDR）；**369 组整体 KW 已标记 `DEPRECATED_INFERENCE`（已废止推断）**，
不再进入正式因素比较与论文主结论（追溯见 29 号表 16_已废止推断、32/33 号审计表）。

实验状态：`EXPERIMENT_FREEZE = TRUE`（模型实验冻结）+ `ANALYSIS_FREEZE = TRUE`（EDA/统计推断方法冻结）。

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

- `outputs/tables/`：每个正式阶段 / 正式问题一张最终审计表（**编号 00–59 连续**，按论文章节归入 `ch3/`~`ch8/`）；
- 各阶段门禁 / 指标 JSON（`outputs/logs/gates|metrics/`）与流水线日志
  （`outputs/logs/data_pipeline_latest.log`）由流水线运行时生成（目录已建立）；
- 专项审计：`ch3/16_company_attribute_semantic_anomaly_audit.xlsx`（源记录公司属性语义槽位异常，
  含 MySQL↔raw 跨源证据）、`ch3/17_preprocessing_refactor_cleanup_audit.xlsx`（预处理重构与冗余清理）、
  `ch3/18_job_skill_extraction_audit.xlsx`（Stage 07 技能需求提取）、
  `ch6/19_skill_eda_scope_audit.xlsx`（Stage 13 技能 EDA 两种统计范围与分层榜单，含两种统计范围稳健性）、
   `ch3/20_modeling_dataset_audit.xlsx`（Stage 12 建模数据集、Feature Manifest 与目标泄漏审计）、
   `ch4/21_eda_statistical_analysis.xlsx`（Stage 13 正式 EDA 与统计检验，15 张子表）、
   `ch7/22_model_comparison.xlsx`（Stage 14 薪资预测模型对比、验证集选模与一次性 test 结果）、
  `ch8/23_ablation_robustness_shap.xlsx`（Stage 15 特征组消融、公司 Group Split、极端值/目标稳健性、
   TreeSHAP 整体与技能排名，11 张子表）、
  `ch5/24_company_field_semantic_audit.xlsx`（Stage 16 公司字段语义核查：公司认证 369 组溯源与问题分类）、
  `ch5/25_final_interpretation_audit.xlsx`（Stage 17 最终解释审计：字段语义修正前后 + 技能 SHAP
  presence 统计方式 + 核心结果回归保护 + 门禁与测试）。
- 正式 EDA 图件：`outputs/figures/eda/` 现仅保留 `fig_6_1_tech_skill_top20`（其余已按第 5.1 节留存标准删除）。
- 建模图件与模型产出：`figures/modeling/` 目录已按第 5.1 节留存标准**整体移除**；
  模型产出仍保留于 `outputs/models/salary_model/`（完整 Pipeline + Feature Manifest + 技能列 +
  类别编码 schema + 文本降维 + 参数）。

### 5.1 图件目录说明（`outputs/figures/`）

`outputs/figures/` 现**只保留论文 docx 中实际出现的 26 张图**（严格逐文件统计范围）：

| 目录 | 论文图号 | 文件（stem） |
| --- | --- | --- |
| `figures/ch3/` | 图 3.3 | `fig_3_3_acquisition_flow` |
| `figures/ch4/` | 图 4.1~4.10 | `fig_4_1_formal_sample_jd_processing_result`、`fig_4_2_salary_midpoint_distribution`、`fig_4_3_main_subcategory_salary_quartile`、`fig_4_4_province_distribution`、`fig_4_5_recruitment_duration_distribution`、`fig_4_6_planned_demand_by_job_category`、`fig_4_7_active_cycle_salary_quartile`、`fig_4_8_publish_cohort_count`、`fig_4_9_salary_by_publish_time`、`fig_4_10_skill_or_category_time_structure` |
| `figures/ch5/` | 图 5.4~5.7 | `fig_5_4_city_education_company_size_salary`、`fig_5_5_certification_salary_distribution`、`fig_5_6_high_effect_benefit_label_cluster`、`fig_5_7_median_regression_cluster_bootstrap` |
| `figures/ch6/` | 图 6.1 / 6.3 / 6.4 | `fig_6_1_tech_skill_top20`、`fig_6_3_category_skill_hit_heatmap`、`fig_6_4_skill_salary_control_dumbbell` |
| `figures/ch7/` | 图 7.8 / 图 7-x | `fig_7_8_error_by_salary_quartile`、`fig_7_actual_vs_predicted_salary` |
| `figures/ch8/` | 图 8.5/8.6/8.9/8.10/8.11 | `fig_8_5_lgbm_catboost_paired_difference`、`fig_8_6_company_split_mae_distribution`、`fig_8_9_shap_beeswarm`、`fig_8_10_skill_shap_contribution`、`fig_8_11_benefit_cluster_permutation` |

**图号来源**：以 docx 内嵌图片的 SHA-256（21/25 命中）与像素尺寸（其余 4 张）逐一匹配到 `图 X.Y` 题注；
第 3 章图与「核心代码」类截图不纳入本目录管理。

**留存标准**：以论文 `docs/paper/*.docx` 内嵌图片为准（docx 即 zip，图在 `word/media/`），
逐文件校验「SHA256 完全一致 或 32×32 灰度归一化互相关 NCC ≥ 0.95」；不满足者一律删除。
据此共删除 181 个 png/pdf（含全部 PDF 与 `*_display` 变体，及 `sci/`、`modeling/` 两个目录）；
论文图表实际来自 `figures/supplementary/`、`figures/` 根目录与 `figures/time/`。图件可由相应脚本重新生成。

## 6. Notebook

`notebooks/` 目录已建立（暂空，以 `.gitkeep` 占位）。后续 EDA / 建模 / 论文出图 Notebook 按统一规范新增；
Stage 00~11 不再保留 Notebook 双实现。

## 7. 运行约定

- 数据工程（Stage 00~11）只用 `scripts/` + `src/` 的 Python 流水线，禁止在 Notebook 中做数据加工；
- 公司属性语义槽位异常只在 Stage 01 确定性修复一次，下游只验证不重复修复；
- 薪资主目标固定为「薪资中点」，面议岗位正式目标为缺失（不做永久填补）；
- 删除任何文件前必须做引用检查（代码 imports / README / docs / Notebook）；
- 定时机制与运行环境由本机 conda 环境提供，命令见第 2 节。

## 8. 结构治理记录

以下登记项已处理，改动均同步更新脚本引用与 README：

1. **审计表编号冲突已消解**：`outputs/tables/` 编号统一为 00~75 且唯一。
   原 `33_business_time_dimension_analysis.xlsx` 与 `ch5/25_final_interpretation_audit.xlsx` 编号冲突，
   前者改为 `ch4/59_business_time_dimension_analysis.xlsx`（写入脚本 `09_business_time_dimension`、读取脚本
   `01_time_cohort_figures` 同步更新，收口到常量 `project_paths.TABLE_BUSINESS_TIME_DIMENSION`）；
   原 `34_visual_evidence_registry.xlsx`（见第 2 条）零现行引用已删除，`ch5/26_stage25_factor_revision.xlsx` 编号唯一化
   （收口到 `project_paths.TABLE_STAGE25_FACTOR_REVISION`）；原 `35_*` 冲突随
   `35_visual_evidence_native_layout_registry.xlsx` 删除而消解。
2. **证据图集已整体移除**：`outputs/figures/evidence_native/`（Stage27.0A 单图）、
   `outputs/figures/evidence/`（Stage27.0 组合图）目录、生成脚本（`36`/`40`/`43`/`44`）
   及布局登记表 `35_visual_evidence_native_layout_registry.xlsx` 已删除；归档脚本（`41`/`42`）
   中的 `evidence_native/*` 图片清单引用随之失效（仅存档留痕）。论文 `.docx` 已嵌入这些截图，正文不受影响。
3. **顶层 `results/` 已归并**：原顶层 `results/` 并入 `outputs/`，拆分为 `outputs/results/`
   （实验数值 CSV/JSON）与 `outputs/deliverables/`（「补全任务」报告 / 派生表 / 派生图）；
   生成脚本 `01_completion_supplement.py` 与交付文档内的路径引用同步更新。
4. **运行期目录已建立**：`docs/methodology`、`docs/records`、`outputs/logs/`（含 `gates`/`metrics`）
   已建立（`.gitkeep` 占位）并纳入 `project_paths.ensure_directories()`，内容由流水线运行时生成；
   `tests/`、`notebooks/`、`docs/prompts/` 三个纯占位目录已移除。
5. **仓库根目录遗留**：`_s24/prot_before.json` 为历史遗留、零引用，已于上一轮删除。
6. **返工脚本已删除**：18 个"零引用且已被取代 / 目标目录已不存在"的返工脚本
   （Stage23.1、Stage26.x 中途稿、Stage27.0 / 27.0A 论文装配链、Word 装配与一次性补丁等）已删除，
   仅存于 Git 历史。
7. **产物命名与编号统一**：`outputs/figures/` 25 张图统一为 `fig_<标签>_<ASCII slug>`（沿用
   角色子目录 `(根)/eda/supplementary/time`）；`outputs/tables/` 审计表重编为**连续 00–59**；
   `outputs/results/` 16 个数值产物按实验分为
   `E1_E3_E4_E5 / E2_E7 / E3_E4_E5_bootstrap / E8` 四组（`project_paths.RESULTS_*`）。
8. **26 系列递进链经评估不合并**：`02_temporal_tightening` → `08_figures` 共 7 个脚本、约 8,270 行，
   含 23 处有意的跨脚本动态装载（`07_figure_rebuild` 需接收 `figures/base` 的 `01`/`02` 模块对象、
   `05_figure_refresh` 需 `03_final_consolidation` 的两个独立模块实例），合并为单文件会显著降低
   可维护性且无法离线验证，故保留现结构。
9. **脚本层重分类与命名规范化**（本轮）：`scripts/` 由「纯论文章节目录」改为「角色 + 章节」结构——
   新增 `pipeline/`（Stage 00~12 数据流水线，自原 `ch3_data/` 迁入）、`figures/base|ch4..ch8/`
   （集中全部出图/重绘脚本）、`experiments/`（E1~E8）、`supplementary/`（补全交付）；
   分析脚本保留章节目录 `ch4_lifecycle/`~`ch8_robust/`。去掉文件名中 `redraw_` / `exp_` /
   `stage26_x` 等冗余中缀。`run_data_pipeline` 阶段表、脚本间动态装载、自路径留痕字符串与
   README 引用同步更新。
10. **各目录内重新编号**（本轮）：除 `pipeline/`（其 `00~12` 即权威 Stage 编号，保持不动）外，
   其余目录按目录内 `01..N` 连续编号——`ch4_lifecycle` 01~09、`ch5_factors` 01~03、
   `ch6_skills`/`ch7_model`/`ch8_robust` 各 01、`experiments` 01~04、`figures/base` 01~03、
   `figures/ch4`~`ch8` 各 01~0N、`supplementary` 01。跨脚本动态装载路径与 README 引用同步更新。

### 8.1 附带的路径收口

- **孤儿产物清理**：删除零引用产物 `outputs/figures/_stage23_1_registry.json`（仅已删除的历史脚本 `18d` 生成）
  与 `outputs/figures/supplementary/_stage26_5_map/`（仅已删除的历史脚本 `26i` 使用的地图缓存，可重新下载）。
- **文件归位**：`data/skill_annotation_sample.xlsx` 从 `data/` 根移到 `data/interim/`，
  生成脚本 `04_E6_skill_annotation_sample.py` 与相关交付文档同步更新。
- **文档门禁路径修正**：`run_data_pipeline.py` 的 `check_documentation()` 由读取不存在的
  `docs/prompts/README.md` 改为读取项目根 `README.md`，必要标记 `Jupyter` 校正为 README 实际包含的 `Notebook`。
- **硬编码路径收口**：出图 / 实验脚本中手写的 `outputs/results`、
  `data/processed/model_splits.parquet` 等改为经 `project_paths`（`OUTPUTS_RESULTS_DIR` 常量）取得。
- **历史审计表清理**：14 张仅被已删除的历史脚本引用的历史审计表已删除（曾归档于
  `outputs/tables/_legacy/`，现仅存于 Git 历史）；`data/processed` 中的 stage26 试验产物仍被
  现行 `scripts/ch4_lifecycle/02_temporal_tightening.py` 引用，保持原位。
- **元数据与产物分离**：出图脚本写出的 4 个 `*_registry.json` 由 `outputs/figures/` 迁至
  `outputs/registries/`（`project_paths.REGISTRIES_DIR`）；「补全任务」交付物由 `outputs/results/`
  迁至 `outputs/deliverables/`（`project_paths.DELIVERABLES_DIR`），使 `outputs/results/` 只保留实验数值。

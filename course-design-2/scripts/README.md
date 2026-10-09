# scripts 目录索引

本目录按**论文章节**分入子目录（Stage 00~12 与第 3~8 章），调用写 `python scripts/<组>/<name>.py`。

| 目录 | 论文对应 | 脚本 |
| --- | --- | --- |
| `ch3_data/` | 数据获取与预处理 | `run_data_pipeline.py`、`00`~`12` |
| `ch4_lifecycle/` | 第 4 章 岗位特征与招聘生命周期 | `13`、`26b`~`26j`、`33`、`47`、`49` |
| `ch5_factors/` | 第 5 章 薪资影响因素 | `16`、`17`、`25`、`46`、`50`~`53` |
| `ch6_skills/` | 第 6 章 技能特征与薪资关联 | `13b`、`54`、`55`、`65` |
| `ch7_model/` | 第 7 章 薪资预测模型 | `14`、`56`、`58` |
| `ch8_robust/` | 第 8 章 稳健性/泛化/解释 | `15`、`45`、`59`~`61` |
| `figures_base/` | 全章出图基座 | `18a`、`18b`、`18c` |
| `supp/` | 补全任务 | `48`、`62` |

> 定位约定已随之调整：脚本用 `Path(__file__).resolve().parents[2]` 定位项目根（较原扁平结构多一层）；跨脚本装载一律写「项目根 + 相对路径」，如 `scripts/figures_base/18a_redraw_eda_modeling_figures.py`。
> 编号说明：脚本编号仅用于排序，**权威分组以本表为准**；存在历史缺号（如缺 `57`），且 `13_run_eda` / `13b_skill_eda_scope_audit` 同属 Stage 13（`13b` 为技能 EDA 支线，避免重号）。

## 1. 流水线入口

| 脚本 | 说明 |
| --- | --- |
| `run_data_pipeline.py` | Stage 00→11 总执行器（`--reuse-existing-raw` / `--force-export-raw`；门禁与回归核验） |

## 2. Stage 00–11 数据流水线

数据工程的唯一正式入口，禁止在 Notebook 中做数据加工。

| 脚本 | 说明 |
| --- | --- |
| `00_export_raw.py` | Stage 00 原始 MySQL 只读导出（`data/raw` 永久不可变） |
| `01_localize_raw_columns.py` | Stage 01 源记录标准化 + 公司属性语义槽位异常修复（唯一正式预处理入口） |
| `02_audit_job_identity.py` | Stage 02 岗位身份一致性审计（实习岗位ID / 详情链接） |
| `03_build_job_observations.py` | Stage 03 岗位观测快照构建（搜索来源折叠 + 分类关系层） |
| `04_build_job_versions.py` | Stage 04 岗位版本时序（核心/完整双签名 + 变化事件） |
| `05_build_unique_jobs.py` | Stage 05 最终岗位实体构建（最终核心版本优先 + 原子替换） |
| `06_prepare_text_corpus.py` | Stage 06 岗位版本文本语料准备 |
| `07_extract_job_text_features.py` | Stage 07 技能与文本特征（→ `ch3/18_job_skill_extraction_audit.xlsx`） |
| `08_build_job_text_semantics.py` | Stage 08 岗位描述语义时序（完整 / 去薪资两种统计范围） |
| `09_resolve_company_entities.py` | Stage 09 公司实体识别（跨地域不自动拆分） |
| `10_build_company_text_semantics.py` | Stage 10 公司简介快照 / 版本 / 语义时序 |
| `11_clean_structured_fields.py` | Stage 11 结构化业务字段清洗与薪资目标解析 |

## 3. Stage 12–17 独立分析层

不进入 Stage 00–11 流水线，按需单独运行。

| 脚本 | 说明 | 主要审计表 |
| --- | --- | --- |
| `12_build_modeling_dataset.py` | 建模宽表与目标泄漏审计 | `ch3/20_modeling_dataset_audit.xlsx` |
| `13_run_eda.py` | 正式 EDA 与统计检验 | `ch4/21_eda_statistical_analysis.xlsx` |
| `13b_skill_eda_scope_audit.py` | 技能 EDA 两种统计范围 + 分层榜单 + 稳健性 | `ch6/19_skill_eda_scope_audit.xlsx` |
| `14_train_salary_model.py` | 模型对比与验证集选模 | `ch7/22_model_comparison.xlsx` |
| `15_ablation_robustness_shap.py` | 消融 / 公司 Group Split / TreeSHAP | `ch8/23_ablation_robustness_shap.xlsx` |
| `16_company_field_semantic_audit.py` | 公司字段语义只读取证（公司认证 vs 标签） | `ch5/24_company_field_semantic_audit.xlsx` |
| `17_final_interpretation_audit.py` | 最终解释审计与封版 | `ch5/25_final_interpretation_audit.xlsx` |

## 4. 论文图件（收口与重绘）

只读冻结产物、只重排版式，不重算统计与模型。

| 脚本 | 产出 |
| --- | --- |
| `18a_redraw_eda_modeling_figures.py` | `outputs/figures/` 根：EDA/建模正式图重绘 |
| `18b_supplementary_figures.py` | `outputs/figures/supplementary/`：图 S01–S16 补图 |
| `18c_stage22_acquisition_flow.py` | 图 S16 数据采集总体流程 |
| `49_redraw_time_cohort_figures.py` | `outputs/figures/time/`：02 / 03 业务时间维度图 |
| `50_redraw_city_education_company_salary.py` | 图 S59 城市/学历/公司规模薪资中点中位数 |
| `51_redraw_certification_salary_figure.py` | 图 S60 公司认证状态薪资分布与组间比较 |
| `52_redraw_benefit_overlap_figure.py` | 图 S62 高效应福利标签共现结构 |
| `53_redraw_median_regression_bootstrap.py` | 图 5-7 中位数回归系数 bootstrap 区间对照 |
| `54_redraw_category_skill_heatmap.py` | 图 S63 岗位细分类 × 技能命中率热力图 |
| `55_redraw_skill_control_dumbbell.py` | 图 S09 控制细分类前后的技能薪资差异 |
| `56_redraw_quartile_error_figure.py` | 图 7-8 薪资四分位分组误差 |
| `58_redraw_model_comparison_bootstrap.py` | 图 8-9 LightGBM/CatBoost 配对 MAE bootstrap |
| `59_redraw_company_split_boxplot.py` | 图 8-10 重复公司分组划分 MAE 分布 |
| `60_redraw_shap_beeswarm.py` | 图 S64 主模型 SHAP 蜂群图 |
| `61_redraw_benefit_cluster_permutation.py` | 图 8-11 福利标签簇置换 MAE 增量 |
| `65_redraw_figure_s44_heatmap.py` | 图 S44 热力图重绘（移除图内「主统计范围」字样） |

## 5. Stage25 / 26 / 27 专项阶段

论文修订配套的受控新增重算（只新增文件、不覆盖既有产物）。

| 脚本 | 说明 | 主要审计表 |
| --- | --- | --- |
| `25_stage25_factor_revision.py` | 多值类别二元检验 / 中位数基线 / 图 S17 | `ch5/26_stage25_factor_revision.xlsx` |
| `26b_stage26_1_temporal_tightening.py` | Stage26.1 招聘生命周期与时序设定收紧 | 44–48 |
| `26c_stage26_2_final_consolidation.py` | Stage26.2 方法修复与补充分析 | 49–54 |
| `26d_stage26_3_structure_finalize.py` | Stage26.3 结构精简配套必要重算 | 55–59 |
| `26e_stage26_3_figure_refresh.py` | Stage26.3 图件重制（图 S34–S49） | — |
| `26f_stage26_4_final_polish.py` | Stage26.4 数据/模型修正配套重算 | 60–64 |
| `26g_stage26_4_figure_rebuild.py` | Stage26.4 图件重构 | — |
| `26j_stage26_6_figures.py` | Stage26.6 图件去留审计 / 地域归属范围 / 重绘 | 70–74 |
| `33_stage26_7_time_and_audit.py` | Stage26.7 业务时间维度专题分析（+ `figures/time/01–03`） | `ch4/59_business_time_dimension_analysis.xlsx` |

## 6. 「补全任务」实验与交付

| 脚本 | 说明 |
| --- | --- |
| `45_exp_E1_E3_E4_E5_model.py` | E1 / E3 / E4 / E5：配对检验、分组误差、消融增量、标签簇置换 |
| `46_exp_E2_E7_company_group_and_median.py` | E2 公司分组重复划分 + E7 中位数回归公司聚集 bootstrap |
| `47_exp_E8_planned_demand_series.py` | E8 计划招聘需求时间序列 |
| `48_exp_E6_skill_annotation_sample.py` | E6 技能抽取人工核验抽样表（→ `data/interim/skill_annotation_sample.xlsx`） |
| `62_completion_supplement.py` | 补全任务交付物（→ `outputs/deliverables/`） |

## 调用约定

- 统一从项目根执行：`python scripts/<name>.py`；全量流水线：`python scripts/ch3_data/run_data_pipeline.py --reuse-existing-raw`；
- 所有脚本以 `Path(__file__).resolve().parents[1]` 定位项目根，**不要**把脚本移入子目录；
- 脚本间复用（`importlib` 动态加载）一律写「项目根 + 相对路径」，如 `str(PROJECT_ROOT / 'scripts' / 'ch4_lifecycle' / '26f_stage26_4_final_polish.py')`；
  重命名脚本时必须同步更新引用（含 `run_data_pipeline.py` 的 `STAGE_SCRIPTS` 与 `.md/.json` 归档记录）。

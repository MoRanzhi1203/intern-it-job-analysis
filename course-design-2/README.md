# course-design-2：互联网 IT 实习岗位薪资影响因素分析与预测

> 本 README 只描述**当前有效架构**；历史返工过程与旧方案已随本轮清理删除，仅存于 Git 历史
> （提交历史即旧实现的长期存档）。方法论文档见
> [`docs/methodology/data_preprocessing_pipeline.md`](docs/methodology/data_preprocessing_pipeline.md)。

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
├─ docs/
│  └─ paper/               论文终稿（.docx / .pdf，只读保护区）
└─ outputs/
   ├─ tables/              审计表（全项目统一编号，含编号冲突，见第 8 节）
   ├─ figures/
   │  ├─ sci/              SCI 投稿风图件
   │  ├─ eda/              Stage 13 正式 EDA 图件
   │  ├─ modeling/         Stage 14 建模图件
   │  ├─ supplementary/    补充图件（含 *_display 展示版，供 Word 嵌图）
   │  ├─ evidence/         证据截图（旧版，Stage27_0，脚本 36~39）
   │  ├─ evidence_native/  证据截图（现行版，Stage27_0a，脚本 40~44，论文构建引用）
   │  └─ time/             时间维度分析图件（脚本 33 / 35）
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

# 静态检查与测试
python -m compileall src scripts tests
python -m pytest tests -q
```

## 3. 阶段路线（Stage 00~11，编号固定不再变更）

| Stage | 职责 | 唯一权威记录 |
| --- | --- | --- |
| 00 | 原始数据导出（MySQL 只读，raw 永久不可变） | `docs/records/01_raw_export_record.md` |
| 01 | 源记录标准化与公司属性语义槽位异常修复 | `docs/records/02_column_localization_record.md` |
| 02 | 岗位身份确认（intern_id 实体键 / URL 标准化 / 1:1 校验） | `docs/records/03_identity_audit_record.md` |
| 03 | 搜索来源关系与观测快照（含去重前公司认证标签映射） | `docs/records/05_observation_snapshot_record.md` |
| 04 | 岗位版本时序（核心 / 完整页面双签名 + 变化事件） | `docs/records/06_job_version_record.md` |
| 05 | 最终岗位实体（最终核心版本优先） | `docs/records/07_final_entity_record.md` |
| 06 | 岗位版本文本语料 | `docs/records/09_text_preprocessing_record.md` |
| 07 | 技能与文本特征 | — （审计表 `16_job_text_preprocessing_audit.xlsx`） |
| 08 | 岗位描述语义时序（完整 / 去薪资双口径） | `docs/records/10_job_text_semantic_record.md` |
| 09 | 公司实体识别 | `docs/records/11_company_identity_record.md` |
| 10 | 公司简介快照 / 版本 / 语义时序 | `docs/records/12_company_text_semantic_record.md` |
| 11 | 结构化字段清洗与薪资目标解析 | `docs/records/14_salary_target_record.md` |

正式问题的专项记录：`13_text_semantic_refinement_record.md`（文本语义封版）、
`15_company_attribute_semantic_anomaly_record.md`（源记录公司属性语义槽位异常）、
`16_preprocessing_refactor_cleanup_record.md`（本轮预处理重构与冗余清理）。

## 3.1 独立脚本层（Stage 12~17，不进入 Stage 00~11 流水线）

| 脚本 | 职责 | 审计表 / 记录 |
| --- | --- | --- |
| `scripts/12_build_modeling_dataset.py` | 建模宽表与泄漏审计 | `28_modeling_dataset_audit.xlsx` / `19_..._record.md` |
| `scripts/13_run_eda.py` | 正式 EDA 与统计检验 | `29_eda_statistical_analysis.xlsx` / `20_..._record.md` |
| `scripts/14_train_salary_model.py` | 模型对比与验证集选模 | `30_model_comparison.xlsx` / `21_..._record.md` |
| `scripts/15_ablation_robustness_shap.py` | 消融 / Company Group Split / TreeSHAP | `31_ablation_robustness_shap.xlsx` / `22_..._record.md` |
| `scripts/16_company_field_semantic_audit.py` | 公司字段语义只读取证（认证 vs 标签） | `32_company_field_semantic_audit.xlsx` / `23_..._record.md` |
| `scripts/17_final_interpretation_audit.py` | 最终解释审计与封版 | `33_final_interpretation_audit.xlsx` / `24_..._record.md` |

术语纪律：**公司认证**（公司认证标签，最佳雇主 / 行业认证）与**公司标签（福利标签）**
（公司标签列表，免费健身设施等）必须严格区分；技能 SHAP 方向以 `presence_direction`
（技能存在时的平均 SHAP 贡献方向）为准，`global_mean_SHAP_direction` 仅作对比，两者均非因果。

统计推断口径（封版）：公司认证 = 有限四类 Kruskal–Wallis + epsilon² + 成对 Mann–Whitney /
Cliff's delta / BH-FDR；公司标签（福利标签）= 描述性统计 + 单标签 present vs absent 二元比较
（两侧样本 ≥ 50，统一 BH-FDR）；**369 组整体 KW 已标记 `DEPRECATED_INFERENCE`（已废止推断）**，
不再进入正式因素比较与论文主结论（追溯见 29 号表 16_已废止推断、32/33 号审计表与记录 20/23/24）。

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

- `outputs/tables/`：每个正式阶段 / 正式问题一张最终审计表（编号 00~25 全项目统一）；
- `outputs/logs/gates/`：各阶段门禁 JSON；`outputs/logs/metrics/`：各阶段指标 JSON；
- `outputs/logs/data_pipeline_latest.log`：最近一次流水线日志；
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
- 正式 EDA 图件：`outputs/figures/eda/`（10 张主图，PNG 600 dpi + PDF，图名规范 `01_...`~`10_...`）。
- 建模图件与模型产出：`outputs/figures/modeling/`（`01_..04_` 模型对比 / 预测 / 残差 / 分组误差，
  `05_..08_` 特征组消融 / Random vs Group Split / SHAP 蜂群图 / 技能 SHAP Top20）、
  `outputs/models/salary_model/`（完整 Pipeline + Feature Manifest + 技能列 + 类别编码 schema + 文本降维 + 参数）。

## 6. 正式 Notebook

| Notebook | 用途 |
| --- | --- |
| `notebooks/01_data_governance_sci_visualization.ipynb` | 数据治理与岗位版本重构可视化（SCI 投稿风，图01~图34） |
| `notebooks/02_formal_eda.ipynb` | Stage 13 正式 EDA：读取封版产物、展示关键表、校验/重绘论文图表（图件统一存 `outputs/figures/eda/`） |

后续 EDA / 建模 / 论文出图 Notebook 按同一规范新增；Stage 00~11 不再保留 Notebook 双实现。

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
   两组 `34_*`（`34_stage25_factor_revision.xlsx` 与 `34_visual_evidence_registry.xlsx`）、
   两组 `35_*`（`35_recruitment_time_field_audit.xlsx` 与
   `35_visual_evidence_native_layout_registry.xlsx`），与第 5 节「全项目统一编号」的约定冲突；
   两组编号均被脚本引用，需统一规划后再调整。
2. **两代证据图集并存**：`outputs/figures/evidence/`（旧，Stage27_0，脚本 36~39）已被
   `outputs/figures/evidence_native/`（现行，Stage27_0a，脚本 40~44）取代，旧脚本仍引用旧集。
3. **顶层 `results/` 与 `outputs/` 功能重叠**：`results/` 为「补全任务」交付物
   （含 `补全任务_完成报告.md` 自述映射到论文段落），未纳入统一结构约定。
4. **README 声明与实际不符**：`tests/`、`notebooks/`、`docs/methodology|records|prompts`、
   `outputs/logs/` 当前不存在，相关叙述保留为待补充目标。
5. **仓库根目录遗留**：`_s24/prot_before.json` 为历史遗留、零引用，已于本轮删除。

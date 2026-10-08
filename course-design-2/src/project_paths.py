# -*- coding: utf-8 -*-
"""项目路径模块：统一识别 course-design-2 根目录并提供全部标准路径。

所有脚本、Notebook 一律通过本模块取路径，禁止硬编码绝对路径，
也禁止在各脚本中重复实现 resolve_project_root()。
"""

from __future__ import annotations

import re

from pathlib import Path

# 项目根目录标志：同时存在 data 与 scripts 目录
ROOT_MARKER_DIRS = ('data', 'scripts')


def resolve_project_root(start: Path | None = None) -> Path:
    """从起始目录向上查找 course-design-2 项目根目录。

    兼容三种启动方式：
    1. 在项目根目录执行 `python scripts/xx.py`；
    2. 在 scripts/ 目录内执行 `python xx.py`；
    3. 在 notebooks/ 或任意子目录内以交互方式导入本模块。
    """
    base = (start or Path.cwd()).resolve()
    for candidate in [base, *base.parents]:
        if all((candidate / name).is_dir() for name in ROOT_MARKER_DIRS):
            return candidate
    raise FileNotFoundError(
        f'未找到 course-design-2 项目根目录（需同时包含 {ROOT_MARKER_DIRS}）：起始目录 {base}'
    )


PROJECT_ROOT = resolve_project_root()

# ---- 数据目录 ----
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
INTERIM_DIR = DATA_DIR / 'interim'
PROCESSED_DIR = DATA_DIR / 'processed'
FEATURES_DIR = DATA_DIR / 'features'

# ---- 配置目录 ----
CONFIG_DIR = PROJECT_ROOT / 'config'
SKILL_CONFIG_PATH = CONFIG_DIR / 'skills.yml'
STOPWORDS_PATH = CONFIG_DIR / 'stopwords.txt'
STRUCTURED_FIELDS_CONFIG_PATH = CONFIG_DIR / 'structured_fields.yml'

RAW_PARQUET = RAW_DIR / 'shixiseng_job_details.parquet'
INTERIM_CN_PARQUET = INTERIM_DIR / 'shixiseng_job_details_cn.parquet'
OBSERVATION_SNAPSHOT_PARQUET = INTERIM_DIR / 'job_observation_snapshots.parquet'
# 旧实现（Stage 03 build_unique_jobs）输出的实体表，迁移期作为差异比较基线一次性留存
LEGACY_UNIQUE_PARQUET = INTERIM_DIR / 'job_details_unique_legacy.parquet'
PROCESSED_UNIQUE_PARQUET = PROCESSED_DIR / 'job_details_unique.parquet'
PROCESSED_UNIQUE_PARQUET_V2 = PROCESSED_DIR / 'job_details_unique_v2.parquet'
PROCESSED_CATEGORY_MEMBERSHIP_PARQUET = PROCESSED_DIR / 'job_category_membership.parquet'
PROCESSED_VERSION_HISTORY_PARQUET = PROCESSED_DIR / 'job_version_history.parquet'
PROCESSED_CHANGE_EVENTS_PARQUET = PROCESSED_DIR / 'job_change_events.parquet'

# ---- 文本语义层（Stage 06~10） ----
TEXT_CORPUS_PARQUET = INTERIM_DIR / 'job_text_version_corpus.parquet'
JOB_TEXT_FEATURES_PARQUET = PROCESSED_DIR / 'job_text_features.parquet'
# ---- Stage 12 建模数据集（一岗一行宽表 + 薪资建模集） ----
JOB_ANALYSIS_DATASET_PARQUET = PROCESSED_DIR / 'job_analysis_dataset.parquet'
JOB_SALARY_MODEL_DATASET_PARQUET = PROCESSED_DIR / 'job_salary_model_dataset.parquet'
JOB_TEXT_EVENTS_PARQUET = PROCESSED_DIR / 'job_text_change_events.parquet'
# ---- 岗位 × 规范技能 long-format 关系表（Stage 07，一行 = 一个岗位 × 一个规范技能） ----
JOB_SKILL_MEMBERSHIP_PARQUET = FEATURES_DIR / 'job_skill_membership.parquet'
COMPANY_ENTITY_MAP_PARQUET = PROCESSED_DIR / 'company_entity_map.parquet'
# 公司简介快照层：company_entity_map ↓ 快照 ↓ history ↓ 事件（Refinement R1 新增）
COMPANY_PROFILE_SNAPSHOTS_PARQUET = INTERIM_DIR / 'company_profile_snapshots.parquet'
COMPANY_PROFILE_HISTORY_PARQUET = PROCESSED_DIR / 'company_profile_history.parquet'
COMPANY_TEXT_EVENTS_PARQUET = PROCESSED_DIR / 'company_text_change_events.parquet'

# ---- Stage 11 结构化业务字段与薪资目标 ----
SALARY_TARGETS_PARQUET = PROCESSED_DIR / 'job_salary_targets.parquet'
STRUCTURED_FEATURES_PARQUET = PROCESSED_DIR / 'job_structured_features.parquet'

# ---- 高维向量（不写入主业务宽表） ----
JOB_TEXT_EMBEDDINGS_NPZ = FEATURES_DIR / 'job_text_embeddings.npz'
JOB_TEXT_EMBEDDING_INDEX_PARQUET = FEATURES_DIR / 'job_text_embedding_index.parquet'
COMPANY_TEXT_EMBEDDINGS_NPZ = FEATURES_DIR / 'company_text_embeddings.npz'
COMPANY_TEXT_EMBEDDING_INDEX_PARQUET = FEATURES_DIR / 'company_text_embedding_index.parquet'

# ---- 输出目录 ----
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
TABLES_DIR = OUTPUTS_DIR / 'tables'
FIGURES_DIR = OUTPUTS_DIR / 'figures'
# ---- 实验脚本结果目录（E1–E8 等 CSV / JSON 数值产物） ----
OUTPUTS_RESULTS_DIR = OUTPUTS_DIR / 'results'
# ---- 图件登记元数据目录（出图脚本写出的 *_registry.json，与图件本体分离） ----
REGISTRIES_DIR = OUTPUTS_DIR / 'registries'
# ---- 「补全任务」等交付物目录（报告 / 派生表 / 派生图，与实验数值产物分离） ----
DELIVERABLES_DIR = OUTPUTS_DIR / 'deliverables'
# ---- Stage 13 正式 EDA 图目录（图名规范 01_... / 02_...） ----
EDA_FIGURES_DIR = FIGURES_DIR / 'eda'
# ---- Stage 14 建模图表目录 ----
MODELING_FIGURES_DIR = FIGURES_DIR / 'modeling'
MODELS_DIR = OUTPUTS_DIR / 'models'
LOGS_DIR = OUTPUTS_DIR / 'logs'
PIPELINE_LOG = LOGS_DIR / 'data_pipeline_latest.log'
GATES_DIR = LOGS_DIR / 'gates'
METRICS_DIR = LOGS_DIR / 'metrics'

# ---- 文档目录 ----
DOCS_DIR = PROJECT_ROOT / 'docs'
RECORDS_DIR = DOCS_DIR / 'records'
# ---- 方法论文档目录（论文「数据预处理」章节的唯一技术来源） ----
METHODOLOGY_DIR = DOCS_DIR / 'methodology'
METHODOLOGY_PREPROCESSING_DOC = METHODOLOGY_DIR / 'data_preprocessing_pipeline.md'

# ---- 归档目录（仅历史证据；已按用户要求移除，历史证据改由 git 历史承载） ----
ARCHIVE_DIR = PROJECT_ROOT / 'archive'
ARCHIVE_PRE_REFACTOR_DIR = ARCHIVE_DIR / 'pre_refactor'

# git 历史中的归档证据路径（用于归档目录移除后的可追溯性校验）
ARCHIVE_HISTORY_PROBES = [
    'course-design-2/archive/pre_refactor/notebooks/03_job_identity_audit.ipynb',
    'course-design-2/archive/pre_refactor/notebooks/04_job_dedup_and_consistency.ipynb',
    'course-design-2/archive/pre_refactor/records/04_unique_job_processing_record.md',
    'course-design-2/archive/pre_refactor/tables/10_唯一岗位质量汇总.xlsx',
]

# ---- 本阶段涉及的表文件名（全项目统一编号，禁止两套编号并存） ----
TABLE_RAW_SCHEMA_AUDIT = 'ch3/01_raw_schema_audit.xlsx'
TABLE_COLUMN_DICTIONARY = 'ch3/02_column_dictionary.csv'
TABLE_IDENTITY_AUDIT = 'ch3/03_identity_audit.xlsx'
TABLE_IDENTITY_CONFLICTS = 'ch3/04_identity_conflicts.xlsx'
TABLE_REGRESSION_CHECK = 'ch3/06_refactor_regression_check.csv'
# ---- 岗位重复观测时序重构（Stage 03~05 正式产物） ----
TABLE_OBSERVATION_AUDIT = 'ch3/07_observation_snapshot_audit.xlsx'
TABLE_VERSION_AUDIT = 'ch3/08_job_version_audit.xlsx'
TABLE_ENTITY_AUDIT = 'ch3/09_final_version_entity_audit.xlsx'
# ---- 文本语义层审计表（Stage 06~10） ----
TABLE_JOB_TEXT_PREPROCESSING = 'ch3/10_job_text_preprocessing_audit.xlsx'
TABLE_JOB_TEXT_SEMANTIC = 'ch3/11_job_text_semantic_audit.xlsx'
TABLE_COMPANY_IDENTITY = 'ch3/12_company_identity_audit.xlsx'
TABLE_COMPANY_TEXT_SEMANTIC = 'ch3/13_company_text_semantic_audit.xlsx'
# ---- 文本语义层 Refinement R1 总修复审计表 ----
TABLE_TEXT_SEMANTIC_REFINEMENT = 'ch3/14_text_semantic_refinement_audit.xlsx'
TABLE_STRUCTURED_FIELD_SALARY = 'ch3/15_structured_field_salary_audit.xlsx'
# ---- Stage 01：源记录公司属性语义槽位异常（22~24 三轮取证已合并为同一张表） ----
TABLE_COMPANY_ATTRIBUTE_ANOMALY = 'ch3/16_company_attribute_semantic_anomaly_audit.xlsx'
COMPANY_ATTRIBUTE_ANOMALY_TABLE_PATH = TABLES_DIR / TABLE_COMPANY_ATTRIBUTE_ANOMALY
# ---- 本轮预处理重构与冗余清理总审计表 ----
TABLE_PREPROCESSING_CLEANUP = 'ch3/17_preprocessing_refactor_cleanup_audit.xlsx'
# ---- Stage 07 技能提取审计表（技能需求分析专用，与 16 号文本预处理审计分开） ----
TABLE_JOB_SKILL_AUDIT = 'ch3/18_job_skill_extraction_audit.xlsx'
# ---- Stage 13 技能 EDA 口径审计表（双口径 + 分层榜单 + 稳健性） ----
TABLE_SKILL_EDA_SCOPE = 'ch6/19_skill_eda_scope_audit.xlsx'
# ---- Stage 12 建模数据集与泄漏审计表 ----
TABLE_MODELING_DATASET_AUDIT = 'ch3/20_modeling_dataset_audit.xlsx'
# ---- Stage 13 正式 EDA 与统计检验 ----
TABLE_EDA_STATISTICAL = 'ch4/21_eda_statistical_analysis.xlsx'
# ---- Stage 14 薪资预测模型对比与验证集选模 ----
TABLE_MODEL_COMPARISON = 'ch7/22_model_comparison.xlsx'
# ---- Stage 15 消融、稳健性与 SHAP ----
TABLE_ABLATION_SHAP = 'ch8/23_ablation_robustness_shap.xlsx'
# ---- Stage 16 公司字段语义核查（只读取证：公司认证 vs 公司标签） ----
TABLE_COMPANY_FIELD_SEMANTIC = 'ch5/24_company_field_semantic_audit.xlsx'
# ---- Stage 17 最终解释审计（字段语义修正 + 技能 SHAP presence 口径） ----
TABLE_FINAL_INTERPRETATION = 'ch5/25_final_interpretation_audit.xlsx'
# ---- Stage26.7 业务时间维度分析（原 33_* 与 33_final_interpretation_audit 编号冲突，统一改为 75） ----
TABLE_BUSINESS_TIME_DIMENSION = 'ch4/59_business_time_dimension_analysis.xlsx'
# ---- Stage25 因素修订（原与 34_visual_evidence_registry 编号冲突，后者已删除，本表编号唯一化保留） ----
TABLE_STAGE25_FACTOR_REVISION = 'ch5/26_stage25_factor_revision.xlsx'
MODEL_SPLITS_PARQUET = PROCESSED_DIR / 'model_splits.parquet'
MODEL_PREDICTIONS_PARQUET = PROCESSED_DIR / 'model_predictions.parquet'
SALARY_MODEL_DIR = MODELS_DIR / 'salary_model'

# ---- 阶段记录文件名 ----
RECORD_REFACTOR_BASELINE = '00_refactor_baseline.md'
RECORD_RAW_EXPORT = '01_raw_export_record.md'
RECORD_COLUMN_LOCALIZATION = '02_column_localization_record.md'
RECORD_IDENTITY_AUDIT = '03_identity_audit_record.md'
# ---- 岗位重复观测时序重构阶段记录 ----
RECORD_OBSERVATION_SNAPSHOT = '05_observation_snapshot_record.md'
RECORD_JOB_VERSION = '06_job_version_record.md'
RECORD_FINAL_ENTITY = '07_final_entity_record.md'
# ---- 文本语义层阶段记录（Stage 06~10） ----
RECORD_TEXT_PREPROCESSING = '09_text_preprocessing_record.md'
RECORD_JOB_TEXT_SEMANTIC = '10_job_text_semantic_record.md'
RECORD_COMPANY_IDENTITY = '11_company_identity_record.md'
RECORD_COMPANY_TEXT_SEMANTIC = '12_company_text_semantic_record.md'
# ---- 文本语义层 Refinement R1 封版记录 ----
RECORD_TEXT_SEMANTIC_REFINEMENT = '13_text_semantic_refinement_record.md'
RECORD_SALARY_TARGET = '14_salary_target_record.md'
# ---- 源记录公司属性语义槽位异常：取证记录（15~16 两轮取证已合并） ----
RECORD_COMPANY_ATTRIBUTE_ANOMALY = '15_company_attribute_semantic_anomaly_record.md'
# ---- 本轮预处理方案重构与项目冗余清理记录 ----
RECORD_PREPROCESSING_CLEANUP = '16_preprocessing_refactor_cleanup_record.md'
# ---- Stage 07 技能需求提取记录 ----
RECORD_JOB_SKILL_EXTRACTION = '17_job_skill_extraction_record.md'
# ---- Stage 06 薪资泄漏修补 + 技能 EDA 双口径封版记录 ----
RECORD_SKILL_EDA_SCOPE = '18_skill_eda_scope_and_leakage_fix_record.md'
# ---- Stage 12 建模数据集记录 ----
RECORD_MODELING_DATASET = '19_modeling_dataset_record.md'
# ---- Stage 13 正式 EDA 记录 ----
RECORD_FORMAL_EDA = '20_formal_eda_record.md'
# ---- Stage 14 薪资预测模型对比记录 ----
RECORD_SALARY_MODEL = '21_salary_model_comparison_record.md'
# ---- Stage 15 消融、稳健性与 SHAP 记录 ----
RECORD_ABLATION_SHAP = '22_ablation_robustness_shap_record.md'
# ---- Stage 16 公司字段语义核查记录 ----
RECORD_COMPANY_FIELD_SEMANTIC = '23_company_field_semantic_audit_record.md'
# ---- Stage 17 最终解释封版记录 ----
RECORD_FINAL_INTERPRETATION = '24_final_interpretation_freeze_record.md'

# ---- Refinement R1 修复前基线（首次覆盖正式结果前固化，禁止人工硬编码） ----
REFINEMENT_BASELINE_JSON = METRICS_DIR / 'refinement_baseline.json'


# ---- 实验数值产物分组（outputs/results/，按产出实验分目录） ----
RESULTS_E1_E3_E4_E5 = OUTPUTS_RESULTS_DIR / 'E1_E3_E4_E5'
RESULTS_E2_E7 = OUTPUTS_RESULTS_DIR / 'E2_E7'
RESULTS_E3_E4_E5_BOOTSTRAP = OUTPUTS_RESULTS_DIR / 'E3_E4_E5_bootstrap'
RESULTS_E8 = OUTPUTS_RESULTS_DIR / 'E8'


def figure_chapter_dir(stem: str, default: Path) -> Path:
    """论文图件按 ``fig_<章>_<序>_`` 前缀路由到 outputs/figures/ch<章>/。"""
    match = re.match(r'fig_(\d+)_\d+_', stem)
    return FIGURES_DIR / f'ch{match.group(1)}' if match else default


def ensure_directories() -> list[Path]:
    """创建流水线所需的全部目录，返回已创建的目录列表。"""
    directories = [
        RAW_DIR, INTERIM_DIR, PROCESSED_DIR, FEATURES_DIR,
        TABLES_DIR, FIGURES_DIR, EDA_FIGURES_DIR, MODELING_FIGURES_DIR,
        MODELS_DIR, SALARY_MODEL_DIR, LOGS_DIR, GATES_DIR, METRICS_DIR,
        OUTPUTS_RESULTS_DIR, RESULTS_E1_E3_E4_E5, RESULTS_E2_E7,
        RESULTS_E3_E4_E5_BOOTSTRAP, RESULTS_E8, REGISTRIES_DIR, DELIVERABLES_DIR,
        RECORDS_DIR, METHODOLOGY_DIR,
        *[FIGURES_DIR / f'ch{i}' for i in range(3, 9)],
        *[TABLES_DIR / f'ch{i}' for i in range(3, 9)],
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def relative_to_root(path: Path) -> str:
    """返回相对项目根目录的路径文本，用于日志与记录展示。"""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)

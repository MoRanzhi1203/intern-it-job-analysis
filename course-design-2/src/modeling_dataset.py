# -*- coding: utf-8 -*-
"""建模数据集构建层（Stage 12）：一岗一行宽表、A/B/C/D/E 特征分组、目标泄漏黑名单。

## 产物

| 产物 | 行数 | 用途 |
| --- | --- | --- |
| `data/processed/job_analysis_dataset.parquet` | 17,144（1 intern_id = 1 行） | EDA / 面议岗位描述 / 技能分析 / 岗位与公司结构分析 |
| `data/processed/job_salary_model_dataset.parquet` | 14,883 | 薪资建模（主目标 = 薪资中点） |

建模样本：非面议 + 解析成功 + 薪资中点有效 + 薪资异常标志为空；
`14,883 + 2,245（面议）+ 16（逻辑异常）= 17,144`。

## 特征分组（Feature Manifest）

- **A 岗位基础**：岗位大类/细分类集合与数量、学历要求与等级、每周到岗、实习月数、
  岗位标签数量、JD 字符数/词数/分段状态、岗位方向、版本信息；
- **B 地域**：工作城市（原文 + 规范值）、是否直辖市（城市等级/区域无可靠来源，不新增）；
- **C 公司**：公司名称/公司实体信息、所属行业、公司性质（缺失保持）、公司规模数值与等级、
  公司认证标签、公司标签数量、公司简介字符数；
- **D 技能**：聚合技能特征（技能总数 + 各技能组计数）+ `技能提取范围` 控制变量；
  高频技能 multi-hot 由 `data/features/job_skill_membership.parquet`（ALL_USABLE 口径）
  在建模阶段临时构造，**不写入 processed 宽表**；
- **E 文本语义**：model-safe BGE 向量以 `文本向量行号` 引用（512 维，索引见
  `data/features/job_text_embedding_index.parquet` 的 `job_text_safe`），
  降维（SVD/PCA 16/32/64）在 Stage 14 **只用训练集拟合**，不在本轮固化。

## 目标泄漏黑名单

X 中禁止出现任何薪资字段或薪资派生字段（`SALARY_DERIVED_BLACKLIST` + `SALARY_COLUMN_PATTERNS`
双重防护）；文本类特征只能来自 model-safe 版本。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import schema, text_utils


def is_multi_value(value) -> bool:
    """多值字段判定（parquet 读取的列表列可能是 list / ndarray / set）。"""
    return isinstance(value, (list, tuple, set, frozenset, np.ndarray))

# ---- 主目标与稳健性口径 ----
TARGET_FIELD = schema.SALARY_MID_FIELD
ROBUSTNESS_TARGET_FIELDS = [schema.SALARY_MIN_FIELD, schema.SALARY_MAX_FIELD,
                            schema.SALARY_SPAN_FIELD]

# ---- 目标泄漏黑名单（显式字段 + 命名模式双重防护） ----
SALARY_DERIVED_BLACKLIST = [
    schema.SALARY_RAW_FIELD,             # 薪资信息（原始文本）
    schema.SALARY_MIN_FIELD,             # 薪资下限
    schema.SALARY_MAX_FIELD,             # 薪资上限
    schema.SALARY_MID_FIELD,             # 薪资中点（主目标）
    schema.SALARY_SPAN_FIELD,            # 薪资区间宽度
    schema.SALARY_UNIT_FIELD,            # 薪资单位
    schema.SALARY_NEGOTIABLE_FIELD,      # 是否面议
    schema.SALARY_PARSE_STATUS_FIELD,    # 薪资解析状态
    schema.SALARY_ANOMALY_FIELD,         # 薪资异常标志
    '历史是否发生薪资变化', schema.SALARY_FIRST_MIN_FIELD, schema.SALARY_FIRST_MID_FIELD,
    schema.SALARY_MID_DRIFT_FIELD,
]
SALARY_COLUMN_PATTERNS = ['薪资', '薪酬', '工资', '面议', '待遇']

# ---- 特征组 ----
GROUP_JOB = 'A_岗位基础'
GROUP_REGION = 'B_地域'
GROUP_COMPANY = 'C_公司'
GROUP_SKILL = 'D_技能'
GROUP_TEXT = 'E_文本语义'
GROUP_TARGET = '目标'
GROUP_META = '键与元数据'

TEXT_VECTOR_ROW_FIELD = '文本向量行号'
TEXT_VECTOR_AVAILABLE_FIELD = '文本向量是否可用'
TEXT_EMPTY_FIELD = '文本是否为空'
HAS_SKILL_FIELD = '是否有技能'

# D 组聚合技能特征（全部复用 Stage 07 已有列，不重复新增）
SKILL_AGGREGATE_FIELDS = [
    schema.SKILL_COUNT_FIELD, '技术技能数', '工程工具数', '数据工具数', '业务能力数', '技术领域数',
    schema.AI_SKILL_COUNT_FIELD, schema.LLM_SKILL_COUNT_FIELD,
    schema.SKILL_LANGUAGE_COUNT_FIELD, schema.SKILL_DATABASE_COUNT_FIELD,
    schema.SKILL_SECURITY_COUNT_FIELD, schema.SKILL_OFFICE_COUNT_FIELD,
]
# 技能阈值候选（§12：只做候选，最终阈值在 Stage 14 validation 上选择）
SKILL_THRESHOLD_CANDIDATES = [50, 80, 100]
SKILL_THRESHOLD_RATIO = 0.005

# ---- 逐字段规格（Feature Manifest 的唯一来源） ----
# 值：特征组 / 数据类型 / 是否数值 / 是否类别 / 是否多值 / 是否参与模型 / 来源 Stage / 备注
COLUMN_SPEC = {
    schema.ID_FIELD: (GROUP_META, '文本', 0, 0, 0, 0, 'Stage 02', '主键（1 intern_id = 1 行）'),
    '岗位大类集合': (GROUP_JOB, '列表', 0, 0, 1, 0, 'Stage 05',
                '多对多来源关系（原始关系存于 job_category_membership.parquet），'
                '保留全部大类（建模阶段 multi-hot）'),
    '岗位大类数量': (GROUP_JOB, '数值', 1, 0, 0, 1, 'Stage 05', ''),
    '岗位细分类集合': (GROUP_JOB, '列表', 0, 0, 1, 0, 'Stage 05',
                 '高频类别 multi-hot + 低频 OTHER，禁止只取第一个'),
    '岗位细分类数量': (GROUP_JOB, '数值', 1, 0, 0, 1, 'Stage 05', ''),
    '学历要求': (GROUP_JOB, '文本', 0, 1, 0, 1, 'Stage 11', '原始文本，保留用于核对等级映射'),
    '学历等级': (GROUP_JOB, '数值', 1, 0, 0, 1, 'Stage 11', '有序等级 0~4'),
    '每周到岗要求': (GROUP_JOB, '文本', 0, 1, 0, 1, 'Stage 11', ''),
    '每周到岗天数': (GROUP_JOB, '数值', 1, 0, 0, 1, 'Stage 11', ''),
    '实习时长要求': (GROUP_JOB, '文本', 0, 1, 0, 1, 'Stage 11', ''),
    '实习月数': (GROUP_JOB, '数值', 1, 0, 0, 1, 'Stage 11', ''),
    '岗位标签列表': (GROUP_JOB, '列表', 0, 0, 1, 0, 'Stage 06', '建模阶段再编码'),
    '岗位标签数量': (GROUP_JOB, '数值', 1, 0, 0, 1, 'Stage 07', ''),
    schema.JD_CHAR_COUNT_FIELD: (GROUP_JOB, '数值', 1, 0, 0, 1, 'Stage 06', ''),
    schema.JD_TOKEN_COUNT_FIELD: (GROUP_JOB, '数值', 1, 0, 0, 1, 'Stage 06', ''),
    schema.JD_SPLIT_STATUS_FIELD: (GROUP_JOB, '文本', 0, 1, 0, 1, 'Stage 06',
                                   '完整 / 部分 / 未识别'),
    schema.JOB_DIRECTION_FIELD: (GROUP_JOB, '文本', 0, 1, 0, 1, 'Stage 07', ''),
    schema.JOB_DIRECTION_MATCH_FIELD: (GROUP_JOB, '数值', 1, 0, 0, 1, 'Stage 07', ''),
    '岗位标题_规范': (GROUP_JOB, '文本', 0, 1, 0, 0, 'Stage 07', '高基数：仅用于标题聚类/审计'),
    schema.ENTITY_CORE_VERSION_COUNT_FIELD: (GROUP_JOB, '数值', 1, 0, 0, 1, 'Stage 04', ''),
    '完整页面版本数': (GROUP_JOB, '数值', 1, 0, 0, 1, 'Stage 04', ''),
    '是否多版本岗位': (GROUP_JOB, '数值', 1, 0, 0, 1, 'Stage 04', ''),
    schema.ENTITY_FINAL_VERSION_FIELD: (GROUP_META, '数值', 1, 0, 0, 0, 'Stage 05',
                                        '最终核心版本号（技能/文本画像对应的版本，元数据）'),
    '工作城市': (GROUP_REGION, '文本', 0, 1, 0, 1, 'Stage 01', '原始城市文本'),
    '工作城市_规范': (GROUP_REGION, '文本', 0, 1, 0, 1, 'Stage 11', '标准化城市'),
    '是否直辖市': (GROUP_REGION, '数值', 1, 0, 0, 1, 'Stage 11', ''),
    '公司名称': (GROUP_COMPANY, '文本', 0, 1, 0, 0, 'Stage 01',
             '高基数标识：用于 Stage 15 公司 Group Split，不作主模型类别特征'),
    'company_entity_id': (GROUP_COMPANY, '文本', 0, 1, 0, 0, 'Stage 09', '公司实体标识（高基数）'),
    '映射置信度': (GROUP_COMPANY, '数值', 1, 0, 0, 1, 'Stage 09', ''),
    '是否需人工复核': (GROUP_COMPANY, '数值', 1, 0, 0, 1, 'Stage 09', ''),
    '是否跨地域': (GROUP_COMPANY, '数值', 1, 0, 0, 1, 'Stage 09', ''),
    '是否存在同名跨地域歧义': (GROUP_COMPANY, '数值', 1, 0, 0, 1, 'Stage 09', ''),
    '公司所在地': (GROUP_COMPANY, '文本', 0, 1, 0, 1, 'Stage 11', '公司侧所在地（与工作城市区分）'),
    '所属行业': (GROUP_COMPANY, '文本', 0, 1, 0, 1, 'Stage 01', ''),
    '公司性质': (GROUP_COMPANY, '文本', 0, 1, 0, 1, 'Stage 11',
             'Stage 01 无法恢复的性质保持缺失，建模编码为「未知」，不回写源字段'),
    '公司规模': (GROUP_COMPANY, '文本', 0, 1, 0, 1, 'Stage 11', '原始区间文本'),
    '公司规模下限': (GROUP_COMPANY, '数值', 1, 0, 0, 1, 'Stage 11', ''),
    '公司规模上限': (GROUP_COMPANY, '数值', 1, 0, 0, 1, 'Stage 11', ''),
    '公司规模中点': (GROUP_COMPANY, '数值', 1, 0, 0, 1, 'Stage 11', ''),
    '公司规模等级': (GROUP_COMPANY, '数值', 1, 0, 0, 1, 'Stage 11', ''),
    '公司规模是否已知': (GROUP_COMPANY, '数值', 1, 0, 0, 1, 'Stage 11', ''),
    schema.SCALE_SHIFT_FIELD: (GROUP_COMPANY, '数值', 1, 0, 0, 1, 'Stage 11',
                               '槽位异常残留标志（恒为 0）'),
    schema.COMPANY_TAG_LIST_FIELD: (GROUP_COMPANY, '列表', 0, 0, 1, 0, 'Stage 03',
                                    '公司认证标签（列表）'),
    '公司标签数量': (GROUP_COMPANY, '数值', 1, 0, 0, 1, 'Stage 06', ''),
    schema.COMPANY_PROFILE_CHAR_COUNT_FIELD: (GROUP_COMPANY, '数值', 1, 0, 0, 1, 'Stage 06', ''),
    schema.SKILL_SET_FIELD: (GROUP_SKILL, '列表', 0, 0, 1, 0, 'Stage 07',
                             'ALL_USABLE 口径规范技能集合（关系引用）'),
    schema.SKILL_GROUP_FIELD: (GROUP_SKILL, '列表', 0, 0, 1, 0, 'Stage 06', '技能组列表'),
    schema.SKILL_FAMILY_LIST_FIELD: (GROUP_SKILL, '列表', 0, 0, 1, 0, 'Stage 07',
                                     '技能一级类型列表'),
    **{column: (GROUP_SKILL, '数值', 1, 0, 0, 1, 'Stage 07', '')
       for column in SKILL_AGGREGATE_FIELDS},
    schema.SKILL_SCOPE_FIELD: (GROUP_SKILL, '文本', 0, 1, 0, 1, 'Stage 07',
                               '技能提取口径控制变量（REQUIREMENT_SECTION / FULL_TEXT_FALLBACK / EMPTY_TEXT）'),
    TEXT_EMPTY_FIELD: (GROUP_SKILL, '数值', 1, 0, 0, 1, 'Stage 12',
                       '无可用技能文本标志：与「有文本但 0 技能」严格区分'),
    HAS_SKILL_FIELD: (GROUP_SKILL, '数值', 1, 0, 0, 1, 'Stage 12', ''),
    TEXT_VECTOR_ROW_FIELD: (GROUP_TEXT, '数值', 1, 0, 0, 0, 'Stage 08',
                            'model-safe BGE 向量行号（512 维不入宽表，建模阶段 join）'),
    TEXT_VECTOR_AVAILABLE_FIELD: (GROUP_TEXT, '数值', 1, 0, 0, 1, 'Stage 08',
                                  'E 组可用性标志（缺失机制控制变量）'),
}
MANIFEST_COLUMNS = ['字段名', '特征组', '来源Stage', '来源文件', '数据类型', '是否数值', '是否类别',
                    '是否多值', '是否参与模型', '是否目标派生', '是否存在泄漏风险', '处理方式', '备注']

SOURCE_FILES = {
    'Stage 01': 'data/interim/shixiseng_job_details_cn.parquet',
    'Stage 02': 'data/processed/job_details_unique.parquet',
    'Stage 03': 'data/processed/job_category_membership.parquet',
    'Stage 04': 'data/processed/job_version_history.parquet',
    'Stage 05': 'data/processed/job_details_unique.parquet',
    'Stage 06': 'data/interim/job_text_version_corpus.parquet',
    'Stage 07': 'data/processed/job_text_features.parquet',
    'Stage 08': 'data/features/job_text_embedding_index.parquet',
    'Stage 09': 'data/processed/company_entity_map.parquet',
    'Stage 11': 'data/processed/job_structured_features.parquet',
    'Stage 12': 'Stage 12 派生控制变量',
}


def salary_derived_columns(columns) -> list:
    """按显式名单 + 命名模式识别目标派生字段。"""
    blocked = []
    for column in columns:
        if column in SALARY_DERIVED_BLACKLIST:
            blocked.append(column)
        elif any(pattern in str(column) for pattern in SALARY_COLUMN_PATTERNS):
            blocked.append(column)
    return sorted(set(blocked))


def valid_salary_mask(salary: pd.DataFrame) -> pd.Series:
    """正式薪资样本掩码（与 Stage 13 技能 EDA 完全一致的口径）。"""
    return (salary[schema.SALARY_NEGOTIABLE_FIELD].eq(0)
            & salary[schema.SALARY_PARSE_STATUS_FIELD].eq('已解析')
            & salary[schema.SALARY_MID_FIELD].notna()
            & salary[schema.SALARY_ANOMALY_FIELD].fillna('').astype(str).str.strip().eq(''))


def build_analysis_dataset(entity: pd.DataFrame, structured: pd.DataFrame,
                           text_features: pd.DataFrame, company_map: pd.DataFrame,
                           text_index: pd.DataFrame) -> pd.DataFrame:
    """构建全量分析集（一岗一行，含 A/B/C/D/E 紧凑特征）。"""
    job_id = schema.ID_FIELD
    base = entity[[job_id, '岗位大类集合', '岗位细分类集合', '公司名称', '所属行业',
                   schema.ENTITY_CORE_VERSION_COUNT_FIELD, '完整页面版本数',
                   '是否多版本岗位', schema.ENTITY_FINAL_VERSION_FIELD]].copy()
    base['岗位大类数量'] = base['岗位大类集合'].map(lambda value: len(text_utils.as_list(value)))
    base['岗位细分类数量'] = base['岗位细分类集合'].map(lambda value: len(text_utils.as_list(value)))

    structured_columns = [job_id, '学历要求', '学历等级', '每周到岗要求', '每周到岗天数',
                          '实习时长要求', '实习月数', '公司性质', '公司规模', '公司规模下限',
                          '公司规模上限', '公司规模中点', '公司规模等级', '公司规模是否已知',
                          schema.SCALE_SHIFT_FIELD, '公司所在地', '工作城市', '工作城市_规范',
                          '是否直辖市']
    text_columns = [job_id, '岗位标题_规范', schema.JOB_DIRECTION_FIELD,
                    schema.JOB_DIRECTION_MATCH_FIELD, schema.JD_CHAR_COUNT_FIELD,
                    schema.JD_TOKEN_COUNT_FIELD, schema.JD_SPLIT_STATUS_FIELD,
                    schema.JOB_TAG_LIST_FIELD, '岗位标签数量', schema.SKILL_SET_FIELD,
                    schema.SKILL_GROUP_FIELD, schema.SKILL_FAMILY_LIST_FIELD,
                    schema.SKILL_SCOPE_FIELD, schema.COMPANY_TAG_LIST_FIELD,
                    schema.COMPANY_PROFILE_CHAR_COUNT_FIELD, *SKILL_AGGREGATE_FIELDS]
    merged = (base
              .merge(structured[structured_columns], on=job_id, how='left')
              .merge(text_features[text_columns], on=job_id, how='left'))

    # C 组：公司实体信息（按 公司名称 + 公司所在地 关联唯一实体）
    company_info = (company_map[['company_name_original', '公司所在地', 'company_entity_id',
                                 '映射置信度', '是否需人工复核', '是否跨地域',
                                 '是否存在同名跨地域歧义']]
                    .rename(columns={'company_name_original': '公司名称'}))
    merged = merged.merge(company_info, on=['公司名称', '公司所在地'], how='left')
    merged['公司标签数量'] = merged[schema.COMPANY_TAG_LIST_FIELD].map(
        lambda value: len(text_utils.as_list(value)))

    # D 组控制变量：区分「无可用文本」与「有文本但 0 技能」
    merged[TEXT_EMPTY_FIELD] = merged[schema.SKILL_SCOPE_FIELD].eq('EMPTY_TEXT').astype(int)
    merged[HAS_SKILL_FIELD] = merged[schema.SKILL_COUNT_FIELD].gt(0).astype(int)

    # E 组：model-safe BGE 向量引用（不在宽表展开 512 维）
    vector = (text_index[text_index['text_type'].eq('job_text_safe')]
              [[job_id, '核心版本号', 'embedding_row']]
              .rename(columns={job_id: '_vector_job', '核心版本号': '_vector_version',
                               'embedding_row': TEXT_VECTOR_ROW_FIELD}))
    merged = (merged.merge(vector, left_on=[job_id, schema.ENTITY_FINAL_VERSION_FIELD],
                           right_on=['_vector_job', '_vector_version'], how='left')
              .drop(columns=['_vector_job', '_vector_version']))
    merged[TEXT_VECTOR_AVAILABLE_FIELD] = merged[TEXT_VECTOR_ROW_FIELD].notna().astype(int)

    ordered = [job_id, '岗位大类集合', '岗位大类数量', '岗位细分类集合', '岗位细分类数量',
               '学历要求', '学历等级', '每周到岗要求', '每周到岗天数', '实习时长要求', '实习月数',
               '岗位标签列表', '岗位标签数量', schema.JD_CHAR_COUNT_FIELD,
               schema.JD_TOKEN_COUNT_FIELD, schema.JD_SPLIT_STATUS_FIELD,
               schema.JOB_DIRECTION_FIELD, schema.JOB_DIRECTION_MATCH_FIELD,
               '岗位标题_规范', schema.ENTITY_CORE_VERSION_COUNT_FIELD, '完整页面版本数',
               '是否多版本岗位', schema.ENTITY_FINAL_VERSION_FIELD,
               '工作城市', '工作城市_规范', '是否直辖市',
               '公司名称', 'company_entity_id', '映射置信度', '是否需人工复核', '是否跨地域',
               '是否存在同名跨地域歧义', '公司所在地', '所属行业', '公司性质', '公司规模',
               '公司规模下限',
               '公司规模上限', '公司规模中点', '公司规模等级', '公司规模是否已知',
               schema.SCALE_SHIFT_FIELD, schema.COMPANY_TAG_LIST_FIELD, '公司标签数量',
               schema.COMPANY_PROFILE_CHAR_COUNT_FIELD,
               schema.SKILL_SET_FIELD, schema.SKILL_GROUP_FIELD, schema.SKILL_FAMILY_LIST_FIELD,
               *SKILL_AGGREGATE_FIELDS, schema.SKILL_SCOPE_FIELD, TEXT_EMPTY_FIELD,
               HAS_SKILL_FIELD, TEXT_VECTOR_ROW_FIELD, TEXT_VECTOR_AVAILABLE_FIELD]
    analysis = merged[ordered].copy()
    analysis[TEXT_VECTOR_ROW_FIELD] = analysis[TEXT_VECTOR_ROW_FIELD].astype('Int64')
    return analysis


def build_salary_model_dataset(analysis: pd.DataFrame, salary: pd.DataFrame) -> tuple:
    """构建薪资建模集（正式样本），并返回样本筛选审计表。"""
    valid = salary[valid_salary_mask(salary)]
    frame = analysis[analysis[schema.ID_FIELD].isin(set(valid[schema.ID_FIELD]))].merge(
        valid[[schema.ID_FIELD, TARGET_FIELD, *ROBUSTNESS_TARGET_FIELDS]], on=schema.ID_FIELD,
        how='inner')
    return frame, build_sample_filter_audit(analysis, salary)


def build_sample_filter_audit(analysis: pd.DataFrame, salary: pd.DataFrame) -> pd.DataFrame:
    """08_模型样本筛选：逐步筛选与 14,883 + 2,245 + 16 = 17,144 恒等式。"""
    total = len(analysis)
    negotiable = int(salary[schema.SALARY_NEGOTIABLE_FIELD].eq(1).sum())
    parsed = int(salary[schema.SALARY_PARSE_STATUS_FIELD].eq('已解析').sum())
    anomaly = int(salary[schema.SALARY_ANOMALY_FIELD].fillna('').astype(str).str.strip().ne('').sum())
    modeling = parsed - anomaly
    rows = [
        {'筛选步骤': '全量最终岗位实体', '岗位数': total, '规则': '1 intern_id = 1 行',
         '校验': ''},
        {'筛选步骤': '剔除薪资面议岗位', '岗位数': -negotiable,
         '规则': f'{schema.SALARY_NEGOTIABLE_FIELD} = 1（正式目标不可用，仍保留在全量分析集）',
         '校验': ''},
        {'筛选步骤': '剔除解析失败/无薪资中点岗位', '岗位数': -(total - negotiable - parsed),
         '规则': f'{schema.SALARY_PARSE_STATUS_FIELD} ≠ 已解析 或 薪资中点为空', '校验': ''},
        {'筛选步骤': '剔除薪资逻辑异常岗位', '岗位数': -anomaly,
         '规则': f'{schema.SALARY_ANOMALY_FIELD} 非空（下限非正数 / 上下限倒置 / 上限非正数）',
         '校验': ''},
        {'筛选步骤': '正式建模样本', '岗位数': modeling, '规则': '主目标 = 薪资中点',
         '校验': f'{modeling} + {negotiable} + {anomaly} = {modeling + negotiable + anomaly}'
                 f'（应为 {total}）'},
    ]
    return pd.DataFrame(rows)


def build_feature_manifest(analysis_columns, model_frame: pd.DataFrame) -> pd.DataFrame:
    """Feature Manifest：逐字段给出特征组、来源、类型、是否参与模型与处理方式。"""
    model_columns = set(model_frame.columns)
    rows = [_manifest_row(column, model_columns) for column in analysis_columns]
    for column in ROBUSTNESS_TARGET_FIELDS:
        rows.append({
            '字段名': column, '特征组': GROUP_TARGET, '来源Stage': 'Stage 11',
            '来源文件': 'data/processed/job_salary_targets.parquet', '数据类型': '数值',
            '是否数值': 1, '是否类别': 0, '是否多值': 0, '是否参与模型': 0, '是否目标派生': 1,
            '是否存在泄漏风险': '是（目标派生）',
            '处理方式': '禁止进入 X；仅用于描述统计与稳健性实验', '备注': '薪资派生字段（黑名单）',
        })
    rows.append({
        '字段名': TARGET_FIELD, '特征组': GROUP_TARGET, '来源Stage': 'Stage 11',
        '来源文件': 'data/processed/job_salary_targets.parquet', '数据类型': '数值',
        '是否数值': 1, '是否类别': 0, '是否多值': 0, '是否参与模型': 0, '是否目标派生': 1,
        '是否存在泄漏风险': '是（主目标 y）', '处理方式': '仅作为 y，禁止进入 X',
        '备注': '主目标 =（薪资下限 + 薪资上限）/ 2',
    })
    rows.append({
        '字段名': 'skill_multi_hot（高频规范技能 multi-hot）',
        '特征组': GROUP_SKILL, '来源Stage': 'Stage 07',
        '来源文件': 'data/features/job_skill_membership.parquet',
        '数据类型': '稀疏 0/1 矩阵', '是否数值': 1, '是否类别': 0, '是否多值': 1,
        '是否参与模型': 1, '是否目标派生': 0, '是否存在泄漏风险': '否',
        '处理方式': '建模阶段用 MultiLabelBinarizer / sparse pivot 临时构造；'
                '阈值候选 50 / 80 / 100 / 0.5% training jobs，最终在 Stage 14 validation 上选择',
        '备注': f'技能口径 = ALL_USABLE；建模样本 {len(model_frame)} 行；'
                'EMPTY_TEXT 不解释为「无技能」，由文本是否为空/技能提取范围显式控制',
    })
    rows.append({
        '字段名': 'bge_svd（model-safe 文本语义降维）',
        '特征组': GROUP_TEXT, '来源Stage': 'Stage 08',
        '来源文件': 'data/features/job_text_embeddings.npz（job_text_safe，512 维）',
        '数据类型': '数值向量', '是否数值': 1, '是否类别': 0, '是否多值': 1,
        '是否参与模型': 1, '是否目标派生': 0, '是否存在泄漏风险': '否（仅使用 model-safe 文本向量）',
        '处理方式': 'SVD/PCA 16 / 32 / 64 维候选，只用训练集拟合后变换验证集/测试集；'
                '原始 512 维仅作敏感性实验',
        '备注': '按 文本向量行号 关联；缺失向量由 文本向量是否可用 标志控制',
    })
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def _manifest_row(column: str, model_columns: set) -> dict:
    spec = COLUMN_SPEC.get(column)
    if spec is None:
        spec = (GROUP_META, '文本', 0, 0, 0, 0, 'Stage 12', '未在 COLUMN_SPEC 中登记（需补登记）')
    group, dtype, is_numeric, is_category, is_multi, in_model, stage, note = spec
    if in_model:
        treatment = ('建模阶段编码（multi-hot / MultiLabelBinarizer）' if is_multi else
                     'OneHotEncoder(handle_unknown=ignore)；缺失编码为「未知」' if is_category else
                     '中位数填补 + 必要时缺失指示；线性模型 StandardScaler')
    else:
        treatment = '不进入 X（键 / 多值引用 / 高基数标识 / 目标派生）'
    return {
        '字段名': column,
        '特征组': group,
        '来源Stage': stage,
        '来源文件': SOURCE_FILES.get(stage, ''),
        '数据类型': dtype,
        '是否数值': int(is_numeric),
        '是否类别': int(is_category),
        '是否多值': int(is_multi),
        '是否参与模型': int(in_model and column in model_columns),
        '是否目标派生': 0,
        '是否存在泄漏风险': '否',
        '处理方式': treatment,
        '备注': note,
    }


def build_skill_threshold_table(membership: pd.DataFrame, model_ids: set,
                                config=None) -> pd.DataFrame:
    """06_技能特征候选：各阈值保留的技能数、矩阵维度与覆盖率。"""
    scopes = ('REQUIREMENT_SECTION', 'FULL_TEXT_FALLBACK')
    frame = membership[membership['match_scope'].isin(scopes)
                       & membership['intern_id'].isin(model_ids)]
    counts = frame.groupby('canonical_skill')['intern_id'].nunique().sort_values(ascending=False)
    denominator = len(model_ids)
    ratio_threshold = int(denominator * SKILL_THRESHOLD_RATIO)
    thresholds = [(f'>= {item}', item) for item in SKILL_THRESHOLD_CANDIDATES]
    thresholds.append((f'>= 0.5% × {denominator}（= {ratio_threshold}）', ratio_threshold))
    rows = []
    for label, threshold in thresholds:
        kept = counts[counts >= threshold]
        jobs = int(frame[frame['canonical_skill'].isin(set(kept.index))]['intern_id'].nunique())
        rows.append({
            '阈值口径': label,
            '阈值（岗位数）': threshold,
            '保留技能数': int(len(kept)),
            '矩阵维度': f'{denominator} × {len(kept)}',
            '至少命中 1 个保留技能的岗位数': jobs,
            '覆盖率': round(jobs / denominator, 6) if denominator else 0.0,
            '保留技能列表': '、'.join(kept.index.tolist()),
            '说明': '本轮只提供候选；最终阈值在 Stage 14 validation 上选择（禁止用 test set 决定）',
        })
    return pd.DataFrame(rows)


def build_cardinality_table(analysis: pd.DataFrame) -> pd.DataFrame:
    """07_类别特征基数：类别 / 多值字段基数与低频处理方案。"""
    columns = ['岗位大类集合', '岗位细分类集合', '学历要求', '每周到岗要求', '实习时长要求',
               schema.JOB_DIRECTION_FIELD, schema.JD_SPLIT_STATUS_FIELD, '工作城市_规范',
               '所属行业', '公司性质', '公司规模', 'company_entity_id', '公司名称',
               schema.COMPANY_TAG_LIST_FIELD, schema.SKILL_GROUP_FIELD]
    rows = []
    for column in dict.fromkeys(columns):
        if column not in analysis.columns:
            continue
        series = analysis[column]
        is_multi = bool(series.map(is_multi_value).any())
        if is_multi:
            unique = int(series.explode().dropna().nunique())
            mean_items = round(float(series.map(len).mean()), 4)
            plan = '高频取值 multi-hot + 低频合并 OTHER（阈值按训练集确定）'
        else:
            unique = int(series.nunique(dropna=True))
            mean_items = 1.0
            top = series.value_counts(dropna=False).head(3)
            plan = f'Top3：{"，".join(f"{key}({value})" for key, value in top.items())}'
        rows.append({
            '字段': column,
            '类型': '多值列表' if is_multi else '单值类别',
            '唯一取值数': unique,
            '是否高基数（>200）': '是' if unique > 200 else '否',
            '每行平均取值数': mean_items,
            '低频/编码方案': plan,
        })
    return pd.DataFrame(rows)


def build_missing_table(analysis: pd.DataFrame, model_frame: pd.DataFrame) -> pd.DataFrame:
    """05_缺失值统计：逐字段缺失数与处理策略（多值列以「空列表」计缺失）。"""
    def count_missing(frame: pd.DataFrame, column: str) -> int:
        if column not in frame.columns:
            return -1
        series = frame[column]
        if series.map(is_multi_value).any():
            return int(series.map(lambda value: value is None
                                  or (is_multi_value(value) and len(value) == 0)).sum())
        return int(series.isna().sum())

    rows = []
    for column in analysis.columns:
        series = analysis[column]
        is_multi = bool(series.map(is_multi_value).any())
        rows.append({
            '字段名': column,
            '分析集缺失数': count_missing(analysis, column),
            '分析集缺失率': round(count_missing(analysis, column) / len(analysis), 6),
            '建模集缺失数': count_missing(model_frame, column),
            '策略': ('多值列表：空列表视为无该关系' if is_multi
                   else '数值：中位数填补 + 必要时缺失指示'
                   if pd.api.types.is_numeric_dtype(series) and series.dtype != object
                   else '类别：编码为「未知」（不回写源数据）'),
        })
    return pd.DataFrame(rows)

# -*- coding: utf-8 -*-
"""正式 EDA 与统计检验计算层（Stage 13）。

## 口径（严格遵循封版）

- EDA 分析单元 = **唯一岗位实体**（17,144，1 intern_id = 1 行），不使用 172,063 条原始搜索观测；
- 薪资分析样本 = **14,883**（非面议 + 解析成功 + 中点有效 + 无逻辑异常）；
- 技能主口径 = REQUIREMENT_SECTION（8,822）；扩展口径 = REQUIREMENT_SECTION + FULL_TEXT_FALLBACK（16,378）；
  EMPTY_TEXT（766）单独报告；
- 技能榜分四层（具体技术技能 / 技术领域 / 业务能力 / 办公工具），组级统计按 intern_id 去重；
- 两组比较 = Mann–Whitney U + Cliff's delta；多组比较 = Kruskal–Wallis（+ 成对 Mann–Whitney + BH-FDR）；
- 所有批量比较做 Benjamini–Hochberg FDR 校正；
- 措辞一律为「关联」，禁止因果表述。

## 复用而非重复实现

`cliff_delta`、`benjamini_hochberg`、`rank_skills`、`layer_rank`、`cooccurrence`、
`rank_robustness`、`group_job_counts` 直接复用 `src/skill_eda.py`（技能口径唯一实现）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import plot_style, project_paths, schema, skill_eda, skill_extraction, text_utils

# 统计与展示约定
MIN_GROUP_SIZE = 30          # 单因素分组展示的最小样本
MIN_TEST_GROUP = 50          # 两组比较做检验的最小样本（不足只做描述）
TOP_CATEGORY = 12
TOP_CATEGORY_SKILL = 15
NO_CAUSAL_NOTE = '描述性关联：禁止表述为因果（例如不得写「某技能导致薪资提高」）'
SALARY_LABEL = '薪资中点（元/天）'

# ---- 公司因素语义分离（Stage 16 取证结论 A2+A1 修正） ----
# 公司认证：Stage 05 正式产物 job_details_unique.parquet 的「公司认证标签」（原子标签仅
# 最佳雇主 / 行业认证），构造互斥四类后作为有限类别的公司因素；
# 公司标签（福利标签）：Stage 12 宽表的「公司标签列表」（免费健身设施 / 弹性工作制 等），
# 属高基数多值因素，**不得**再称为「公司认证」。
CERT_EMPLOYER = '最佳雇主'
CERT_INDUSTRY = '行业认证'
CERT_CLASS_FIELD = '公司认证类别'
CERT_CLASS_NONE = '无认证'
CERT_CLASS_BOTH = '两者均有'
CERT_CLASS_ORDER = [CERT_CLASS_NONE, CERT_INDUSTRY, CERT_EMPLOYER, CERT_CLASS_BOTH]
COMPANY_CERT_LABEL = '公司认证'
COMPANY_TAG_LABEL = '公司标签（福利标签）'
TAG_FACTOR_KEY = 'company_tags'
CERT_FACTOR_KEY = 'company_certification'

# ---- 公司标签（福利标签）正式口径：描述性统计 + 单标签 present/absent 二元比较 ----
# 公司标签是高基数多值字段（同一岗位可同时命中多个标签），标签组之间**不独立**，
# 因此禁止把数百个重叠标签当作一个整体做 Kruskal–Wallis 正式推断。
TAG_ANALYSIS_TYPE = '描述性多值标签'
DEPRECATED_INFERENCE_FLAG = 'DEPRECATED_INFERENCE（已废止推断）'
DEPRECATED_TAG_KW_VALUE = 0.471492
# 二元比较样本量门槛：复用既有两组检验常量（50/50），不新造阈值
MIN_BINARY_GROUP_SIZE = MIN_TEST_GROUP
# Cliff's delta 通用分级（Romano et al. 2006，学界通行阈值），本项目首次用于标签效应量分级
CLIFF_EFFECT_THRESHOLDS = (0.147, 0.33, 0.474)
CLIFF_EFFECT_LABELS = ('negligible（可忽略）', 'small（小）', 'medium（中）', 'large（大）')


def cliff_effect_label(delta) -> str:
    """Cliff's delta 效应分级（学界通行阈值，非本项目自定）。"""
    if delta is None or (isinstance(delta, float) and np.isnan(delta)):
        return '未检验'
    magnitude = abs(float(delta))
    for threshold, label in zip(CLIFF_EFFECT_THRESHOLDS, CLIFF_EFFECT_LABELS):
        if magnitude < threshold:
            return label
    return CLIFF_EFFECT_LABELS[-1]


def certification_atoms(certification: pd.DataFrame) -> list:
    """公司认证原子标签全集（未知值逐项返回，禁止静默忽略）。"""
    series = certification[schema.CERT_TAG_FIELD]
    atoms = sorted({str(tag) for value in series.dropna() for tag in value})
    unknown = sorted(set(atoms) - {CERT_EMPLOYER, CERT_INDUSTRY})
    return atoms, unknown


def certification_classes(values) -> str:
    """把公司认证标签 list[str] 映射为互斥四类（无认证 / 最佳雇主 / 行业认证 / 两者均有）。"""
    tags = {str(tag) for tag in values} if text_utils.as_list(values) else set()
    unknown = tags - {CERT_EMPLOYER, CERT_INDUSTRY}
    if unknown:
        raise ValueError(f'出现未登记的公司认证标签：{sorted(unknown)}（禁止静默忽略）')
    has_employer = CERT_EMPLOYER in tags
    has_industry = CERT_INDUSTRY in tags
    if has_employer and has_industry:
        return CERT_CLASS_BOTH
    if has_employer:
        return CERT_EMPLOYER
    if has_industry:
        return CERT_INDUSTRY
    return CERT_CLASS_NONE


def attach_company_certification(frame: pd.DataFrame,
                                 certification: pd.DataFrame) -> pd.DataFrame:
    """把公司认证标签（Stage 05 正式产物）并入 EDA 帧并派生互斥四类。

    只读引用上游正式产物，**不修改 Stage 12 宽表**、不新增模型特征。
    """
    atoms, unknown = certification_atoms(certification)
    if unknown:
        raise ValueError(f'出现未登记的公司认证标签：{unknown}（禁止静默忽略）')
    merged = frame.merge(certification[[schema.ID_FIELD, schema.CERT_TAG_FIELD]],
                         on=schema.ID_FIELD, how='left')
    if merged[schema.CERT_TAG_FIELD].isna().any():
        missing = int(merged[schema.CERT_TAG_FIELD].isna().sum())
        raise ValueError(f'{missing} 个岗位缺少公司认证标签，无法构造互斥认证类别')
    merged[CERT_CLASS_FIELD] = merged[schema.CERT_TAG_FIELD].map(certification_classes)
    return merged.drop(columns=[schema.CERT_TAG_FIELD])


# ---------------------------------------------------------------- 基础统计

def describe(values) -> dict:
    """单变量描述统计（含分位数与 IQR）。"""
    array = np.asarray(pd.Series(values).dropna(), dtype='float64')
    if array.size == 0:
        return {'n': 0}
    return {
        'n': int(array.size),
        '均值': round(float(array.mean()), 4),
        '中位数': round(float(np.median(array)), 4),
        '标准差': round(float(array.std(ddof=1)) if array.size > 1 else 0.0, 4),
        'IQR': round(float(np.percentile(array, 75) - np.percentile(array, 25)), 4),
        'P10': round(float(np.percentile(array, 10)), 4),
        'P25': round(float(np.percentile(array, 25)), 4),
        'P50': round(float(np.percentile(array, 50)), 4),
        'P75': round(float(np.percentile(array, 75)), 4),
        'P90': round(float(np.percentile(array, 90)), 4),
        'min': round(float(array.min()), 4),
        'max': round(float(array.max()), 4),
    }


def kruskal_wallis(groups) -> dict:
    """多组比较：Kruskal–Wallis H + 效应量 epsilon²。"""
    from scipy import stats  # noqa: PLC0415

    arrays = [np.asarray(group, dtype='float64') for group in groups if len(group) > 0]
    if len(arrays) < 2:
        return {'H统计量': None, 'p值': None, 'epsilon平方': None, '组数': len(arrays)}
    h_stat, p_value = stats.kruskal(*arrays)
    total = sum(len(item) for item in arrays)
    epsilon_sq = ((h_stat - len(arrays) + 1) / (total - len(arrays))) if total > len(arrays) else 0.0
    return {
        'H统计量': round(float(h_stat), 6),
        'p值': float(p_value),
        'epsilon平方': round(float(max(epsilon_sq, 0.0)), 6),
        '组数': len(arrays),
    }


def two_group_stats(present, absent, min_group: int = MIN_TEST_GROUP) -> dict:
    """两组比较：Mann–Whitney U + Cliff's delta（样本不足只做描述）。"""
    from scipy import stats  # noqa: PLC0415

    left = np.asarray(pd.Series(present).dropna(), dtype='float64')
    right = np.asarray(pd.Series(absent).dropna(), dtype='float64')
    row = {
        'n_present': int(left.size), 'n_absent': int(right.size),
        'median_present': round(float(np.median(left)), 4) if left.size else None,
        'median_absent': round(float(np.median(right)), 4) if right.size else None,
        'IQR_present': round(float(np.percentile(left, 75) - np.percentile(left, 25)), 4)
        if left.size else None,
        'IQR_absent': round(float(np.percentile(right, 75) - np.percentile(right, 25)), 4)
        if right.size else None,
        'median_diff': (round(float(np.median(left) - np.median(right)), 4)
                        if left.size and right.size else None),
        "cliff_delta": (round(skill_eda.cliff_delta(left, right), 6)
                        if left.size and right.size else None),
        '测试是否达标': '是' if min(left.size, right.size) >= min_group else '否（仅描述）',
    }
    if min(left.size, right.size) >= min_group:
        u_stat, p_value = stats.mannwhitneyu(left, right, alternative='two-sided')
        row['u_stat'] = float(u_stat)
        row['p_raw'] = float(p_value)
    else:
        row['u_stat'] = None
        row['p_raw'] = None
    return row


def add_fdr(table: pd.DataFrame, p_column: str = 'p_raw',
            output_column: str = 'p_adjusted_bh', alpha: float = 0.05) -> pd.DataFrame:
    """Benjamini–Hochberg FDR 校正（复用 skill_eda.benjamini_hochberg）。"""
    frame = table.copy()
    if p_column not in frame.columns:
        frame[output_column] = None
        frame['FDR显著'] = '否'
        return frame
    mask = frame[p_column].notna()
    adjusted = pd.Series([None] * len(frame), dtype=object)
    if mask.any():
        adjusted.loc[mask] = list(skill_eda.benjamini_hochberg(frame.loc[mask, p_column].to_numpy()))
    frame[output_column] = pd.to_numeric(adjusted, errors='coerce')
    frame['FDR显著'] = np.where(frame[output_column].notna() & (frame[output_column] < alpha),
                             '是', '否')
    return frame


# ---------------------------------------------------------------- 样本与结构表

def build_sample_overview(analysis: pd.DataFrame, model: pd.DataFrame,
                          salary: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """01_样本概况。"""
    parsed = int(salary[schema.SALARY_PARSE_STATUS_FIELD].eq('已解析').sum())
    anomaly = int(salary[schema.SALARY_ANOMALY_FIELD].fillna('').astype(str).str.strip().ne('').sum())
    negotiable = int(salary[schema.SALARY_NEGOTIABLE_FIELD].eq(1).sum())
    category_jobs = analysis['岗位大类集合'].explode().dropna().nunique()
    sub_category_jobs = analysis['岗位细分类集合'].explode().dropna().nunique()
    rows = [
        {'指标': '全量岗位数（EDA 分析单元）', '数值': len(analysis),
         '说明': '1 intern_id = 1 行；不使用 172,063 条原始搜索观测'},
        {'指标': '正式薪资分析样本', '数值': len(model), '说明': f'{parsed} − {anomaly}（逻辑异常）'},
        {'指标': '薪资面议岗位数', '数值': negotiable,
         '说明': f'占比 {negotiable / len(analysis):.2%}（仅在全量分析集做描述性分析）'},
        {'指标': '薪资逻辑异常岗位数', '数值': anomaly, '说明': '不进正式薪资分析样本'},
        {'指标': '城市数（工作城市_规范）', '数值': int(analysis['工作城市_规范'].nunique()),
         '说明': '标准化后的城市取值数'},
        {'指标': '岗位大类数', '数值': int(category_jobs), '说明': '多值关系，含全部大类取值'},
        {'指标': '岗位细分类数', '数值': int(sub_category_jobs), '说明': '多值关系，含全部细分类取值'},
        {'指标': '公司数（原始名称）', '数值': int(analysis['公司名称'].nunique()), '说明': ''},
        {'指标': '公司实体数', '数值': int(analysis['company_entity_id'].nunique()),
         '说明': 'Stage 09 公司实体（含跨地域歧义标记）'},
        {'指标': '所属行业数', '数值': int(analysis['所属行业'].nunique()), '说明': ''},
        {'指标': '技能提取口径样本', '数值': '',
         '说明': f'主口径 REQUIREMENT_SECTION '
                 f'{int((analysis[schema.SKILL_SCOPE_FIELD] == "REQUIREMENT_SECTION").sum())}；'
                 f'FULL_TEXT_FALLBACK '
                 f'{int((analysis[schema.SKILL_SCOPE_FIELD] == "FULL_TEXT_FALLBACK").sum())}；'
                 f'EMPTY_TEXT {int((analysis[schema.SKILL_SCOPE_FIELD] == "EMPTY_TEXT").sum())}'},
        {'指标': '主口径岗位数', '数值': int(
            (analysis[schema.SKILL_SCOPE_FIELD] == 'REQUIREMENT_SECTION').sum()),
         '说明': 'REQUIREMENT_SECTION（技能主分析分母）'},
        {'指标': '扩展口径 fallback 岗位数', '数值': int(
            (analysis[schema.SKILL_SCOPE_FIELD] == 'FULL_TEXT_FALLBACK').sum()),
         '说明': 'FULL_TEXT_FALLBACK（扩展口径组成部分）'},
        {'指标': '空文本岗位数', '数值': int(
            (analysis[schema.SKILL_SCOPE_FIELD] == 'EMPTY_TEXT').sum()),
         '说明': 'EMPTY_TEXT：无可用技能文本，单独报告，不解释为「无技能要求」'},
        {'指标': '技能关系行数（long-format）', '数值': len(membership),
         '说明': '(intern_id, canonical_skill) 唯一'},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 结构化因素与薪资

def _factor_frame(model: pd.DataFrame) -> pd.DataFrame:
    frame = model.copy()
    frame['_salary'] = frame[schema.SALARY_MID_FIELD]
    return frame


def build_group_salary(model: pd.DataFrame, group_kind: str) -> tuple:
    """按因素分组的中位数/IQR 表 + Kruskal–Wallis 检验。

    group_kind: 'category'（岗位大类，多值）/ 'sub_category'（岗位细分类，多值）/
    'city' / 'education' / 'company_scale' / 'company_nature' / 'industry' /
    'attendance' / 'duration' / 'company_certification'（公司认证，互斥四类，单值）。

    公司认证（最佳雇主 / 行业认证）与公司标签（福利标签）是**两个不同因素**，
    禁止互相替代或合并命名；**公司标签不在此函数内做整体 Kruskal–Wallis**
    （高基数多值标签组互不独立），改用 `build_company_tag_descriptive` +
    `build_binary_multivalue_factor_tests`。
    """
    frame = _factor_frame(model)
    specification = {
        'category': ('岗位大类集合', '多值', '岗位大类'),
        'sub_category': ('岗位细分类集合', '多值', '岗位细分类'),
        'city': ('工作城市_规范', '单值', '工作城市'),
        'education': ('学历要求', '单值', '学历要求'),
        'company_scale': ('公司规模', '单值', '公司规模'),
        'company_nature': ('公司性质', '单值', '公司性质'),
        'industry': ('所属行业', '单值', '公司所属行业'),
        'attendance': ('每周到岗要求', '单值', '每周到岗要求'),
        'duration': ('实习时长要求', '单值', '实习时长要求'),
        CERT_FACTOR_KEY: (CERT_CLASS_FIELD, '单值', COMPANY_CERT_LABEL),
    }[group_kind]
    column, kind, label = specification
    if kind == '多值':
        exploded = frame.explode(column).dropna(subset=[column])
        grouped = exploded.groupby(column)
    else:
        grouped = frame.fillna({column: '缺失'}).groupby(column)
    rows = []
    groups_for_test = []
    for value, group in grouped:
        values = group['_salary'].dropna()
        if len(values) < MIN_GROUP_SIZE:
            continue
        rows.append({
            '因素': label, '取值': value, '样本数': int(len(values)),
            '中位数': round(float(values.median()), 4),
            'IQR': round(float(values.quantile(0.75) - values.quantile(0.25)), 4),
            'P25': round(float(values.quantile(0.25)), 4),
            'P75': round(float(values.quantile(0.75)), 4),
        })
        groups_for_test.append(values.to_numpy())
    table = pd.DataFrame(rows).sort_values('中位数', ascending=False).reset_index(drop=True)
    test = kruskal_wallis(groups_for_test)
    if kind == '多值':
        note = (f'仅纳入样本数 ≥ {MIN_GROUP_SIZE} 的取值；'
                '多值因素按岗位-取值展开（同一岗位可计入多个取值）')
    else:
        note = f'仅纳入样本数 ≥ {MIN_GROUP_SIZE} 的取值'
    test.update({'因素': label, '参与检验组数': len(groups_for_test), '说明': note})
    return table, test


def build_pairwise_tests(model: pd.DataFrame, group_kind: str, top_n: int = 8) -> pd.DataFrame:
    """成对 Mann–Whitney + Cliff's delta + BH-FDR（按中位数降序取前 n 个取值）。"""
    table, _test = build_group_salary(model, group_kind)
    if table.empty:
        return pd.DataFrame()
    label = table['因素'].iloc[0]
    frame = _factor_frame(model)
    column = {
        'category': '岗位大类集合', 'sub_category': '岗位细分类集合', 'city': '工作城市_规范',
        'education': '学历要求', 'company_scale': '公司规模', 'company_nature': '公司性质',
        'industry': '所属行业', 'attendance': '每周到岗要求', 'duration': '实习时长要求',
        CERT_FACTOR_KEY: CERT_CLASS_FIELD,
    }[group_kind]
    is_multi = group_kind in ('category', 'sub_category')
    values = table['取值'].head(top_n).tolist()
    rows = []
    for index, left in enumerate(values):
        for right in values[index + 1:]:
            left_values = _group_values(frame, column, left, is_multi)
            right_values = _group_values(frame, column, right, is_multi)
            if min(len(left_values), len(right_values)) < MIN_GROUP_SIZE:
                continue
            stats_row = two_group_stats(left_values, right_values)
            stats_row.update({'因素': label, '取值A': left, '取值B': right,
                              'median_diff_A_minus_B': stats_row.pop('median_diff')})
            rows.append(stats_row)
    table_out = pd.DataFrame(rows)
    if table_out.empty:
        return table_out
    table_out = add_fdr(table_out)
    table_out.insert(0, '检验块', '成对比较（Mann–Whitney + BH-FDR）')
    table_out['说明'] = NO_CAUSAL_NOTE
    return table_out.sort_values('p_raw').reset_index(drop=True)


def _group_values(frame: pd.DataFrame, column: str, value, is_multi: bool) -> np.ndarray:
    if is_multi:
        mask = frame[column].map(lambda items: value in set(text_utils.as_list(items)))
    else:
        mask = frame[column].fillna('缺失').eq(value)
    return frame.loc[mask, '_salary'].dropna().to_numpy()


# ---------------------------------------------------------------- 公司标签（福利标签）正式口径

def build_company_tag_descriptive(model: pd.DataFrame,
                                  list_field: str = schema.COMPANY_TAG_LIST_FIELD,
                                  salary_field: str = schema.SALARY_MID_FIELD) -> pd.DataFrame:
    """06_公司因素薪资 · 公司标签描述统计（高基数多值标签，**不做整体 Kruskal–Wallis**）。

    输出：标签 / 岗位数 / 岗位占比 / 薪资样本数 / 薪资中位数 / IQR / P25 / P75 / 分析类型。
    岗位占比分母 = 全量分析岗位数同口径的薪资样本数（14,883）。
    """
    frame = model[[schema.ID_FIELD, list_field, salary_field]].copy()
    total_jobs = int(len(frame))
    exploded = frame.explode(list_field).dropna(subset=[list_field])
    exploded = exploded.assign(**{list_field: exploded[list_field].astype(str)})
    rows = []
    for tag, block in exploded.groupby(list_field):
        jobs = int(block[schema.ID_FIELD].nunique())
        values = block[salary_field].dropna().to_numpy(dtype='float64')
        rows.append({
            '因素': COMPANY_TAG_LABEL, '标签': tag, '取值': tag,
            '岗位数': jobs, '岗位占比': round(jobs / total_jobs, 6) if total_jobs else 0.0,
            '样本数': int(values.size),
            '薪资中位数': round(float(np.median(values)), 4) if values.size else None,
            '中位数': round(float(np.median(values)), 4) if values.size else None,
            'IQR': round(float(np.percentile(values, 75) - np.percentile(values, 25)), 4)
            if values.size else None,
            'P25': round(float(np.percentile(values, 25)), 4) if values.size else None,
            'P75': round(float(np.percentile(values, 75)), 4) if values.size else None,
            '分析类型': TAG_ANALYSIS_TYPE,
        })
    table = pd.DataFrame(rows).sort_values(['岗位数', '标签'],
                                           ascending=[False, True]).reset_index(drop=True)
    table.insert(0, '排名', np.arange(1, len(table) + 1))
    table['说明'] = ('高基数多值标签：同一岗位可同时命中多个标签，标签组之间不独立，'
                    '禁止整体 Kruskal–Wallis 正式推断；本表仅作描述性统计')
    return table


def tag_cluster_overlap(model: pd.DataFrame, tag_binary: pd.DataFrame, top_n: int = 5,
                        list_field: str = schema.COMPANY_TAG_LIST_FIELD) -> dict:
    """|Cliff's delta| 最高标签之间的岗位重叠（避免把同一标签簇当作多个独立关联）。"""
    if tag_binary.empty:
        return {}
    ranked = tag_binary.reindex(tag_binary['cliff_delta'].abs()
                                .sort_values(ascending=False).index)
    tags = ranked.head(top_n)['标签'].astype(str).tolist()
    sets = []
    for tag in tags:
        mask = model[list_field].map(lambda items: tag in {str(item) for item in items})
        sets.append(set(model.loc[mask, schema.ID_FIELD]))
    if len(sets) < 2:
        return {'标签': tags}
    jaccard = max(len(left & right) / len(left | right) if left | right else 0.0
                  for index, left in enumerate(sets) for right in sets[index + 1:])
    return {
        '标签': tags,
        '两两最大Jaccard': round(float(jaccard), 4),
        '全部交集岗位数': int(len(set.intersection(*sets))),
        '全部并集岗位数': int(len(set.union(*sets))),
        '说明': '|Cliff\'s delta| 最高的标签高度共现（同一批岗位的标签簇），'
                '不能当作彼此独立的薪资关联证据；论文呈现薪资关联榜时必须同时给出共现说明。',
    }


def build_binary_multivalue_factor_tests(
        model: pd.DataFrame, list_field: str = schema.COMPANY_TAG_LIST_FIELD,
        salary_field: str = schema.SALARY_MID_FIELD,
        min_present: int = MIN_BINARY_GROUP_SIZE,
        min_absent: int = MIN_BINARY_GROUP_SIZE,
        label: str = COMPANY_TAG_LABEL) -> pd.DataFrame:
    """公司标签正式推断：单标签 present vs absent 二元比较。

    对每个标签构造互斥二组（present ∩ absent = ∅，present_n + absent_n = 薪资样本数），
    仅纳入 present_n ≥ `min_present` 且 absent_n ≥ `min_absent` 的标签；
    统一做 Mann–Whitney U + Cliff's delta + Benjamini–Hochberg FDR。
    """
    frame = model[[schema.ID_FIELD, list_field, salary_field]].copy()
    frame['_salary'] = frame[salary_field]
    total_jobs = int(len(frame))
    membership = frame[[schema.ID_FIELD, list_field]].explode(list_field).dropna(subset=[list_field])
    membership[list_field] = membership[list_field].astype(str)
    present_map = membership.groupby(list_field)[schema.ID_FIELD].apply(set).to_dict()
    rows = []
    for tag, jobs in present_map.items():
        present = frame.loc[frame[schema.ID_FIELD].isin(jobs), '_salary'].dropna().to_numpy('float64')
        absent = frame.loc[~frame[schema.ID_FIELD].isin(jobs), '_salary'].dropna().to_numpy('float64')
        if len(present) < min_present or len(absent) < min_absent:
            continue
        stats_row = two_group_stats(present, absent)
        stats_row.update({
            '因素': label, '标签': tag, '取值A': 'present（具有该标签）',
            '取值B': 'absent（不具有该标签）',
            '岗位数': int(len(jobs)),
            '岗位占比': round(len(jobs) / total_jobs, 6) if total_jobs else 0.0,
            'median_diff_A_minus_B': stats_row.pop('median_diff'),
            'effect_label': cliff_effect_label(stats_row.get('cliff_delta')),
        })
        rows.append(stats_row)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = add_fdr(table)
    table.insert(0, '检验块', '单标签二元比较（present vs absent；Mann–Whitney + BH-FDR）')
    table['说明'] = (NO_CAUSAL_NOTE + '；该标签存在与薪资分布差异为描述性关联，'
                    '不等于该标签造成薪资变化')
    return table.sort_values('p_raw').reset_index(drop=True)


# ---------------------------------------------------------------- 技能需求（复用封版口径）

def build_skill_demand(membership: pd.DataFrame, universe: dict, config) -> tuple:
    """07_技能需求：主口径分层榜单 + 扩展口径排名。"""
    main_rank = skill_eda.rank_skills(membership, universe[skill_eda.SCOPE_MAIN], config,
                                      scopes=skill_eda.SCOPE_MAIN)
    ext_rank = skill_eda.rank_skills(membership, universe['ALL_USABLE'], config,
                                     scopes=skill_eda.ALL_USABLE_SCOPES)
    layers = skill_extraction.resolve_rank_layers(config)
    detail = skill_eda.layer_rank(main_rank, skill_eda.LAYER_TECHNICAL, 20).copy()
    detail.insert(0, '榜单', '核心技术技能 Top20（主口径，分母 8,822）')
    blocks = [detail]
    for layer, title, top_n in [
        (skill_eda.LAYER_DOMAIN, '技术领域 Top15（主口径，分母 8,822）', 15),
        (skill_eda.LAYER_BUSINESS, '业务能力（主口径，分母 8,822）', None),
        (skill_eda.LAYER_OFFICE, '办公工具（主口径，分母 8,822）', None),
    ]:
        table = skill_eda.layer_rank(main_rank, layer, top_n).copy()
        table.insert(0, '榜单', title)
        blocks.append(table)
    extended = ext_rank.head(20).copy()
    extended.insert(0, '榜单', '扩展口径 Top20（分母 16,378）')
    extended['层级'] = extended['技能标准名'].map(skill_extraction.skill_layer_map(config))
    blocks.append(extended)
    demand = pd.concat(blocks, ignore_index=True, sort=False)
    demand['口径说明'] = ('主口径 = REQUIREMENT_SECTION（企业明确要求段落）；'
                       '扩展口径 = REQUIREMENT_SECTION + FULL_TEXT_FALLBACK；'
                       'EMPTY_TEXT 单独报告，不解释为「无技能要求」')
    return demand, main_rank, ext_rank, layers


def build_category_skill_matrix(membership: pd.DataFrame, analysis: pd.DataFrame, universe: dict,
                                main_rank: pd.DataFrame) -> pd.DataFrame:
    """13.6 岗位细分类 × 高频技能命中率矩阵（分母 = 细分类内部岗位数）。"""
    categories = analysis[[schema.ID_FIELD, '岗位细分类集合']].explode('岗位细分类集合')
    categories = categories.dropna().rename(columns={'岗位细分类集合': '岗位细分类'})
    return skill_eda.category_skill_matrix(membership, categories,
                                           universe[skill_eda.SCOPE_MAIN], main_rank,
                                           scopes=skill_eda.SCOPE_MAIN,
                                           top_categories=TOP_CATEGORY,
                                           top_skills=TOP_CATEGORY_SKILL)


def build_skill_salary(membership: pd.DataFrame, salary_layer: pd.DataFrame, model_ids: set,
                       analysis: pd.DataFrame, universe: dict,
                       main_rank: pd.DataFrame) -> tuple:
    """08/09/13：技能薪资（逐技能）、岗位细分类内、技能数量档。

    逐技能与细分类内对比的限制集 = 主口径 ∩ 正式薪资样本；
    技能数量档使用全部正式薪资样本（技能来自 ALL_USABLE 口径）。
    """
    base_ids = set(universe[skill_eda.SCOPE_MAIN]) & set(model_ids)
    focus = sorted(set(main_rank[main_rank['层级'] == skill_eda.LAYER_TECHNICAL]
                       .head(20)['技能标准名'].tolist())
                   | set(['Python', 'Java', 'SQL', 'C++', 'Excel', '大模型', 'Agent']))
    per_skill = skill_eda.salary_association(membership, salary_layer, base_ids, focus,
                                            scopes=skill_eda.SCOPE_MAIN)
    per_skill = add_fdr(per_skill, p_column='p值', output_column='BH_FDR_q值')
    per_skill['说明'] = NO_CAUSAL_NOTE
    per_skill = per_skill.sort_values('中位数差', ascending=False).reset_index(drop=True)

    categories = analysis[[schema.ID_FIELD, '岗位细分类集合']].explode('岗位细分类集合')
    categories = categories.dropna().rename(columns={'岗位细分类集合': '岗位细分类'})
    within = skill_eda.salary_association_within_category(
        membership, salary_layer, categories, base_ids,
        ['Python', 'Java', 'SQL', 'Excel', '大模型', 'Agent'], scopes=skill_eda.SCOPE_MAIN)
    if not within.empty:
        within['说明'] = ('同岗位细分类内部的有/无技能对比（弱化「算法岗本身高薪」等类别结构混杂）；'
                       + NO_CAUSAL_NOTE)

    counts = (membership[membership['match_scope'].isin(skill_eda.ALL_USABLE_SCOPES)
                         & membership['intern_id'].isin(model_ids)]
              .groupby('intern_id')['canonical_skill'].nunique()
              .reindex(list(model_ids), fill_value=0))
    valid_mask = skill_eda.valid_salary_mask(salary_layer)
    mid_by_job = (salary_layer.loc[valid_mask]
                  .set_index(schema.ID_FIELD)[schema.SALARY_MID_FIELD].to_dict())
    merged = pd.DataFrame({'技能数': counts.values}, index=counts.index)
    merged['_salary'] = [mid_by_job.get(job) for job in merged.index]
    merged = merged[merged['_salary'].notna()]
    bucket = pd.cut(merged['技能数'], bins=[-1, 0, 1, 2, 3, 4, 10_000],
                    labels=['0', '1', '2', '3', '4', '5+'])
    rows = []
    groups = []
    for name, group in merged.groupby(bucket):
        values = group['_salary'].dropna()
        rows.append({'技能数量档': str(name), '样本数': int(len(values)),
                     '中位数': round(float(values.median()), 4),
                     'IQR': round(float(values.quantile(0.75) - values.quantile(0.25)), 4),
                     'P25': round(float(values.quantile(0.25)), 4),
                     'P75': round(float(values.quantile(0.75)), 4)})
        groups.append(values.to_numpy())
    count_table = pd.DataFrame(rows)
    test = kruskal_wallis(groups)
    test.update({'因素': '技能数量档（0/1/2/3/4/5+）', '参与检验组数': len(groups),
                 '说明': '技能口径 = ALL_USABLE；' + NO_CAUSAL_NOTE})
    return per_skill, within, count_table, test


def build_robustness(model: pd.DataFrame, main_rank: pd.DataFrame,
                     ext_rank: pd.DataFrame) -> pd.DataFrame:
    """12_稳健性：双口径排名一致性 + 薪资 1%/99% 截断对照。"""
    summary, detail = skill_eda.rank_robustness(main_rank, ext_rank)
    rows = [{'项目': key, '数值': value,
             '说明': '主口径 = REQUIREMENT_SECTION；扩展口径 = REQUIREMENT_SECTION + '
                     'FULL_TEXT_FALLBACK（复用 27 号审计口径，未重新设计）'}
            for key, value in summary.items()]
    salary = model[schema.SALARY_MID_FIELD].dropna()
    low, high = np.percentile(salary, 1), np.percentile(salary, 99)
    truncated = salary.clip(lower=low, upper=high)
    rows.append({'项目': '薪资 1%/99% 截断边界', '数值': f'{low:.1f} ~ {high:.1f}',
                 '说明': '主分析保留真实数据，截断仅作稳健性参考'})
    for label, values in [('原始', salary), ('1%/99% 截断', truncated)]:
        stats = describe(values)
        rows.append({'项目': f'{label} 中位数 / IQR',
                     '数值': f"{stats['中位数']} / {stats['IQR']}",
                     '说明': f"n = {stats['n']}"})
    skill_top = main_rank.head(20)[['技能标准名', '岗位数', '岗位占比']].to_dict('records')
    rows.append({'项目': '主口径 Top20 技能（用于稳健性核对）',
                 '数值': '、'.join(f"{item['技能标准名']}({item['岗位数']})" for item in skill_top),
                 '说明': '与 27 号审计表一致'})
    rows.append({'项目': '排名差明细（Top50）',
                 '数值': f'{len(detail)} 行', '说明': '见 29 号表 12_稳健性 明细块'})
    table = pd.DataFrame(rows)
    detail_block = detail.copy()
    detail_block.insert(0, '项目', 'Top50 技能排名对照')
    detail_block['数值'] = detail_block['排名差']
    return pd.concat([table, detail_block], ignore_index=True, sort=False)


# ---------------------------------------------------------------- 图表（统一 SCI 风格）

def _prepare_figure_dir() -> None:
    """正式 EDA 图统一保存到 outputs/figures/eda/（复用 SCI 保存与校验流程）。"""
    project_paths.EDA_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_style.SCI_FIGURES_DIR = project_paths.EDA_FIGURES_DIR


def _finish(fig, stem: str, caption: str, subfigures=None, meta=None, registry=None) -> dict:
    diagnostics = plot_style.save_sci_figure(fig, stem, caption, subfigures=subfigures, meta=meta)
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plt.close(fig)
    if registry is not None:
        registry.append(diagnostics)
    return diagnostics


def figure_sample_structure(overview_rows: pd.DataFrame, registry: list) -> dict:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    lookup = overview_rows.set_index('指标')['数值']
    counts = {
        '全量岗位': int(lookup['全量岗位数（EDA 分析单元）']),
        '薪资可解析': int(lookup['正式薪资分析样本']) + int(lookup['薪资逻辑异常岗位数']),
        '正式薪资样本': int(lookup['正式薪资分析样本']),
        '薪资面议': int(lookup['薪资面议岗位数']),
        '逻辑异常': int(lookup['薪资逻辑异常岗位数']),
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.0))
    ax = axes[0]
    series = pd.Series(counts)
    plot_style.bar_ranked(ax, series, xlabel='岗位数', ylabel='样本口径')
    plot_style.apply_sci_axis(ax, grid_axis='x')
    plot_style.format_integer_axis(ax, axis='x')
    plot_style.add_subfigure_caption(ax, 'a', '样本口径构成（单位：岗位）')
    ax2 = axes[1]
    layer = pd.Series({
        'REQUIREMENT_SECTION': int(lookup.get('主口径岗位数', 8822)),
        'FULL_TEXT_FALLBACK': int(lookup.get('扩展口径 fallback 岗位数', 7556)),
        'EMPTY_TEXT': int(lookup.get('空文本岗位数', 766)),
    })
    plot_style.bar_ranked(ax2, layer, color=plot_style.ACCENT_COLOR,
                          xlabel='岗位数', ylabel='技能提取口径')
    plot_style.apply_sci_axis(ax2, grid_axis='x')
    plot_style.format_integer_axis(ax2, axis='x')
    plot_style.add_subfigure_caption(ax2, 'b', '技能提取口径构成（单位：岗位）')
    fig.subplots_adjust(wspace=0.35, bottom=0.3)
    plot_style.add_bottom_caption(fig, '图01 正式 EDA 样本与技能口径结构')
    return _finish(fig, '01_sample_structure', '图01 正式 EDA 样本与技能口径结构',
                   subfigures=[('a', '样本口径构成（单位：岗位）', axes[0]),
                               ('b', '技能提取口径构成（单位：岗位）', axes[1])],
                   meta={'图表类型': '柱状图', '数据来源': 'job_analysis_dataset / 27 号审计表'},
                   registry=registry)


def figure_salary_distribution(model: pd.DataFrame, registry: list) -> dict:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    values = model[schema.SALARY_MID_FIELD].dropna().to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.0))
    ax = axes[0]
    plot_style.hist_discrete(ax, values, discrete=False, bins=40,
                             xlabel=SALARY_LABEL, ylabel='岗位数')
    plot_style.apply_sci_axis(ax, grid_axis='y')
    plot_style.annotate_stats(ax, values)
    plot_style.add_subfigure_caption(ax, 'a', '薪资中点分布直方图')
    ax2 = axes[1]
    sorted_values = np.sort(values)
    ecdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    ax2.plot(sorted_values, ecdf, color=plot_style.MAIN_COLOR, linewidth=1.1)
    ax2.set_xlabel(SALARY_LABEL)
    ax2.set_ylabel('累计占比')
    plot_style.apply_sci_axis(ax2)
    plot_style.format_percent_axis(ax2, axis='y')
    plot_style.add_subfigure_caption(ax2, 'b', '薪资中点经验累积分布（ECDF）')
    fig.subplots_adjust(wspace=0.3, bottom=0.3)
    caption = f'图02 正式薪资样本薪资中点分布（n = {len(values):,}，含极端值，未强制缩尾）'
    plot_style.add_bottom_caption(fig, caption)
    return _finish(fig, '02_salary_distribution', caption,
                   subfigures=[('a', '薪资中点分布直方图', axes[0]),
                               ('b', '薪资中点经验累积分布（ECDF）', axes[1])],
                   meta={'图表类型': '直方图 + ECDF', '数据来源': 'job_salary_model_dataset'},
                   registry=registry)


def figure_category_salary(category_table: pd.DataFrame, registry: list) -> dict:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    frame = category_table.head(TOP_CATEGORY).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    positions = np.arange(len(frame))
    ax.barh(positions, frame['中位数'], height=0.62, color=plot_style.MAIN_COLOR,
            edgecolor='black', linewidth=0.5,
            xerr=frame['IQR'] / 2, error_kw={'elinewidth': 0.8, 'capsize': 2.5})
    ax.set_yticks(positions)
    ax.set_yticklabels(frame['取值'])
    ax.set_xlabel(SALARY_LABEL)
    ax.set_ylabel('岗位细分类')
    plot_style.apply_sci_axis(ax, grid_axis='x')
    plot_style.add_bottom_caption(fig, '图03 主要岗位细分类薪资中点中位数（误差线为 IQR/2）')
    fig.subplots_adjust(left=0.28, bottom=0.16)
    return _finish(fig, '03_category_salary',
                   '图03 主要岗位细分类薪资中点中位数（误差线为 IQR/2）',
                   meta={'图表类型': '横向柱状图', '数据来源': 'job_salary_model_dataset'},
                   registry=registry)


def figure_structured_factors(city_table: pd.DataFrame, education_table: pd.DataFrame,
                              scale_table: pd.DataFrame, registry: list) -> dict:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.0))
    panels = [('a', '城市（Top8）', city_table, '工作城市'),
              ('b', '学历要求', education_table, '学历要求'),
              ('c', '公司规模', scale_table, '公司规模')]
    for (letter, title, table, _label), ax in zip(panels, axes):
        frame = table.head(8).iloc[::-1]
        positions = np.arange(len(frame))
        ax.barh(positions, frame['中位数'], height=0.6, color=plot_style.MAIN_COLOR,
                edgecolor='black', linewidth=0.5)
        ax.set_yticks(positions)
        ax.set_yticklabels([str(item) for item in frame['取值']])
        ax.set_xlabel(SALARY_LABEL)
        ax.set_ylabel(title)
        plot_style.apply_sci_axis(ax, grid_axis='x')
        plot_style.add_subfigure_caption(ax, letter, f'{title}薪资中点中位数')
    fig.subplots_adjust(wspace=0.55, bottom=0.3)
    caption = '图04 城市、学历与公司规模的薪资中点中位数对比（各因素仅展示样本数 ≥ 30 的取值）'
    plot_style.add_bottom_caption(fig, caption)
    return _finish(fig, '04_structured_factor_salary', caption,
                   subfigures=[(letter, f'{title}薪资中点中位数', ax)
                               for (letter, title, _table, _l), ax in zip(panels, axes)],
                   meta={'图表类型': '三联横向柱状图', '数据来源': 'job_salary_model_dataset'},
                   registry=registry)


def figure_tech_skill_top20(rank_table: pd.DataFrame, registry: list) -> dict:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    frame = rank_table.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    positions = np.arange(len(frame))
    ax.barh(positions, frame['岗位数'], height=0.66, color=plot_style.MAIN_COLOR,
            edgecolor='black', linewidth=0.5)
    ax.set_yticks(positions)
    ax.set_yticklabels(frame['技能标准名'])
    ax.set_xlabel('明确要求该技能的岗位数（主口径）')
    ax.set_ylabel('具体技术技能')
    plot_style.apply_sci_axis(ax, grid_axis='x')
    plot_style.format_integer_axis(ax, axis='x')
    caption = '图05 核心技术技能 Top20（主口径 REQUIREMENT_SECTION，分母 8,822 个岗位）'
    plot_style.add_bottom_caption(fig, caption)
    fig.subplots_adjust(left=0.3, bottom=0.16)
    return _finish(fig, 'fig_05_tech_skill_top20', caption,
                   meta={'图表类型': '横向柱状图', '数据来源': 'job_skill_membership（主口径）'},
                   registry=registry)


def figure_skill_layers(domain_table: pd.DataFrame, business_table: pd.DataFrame,
                        office_table: pd.DataFrame, registry: list) -> dict:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.2))
    panels = [('a', '技术领域', domain_table, 12),
              ('b', '业务能力', business_table, 8),
              ('c', '办公工具', office_table, 8)]
    for (letter, title, table, top_n), ax in zip(panels, axes):
        frame = table.head(top_n).iloc[::-1]
        positions = np.arange(len(frame))
        ax.barh(positions, frame['岗位数'], height=0.6, color=plot_style.ACCENT_COLOR,
                edgecolor='black', linewidth=0.5)
        ax.set_yticks(positions)
        ax.set_yticklabels([str(item) for item in frame['技能标准名']])
        ax.set_xlabel('岗位数（主口径）')
        ax.set_ylabel(title)
        plot_style.apply_sci_axis(ax, grid_axis='x')
        plot_style.format_integer_axis(ax, axis='x')
        plot_style.add_subfigure_caption(ax, letter, f'{title}需求岗位数')
    fig.subplots_adjust(wspace=0.6, bottom=0.3)
    caption = ('图06 技能需求的分层结构：技术领域 / 业务能力 / 办公工具'
               '（主口径，分母 8,822，各组独立统计不跨层相加）')
    plot_style.add_bottom_caption(fig, caption)
    return _finish(fig, '06_skill_layer_structure', caption,
                   subfigures=[(letter, f'{title}需求岗位数', ax)
                               for (letter, title, _t, _n), ax in zip(panels, axes)],
                   meta={'图表类型': '三联横向柱状图', '数据来源': 'job_skill_membership（主口径）'},
                   registry=registry)


def figure_category_skill_heatmap(matrix: pd.DataFrame, registry: list) -> dict:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if matrix.empty:
        raise ValueError('岗位细分类技能矩阵为空，无法出图')
    values = matrix.drop(columns=['细分类岗位数']).set_index('岗位细分类')
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    image = ax.imshow(values.to_numpy(dtype='float64'), aspect='auto', cmap='cividis',
                      vmin=0.0, vmax=float(np.nanmax(values.to_numpy(dtype='float64'))))
    ax.set_xticks(np.arange(values.shape[1]))
    ax.set_xticklabels(values.columns, rotation=60, ha='right')
    ax.set_yticks(np.arange(values.shape[0]))
    ax.set_yticklabels([f'{name}（n={int(count)}）' for name, count
                        in zip(matrix['岗位细分类'], matrix['细分类岗位数'])])
    ax.set_xlabel('具体技术技能（Top15）')
    ax.set_ylabel('岗位细分类（Top12）')
    plot_style.apply_sci_axis(ax, grid=False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label('细分类内部命中率', fontsize=plot_style.FONT_SIZES['axis_label'])
    caption = '图07 岗位细分类 × 技能命中率热力图（主口径，值为细分类内部命中率，非 raw count）'
    plot_style.add_bottom_caption(fig, caption)
    fig.subplots_adjust(left=0.22, bottom=0.22)
    return _finish(fig, '07_category_skill_heatmap', caption,
                   meta={'图表类型': '热力图', '数据来源': 'job_skill_membership（主口径）'},
                   registry=registry)


def figure_skill_cooccurrence(cooc_table: pd.DataFrame, top_n: int = 15,
                              registry: list = None) -> dict:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if cooc_table.empty:
        raise ValueError('技能共现表为空，无法出图')
    skills = []
    for row in cooc_table.head(40).itertuples(index=False):
        for skill in (row.技能A, row.技能B):
            if skill not in skills:
                skills.append(skill)
    skills = skills[:top_n]
    matrix = pd.DataFrame(np.nan, index=skills, columns=skills)
    for row in cooc_table.itertuples(index=False):
        if row.技能A in skills and row.技能B in skills:
            matrix.loc[row.技能A, row.技能B] = row.Lift
            matrix.loc[row.技能B, row.技能A] = row.Lift
    np.fill_diagonal(matrix.values, 1.0)
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    image = ax.imshow(matrix.to_numpy(dtype='float64'), cmap='magma', vmin=1.0,
                      vmax=float(np.nanmax(matrix.to_numpy(dtype='float64'))))
    ax.set_xticks(np.arange(len(skills)))
    ax.set_xticklabels(skills, rotation=60, ha='right')
    ax.set_yticks(np.arange(len(skills)))
    ax.set_yticklabels(skills)
    ax.set_xlabel('具体技术技能')
    ax.set_ylabel('具体技术技能')
    plot_style.apply_sci_axis(ax, grid=False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label('共现 Lift', fontsize=plot_style.FONT_SIZES['axis_label'])
    caption = '图08 具体技术技能共现热力图（共现岗位数 ≥ 30，颜色为 Lift，按 Lift 排序）'
    plot_style.add_bottom_caption(fig, caption)
    fig.subplots_adjust(left=0.2, bottom=0.24)
    return _finish(fig, '08_skill_cooccurrence', caption,
                   meta={'图表类型': '热力图', '数据来源': 'job_skill_membership（主口径）'},
                   registry=registry)


def figure_skill_salary(per_skill: pd.DataFrame, count_table: pd.DataFrame,
                        registry: list) -> dict:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    focus = ['Python', 'Java', 'SQL', 'C++', 'Excel', '大模型', 'Agent']
    frame = per_skill[per_skill['技能标准名'].isin(focus)].copy()
    frame = frame.sort_values('中位数差')
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    ax = axes[0]
    positions = np.arange(len(frame))
    ax.barh(positions - 0.18, frame['有技能薪资中位数'], height=0.32, color=plot_style.MAIN_COLOR,
            edgecolor='black', linewidth=0.5, label='明确要求该技能')
    ax.barh(positions + 0.18, frame['无技能薪资中位数'], height=0.32, color=plot_style.MUTED_COLOR,
            edgecolor='black', linewidth=0.5, label='未明确要求该技能')
    ax.set_yticks(positions)
    ax.set_yticklabels(frame['技能标准名'])
    ax.set_xlabel(SALARY_LABEL)
    ax.set_ylabel('重点技能')
    ax.legend(loc='lower right', frameon=False)
    plot_style.apply_sci_axis(ax, grid_axis='x')
    plot_style.add_subfigure_caption(ax, 'a', '重点技能有/无对比（中位数）')
    ax2 = axes[1]
    count_frame = count_table.copy()
    positions = np.arange(len(count_frame))
    ax2.bar(positions, count_frame['中位数'], width=0.6, color=plot_style.MAIN_COLOR,
            edgecolor='black', linewidth=0.5)
    ax2.errorbar(positions, count_frame['中位数'], yerr=count_frame['IQR'] / 2, fmt='none',
                 ecolor='black', elinewidth=0.8, capsize=2.5)
    ax2.set_xticks(positions)
    ax2.set_xticklabels(count_frame['技能数量档'])
    ax2.set_xlabel('岗位技能数量档')
    ax2.set_ylabel(SALARY_LABEL)
    plot_style.apply_sci_axis(ax2, grid_axis='y')
    plot_style.add_subfigure_caption(ax2, 'b', '技能数量档与薪资中位数（误差线为 IQR/2）')
    fig.subplots_adjust(wspace=0.35, bottom=0.3)
    caption = ('图09 技能与薪资的描述性关联（主口径；中位数差异不代表因果关系）')
    plot_style.add_bottom_caption(fig, caption)
    return _finish(fig, '09_skill_salary', caption,
                   subfigures=[('a', '重点技能有/无对比（中位数）', axes[0]),
                               ('b', '技能数量档与薪资中位数（误差线为 IQR/2）', axes[1])],
                   meta={'图表类型': '对比柱状图', '数据来源': 'job_salary_model_dataset'},
                   registry=registry)


def figure_scope_robustness(main_rank: pd.DataFrame, ext_rank: pd.DataFrame,
                            summary: dict, registry: list) -> dict:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    main_order = {name: index + 1 for index, name in enumerate(main_rank['技能标准名'])}
    ext_order = {name: index + 1 for index, name in enumerate(ext_rank['技能标准名'])}
    shared = [name for name in main_order if name in ext_order]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    ax = axes[0]
    xs = [main_order[name] for name in shared]
    ys = [ext_order[name] for name in shared]
    ax.scatter(xs, ys, s=12, color=plot_style.MAIN_COLOR, alpha=0.75, edgecolor='none')
    limit = max(max(xs), max(ys))
    ax.plot([1, limit], [1, limit], color=plot_style.MUTED_COLOR, linewidth=0.9,
            linestyle='--')
    ax.set_xlabel('主口径排名（REQUIREMENT_SECTION）')
    ax.set_ylabel('扩展口径排名（ALL_USABLE）')
    ax.text(0.97, 0.06, f'Spearman = {summary["Spearman 排名相关"]:.3f}',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=plot_style.FONT_SIZES['annotation'])
    plot_style.apply_sci_axis(ax)
    plot_style.add_subfigure_caption(ax, 'a', '双口径技能排名对照')
    ax2 = axes[1]
    labels = ['Top10 overlap', 'Top20 overlap']
    values = [summary['Top10 overlap 比例'], summary['Top20 overlap 比例']]
    positions = np.arange(len(labels))
    ax2.bar(positions, values, width=0.5, color=plot_style.ACCENT_COLOR, edgecolor='black',
            linewidth=0.5)
    for position, value in zip(positions, values):
        ax2.text(position, value + 0.02, f'{value:.0%}', ha='center', va='bottom',
                 fontsize=plot_style.FONT_SIZES['annotation'])
    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel('一致性指标')
    ax2.set_ylabel('重合比例')
    plot_style.apply_sci_axis(ax2, grid_axis='y')
    plot_style.format_percent_axis(ax2, axis='y')
    plot_style.add_subfigure_caption(ax2, 'b', '双口径 Top 榜重合比例')
    fig.subplots_adjust(wspace=0.32, bottom=0.3)
    caption = '图10 技能口径稳健性（主口径 vs 扩展口径，排名一致说明结论稳健）'
    plot_style.add_bottom_caption(fig, caption)
    return _finish(fig, '10_scope_robustness', caption,
                   subfigures=[('a', '双口径技能排名对照', axes[0]),
                               ('b', '双口径 Top 榜重合比例', axes[1])],
                   meta={'图表类型': '散点 + 柱状图', '数据来源': 'job_skill_membership'},
                   registry=registry)

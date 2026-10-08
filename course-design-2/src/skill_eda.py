# -*- coding: utf-8 -*-
"""技能需求 EDA 的计算层（Stage 13 口径封版，唯一实现）。

## 三种文本口径（match_scope，来自 Stage 07）

| 口径 | 岗位数（当前） | 用途 |
| --- | --- | --- |
| REQUIREMENT_SECTION | 8,822 | **论文正文主口径**：企业明确提出的任职要求段落 |
| FULL_TEXT_FALLBACK | 7,556 | 扩展技能画像 / 建模特征 / 稳健性分析（模型安全 JD 全文） |
| EMPTY_TEXT | 766 | 独立缺失口径：无可用技能文本，**不得解释为「企业没有技能要求」** |

主口径 = REQUIREMENT_SECTION；扩展口径 = REQUIREMENT_SECTION + FULL_TEXT_FALLBACK（排除 EMPTY_TEXT）。

## 榜单层级（复用 feature_family / group，见 config/skills.yml → rank_layers）

A 具体技术技能 / B 技术领域 / C 业务能力 / D 办公工具；
层级内统计一律按 intern_id 去重，**禁止跨层级或同层内简单相加**
（Excel + PPT + Word + 办公软件 ≠ 办公工具岗位数）。

## 统计口径

- 分母：主口径 = REQUIREMENT_SECTION 岗位数；扩展口径 = 有可用文本岗位数；
- 技能与薪资：非面议 + 解析成功 + 薪资中点有效 + 无逻辑异常；
  Mann–Whitney U + Cliff's delta + Benjamini–Hochberg FDR；
- 一律为描述性关联，禁止因果表述。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import schema, skill_extraction

# ---- 口径定义 ----
SCOPE_MAIN = 'REQUIREMENT_SECTION'
SCOPE_FALLBACK = 'FULL_TEXT_FALLBACK'
SCOPE_EMPTY = 'EMPTY_TEXT'
ALL_USABLE_SCOPES = (SCOPE_MAIN, SCOPE_FALLBACK)
# 榜单层级名称（与 config/skills.yml → rank_layers 一致）
LAYER_TECHNICAL = '具体技术技能'
LAYER_DOMAIN = '技术领域'
LAYER_BUSINESS = '业务能力'
LAYER_OFFICE = '办公工具'

# 技能共现：最小共同岗位数（低于该值不进入正式排序，避免高频技能天然高共现）
COOCCURRENCE_MIN_JOINT = 30
# 技能与薪资：最小可用样本（两侧均不足时只做描述，不做检验）
SALARY_MIN_GROUP = 50
# 技能与薪资回归的正式显著性水平
FDR_ALPHA = 0.05


def load_scope_universe(features: pd.DataFrame) -> dict:
    """返回各口径的岗位 ID 集合与样本规模（分母来自 Stage 07 的技能提取范围）。"""
    scope_series = features[schema.SKILL_SCOPE_FIELD]
    job_ids = features[schema.ID_FIELD]
    main = set(job_ids[scope_series.eq(SCOPE_MAIN)])
    fallback = set(job_ids[scope_series.eq(SCOPE_FALLBACK)])
    empty = set(job_ids[scope_series.eq(SCOPE_EMPTY)])
    return {
        SCOPE_MAIN: main,
        SCOPE_FALLBACK: fallback,
        SCOPE_EMPTY: empty,
        'ALL_USABLE': main | fallback,
    }


def filter_membership(membership: pd.DataFrame, scopes) -> pd.DataFrame:
    """按 match_scope 筛选 long-format（不生成 strict/full 两张重复表）。"""
    wanted = {scopes} if isinstance(scopes, str) else set(scopes)
    return membership[membership['match_scope'].isin(wanted)]


def rank_skills(membership: pd.DataFrame, universe_ids: set, config,
                scopes=None) -> pd.DataFrame:
    """技能排名：岗位数按 intern_id 去重，占比分母为指定口径的岗位数。"""
    frame = membership if scopes is None else filter_membership(membership, scopes)
    frame = frame[frame[schema.SKILL_MEMBERSHIP_ID_FIELD].isin(universe_ids)]
    denominator = len(universe_ids)
    layer_map = skill_extraction.skill_layer_map(config)
    grouped = (frame.groupby(schema.SKILL_MEMBERSHIP_SKILL_FIELD)
               [schema.SKILL_MEMBERSHIP_ID_FIELD].nunique().sort_values(ascending=False))
    rows = []
    for skill, job_count in grouped.items():
        rows.append({
            '排名': 0,
            '技能标准名': skill,
            '层级': layer_map.get(skill, ''),
            'feature_family': config.skill_to_family.get(skill, ''),
            '技能组': config.skill_to_group.get(skill, ''),
            '岗位数': int(job_count),
            '岗位占比': round(job_count / denominator, 6) if denominator else 0.0,
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(columns=['排名', '技能标准名', '层级', 'feature_family', '技能组',
                                     '岗位数', '岗位占比'])
    table = table.sort_values(['岗位数', '技能标准名'], ascending=[False, True]).reset_index(drop=True)
    table['排名'] = np.arange(1, len(table) + 1)
    return table


def layer_rank(rank_table: pd.DataFrame, layer: str, top_n: int = None) -> pd.DataFrame:
    """按层级筛选榜单（层级划分来自 skills.yml rank_layers）。"""
    subset = rank_table[rank_table['层级'] == layer].reset_index(drop=True)
    subset['层级内排名'] = np.arange(1, len(subset) + 1)
    return subset.head(top_n) if top_n else subset


def group_job_counts(membership: pd.DataFrame, universe_ids: set, group: str,
                     config) -> tuple:
    """技能组岗位数（按 intern_id 去重）与技能项次，并给出「简单相加」的对照值。"""
    members = config.group_skills([group])
    frame = membership[membership[schema.SKILL_MEMBERSHIP_ID_FIELD].isin(universe_ids)
                       & membership[schema.SKILL_MEMBERSHIP_SKILL_FIELD].isin(members)]
    unique_jobs = int(frame[schema.SKILL_MEMBERSHIP_ID_FIELD].nunique())
    item_total = int(len(frame))
    naive_sum = int(frame.groupby(schema.SKILL_MEMBERSHIP_SKILL_FIELD)
                    [schema.SKILL_MEMBERSHIP_ID_FIELD].nunique().sum())
    return unique_jobs, item_total, naive_sum


def category_skill_matrix(membership: pd.DataFrame, categories: pd.DataFrame,
                          universe_ids: set, rank_table: pd.DataFrame, scopes=None,
                          top_categories: int = 15, top_skills: int = 20) -> pd.DataFrame:
    """岗位细分类 × 技能命中率（细分类内部占比，不是 raw count）。"""
    frame = membership if scopes is None else filter_membership(membership, scopes)
    frame = frame[frame[schema.SKILL_MEMBERSHIP_ID_FIELD].isin(universe_ids)]
    top_skill_names = rank_table.head(top_skills)['技能标准名'].tolist()
    frame = frame[frame[schema.SKILL_MEMBERSHIP_SKILL_FIELD].isin(top_skill_names)]
    merged = frame.merge(categories[[schema.ID_FIELD, '岗位细分类']], left_on='intern_id',
                         right_on=schema.ID_FIELD, how='inner')
    size_by_category = (categories[categories[schema.ID_FIELD].isin(universe_ids)]
                        .groupby('岗位细分类')[schema.ID_FIELD].nunique())
    top_category_names = size_by_category.sort_values(ascending=False).head(top_categories).index
    merged = merged[merged['岗位细分类'].isin(top_category_names)]
    if merged.empty:
        return pd.DataFrame()
    pivot = (merged.pivot_table(index='岗位细分类', columns='canonical_skill',
                                values='intern_id', aggfunc='nunique', fill_value=0)
             .reindex(columns=top_skill_names, fill_value=0))
    share = pivot.div(size_by_category.reindex(pivot.index), axis=0).round(6)
    share['细分类岗位数'] = size_by_category.reindex(pivot.index).astype(int)
    return share.reset_index().sort_values('细分类岗位数', ascending=False)


def cooccurrence(membership: pd.DataFrame, universe_ids: set, rank_table: pd.DataFrame,
                 layer: str = LAYER_TECHNICAL, top_n: int = 25,
                 min_joint: int = COOCCURRENCE_MIN_JOINT, scopes=None) -> pd.DataFrame:
    """具体技能共现（共现岗位数 / Jaccard / Lift），按 Lift 排序。"""
    frame = membership if scopes is None else filter_membership(membership, scopes)
    frame = frame[frame[schema.SKILL_MEMBERSHIP_ID_FIELD].isin(universe_ids)]
    skills = rank_table[rank_table['层级'] == layer].head(top_n)['技能标准名'].tolist()
    frame = frame[frame[schema.SKILL_MEMBERSHIP_SKILL_FIELD].isin(skills)]
    if frame.empty:
        return pd.DataFrame()
    matrix = pd.crosstab(frame[schema.SKILL_MEMBERSHIP_ID_FIELD],
                         frame[schema.SKILL_MEMBERSHIP_SKILL_FIELD]).gt(0)
    denominator = int(matrix.shape[0])
    counts = matrix.sum(axis=0).to_dict()
    rows = []
    for i, left in enumerate(skills):
        if left not in matrix.columns:
            continue
        for right in skills[i + 1:]:
            if right not in matrix.columns:
                continue
            joint = int((matrix[left] & matrix[right]).sum())
            if joint < min_joint:
                continue
            union = int((matrix[left] | matrix[right]).sum())
            expected = (counts[left] / denominator) * (counts[right] / denominator)
            lift = (joint / denominator) / expected if expected else 0.0
            rows.append({
                '技能A': left, '技能B': right,
                '共现岗位数': joint,
                'Jaccard': round(joint / union, 6) if union else 0.0,
                'Lift': round(lift, 6),
                'A岗位数': int(counts[left]), 'B岗位数': int(counts[right]),
            })
    table = pd.DataFrame(rows)
    return (table.sort_values(['Lift', '共现岗位数'], ascending=[False, False]).reset_index(drop=True)
            if not table.empty else table)


def cliff_delta(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """Cliff's delta（效应量）：P(a > b) - P(a < b)，取值 [-1, 1]。"""
    if len(sample_a) == 0 or len(sample_b) == 0:
        return 0.0
    greater = 0
    less = 0
    for value in sample_a:
        greater += int(np.sum(value > sample_b))
        less += int(np.sum(value < sample_b))
    return (greater - less) / (len(sample_a) * len(sample_b))


def benjamini_hochberg(p_values) -> np.ndarray:
    """Benjamini–Hochberg FDR 校正（不依赖 statsmodels）。"""
    values = np.asarray(list(p_values), dtype=float)
    if values.size == 0:
        return values
    order = np.argsort(values)
    ranked = values[order]
    count = values.size
    adjusted = ranked * count / (np.arange(1, count + 1))
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    result = np.empty(count, dtype=float)
    result[order] = adjusted
    return result


def valid_salary_mask(salary: pd.DataFrame) -> pd.Series:
    """正式薪资样本掩码：非面议 + 解析成功 + 薪资中点有效 + 无逻辑异常。

    注意：`薪资异常标志` 为文本列，正常行为空字符串（不是 0/NaN）。
    """
    return (salary[schema.SALARY_NEGOTIABLE_FIELD].eq(0)
            & salary[schema.SALARY_PARSE_STATUS_FIELD].eq('已解析')
            & salary[schema.SALARY_MID_FIELD].notna()
            & salary[schema.SALARY_ANOMALY_FIELD].fillna('').astype(str).str.strip().eq(''))


def salary_association(membership: pd.DataFrame, salary: pd.DataFrame, universe_ids: set,
                       skills, scopes=None, min_group: int = SALARY_MIN_GROUP) -> pd.DataFrame:
    """技能 × 薪资的描述性关联（Mann–Whitney U + Cliff's delta + BH-FDR）。"""
    from scipy import stats  # noqa: PLC0415 - 仅在需要统计检验时导入

    frame = membership if scopes is None else filter_membership(membership, scopes)
    frame = frame[frame[schema.SKILL_MEMBERSHIP_ID_FIELD].isin(universe_ids)]
    valid = salary[valid_salary_mask(salary)]
    valid = valid[valid[schema.ID_FIELD].isin(universe_ids)]
    mid_by_job = valid.set_index(schema.ID_FIELD)[schema.SALARY_MID_FIELD].to_dict()
    jobs_by_skill = (frame.groupby(schema.SKILL_MEMBERSHIP_SKILL_FIELD)
                     [schema.SKILL_MEMBERSHIP_ID_FIELD].apply(set).to_dict())
    rows = []
    for skill in skills:
        present_jobs = jobs_by_skill.get(skill, set())
        present = np.array([mid_by_job[job] for job in present_jobs if job in mid_by_job], dtype=float)
        absent_jobs = [job for job in mid_by_job if job not in present_jobs]
        absent = np.array([mid_by_job[job] for job in absent_jobs], dtype=float)
        if len(present) == 0 or len(absent) == 0:
            continue
        u_stat, p_value = ((np.nan, np.nan) if (len(present) < min_group or len(absent) < min_group)
                           else stats.mannwhitneyu(present, absent, alternative='two-sided'))
        rows.append({
            '技能标准名': skill,
            '有技能岗位数': int(len(present)),
            '无技能岗位数': int(len(absent)),
            '有技能薪资中位数': round(float(np.median(present)), 4),
            '无技能薪资中位数': round(float(np.median(absent)), 4),
            '中位数差': round(float(np.median(present) - np.median(absent)), 4),
            '有技能IQR': round(float(np.percentile(present, 75) - np.percentile(present, 25)), 4),
            '无技能IQR': round(float(np.percentile(absent, 75) - np.percentile(absent, 25)), 4),
            'MannWhitneyU': None if np.isnan(u_stat) else float(u_stat),
            'p值': None if np.isnan(p_value) else float(p_value),
            "Cliff's delta": round(cliff_delta(present, absent), 6),
            '检验是否达标': '是' if (len(present) >= min_group and len(absent) >= min_group) else '否（仅描述）',
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    testable = table['p值'].notna()
    adjusted = pd.Series([None] * len(table), dtype=object)
    if testable.any():
        adjusted.loc[testable] = list(benjamini_hochberg(table.loc[testable, 'p值'].to_numpy()))
    table['BH_FDR_q值'] = pd.to_numeric(adjusted, errors='coerce')
    table['FDR显著'] = np.where(table['BH_FDR_q值'].notna() & (table['BH_FDR_q值'] < FDR_ALPHA),
                             '是', '否')
    return table.sort_values(['有技能岗位数', '技能标准名'],
                             ascending=[False, True]).reset_index(drop=True)


def salary_association_within_category(membership: pd.DataFrame, salary: pd.DataFrame,
                                       categories: pd.DataFrame, universe_ids: set, skills,
                                       scopes=None, top_n: int = 5) -> pd.DataFrame:
    """控制岗位细分类后的有/无技能薪资中位数差（降低岗位类别结构混杂）。"""
    frame = membership if scopes is None else filter_membership(membership, scopes)
    frame = frame[frame[schema.SKILL_MEMBERSHIP_ID_FIELD].isin(universe_ids)]
    frame = frame.merge(categories[[schema.ID_FIELD, '岗位细分类']], left_on='intern_id',
                        right_on=schema.ID_FIELD, how='inner')
    valid = salary[valid_salary_mask(salary)]
    frame = frame.merge(valid[[schema.ID_FIELD, schema.SALARY_MID_FIELD]],
                        left_on='intern_id', right_on=schema.ID_FIELD, how='inner')
    rows = []
    for skill in skills:
        present = set(frame.loc[frame['canonical_skill'] == skill, 'intern_id'])
        if not present:
            continue
        for category, group in frame.groupby('岗位细分类'):
            jobs = set(group['intern_id'])
            has = jobs & present
            has_not = jobs - present
            if len(has) < 10 or len(has_not) < 10:
                continue
            mid = group.drop_duplicates('intern_id').set_index('intern_id')[schema.SALARY_MID_FIELD]
            rows.append({
                '技能标准名': skill, '岗位细分类': category,
                '细分类样本数': int(len(jobs)),
                '有技能岗位数': int(len(has)), '无技能岗位数': int(len(has_not)),
                '有技能薪资中位数': round(float(mid.reindex(list(has)).median()), 4),
                '无技能薪资中位数': round(float(mid.reindex(list(has_not)).median()), 4),
                '中位数差': round(float(mid.reindex(list(has)).median()
                                  - mid.reindex(list(has_not)).median()), 4),
            })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = table.sort_values(['技能标准名', '中位数差'], ascending=[True, False])
    return table.groupby('技能标准名', as_index=False).head(top_n).reset_index(drop=True)


def rank_robustness(main_rank: pd.DataFrame, extended_rank: pd.DataFrame) -> tuple:
    """双口径稳健性：Top10 / Top20 overlap 与 Spearman 排名相关。"""
    from scipy import stats  # noqa: PLC0415 - 仅在需要统计检验时导入

    main_order = main_rank['技能标准名'].tolist()
    extended_order = extended_rank['技能标准名'].tolist()
    shared = [skill for skill in main_order if skill in set(extended_order)]
    main_rank_map = {skill: index + 1 for index, skill in enumerate(main_order)}
    ext_rank_map = {skill: index + 1 for index, skill in enumerate(extended_order)}
    if len(shared) >= 2:
        spearman, p_value = stats.spearmanr([main_rank_map[skill] for skill in shared],
                                            [ext_rank_map[skill] for skill in shared])
    else:
        spearman, p_value = float('nan'), float('nan')
    summary = {
        '共有技能数': len(shared),
        'Top10 overlap': len(set(main_order[:10]) & set(extended_order[:10])),
        'Top10 overlap 比例': round(len(set(main_order[:10]) & set(extended_order[:10]))
                                 / max(1, min(10, len(main_order), len(extended_order))), 6),
        'Top20 overlap': len(set(main_order[:20]) & set(extended_order[:20])),
        'Top20 overlap 比例': round(len(set(main_order[:20]) & set(extended_order[:20]))
                                 / max(1, min(20, len(main_order), len(extended_order))), 6),
        'Spearman 排名相关': None if np.isnan(spearman) else round(float(spearman), 6),
        'Spearman p值': None if np.isnan(p_value) else float(p_value),
    }
    detail = pd.DataFrame([
        {'技能标准名': skill, '主口径排名': main_rank_map[skill],
         '扩展口径排名': ext_rank_map.get(skill), '排名差': ext_rank_map.get(skill, np.nan) - main_rank_map[skill]}
        for skill in main_order[:50]])
    return summary, detail

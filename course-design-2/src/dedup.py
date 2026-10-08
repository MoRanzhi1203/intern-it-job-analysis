# -*- coding: utf-8 -*-
"""去重与唯一岗位构建模块。

职责：
- 重复岗位分组统计；
- 重复组业务字段一致性审计与冲突明细（支持 list[str] 字段）；
- 岗位大类/细分类关系构建与来源聚合（集合 + 命中数 + 稳定 JSON 计数）；
- 业务字段冲突汇总（业务字段冲突数 / 关键业务字段冲突标志）；
- 代表记录选择（四级规则，禁止 keep='first' 直取）；
- 唯一岗位表构造。
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from . import schema
from .quality import count_blank

# 代表记录选择依据取值
BASIS_SINGLE = '唯一记录'
BASIS_COMPLETENESS = '完整度最高'
BASIS_UPDATED = '完整度相同_更新时间最新'
BASIS_CREATED = '完整度相同_创建时间最新'
BASIS_ORDER = '完全一致_原始首条'

COMPLETENESS_FIELD = '_业务字段非空数量'
ORDER_FIELD = '_原始行顺序'


def canonicalize_value(value):
    """把任意值归一化为稳定、可哈希、可比较的表示。

    支持 list / tuple / set / dict / numpy 标量 / pandas NA / 字符串 / 数字 / datetime。
    用于业务字段一致性比较、冲突值列表输出与回归检查。
    """
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return tuple(canonicalize_value(item) for item in value.tolist())
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(canonicalize_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(key), canonicalize_value(item)) for key, item in value.items()))
    if isinstance(value, np.generic):
        return canonicalize_value(value.item())
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def dump_category_counts(series: pd.Series) -> str:
    """把类别序列渲染为稳定 JSON 计数串（key 排序、中文不转义、次数为 int）。"""
    counts: dict = {}
    for value in series.dropna():
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return json.dumps({key: counts[key] for key in sorted(counts)}, ensure_ascii=False)


def build_category_membership(frame: pd.DataFrame) -> pd.DataFrame:
    """构造岗位—分类关系表（一行 = 岗位ID + 岗位大类 + 岗位细分类 + 原始命中次数）。"""
    membership = (frame.groupby([schema.ID_FIELD, schema.CATEGORY_GROUP_FIELD,
                                 schema.CATEGORY_ITEM_FIELD])
                  .size().rename('原始命中次数').reset_index())
    membership = membership[schema.CATEGORY_MEMBERSHIP_COLUMNS]
    return membership.sort_values(list(schema.CATEGORY_MEMBERSHIP_COLUMNS[:3])).reset_index(drop=True)


def aggregate_source_categories(frame: pd.DataFrame,
                                records_per_id: pd.Series) -> pd.DataFrame:
    """按岗位聚合来源分类：集合（list[str]）+ 命中数 + 稳定 JSON 计数 + 原始记录数。"""
    id_field = schema.ID_FIELD
    index = pd.Index(sorted(frame[id_field].unique()), name=id_field)
    grouped = frame.groupby(id_field)
    aggregated = pd.DataFrame(index=index)

    set_fields = schema.CATEGORY_SET_FIELDS
    hit_fields = schema.CATEGORY_HIT_FIELDS
    count_fields = schema.CATEGORY_COUNT_FIELDS
    for set_field, hit_field, count_field, source_field in zip(
            set_fields, hit_fields, count_fields,
            (schema.CATEGORY_GROUP_FIELD, schema.CATEGORY_ITEM_FIELD)):
        aggregated[set_field] = grouped[source_field].agg(lambda s: sorted(set(s.dropna())))
        aggregated[hit_field] = grouped[source_field].nunique(dropna=True)
        aggregated[count_field] = grouped[source_field].agg(dump_category_counts)

    aggregated[schema.ORIGINAL_RECORD_FIELD] = records_per_id
    aggregated['来源搜索页码集合'] = grouped[schema.PAGE_FIELD].agg(lambda s: sorted(set(s.dropna())))
    return aggregated


def aggregate_categories_from_membership(membership: pd.DataFrame) -> pd.DataFrame:
    """从岗位分类关系表恢复主表的分类聚合字段（集合 / 命中数 / 计数 / 原始记录数）。

    权威来源是 `job_category_membership.parquet`，本函数不依赖原始观测明细。
    """
    id_field = schema.ID_FIELD
    set_fields = schema.CATEGORY_SET_FIELDS
    hit_fields = schema.CATEGORY_HIT_FIELDS
    count_fields = schema.CATEGORY_COUNT_FIELDS
    grouped = membership.groupby(id_field, sort=True)

    aggregated = pd.DataFrame(index=pd.Index(sorted(membership[id_field].unique()),
                                             name=id_field))
    for set_field, hit_field, count_field, source_field in zip(
            set_fields, hit_fields, count_fields,
            (schema.CATEGORY_GROUP_FIELD, schema.CATEGORY_ITEM_FIELD)):
        aggregated[set_field] = grouped[source_field].agg(lambda s: sorted(set(s)))
        aggregated[hit_field] = grouped[source_field].nunique()
        aggregated[count_field] = grouped.apply(
            lambda part, column=source_field: category_counts_json(part, column),
            include_groups=False)
    aggregated[schema.ORIGINAL_RECORD_FIELD] = grouped['原始命中次数'].sum().astype('int64')
    return aggregated


def category_counts_json(part: pd.DataFrame, column: str) -> str:
    """按类别累计「原始命中次数」并渲染为稳定 JSON 串（key 排序、中文不转义）。"""
    counts: dict = {}
    for value, hits in zip(part[column], part['原始命中次数']):
        key = str(value)
        counts[key] = counts.get(key, 0) + int(hits)
    return json.dumps({key: counts[key] for key in sorted(counts)}, ensure_ascii=False)


def reconcile_category_membership(membership: pd.DataFrame,
                                  aggregated: pd.DataFrame) -> dict:
    """校验关系表能否完整还原主表的岗位大类/细分类集合，且岗位ID集合一致。"""
    id_field = schema.ID_FIELD
    group_field, item_field = schema.CATEGORY_SET_FIELDS
    group_map = membership.groupby(id_field)[schema.CATEGORY_GROUP_FIELD].agg(
        lambda s: tuple(sorted(set(s)))).to_dict()
    item_map = membership.groupby(id_field)[schema.CATEGORY_ITEM_FIELD].agg(
        lambda s: tuple(sorted(set(s)))).to_dict()
    ids = list(aggregated.index)
    return {
        'membership_rows': int(len(membership)),
        'membership_ids': int(membership[id_field].nunique()),
        'group_set_match': all(group_map.get(i) == tuple(aggregated.at[i, group_field]) for i in ids),
        'item_set_match': all(item_map.get(i) == tuple(aggregated.at[i, item_field]) for i in ids),
        'id_set_match': set(ids) == set(membership[id_field].unique()),
    }


def summarize_conflicts(frame: pd.DataFrame, multi_value_ids: dict,
                        business_fields=None, key_fields=None) -> pd.DataFrame:
    """按岗位统计「业务字段冲突数」与「关键业务字段冲突标志」。"""
    business_fields = business_fields or schema.BUSINESS_FIELDS
    key_fields = key_fields or schema.KEY_CONFLICT_FIELDS
    id_field = schema.ID_FIELD
    index = pd.Index(sorted(frame[id_field].unique()), name=id_field)

    flags = pd.DataFrame(index=index)
    for field in business_fields:
        ids = list(multi_value_ids.get(field, []))
        flags[field] = index.isin(ids).astype('int64')

    summary = pd.DataFrame(index=index)
    summary[schema.CONFLICT_COUNT_FIELD] = flags.sum(axis=1).astype('int64')
    summary[schema.KEY_CONFLICT_FLAG_FIELD] = flags[list(key_fields)].max(axis=1).astype('int64')
    return summary


def select_representatives(frame: pd.DataFrame, business_fields=None,
                           time_fields=None) -> tuple:
    """按四级规则选择代表记录，返回（代表记录表, 选择依据分布表）。"""
    business_fields = business_fields or schema.BUSINESS_FIELDS
    time_fields = time_fields or schema.TIME_FIELDS
    created_field, updated_field = time_fields

    working = frame.copy()
    working[COMPLETENESS_FIELD] = len(business_fields) - count_blank(working, business_fields)
    working[ORDER_FIELD] = np.arange(len(working), dtype='int64')

    id_field = schema.ID_FIELD
    records_per_id = working.groupby(id_field).size()

    rank_completeness = working.groupby(id_field)[COMPLETENESS_FIELD].rank(
        method='min', ascending=False)
    completeness_tie = (rank_completeness.eq(1)
                        .groupby(working[id_field]).transform('sum'))

    top_completeness = working.loc[rank_completeness.eq(1)].copy()
    top_completeness['_更新时间并列数'] = (
        top_completeness.groupby(id_field)[updated_field]
        .rank(method='min', ascending=False).eq(1)
        .groupby(top_completeness[id_field]).transform('sum')
    )
    top_updated = top_completeness.loc[top_completeness['_更新时间并列数'].eq(1)].copy()
    top_updated['_创建时间并列数'] = (
        top_updated.groupby(id_field)[created_field]
        .rank(method='min', ascending=False).eq(1)
        .groupby(top_updated[id_field]).transform('sum')
    )

    profile = pd.DataFrame({
        '组内记录数': records_per_id,
        '完整度并列数': completeness_tie.groupby(working[id_field]).first(),
        '更新时间并列数': top_completeness.groupby(id_field)['_更新时间并列数'].first(),
        '创建时间并列数': top_updated.groupby(id_field)['_创建时间并列数'].first(),
    })

    representative = (
        working.sort_values([COMPLETENESS_FIELD, updated_field, created_field, ORDER_FIELD],
                            ascending=[False, False, False, True])
        .drop_duplicates(subset=id_field, keep='first')
    )
    profile = profile.reindex(representative[id_field].values)

    representative = representative.copy()
    representative['代表记录选择依据'] = np.select(
        [
            profile['组内记录数'].eq(1).to_numpy(),
            profile['完整度并列数'].eq(1).to_numpy(),
            profile['更新时间并列数'].eq(1).to_numpy(),
            profile['创建时间并列数'].eq(1).to_numpy(),
        ],
        [BASIS_SINGLE, BASIS_COMPLETENESS, BASIS_UPDATED, BASIS_CREATED],
        default=BASIS_ORDER,
    )
    basis_table = (representative['代表记录选择依据'].value_counts()
                   .rename('岗位数量').reset_index())
    basis_table.columns = ['代表记录选择依据', '岗位数量']
    return representative, basis_table


def build_unique_jobs(representative: pd.DataFrame, source_aggregated: pd.DataFrame,
                      conflict_summary: pd.DataFrame, entity_summary: pd.DataFrame | None = None,
                      columns=None) -> pd.DataFrame:
    """构造唯一岗位表：规范化链接 + 业务字段 + 时间字段 + 分类聚合 + 冲突汇总 + 选择依据。

    ``entity_summary`` 非空时额外并入版本摘要字段，输出列切换为
    ``schema.ENTITY_JOB_COLUMNS``（最终岗位实体层）。
    """
    id_field = schema.ID_FIELD
    columns = list(columns or (schema.ENTITY_JOB_COLUMNS if entity_summary is not None
                               else schema.UNIQUE_JOB_COLUMNS))
    merged_fields = list(schema.SOURCE_AGG_FIELDS) + [schema.CONFLICT_COUNT_FIELD,
                                                      schema.KEY_CONFLICT_FLAG_FIELD]
    if entity_summary is not None:
        merged_fields += list(entity_summary.columns)
    unique = pd.DataFrame({id_field: representative[id_field].to_numpy()})
    unique[schema.URL_FIELD] = representative[schema.URL_NORM_FIELD].to_numpy()

    for column in columns:
        if column in (id_field, schema.URL_FIELD) or column in merged_fields:
            continue
        unique[column] = representative[column].to_numpy()

    unique = unique.merge(source_aggregated[schema.SOURCE_AGG_FIELDS],
                          left_on=id_field, right_index=True, how='left')
    unique = unique.merge(conflict_summary, left_on=id_field, right_index=True, how='left')
    if entity_summary is not None:
        unique = unique.merge(entity_summary, left_on=id_field, right_index=True, how='left')
    return unique[columns]

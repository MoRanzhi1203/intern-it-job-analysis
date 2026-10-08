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


def format_canonical_value(value) -> str:
    """把归一化值渲染为可读文本（list 型渲染为 JSON 数组）。"""
    if value is None:
        return ''
    if isinstance(value, tuple):
        return json.dumps([item for item in value], ensure_ascii=False, default=str)
    return str(value)


def _nunique_per_id(frame: pd.DataFrame, field: str) -> pd.Series:
    """按岗位ID统计字段唯一值数量（比较前统一归一化，支持 list[str]）。"""
    canonical = frame[field].map(canonicalize_value)
    return canonical.groupby(frame[schema.ID_FIELD]).nunique(dropna=True)


def dump_category_counts(series: pd.Series) -> str:
    """把类别序列渲染为稳定 JSON 计数串（key 排序、中文不转义、次数为 int）。"""
    counts: dict = {}
    for value in series.dropna():
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return json.dumps({key: counts[key] for key in sorted(counts)}, ensure_ascii=False)


def profile_duplicate_groups(frame: pd.DataFrame, id_field: str = schema.ID_FIELD):
    """重复岗位基础统计，返回（每岗位记录数 Series, 统计表, 分布表）。"""
    records_per_id = frame.groupby(id_field).size().rename(schema.ORIGINAL_RECORD_FIELD)
    duplicate_groups = int((records_per_id > 1).sum())
    single_groups = int((records_per_id == 1).sum())

    profile = pd.DataFrame([
        {'指标': '原始记录数', '数值': len(frame)},
        {'指标': '唯一岗位数', '数值': int(frame[id_field].nunique())},
        {'指标': '重复组数量（记录数>1）', '数值': duplicate_groups},
        {'指标': '单记录组数量', '数值': single_groups},
        {'指标': '重复组覆盖原始记录数', '数值': int(records_per_id[records_per_id > 1].sum())},
        {'指标': '平均每岗位原始记录数', '数值': round(float(records_per_id.mean()), 4)},
        {'指标': '原始记录数 P50', '数值': float(records_per_id.quantile(0.5))},
        {'指标': '原始记录数 P95', '数值': float(records_per_id.quantile(0.95))},
        {'指标': '原始记录数 Max', '数值': int(records_per_id.max())},
    ])
    distribution = (records_per_id.value_counts().sort_index()
                    .rename('岗位数量').reset_index())
    distribution.columns = [schema.ORIGINAL_RECORD_FIELD, '岗位数量']
    return records_per_id, profile, distribution


def field_consistency_audit(frame: pd.DataFrame, records_per_id: pd.Series,
                            audit_fields=None, time_fields=None,
                            business_fields=None) -> tuple:
    """逐字段统计重复组内一致性，返回（统计表, {字段: 多值岗位ID}）。

    list[str] 字段（公司认证标签、岗位大类集合等）先归一化为 tuple 再比较。
    """
    audit_fields = audit_fields or schema.AUDIT_FIELDS
    time_fields = time_fields or schema.TIME_FIELDS
    business_fields = business_fields or schema.BUSINESS_FIELDS

    duplicate_groups = int((records_per_id > 1).sum())
    rows = []
    multi_value_ids: dict = {}
    for field in list(audit_fields) + list(time_fields):
        nunique_series = _nunique_per_id(frame, field)
        field_multi_ids = nunique_series[nunique_series > 1].index
        multi_value_ids[field] = field_multi_ids
        if field in time_fields:
            category = '采集记录时间'
        elif field in business_fields:
            category = '业务字段'
        else:
            category = '链接'
        rows.append({
            '字段名称': field,
            '重复岗位组数量': duplicate_groups,
            '完全一致岗位组数量': int((nunique_series <= 1).sum()),
            '存在多值岗位组数量': int(len(field_multi_ids)),
            '多值岗位组比例': round(len(field_multi_ids) / duplicate_groups, 6) if duplicate_groups else 0.0,
            '最大唯一值数量': int(nunique_series.max()),
            '涉及原始记录数': int(frame[schema.ID_FIELD].isin(field_multi_ids).sum()),
            '字段类别': category,
            '是否关键业务字段': '是' if field in schema.KEY_CONFLICT_FIELDS else '否',
            '是否业务字段': '是' if field in business_fields else '否',
        })
    table = pd.DataFrame(rows).sort_values(
        '存在多值岗位组数量', ascending=False).reset_index(drop=True)
    return table, multi_value_ids


def build_conflict_details(frame: pd.DataFrame, multi_value_ids: dict,
                           conflict_fields, records_per_id: pd.Series) -> tuple:
    """构造业务字段冲突概览与明细。

    明细固定列：实习岗位ID / 字段名称 / 唯一值数量 / 不同值列表 / 岗位标题 /
    公司名称 / 原始记录数 / 最早数据更新时间 / 最新数据更新时间。
    """
    id_index = frame.groupby(schema.ID_FIELD)
    base = pd.DataFrame(index=pd.Index(sorted(frame[schema.ID_FIELD].unique()),
                                       name=schema.ID_FIELD))
    base['岗位标题'] = id_index['岗位标题'].first()
    base['公司名称'] = id_index['公司名称'].first()
    base[schema.ORIGINAL_RECORD_FIELD] = records_per_id
    base['最早数据更新时间'] = id_index['数据更新时间'].min()
    base['最新数据更新时间'] = id_index['数据更新时间'].max()

    detail_frames = []
    overview_rows = []
    duplicate_groups = int((records_per_id > 1).sum())
    for field in conflict_fields:
        target_ids = multi_value_ids[field]
        if len(target_ids) == 0:
            continue
        sub = frame[frame[schema.ID_FIELD].isin(target_ids)]
        grouped = sub[field].map(canonicalize_value).groupby(sub[schema.ID_FIELD])
        nunique_series = grouped.nunique(dropna=True)
        detail = pd.DataFrame({
            '唯一值数量': nunique_series,
            '不同值列表': grouped.agg(
                lambda s: sorted({format_canonical_value(item) for item in s.dropna()})),
        })
        detail = detail.join(base, how='left').reset_index()
        detail.insert(1, '字段名称', field)
        detail['不同值列表'] = detail['不同值列表'].map(
            lambda values: '、'.join(str(v) for v in values) if isinstance(values, list) else '')
        detail_frames.append(detail)
        overview_rows.append({
            '字段名称': field,
            '冲突岗位组数量': int(len(target_ids)),
            '涉及原始记录数': int(frame[schema.ID_FIELD].isin(target_ids).sum()),
            '多值岗位组比例': round(len(target_ids) / duplicate_groups, 6) if duplicate_groups else 0.0,
            '最大唯一值数量': int(nunique_series.max()),
            '是否关键业务字段': '是' if field in schema.KEY_CONFLICT_FIELDS else '否',
        })

    detail_table = (pd.concat(detail_frames, ignore_index=True)
                    if detail_frames else pd.DataFrame(
                        columns=[schema.ID_FIELD, '字段名称', '唯一值数量', '不同值列表',
                                 '岗位标题', '公司名称', schema.ORIGINAL_RECORD_FIELD,
                                 '最早数据更新时间', '最新数据更新时间']))
    overview = (pd.DataFrame(overview_rows)
                .sort_values('冲突岗位组数量', ascending=False)
                .reset_index(drop=True) if overview_rows else pd.DataFrame(
                    columns=['字段名称', '冲突岗位组数量', '涉及原始记录数',
                             '多值岗位组比例', '最大唯一值数量', '是否关键业务字段']))
    return overview, detail_table


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


def _value_blocks(values, moments) -> list:
    """按时间顺序把取值序列切分为「取值块」（同一取值连续出现算一块）。"""
    blocks: list = []
    previous = None
    for value, moment in zip(values, moments):
        if not blocks or value != previous:
            blocks.append({'取值': value, '条数': 0, '首次时间': moment, '末次时间': moment})
            previous = value
        blocks[-1]['条数'] += 1
        blocks[-1]['末次时间'] = moment
    return blocks


def build_manual_review_queue(frame: pd.DataFrame, unique_frame: pd.DataFrame,
                              review_fields=None,
                              time_field: str = '数据更新时间') -> tuple:
    """构造「版本漂移冲突」人工复核队列。

    针对同 ID 下多值的关注字段（薪资信息 / 公司名称 / 工作城市 / 工作地址），
    逐（岗位ID, 字段）输出取值明细、时间分段数与是否时间连续切换，供人工判定
    「是同一岗位的页面版本变化」还是「两个岗位被错误合并」。

    本函数只读取数据、不修改主唯一岗位表。
    """
    review_fields = list(review_fields or schema.REVIEW_FIELDS)
    id_field = schema.ID_FIELD
    nunique_by_field = {field: _nunique_per_id(frame, field) for field in review_fields}
    review_ids = set()
    for field in review_fields:
        series = nunique_by_field[field]
        review_ids |= set(series.index[series > 1])
    representatives = unique_frame.set_index(id_field)

    detail_rows, block_rows = [], []
    for job_id in sorted(review_ids):
        sub = frame.loc[frame[id_field] == job_id].sort_values(time_field, kind='stable')
        for field in review_fields:
            values = sub[field].fillna('').astype(str)
            if values.nunique() <= 1:
                continue
            blocks = _value_blocks(values.tolist(), sub[time_field].astype(str).tolist())
            counts = values.value_counts()
            majority_value = str(counts.idxmax())
            picked = str(representatives.at[job_id, field]) if field in representatives.columns else ''
            is_continuous = len(blocks) == int(values.nunique())
            detail_rows.append({
                id_field: job_id,
                '岗位标题': representatives.at[job_id, '岗位标题'],
                '公司名称': representatives.at[job_id, '公司名称'],
                '冲突字段': field,
                '不同取值数': int(values.nunique()),
                '原始记录数': int(representatives.at[job_id, schema.ORIGINAL_RECORD_FIELD]),
                '取值块数': len(blocks),
                '是否时间连续切换': '是' if is_continuous else '否（存在交错，需重点核查）',
                '多数取值': majority_value,
                '多数取值条数': int(counts.max()),
                '代表记录取值': picked,
                '代表记录是否取到多数值': '是' if picked == majority_value else '否',
                '取值明细': '；'.join(f'{key}×{value}' for key, value in counts.items()),
                '最早数据更新时间': sub[time_field].min(),
                '最新数据更新时间': sub[time_field].max(),
                schema.CONFLICT_COUNT_FIELD: int(representatives.at[job_id, schema.CONFLICT_COUNT_FIELD]),
                schema.KEY_CONFLICT_FLAG_FIELD: int(representatives.at[
                    job_id, schema.KEY_CONFLICT_FLAG_FIELD]),
            })
            total = int(len(values))
            for order, block in enumerate(blocks, start=1):
                block_rows.append({
                    id_field: job_id,
                    '冲突字段': field,
                    '段序号': order,
                    '取值': block['取值'],
                    '条数': block['条数'],
                    '占该岗位该字段记录比例': round(block['条数'] / total, 6),
                    '首次数据更新时间': block['首次时间'],
                    '末次数据更新时间': block['末次时间'],
                })

    detail_table = pd.DataFrame(detail_rows, columns=[
        id_field, '岗位标题', '公司名称', '冲突字段', '不同取值数', '原始记录数', '取值块数',
        '是否时间连续切换', '多数取值', '多数取值条数', '代表记录取值', '代表记录是否取到多数值',
        '取值明细', '最早数据更新时间', '最新数据更新时间',
        schema.CONFLICT_COUNT_FIELD, schema.KEY_CONFLICT_FLAG_FIELD])
    block_table = pd.DataFrame(block_rows, columns=[
        id_field, '冲突字段', '段序号', '取值', '条数', '占该岗位该字段记录比例',
        '首次数据更新时间', '末次数据更新时间'])
    interleaved = int((detail_table['是否时间连续切换'] != '是').sum()) if len(detail_table) else 0
    summary = {
        'review_job_count': int(len(review_ids)),
        'review_row_count': int(len(detail_table)),
        'review_block_count': int(len(block_table)),
        'review_field_count': int(detail_table['冲突字段'].nunique()) if len(detail_table) else 0,
        'review_continuous_rows': int(len(detail_table)) - interleaved,
        'review_interleaved_rows': interleaved,
        'review_minority_picked_rows': (int((detail_table['代表记录是否取到多数值'] == '否').sum())
                                        if len(detail_table) else 0),
    }
    return detail_table, block_table, summary


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

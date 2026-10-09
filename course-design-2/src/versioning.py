# -*- coding: utf-8 -*-
"""岗位版本时序模块：签名、观测快照折叠、版本序列、变化事件与最终版本选择。

三层建模：

1. 观测快照层：同一 ``实习岗位ID + 数据更新时间`` 下的搜索来源重复被折叠，得到
   一个「岗位在某观测时点的业务状态」；
2. 版本时序层：按观测时间排序后，对连续相同的状态签名做压缩，得到核心业务版本
   与完整页面版本；
3. 变化事件层：相邻核心版本逐字段比较，输出字段级变化事件。

核心约定：

- 版本的本质是「连续时间段内的业务状态」，因此 ``A → B → A`` 必须保留为三个版本；
- 签名必须是稳定、可复现的（稳定 JSON + SHA256），禁止使用 Python 内存 hash；
- 所有输出排序固定，重复运行结果一致。
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime

import numpy as np
import pandas as pd

from . import schema

# 签名序列化使用的固定分隔符（保证稳定 JSON 文本可复现）
_JSON_SEPARATORS = (',', ':')


def _json_default(value):
    """JSON 默认序列化：datetime / numpy 标量统一转稳定文本或原生类型。"""
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.floating):
        number = float(value)
        return int(number) if number.is_integer() else number
    return str(value)


def _stable_text(value) -> str:
    """取值的稳定 JSON 文本（用于排序键与签名）。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)


def canonicalize_business_value(value):
    """把业务字段取值归一化为稳定、可比较、可哈希的规范形式。

    规则：
    - 空值（None / NaN / NaT / 空字符串）统一归一为 ``None``；
    - 字符串去除无意义首尾空格；
    - list / tuple / set 去空、去重、固定排序，返回 tuple（可哈希，JSON 序列化为数组）；
    - dict 按 key 固定排序；
    - numpy 标量转为原生类型；
    - 整数值的浮点表示归一为 int。
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, np.ndarray):
        return canonicalize_business_value(value.tolist())
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [canonicalize_business_value(item) for item in value]
        unique: dict = {}
        for item in items:
            if item is None:
                continue
            unique.setdefault(_stable_text(item), item)
        return tuple(unique[key] for key in sorted(unique))
    if isinstance(value, dict):
        return {str(key): canonicalize_business_value(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, np.generic):
        return canonicalize_business_value(value.item())
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def render_value(value) -> str:
    """把取值渲染为可读文本（list/tuple/dict 渲染为 JSON，空值渲染为空串）。"""
    canonical = canonicalize_business_value(value)
    if canonical is None:
        return ''
    if isinstance(canonical, tuple):
        return _stable_text(list(canonical))
    if isinstance(canonical, (list, dict)):
        return _stable_text(canonical)
    return str(canonical)


def build_signature(values: dict, fields) -> str:
    """构造字段集合的稳定签名：先稳定 JSON，再 SHA256。"""
    payload = {field: canonicalize_business_value(values.get(field)) for field in sorted(fields)}
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      default=_json_default, separators=_JSON_SEPARATORS)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def build_signature_series(frame: pd.DataFrame, fields, name: str = 'signature') -> pd.Series:
    """整列构造签名，保证同一输入产生完全一致的签名序列。"""
    ordered = sorted(fields)
    values = frame[ordered].to_numpy(dtype=object)
    signatures = []
    for row in values:
        payload = {field: canonicalize_business_value(value)
                   for field, value in zip(ordered, row)}
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                          default=_json_default, separators=_JSON_SEPARATORS)
        signatures.append(hashlib.sha256(text.encode('utf-8')).hexdigest())
    return pd.Series(signatures, index=frame.index, name=name)


def _canonical_fields_series(frame: pd.DataFrame, fields) -> dict:
    """逐字段归一化，返回 {字段: 归一化 ndarray}。"""
    return {field: np.array([canonicalize_business_value(value)
                             for value in frame[field].to_numpy(dtype=object)], dtype=object)
            for field in fields}


def collapse_observation_snapshots(frame: pd.DataFrame, url_field: str,
                                   business_fields=None) -> tuple:
    """折叠同一「岗位ID + 观测时间」下的搜索来源重复，生成观测快照。

    返回（快照表, 同时间点业务冲突审计表, 统计字典）。

    同时间点业务字段完全一致 → 折叠为 1 行快照；
    存在差异 → 记录到冲突审计表（不随机取值、不静默覆盖）。
    """
    business_fields = list(business_fields or schema.BUSINESS_FIELDS)
    id_field = schema.ID_FIELD
    time_field = schema.OBSERVATION_TIME_FIELD
    working = frame.copy()
    working['_原始顺序'] = np.arange(len(working), dtype='int64')
    working['_业务签名'] = build_signature_series(working, business_fields, '_业务签名')
    working = working.sort_values(
        [id_field, time_field, '数据创建时间', '_原始顺序'], kind='stable')

    grouped = working.groupby([id_field, time_field], sort=True)
    stats = pd.DataFrame({
        schema.SNAPSHOT_RECORD_COUNT_FIELD: grouped.size(),
        schema.SNAPSHOT_GROUP_COUNT_FIELD: grouped[schema.CATEGORY_GROUP_FIELD].nunique(dropna=True),
        schema.SNAPSHOT_ITEM_COUNT_FIELD: grouped[schema.CATEGORY_ITEM_FIELD].nunique(dropna=True),
        '_业务状态数': grouped['_业务签名'].nunique(dropna=False),
        '数据创建时间': grouped['数据创建时间'].min(),
    })

    # 同时间点业务字段一致时，组内取值归一化后完全等价；冲突组进入审计表。
    # 取排序后的组内首行作为快照取值，不丢弃任何冲突组。
    first_rows = working.groupby([id_field, time_field], sort=True).head(1)
    snapshot = first_rows[[id_field, url_field, time_field, *business_fields]].copy()
    snapshot = snapshot.rename(columns={url_field: schema.URL_FIELD})
    snapshot = snapshot.merge(
        stats.reset_index()[[id_field, time_field,
                             schema.SNAPSHOT_RECORD_COUNT_FIELD,
                             schema.SNAPSHOT_GROUP_COUNT_FIELD,
                             schema.SNAPSHOT_ITEM_COUNT_FIELD, '数据创建时间']],
        on=[id_field, time_field], how='left')
    snapshot[schema.URL_FIELD] = snapshot[schema.URL_FIELD].fillna('')
    snapshot['数据更新时间'] = snapshot[time_field]
    snapshot = snapshot[schema.SNAPSHOT_COLUMNS]
    snapshot = snapshot.sort_values(
        [id_field, schema.OBSERVATION_TIME_FIELD, '数据创建时间'], kind='stable'
    ).reset_index(drop=True)

    conflicts = _build_observation_conflicts(working, stats, business_fields)
    summary = {
        'raw_rows': int(len(frame)),
        'snapshot_rows': int(len(snapshot)),
        'collapsed_rows': int(len(frame) - len(snapshot)),
        'observation_jobs': int(snapshot[id_field].nunique()),
        'same_time_conflict_groups': int(conflicts.shape[0]),
    }
    return snapshot, conflicts, summary


def _build_observation_conflicts(working: pd.DataFrame, stats: pd.DataFrame,
                                 business_fields) -> pd.DataFrame:
    """同时间点（岗位ID + 观测时间）业务字段不一致的审计明细。"""
    id_field = schema.ID_FIELD
    time_field = schema.OBSERVATION_TIME_FIELD
    conflict_index = stats.index[stats['_业务状态数'] > 1]
    columns = [id_field, time_field, schema.SNAPSHOT_RECORD_COUNT_FIELD, '冲突字段数',
               '差异字段列表', '岗位标题', '公司名称']
    if len(conflict_index) == 0:
        return pd.DataFrame(columns=columns)

    canonical = _canonical_fields_series(working, business_fields)
    rows = []
    for job_id, moment in conflict_index:
        mask = ((working[id_field] == job_id) & (working[time_field] == moment)).to_numpy()
        positions = np.flatnonzero(mask)
        differing = []
        for field in business_fields:
            values = {canonical[field][position] for position in positions}
            if len(values) > 1:
                differing.append(field)
        first = positions[0]
        rows.append({
            id_field: job_id,
            time_field: moment,
            schema.SNAPSHOT_RECORD_COUNT_FIELD: int(len(positions)),
            '冲突字段数': len(differing),
            '差异字段列表': json.dumps(differing, ensure_ascii=False),
            '岗位标题': canonical['岗位标题'][first],
            '公司名称': canonical['公司名称'][first],
        })
    return pd.DataFrame(rows, columns=columns)


def assign_sequential_versions(frame: pd.DataFrame, sort_fields, signature_field: str,
                               version_field: str) -> pd.DataFrame:
    """按观测顺序对连续相同签名压缩，生成版本号（A→B→A 保留为三段）。"""
    ordered = frame.sort_values(list(sort_fields), kind='stable').copy()
    id_field = schema.ID_FIELD
    grouped = ordered.groupby(id_field, sort=False)[signature_field]
    ordered[version_field] = grouped.transform(
        lambda series: series.ne(series.shift()).cumsum()).astype('int64')
    return ordered


def assign_snapshot_versions(snapshots: pd.DataFrame, core_fields=None,
                             full_fields=None) -> pd.DataFrame:
    """计算观测快照的核心/完整页面版本号

    排序字段：观测时间 → 数据创建时间 → 稳定原始顺序。
    """
    core_fields = list(core_fields or schema.CORE_SIGNATURE_FIELDS)
    full_fields = list(full_fields or schema.FULL_SIGNATURE_FIELDS)
    working = snapshots.copy()
    working['_版本顺序'] = np.arange(len(working), dtype='int64')
    working[schema.CORE_SIGNATURE_FIELD] = build_signature_series(
        working, core_fields, schema.CORE_SIGNATURE_FIELD)
    working[schema.FULL_SIGNATURE_FIELD] = build_signature_series(
        working, full_fields, schema.FULL_SIGNATURE_FIELD)
    sort_fields = [schema.ID_FIELD, schema.OBSERVATION_TIME_FIELD, '数据创建时间', '_版本顺序']
    working = assign_sequential_versions(
        working, sort_fields, schema.CORE_SIGNATURE_FIELD, schema.CORE_VERSION_FIELD)
    working = assign_sequential_versions(
        working, sort_fields, schema.FULL_SIGNATURE_FIELD, schema.FULL_VERSION_FIELD)
    return working


def build_version_history(snapshots: pd.DataFrame, core_fields=None,
                          full_fields=None) -> tuple:
    """构建岗位版本时序表。

    返回（版本历史, 带核心/完整版本号的快照表, 变化事件表, 统计字典）。
    """
    core_fields = list(core_fields or schema.CORE_SIGNATURE_FIELDS)
    full_fields = list(full_fields or schema.FULL_SIGNATURE_FIELDS)
    id_field = schema.ID_FIELD
    time_field = schema.OBSERVATION_TIME_FIELD

    working = assign_snapshot_versions(snapshots, core_fields, full_fields)

    grouped = working.groupby([id_field, schema.CORE_VERSION_FIELD], sort=True)
    aggregates = grouped.agg(**{
        schema.VERSION_FIRST_TIME_FIELD: (time_field, 'min'),
        schema.VERSION_LAST_TIME_FIELD: (time_field, 'max'),
        schema.VERSION_OBS_COUNT_FIELD: (time_field, 'size'),
        schema.VERSION_FULL_COUNT_FIELD: (schema.FULL_VERSION_FIELD, 'nunique'),
        schema.FULL_VERSION_FIELD: (schema.FULL_VERSION_FIELD, 'max'),
        schema.CORE_SIGNATURE_FIELD: (schema.CORE_SIGNATURE_FIELD, 'first'),
        schema.FULL_SIGNATURE_FIELD: (schema.FULL_SIGNATURE_FIELD, 'last'),
    }).reset_index()

    last_state = working.groupby([id_field, schema.CORE_VERSION_FIELD], sort=True).tail(1)
    last_state = last_state[[id_field, schema.CORE_VERSION_FIELD, schema.URL_FIELD,
                             *full_fields]].reset_index(drop=True)
    versions = aggregates.merge(last_state, on=[id_field, schema.CORE_VERSION_FIELD], how='left')
    versions = versions.sort_values([id_field, schema.CORE_VERSION_FIELD], kind='stable')

    versions[schema.VERSION_NEXT_TIME_FIELD] = versions.groupby(id_field, sort=False)[
        schema.VERSION_FIRST_TIME_FIELD].shift(-1)
    versions[schema.VERSION_DURATION_FIELD] = (
        (versions[schema.VERSION_LAST_TIME_FIELD] - versions[schema.VERSION_FIRST_TIME_FIELD])
        .dt.total_seconds() / 86400).round(6)
    versions[schema.VERSION_GAP_FIELD] = (
        (versions[schema.VERSION_NEXT_TIME_FIELD] - versions[schema.VERSION_LAST_TIME_FIELD])
        .dt.total_seconds() / 86400).round(6)
    versions[schema.VERSION_IS_FINAL_FIELD] = (
        versions[schema.CORE_VERSION_FIELD]
        == versions.groupby(id_field, sort=False)[schema.CORE_VERSION_FIELD].transform('max')
    ).astype('int64')

    versions, change_events = build_change_events(versions, core_fields)
    versions = versions[schema.VERSION_HISTORY_COLUMNS].reset_index(drop=True)

    summary = {
        'core_versions': int(len(versions)),
        'full_versions': int(versions.groupby(id_field)[schema.FULL_VERSION_FIELD].max().sum()),
        'observation_snapshots': int(len(working)),
        'change_events': int(len(change_events)),
    }
    return versions, working, change_events, summary


def compare_versions(previous_values: dict, current_values: dict, fields) -> list:
    """逐字段比较两套已归一化取值，返回发生变化的字段列表。"""
    return [field for field in fields if previous_values[field] != current_values[field]]


def build_change_events(versions: pd.DataFrame, core_fields=None) -> tuple:
    """计算相邻核心版本的变化字段、重点标志，并生成字段级变化事件表。"""
    core_fields = list(core_fields or schema.CORE_SIGNATURE_FIELDS)
    id_field = schema.ID_FIELD
    version_field = schema.CORE_VERSION_FIELD
    ordered = versions.sort_values([id_field, version_field], kind='stable').reset_index(drop=True)
    canonical = _canonical_fields_series(ordered, core_fields)
    version_numbers = ordered[version_field].to_numpy()
    job_ids = ordered[id_field].to_numpy(dtype=object)

    changed_counts = np.zeros(len(ordered), dtype='int64')
    changed_lists: list = []
    flag_arrays = {name: np.zeros(len(ordered), dtype='int64')
                   for name in schema.VERSION_CHANGE_FLAG_FIELDS}
    event_rows = []

    previous_values = None
    previous_job = None
    for position in range(len(ordered)):
        current_values = {field: canonical[field][position] for field in core_fields}
        # 跨岗位不得比较：新岗位的第一个核心版本视为无上一版本
        if previous_values is None or job_ids[position] != previous_job:
            changed = []
        else:
            changed = compare_versions(previous_values, current_values, core_fields)
        changed_counts[position] = len(changed)
        changed_lists.append(json.dumps(changed, ensure_ascii=False))
        for name, fields in schema.CHANGE_FLAG_FIELDS.items():
            if any(field in changed for field in fields):
                flag_arrays[name][position] = 1
        if changed:
            for field in changed:
                event_rows.append({
                    id_field: job_ids[position],
                    '旧核心版本号': int(version_numbers[position] - 1),
                    '新核心版本号': int(version_numbers[position]),
                    '变化时间': ordered.at[position, schema.VERSION_FIRST_TIME_FIELD],
                    '变化字段': field,
                    '旧值': render_value(previous_values[field]),
                    '新值': render_value(current_values[field]),
                    '岗位标题': render_value(current_values['岗位标题']),
                    '公司名称': render_value(current_values['公司名称']),
                    '工作城市': render_value(current_values['工作城市']),
                })
        previous_values = current_values
        previous_job = job_ids[position]

    ordered[schema.VERSION_CHANGED_COUNT_FIELD] = changed_counts
    ordered[schema.VERSION_CHANGED_LIST_FIELD] = changed_lists
    for name, array in flag_arrays.items():
        ordered[name] = array

    events = pd.DataFrame(event_rows, columns=schema.CHANGE_EVENT_COLUMNS)
    if len(events):
        events = events.sort_values([id_field, '新核心版本号', '变化字段'], kind='stable'
                                    ).reset_index(drop=True)
    return ordered, events


def select_final_version(versions: pd.DataFrame) -> pd.DataFrame:
    """取出每个岗位的最终核心版本。"""
    return versions.loc[versions[schema.VERSION_IS_FINAL_FIELD] == 1].reset_index(drop=True)


def field_multi_value_ids(frame: pd.DataFrame, field: str) -> set:
    """返回该字段在同一岗位下存在多个不同取值的岗位ID集合（归一化后比较）。"""
    id_field = schema.ID_FIELD
    canonical = frame[field].map(canonicalize_business_value)
    nunique = canonical.groupby(frame[id_field]).nunique(dropna=True)
    return set(nunique.index[nunique > 1])


def field_multi_value_stats(frame: pd.DataFrame, fields, id_field: str | None = None) -> pd.DataFrame:
    """按字段统计「同一岗位历史多值」的岗位数（统计范围与最终实体冲突统计一致）。"""
    id_field = id_field or schema.ID_FIELD
    rows = []
    for field in fields:
        canonical = frame[field].map(canonicalize_business_value)
        nunique = canonical.groupby(frame[id_field]).nunique(dropna=True)
        multi = nunique[nunique > 1]
        rows.append({
            '字段名称': field,
            '同岗位多值岗位数': int(len(multi)),
            '涉及观测快照数': int(frame[id_field].isin(multi.index).sum()),
            '最大唯一值数量': int(nunique.max()) if len(nunique) else 0,
            '是否核心业务签名字段': '是' if field in schema.CORE_SIGNATURE_FIELDS else '否',
        })
    return (pd.DataFrame(rows).sort_values(['同岗位多值岗位数', '字段名称'], ascending=[False, True])
            .reset_index(drop=True))


def version_job_summary(versions: pd.DataFrame) -> pd.DataFrame:
    """按岗位汇总版本摘要：核心/完整版本数、是否多版本、历史变化标志。"""
    id_field = schema.ID_FIELD
    grouped = versions.groupby(id_field, sort=True)
    summary = pd.DataFrame({
        schema.ENTITY_CORE_VERSION_COUNT_FIELD: grouped[schema.CORE_VERSION_FIELD].max(),
        schema.ENTITY_FULL_VERSION_COUNT_FIELD: grouped[schema.FULL_VERSION_FIELD].max(),
    })
    summary[schema.ENTITY_MULTI_VERSION_FIELD] = (
        summary[schema.ENTITY_CORE_VERSION_COUNT_FIELD] > 1).astype('int64')
    for flag_field, entity_field in schema.VERSION_FLAG_TO_ENTITY_FIELD.items():
        summary[entity_field] = grouped[flag_field].max().astype('int64')
    return summary


def find_aba_jobs(versions: pd.DataFrame) -> pd.DataFrame:
    """统计「同一核心业务状态再次出现」的岗位（A→B→A 回退，正常保留不合并）。"""
    id_field = schema.ID_FIELD
    signature_field = schema.CORE_SIGNATURE_FIELD
    version_field = schema.CORE_VERSION_FIELD
    rows = []
    for job_id, group in versions.groupby(id_field, sort=True):
        ordered = group.sort_values(version_field, kind='stable')
        signatures = ordered[signature_field].tolist()
        first_seen: dict = {}
        repeats = 0
        for position, signature in enumerate(signatures, start=1):
            if signature in first_seen:
                repeats += 1
            else:
                first_seen[signature] = position
        if repeats:
            rows.append({
                id_field: job_id,
                '核心版本数': int(len(signatures)),
                '重复出现状态次数': int(repeats),
                '回退版本号列表': json.dumps(
                    [position for position, signature in enumerate(signatures, start=1)
                     if first_seen.get(signature, position) != position], ensure_ascii=False),
                '版本签名序列': json.dumps(
                    [signature[:12] for signature in signatures], ensure_ascii=False),
                '岗位标题': render_value(ordered['岗位标题'].iloc[-1]),
                '公司名称': render_value(ordered['公司名称'].iloc[-1]),
            })
    return pd.DataFrame(rows, columns=[id_field, '核心版本数', '重复出现状态次数',
                                       '回退版本号列表', '版本签名序列', '岗位标题', '公司名称'])


def check_version_time_order(versions: pd.DataFrame, snapshots: pd.DataFrame,
                             versioned_snapshots: pd.DataFrame | None = None) -> pd.DataFrame:
    """版本时间与版本号异常检查（倒序 / 负间隔 / 版本号断裂）。

    ``versioned_snapshots`` 用于校验观测级完整页面版本号的连续性；
    未提供时退化为仅校验核心版本层。
    """
    id_field = schema.ID_FIELD
    version_field = schema.CORE_VERSION_FIELD
    full_field = schema.FULL_VERSION_FIELD
    ordered = versions.sort_values([id_field, version_field], kind='stable')
    issues = []

    def _add(kind: str, affected_ids, detail: str, limit: int = 50) -> None:
        ids = list(affected_ids)
        if not ids:
            return
        for job_id in ids[:limit]:
            issues.append({'异常类型': kind, id_field: job_id,
                           '说明': f'{detail}（共 {len(ids)} 个岗位）'})

    _add('版本时间长于末次观测', ordered.loc[
        ordered[schema.VERSION_FIRST_TIME_FIELD] > ordered[schema.VERSION_LAST_TIME_FIELD],
        id_field].unique(), '版本首次观测时间 > 版本末次观测时间')
    _add('下一版本早于本版本结束', ordered.loc[
        ordered[schema.VERSION_NEXT_TIME_FIELD].notna()
        & (ordered[schema.VERSION_NEXT_TIME_FIELD] < ordered[schema.VERSION_LAST_TIME_FIELD]),
        id_field].unique(), '下一版本开始时间 < 当前版本末次观测时间')
    _add('负持续时间', ordered.loc[ordered[schema.VERSION_DURATION_FIELD] < 0, id_field].unique(),
         '已观测持续时间_天 < 0')
    _add('负版本间隔', ordered.loc[ordered[schema.VERSION_GAP_FIELD] < 0, id_field].unique(),
         '到下一版本间隔_天 < 0')

    bounds = ordered.groupby(id_field)[version_field].agg(['min', 'max', 'nunique'])
    broken = bounds[(bounds['min'] != 1) | (bounds['max'] != bounds['nunique'])]
    _add('核心版本号断裂', broken.index, '同一岗位核心版本号存在跳号或重复')

    full_max = ordered.groupby(id_field)[full_field].max()
    fewer = full_max[full_max < bounds['max'].reindex(full_max.index)]
    _add('完整版本数少于核心版本数', fewer.index, '完整版本数 < 核心版本数')

    source = versioned_snapshots if versioned_snapshots is not None else ordered
    if full_field in source.columns:
        full_bounds = source.groupby(id_field)[full_field].agg(['min', 'max', 'nunique'])
        full_broken = full_bounds[(full_bounds['min'] != 1)
                                  | (full_bounds['max'] != full_bounds['nunique'])]
        _add('完整版本号断裂', full_broken.index, '观测级完整页面版本号存在跳号')

    snapshot_jobs = set(snapshots[id_field].unique())
    version_jobs = set(versions[id_field].unique())
    if snapshot_jobs != version_jobs:
        issues.append({'异常类型': '岗位ID集合不一致', id_field: '',
                       '说明': f'快照岗位 {len(snapshot_jobs)} / 版本岗位 {len(version_jobs)}'})

    return pd.DataFrame(issues, columns=['异常类型', id_field, '说明'])

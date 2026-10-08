# -*- coding: utf-8 -*-
"""公司简介版本化模块：快照层 → 版本层 → 变化事件层。

层次结构（提示词第 23 节）：

```text
company_entity_map
      ↓
company_profile_snapshots     一行 = 一个正式公司实体 × 一个观测时间
      ↓
company_profile_history       只取 CONSISTENT 快照，连续相同简介压缩为一个版本
      ↓
company_text_change_events    相邻版本之间的企业简介语义变化事件
```

核心规则：

1. 同一公司同一时间存在多个非空简介 → ``AMBIGUOUS``，**禁止**用多数表决直接生成正式版本；
2. 只有 ``CONSISTENT`` 快照允许进入 ``company_profile_history``；
3. ``MISSING``（无任何非空简介）不得生成伪版本；
4. A → B → A 必须保留为 V1 → V2 → V3，禁止合并分离的相同状态；
5. 跨越 AMBIGUOUS 时间点的相邻版本，事件必须记录跨越歧义快照数量。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import company_identity, schema, text_utils

# 快照候选列表中的分隔符（写 Excel 时人类可读，Parquet 中保持 list）
CANDIDATE_SEPARATOR = '␟'


def _unique_sorted(values) -> list:
    """把一列列表值合并为去重排序后的列表（确定性输出）。"""
    merged: set = set()
    for value in values:
        merged.update(str(item) for item in text_utils.as_list(value) if str(item).strip())
    return sorted(merged)


def _aggregate_auxiliary(values, default='') -> str:
    """辅助状态字段聚合：出现次数最多者优先，并列时字典序最小（保证确定性）。"""
    counts: dict = {}
    for value in values:
        text = '' if value is None else str(value).strip()
        if not text:
            continue
        counts[text] = counts.get(text, 0) + 1
    if not counts:
        return default
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def build_company_profile_snapshots(mapped: pd.DataFrame) -> pd.DataFrame:
    """构建公司简介快照层（一行 = 公司实体 × 观测时间）。

    ``mapped`` 必须包含：company_entity_id、观测时间、公司简介_清洗、实习岗位ID
    以及辅助状态字段（公司标签列表、所属行业、公司性质、公司规模、所在地集合）。
    快照状态：CONSISTENT（唯一非空简介）/ AMBIGUOUS（多个并行简介）/ MISSING（无简介）。
    """
    entity_field = schema.COMPANY_ENTITY_ID_FIELD
    time_field = schema.OBSERVATION_TIME_FIELD
    profile_field = schema.COMPANY_PROFILE_CLEAN_FIELD
    id_field = schema.ID_FIELD
    if mapped.empty:
        return pd.DataFrame(columns=schema.COMPANY_PROFILE_SNAPSHOT_COLUMNS)

    frame = mapped.copy()
    frame[profile_field] = frame[profile_field].fillna('').astype(str).str.strip()
    if schema.COMPANY_LOCATION_SET_FIELD in frame.columns:
        location_column = schema.COMPANY_LOCATION_SET_FIELD
    elif '公司所在地' in frame.columns:
        location_column = '公司所在地'
    else:
        location_column = None
    keys = [entity_field, time_field]

    base = (frame.groupby(keys, as_index=False, sort=True)
            .agg(**{schema.COMPANY_SNAPSHOT_JOB_COUNT_FIELD: (id_field, 'nunique'),
                    schema.COMPANY_SNAPSHOT_OBS_COUNT_FIELD: (id_field, 'size')}))
    aux = (frame.groupby(keys, as_index=False, sort=True)
           .agg(公司标签列表_合并=(schema.COMPANY_TAG_LIST_FIELD, _unique_sorted),
                所属行业=('所属行业', _aggregate_auxiliary),
                公司性质=('公司性质', _aggregate_auxiliary),
                公司规模=('公司规模', _aggregate_auxiliary),
                来源岗位集合=(id_field, lambda values: sorted({str(v) for v in values})),
                所在地集合=(location_column, _unique_sorted) if location_column
                else (id_field, lambda values: [])))
    snapshots = base.merge(aux, on=keys, how='left')

    non_empty = frame[frame[profile_field] != '']
    candidates = (non_empty.groupby([*keys, profile_field], as_index=False, sort=True)
                  .agg(候选支持岗位数=(id_field, 'nunique')))
    candidate_count = (candidates.groupby(keys, as_index=False, sort=True)
                       .agg(**{schema.COMPANY_SNAPSHOT_CANDIDATE_COUNT_FIELD:
                               (profile_field, 'size')}))
    snapshots = snapshots.merge(candidate_count, on=keys, how='left')
    snapshots[schema.COMPANY_SNAPSHOT_CANDIDATE_COUNT_FIELD] = (
        snapshots[schema.COMPANY_SNAPSHOT_CANDIDATE_COUNT_FIELD].fillna(0).astype(int))

    ordered = candidates.sort_values([*keys, '候选支持岗位数', profile_field],
                                     ascending=[True, True, False, True])
    main = (ordered.groupby(keys, as_index=False, sort=True).first()[
        [*keys, profile_field, '候选支持岗位数']]
        .rename(columns={profile_field: schema.COMPANY_SNAPSHOT_MAIN_PROFILE_FIELD,
                         '候选支持岗位数': schema.COMPANY_SNAPSHOT_MAIN_SUPPORT_FIELD}))
    snapshots = snapshots.merge(main, on=keys, how='left')
    snapshots[schema.COMPANY_SNAPSHOT_MAIN_PROFILE_FIELD] = snapshots[
        schema.COMPANY_SNAPSHOT_MAIN_PROFILE_FIELD].fillna('')
    snapshots[schema.COMPANY_SNAPSHOT_MAIN_SUPPORT_FIELD] = snapshots[
        schema.COMPANY_SNAPSHOT_MAIN_SUPPORT_FIELD].fillna(0).astype(int)

    listed = (candidates.sort_values([*keys, profile_field], ascending=True)
              .groupby(keys, as_index=False, sort=True)
              .agg(**{schema.COMPANY_SNAPSHOT_CANDIDATE_LIST_FIELD: (profile_field, list),
                      schema.COMPANY_SNAPSHOT_CANDIDATE_SUPPORT_FIELD: ('候选支持岗位数', list)}))
    snapshots = snapshots.merge(listed, on=keys, how='left')

    count = snapshots[schema.COMPANY_SNAPSHOT_CANDIDATE_COUNT_FIELD]
    snapshots[schema.COMPANY_SNAPSHOT_STATE_FIELD] = np.where(
        count == 0, 'MISSING', np.where(count == 1, 'CONSISTENT', 'AMBIGUOUS'))
    snapshots[schema.COMPANY_SNAPSHOT_FORMAL_FIELD] = (
        snapshots[schema.COMPANY_SNAPSHOT_STATE_FIELD] == 'CONSISTENT').astype(int)
    snapshots[schema.COMPANY_SNAPSHOT_MAIN_SUPPORT_RATE_FIELD] = (
        snapshots[schema.COMPANY_SNAPSHOT_MAIN_SUPPORT_FIELD]
        / snapshots[schema.COMPANY_SNAPSHOT_JOB_COUNT_FIELD]).round(6)
    snapshots['公司规范名称'] = snapshots[entity_field]
    snapshots[schema.COMPANY_TAG_LIST_FIELD] = snapshots.pop('公司标签列表_合并')
    for column in (schema.COMPANY_SNAPSHOT_CANDIDATE_LIST_FIELD,
                   schema.COMPANY_SNAPSHOT_CANDIDATE_SUPPORT_FIELD):
        snapshots[column] = snapshots[column].map(text_utils.as_list)

    for column in schema.COMPANY_PROFILE_SNAPSHOT_COLUMNS:
        if column not in snapshots.columns:
            snapshots[column] = None
    return snapshots[schema.COMPANY_PROFILE_SNAPSHOT_COLUMNS].sort_values(
        [entity_field, time_field]).reset_index(drop=True)


def validate_company_snapshot_uniqueness(snapshots: pd.DataFrame) -> dict:
    """断言 company_entity_id + 观测时间 在快照层严格唯一。"""
    if snapshots.empty:
        return {'唯一': True, '行数': 0, '重复键数': 0}
    keys = snapshots[[schema.COMPANY_ENTITY_ID_FIELD, schema.OBSERVATION_TIME_FIELD]]
    duplicates = int(keys.duplicated().sum())
    return {'唯一': duplicates == 0, '行数': int(len(snapshots)), '重复键数': duplicates}


def build_company_profile_history(snapshots: pd.DataFrame) -> tuple:
    """只取 CONSISTENT 快照，按时间升序压缩连续相同简介为公司简介版本。"""
    entity_field = schema.COMPANY_ENTITY_ID_FIELD
    time_field = schema.OBSERVATION_TIME_FIELD
    version_field = schema.COMPANY_PROFILE_VERSION_FIELD
    # CONSISTENT 快照只有唯一非空简介，主候选简介即该快照的正式简介
    profile_field = schema.COMPANY_SNAPSHOT_MAIN_PROFILE_FIELD
    if snapshots.empty:
        return pd.DataFrame(columns=schema.COMPANY_PROFILE_HISTORY_COLUMNS), {
            'consistent_snapshots': 0, 'entities': 0, 'versions': 0}

    formal = snapshots[snapshots[schema.COMPANY_SNAPSHOT_FORMAL_FIELD] == 1].copy()
    formal = formal.sort_values([entity_field, time_field]).reset_index(drop=True)
    records: list = []
    counters: dict = {}
    previous_entity = None
    previous_profile = None
    for row in formal.to_dict('records'):
        entity_id = row[entity_field]
        profile = row[profile_field] or ''
        if entity_id != previous_entity:
            previous_entity = entity_id
            previous_profile = None
        if profile != previous_profile or not records:
            counters[entity_id] = counters.get(entity_id, 0) + 1
            records.append({
                entity_field: entity_id,
                '公司规范名称': entity_id,
                version_field: counters[entity_id],
                schema.TEXT_FIRST_TIME_FIELD: row[time_field],
                schema.TEXT_LAST_TIME_FIELD: row[time_field],
                schema.COMPANY_PROFILE_OBS_COUNT_FIELD: 0,
                schema.COMPANY_PROFILE_CLEAN_FIELD: profile,
                schema.COMPANY_TAG_LIST_FIELD: row[schema.COMPANY_TAG_LIST_FIELD],
                '所属行业': row['所属行业'],
                '公司性质': row['公司性质'],
                '公司规模': row['公司规模'],
                '公司所在地': '、'.join(text_utils.as_list(
                    row[schema.COMPANY_LOCATION_SET_FIELD])),
                schema.COMPANY_PROFILE_SIGNATURE_FIELD: (
                    company_identity.company_fingerprint(profile)),
                '来源岗位数': 0,
                '_岗位集合': set(),
            })
            previous_profile = profile
        current = records[-1]
        current[schema.TEXT_LAST_TIME_FIELD] = row[time_field]
        current[schema.COMPANY_PROFILE_OBS_COUNT_FIELD] += 1
        current['_岗位集合'] |= set(text_utils.as_list(
            row[schema.COMPANY_SNAPSHOT_JOB_SET_FIELD]))
    for record in records:
        record['来源岗位数'] = len(record.pop('_岗位集合'))
    history = pd.DataFrame(records)
    if history.empty:
        history = pd.DataFrame(columns=schema.COMPANY_PROFILE_HISTORY_COLUMNS)
    else:
        history = history[schema.COMPANY_PROFILE_HISTORY_COLUMNS].sort_values(
            [entity_field, version_field]).reset_index(drop=True)
    statistics = {
        'consistent_snapshots': int(len(formal)),
        'entities': int(history[entity_field].nunique()) if not history.empty else 0,
        'versions': int(len(history)),
    }
    return history, statistics


def build_company_profile_change_events(history: pd.DataFrame,
                                        snapshots: pd.DataFrame) -> tuple:
    """构建公司简介变化事件（含跨歧义快照说明字段）。"""
    entity_field = schema.COMPANY_ENTITY_ID_FIELD
    version_field = schema.COMPANY_PROFILE_VERSION_FIELD
    profile_field = schema.COMPANY_PROFILE_CLEAN_FIELD
    time_field = schema.OBSERVATION_TIME_FIELD
    if history.empty:
        return pd.DataFrame(columns=schema.COMPANY_TEXT_EVENT_COLUMNS), {}

    ambiguous_by_entity: dict = {}
    if not snapshots.empty:
        ambiguous = snapshots[snapshots[schema.COMPANY_SNAPSHOT_STATE_FIELD] == 'AMBIGUOUS']
        ambiguous_by_entity = {
            entity_id: sorted(group[time_field].tolist())
            for entity_id, group in ambiguous.groupby(entity_field, sort=True)}

    ordered = history.sort_values([entity_field, version_field])
    rows = []
    for entity_id, group in ordered.groupby(entity_field, sort=False):
        records = group.to_dict('records')
        ambiguous_times = ambiguous_by_entity.get(entity_id, [])
        for index in range(1, len(records)):
            old, new = records[index - 1], records[index]
            crossed = [time for time in ambiguous_times
                       if old[schema.TEXT_LAST_TIME_FIELD] < time
                       <= new[schema.TEXT_FIRST_TIME_FIELD]]
            rows.append({
                entity_field: entity_id,
                '公司规范名称': new['公司规范名称'],
                '旧公司简介版本号': int(old[version_field]),
                '新公司简介版本号': int(new[version_field]),
                '变化时间': new[schema.TEXT_FIRST_TIME_FIELD],
                '旧公司简介': old[profile_field],
                '新公司简介': new[profile_field],
                '旧文本字符数': text_utils.char_length(old[profile_field]),
                '新文本字符数': text_utils.char_length(new[profile_field]),
                schema.COMPANY_EVENT_AMBIGUOUS_FLAG_FIELD: int(bool(crossed)),
                schema.COMPANY_EVENT_AMBIGUOUS_COUNT_FIELD: int(len(crossed)),
                schema.COMPANY_EVENT_OLD_SNAPSHOT_COUNT_FIELD: int(
                    old[schema.COMPANY_PROFILE_OBS_COUNT_FIELD]),
                schema.COMPANY_EVENT_NEW_SNAPSHOT_COUNT_FIELD: int(
                    new[schema.COMPANY_PROFILE_OBS_COUNT_FIELD]),
                schema.COMPANY_EVENT_OLD_JOB_COUNT_FIELD: int(old['来源岗位数']),
                schema.COMPANY_EVENT_NEW_JOB_COUNT_FIELD: int(new['来源岗位数']),
            })
    events = pd.DataFrame(rows)
    statistics = {
        'events': int(len(events)),
        'ambiguous_cross_events': int(events[schema.COMPANY_EVENT_AMBIGUOUS_FLAG_FIELD].sum())
        if not events.empty else 0,
    }
    return events, statistics

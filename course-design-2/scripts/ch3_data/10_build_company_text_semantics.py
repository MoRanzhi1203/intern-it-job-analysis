# -*- coding: utf-8 -*-
"""Stage 10：公司简介快照 / 版本 / 语义时序（必须先完成公司实体识别）。

处理路线（Refinement R1 新增快照层）：

```text
data/processed/company_entity_map.parquet          公司实体映射层（仅正式实体）
data/interim/job_observation_snapshots.parquet     岗位观测快照层（含公司简介与观测时间）
        ↓  按公司实体 + 观测时间 折叠岗位重复
        ↓
data/interim/company_profile_snapshots.parquet     公司简介快照层（CONSISTENT / AMBIGUOUS / MISSING）
        ↓  只取 CONSISTENT，连续相同简介压缩为一个版本（A→B→A 保留三版本）
        ↓
data/processed/company_profile_history.parquet     公司简介版本层
data/processed/company_text_change_events.parquet  公司简介语义变化事件层
data/features/company_text_embeddings.npz          公司简介句向量（不写入主业务宽表）
```

关键规则：

- 同一公司同一时间存在多个非空简介 → ``AMBIGUOUS``，**禁止**多数表决生成正式版本；
- ``MISSING`` 不得生成伪版本；
- 跨越 AMBIGUOUS 时间点的相邻版本，事件必须记录跨越歧义快照数量。

用法：
    python scripts/ch3_data/10_build_company_text_semantics.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import (company_identity, company_profile_versioning, io_utils,  # noqa: E402
                 project_paths, quality, refinement, schema, skill_extraction,
                 text_semantics, text_utils)

STAGE = 'stage_10'
TITLE = 'Stage 10 公司简介快照 / 版本 / 语义时序'

GLOBALIZATION_KEYWORDS = ['出海', '全球化', '国际化', '海外', '全球']


def load_inputs() -> tuple:
    """读取公司实体映射与岗位观测快照。"""
    entity_map = io_utils.read_parquet(project_paths.COMPANY_ENTITY_MAP_PARQUET)
    snapshots = io_utils.read_parquet(project_paths.OBSERVATION_SNAPSHOT_PARQUET)
    return entity_map, snapshots


def build_entity_lookup(entity_map: pd.DataFrame) -> dict:
    """构建 (公司名称, 公司所在地) → 实体ID 映射（仅正式实体，默认要求无需人工复核）。"""
    formal = company_identity.resolve_formal_entities(entity_map)
    lookup = {}
    for record in formal.to_dict('records'):
        key = (str(record[schema.COMPANY_NAME_ORIGINAL_FIELD]).strip(),
               str(record['公司所在地']).strip())
        lookup[key] = record[schema.COMPANY_ENTITY_ID_FIELD]
    return lookup


def map_observations(snapshots: pd.DataFrame, lookup: dict) -> tuple:
    """把岗位观测快照映射到公司实体，并整理快照层输入。"""
    id_field = schema.ID_FIELD
    time_field = schema.OBSERVATION_TIME_FIELD
    frame = snapshots[[id_field, time_field, '公司名称', '公司所在地', '公司简介',
                       '公司标签', '所属行业', '公司性质', '公司规模']].copy()
    frame['公司名称'] = frame['公司名称'].fillna('').astype(str).str.strip()
    frame['公司所在地'] = frame['公司所在地'].fillna('').astype(str).str.strip()
    frame[schema.COMPANY_PROFILE_CLEAN_FIELD] = frame['公司简介'].map(text_utils.clean_text)
    keys = list(zip(frame['公司名称'], frame['公司所在地']))
    frame[schema.COMPANY_ENTITY_ID_FIELD] = [lookup.get(key) for key in keys]
    mapped = frame[frame[schema.COMPANY_ENTITY_ID_FIELD].notna()].copy()
    mapped[schema.COMPANY_TAG_LIST_FIELD] = mapped['公司标签'].map(text_utils.split_tags)
    mapped[schema.COMPANY_LOCATION_SET_FIELD] = mapped['公司所在地'].map(
        lambda value: [value] if value else [])
    statistics = {
        'snapshot_rows': int(len(frame)),
        'mapped_rows': int(len(mapped)),
        'unmapped_rows': int(len(frame) - len(mapped)),
    }
    return mapped, statistics


def build_profile_embeddings(history: pd.DataFrame,
                             embedding: text_semantics.EmbeddingResult) -> dict:
    """公司简介向量：按「文本类型 + 语料 SHA256 + 模型配置」复用缓存。"""
    entity_field = schema.COMPANY_ENTITY_ID_FIELD
    version_field = schema.COMPANY_PROFILE_VERSION_FIELD
    profile_field = schema.COMPANY_PROFILE_CLEAN_FIELD
    artifacts: dict = {'arrays': {}, 'row_index': {}, 'cache_status': '', 'fingerprint': ''}
    if history.empty or not embedding.available:
        return artifacts
    texts = history[profile_field].fillna('').astype(str).tolist()
    fingerprint = text_semantics.corpus_fingerprint(texts)
    artifacts['fingerprint'] = fingerprint
    positions = np.where(np.array([bool(value.strip()) for value in texts]))[0]
    expected = {text_semantics.TEXT_TYPE_COMPANY: fingerprint}
    sizes = {text_semantics.TEXT_TYPE_COMPANY: int(len(positions))}
    cache = text_semantics.load_array_cache(
        project_paths.COMPANY_TEXT_EMBEDDINGS_NPZ,
        text_semantics.model_config(embedding), expected, expected_sizes=sizes)
    cached = cache['arrays'].get(text_semantics.TEXT_TYPE_COMPANY)
    if cached is not None:
        matrix = cached
        artifacts['cache_status'] = cache['cache_status'][text_semantics.TEXT_TYPE_COMPANY]
    else:
        positions, matrix = text_semantics.embed_non_empty(embedding, texts)
        artifacts['cache_status'] = '本次新建'
    artifacts['arrays']['company_profile'] = matrix
    row_index = {}
    for row, position in enumerate(positions):
        record = history.iloc[int(position)]
        row_index[(record[entity_field], int(record[version_field]))] = row
    artifacts['row_index'] = row_index
    return artifacts


def build_change_events(history: pd.DataFrame, snapshots: pd.DataFrame,
                        embedding: text_semantics.EmbeddingResult,
                        artifacts: dict, config: skill_extraction.SkillConfig) -> tuple:
    """构建公司简介语义变化事件层（含跨歧义快照说明字段）。"""
    entity_field = schema.COMPANY_ENTITY_ID_FIELD
    version_field = schema.COMPANY_PROFILE_VERSION_FIELD
    profile_field = schema.COMPANY_PROFILE_CLEAN_FIELD
    base_events, ambiguity_stats = company_profile_versioning.build_company_profile_change_events(
        history, snapshots)
    if base_events.empty:
        return pd.DataFrame(columns=schema.COMPANY_TEXT_EVENT_COLUMNS), ambiguity_stats, {}

    ordered = history.sort_values([entity_field, version_field]).reset_index(drop=True)
    vectorizer, tfidf_matrix = text_semantics.build_tfidf_matrix(
        ordered[profile_field].fillna('').astype(str).tolist(), text_utils.tokenize, min_df=1)
    tfidf_row_index = {(record[entity_field], int(record[version_field])): position
                       for position, record in enumerate(ordered.to_dict('records'))}
    pair_keys = [((record[entity_field], int(record['旧公司简介版本号'])),
                  (record[entity_field], int(record['新公司简介版本号'])))
                 for record in base_events.to_dict('records')]
    tfidf_values = text_semantics.tfidf_cosine_for_pairs(tfidf_matrix, tfidf_row_index, pair_keys)
    if embedding.available:
        semantic_values = text_semantics.similarity_for_pairs(
            artifacts['arrays']['company_profile'], artifacts['row_index'], pair_keys)
    else:
        semantic_values = np.full(len(pair_keys), np.nan)

    keyword_counter: Counter = Counter()
    rows = []
    ai_event_count = 0
    for position, record in enumerate(base_events.to_dict('records')):
        old_text = record['旧公司简介'] or ''
        new_text = record['新公司简介'] or ''
        tfidf_cosine = float(tfidf_values[position])
        cosine = float(semantic_values[position])
        distance = (1.0 - cosine) if not np.isnan(cosine) else float('nan')
        old_keywords = set(skill_extraction.keyword_match_plain(old_text, config.company_keywords))
        new_keywords = set(skill_extraction.keyword_match_plain(new_text, config.company_keywords))
        added_keywords = sorted(new_keywords - old_keywords)
        removed_keywords = sorted(old_keywords - new_keywords)
        for keyword in added_keywords:
            keyword_counter[keyword] += 1
        added_ai = sorted(skill_extraction.keyword_match_plain(
            ' '.join(added_keywords), config.ai_keywords))
        new_ai_flag = int(bool(added_ai) or bool(
            set(skill_extraction.keyword_match_plain(new_text, config.ai_keywords))
            - set(skill_extraction.keyword_match_plain(old_text, config.ai_keywords))))
        llm_added = sorted(set(skill_extraction.keyword_match_plain(new_text, config.llm_skills))
                           - set(skill_extraction.keyword_match_plain(old_text, config.llm_skills)))
        ai_event_count += new_ai_flag
        old_chars = text_utils.char_length(old_text)
        new_chars = text_utils.char_length(new_text)
        row = dict(record)
        row.update({
            '旧文本字符数': old_chars,
            '新文本字符数': new_chars,
            '文本长度变化': new_chars - old_chars,
            '文本长度变化率': ((new_chars - old_chars) / old_chars) if old_chars else float('nan'),
            'TFIDF余弦相似度': tfidf_cosine,
            'Embedding语义相似度': cosine,
            'Embedding语义距离': distance,
            '新增业务关键词': added_keywords,
            '删除业务关键词': removed_keywords,
            '新增关键词数': len(added_keywords),
            '删除关键词数': len(removed_keywords),
            '是否新增AI关键词': int(bool(added_ai)),
            '是否新增大模型关键词': int(bool(llm_added)),
        })
        rows.append(row)
    events = pd.DataFrame(rows)
    thresholds = text_semantics.quantile_thresholds(events['Embedding语义距离'],
                                                    prefix='公司简介语义距离')
    events[schema.COMPANY_TEXT_EVENT_COLUMNS[-1]] = [
        text_semantics.classify_company_text_change(row, thresholds)
        for row in events.to_dict('records')]
    events = events[schema.COMPANY_TEXT_EVENT_COLUMNS]
    statistics = {
        'thresholds': thresholds,
        'keyword_counter': keyword_counter,
        'ai_event_count': ai_event_count,
        'tfidf_vocabulary': int(len(vectorizer.vocabulary_)) if tfidf_matrix is not None else 0,
        **ambiguity_stats,
    }
    return events, ambiguity_stats, statistics


def build_audit_tables(snapshots: pd.DataFrame, history: pd.DataFrame, events: pd.DataFrame,
                       statistics: dict, fold_stats: dict,
                       embedding: text_semantics.EmbeddingResult,
                       baseline_company: dict) -> dict:
    """构建 19 号公司简介语义审计表（快照层 + 版本层 + 事件层）。"""
    entity_field = schema.COMPANY_ENTITY_ID_FIELD
    version_field = schema.COMPANY_PROFILE_VERSION_FIELD
    state_field = schema.COMPANY_SNAPSHOT_STATE_FIELD
    total_snapshots = int(len(snapshots))
    state_counts = snapshots[state_field].value_counts().to_dict()
    ambiguous_entities = int(snapshots.loc[snapshots[state_field] == 'AMBIGUOUS',
                                           entity_field].nunique())

    overview = pd.DataFrame([
        {'指标': '公司简介快照总数', '数值': total_snapshots},
        {'指标': 'CONSISTENT 快照数', '数值': int(state_counts.get('CONSISTENT', 0))},
        {'指标': 'AMBIGUOUS 快照数', '数值': int(state_counts.get('AMBIGUOUS', 0))},
        {'指标': 'MISSING 快照数', '数值': int(state_counts.get('MISSING', 0))},
        {'指标': '同时间多简介涉及公司数', '数值': ambiguous_entities},
        {'指标': '进入正式版本历史的实体数', '数值': fold_stats.get('history_entities', 0)},
        {'指标': '公司简介版本总数', '数值': fold_stats.get('profile_versions', 0)},
        {'指标': '公司简介语义变化事件数', '数值': int(len(events))},
        {'指标': '有简介版本变化的公司数',
         '数值': int(events[entity_field].nunique()) if not events.empty else 0},
        {'指标': '跨歧义快照的变化事件数',
         '数值': int(events[schema.COMPANY_EVENT_AMBIGUOUS_FLAG_FIELD].sum())
         if not events.empty else 0},
        {'指标': '区间内部跨越歧义快照的版本数',
         '数值': fold_stats.get('interior_ambiguous_spans', 0)},
        {'指标': '区间内部跨越缺失快照的版本数',
         '数值': fold_stats.get('interior_missing_spans', 0)},
        {'指标': '映射成功的岗位观测行数', '数值': fold_stats.get('mapped_rows', 0)},
        {'指标': '未能映射到正式实体的行数', '数值': fold_stats.get('unmapped_rows', 0)},
        {'指标': '新增业务关键词事件数',
         '数值': int((events['新增关键词数'] > 0).sum()) if not events.empty else 0},
        {'指标': '新增 AI 关键词事件数', '数值': int(statistics.get('ai_event_count', 0))},
        {'指标': '新增大模型关键词事件数',
         '数值': int(events['是否新增大模型关键词'].sum()) if not events.empty else 0},
        {'指标': '句向量模型状态', '数值': embedding.status},
        {'指标': '句向量模型名称', '数值': embedding.model_name},
    ])

    candidate_distribution = (snapshots[schema.COMPANY_SNAPSHOT_CANDIDATE_COUNT_FIELD]
                              .value_counts().sort_index()
                              .rename_axis('非空简介候选数').reset_index(name='快照数'))
    candidate_distribution['占比'] = (candidate_distribution['快照数']
                                 / max(total_snapshots, 1)).round(6)

    ambiguous_times = snapshots[snapshots[state_field] == 'AMBIGUOUS'][
        [entity_field, schema.OBSERVATION_TIME_FIELD, schema.COMPANY_SNAPSHOT_CANDIDATE_COUNT_FIELD,
         schema.COMPANY_SNAPSHOT_CANDIDATE_LIST_FIELD,
         schema.COMPANY_SNAPSHOT_MAIN_PROFILE_FIELD,
         schema.COMPANY_SNAPSHOT_MAIN_SUPPORT_FIELD,
         schema.COMPANY_SNAPSHOT_MAIN_SUPPORT_RATE_FIELD,
         schema.COMPANY_SNAPSHOT_JOB_COUNT_FIELD]].copy()
    if not ambiguous_times.empty:
        ambiguous_times[schema.COMPANY_SNAPSHOT_CANDIDATE_LIST_FIELD] = ambiguous_times[
            schema.COMPANY_SNAPSHOT_CANDIDATE_LIST_FIELD].map(
            lambda items: ' ｜ '.join(text_utils.as_list(items))[:500])
        ambiguous_times[schema.COMPANY_SNAPSHOT_MAIN_PROFILE_FIELD] = ambiguous_times[
            schema.COMPANY_SNAPSHOT_MAIN_PROFILE_FIELD].map(lambda text: str(text)[:200])
        ambiguous_times = ambiguous_times.sort_values(
            [entity_field, schema.OBSERVATION_TIME_FIELD])

    support_rate = snapshots[schema.COMPANY_SNAPSHOT_MAIN_SUPPORT_RATE_FIELD].dropna()
    if not support_rate.empty:
        edges = np.linspace(0, 1, 11)
        bucket = pd.cut(support_rate, bins=edges, include_lowest=True).value_counts().sort_index()
        support_table = pd.DataFrame([
            {'支持率区间下限': round(float(interval.left), 4),
             '支持率区间上限': round(float(interval.right), 4),
             '快照数': int(count),
             '占比': round(int(count) / len(support_rate), 6)}
            for interval, count in bucket.items()])
    else:
        support_table = pd.DataFrame(columns=['支持率区间下限', '支持率区间上限', '快照数', '占比'])

    per_entity = (history.groupby(entity_field, as_index=False)
                  .agg(公司简介版本数=(version_field, 'max'),
                       版本观测次数合计=(schema.COMPANY_PROFILE_OBS_COUNT_FIELD, 'sum'),
                       来源岗位数=('来源岗位数', 'max'))
                  .sort_values('公司简介版本数', ascending=False))
    version_distribution = (per_entity['公司简介版本数'].value_counts().sort_index()
                            .rename_axis('公司简介版本数').reset_index(name='公司数'))
    version_distribution['占比'] = (version_distribution['公司数']
                              / max(len(per_entity), 1)).round(6)

    def _pair(name, baseline_value, new_value, reason) -> list:
        table = refinement.compare_metric_rows([(name, baseline_value, new_value, reason)])
        return table.to_dict('records')

    version_before_after = pd.DataFrame(
        _pair('公司简介版本数', baseline_company.get('旧公司简介版本数'), int(len(history)),
              '只允许 CONSISTENT 快照进入正式版本历史，同时间多简介不再串成时间版本')
        + _pair('正式公司实体数', baseline_company.get('旧正式公司实体数'),
                fold_stats.get('history_entities'),
                '跨地域不再自动拆分，MULTI_LOCATION_AMBIGUOUS 等映射不再进入正式公司时序'))
    event_before_after = pd.DataFrame(
        _pair('公司简介变化事件数', baseline_company.get('旧公司简介变化事件数'), int(len(events)),
              '随版本层重算，并新增跨越歧义快照说明字段')
        + _pair('有变化公司数', baseline_company.get('旧有变化公司数'),
                int(events[entity_field].nunique()) if not events.empty else 0,
                '随版本层与事件层重算'))

    cross_ambiguous = events[events[schema.COMPANY_EVENT_AMBIGUOUS_FLAG_FIELD] == 1] \
        if not events.empty else events

    distance_table = pd.DataFrame([
        {'指标': '公司简介语义距离有效样本数',
         '数值': int(events['Embedding语义距离'].notna().sum()) if not events.empty else 0},
        {'指标': '公司简介语义距离均值',
         '数值': round(float(events['Embedding语义距离'].mean()), 6) if not events.empty else np.nan},
        {'指标': '公司简介语义距离中位数',
         '数值': float(events['Embedding语义距离'].median()) if not events.empty else np.nan},
        {'指标': '公司简介语义距离P90',
         '数值': float(events['Embedding语义距离'].quantile(0.9)) if not events.empty else np.nan},
        {'指标': '公司简介语义距离P95',
         '数值': float(events['Embedding语义距离'].quantile(0.95)) if not events.empty else np.nan},
        {'指标': 'TF-IDF 词表规模（词面基线）', '数值': statistics.get('tfidf_vocabulary', 0)},
    ])

    candidate_table = (events[schema.COMPANY_TEXT_EVENT_COLUMNS[-1]].value_counts()
                       .rename_axis('公司简介变化候选类型').reset_index(name='事件数')) \
        if not events.empty else pd.DataFrame(columns=['公司简介变化候选类型', '事件数'])
    if not candidate_table.empty:
        candidate_table['占比'] = (candidate_table['事件数'] / len(events)).round(6)

    return {
        '01_公司简介快照总体统计': overview,
        '02_同时间简介候选数分布': candidate_distribution,
        '03_AMBIGUOUS公司时间点': ambiguous_times,
        '04_主候选支持率分布': support_table,
        '05_正式公司简介版本统计': version_distribution,
        '06_重构前后版本数量对比': version_before_after,
        '07_重构前后事件数量对比': event_before_after,
        '08_跨歧义快照变化事件': cross_ambiguous,
        '09_公司简介语义距离': distance_table,
        '10_公司简介变化候选类型': candidate_table,
        '11_公司简介语义变化事件': events.head(500) if not events.empty else events,
        '12_公司实体版本明细': per_entity.head(500),
        '13_语义距离分位数': text_semantics.build_threshold_source(
            statistics.get('thresholds', {})),
        '14_新增业务关键词统计': pd.DataFrame(
            [{'新增业务关键词': keyword, '出现事件数': count}
             for keyword, count in statistics.get('keyword_counter', Counter()).most_common()])
        if statistics.get('keyword_counter') else pd.DataFrame(
            columns=['新增业务关键词', '出现事件数']),
    }


def build_record_lines(metrics: dict, audit: dict, embedding: text_semantics.EmbeddingResult,
                       artifacts: dict) -> list:
    """生成 12 号公司简介语义时序阶段记录。"""
    overview = audit['01_公司简介快照总体统计'].set_index('指标')['数值']
    lines = [
        '# 阶段记录：Stage 10 公司简介快照 / 版本 / 语义时序（Refinement R1）',
        '',
        '> 本文件由 `scripts/ch3_data/10_build_company_text_semantics.py` 生成，全部数字来自真实运行结果。',
        '',
        '## 1. 与岗位层的关键区别',
        '',
        '- 公司简介时序**以公司实体为单位**，不是以岗位ID为单位；',
        '- 先在「公司实体 + 观测时间」上折叠岗位重复，避免岗位数量多的公司在企业语义时序中被重复加权；',
        '- 新增**公司简介快照层**：同一公司同一时间存在多个非空简介时标记 AMBIGUOUS，禁止用多数表决生成正式版本；',
        '- 一次企业简介变化最多记作一条事件，不因该公司发布多个岗位而重复计数。',
        '',
        '## 2. 快照层状态',
        '',
        f"- 公司简介快照总数：{int(overview['公司简介快照总数'])}；",
        f"- CONSISTENT：{int(overview['CONSISTENT 快照数'])}；",
        f"- AMBIGUOUS：{int(overview['AMBIGUOUS 快照数'])}；",
        f"- MISSING：{int(overview['MISSING 快照数'])}；",
        f"- 同时间多简介涉及公司数：{int(overview['同时间多简介涉及公司数'])}。",
        '',
        '> 只有 CONSISTENT 快照允许进入正式公司简介版本历史；AMBIGUOUS / MISSING 一律排除，',
        '且即使主候选支持率很高也不自动进入正式时序。',
        '',
        '## 3. 规模',
        '',
        f"- 进入正式公司时序的公司实体数：{metrics['entities_with_history']}；",
        f"- 公司简介版本总数：{int(overview['公司简介版本总数'])}；",
        f"- 公司简介语义变化事件数：{int(overview['公司简介语义变化事件数'])}；",
        f"- 有简介版本变化的公司数：{int(overview['有简介版本变化的公司数'])}；",
        f"- 跨歧义快照的变化事件数：{int(overview['跨歧义快照的变化事件数'])}；",
        f"- 未能映射到正式实体的观测行数：{int(overview['未能映射到正式实体的行数'])}。",
        '',
        '## 4. 相似度口径与阈值',
        '',
        f"- 深层语义：`{embedding.model_name}`（状态 {embedding.status}，"
        f"维度 {int(embedding.dimension) if embedding.available else 'NOT_RUN'}）；",
        f"- TF-IDF 词表规模（词面基线）：{metrics['tfidf_vocabulary']}；",
        f"- 向量缓存状态：{artifacts.get('cache_status', '—')}（键 = 文本类型 + 语料 SHA256 + 模型配置）；",
        '- Embedding 语义距离 = 1 − 余弦相似度；TF-IDF 余弦仅为词面基线。',
        '',
        '| 指标 | 阈值 | 样本数 | 来源 |',
        '| --- | --- | --- | --- |',
    ]
    for row in audit['13_语义距离分位数'].itertuples(index=False):
        value = row.阈值
        threshold = value if isinstance(value, str) else (
            '—' if pd.isna(value) else f'{float(value):.6f}')
        sample = '—' if pd.isna(row.样本数) else f'{int(row.样本数)}'
        lines.append(f'| {row.指标} | {threshold} | {sample} | {row.来源} |')
    lines += [
        '',
        '## 5. 变化候选类型',
        '',
        '| 公司简介变化候选类型 | 事件数 | 占比 |',
        '| --- | --- | --- |',
    ]
    for row in audit['10_公司简介变化候选类型'].itertuples(index=False):
        lines.append(f'| {row.公司简介变化候选类型} | {int(row.事件数)} | {row.占比:.2%} |')
    lines += [
        '',
        '> 企业级变化同样全部保留「候选」字样，需人工复核后方可作为结论。',
        '',
        '## 6. 跨歧义快照事件（前 10 条）',
        '',
        '| company_entity_id | 旧版本 | 新版本 | 变化时间 | 跨越歧义快照数量 |',
        '| --- | --- | --- | --- | --- |',
    ]
    cross = audit['08_跨歧义快照变化事件']
    for row in (cross.head(10).itertuples(index=False) if not cross.empty else []):
        lines.append(f'| {row.company_entity_id} | {int(row.旧公司简介版本号)} | '
                     f'{int(row.新公司简介版本号)} | {row.变化时间} | '
                     f'{int(row.跨越歧义快照数量)} |')
    lines += [
        '',
        '> A → B 之间若存在 AMBIGUOUS 快照，事件必须记录跨越歧义快照数量，',
        '避免把变化时间（新版本首次有效观测时间）误读为精确发生时刻。',
        '',
        '## 7. 重构前后对比',
        '',
        '| 指标 | 旧值 | 新值 | 绝对差 | 相对差 | 是否预期变化 | 变化原因 |',
        '| --- | --- | --- | --- | --- | --- | --- |',
    ]
    for table in (audit['06_重构前后版本数量对比'], audit['07_重构前后事件数量对比']):
        for row in table.itertuples(index=False):
            lines.append(f'| {row.指标} | {row.旧值} | {row.新值} | {row.绝对差} | {row.相对差} | '
                         f'{row.是否预期变化} | {row.变化原因} |')
    lines += [
        '',
        '## 8. 业务 / AI 关键词变化（Top 15）',
        '',
        '| 新增业务关键词 | 出现事件数 |',
        '| --- | --- |',
    ]
    for row in audit['14_新增业务关键词统计'].head(15).itertuples(index=False):
        lines.append(f'| {row.新增业务关键词} | {int(row.出现事件数)} |')
    lines += [
        '',
        f"- 新增 AI 关键词事件数：{int(overview['新增 AI 关键词事件数'])}；",
        f"- 新增大模型关键词事件数：{int(overview['新增大模型关键词事件数'])}。",
        '',
        '## 9. 句向量存储与复现',
        '',
        f"- 向量文件：`{metrics['embeddings_npz']}`；索引表：`{metrics['embeddings_index']}`；",
        f"- 复现记录：`{metrics['embeddings_meta']}`；",
        f"- 向量数组：{metrics['embedding_shapes']}；",
        '- 高维向量不写入 `company_profile_history.parquet`。',
        '',
        '## 10. 运行门禁',
        '',
        '| 门禁项 | 状态 |',
        '| --- | --- |',
    ]
    for name, result in metrics['gates'].items():
        lines.append(f'| {name} | {result} |')
    lines += [
        '',
        '## 11. 产物清单',
        '',
        f"- 公司简介快照：`{metrics['snapshot_path']}`（{metrics['snapshot_rows']} 行）；",
        f"- 公司简介版本：`{metrics['history_path']}`（{metrics['history_rows']} 行）；",
        f"- 公司简介语义事件：`{metrics['events_path']}`（{metrics['event_rows']} 行）；",
        f"- 审计表：`{metrics['audit_path']}`；",
        f"- 总修复审计表：`{metrics['refinement_audit']}`；",
        f"- 封版记录：`{metrics['refinement_record']}`。",
        '',
        '> 公司简介语义时序属于**公司实体层增强分析**，其结果不与岗位层语义事件混用计数。',
        '',
    ]
    return lines


def main() -> int:
    quality.configure_logging()
    gates = quality.GateRegistry(STAGE)

    quality.stage_banner(STAGE, TITLE)
    baseline = refinement.capture_company_baseline()
    baseline_company = (baseline.get('company') or {}).get('指标') or {}
    entity_map, observation_snapshots = load_inputs()
    print(f'输入: 公司实体映射 {len(entity_map)} 行 / 观测快照 {len(observation_snapshots)} 行')

    lookup = build_entity_lookup(entity_map)
    formal_entities = company_identity.resolve_formal_entities(entity_map)
    mapped, map_stats = map_observations(observation_snapshots, lookup)
    snapshots = company_profile_versioning.build_company_profile_snapshots(mapped)
    state_counts = snapshots[schema.COMPANY_SNAPSHOT_STATE_FIELD].value_counts().to_dict()

    gates.check('COMPANY_PROFILE_SNAPSHOT_BUILD',
                not snapshots.empty
                and set(state_counts) <= set(schema.COMPANY_SNAPSHOT_STATES)
                and len(snapshots) == int(mapped.groupby(
                    [schema.COMPANY_ENTITY_ID_FIELD, schema.OBSERVATION_TIME_FIELD]).ngroups),
                f'{len(snapshots)} 个「公司实体 × 观测时间」快照，状态分布 '
                f'{ {key: int(value) for key, value in state_counts.items()} }')

    uniqueness = company_profile_versioning.validate_company_snapshot_uniqueness(snapshots)
    gates.check('COMPANY_PROFILE_SNAPSHOT_UNIQUE', bool(uniqueness['唯一']),
                f'company_entity_id + 观测时间 严格唯一（行数 {uniqueness["行数"]}，'
                f'重复键 {uniqueness["重复键数"]}）')

    ambiguous = snapshots[snapshots[schema.COMPANY_SNAPSHOT_STATE_FIELD] == 'AMBIGUOUS']
    multi_candidate = snapshots[snapshots[schema.COMPANY_SNAPSHOT_CANDIDATE_COUNT_FIELD] > 1]
    ambiguous_rates = ambiguous[schema.COMPANY_SNAPSHOT_MAIN_SUPPORT_RATE_FIELD]
    gates.check('COMPANY_SAME_TIME_CONFLICT_AUDIT',
                len(ambiguous) == len(multi_candidate)
                and bool(ambiguous_rates.between(0, 1).all()) if len(ambiguous) else True,
                f'同时间多简介快照 {len(ambiguous)} 个（涉及公司 '
                f'{int(ambiguous[schema.COMPANY_ENTITY_ID_FIELD].nunique())} 家），'
                '主候选支持率仅用于人工审计，不用于自动进入正式时序')

    history, fold_stats = company_profile_versioning.build_company_profile_history(snapshots)
    fold_stats['history_entities'] = fold_stats.get('entities', 0)
    fold_stats['profile_versions'] = fold_stats.get('versions', 0)
    consistent = snapshots[snapshots[schema.COMPANY_SNAPSHOT_STATE_FIELD] == 'CONSISTENT']
    consistent_times = {}
    for row in consistent.to_dict('records'):
        consistent_times.setdefault(row[schema.COMPANY_ENTITY_ID_FIELD], set()).add(
            row[schema.OBSERVATION_TIME_FIELD])
    ambiguous_times = {}
    missing_times = {}
    for state, target in (('AMBIGUOUS', ambiguous_times), ('MISSING', missing_times)):
        banned = snapshots[snapshots[schema.COMPANY_SNAPSHOT_STATE_FIELD] == state]
        for row in banned.to_dict('records'):
            target.setdefault(row[schema.COMPANY_ENTITY_ID_FIELD], set()).add(
                row[schema.OBSERVATION_TIME_FIELD])
    violations = 0
    interior_ambiguous = 0
    interior_missing = 0
    for row in history.to_dict('records'):
        entity_id = row[schema.COMPANY_ENTITY_ID_FIELD]
        first = row[schema.TEXT_FIRST_TIME_FIELD]
        last = row[schema.TEXT_LAST_TIME_FIELD]
        # 版本边界必须是 CONSISTENT 快照时间（真实不变量）
        if first not in consistent_times.get(entity_id, set()) \
                or last not in consistent_times.get(entity_id, set()):
            violations += 1
            continue
        # 区间内部跨越 AMBIGUOUS / MISSING 快照不算违规（压缩只针对 CONSISTENT 快照），
        # 但必须计数留痕，避免读者误以为版本区间内部完全被观测覆盖
        if any(first < time < last for time in ambiguous_times.get(entity_id, set())):
            interior_ambiguous += 1
        if any(first < time < last for time in missing_times.get(entity_id, set())):
            interior_missing += 1
    gates.check('AMBIGUOUS_SNAPSHOT_EXCLUSION',
                violations == 0
                and int(history[schema.COMPANY_PROFILE_OBS_COUNT_FIELD].sum())
                == int(len(consistent))
                and fold_stats['consistent_snapshots'] == int(len(consistent)),
                f'AMBIGUOUS {len(ambiguous)} / MISSING '
                f'{int((snapshots[schema.COMPANY_SNAPSHOT_STATE_FIELD] == "MISSING").sum())} '
                f'快照全部不进入版本层；{len(history)} 个版本的有效观测数合计 '
                f'{int(history[schema.COMPANY_PROFILE_OBS_COUNT_FIELD].sum())} == CONSISTENT '
                f'{int(len(consistent))}，边界越界版本数 {violations}；'
                f'区间内部跨越歧义快照的版本 {interior_ambiguous} 个、跨越缺失快照的版本 '
                f'{interior_missing} 个（已计数留痕）')

    fold_stats['interior_ambiguous_spans'] = int(interior_ambiguous)
    fold_stats['interior_missing_spans'] = int(interior_missing)

    contiguity_ok = True
    for entity_id, group in history.groupby(schema.COMPANY_ENTITY_ID_FIELD, sort=False):
        versions = sorted(int(value) for value in group[schema.COMPANY_PROFILE_VERSION_FIELD])
        if versions != list(range(1, len(versions) + 1)):
            contiguity_ok = False
            break
    time_order_ok = bool((history.groupby(schema.COMPANY_ENTITY_ID_FIELD, sort=False)
                          [schema.TEXT_FIRST_TIME_FIELD]
                          .apply(lambda values: list(values) == sorted(values))).all()) \
        if not history.empty else True
    gates.check('COMPANY_VERSION_REBUILD',
                contiguity_ok and time_order_ok
                and len(history) == fold_stats['profile_versions']
                and set(history[schema.COMPANY_ENTITY_ID_FIELD].unique())
                <= set(formal_entities[schema.COMPANY_ENTITY_ID_FIELD].unique()),
                f'{fold_stats["history_entities"]} 个公司实体 → {len(history)} 个连续简介版本'
                '（版本号连续、时间非降序，A→B→A 保留为三个版本；'
                '全部实体均来自正式时序准入集合，'
                'MULTI_LOCATION_AMBIGUOUS / HIGH_CONFIDENCE_MATCH / UNRESOLVED 出现 0 次）')
    gates.check('COMPANY_PROFILE_VERSION_BUILD',
                set(history.columns) == set(schema.COMPANY_PROFILE_HISTORY_COLUMNS),
                f'公司简介版本层 {len(history)} 行 × {history.shape[1]} 列，字段与 schema 一致')

    config = skill_extraction.load_skill_config()
    embedding = text_semantics.load_embedding_model()
    artifacts = build_profile_embeddings(history, embedding)
    events, ambiguity_stats, statistics = build_change_events(history, snapshots, embedding,
                                                              artifacts, config)
    expected_events = int(sum(max(int(count) - 1, 0)
                              for count in history.groupby(
                                  schema.COMPANY_ENTITY_ID_FIELD).size()))
    gates.check('COMPANY_EVENT_REBUILD',
                len(events) == expected_events
                and schema.COMPANY_EVENT_AMBIGUOUS_COUNT_FIELD in events.columns,
                f'{len(events)} 条公司简介语义变化事件 == sum(max(版本数-1,0)) = '
                f'{expected_events}，跨歧义快照事件 '
                f'{int(events[schema.COMPANY_EVENT_AMBIGUOUS_FLAG_FIELD].sum()) if len(events) else 0} 条')
    gates.check('COMPANY_TEXT_EVENT_BUILD',
                ('候选' in ''.join(events[schema.COMPANY_TEXT_EVENT_COLUMNS[-1]].unique())
                 if len(events) else True)
                and set(events.columns) == set(schema.COMPANY_TEXT_EVENT_COLUMNS),
                f'{len(events)} 条公司简介语义变化事件，涉及 '
                f'{events[schema.COMPANY_ENTITY_ID_FIELD].nunique() if len(events) else 0} 家公司')

    snapshot_path = io_utils.write_parquet(snapshots, project_paths.COMPANY_PROFILE_SNAPSHOTS_PARQUET)
    history_path = io_utils.write_parquet(history, project_paths.COMPANY_PROFILE_HISTORY_PARQUET)
    event_path = io_utils.write_parquet(events, project_paths.COMPANY_TEXT_EVENTS_PARQUET)

    if embedding.available and artifacts['arrays']:
        index_frame = pd.DataFrame([
            {schema.COMPANY_ENTITY_ID_FIELD: key[0],
             schema.COMPANY_PROFILE_VERSION_FIELD: int(key[1]),
             'text_type': text_semantics.TEXT_TYPE_COMPANY,
             'embedding_row': int(row),
             'model_name': embedding.model_name}
            for key, row in sorted(artifacts['row_index'].items())])
        meta = embedding.reproduction_record(input_field=schema.COMPANY_PROFILE_CLEAN_FIELD)
        meta['模型配置'] = text_semantics.model_config(embedding)
        meta['语料指纹'] = {text_semantics.TEXT_TYPE_COMPANY: artifacts['fingerprint']}
        meta['向量数组形状'] = {'company_profile': list(
            artifacts['arrays']['company_profile'].shape)}
        meta['索引行数'] = int(len(index_frame))
        meta['公司实体数'] = int(fold_stats['history_entities'])
        meta['向量来源'] = artifacts['cache_status']
        saved = text_semantics.save_embeddings(
            project_paths.COMPANY_TEXT_EMBEDDINGS_NPZ,
            project_paths.COMPANY_TEXT_EMBEDDING_INDEX_PARQUET,
            artifacts['arrays'], index_frame, meta)
    else:
        saved = {'npz': 'NOT_RUN', 'index': 'NOT_RUN', 'meta': 'NOT_RUN', 'keys': {}}

    fold_stats.update(map_stats)
    audit_tables = build_audit_tables(snapshots, history, events, statistics, fold_stats,
                                      embedding, baseline_company)
    audit_path = io_utils.write_excel(project_paths.TABLES_DIR / project_paths.TABLE_COMPANY_TEXT_SEMANTIC,
                                      audit_tables)
    gates.check('COMPANY_TEXT_AUDIT_EXPORT',
                snapshot_path.exists() and history_path.exists() and event_path.exists()
                and audit_path.exists()
                and {'01_公司简介快照总体统计', '02_同时间简介候选数分布', '03_AMBIGUOUS公司时间点',
                     '04_主候选支持率分布', '05_正式公司简介版本统计', '06_重构前后版本数量对比',
                     '07_重构前后事件数量对比', '08_跨歧义快照变化事件'} <= set(audit_tables),
                f'快照 {len(snapshots)} 行、版本 {len(history)} 行、事件 {len(events)} 行、'
                f'审计表 {len(audit_tables)} 张子表已写出')

    snapshot_stats = {
        'snapshot_rows': int(len(snapshots)),
        'consistent_snapshots': int(state_counts.get('CONSISTENT', 0)),
        'ambiguous_snapshots': int(state_counts.get('AMBIGUOUS', 0)),
        'missing_snapshots': int(state_counts.get('MISSING', 0)),
        'ambiguous_entities': int(ambiguous[schema.COMPANY_ENTITY_ID_FIELD].nunique())
        if len(ambiguous) else 0,
        'snapshot_state_table': snapshots[schema.COMPANY_SNAPSHOT_STATE_FIELD]
        .value_counts().rename_axis('公司简介快照状态').reset_index(name='快照数')
        .assign(占比=lambda frame: (frame['快照数'] / max(len(snapshots), 1)).round(6))
        .to_dict('records'),
    }
    state_table = pd.DataFrame(snapshot_stats['snapshot_state_table'])
    if not state_table.empty:
        state_table['涉及公司数'] = state_table['公司简介快照状态'].map(
            snapshots.groupby(schema.COMPANY_SNAPSHOT_STATE_FIELD)
            [schema.COMPANY_ENTITY_ID_FIELD].nunique())
        snapshot_stats['snapshot_conflict_table'] = state_table.to_dict('records')
    else:
        snapshot_stats['snapshot_conflict_table'] = []

    metrics = {
        'snapshot_path': project_paths.relative_to_root(snapshot_path),
        'snapshot_rows': int(len(snapshots)),
        'history_path': project_paths.relative_to_root(history_path),
        'history_rows': int(len(history)),
        'events_path': project_paths.relative_to_root(event_path),
        'event_rows': int(len(events)),
        'audit_path': project_paths.relative_to_root(audit_path),
        'formal_entities': int(formal_entities[schema.COMPANY_ENTITY_ID_FIELD].nunique()),
        'entities_with_history': int(fold_stats.get('history_entities', 0)),
        'entities_with_change': int(events[schema.COMPANY_ENTITY_ID_FIELD].nunique())
        if len(events) else 0,
        'unmapped_rows': int(map_stats['unmapped_rows']),
        'ambiguous_cross_events': int(ambiguity_stats.get('ambiguous_cross_events', 0)),
        'ai_events': int(statistics.get('ai_event_count', 0)),
        'llm_events': int(events['是否新增大模型关键词'].sum()) if len(events) else 0,
        'tfidf_vocabulary': statistics.get('tfidf_vocabulary', 0),
        'semantic_p50': statistics['thresholds'].get('公司简介语义距离P50', np.nan),
        'semantic_p90': statistics['thresholds'].get('公司简介语义距离P90', np.nan),
        'semantic_p95': statistics['thresholds'].get('公司简介语义距离P95', np.nan),
        'embedding_status': embedding.status,
        'embedding_model': embedding.model_name,
        'embedding_dimension': int(embedding.dimension) if embedding.available else 0,
        'embeddings_npz': saved['npz'],
        'embeddings_index': saved['index'],
        'embeddings_meta': saved['meta'],
        'embedding_shapes': {name: list(array.shape)
                             for name, array in artifacts['arrays'].items()},
        'metrics_path': project_paths.relative_to_root(
            project_paths.METRICS_DIR / f'{STAGE}.json'),
    }
    metrics.update(snapshot_stats)
    metrics['gates'] = {name: result['status'] for name, result in gates.results.items()}
    io_utils.write_json(project_paths.METRICS_DIR / f'{STAGE}.json', metrics)

    refinement_result = refinement.write_refinement_outputs(baseline=baseline)
    metrics['refinement_audit'] = refinement_result['audit_path']
    metrics['refinement_record'] = refinement_result['record_path']

    record_path = io_utils.write_markdown(
        project_paths.RECORDS_DIR / project_paths.RECORD_COMPANY_TEXT_SEMANTIC,
        build_record_lines(metrics, audit_tables, embedding, artifacts))
    print(f'阶段记录: {project_paths.relative_to_root(record_path)}')
    print(f"总修复审计表: {metrics['refinement_audit']}")

    quality.print_gates(gates.results, order=quality.STAGE_GATE_MAP[STAGE])
    gates.save()
    return 0


if __name__ == '__main__':
    sys.exit(main())

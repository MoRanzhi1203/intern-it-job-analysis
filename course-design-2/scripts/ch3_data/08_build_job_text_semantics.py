# -*- coding: utf-8 -*-
"""Stage 08：岗位描述语义时序（双语义口径：完整文本 / 去薪资安全文本）。

处理路线：

data/interim/job_text_version_corpus.parquet   岗位版本文本语料层（含语义分析版与模型安全版）
data/processed/job_version_history.parquet     岗位版本时序层
data/processed/job_change_events.parquet       字段变化事件层
        ↓  同一岗位相邻核心版本 V(k-1) → V(k) 的文本对
        ↓  字面/词面指标：字符数、词数、Token Jaccard、编辑相似度
        ↓  **完整口径**：语义分析版全文 → TF-IDF 余弦 / BGE 余弦 / 语义距离
        ↓  **去薪资口径**：模型安全版全文 → TF-IDF 余弦 / BGE 余弦 / 语义距离（正式主判据）
        ↓  完整减去薪资安全语义距离差（仅作诊断，禁止解释为「薪资文本贡献率」）
        ↓  分段语义距离：职责 / 任职要求 / 技能段（不做人为加权综合）
        ↓  BGE 真实 tokenizer 的 token 长度与截断审计（禁止用字符数近似）
        ↓  由真实分布确定 P50/P75/P90/P95，据此判定显著变化候选
        ↓
data/processed/job_text_change_events.parquet  岗位文本语义变化事件层
data/features/job_text_embeddings.npz          高维句向量（不写入主业务宽表）
data/features/job_text_embedding_index.parquet 向量索引（含 token 截断审计字段）

Embedding 缓存键 = 文本类型 + 语料 SHA256 + 模型配置；
安全文本优先复用完整版向量，仅在文本不同时按哈希去重后编码。

若句向量模型不可用：明确标记 NOT_RUN，绝不用 TF-IDF 冒充深层语义。

用法：
    python scripts/ch3_data/08_build_job_text_semantics.py
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

from src import (io_utils, project_paths, quality, refinement, schema,  # noqa: E402
                 skill_extraction, text_semantics, text_utils)

STAGE = 'stage_08'
TITLE = 'Stage 08 岗位描述语义时序（完整 / 去薪资双口径）'

# 分段文本 → 语义指标列名（严格对应提示词字段命名）
SECTION_SPECS = {
    '职责': {
        'text': '职责文本',
        'similarity': '职责语义相似度',
        'distance': '职责语义距离',
        'char_rate': '职责段字符数变化率',
    },
    '要求': {
        'text': '任职要求文本',
        'similarity': '任职要求语义相似度',
        'distance': '任职要求语义距离',
        'char_rate': '任职要求段字符数变化率',
    },
    '技能': {
        'text': '技能要求文本',
        'similarity': '技能段语义相似度',
        'distance': '技能段语义距离',
        'char_rate': '技能段字符数变化率',
    },
}
SECTION_TEXT_TYPES = {'职责': text_semantics.TEXT_TYPE_DUTY,
                      '要求': text_semantics.TEXT_TYPE_REQUIRE,
                      '技能': text_semantics.TEXT_TYPE_SKILL}

# 语料列 → 文本类型（顺序即编码顺序）
BASE_TEXT_TYPES = {
    text_semantics.TEXT_TYPE_JOB_FULL: schema.JD_SEMANTIC_FIELD,
    text_semantics.TEXT_TYPE_DUTY: '职责文本',
    text_semantics.TEXT_TYPE_REQUIRE: '任职要求文本',
    text_semantics.TEXT_TYPE_SKILL: '技能要求文本',
}
SAFE_TEXT_TYPE = text_semantics.TEXT_TYPE_JOB_SAFE

# 人工复核抽样数量（P50 / P75 / P90 / P95 各抽多少条）
REVIEW_SAMPLE_PER_QUANTILE = 6
# 显著 / 极端变化候选阈值分位
SIGNIFICANT_QUANTILE = 0.90
EXTREME_QUANTILE = 0.95
# 双口径向量一致性容差（完整文本 == 安全文本时两套 Embedding 必须逐行一致）
VECTOR_ATOL = 1e-6


def load_inputs() -> tuple:
    """读取语料、版本历史与字段变化事件。"""
    corpus = io_utils.read_parquet(project_paths.TEXT_CORPUS_PARQUET)
    versions = io_utils.read_parquet(project_paths.PROCESSED_VERSION_HISTORY_PARQUET)
    events = io_utils.read_parquet(project_paths.PROCESSED_CHANGE_EVENTS_PARQUET)
    return corpus, versions, events


def build_pairs(versions: pd.DataFrame) -> pd.DataFrame:
    """构造同一岗位相邻核心版本的文本对（一次版本切换 = 至多一条文本事件）。"""
    id_field = schema.ID_FIELD
    version_field = schema.CORE_VERSION_FIELD
    ordered = versions.sort_values([id_field, version_field]).copy()
    ordered['_prev_version'] = ordered.groupby(id_field)[version_field].shift(1)
    pairs = ordered[ordered['_prev_version'].notna()].copy()
    pairs['_prev_version'] = pairs['_prev_version'].astype('int64')
    return pairs


def corpus_texts(corpus: pd.DataFrame) -> dict:
    """按语料行顺序取出各文本类型的文本（含安全文本）。"""
    texts = {text_type: corpus[column].fillna('').astype(str).tolist()
             for text_type, column in BASE_TEXT_TYPES.items()}
    texts[SAFE_TEXT_TYPE] = corpus[schema.JD_SAFE_FIELD].fillna('').astype(str).tolist()
    return texts


def build_embeddings(corpus: pd.DataFrame,
                     embedding: text_semantics.EmbeddingResult) -> dict:
    """构建（或复用）五类句向量，并生成含 token 审计字段的向量索引。

    安全文本处理顺序（提示词第 6 节）：
    1. 安全文本 == 完整文本 → 直接复用完整版向量，不重新推理；
    2. 其余安全文本按文本 SHA256 去重后只编码一次；
    3. 全量向量（完整 / 职责 / 要求 / 技能）按「文本类型 + 语料指纹 + 模型配置」复用缓存。
    """
    id_field = schema.ID_FIELD
    version_field = schema.CORE_VERSION_FIELD
    texts = corpus_texts(corpus)
    ids = corpus[id_field].tolist()
    versions = [int(value) for value in corpus[version_field].tolist()]
    # (岗位ID, 核心版本号) → 语料行位置（token 审计按行位置对齐，避免重复扫描语料）
    position_of_key = {(ids[position], versions[position]): position
                       for position in range(len(ids))}
    masks = {name: np.array([bool(value.strip()) for value in values])
             for name, values in texts.items()}
    fingerprints = {name: text_semantics.corpus_fingerprint(values)
                    for name, values in texts.items()}
    expected_sizes = {name: int(mask.sum()) for name, mask in masks.items()}

    artifacts: dict = {
        'arrays': {}, 'index_rows': [], 'row_index': {}, 'counters': {},
        'cache_status': {}, 'fingerprints': fingerprints,
        'model_max_tokens': 0, 'token_stats': [], 'token_distribution': [],
        'token_lengths': {},
    }
    if not embedding.available:
        return artifacts

    config = text_semantics.model_config(embedding)
    max_tokens = text_semantics.model_max_tokens(embedding)
    artifacts['model_max_tokens'] = int(max_tokens)
    legacy_fingerprint = text_semantics.legacy_corpus_fingerprint(
        texts[text_semantics.TEXT_TYPE_JOB_FULL])
    cache = text_semantics.load_array_cache(
        project_paths.JOB_TEXT_EMBEDDINGS_NPZ, config, fingerprints,
        expected_sizes=expected_sizes, legacy_fingerprint=legacy_fingerprint)
    artifacts['cache_status'] = dict(cache['cache_status'])

    # ---- 1) 完整文本 / 分段文本：缓存命中直接复用，否则真实编码 ----
    for text_type, values in texts.items():
        if text_type == SAFE_TEXT_TYPE:
            continue
        positions = np.where(masks[text_type])[0]
        cached = cache['arrays'].get(text_type)
        if cached is not None:
            matrix = cached
            status = cache['cache_status'].get(text_type, '复用缓存')
        else:
            positions, matrix = text_semantics.embed_non_empty(embedding, values)
            status = '本次新建'
        artifacts['arrays'][text_type] = matrix
        artifacts['row_index'][text_type] = {}
        for row, position in enumerate(positions):
            artifacts['row_index'][text_type][(ids[int(position)], versions[int(position)])] = row

    # ---- 2) 安全文本：复用完整版向量 + 哈希去重编码 ----
    full_type = text_semantics.TEXT_TYPE_JOB_FULL
    full_values = texts[full_type]
    safe_values = texts[SAFE_TEXT_TYPE]
    full_matrix = artifacts['arrays'][full_type]
    position_of_full_key = dict(artifacts['row_index'][full_type])

    safe_positions = np.where(masks[SAFE_TEXT_TYPE])[0]
    counters = {
        '安全文本总数': int(len(safe_positions)),
        '缓存整体复用数': 0,
        '直接复用完整版向量数': 0,
        '需要新编码的样本数': 0,
        '哈希去重后新编码文本数': 0,
        '实际新增编码数': 0,
        '去重节省编码数': 0,
        '缓存复用比例': 0.0,
        '安全文本编码来源': '',
    }
    same_text_rows = 0
    for position in safe_positions:
        position = int(position)
        if safe_values[position].strip() and \
                safe_values[position].strip() == full_values[position].strip():
            same_text_rows += 1
    cached_safe = cache['arrays'].get(SAFE_TEXT_TYPE)
    counters['完整与安全文本相同的行数'] = int(same_text_rows)
    if cached_safe is not None and cached_safe.shape[0] == len(safe_positions):
        safe_matrix = cached_safe
        counters['缓存整体复用数'] = int(len(safe_positions))
        counters['缓存复用比例'] = 1.0
        counters['安全文本编码来源'] = cache['cache_status'].get(SAFE_TEXT_TYPE, '复用缓存')
    else:
        safe_matrix = np.zeros((len(safe_positions), embedding.dimension), dtype='float32')
        direct_index: dict = {}
        pending: dict = {}
        for row, position in enumerate(safe_positions):
            position = int(position)
            safe_text = safe_values[position].strip()
            full_text = full_values[position].strip()
            key = (ids[position], versions[position])
            if safe_text == full_text and full_text:
                direct_index[row] = position_of_full_key.get(key)
                continue
            pending.setdefault(text_semantics.text_hash(safe_text), []).append((row, safe_text))
        counters['直接复用完整版向量数'] = len(direct_index)
        counters['需要新编码的样本数'] = int(sum(len(items) for items in pending.values()))
        counters['哈希去重后新编码文本数'] = int(len(pending))
        counters['去重节省编码数'] = (counters['需要新编码的样本数']
                                - counters['哈希去重后新编码文本数'])
        for row, full_row in direct_index.items():
            if full_row is not None:
                safe_matrix[row] = full_matrix[full_row]
        if pending:
            unique_texts = [items[0][1] for _hash, items in sorted(pending.items())]
            unique_vectors = text_semantics.encode_texts(embedding, unique_texts)
            counters['实际新增编码数'] = int(len(unique_texts))
            for (hash_key, items), vector in zip(sorted(pending.items()), unique_vectors):
                for row, _text in items:
                    safe_matrix[row] = vector
        total = counters['安全文本总数']
        counters['缓存复用比例'] = (round(counters['直接复用完整版向量数'] / total, 6)
                              if total else 0.0)
        counters['安全文本编码来源'] = '本次编码（复用完整版向量 + 哈希去重新增）'
    artifacts['arrays'][SAFE_TEXT_TYPE] = safe_matrix
    artifacts['row_index'][SAFE_TEXT_TYPE] = {}
    for row, position in enumerate(safe_positions):
        position = int(position)
        artifacts['row_index'][SAFE_TEXT_TYPE][
            (ids[position], versions[position])] = row
    artifacts['counters'] = counters

    # ---- 3) 真实 tokenizer 的 token 长度与截断审计 ----
    distribution_rows = []
    for text_type, values in texts.items():
        lengths = text_semantics.token_lengths(embedding, values)
        artifacts['token_lengths'][text_type] = lengths
        non_empty = masks[text_type]
        artifacts['token_stats'].append(text_semantics.truncation_frame(
            text_semantics.TEXT_TYPE_LABELS[text_type], lengths[non_empty], max_tokens))
        distribution_rows.append(text_semantics.token_distribution_frame(
            text_semantics.TEXT_TYPE_LABELS[text_type], lengths[non_empty], max_tokens))
    artifacts['token_distribution'] = pd.concat(distribution_rows, ignore_index=True) \
        if distribution_rows else pd.DataFrame()

    # ---- 4) 向量索引（含 token 审计字段） ----
    model_name = embedding.model_name
    for text_type, values in texts.items():
        lengths = artifacts['token_lengths'][text_type]
        for (key, row) in sorted(artifacts['row_index'][text_type].items(),
                                 key=lambda item: (item[0][0], item[0][1])):
            position = position_of_key.get(key)
            token_count = int(lengths[int(position)]) if position is not None else 0
            exceed = max(token_count - max_tokens, 0)
            artifacts['index_rows'].append({
                id_field: key[0],
                version_field: int(key[1]),
                'text_type': text_type,
                'embedding_row': int(row),
                'model_name': model_name,
                schema.TOKEN_RAW_COUNT_FIELD: token_count,
                schema.TOKEN_MAX_COUNT_FIELD: int(max_tokens),
                schema.TOKEN_TRUNCATED_FIELD: int(exceed > 0),
                schema.TOKEN_EXCEED_FIELD: int(exceed),
                schema.TOKEN_RETENTION_FIELD: (round(min(max_tokens / token_count, 1.0), 6)
                                               if token_count > 0 else 1.0),
            })
    return artifacts


def _pair_values(matrix, row_index: dict, pairs) -> np.ndarray:
    """逐对余弦相似度；任一侧文本为空（无向量）时返回 NaN。"""
    if matrix is None or matrix.size == 0:
        return np.full(len(pairs), np.nan)
    return text_semantics.similarity_for_pairs(matrix, row_index, pairs)


def build_events(corpus: pd.DataFrame, versions: pd.DataFrame, change_events: pd.DataFrame,
                 matcher: skill_extraction.SkillMatcher,
                 embedding: text_semantics.EmbeddingResult, artifacts: dict) -> tuple:
    """构建岗位文本语义变化事件表（完整口径 + 去薪资口径）。"""
    id_field = schema.ID_FIELD
    version_field = schema.CORE_VERSION_FIELD
    pairs = build_pairs(versions)
    pair_records = pairs.to_dict('records')
    pair_keys = [((record[id_field], int(record['_prev_version'])),
                  (record[id_field], int(record[version_field]))) for record in pair_records]

    corpus_lookup = {}
    position_of_key = {}
    for position, record in enumerate(corpus.to_dict('records')):
        key = (record[id_field], int(record[version_field]))
        corpus_lookup[key] = record
        position_of_key[key] = position

    # ---- 业务字段变化对齐（版本历史变化标志 + 变化事件中的学历要求） ----
    flag_fields = {
        '是否同时薪资变化': '是否薪资变化',
        '是否同时城市变化': '是否城市变化',
        '是否同时公司变化': '是否公司变化',
        '是否同时岗位标题变化': '是否标题变化',
    }
    degree_changes = set()
    if not change_events.empty:
        degree_rows = change_events[change_events['变化字段'] == '学历要求']
        degree_changes = {(record[id_field], int(record['新核心版本号']))
                          for record in degree_rows.to_dict('records')}

    # ---- TF-IDF 词面基线：完整口径与去薪资口径分别拟合 ----
    full_texts = corpus[schema.JD_SEMANTIC_FIELD].fillna('').astype(str).tolist()
    safe_texts = corpus[schema.JD_SAFE_FIELD].fillna('').astype(str).tolist()
    vectorizer_full, matrix_full = text_semantics.build_tfidf_matrix(full_texts, text_utils.tokenize)
    vectorizer_safe, matrix_safe = text_semantics.build_tfidf_matrix(safe_texts, text_utils.tokenize)
    tfidf_row_index = position_of_key
    tfidf_full_values = text_semantics.tfidf_cosine_for_pairs(matrix_full, tfidf_row_index, pair_keys)
    tfidf_safe_values = text_semantics.tfidf_cosine_for_pairs(matrix_safe, tfidf_row_index, pair_keys)

    full_type = text_semantics.TEXT_TYPE_JOB_FULL
    safe_type = text_semantics.TEXT_TYPE_JOB_SAFE
    full_similarity = _pair_values(artifacts['arrays'].get(full_type),
                                   artifacts['row_index'].get(full_type, {}), pair_keys) \
        if embedding.available else np.full(len(pair_keys), np.nan)
    safe_similarity = _pair_values(artifacts['arrays'].get(safe_type),
                                   artifacts['row_index'].get(safe_type, {}), pair_keys) \
        if embedding.available else np.full(len(pair_keys), np.nan)
    section_values = {}
    for label, text_type in SECTION_TEXT_TYPES.items():
        if not embedding.available:
            continue
        section_values[label] = _pair_values(artifacts['arrays'].get(text_type),
                                             artifacts['row_index'].get(text_type, {}), pair_keys)

    # 分段字符长度预先算好，循环内不再分词
    section_lengths = {}
    corpus_records = corpus.to_dict('records')
    for label, spec in SECTION_SPECS.items():
        section_lengths[label] = {
            (record[id_field], int(record[version_field])): text_utils.char_length(record[spec['text']])
            for record in corpus_records}

    full_tokens = artifacts['token_lengths'].get(full_type)
    safe_tokens = artifacts['token_lengths'].get(safe_type)
    max_tokens = artifacts['model_max_tokens'] or 0

    def _truncated(key, lengths) -> int:
        position = position_of_key.get(key)
        if position is None or lengths is None or not max_tokens:
            return 0
        return int(int(lengths[position]) > max_tokens)

    rows = []
    added_skill_counter: Counter = Counter()
    removed_skill_counter: Counter = Counter()
    candidate_counter: Counter = Counter()
    llm_skills = skill_extraction.llm_skill_set(matcher.config)
    for pair_position, record in enumerate(pair_records):
        job_id = record[id_field]
        old_version = int(record['_prev_version'])
        new_version = int(record[version_field])
        old_key, new_key = (job_id, old_version), (job_id, new_version)
        old_row = corpus_lookup.get(old_key, {})
        new_row = corpus_lookup.get(new_key, {})
        old_text = old_row.get(schema.JD_SEMANTIC_FIELD, '') or ''
        new_text = new_row.get(schema.JD_SEMANTIC_FIELD, '') or ''
        old_safe = old_row.get(schema.JD_SAFE_FIELD, '') or ''
        new_safe = new_row.get(schema.JD_SAFE_FIELD, '') or ''

        old_chars = text_utils.char_length(old_text)
        new_chars = text_utils.char_length(new_text)

        def _distance(cosine: float, old_value: str, new_value: str) -> tuple:
            if not embedding.available or np.isnan(cosine) \
                    or not old_value.strip() or not new_value.strip():
                return float('nan'), float('nan')
            similarity = float(cosine)
            return similarity, float(1.0 - similarity)

        full_cosine, full_distance = _distance(
            float(full_similarity[pair_position]), old_text, new_text)
        safe_cosine, safe_distance = _distance(
            float(safe_similarity[pair_position]), old_safe, new_safe)
        gap = (float(full_distance - safe_distance)
               if not np.isnan(full_distance) and not np.isnan(safe_distance) else float('nan'))

        section_distances = {}
        section_char_rates = {}
        for label, spec in SECTION_SPECS.items():
            old_section = old_row.get(spec['text']) or ''
            new_section = new_row.get(spec['text']) or ''
            pair_similarity = (float(section_values[label][pair_position])
                               if label in section_values else float('nan'))
            if old_section.strip() and new_section.strip() and not np.isnan(pair_similarity):
                section_distances[spec['similarity']] = pair_similarity
                section_distances[spec['distance']] = float(1.0 - pair_similarity)
            else:
                section_distances[spec['similarity']] = float('nan')
                section_distances[spec['distance']] = float('nan')
            old_length = section_lengths[label].get(old_key, 0)
            new_length = section_lengths[label].get(new_key, 0)
            section_char_rates[spec['char_rate']] = (
                (new_length - old_length) / old_length if old_length else float('nan'))

        old_skills = set(text_utils.as_list(old_row.get(schema.SKILL_SET_FIELD)))
        new_skills = set(text_utils.as_list(new_row.get(schema.SKILL_SET_FIELD)))
        added = sorted(new_skills - old_skills)
        removed = sorted(old_skills - new_skills)
        kept = sorted(old_skills & new_skills)
        skill_union = old_skills | new_skills
        for skill in added:
            added_skill_counter[skill] += 1
        for skill in removed:
            removed_skill_counter[skill] += 1

        old_full_trunc = _truncated(old_key, full_tokens)
        new_full_trunc = _truncated(new_key, full_tokens)
        old_safe_trunc = _truncated(old_key, safe_tokens)
        new_safe_trunc = _truncated(new_key, safe_tokens)
        row = {
            id_field: job_id,
            '旧核心版本号': old_version,
            '新核心版本号': new_version,
            '变化时间': record[schema.VERSION_FIRST_TIME_FIELD],
            '岗位标题': record['岗位标题'],
            '公司名称': record['公司名称'],
            schema.JD_RAW_FIELD: old_row.get(schema.JD_RAW_FIELD),
            '新岗位描述': new_row.get(schema.JD_RAW_FIELD),
            '旧文本字符数': old_chars,
            '新文本字符数': new_chars,
            '字符数变化': new_chars - old_chars,
            '字符数变化率': ((new_chars - old_chars) / old_chars) if old_chars else float('nan'),
            '旧文本词数': text_utils.count_tokens(old_text),
            '新文本词数': text_utils.count_tokens(new_text),
            '文本完全一致标志': int(text_utils.texts_identical(old_text, new_text)),
            'Token_Jaccard相似度': text_utils.token_jaccard(old_text, new_text),
            '编辑相似度': text_utils.edit_similarity(old_text, new_text),
            schema.TFIDF_FULL_FIELD: float(tfidf_full_values[pair_position]),
            schema.JD_SIMILARITY_FULL_FIELD: full_cosine,
            schema.JD_DISTANCE_FULL_FIELD: full_distance,
            schema.TFIDF_SAFE_FIELD: float(tfidf_safe_values[pair_position]),
            schema.JD_SIMILARITY_SAFE_FIELD: safe_cosine,
            schema.JD_DISTANCE_SAFE_FIELD: safe_distance,
            schema.DISTANCE_GAP_FIELD: gap,
            **section_distances,
            **section_char_rates,
            '旧技能集合': sorted(old_skills),
            '新技能集合': sorted(new_skills),
            '新增技能集合': added,
            '删除技能集合': removed,
            '保留技能集合': kept,
            '新增技能数': len(added),
            '删除技能数': len(removed),
            '技能变化数': len(added) + len(removed),
            '技能Jaccard相似度': (len(kept) / len(skill_union)) if skill_union else float('nan'),
            '是否新增LLM': int('大模型' in added),
            '是否删除LLM': int('大模型' in removed),
            '是否新增RAG': int('RAG' in added),
            '是否删除RAG': int('RAG' in removed),
            '是否新增Agent': int('Agent' in added),
            '是否删除Agent': int('Agent' in removed),
            '是否新增大模型相关技能': int(bool(set(added) & llm_skills)),
            '是否删除大模型相关技能': int(bool(set(removed) & llm_skills)),
            '是否同时薪资变化': int(record[flag_fields['是否同时薪资变化']]),
            '是否同时城市变化': int(record[flag_fields['是否同时城市变化']]),
            '是否同时公司变化': int(record[flag_fields['是否同时公司变化']]),
            '是否同时岗位标题变化': int(record[flag_fields['是否同时岗位标题变化']]),
            '是否同时学历要求变化': int(new_key in degree_changes),
            '旧文本是否截断_完整': old_full_trunc,
            '新文本是否截断_完整': new_full_trunc,
            '旧文本是否截断_去薪资': old_safe_trunc,
            '新文本是否截断_去薪资': new_safe_trunc,
            '版本对是否存在截断': int(bool(old_full_trunc or new_full_trunc
                                       or old_safe_trunc or new_safe_trunc)),
        }
        # 词数变化依赖已算好的分词结果
        row['词数变化'] = row['新文本词数'] - row['旧文本词数']
        rows.append(row)

    events = pd.DataFrame(rows)

    # ---- 阈值必须来自真实分布（双口径分别统计） ----
    thresholds: dict = {}
    thresholds.update(text_semantics.quantile_thresholds(
        events[schema.JD_DISTANCE_FULL_FIELD], prefix=schema.JD_DISTANCE_FULL_FIELD))
    thresholds.update(text_semantics.quantile_thresholds(
        events[schema.JD_DISTANCE_SAFE_FIELD], prefix=schema.JD_DISTANCE_SAFE_FIELD))
    for label, spec in SECTION_SPECS.items():
        thresholds.update(text_semantics.quantile_thresholds(
            events[spec['distance']], prefix=spec['distance']))

    p90_full = thresholds[f'{schema.JD_DISTANCE_FULL_FIELD}P90']
    p95_full = thresholds[f'{schema.JD_DISTANCE_FULL_FIELD}P95']
    p90_safe = thresholds[f'{schema.JD_DISTANCE_SAFE_FIELD}P90']
    p95_safe = thresholds[f'{schema.JD_DISTANCE_SAFE_FIELD}P95']
    events[schema.SIGNIFICANT_FULL_CALIBER_FIELD] = (
        events[schema.JD_DISTANCE_FULL_FIELD].ge(p90_full)
        & events[schema.JD_DISTANCE_FULL_FIELD].notna()).astype(int)
    events[schema.EXTREME_FULL_CALIBER_FIELD] = (
        events[schema.JD_DISTANCE_FULL_FIELD].ge(p95_full)
        & events[schema.JD_DISTANCE_FULL_FIELD].notna()).astype(int)
    events[schema.SIGNIFICANT_FIELD] = (
        events[schema.JD_DISTANCE_SAFE_FIELD].ge(p90_safe)
        & events[schema.JD_DISTANCE_SAFE_FIELD].notna()).astype(int)
    events[schema.EXTREME_FIELD] = (
        events[schema.JD_DISTANCE_SAFE_FIELD].ge(p95_safe)
        & events[schema.JD_DISTANCE_SAFE_FIELD].notna()).astype(int)

    candidate_types = []
    for row in events.to_dict('records'):
        label = text_semantics.classify_job_text_change(
            row, thresholds, distance_field=schema.JD_DISTANCE_SAFE_FIELD)
        candidate_types.append(label)
        candidate_counter[label] += 1
    events[schema.CANDIDATE_TYPE_FIELD] = candidate_types

    statistics = {
        'thresholds': thresholds,
        'added_skill_counter': added_skill_counter,
        'removed_skill_counter': removed_skill_counter,
        'candidate_counter': candidate_counter,
        'tfidf_vocabulary': (int(len(vectorizer_full.vocabulary_)) if matrix_full is not None else 0),
        'tfidf_vocabulary_safe': (int(len(vectorizer_safe.vocabulary_))
                                  if matrix_safe is not None else 0),
        'significant_full_caliber': int(events[schema.SIGNIFICANT_FULL_CALIBER_FIELD].sum()),
        'extreme_full_caliber': int(events[schema.EXTREME_FULL_CALIBER_FIELD].sum()),
        'dropped_by_salary_text': int(((events[schema.SIGNIFICANT_FULL_CALIBER_FIELD] == 1)
                                       & (events[schema.SIGNIFICANT_FIELD] == 0)).sum()),
        'entered_by_safe_caliber': int(((events[schema.SIGNIFICANT_FULL_CALIBER_FIELD] == 0)
                                        & (events[schema.SIGNIFICANT_FIELD] == 1)).sum()),
        'identical_text_versions': int(events['文本完全一致标志'].sum()),
        'truncated_pairs': int(events['版本对是否存在截断'].sum()),
    }
    return events[schema.JOB_TEXT_EVENT_COLUMNS], statistics


def _distribution_stats(series) -> dict:
    """描述性统计（均值 / 标准差 / 中位数 / P75 / P90 / P95）。"""
    values = pd.Series(series, dtype='float64').dropna()
    if values.empty:
        return {'样本数': 0, '均值': np.nan, '标准差': np.nan, '中位数': np.nan,
                'P75': np.nan, 'P90': np.nan, 'P95': np.nan}
    return {
        '样本数': int(values.size),
        '均值': round(float(values.mean()), 6),
        '标准差': round(float(values.std(ddof=1)) if values.size > 1 else 0.0, 6),
        '中位数': round(float(values.median()), 6),
        'P75': round(float(values.quantile(0.75)), 6),
        'P90': round(float(values.quantile(0.90)), 6),
        'P95': round(float(values.quantile(0.95)), 6),
    }


def build_audit_tables(events: pd.DataFrame, statistics: dict, artifacts: dict,
                       embedding: text_semantics.EmbeddingResult,
                       matcher: skill_extraction.SkillMatcher) -> dict:
    """构建 17 号岗位文本语义审计表（双口径 + token 截断审计，共 21 张子表）。"""
    id_field = schema.ID_FIELD
    thresholds = statistics['thresholds']
    total = len(events)
    full_distance = events[schema.JD_DISTANCE_FULL_FIELD]
    safe_distance = events[schema.JD_DISTANCE_SAFE_FIELD]
    full_valid = text_semantics.validate_semantic_values(
        full_distance, schema.JD_DISTANCE_FULL_FIELD)
    safe_valid = text_semantics.validate_semantic_values(
        safe_distance, schema.JD_DISTANCE_SAFE_FIELD)
    similarity_valid = text_semantics.validate_semantic_values(
        events[schema.JD_SIMILARITY_SAFE_FIELD], schema.JD_SIMILARITY_SAFE_FIELD,
        lower=-1.0, upper=1.0)

    overview = pd.DataFrame([
        {'指标': '岗位描述变化版本切换数', '数值': total},
        {'指标': '涉及岗位数', '数值': int(events[id_field].nunique())},
        {'指标': '文本完全一致的版本切换数', '数值': int(events['文本完全一致标志'].sum())},
        {'指标': '有技能新增的版本切换数', '数值': int((events['新增技能数'] > 0).sum())},
        {'指标': '有技能删除的版本切换数', '数值': int((events['删除技能数'] > 0).sum())},
        {'指标': '技能集合完全不变的版本切换数', '数值': int((events['技能变化数'] == 0).sum())},
        {'指标': '显著语义变化候选数（去薪资 ≥P90）', '数值': int(events[schema.SIGNIFICANT_FIELD].sum())},
        {'指标': '极端语义变化候选数（去薪资 ≥P95）', '数值': int(events[schema.EXTREME_FIELD].sum())},
        {'指标': '旧口径显著候选数（完整 ≥P90）',
         '数值': int(events[schema.SIGNIFICANT_FULL_CALIBER_FIELD].sum())},
        {'指标': '旧口径极端候选数（完整 ≥P95）',
         '数值': int(events[schema.EXTREME_FULL_CALIBER_FIELD].sum())},
        {'指标': '因薪资文本影响退出显著候选的事件数',
         '数值': int(statistics['dropped_by_salary_text'])},
        {'指标': '因新口径进入显著候选的事件数',
         '数值': int(statistics['entered_by_safe_caliber'])},
        {'指标': 'TF-IDF 词表规模（完整口径）', '数值': statistics['tfidf_vocabulary']},
        {'指标': 'TF-IDF 词表规模（去薪资口径）', '数值': statistics['tfidf_vocabulary_safe']},
        {'指标': '去薪资语义距离缺失数', '数值': int(safe_distance.isna().sum())},
        {'指标': '完整语义距离缺失数', '数值': int(full_distance.isna().sum())},
        {'指标': '版本对存在截断数', '数值': int(events['版本对是否存在截断'].sum())},
        {'指标': '句向量模型状态', '数值': embedding.status},
        {'指标': '句向量模型名称', '数值': embedding.model_name},
        {'指标': '句向量维度', '数值': int(embedding.dimension) if embedding.available else 0},
        {'指标': '模型最大 token 数', '数值': int(artifacts.get('model_max_tokens', 0))},
    ])

    # ---- 02 语义距离分布（双口径共用同一组区间边界） ----
    def _histogram(edges) -> pd.DataFrame:
        rows = []
        for left, right in zip(edges[:-1], edges[1:]):
            full_count = int(((full_distance >= left) & (full_distance < right)).sum())
            safe_count = int(((safe_distance >= left) & (safe_distance < right)).sum())
            rows.append({'区间下限': round(float(left), 4), '区间上限': round(float(right), 4),
                         '完整口径版本切换数': full_count,
                         '完整占比': round(full_count / max(total, 1), 6),
                         '去薪资口径版本切换数': safe_count,
                         '去薪资占比': round(safe_count / max(total, 1), 6)})
        return pd.DataFrame(rows)
    distance_max = max(1.0, float(pd.concat([full_distance, safe_distance]).max()))
    histogram = _histogram(np.linspace(0, distance_max, 21))

    # ---- 03 / 15 分位数来源 ----
    full_threshold_table = text_semantics.build_threshold_source(
        {key: value for key, value in thresholds.items()
         if key.startswith(schema.JD_DISTANCE_FULL_FIELD)})
    safe_threshold_table = text_semantics.build_threshold_source(
        {key: value for key, value in thresholds.items()
         if key.startswith(schema.JD_DISTANCE_SAFE_FIELD)},
        extra={'显著语义变化候选阈值分位': SIGNIFICANT_QUANTILE,
               '极端语义变化候选阈值分位': EXTREME_QUANTILE,
               '阈值确定方式': '先统计真实分布分位数，再结合人工复核样本确认（去薪资口径为正式主判据）'})

    # ---- 04 / 21 显著与极端候选 ----
    significant_columns = [id_field, '旧核心版本号', '新核心版本号',
                           schema.JD_DISTANCE_SAFE_FIELD, schema.JD_DISTANCE_FULL_FIELD,
                           schema.DISTANCE_GAP_FIELD, schema.TFIDF_SAFE_FIELD,
                           '新增技能集合', '删除技能集合', schema.CANDIDATE_TYPE_FIELD,
                           schema.SIGNIFICANT_FULL_CALIBER_FIELD, '岗位标题']
    significant = (events[events[schema.SIGNIFICANT_FIELD] == 1]
                   .sort_values(schema.JD_DISTANCE_SAFE_FIELD, ascending=False)
                   [significant_columns])
    extreme = (events[events[schema.EXTREME_FIELD] == 1]
               .sort_values(schema.JD_DISTANCE_SAFE_FIELD, ascending=False)
               [significant_columns])

    def _skill_table(counter: Counter, name: str) -> pd.DataFrame:
        table = pd.DataFrame(
            [{'技能标准名': skill,
              '技能一级类型': matcher.config.skill_to_family.get(skill, ''),
              '技能组': matcher.config.skill_to_group.get(skill, '其他'),
              name: count}
             for skill, count in counter.most_common()])
        if table.empty:
            table = pd.DataFrame(columns=['技能标准名', '技能一级类型', '技能组', name])
        return table

    llm_rows = []
    for skill in sorted(skill_extraction.llm_skill_set(matcher.config)):
        added = int(events['新增技能集合'].map(lambda items: skill in text_utils.as_list(items)).sum())
        removed = int(events['删除技能集合'].map(lambda items: skill in text_utils.as_list(items)).sum())
        llm_rows.append({'大模型相关技能': skill, '新增版本切换数': added, '删除版本切换数': removed,
                         '净变化': added - removed})
    llm_table = pd.DataFrame(llm_rows).sort_values('新增版本切换数', ascending=False)

    def _salary_table(distance_column: str, label: str, frame: pd.DataFrame) -> list:
        rows = []
        for name, flag in (('薪资变化版本', 1), ('薪资稳定版本', 0)):
            subset = frame[frame['是否同时薪资变化'] == flag]
            stats = _distribution_stats(subset[distance_column])
            rows.append({'口径': label, '分组': name, '版本切换数': int(len(subset)),
                         '语义距离均值': stats['均值'], '标准差': stats['标准差'],
                         '中位数': stats['中位数'], 'P75': stats['P75'],
                         'P90': stats['P90'], 'P95': stats['P95'],
                         '显著语义变化候选数': int(subset[schema.SIGNIFICANT_FIELD].sum()),
                         '新增技能事件数': int(subset['新增技能数'].sum())})
        return rows

    salary_old = pd.DataFrame(_salary_table(schema.JD_DISTANCE_FULL_FIELD, '完整口径（旧）', events))
    salary_old['说明'] = '旧口径，仅作回归对照，不作为正式结论'
    nontruncated = events[events['版本对是否存在截断'] == 0]
    salary_new = pd.DataFrame(_salary_table(schema.JD_DISTANCE_SAFE_FIELD, '去薪资口径（全样本）', events)
                              + _salary_table(schema.JD_DISTANCE_SAFE_FIELD, '去薪资口径（非截断样本）',
                                              nontruncated))
    salary_new['说明'] = '正式口径：仅描述性关联统计，不做因果推断，不执行统计显著性检验'

    # ---- 14 完整 vs 去薪资语义距离 ----
    caliber_rows = []
    for group_name, subset in (('总体', events),
                               ('薪资变化版本', events[events['是否同时薪资变化'] == 1]),
                               ('薪资稳定版本', events[events['是否同时薪资变化'] == 0])):
        full_stats = _distribution_stats(subset[schema.JD_DISTANCE_FULL_FIELD])
        safe_stats = _distribution_stats(subset[schema.JD_DISTANCE_SAFE_FIELD])
        gap_stats = _distribution_stats(subset[schema.DISTANCE_GAP_FIELD])
        caliber_rows.append({
            '分组': group_name,
            '有效样本数': safe_stats['样本数'],
            '完整口径均值': full_stats['均值'], '完整口径中位数': full_stats['中位数'],
            '完整口径P90': full_stats['P90'], '完整口径P95': full_stats['P95'],
            '去薪资口径均值': safe_stats['均值'], '去薪资口径中位数': safe_stats['中位数'],
            '去薪资口径P90': safe_stats['P90'], '去薪资口径P95': safe_stats['P95'],
            '完整减去薪资安全语义距离差_均值': gap_stats['均值'],
            '完整减去薪资安全语义距离差_中位数': gap_stats['中位数'],
            '完整减去薪资安全语义距离差_P90': gap_stats['P90'],
            '完整减去薪资安全语义距离差_最大值': (
                round(float(subset[schema.DISTANCE_GAP_FIELD].max()), 6)
                if subset[schema.DISTANCE_GAP_FIELD].notna().any() else np.nan),
            '差值大于0比例': (round(float((subset[schema.DISTANCE_GAP_FIELD] > 0).mean()), 6)
                        if subset[schema.DISTANCE_GAP_FIELD].notna().any() else np.nan),
        })
    caliber_table = pd.DataFrame(caliber_rows)
    caliber_table['说明'] = ('Embedding 距离不是可加性分解，该差值仅作诊断，'
                        '禁止解释为「薪资文本贡献率」')

    # ---- 16 / 17 token 长度与截断 ----
    token_table = pd.DataFrame(artifacts.get('token_distribution', pd.DataFrame()))
    truncation_table = pd.DataFrame(artifacts.get('token_stats', []))

    # ---- 19 非截断样本敏感性 ----
    sensitivity_rows = []
    for scope, subset in (('全样本', events), ('非截断样本', nontruncated),
                          ('至少一侧被截断', events[events['版本对是否存在截断'] == 1])):
        for column, label in ((schema.JD_DISTANCE_SAFE_FIELD, '去薪资口径'),
                              (schema.JD_DISTANCE_FULL_FIELD, '完整口径')):
            stats = _distribution_stats(subset[column])
            sensitivity_rows.append({
                '样本口径': scope, '语义口径': label, '版本切换数': int(len(subset)),
                '有效样本数': stats['样本数'], '均值': stats['均值'],
                '中位数': stats['中位数'], 'P90': stats['P90'], 'P95': stats['P95'],
                '显著语义变化候选数': int(subset[schema.SIGNIFICANT_FIELD].sum()),
            })
    sensitivity_table = pd.DataFrame(sensitivity_rows)
    sensitivity_table['说明'] = '不因截断比例高低删除样本，仅提供非截断样本对照'

    # ---- 20 数值检查 ----
    value_check = pd.DataFrame([full_valid, safe_valid, similarity_valid])

    # ---- 13 人工复核样本 ----
    review_rows = []
    for label, quantile in (('P50', 0.50), ('P75', 0.75), ('P90', 0.90), ('P95', 0.95)):
        target = thresholds[f'{schema.JD_DISTANCE_SAFE_FIELD}{label}']
        pool = events[events[schema.JD_DISTANCE_SAFE_FIELD].notna()].copy()
        pool['_distance_to_target'] = (pool[schema.JD_DISTANCE_SAFE_FIELD] - target).abs()
        sample = pool.nsmallest(REVIEW_SAMPLE_PER_QUANTILE, '_distance_to_target')
        for row in sample.to_dict('records'):
            review_rows.append({
                '分位点': label,
                '该分位阈值': round(float(target), 6),
                id_field: row[id_field],
                '旧版本': int(row['旧核心版本号']),
                '新版本': int(row['新核心版本号']),
                '旧文本摘要': str(row[schema.JD_RAW_FIELD] or '')[:120],
                '新文本摘要': str(row['新岗位描述'] or '')[:120],
                '完整语义距离': row[schema.JD_DISTANCE_FULL_FIELD],
                '去薪资语义距离': row[schema.JD_DISTANCE_SAFE_FIELD],
                '新增技能': '、'.join(text_utils.as_list(row['新增技能集合'])),
                '删除技能': '、'.join(text_utils.as_list(row['删除技能集合'])),
                '候选类型': row[schema.CANDIDATE_TYPE_FIELD],
                '人工结论': '',
                '备注': '',
            })
    review_table = pd.DataFrame(review_rows)

    def _section_table(prefix: str, name: str) -> pd.DataFrame:
        values = events[prefix].dropna()
        rows = [{'指标': f'{name}有效版本切换数', '数值': int(len(values))},
                {'指标': f'{name}缺失数', '数值': int(events[prefix].isna().sum())},
                {'指标': f'{name}均值', '数值': round(float(values.mean()), 6) if not values.empty else np.nan},
                {'指标': f'{name}中位数', '数值': float(values.median()) if not values.empty else np.nan},
                {'指标': f'{name}P90', '数值': float(values.quantile(0.9)) if not values.empty else np.nan},
                {'指标': f'{name}最大值', '数值': float(values.max()) if not values.empty else np.nan}]
        return pd.DataFrame(rows)

    candidate_table = pd.DataFrame(
        [{'语义变化候选类型': label, '版本切换数': count,
          '占比': round(count / total, 6) if total else 0,
          '是否含候选字样': int('候选' in label)}
         for label, count in statistics['candidate_counter'].most_common()])

    return {
        '01_语义事件总体统计': overview,
        '02_语义距离分布': histogram,
        '03_完整语义距离分位数': full_threshold_table,
        '04_显著语义变化候选': significant,
        '05_新增技能统计': _skill_table(statistics['added_skill_counter'], '新增版本切换数'),
        '06_删除技能统计': _skill_table(statistics['removed_skill_counter'], '删除版本切换数'),
        '07_大模型技能新增统计': llm_table,
        '08_薪资变化vs完整语义_旧口径': salary_old,
        '09_职责语义变化': _section_table('职责语义距离', '职责语义距离'),
        '10_要求语义变化': _section_table('任职要求语义距离', '任职要求语义距离'),
        '11_技能段语义变化': _section_table('技能段语义距离', '技能段语义距离'),
        '12_语义变化候选类型': candidate_table,
        '13_人工复核样本': review_table,
        '14_完整vs去薪资语义距离': caliber_table,
        '15_去薪资语义距离分位数': safe_threshold_table,
        '16_BGE_Token长度分布': token_table,
        '17_BGE_截断统计': truncation_table,
        '18_薪资变化vs去薪资语义': salary_new,
        '19_非截断样本敏感性': sensitivity_table,
        '20_语义距离数值检查': value_check,
        '21_极端语义变化候选': extreme,
    }


def build_record_lines(metrics: dict, audit: dict, embedding: text_semantics.EmbeddingResult,
                       artifacts: dict) -> list:
    """生成 10 号岗位文本语义时序阶段记录。"""
    overview = audit['01_语义事件总体统计'].set_index('指标')['数值']
    quantiles = audit['15_去薪资语义距离分位数']
    counters = artifacts.get('counters', {})
    lines = [
        '# 阶段记录：Stage 08 岗位描述语义时序（完整 / 去薪资双口径）',
        '',
        '> 本文件由 `scripts/ch3_data/08_build_job_text_semantics.py` 生成，全部数字来自真实运行结果。',
        '',
        '## 1. 时序口径',
        '',
        '- 观测单位：同一岗位的**相邻核心版本**文本对 `V(k-1) → V(k)`；',
        '- 事件时间：新核心版本的**版本首次观测时间**（不使用发布时间替代）；',
        '- 口径名称：非平衡多时点文本观测（岗位版本文本语义时序）；',
        '- 一次核心版本切换最多记作一条文本语义变化事件，不因搜索分类命中数重复计数。',
        '',
        '## 2. 双语义口径（Refinement R1 核心变更）',
        '',
        '| 口径 | 输入文本 | 用途 |',
        '| --- | --- | --- |',
        '| 完整口径 `_完整` | 岗位描述_语义分析版 | 页面整体变化辅助指标、与旧结果对照 |',
        '| 去薪资口径 `_去薪资` | 岗位描述_模型安全版 | **正式主判据**：职责/要求/技能真实语义变化、薪资关联分析 |',
        '',
        '> 完整减去薪资安全语义距离差仅作诊断：Embedding 距离不是可加性分解，'
        '禁止解释为「薪资文本贡献率」。',
        '',
        '## 3. 事件规模',
        '',
        f"- 岗位描述变化版本切换数：{metrics['event_rows']}；",
        f"- 涉及岗位数：{int(overview['涉及岗位数'])}；",
        f"- 文本完全一致的版本切换数：{int(overview['文本完全一致的版本切换数'])}；",
        f"- 有技能新增 / 删除的版本切换数：{int(overview['有技能新增的版本切换数'])} / "
        f"{int(overview['有技能删除的版本切换数'])}；",
        f"- 技能集合完全不变的版本切换数：{int(overview['技能集合完全不变的版本切换数'])}。",
        '',
        '## 4. 两类相似度严格区分',
        '',
        '| 指标 | 性质 | 用途 |',
        '| --- | --- | --- |',
        '| Token Jaccard / 编辑相似度 | 字面词面 | 词形变化描述 |',
        '| TF-IDF 余弦相似度 | 词面基线 | 关键词变化对照，**不等于深层语义** |',
        '| Embedding 余弦相似度 | 深层语义 | 语义相似度，语义距离 = 1 − 相似度 |',
        '',
        f"- 句向量模型：`{embedding.model_name}`；状态 {embedding.status}；"
        f"维度 {int(embedding.dimension) if embedding.available else 'NOT_RUN'}；",
        f"- TF-IDF 词表规模：完整口径 {int(overview['TF-IDF 词表规模（完整口径）'])}、"
        f"去薪资口径 {int(overview['TF-IDF 词表规模（去薪资口径）'])}。",
        '',
        '## 5. 安全文本 Embedding 复用策略',
        '',
        f"- 安全文本总数（非空）：{counters.get('安全文本总数', '—')}；",
        f"- 缓存整体复用数：{counters.get('缓存整体复用数', '—')}；",
        f"- 直接复用完整版向量数：{counters.get('直接复用完整版向量数', '—')}；",
        f"- 需要新编码的样本数：{counters.get('需要新编码的样本数', '—')}；",
        f"- 哈希去重后新编码文本数：{counters.get('哈希去重后新编码文本数', '—')}；",
        f"- 去重节省编码数：{counters.get('去重节省编码数', '—')}；",
        f"- 缓存复用比例：{counters.get('缓存复用比例', '—')}；",
        f"- 编码来源：{counters.get('安全文本编码来源', '—')}。",
        '',
        '## 6. BGE token 长度与截断审计（真实 tokenizer）',
        '',
        '| 文本类型 | 文本数 | Token P50 | Token P95 | Token P99 | Token Max | 截断文本数 | 截断比例 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for row in audit['17_BGE_截断统计'].itertuples(index=False):
        lines.append(f'| {row.文本类型} | {int(row.文本数)} | {row.Token_P50:.1f} | {row.Token_P95:.1f} | '
                     f'{row.Token_P99:.1f} | {int(row.Token_Max)} | {int(row.截断文本数)} | '
                     f'{row.截断比例:.2%} |')
    lines += [
        '',
        f"- 模型最大 token 数：{int(artifacts.get('model_max_tokens', 0))}"
        '（由模型 tokenizer 真实计算，禁止用字符数代替）；',
        f"- 版本对存在截断数：{int(overview['版本对存在截断数'])}；",
        '- 不因截断比例高低删除样本，另出「非截断样本」对照统计（19 号子表）。',
        '',
        '## 7. 阈值来源（禁止拍脑袋）',
        '',
        '| 指标 | 阈值 | 样本数 | 来源 |',
        '| --- | --- | --- | --- |',
    ]
    for row in quantiles.itertuples(index=False):
        value = row.阈值
        threshold = value if isinstance(value, str) else ('—' if pd.isna(value) else f'{float(value):.6f}')
        sample = '—' if pd.isna(row.样本数) else f'{int(row.样本数)}'
        lines.append(f'| {row.指标} | {threshold} | {sample} | {row.来源} |')
    lines += [
        '',
        f"- 显著语义变化候选：去薪资语义距离 ≥ P90；极端候选：≥ P95；",
        f"- 候选数量（去薪资口径）：P90 候选 {metrics['significant_events']} 条、"
        f"P95 候选 {metrics['extreme_events']} 条；",
        f"- 旧口径候选数（完整语义距离）：P90 {metrics['significant_full_caliber']} 条、"
        f"P95 {metrics['extreme_full_caliber']} 条；",
        f"- 因薪资文本影响退出显著候选：{metrics['dropped_by_salary_text']} 条；"
        f"因新口径进入显著候选：{metrics['entered_by_safe_caliber']} 条。",
        '',
        '## 8. 薪资变化 vs JD 语义变化（正式口径）',
        '',
        '| 口径 | 分组 | 版本切换数 | 均值 | 标准差 | 中位数 | P90 | P95 | 显著候选数 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for row in audit['18_薪资变化vs去薪资语义'].itertuples(index=False):
        lines.append(f'| {row.口径} | {row.分组} | {int(row.版本切换数)} | {row.语义距离均值} | '
                     f'{row.标准差} | {row.中位数} | {row.P90} | {row.P95} | '
                     f'{int(row.显著语义变化候选数)} |')
    lines += [
        '',
        '> 以上仅为**描述性与关联性**统计；本轮不执行统计显著性检验，'
        '后续分析阶段再做 Mann–Whitney U / Cliff\'s delta / Bootstrap CI，且不得提前因果解释。',
        '',
        '## 9. 分段语义变化（不做人为加权综合）',
        '',
        '| 分段 | 有效版本切换数 | 缺失数 | 均值 | 中位数 | P90 |',
        '| --- | --- | --- | --- | --- | --- |',
    ]
    for name, table, prefix in (('职责段', audit['09_职责语义变化'], '职责语义距离'),
                                ('任职要求段', audit['10_要求语义变化'], '任职要求语义距离'),
                                ('技能段', audit['11_技能段语义变化'], '技能段语义距离')):
        values = table.set_index('指标')['数值']
        lines.append(f"| {name} | {int(values[f'{prefix}有效版本切换数'])} | "
                     f"{int(values[f'{prefix}缺失数'])} | {values[f'{prefix}均值']} | "
                     f"{values[f'{prefix}中位数']} | {values[f'{prefix}P90']} |")
    lines += [
        '',
        '> 四个语义维度（去薪资全文 / 职责 / 任职要求 / 技能段）分别解释，'
        '禁止人为设定 0.35/0.30/0.35 这类未经验证的加权综合距离。',
        '',
        '## 10. 语义变化候选类型',
        '',
        '| 语义变化候选类型 | 版本切换数 | 占比 |',
        '| --- | --- | --- |',
    ]
    for row in audit['12_语义变化候选类型'].itertuples(index=False):
        lines.append(f'| {row.语义变化候选类型} | {int(row.版本切换数)} | {row.占比:.2%} |')
    lines += [
        '',
        '> 全部标签均保留「候选」字样，未经人工复核不得升级为确定结论。',
        '',
        '## 11. 版本历史摘要回写',
        '',
        '| 字段 | 口径 |',
        '| --- | --- |',
        '| 相对上一版本JD语义距离 | **兼容字段，口径已切换为去薪资语义距离** |',
        '| 相对上一版本JD语义距离_完整 | 完整文本口径 |',
        '| 相对上一版本JD语义距离_去薪资 | 去薪资口径（与兼容字段数值相同） |',
        '| 相对上一版本新增/删除技能数 | 相邻核心版本技能集合差 |',
        '| 是否显著JD语义变化候选 | 去薪资语义距离是否 ≥ P90 |',
        '',
        f"- 回写后版本历史列数：{metrics['version_columns']}；",
        '',
        '## 12. 句向量存储与复现',
        '',
        f"- 向量文件：`{metrics['embeddings_npz']}`；索引表：`{metrics['embeddings_index']}`；",
        f"- 复现记录：`{metrics['embeddings_meta']}`；",
        f"- 向量数组：{metrics['embedding_shapes']}；",
        '- 缓存键 = 文本类型 + 语料 SHA256 + 模型配置（job_text_full / job_text_safe / '
        'duty_text / require_text / skill_text）；',
        '- 高维向量不写入 `job_details_unique.parquet` / `job_version_history.parquet`。',
        '',
        '## 13. 运行门禁',
        '',
        '| 门禁项 | 状态 |',
        '| --- | --- |',
    ]
    for name, result in metrics['gates'].items():
        lines.append(f'| {name} | {result} |')
    lines += [
        '',
        '## 14. 产物清单',
        '',
        f"- 文本语义事件：`{metrics['events_path']}`；",
        f"- 审计表：`{metrics['audit_path']}`；",
        f"- 指标 JSON：`{metrics['metrics_path']}`。",
        '',
    ]
    return lines


def main() -> int:
    quality.configure_logging()
    gates = quality.GateRegistry(STAGE)

    quality.stage_banner(STAGE, TITLE)
    # 覆盖正式结果之前先固化修复前基线（幂等：已捕获则不覆盖）
    refinement.capture_job_baseline()
    corpus, versions, change_events = load_inputs()
    print(f'输入: 语料 {len(corpus)} 行 / 版本历史 {len(versions)} 行 / 字段变化事件 {len(change_events)} 行')
    pairs = build_pairs(versions)
    expected_events = int(sum(max(count - 1, 0)
                              for count in versions.groupby(schema.ID_FIELD).size()))
    gates.check('JOB_TEXT_VERSION_RECONCILE',
                len(corpus) == len(versions)
                and corpus[schema.ID_FIELD].nunique() == versions[schema.ID_FIELD].nunique(),
                f'语料 {len(corpus)} 行与版本历史 {len(versions)} 行一一对应，'
                f'岗位数一致（{corpus[schema.ID_FIELD].nunique()}）')

    config = skill_extraction.load_skill_config()
    matcher = skill_extraction.SkillMatcher(config)

    embedding = text_semantics.load_embedding_model()
    if embedding.available:
        gates.check('EMBEDDING_MODEL_LOAD', True,
                    f'{embedding.model_name} 维度 {embedding.dimension}，'
                    f'device={embedding.device}，用时 {embedding.load_seconds:.1f}s')
    else:
        gates.mark_not_run('EMBEDDING_MODEL_LOAD',
                           f'句向量模型不可用：{embedding.reason}（不使用 TF-IDF 冒充深层语义）')

    artifacts = build_embeddings(corpus, embedding)
    events, statistics = build_events(corpus, versions, change_events, matcher,
                                      embedding, artifacts)
    counters = artifacts.get('counters', {})
    audit_tables = build_audit_tables(events, statistics, artifacts, embedding, matcher)

    if embedding.available:
        arrays = artifacts['arrays']
        gates.check('EMBEDDING_BUILD',
                    len(arrays) == 5
                    and all(array.shape[1] == embedding.dimension for array in arrays.values()),
                    f'生成 {len(arrays)} 组向量（完整 / 去薪资 / 职责 / 要求 / 技能），'
                    f'索引 {len(artifacts["index_rows"])} 行')
    else:
        gates.mark_not_run('EMBEDDING_BUILD', '句向量模型不可用，未生成任何向量（禁止伪造）')

    # ---- 安全文本向量与完整版逐行一致性（完整文本 == 安全文本时） ----
    if embedding.available:
        full_mask = (corpus[schema.JD_SEMANTIC_FIELD].fillna('').astype(str).str.strip()
                     == corpus[schema.JD_SAFE_FIELD].fillna('').astype(str).str.strip())
        identical_positions = corpus.index[full_mask
                                           & corpus[schema.JD_SEMANTIC_FIELD]
                                           .fillna('').astype(str).str.strip().ne('')]
        full_index = artifacts['row_index'][text_semantics.TEXT_TYPE_JOB_FULL]
        safe_index = artifacts['row_index'][SAFE_TEXT_TYPE]
        full_matrix = artifacts['arrays'][text_semantics.TEXT_TYPE_JOB_FULL]
        safe_matrix = artifacts['arrays'][SAFE_TEXT_TYPE]
        max_diff = 0.0
        compared = 0
        for position in identical_positions:
            record = corpus.iloc[int(position)]
            key = (record[schema.ID_FIELD], int(record[schema.CORE_VERSION_FIELD]))
            full_row, safe_row = full_index.get(key), safe_index.get(key)
            if full_row is None or safe_row is None:
                continue
            compared += 1
            max_diff = max(max_diff, float(np.abs(full_matrix[full_row]
                                                  - safe_matrix[safe_row]).max()))
        gates.check('FULL_SAFE_EMBEDDING_REUSE',
                    compared > 0 and max_diff <= VECTOR_ATOL
                    and counters.get('缓存整体复用数', 0)
                    + counters.get('直接复用完整版向量数', 0)
                    + counters.get('需要新编码的样本数', 0)
                    == counters.get('安全文本总数', -1)
                    and counters.get('需要新编码的样本数', 0)
                    - counters.get('哈希去重后新编码文本数', 0)
                    == counters.get('去重节省编码数', -1)
                    and (counters.get('缓存整体复用数', 0) > 0
                         or counters.get('直接复用完整版向量数', 0)
                         == counters.get('完整与安全文本相同的行数', -1)),
                    f'{compared} 个「完整文本 == 安全文本」版本两套向量逐行一致'
                    f'（最大差异 {max_diff:.2e} ≤ {VECTOR_ATOL}）；'
                    f'安全文本 {counters.get("安全文本总数", 0)} 行 = 缓存整体复用 '
                    f'{counters.get("缓存整体复用数", 0)} + 复用完整版向量 '
                    f'{counters.get("直接复用完整版向量数", 0)} + 需编码 '
                    f'{counters.get("需要新编码的样本数", 0)}，其中哈希去重后实编码 '
                    f'{counters.get("哈希去重后新编码文本数", 0)}（去重节省 '
                    f'{counters.get("去重节省编码数", 0)}）')
        gates.check('SAFE_TEXT_EMBEDDING_BUILD',
                    safe_matrix.shape == (counters.get('安全文本总数', -1), embedding.dimension)
                    and int(corpus[schema.JD_SAFE_RESIDUAL_FIELD].sum()) == 0,
                    f'安全文本向量 {safe_matrix.shape[0]} 行 × {safe_matrix.shape[1]} 维，'
                    f'模型安全版薪资残留 {int(corpus[schema.JD_SAFE_RESIDUAL_FIELD].sum())} 处，'
                    f'编码来源：{counters.get("安全文本编码来源", "")}')
    else:
        gates.mark_not_run('FULL_SAFE_EMBEDDING_REUSE', '句向量不可用，无法核对一致性')
        gates.mark_not_run('SAFE_TEXT_EMBEDDING_BUILD', '句向量不可用，未生成安全文本向量')

    distance_report = text_semantics.validate_semantic_values(
        events[schema.JD_DISTANCE_FULL_FIELD], schema.JD_DISTANCE_FULL_FIELD)
    if embedding.available:
        gates.check('SEMANTIC_DISTANCE_CHECK',
                    distance_report['是否通过'] and distance_report['Inf数量'] == 0,
                    f'完整语义距离 NaN {distance_report["缺失数"]}、Inf {distance_report["Inf数量"]}、'
                    f'越界 {distance_report["越界数量"]}')
    else:
        gates.mark_not_run('SEMANTIC_DISTANCE_CHECK', '句向量不可用，语义距离列为缺失值')

    safe_report = text_semantics.validate_semantic_values(
        events[schema.JD_DISTANCE_SAFE_FIELD], schema.JD_DISTANCE_SAFE_FIELD)
    similarity_report = text_semantics.validate_semantic_values(
        events[schema.JD_SIMILARITY_SAFE_FIELD], schema.JD_SIMILARITY_SAFE_FIELD,
        lower=-1.0, upper=1.0)
    if embedding.available:
        gates.check('SAFE_SEMANTIC_DISTANCE_CHECK',
                    safe_report['是否通过'] and similarity_report['是否通过'],
                    f'去薪资语义距离 NaN {safe_report["缺失数"]}、Inf {safe_report["Inf数量"]}、'
                    f'越界 {safe_report["越界数量"]}；去薪资相似度越界 '
                    f'{similarity_report["越界数量"]}')
    else:
        gates.mark_not_run('SAFE_SEMANTIC_DISTANCE_CHECK', '句向量不可用，去薪资语义距离缺失')

    salary_changed = events[events['是否同时薪资变化'] == 1]
    salary_stable = events[events['是否同时薪资变化'] == 0]
    same_safe_pairs = 0
    nonzero_same_safe = 0
    salary_only_changed = 0
    if embedding.available and not salary_changed.empty:
        # 真实可验证的不变量：去薪资文本完全相同的版本对，去薪资语义距离必须为 0
        # （即纯薪资文字调整在正式口径下不再产生语义距离）
        version_text = {(record[schema.ID_FIELD], int(record[schema.CORE_VERSION_FIELD])):
                        (record[schema.JD_SAFE_FIELD] or '', record[schema.JD_SEMANTIC_FIELD] or '')
                        for record in corpus.to_dict('records')}
        for row in events.to_dict('records'):
            job_id = row[schema.ID_FIELD]
            old_safe, old_full = version_text.get((job_id, int(row['旧核心版本号'])), ('', ''))
            new_safe, new_full = version_text.get((job_id, int(row['新核心版本号'])), ('', ''))
            if not old_safe.strip() or not new_safe.strip():
                continue
            if old_safe.strip() != new_safe.strip():
                continue
            same_safe_pairs += 1
            distance = row[schema.JD_DISTANCE_SAFE_FIELD]
            if not np.isnan(distance) and abs(float(distance)) > VECTOR_ATOL:
                nonzero_same_safe += 1
            if old_full.strip() != new_full.strip():
                salary_only_changed += 1
        mean_safe = float(salary_changed[schema.JD_DISTANCE_SAFE_FIELD].mean())
        mean_full = float(salary_changed[schema.JD_DISTANCE_FULL_FIELD].mean())
        gates.check('SAFE_SALARY_ALIGNMENT',
                    len(salary_changed) + len(salary_stable) == len(events)
                    and salary_changed[schema.JD_DISTANCE_SAFE_FIELD].notna().any()
                    and salary_stable[schema.JD_DISTANCE_SAFE_FIELD].notna().any()
                    and same_safe_pairs > 0 and nonzero_same_safe == 0,
                    f'薪资变化版本 {len(salary_changed)} 条 / 薪资稳定版本 {len(salary_stable)} 条，'
                    f'合计等于事件总数 {len(events)}；去薪资文本相同的版本对 {same_safe_pairs} 条，'
                    f'其中去薪资语义距离非零 {nonzero_same_safe} 条（纯薪资文字调整不产生语义距离，'
                    f'其中 {salary_only_changed} 条仅薪资文本变化）；'
                    f'薪资变化组去薪资均值 {mean_safe:.6f} / 完整均值 {mean_full:.6f}（仅描述性对照）')
    else:
        gates.mark_not_run('SAFE_SALARY_ALIGNMENT', '句向量不可用或薪资变化样本为空')

    token_table = pd.DataFrame(artifacts.get('token_stats', []))
    index_frame = pd.DataFrame(artifacts['index_rows'])
    if embedding.available and not token_table.empty:
        max_tokens = int(artifacts['model_max_tokens'])
        exceed_ok = bool((index_frame[schema.TOKEN_EXCEED_FIELD]
                          == np.maximum(index_frame[schema.TOKEN_RAW_COUNT_FIELD] - max_tokens,
                                        0)).all())
        retention_ok = bool(index_frame[schema.TOKEN_RETENTION_FIELD].between(0, 1).all())
        gates.check('TOKEN_LENGTH_AUDIT',
                    max_tokens > 0 and len(token_table) == 5 and exceed_ok and retention_ok,
                    f'{len(token_table)} 类文本由真实 tokenizer 统计 token 长度，'
                    f'模型最大 token 数 {max_tokens}；索引 {len(index_frame)} 行满足'
                    '超出token数 = max(原始-最大, 0)，理论保留比例 ∈ (0,1]')
    else:
        gates.mark_not_run('TOKEN_LENGTH_AUDIT', '句向量模型不可用，无法进行真实 token 审计')

    sensitivity_table = audit_tables['19_非截断样本敏感性']
    truncated_pairs = int(events['版本对是否存在截断'].sum())
    full_scope_rows = int(sensitivity_table.loc[sensitivity_table['样本口径'] == '全样本',
                                                '版本切换数'].iloc[0] * 2)
    gates.check('TRUNCATION_SENSITIVITY',
                not sensitivity_table.empty
                and full_scope_rows == len(events) * 2
                and int(sensitivity_table.loc[sensitivity_table['样本口径'] == '至少一侧被截断',
                                              '版本切换数'].iloc[0]) == truncated_pairs,
                f'全样本 {len(events)} 条 / 非截断 {len(events) - truncated_pairs} 条 / '
                f'至少一侧截断 {truncated_pairs} 条，两套描述性对照均已输出（未删除样本）')

    thresholds = statistics['thresholds']
    if statistics['tfidf_vocabulary'] > 0 and statistics['tfidf_vocabulary_safe'] > 0:
        gates.check('TFIDF_BASELINE_BUILD', True,
                    f'TF-IDF 词表（完整 {statistics["tfidf_vocabulary"]} / '
                    f'去薪资 {statistics["tfidf_vocabulary_safe"]}），仅作词面基线，不冒充深层语义')
    else:
        gates.check('TFIDF_BASELINE_BUILD', False, 'TF-IDF 词表为空')

    event_path = io_utils.write_parquet(events, project_paths.JOB_TEXT_EVENTS_PARQUET)
    gates.check('JOB_TEXT_EVENT_BUILD',
                len(events) == len(pairs) == expected_events
                and '候选' in ''.join(events[schema.CANDIDATE_TYPE_FIELD].unique()),
                f'{len(events)} 条文本语义事件 == sum(max(核心版本数-1,0)) = {expected_events}，'
                '候选类型标注全部保留「候选」字样')

    # ---- 版本历史摘要回写（仅摘要字段，禁止把 NLP 明细塞进主版本表） ----
    summary = events[[schema.ID_FIELD, '新核心版本号', schema.JD_DISTANCE_SAFE_FIELD,
                      schema.JD_DISTANCE_FULL_FIELD, '新增技能数', '删除技能数',
                      schema.SIGNIFICANT_FIELD]].rename(columns={
        '新核心版本号': schema.CORE_VERSION_FIELD,
        schema.JD_DISTANCE_SAFE_FIELD: '相对上一版本JD语义距离_去薪资',
        schema.JD_DISTANCE_FULL_FIELD: '相对上一版本JD语义距离_完整',
        '新增技能数': '相对上一版本新增技能数',
        '删除技能数': '相对上一版本删除技能数',
        schema.SIGNIFICANT_FIELD: '是否显著JD语义变化候选',
    })
    summary['相对上一版本JD语义距离'] = summary['相对上一版本JD语义距离_去薪资']
    summary = summary[[schema.ID_FIELD, schema.CORE_VERSION_FIELD,
                       *schema.VERSION_TEXT_SUMMARY_FIELDS]]
    base_versions = versions[[column for column in versions.columns
                              if column not in schema.VERSION_TEXT_SUMMARY_FIELDS]]
    versions_enriched = base_versions.merge(
        summary, on=[schema.ID_FIELD, schema.CORE_VERSION_FIELD], how='left')
    versions_enriched['是否显著JD语义变化候选'] = (
        versions_enriched['是否显著JD语义变化候选'].fillna(0).astype(int))
    write_order = list(base_versions.columns) + list(schema.VERSION_TEXT_SUMMARY_FIELDS)
    versions_enriched = versions_enriched[write_order]
    io_utils.write_parquet(versions_enriched, project_paths.PROCESSED_VERSION_HISTORY_PARQUET)
    compat_ok = bool((versions_enriched['相对上一版本JD语义距离'].fillna(-1)
                      - versions_enriched['相对上一版本JD语义距离_去薪资'].fillna(-1)).abs()
                     .lt(1e-9).all())
    gates.check('JOB_TEXT_VERSION_SUMMARY',
                set(schema.VERSION_TEXT_SUMMARY_FIELDS) <= set(versions_enriched.columns)
                and compat_ok,
                f'版本历史新增摘要字段 {list(schema.VERSION_TEXT_SUMMARY_FIELDS)}，'
                f'共 {versions_enriched.shape[1]} 列；兼容字段「相对上一版本JD语义距离」'
                '口径已切换到去薪资语义距离')

    # ---- 句向量落盘与复现记录 ----
    if embedding.available and artifacts['index_rows']:
        index_frame = pd.DataFrame(artifacts['index_rows'])[schema.EMBEDDING_INDEX_COLUMNS]
        meta = embedding.reproduction_record(
            input_field=f'{schema.JD_SEMANTIC_FIELD} / {schema.JD_SAFE_FIELD} 及其分段')
        meta['模型配置'] = text_semantics.model_config(embedding)
        meta['语料指纹'] = artifacts['fingerprints']
        meta['语料行数'] = int(len(corpus))
        meta['向量数组形状'] = {name: list(array.shape)
                           for name, array in artifacts['arrays'].items()}
        meta['索引行数'] = int(len(index_frame))
        meta['事件总数'] = int(len(events))
        meta['模型最大token数'] = int(artifacts['model_max_tokens'])
        meta['安全文本编码统计'] = counters
        meta['缓存状态'] = artifacts['cache_status']
        saved = text_semantics.save_embeddings(
            project_paths.JOB_TEXT_EMBEDDINGS_NPZ,
            project_paths.JOB_TEXT_EMBEDDING_INDEX_PARQUET,
            artifacts['arrays'], index_frame, meta)
        artifacts['saved'] = saved
    else:
        artifacts['saved'] = {'npz': 'NOT_RUN', 'index': 'NOT_RUN', 'meta': 'NOT_RUN', 'keys': {}}

    audit_path = io_utils.write_excel(project_paths.TABLES_DIR / project_paths.TABLE_JOB_TEXT_SEMANTIC,
                                      audit_tables)
    gates.check('JOB_TEXT_AUDIT_EXPORT',
                event_path.exists() and audit_path.exists()
                and {'14_完整vs去薪资语义距离', '15_去薪资语义距离分位数', '16_BGE_Token长度分布',
                     '17_BGE_截断统计', '18_薪资变化vs去薪资语义',
                     '19_非截断样本敏感性', '13_人工复核样本'} <= set(audit_tables),
                f'文本事件表与 {len(audit_tables)} 张语义审计子表已写出（含双口径与截断审计）')

    metrics = {
        'corpus_rows': int(len(corpus)),
        'event_rows': int(len(events)),
        'event_jobs': int(events[schema.ID_FIELD].nunique()),
        'events_path': project_paths.relative_to_root(event_path),
        'audit_path': project_paths.relative_to_root(audit_path),
        'version_columns': int(versions_enriched.shape[1]),
        'significant_events': int(events[schema.SIGNIFICANT_FIELD].sum()),
        'extreme_events': int(events[schema.EXTREME_FIELD].sum()),
        'significant_full_caliber': int(statistics['significant_full_caliber']),
        'extreme_full_caliber': int(statistics['extreme_full_caliber']),
        'dropped_by_salary_text': int(statistics['dropped_by_salary_text']),
        'entered_by_safe_caliber': int(statistics['entered_by_safe_caliber']),
        'identical_text_versions': int(statistics['identical_text_versions']),
        'safe_identical_pairs': int(same_safe_pairs),
        'salary_only_changed_pairs': int(salary_only_changed),
        'salary_residual': int(corpus[schema.JD_SAFE_RESIDUAL_FIELD].sum()),
        'reused_safe_vectors': int(counters.get('直接复用完整版向量数', 0)),
        'new_safe_encodings': int(counters.get('哈希去重后新编码文本数', 0)),
        'safe_text_total': int(counters.get('安全文本总数', 0)),
        'safe_cache_reused': int(counters.get('缓存整体复用数', 0)),
        'safe_cache_reuse_ratio': counters.get('缓存复用比例', 0.0),
        'new_llm_events': int(events['是否新增LLM'].sum()),
        'new_rag_events': int(events['是否新增RAG'].sum()),
        'new_agent_events': int(events['是否新增Agent'].sum()),
        'new_llm_related_events': int(events['是否新增大模型相关技能'].sum()),
        'model_max_tokens': int(artifacts.get('model_max_tokens', 0)),
        'truncated_pairs': int(statistics['truncated_pairs']),
        'semantic_p50': thresholds[f'{schema.JD_DISTANCE_SAFE_FIELD}P50'],
        'semantic_p75': thresholds[f'{schema.JD_DISTANCE_SAFE_FIELD}P75'],
        'semantic_p90': thresholds[f'{schema.JD_DISTANCE_SAFE_FIELD}P90'],
        'semantic_p95': thresholds[f'{schema.JD_DISTANCE_SAFE_FIELD}P95'],
        'semantic_p50_full': thresholds[f'{schema.JD_DISTANCE_FULL_FIELD}P50'],
        'semantic_p90_full': thresholds[f'{schema.JD_DISTANCE_FULL_FIELD}P90'],
        'semantic_p95_full': thresholds[f'{schema.JD_DISTANCE_FULL_FIELD}P95'],
        'embedding_status': embedding.status,
        'embedding_model': embedding.model_name,
        'embedding_dimension': int(embedding.dimension) if embedding.available else 0,
        'embeddings_npz': artifacts['saved']['npz'],
        'embeddings_index': artifacts['saved']['index'],
        'embeddings_meta': artifacts['saved']['meta'],
        'embedding_shapes': {name: list(array.shape)
                             for name, array in artifacts['arrays'].items()},
        'caliber_summary': audit_tables['14_完整vs去薪资语义距离'].to_dict('records'),
        'safe_thresholds': audit_tables['15_去薪资语义距离分位数'].to_dict('records'),
        'token_stats': audit_tables['17_BGE_截断统计'].to_dict('records'),
        'salary_alignment': audit_tables['18_薪资变化vs去薪资语义'].to_dict('records'),
        'truncation_sensitivity_note': (
            f"全样本 {len(events)} 条 / 非截断 {len(events) - statistics['truncated_pairs']} 条 / "
            f"至少一侧截断 {statistics['truncated_pairs']} 条"),
        'metrics_path': project_paths.relative_to_root(
            project_paths.METRICS_DIR / f'{STAGE}.json'),
    }
    metrics['gates'] = {name: result['status'] for name, result in gates.results.items()}
    io_utils.write_json(project_paths.METRICS_DIR / f'{STAGE}.json', metrics)

    record_path = io_utils.write_markdown(
        project_paths.RECORDS_DIR / project_paths.RECORD_JOB_TEXT_SEMANTIC,
        build_record_lines(metrics, audit_tables, embedding, artifacts))
    print(f'阶段记录: {project_paths.relative_to_root(record_path)}')

    quality.print_gates(gates.results, order=quality.STAGE_GATE_MAP[STAGE])
    gates.save()
    return 0


if __name__ == '__main__':
    sys.exit(main())

# -*- coding: utf-8 -*-
"""文本语义模块：句向量、TF-IDF 词面基线、语义距离、阈值与变化候选分类。

严格区分两类相似度（禁止混用）：

- ``TF-IDF 余弦相似度``：**词面**基线，用于对照与关键词分析；
- ``Embedding 余弦相似度``：**深层语义**，来自 sentence-transformers 真实句向量。

若句向量模型无法加载，本模块**明确返回 NOT_RUN**，绝不伪造向量、不随机生成、
也不用 TF-IDF 冒充深层语义。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import io_utils, project_paths, schema, text_utils

# ---- 句向量模型默认配置（真实模型，非替代品） ----
EMBEDDING_MODEL_NAME = 'BAAI/bge-small-zh-v1.5'
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_NORMALIZE = True
EMBEDDING_MAX_SEQ_LENGTH = 512
EMBEDDING_DEVICE = 'cpu'

# ---- 文本类型标识（Embedding 缓存键 = 文本类型 + 语料 SHA256 + 模型配置） ----
TEXT_TYPE_JOB_FULL = 'job_text_full'
TEXT_TYPE_JOB_SAFE = 'job_text_safe'
TEXT_TYPE_DUTY = 'duty_text'
TEXT_TYPE_REQUIRE = 'require_text'
TEXT_TYPE_SKILL = 'skill_text'
TEXT_TYPE_COMPANY = 'company_profile'
TEXT_TYPE_LABELS = {
    TEXT_TYPE_JOB_FULL: '岗位描述_语义分析版（完整文本统计范围）',
    TEXT_TYPE_JOB_SAFE: '岗位描述_模型安全版（去薪资统计范围）',
    TEXT_TYPE_DUTY: '职责文本',
    TEXT_TYPE_REQUIRE: '任职要求文本',
    TEXT_TYPE_SKILL: '技能要求文本',
    TEXT_TYPE_COMPANY: '公司简介_清洗',
}
# 历史缓存数组键名（Refinement R1 之前的 npz 键）：仅在语料指纹与模型配置一致时允许迁移复用
LEGACY_ARRAY_KEYS = {
    TEXT_TYPE_JOB_FULL: 'job_text',
    TEXT_TYPE_JOB_SAFE: None,
    TEXT_TYPE_DUTY: 'duty_text',
    TEXT_TYPE_REQUIRE: 'require_text',
    TEXT_TYPE_SKILL: 'skill_text',
    TEXT_TYPE_COMPANY: 'company_profile',
}

GLOBALIZATION_KEYWORDS = ['出海', '全球化', '国际化', '海外', '全球']
TECH_DIRECTION_KEYWORDS = ['技术', '研发', '算法', '架构', '平台', '云', '开源', '智能']


@dataclass
class EmbeddingResult:
    """句向量构建结果：要么真实可用，要么明确 NOT_RUN。"""

    status: str
    model: object = None
    model_name: str = ''
    dimension: int = 0
    normalize_embeddings: bool = EMBEDDING_NORMALIZE
    batch_size: int = EMBEDDING_BATCH_SIZE
    device: str = EMBEDDING_DEVICE
    max_seq_length: int = 0
    model_revision: str = ''
    reason: str = ''
    load_seconds: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.status == 'PASS' and self.model is not None

    def reproduction_record(self, input_field: str) -> dict:
        """复现记录（提示词第 39 节要求字段）。"""
        return {
            '模型名称': self.model_name,
            '模型版本_revision': self.model_revision,
            '向量维度': int(self.dimension),
            'normalize_embeddings': bool(self.normalize_embeddings),
            'batch_size': int(self.batch_size),
            'device': self.device,
            'max_seq_length': int(self.max_seq_length),
            '输入文本字段': input_field,
            '模型加载用时_秒': round(self.load_seconds, 2),
            '生成时间': time.strftime('%Y-%m-%d %H:%M:%S'),
            '状态': self.status,
            '失败原因': self.reason,
        }


def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME,
                         device: str = EMBEDDING_DEVICE) -> EmbeddingResult:
    """加载真实句向量模型；任何异常都返回 NOT_RUN，禁止伪造向量。"""
    started = time.time()
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        model = SentenceTransformer(model_name, device=device)
        model.max_seq_length = EMBEDDING_MAX_SEQ_LENGTH
        # sentence-transformers 6.x 起方法改名为 get_embedding_dimension
        dimension_getter = (getattr(model, 'get_embedding_dimension', None)
                            or getattr(model, 'get_sentence_embedding_dimension', None))
        dimension = int(dimension_getter())
        revision = ''
        try:
            config = getattr(model[0].auto_model, 'config', None)
            revision = getattr(config, '_name_or_path', '') or ''
        except (IndexError, KeyError, AttributeError):
            revision = ''
        return EmbeddingResult(status='PASS', model=model, model_name=model_name,
                               dimension=dimension, max_seq_length=EMBEDDING_MAX_SEQ_LENGTH,
                               model_revision=revision, load_seconds=time.time() - started)
    except Exception as error:  # noqa: BLE001 - 必须捕获全部失败原因并如实记录
        return EmbeddingResult(status='NOT_RUN', model_name=model_name,
                               reason=f'{type(error).__name__}: {error}',
                               load_seconds=time.time() - started)


def encode_texts(result: EmbeddingResult, texts, batch_size: int = EMBEDDING_BATCH_SIZE) -> np.ndarray:
    """对文本列表编码；输入为空时返回空数组（不生成任何伪向量）。"""
    if not result.available:
        raise RuntimeError(f'句向量模型不可用（{result.status}）：{result.reason}')
    clean = ['' if text is None else str(text) for text in texts]
    if not clean:
        return np.zeros((0, result.dimension), dtype='float32')
    vectors = result.model.encode(clean, batch_size=batch_size,
                                 normalize_embeddings=result.normalize_embeddings,
                                 show_progress_bar=False, convert_to_numpy=True)
    return np.asarray(vectors, dtype='float32')


def embed_non_empty(result: EmbeddingResult, texts, batch_size: int = EMBEDDING_BATCH_SIZE
                    ) -> tuple:
    """只对非空文本编码，返回 (行号数组, 向量矩阵)。

    空文本一律不参与编码（禁止对空字符串生成伪 Embedding）。
    """
    series = pd.Series([('' if text is None else str(text).strip()) for text in texts])
    positions = np.where(series.str.len() > 0)[0]
    if len(positions) == 0:
        return positions, np.zeros((0, result.dimension), dtype='float32')
    vectors = encode_texts(result, series.iloc[positions].tolist(), batch_size=batch_size)
    return positions, vectors


def save_embeddings(npz_path: Path, index_path: Path, arrays: dict,
                    index_frame: pd.DataFrame, meta: dict,
                    meta_path: Path | None = None) -> dict:
    """保存高维向量（npz）与索引表（parquet），并写出复现记录 JSON。"""
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    io_utils.write_parquet(index_frame, index_path)
    target_meta = meta_path or npz_path.with_name(npz_path.stem + '_meta.json')
    io_utils.write_json(target_meta, meta)
    return {
        'npz': str(npz_path.relative_to(project_paths.PROJECT_ROOT)),
        'index': str(index_path.relative_to(project_paths.PROJECT_ROOT)),
        'meta': str(target_meta.relative_to(project_paths.PROJECT_ROOT)),
        'keys': {name: list(array.shape) for name, array in arrays.items()},
    }


def load_embedding_meta(npz_path: Path) -> dict:
    """读取向量复现记录。"""
    meta_path = npz_path.with_name(npz_path.stem + '_meta.json')
    return io_utils.read_json(meta_path) or {}


def corpus_fingerprint(texts) -> str:
    """语料指纹（SHA256）：用于判断已有句向量是否仍与当前语料一致（缓存复用校验）。"""
    import hashlib  # noqa: PLC0415

    hasher = hashlib.sha256()
    for text in texts:
        hasher.update(('' if text is None else str(text)).encode('utf-8'))
        hasher.update(b'\x1f')
    return hasher.hexdigest()


def legacy_corpus_fingerprint(texts) -> str:
    """Refinement R1 之前的语料指纹（SHA1，仅覆盖岗位描述全文），仅用于缓存迁移校验。"""
    import hashlib  # noqa: PLC0415

    hasher = hashlib.sha1()
    for text in texts:
        hasher.update(('' if text is None else str(text)).encode('utf-8'))
        hasher.update(b'\x1f')
    return hasher.hexdigest()


def text_hash(text) -> str:
    """单条文本哈希（SHA256 前 16 位）：相同安全文本按哈希去重后只编码一次。"""
    import hashlib  # noqa: PLC0415

    return hashlib.sha256(('' if text is None else str(text)).encode('utf-8')).hexdigest()[:16]


def model_config(result: EmbeddingResult) -> dict:
    """模型配置指纹：缓存复用必须同时满足语料指纹一致 + 模型配置一致。"""
    return {
        '模型名称': result.model_name,
        '向量维度': int(result.dimension),
        'normalize_embeddings': bool(result.normalize_embeddings),
        'max_seq_length': int(result.max_seq_length),
        'device': result.device,
    }


def load_array_cache(npz_path: Path, expected_config: dict, expected_fingerprints: dict,
                     expected_sizes: dict | None = None,
                     legacy_fingerprint: str | None = None) -> dict:
    """按「文本类型 + 语料 SHA256 + 模型配置」校验并读取已有向量数组。

    返回 ``{'arrays': {文本类型: 矩阵}, 'cache_status': {文本类型: 说明}}``；
    未通过校验的文本类型不会出现在 arrays 中，调用方必须重新编码。

    迁移规则（Refinement R1 之前只有单一语料指纹，且只覆盖岗位描述全文）：
    当旧meta的单个 SHA1 指纹与当前全文语料一致、且模型配置一致时，允许复用旧数组；
    由于分段文本是全文语料的确定性函数（清洗 → 分段，逻辑未变），
    全文指纹一致即保证分段文本一致，因此分段数组可安全迁移。
    """
    npz_path = Path(npz_path)
    result: dict = {'arrays': {}, 'cache_status': {}}
    expected_sizes = expected_sizes or {}
    if not npz_path.exists():
        return result
    meta = load_embedding_meta(npz_path)
    if not meta:
        return result
    stored_fingerprints = meta.get('语料指纹')
    typed_fingerprints = stored_fingerprints if isinstance(stored_fingerprints, dict) else {}
    stored_config = meta.get('模型配置') or dict(meta)
    config_ok = all(stored_config.get(key) == value for key, value in expected_config.items())
    legacy_ok = (bool(legacy_fingerprint) and not isinstance(stored_fingerprints, dict)
                 and stored_fingerprints == legacy_fingerprint)
    with np.load(npz_path) as payload:
        available = {name: payload[name] for name in payload.files}
    for text_type, fingerprint in expected_fingerprints.items():
        array = None
        source = ''
        if config_ok and typed_fingerprints.get(text_type) == fingerprint:
            array, source = available.get(text_type), f'复用缓存（{text_type}）'
        else:
            legacy_key = LEGACY_ARRAY_KEYS.get(text_type)
            if legacy_ok and legacy_key and legacy_key in available:
                array, source = available[legacy_key], f'迁移复用旧缓存（{legacy_key}）'
        if array is None:
            continue
        expected_size = expected_sizes.get(text_type)
        if expected_size is not None and array.shape[0] != expected_size:
            result['cache_status'][text_type] = (
                f'缓存行数 {array.shape[0]} 与当前非空文本数 {expected_size} 不一致，重新编码')
            continue
        result['arrays'][text_type] = array
        result['cache_status'][text_type] = source
    return result


# ---------------------------------------------------------------- Token 审计

def model_max_tokens(result: EmbeddingResult) -> int:
    """模型允许的最大 token 数（由真实 tokenizer 与模型配置共同决定）。"""
    tokenizer = getattr(result.model, 'tokenizer', None)
    limits = []
    if tokenizer is not None:
        tokenizer_limit = getattr(tokenizer, 'model_max_length', None)
        if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 10 ** 6:
            limits.append(tokenizer_limit)
    if result.max_seq_length:
        limits.append(int(result.max_seq_length))
    return int(min(limits)) if limits else 0


def token_lengths(result: EmbeddingResult, texts, chunk_size: int = 512) -> np.ndarray:
    """用模型真实 tokenizer 统计**未截断** token 长度（禁止用字符数近似）。"""
    tokenizer = getattr(result.model, 'tokenizer', None)
    if tokenizer is None:
        raise RuntimeError('模型未提供 tokenizer，无法进行真实 token 长度审计')
    values = ['' if text is None else str(text) for text in texts]
    lengths = np.zeros(len(values), dtype='int64')
    for start in range(0, len(values), chunk_size):
        chunk = values[start:start + chunk_size]
        encoded = tokenizer(chunk, truncation=False, add_special_tokens=True,
                            return_attention_mask=False, verbose=False)
        for offset, ids in enumerate(encoded['input_ids']):
            lengths[start + offset] = len(ids)
    return lengths


def truncation_frame(text_type: str, lengths: np.ndarray, max_tokens: int) -> dict:
    """截断审计：超出 token 数 = max(原始token数 - 模型最大token数, 0)。"""
    values = np.asarray(lengths, dtype='float64')
    total = int(values.size)
    exceed = np.maximum(values - max_tokens, 0)
    truncated = int((exceed > 0).sum())
    with np.errstate(divide='ignore', invalid='ignore'):
        retention = np.where(values > 0, np.minimum(max_tokens / np.maximum(values, 1), 1.0), 1.0)
    return {
        '文本类型': text_type,
        '文本数': total,
        'Token_P50': round(float(np.percentile(values, 50)), 2) if total else np.nan,
        'Token_P95': round(float(np.percentile(values, 95)), 2) if total else np.nan,
        'Token_P99': round(float(np.percentile(values, 99)), 2) if total else np.nan,
        'Token_Max': int(values.max()) if total else np.nan,
        '模型最大token数': int(max_tokens),
        '截断文本数': truncated,
        '截断比例': round(truncated / total, 6) if total else 0.0,
        '平均理论保留比例': round(float(retention.mean()), 6) if total else 1.0,
        '超出token数均值': round(float(exceed.mean()), 6) if total else 0.0,
    }


def token_distribution_frame(text_type: str, lengths: np.ndarray,
                             max_tokens: int, bins: int = 20) -> pd.DataFrame:
    """token 长度分布分桶表（审计用，只做描述性统计）。"""
    values = np.asarray(lengths, dtype='float64')
    if values.size == 0:
        return pd.DataFrame(columns=['文本类型', '区间下限', '区间上限', '文本数', '占比'])
    upper = max(float(values.max()), max_tokens)
    edges = np.linspace(0, upper, bins + 1)
    bucket = pd.cut(values, bins=edges, include_lowest=True)
    counts = bucket.value_counts().sort_index()
    rows = [{'文本类型': text_type,
             '区间下限': round(float(interval.left), 2),
             '区间上限': round(float(interval.right), 2),
             '文本数': int(count),
             '占比': round(int(count) / values.size, 6)}
            for interval, count in counts.items()]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 相似度与距离

def semantic_distance(similarity) -> np.ndarray:
    """semantic_distance = 1 - semantic_similarity（提示词第 18 节定义）。"""
    values = np.asarray(similarity, dtype='float64')
    return 1.0 - values


def similarity_for_pairs(matrix: np.ndarray, row_index: dict, pairs) -> np.ndarray:
    """按 (左侧键, 右侧键) 列表计算逐对余弦相似度。"""
    result = np.full(len(pairs), np.nan, dtype='float64')
    for position, (left_key, right_key) in enumerate(pairs):
        left_row = row_index.get(left_key)
        right_row = row_index.get(right_key)
        if left_row is None or right_row is None:
            continue
        vector_a = matrix[left_row]
        vector_b = matrix[right_row]
        denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        if denominator == 0:
            continue
        result[position] = float(np.clip(np.dot(vector_a, vector_b) / denominator, -1.0, 1.0))
    return result


# ---------------------------------------------------------------- TF-IDF 词面基线

def build_tfidf_matrix(texts, tokenize_func, min_df: int = 2, max_features: int = 20000) -> tuple:
    """构建 TF-IDF 词面基线矩阵，返回 (vectorizer, matrix)。"""
    from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415

    corpus = ['' if text is None else str(text) for text in texts]

    def analyzer(text):
        return tokenize_func(text)

    vectorizer = TfidfVectorizer(analyzer=analyzer, min_df=min_df, max_features=max_features,
                                 token_pattern=None, lowercase=False)
    matrix = vectorizer.fit_transform(corpus) if corpus else None
    return vectorizer, matrix


def tfidf_cosine_for_pairs(matrix, row_index: dict, pairs) -> np.ndarray:
    """按文本对计算 TF-IDF 余弦相似度（词面指标，不等于深层语义）。

    全程使用稀疏行内积，不做任何稠密化，避免大语料下反复分配巨量内存。
    """
    if matrix is None:
        return np.full(len(pairs), np.nan, dtype='float64')
    result = np.full(len(pairs), np.nan, dtype='float64')
    for position, (left_key, right_key) in enumerate(pairs):
        left_row = row_index.get(left_key)
        right_row = row_index.get(right_key)
        if left_row is None or right_row is None:
            continue
        vector_a = matrix[left_row]
        vector_b = matrix[right_row]
        denominator = float(np.sqrt(vector_a.multiply(vector_a).sum())
                            * np.sqrt(vector_b.multiply(vector_b).sum()))
        if denominator == 0:
            continue
        result[position] = float(np.clip(float(vector_a.multiply(vector_b).sum()) / denominator,
                                         -1.0, 1.0))
    return result


# ---------------------------------------------------------------- 阈值与检查

def quantile_thresholds(values, prefix: str = '') -> dict:
    """统计 P50 / P75 / P90 / P95（阈值取自真实分布）。"""
    series = pd.Series(values, dtype='float64').dropna()
    if series.empty:
        return {f'{prefix}P50': float('nan'), f'{prefix}P75': float('nan'),
                f'{prefix}P90': float('nan'), f'{prefix}P95': float('nan'),
                f'{prefix}样本数': 0}
    return {
        f'{prefix}P50': round(float(series.quantile(0.50)), 6),
        f'{prefix}P75': round(float(series.quantile(0.75)), 6),
        f'{prefix}P90': round(float(series.quantile(0.90)), 6),
        f'{prefix}P95': round(float(series.quantile(0.95)), 6),
        f'{prefix}样本数': int(series.notna().sum()),
    }


def validate_semantic_values(series, name: str, lower: float = 0.0,
                             upper: float = 2.0) -> dict:
    """检查语义距离/相似度是否存在 NaN、Inf 与越界值。"""
    values = pd.Series(series, dtype='float64')
    non_null = values.dropna()
    report = {
        '字段': name,
        '记录数': int(len(values)),
        '非空数': int(non_null.size),
        '缺失数': int(values.isna().sum()),
        'Inf数量': int(np.isinf(non_null).sum()),
        '越界数量': int(((non_null < lower) | (non_null > upper)).sum()),
    }
    report['是否通过'] = report['Inf数量'] == 0 and report['越界数量'] == 0
    return report


def _value(row, key, default=float('nan')):
    value = row.get(key, default)
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if not np.isnan(result) else default


# ---------------------------------------------------------------- 变化候选分类

def classify_job_text_change(row: dict, thresholds: dict,
                             distance_field: str = schema.JD_DISTANCE_SAFE_FIELD) -> str:
    """岗位文本变化候选类型（规则可解释，未人工确认前一律为「候选」）。

    正式主判据为**去薪资语义距离**；完整文本统计范围只作页面整体变化对照。
    """
    distance = _value(row, distance_field)
    added = _value(row, '新增技能数', 0.0)
    removed = _value(row, '删除技能数', 0.0)
    duty_distance = _value(row, '职责语义距离')
    require_distance = _value(row, '任职要求语义距离')
    skill_distance = _value(row, '技能段语义距离')
    duty_growth = _value(row, '职责段字符数变化率')
    degree_changed = int(row.get('是否同时学历要求变化', 0) or 0)
    skill_changes = added + removed
    p50 = thresholds.get(f'{distance_field}P50', float('nan'))
    p75 = thresholds.get(f'{distance_field}P75', float('nan'))
    p90 = thresholds.get(f'{distance_field}P90', float('nan'))
    p95 = thresholds.get(f'{distance_field}P95', float('nan'))

    if not np.isnan(distance) and not np.isnan(p95) and distance >= p95 and skill_changes >= 3:
        return '岗位方向迁移候选'
    if not np.isnan(distance) and not np.isnan(p90) and distance >= p90 and added > 0 and removed > 0:
        return '岗位方向迁移候选'
    if added > 0 and not np.isnan(skill_distance) and not np.isnan(p90) and skill_distance >= p90:
        return '技能要求增加候选'
    if added > 0 and removed == 0:
        return '技能要求增加候选'
    if removed > 0 and added == 0 and skill_changes > 0:
        return '技能要求删除候选'
    if not np.isnan(require_distance) and not np.isnan(p75) and require_distance >= p75 and degree_changed:
        return '任职门槛变化候选'
    if not np.isnan(duty_growth) and duty_growth >= 0.20 and not np.isnan(duty_distance) \
            and not np.isnan(p75) and duty_distance >= p75:
        return '岗位职责扩展候选'
    if not np.isnan(duty_growth) and duty_growth <= -0.20:
        return '岗位职责收缩候选'
    duty_quiet = np.isnan(duty_distance) or (not np.isnan(p50) and duty_distance <= p50)
    require_quiet = np.isnan(require_distance) or (not np.isnan(p50) and require_distance <= p50)
    if skill_changes == 0 and duty_quiet and require_quiet and not np.isnan(distance) and distance > 0:
        return '非核心信息变化候选'
    if skill_changes == 0 and not np.isnan(distance) and not np.isnan(p50) and distance <= p50:
        return '文本润色候选'
    return '非核心信息变化候选'


def classify_company_text_change(row: dict, thresholds: dict) -> str:
    """公司简介变化候选类型（全部保留「候选」字样）。"""
    distance = _value(row, 'Embedding语义距离')
    if np.isnan(distance):
        distance = _value(row, '词面语义距离')
    added_business = text_utils.as_list(row.get('新增业务关键词'))
    removed_business = text_utils.as_list(row.get('删除业务关键词'))
    new_ai = int(row.get('是否新增AI关键词', 0) or 0)
    new_llm = int(row.get('是否新增大模型关键词', 0) or 0)
    p50 = thresholds.get('公司简介语义距离P50', float('nan'))
    p75 = thresholds.get('公司简介语义距离P75', float('nan'))

    if new_llm or new_ai:
        return 'AI/大模型方向强化候选'
    if any(keyword in GLOBALIZATION_KEYWORDS for keyword in added_business):
        return '全球化叙事变化候选'
    if added_business and not np.isnan(distance) and not np.isnan(p75) and distance >= p75:
        return '业务范围扩展候选'
    if removed_business and not np.isnan(distance) and not np.isnan(p75) and distance >= p75:
        return '企业定位变化候选'
    if added_business:
        return '业务范围扩展候选'
    if any(keyword in TECH_DIRECTION_KEYWORDS for keyword in added_business + removed_business):
        return '技术方向变化候选'
    if not np.isnan(distance) and not np.isnan(p50) and distance <= p50:
        return '文字润色候选'
    return '品牌宣传变化候选'


def build_threshold_source(thresholds: dict, extra: dict | None = None) -> pd.DataFrame:
    """把阈值来源写成审计表（阈值取自真实分布，并记录样本量与人工依据）。"""
    rows = []
    for key, value in thresholds.items():
        if key.endswith('样本数'):
            continue
        base = key.rsplit('P', 1)[0]
        rows.append({
            '指标': key,
            '阈值': round(float(value), 6) if not np.isnan(value) else np.nan,
            '样本数': thresholds.get(f'{base}样本数', np.nan),
            '来源': '全体相邻版本语义距离的真实分位数统计',
            '用途': f'{base} 显著变化候选判定',
        })
    if extra:
        for key, value in extra.items():
            rows.append({'指标': key, '阈值': value, '样本数': np.nan,
                         '来源': '人工复核与业务统计范围', '用途': '候选类型规则'})
    return pd.DataFrame(rows)

# -*- coding: utf-8 -*-
"""薪资预测训练层（Stage 14）：固定划分、train-only 预处理、验证集选模、一次性 test 评估。

## 核心约束

1. **同一划分**：train 70% / validation 15% / test 15%（`random_state=42`，按薪资中点分位分箱分层），
   所有模型复用同一划分，不重复随机分割；
2. **预处理只 fit 在 train**：数值填补、缺失指示、类别编码、StandardScaler、
   技能频率筛选与列集合、文本降维全部由 `SalaryFeatureAssembler` 在 train 上拟合；
3. **选择只用 validation**：技能阈值（50/80/100/0.5% train）、文本维度（16/32/64）、
   各模型小网格超参数均以 validation MAE 选择；
4. **test 只用一次**：最终配置锁定后评估一次，禁止据 test 反复调参；
5. **公司 Group Split** 只准备不用于选主模型（留给 Stage 15 稳健性实验）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from src import schema, skill_eda, text_utils

SPLIT_RANDOM_STATE = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)
SPLIT_LABELS = ('train', 'validation', 'test')
SPLIT_BINS = 10
MAIN_METRIC = 'MAE'
SKILL_THRESHOLD_CANDIDATES = (50, 80, 100, 0.005)   # 0.005 = 训练集岗位数的 0.5%
TEXT_DIM_CANDIDATES = (16, 32, 64)
MULTI_VALUE_COLUMNS = ('岗位大类集合', '岗位细分类集合', schema.COMPANY_TAG_LIST_FIELD)
TOP_MULTI_VALUES = 30
MIN_CATEGORY_FREQUENCY = 20
UNKNOWN_LABEL = '__未知__'
OTHER_LABEL = '__其他__'


# ---------------------------------------------------------------- 数据划分

def build_splits(model_frame: pd.DataFrame, random_state: int = SPLIT_RANDOM_STATE,
                 n_bins: int = SPLIT_BINS) -> pd.DataFrame:
    """按薪资中点分位分箱分层划分 train/validation/test（不写入目标值）。"""
    salary = model_frame[schema.SALARY_MID_FIELD].to_numpy()
    bins = pd.qcut(pd.Series(salary), q=n_bins, labels=False, duplicates='drop')
    stratify = bins.fillna(0).astype(int).to_numpy()
    job_ids = model_frame[schema.ID_FIELD].to_numpy()
    split = np.empty(len(job_ids), dtype=object)

    holder = StratifiedShuffleSplit(n_splits=1, test_size=SPLIT_RATIOS[1] + SPLIT_RATIOS[2],
                                    random_state=random_state)
    train_index, rest_index = next(holder.split(job_ids.reshape(-1, 1), stratify))
    split[train_index] = 'train'
    holder2 = StratifiedShuffleSplit(
        n_splits=1, test_size=SPLIT_RATIOS[2] / (SPLIT_RATIOS[1] + SPLIT_RATIOS[2]),
        random_state=random_state)
    valid_rel, test_rel = next(holder2.split(job_ids[rest_index].reshape(-1, 1),
                                            stratify[rest_index]))
    split[rest_index[valid_rel]] = 'validation'
    split[rest_index[test_rel]] = 'test'
    return pd.DataFrame({schema.ID_FIELD: job_ids, 'split': split})


def build_company_group_split(model_frame: pd.DataFrame,
                              random_state: int = SPLIT_RANDOM_STATE) -> pd.DataFrame:
    """公司实体 Group Split（同一公司的岗位只出现在一个子集中）。"""
    groups = model_frame['company_entity_id'].fillna('__无实体__').to_numpy()
    job_ids = model_frame[schema.ID_FIELD].to_numpy()
    split = np.empty(len(job_ids), dtype=object)
    holder = GroupShuffleSplit(n_splits=1, test_size=SPLIT_RATIOS[1] + SPLIT_RATIOS[2],
                               random_state=random_state)
    train_index, rest_index = next(holder.split(job_ids.reshape(-1, 1), groups=groups))
    split[train_index] = 'train'
    holder2 = GroupShuffleSplit(
        n_splits=1, test_size=SPLIT_RATIOS[2] / (SPLIT_RATIOS[1] + SPLIT_RATIOS[2]),
        random_state=random_state)
    valid_rel, test_rel = next(holder2.split(job_ids[rest_index].reshape(-1, 1),
                                            groups=groups[rest_index]))
    split[rest_index[valid_rel]] = 'validation'
    split[rest_index[test_rel]] = 'test'
    return pd.DataFrame({schema.ID_FIELD: job_ids, 'company_group_split': split,
                         'company_entity_id': groups})


# ---------------------------------------------------------------- 特征装配

@dataclass
class FeatureSchema:
    """特征装配的 train-only 拟合结果（可序列化，供模型产出与测试核对）。"""
    numeric_columns: list = field(default_factory=list)
    categorical_columns: list = field(default_factory=list)
    multi_value_columns: list = field(default_factory=list)
    numeric_medians: dict = field(default_factory=dict)
    numeric_indicator_columns: list = field(default_factory=list)
    numeric_mean: list = field(default_factory=list)
    numeric_std: list = field(default_factory=list)
    category_levels: dict = field(default_factory=dict)
    multi_value_levels: dict = field(default_factory=dict)
    skill_columns: list = field(default_factory=list)
    skill_threshold: int = 0
    min_category_frequency: int = MIN_CATEGORY_FREQUENCY
    top_multi_values: int = TOP_MULTI_VALUES
    text_dim: int = 0
    scale_numeric: bool = False
    fit_rows: int = 0
    fit_scope: str = 'train'
    text_mean: list = field(default_factory=list)
    text_components: list = field(default_factory=list)

    @property
    def dimension(self) -> int:
        numeric = len(self.numeric_columns) + len(self.numeric_indicator_columns)
        categorical = sum(len(levels) for levels in self.category_levels.values())
        multi = sum(len(levels) for levels in self.multi_value_levels.values())
        return int(numeric + categorical + multi + len(self.skill_columns) + self.text_dim)


class SalaryFeatureAssembler:
    """在 train 上拟合的薪资特征装配器（数值 / 类别 / 多值 / 技能 / 文本语义）。"""

    def __init__(self, numeric_columns, categorical_columns, multi_value_columns=None,
                 skill_threshold: int = 50, text_dim: int = 0, scale_numeric: bool = False,
                 min_category_frequency: int = MIN_CATEGORY_FREQUENCY,
                 top_multi_values: int = TOP_MULTI_VALUES):
        self.schema = FeatureSchema(
            numeric_columns=list(numeric_columns), categorical_columns=list(categorical_columns),
            multi_value_columns=list(multi_value_columns or []), skill_threshold=skill_threshold,
            text_dim=int(text_dim), scale_numeric=bool(scale_numeric),
            min_category_frequency=int(min_category_frequency),
            top_multi_values=int(top_multi_values))

    # ---- 拟合 ----
    def fit(self, frame: pd.DataFrame, skill_map: dict,
            text_matrix: np.ndarray | None = None) -> 'SalaryFeatureAssembler':
        spec = self.schema
        spec.fit_rows = int(len(frame))
        for column in spec.numeric_columns:
            series = pd.to_numeric(frame[column], errors='coerce')
            spec.numeric_medians[column] = float(series.median(skipna=True)) \
                if series.notna().any() else 0.0
            if bool(series.isna().any()):
                spec.numeric_indicator_columns.append(column)
        for column in spec.categorical_columns:
            series = frame[column].fillna(UNKNOWN_LABEL).astype(str)
            counts = series.value_counts()
            kept = [value for value, count in counts.items()
                    if count >= spec.min_category_frequency]
            spec.category_levels[column] = kept + ([] if OTHER_LABEL in kept else [OTHER_LABEL])
        for column in spec.multi_value_columns:
            exploded = frame[column].explode().dropna().astype(str)
            spec.multi_value_levels[column] = exploded.value_counts().head(
                spec.top_multi_values).index.tolist()
        counts = {}
        for job_id in frame[schema.ID_FIELD]:
            for skill in skill_map.get(job_id, ()):
                counts[skill] = counts.get(skill, 0) + 1
        spec.skill_columns = sorted(skill for skill, count in counts.items()
                                    if count >= spec.skill_threshold)
        if spec.text_dim > 0:
            matrix = self._text_slice(frame, text_matrix)
            usable = np.abs(matrix).sum(axis=1) > 0
            if usable.sum() >= spec.text_dim + 1:
                reducer = TruncatedSVD(n_components=spec.text_dim, random_state=SPLIT_RANDOM_STATE)
                reducer.fit(matrix[usable])
                spec.text_mean = reducer.mean_.tolist() if hasattr(reducer, 'mean_') else []
                spec.text_components = reducer.components_.tolist()
        if spec.scale_numeric and spec.numeric_columns:
            dense = self._numeric_dense(frame)
            spec.numeric_mean = dense.mean(axis=0).tolist()
            spec.numeric_std = np.where(dense.std(axis=0) == 0, 1.0, dense.std(axis=0)).tolist()
        return self

    # ---- 变换 ----
    def transform(self, frame: pd.DataFrame, skill_map: dict,
                  text_matrix: np.ndarray | None = None) -> sparse.csr_matrix:
        spec = self.schema
        blocks = [sparse.csr_matrix(self._numeric_dense(frame))]
        for column in spec.categorical_columns:
            blocks.append(self._one_hot_categorical(frame, column))
        for column in spec.multi_value_columns:
            blocks.append(self._one_hot_multi(frame, column))
        if spec.skill_columns:
            blocks.append(self._one_hot_skill(frame, skill_map))
        if spec.text_dim > 0 and spec.text_components:
            blocks.append(sparse.csr_matrix(self._text_reduced(frame, text_matrix)))
        return sparse.hstack(blocks, format='csr')

    def feature_names(self) -> list:
        """矩阵列名（顺序与 transform 的块顺序严格一致，供 SHAP 与审计对齐）。"""
        spec = self.schema
        names = list(spec.numeric_columns)
        names += [f'{column}__缺失' for column in spec.numeric_indicator_columns]
        for column in spec.categorical_columns:
            names += [f'{column}={level}' for level in spec.category_levels.get(column, [])]
        for column in spec.multi_value_columns:
            names += [f'{column}={level}' for level in spec.multi_value_levels.get(column, [])]
        names += [f'技能={skill}' for skill in spec.skill_columns]
        names += [f'文本SVD{i + 1}' for i in range(spec.text_dim)]
        return names

    # ---- 各子块 ----
    def _numeric_dense(self, frame: pd.DataFrame) -> np.ndarray:
        spec = self.schema
        columns = [pd.to_numeric(frame[column], errors='coerce').to_numpy(dtype='float64')
                   for column in spec.numeric_columns]
        dense = np.column_stack(columns) if columns else np.zeros((len(frame), 0))
        for index, column in enumerate(spec.numeric_columns):
            missing = np.isnan(dense[:, index])
            if missing.any():
                dense[missing, index] = spec.numeric_medians.get(column, 0.0)
        if spec.numeric_indicator_columns:
            indicators = np.column_stack([
                pd.to_numeric(frame[column], errors='coerce').isna().to_numpy(dtype='float64')
                for column in spec.numeric_indicator_columns])
            dense = np.hstack([dense, indicators])
        if spec.scale_numeric and spec.numeric_mean:
            mean = np.asarray(spec.numeric_mean, dtype='float64')
            std = np.asarray(spec.numeric_std, dtype='float64')
            if dense.shape[1] == len(mean):
                dense = (dense - mean) / std
        return dense

    def _one_hot_categorical(self, frame: pd.DataFrame, column: str) -> sparse.csr_matrix:
        spec = self.schema
        levels = spec.category_levels.get(column, [])
        index = {value: position for position, value in enumerate(levels)}
        other = index.get(OTHER_LABEL, len(levels) - 1 if levels else 0)
        series = frame[column].fillna(UNKNOWN_LABEL).astype(str)
        rows = np.arange(len(frame))
        cols = np.array([index.get(value, other) for value in series], dtype='int64')
        data = np.ones(len(frame), dtype='float64')
        return sparse.csr_matrix((data, (rows, cols)), shape=(len(frame), max(len(levels), 1)))

    def _one_hot_multi(self, frame: pd.DataFrame, column: str) -> sparse.csr_matrix:
        spec = self.schema
        levels = spec.multi_value_levels.get(column, [])
        index = {value: position for position, value in enumerate(levels)}
        rows, cols = [], []
        for row_position, items in enumerate(frame[column]):
            for item in text_utils.as_list(items):
                position = index.get(str(item))
                if position is not None:
                    rows.append(row_position)
                    cols.append(position)
        data = np.ones(len(rows), dtype='float64')
        return sparse.csr_matrix((data, (rows, cols)), shape=(len(frame), max(len(levels), 1)))

    def _one_hot_skill(self, frame: pd.DataFrame, skill_map: dict) -> sparse.csr_matrix:
        spec = self.schema
        index = {skill: position for position, skill in enumerate(spec.skill_columns)}
        rows, cols = [], []
        for row_position, job_id in enumerate(frame[schema.ID_FIELD]):
            for skill in skill_map.get(job_id, ()):
                position = index.get(skill)
                if position is not None:
                    rows.append(row_position)
                    cols.append(position)
        data = np.ones(len(rows), dtype='float64')
        return sparse.csr_matrix((data, (rows, cols)),
                                 shape=(len(frame), max(len(spec.skill_columns), 1)))

    def _text_slice(self, frame: pd.DataFrame, text_matrix: np.ndarray | None) -> np.ndarray:
        if text_matrix is None:
            return np.zeros((len(frame), 1))
        return np.asarray(text_matrix, dtype='float32')

    def _text_reduced(self, frame: pd.DataFrame, text_matrix: np.ndarray | None) -> np.ndarray:
        spec = self.schema
        if text_matrix is None:
            return np.zeros((len(frame), spec.text_dim))
        matrix = np.asarray(text_matrix, dtype='float32')
        components = np.asarray(spec.text_components, dtype='float32')
        return matrix @ components.T


def build_skill_map(membership: pd.DataFrame, scopes) -> dict:
    """岗位 → 规范技能集合（按技能统计范围筛选，intern_id 去重）。"""
    frame = membership[membership['match_scope'].isin(scopes)]
    grouped = frame.groupby('intern_id')['canonical_skill'].apply(lambda values: tuple(set(values)))
    return grouped.to_dict()


def load_text_matrix(indexed_job_ids, text_path: Path, index_path: Path) -> np.ndarray:
    """按 intern_id 取 model-safe BGE 向量（缺失补 0），返回稠密矩阵。"""
    index = pd.read_parquet(index_path)
    index = index[index['text_type'].eq('job_text_safe')]
    with np.load(text_path) as data:
        vectors = data['job_text_safe']
    mapping = dict(zip(index[schema.ID_FIELD], index['embedding_row']))
    matrix = np.zeros((len(indexed_job_ids), vectors.shape[1]), dtype='float32')
    for position, job_id in enumerate(indexed_job_ids):
        row = mapping.get(job_id)
        if row is not None and row < vectors.shape[0]:
            matrix[position] = vectors[row]
    return matrix


# ---------------------------------------------------------------- 建模与评估

def regression_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: PLC0415

    truth = np.asarray(y_true, dtype='float64')
    prediction = np.asarray(y_pred, dtype='float64')
    return {
        'MAE': round(float(mean_absolute_error(truth, prediction)), 6),
        'RMSE': round(float(np.sqrt(mean_squared_error(truth, prediction))), 6),
        'R2': round(float(r2_score(truth, prediction)), 6),
        'n': int(len(truth)),
    }


def make_model(model_key: str, params: dict, random_state: int = SPLIT_RANDOM_STATE):
    """构建模型实例；依赖缺失返回 None（调用方标记 NOT_RUN）。"""
    from sklearn.dummy import DummyRegressor  # noqa: PLC0415
    from sklearn.ensemble import RandomForestRegressor  # noqa: PLC0415
    from sklearn.linear_model import Ridge  # noqa: PLC0415

    if model_key == 'Dummy':
        return DummyRegressor(strategy='mean')
    if model_key == 'Ridge':
        return Ridge(alpha=params.get('alpha', 1.0), random_state=random_state)
    if model_key == 'RandomForest':
        return RandomForestRegressor(
            n_estimators=params.get('n_estimators', 300),
            max_depth=params.get('max_depth'),
            min_samples_leaf=params.get('min_samples_leaf', 1),
            max_features=params.get('max_features', 'sqrt'),
            n_jobs=-1, random_state=random_state)
    if model_key == 'CatBoost':
        from catboost import CatBoostRegressor  # noqa: PLC0415
        return CatBoostRegressor(
            depth=params.get('depth', 6), learning_rate=params.get('learning_rate', 0.05),
            iterations=params.get('iterations', 500), random_seed=random_state,
            verbose=0, allow_writing_files=False)
    if model_key == 'LightGBM':
        from lightgbm import LGBMRegressor  # noqa: PLC0415
        return LGBMRegressor(
            n_estimators=params.get('n_estimators', 400),
            learning_rate=params.get('learning_rate', 0.05),
            num_leaves=params.get('num_leaves', 31), max_depth=params.get('max_depth', -1),
            random_state=random_state, verbose=-1, n_jobs=-1)
    raise ValueError(f'未知模型: {model_key}')


def fit_and_predict(matrix_train, y_train, matrix_valid, matrix_test, model_key, params) -> dict:
    """在 train 上拟合、在 validation/test 上预测（test 由调用方决定是否评估）。"""
    model = make_model(model_key, params)
    model.fit(matrix_train, y_train)
    return {
        'model': model,
        'valid_pred': np.asarray(model.predict(matrix_valid), dtype='float64'),
        'test_pred': (np.asarray(model.predict(matrix_test), dtype='float64')
                      if matrix_test is not None else None),
    }


def skill_count_table(skill_map: dict, job_ids) -> dict:
    return {job_id: len(skill_map.get(job_id, ())) for job_id in job_ids}


def skill_frequency_from_train(membership: pd.DataFrame, train_ids, scopes) -> pd.Series:
    """技能频率只从 train 统计（门禁与测试核对用）。"""
    frame = membership[membership['match_scope'].isin(scopes)
                       & membership['intern_id'].isin(set(train_ids))]
    return frame.groupby('canonical_skill')['intern_id'].nunique().sort_values(ascending=False)


def selected_skill_columns(frequency: pd.Series, threshold: int) -> list:
    return sorted(frequency[frequency >= threshold].index.tolist())


def load_scope_universe(analysis: pd.DataFrame) -> dict:
    """技能统计范围样本集（复用封版统计范围，Stage 12/13 同一实现）。"""
    return skill_eda.load_scope_universe(analysis)


def dump_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                    encoding='utf-8')
    return path

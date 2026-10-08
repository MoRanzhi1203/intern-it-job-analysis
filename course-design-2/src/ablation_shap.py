# -*- coding: utf-8 -*-
"""特征组消融、公司 Group Split 稳健性、技能价值 bootstrap 与 TreeSHAP 解释层（Stage 15）。

## 消融定义（Stage 14 主模型配置固定）

| 配置 | 特征组 |
| --- | --- |
| Base | A + B + C |
| Base+Skill | A + B + C + D |
| Full-Skill | A + B + C + E |
| Full | A + B + C + D + E |

公平性：**同一 split、同一模型族（LightGBM）、同一超参数、同一随机种子**，只改变特征组。

## 口径

- 技能组 D = `job_skill_membership`（ALL_USABLE）multi-hot（阈值 = Stage 14 选定值）+ 技能聚合计数列 +
  `技能提取范围` / 文本可用性控制列；
- 文本组 E = model-safe BGE 向量（训练集拟合 SVD 16 维）+ `文本向量是否可用` 标志；
- 公司 Group Split 按 `company_entity_id` 分组，同公司不跨 train/validation/test；
- SHAP 只对 Stage 14 正式主模型执行（LightGBM 原生 TreeSHAP），仅解释模型关联与预测贡献，
  **禁止解释为因果**；技能 SHAP 必须同时给出岗位频率、mean|SHAP| 与方向性。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import modeling_dataset, schema

ABLATION_CONFIGS = {
    'Base': ('A', 'B', 'C'),
    'Base+Skill': ('A', 'B', 'C', 'D'),
    'Full-Skill': ('A', 'B', 'C', 'E'),
    'Full': ('A', 'B', 'C', 'D', 'E'),
}
CONFIG_ORDER = list(ABLATION_CONFIGS)
NO_CAUSAL_NOTE = '模型关联与预测贡献解释'
SKILL_INCREMENT_PAIRS = [('Base+Skill', 'Base'), ('Full', 'Full-Skill')]
TARGET_VARIANTS = [schema.SALARY_MIN_FIELD, schema.SALARY_MAX_FIELD]
BOOTSTRAP_ROUNDS = 1000
BOOTSTRAP_SEED = 42
GROUP_OF_GROUP_LABEL = {
    modeling_dataset.GROUP_JOB: 'A', modeling_dataset.GROUP_REGION: 'B',
    modeling_dataset.GROUP_COMPANY: 'C', modeling_dataset.GROUP_SKILL: 'D',
    modeling_dataset.GROUP_TEXT: 'E',
}


def column_group(column: str) -> str:
    """字段所属特征组字母（来自 Stage 12 的 Feature Manifest 分组）。"""
    spec = modeling_dataset.COLUMN_SPEC.get(column)
    if spec is None:
        if column in modeling_dataset.SKILL_AGGREGATE_FIELDS:
            return 'D'
        if column in ('文本是否为空', '是否有技能', '技能提取范围'):
            return 'D'
        if column in ('文本向量行号', '文本向量是否可用'):
            return 'E'
        return 'A'
    return GROUP_OF_GROUP_LABEL.get(spec[0], 'A')


def split_columns_by_group(numeric_columns, categorical_columns, multi_columns) -> dict:
    """把 Stage 14 使用的字段按 A/B/C/D/E 分组（组内顺序保持 Stage 14 顺序）。"""
    grouped = {letter: {'numeric': [], 'categorical': [], 'multi': []} for letter in 'ABCDE'}
    for column in numeric_columns:
        grouped[column_group(column)]['numeric'].append(column)
    for column in categorical_columns:
        grouped[column_group(column)]['categorical'].append(column)
    for column in multi_columns:
        grouped[column_group(column)]['multi'].append(column)
    return grouped


def build_assembler_for_groups(groups, grouped_columns, skill_threshold: int, text_dim: int):
    """按启用的特征组构造装配器（D 组技能阈值 = 停用时给不可能达到的高阈值）。"""
    numeric, categorical, multi = [], [], []
    for letter in groups:
        numeric += grouped_columns[letter]['numeric']
        categorical += grouped_columns[letter]['categorical']
        multi += grouped_columns[letter]['multi']
    effective_threshold = skill_threshold if 'D' in groups else 10 ** 9
    effective_text_dim = text_dim if 'E' in groups else 0
    from src import model_training  # noqa: PLC0415

    return model_training.SalaryFeatureAssembler(
        numeric, categorical, multi, skill_threshold=effective_threshold,
        text_dim=effective_text_dim)


def bootstrap_mae_difference(y_true, pred_a, pred_b, rounds: int = BOOTSTRAP_ROUNDS,
                             seed: int = BOOTSTRAP_SEED) -> dict:
    """成对 bootstrap 的 MAE 差值 95% CI（差值 = MAE(B) − MAE(A)，负值表示 A 更好）。"""
    truth = np.asarray(y_true, dtype='float64')
    left = np.asarray(pred_a, dtype='float64')
    right = np.asarray(pred_b, dtype='float64')
    rng = np.random.default_rng(seed)
    size = len(truth)
    deltas = np.empty(rounds, dtype='float64')
    for index in range(rounds):
        sample = rng.integers(0, size, size)
        deltas[index] = (np.abs(truth[sample] - right[sample]).mean()
                         - np.abs(truth[sample] - left[sample]).mean())
    return {
        'MAE_A': round(float(np.abs(truth - left).mean()), 6),
        'MAE_B': round(float(np.abs(truth - right).mean()), 6),
        'ΔMAE（B − A）': round(float(np.abs(truth - right).mean()
                                    - np.abs(truth - left).mean()), 6),
        'bootstrap_CI95_下界': round(float(np.percentile(deltas, 2.5)), 6),
        'bootstrap_CI95_上界': round(float(np.percentile(deltas, 97.5)), 6),
        'CI是否跨0': '是' if np.percentile(deltas, 2.5) <= 0 <= np.percentile(deltas, 97.5)
                    else '否',
        'bootstrap轮数': rounds, '随机种子': seed,
        '说明': 'ΔMAE = MAE(B) − MAE(A)；负值表示 B 优于 A；'
                'CI 不跨 0 表示差异在 95% 置信水平上稳定',
    }


def tree_shap_values(model, matrix) -> np.ndarray:
    """LightGBM 原生 TreeSHAP（pred_contrib）：返回 (n, n_features) 的 SHAP 矩阵。

    稀疏输入时 LightGBM 返回稀疏贡献矩阵（形状 n × (特征数 + 1)，最后一列为基准值
    base value），统一转为稠密并去掉基准值列。
    """
    contrib = model.predict(matrix, pred_contrib=True)
    if hasattr(contrib, 'toarray'):
        contrib = contrib.toarray()
    contrib = np.asarray(contrib, dtype='float64')
    if contrib.ndim == 1:
        contrib = contrib.reshape(len(matrix), -1)
    return contrib[:, :-1]


def dense_matrix(matrix) -> np.ndarray:
    """稀疏 / 稠密特征矩阵统一转为 float 稠密数组（供绘图使用）。"""
    if hasattr(matrix, 'toarray'):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype='float64')


def shap_overall_table(shap_matrix: np.ndarray, feature_names) -> pd.DataFrame:
    """07_SHAP总排名：mean|SHAP| 降序的整体特征重要性。"""
    magnitude = np.abs(shap_matrix).mean(axis=0)
    direction = np.sign(shap_matrix.mean(axis=0))
    share = magnitude / magnitude.sum() if magnitude.sum() else magnitude
    table = pd.DataFrame({
        '特征': list(feature_names), 'mean_abs_SHAP': np.round(magnitude, 6),
        'mean_SHAP': np.round(shap_matrix.mean(axis=0), 6),
        '方向性': np.where(direction > 0, '正向（提高预测薪资）',
                       np.where(direction < 0, '负向（降低预测薪资）', '中性')),
        '重要性占比': np.round(share, 6),
    }).sort_values('mean_abs_SHAP', ascending=False).reset_index(drop=True)
    table.insert(0, '排名', np.arange(1, len(table) + 1))
    table['说明'] = NO_CAUSAL_NOTE
    return table


def presence_direction(mean_present: float, tolerance: float = 1e-8) -> str:
    """presence-conditioned 方向定义（技能存在时的平均 SHAP 贡献方向）。"""
    if mean_present > tolerance:
        return '正向预测贡献'
    if mean_present < -tolerance:
        return '负向预测贡献'
    return '中性/弱影响'


def global_direction(mean_value: float) -> str:
    """整体 mean(SHAP) 方向（历史口径，保留用于对比）。"""
    if mean_value > 0:
        return '正向（提高预测薪资）'
    if mean_value < 0:
        return '负向（降低预测薪资）'
    return '中性'


def shap_skill_table(shap_matrix: np.ndarray, feature_names, membership: pd.DataFrame,
                     universe_ids, scopes, presence_matrix=None) -> pd.DataFrame:
    """08_技能SHAP：技能特征频率 + mean|SHAP| + 全局方向 + presence-conditioned 解释。

    `universe_ids` = Stage 12 建模样本（14,883 个岗位待估），模型样本频率 = 岗位数 / 建模样本数；
    `presence_matrix` = 与 `shap_matrix` **同一批行**（原 test 子集）的特征矩阵，
    用于判定技能 0/1 存在性：test_present_n + test_absent_n = test 行数。

    两个分母严格区分：模型样本频率用 14,883；presence 用 test 样本数，禁止混用。
    """
    names = np.asarray(feature_names)
    skill_mask = np.array([name.startswith('技能=') for name in names])
    if not skill_mask.any():
        return pd.DataFrame()
    positions = np.flatnonzero(skill_mask)
    skill_names = [str(names[index]).replace('技能=', '') for index in positions]
    frame = membership[membership['match_scope'].isin(scopes)
                       & membership['intern_id'].isin(set(universe_ids))]
    frequency = frame.groupby('canonical_skill')['intern_id'].nunique()
    denominator = len(set(universe_ids))
    frequency_basis = f'Stage 12 建模样本 {denominator:,} 个岗位（ALL_USABLE 技能口径）'
    presence = np.asarray(presence_matrix.todense()) if hasattr(presence_matrix, 'todense') \
        else np.asarray(presence_matrix)
    test_rows = int(shap_matrix.shape[0])
    rows = []
    for position, skill in enumerate(skill_names):
        column = int(positions[position])
        values = shap_matrix[:, column]
        present = presence[:, column] > 0 if presence_matrix is not None \
            else np.zeros(len(values), dtype=bool)
        present_values = values[present]
        absent_values = values[~present]
        jobs = int(frequency.get(skill, 0))
        mean_value = float(values.mean())
        mean_present = float(present_values.mean()) if present_values.size else 0.0
        rows.append({
            '技能': skill,
            '岗位数': jobs,
            'model_job_count': jobs,
            '岗位频率': round(jobs / denominator, 6) if denominator else 0.0,
            'model_job_frequency': round(jobs / denominator, 6) if denominator else 0.0,
            'test_present_n': int(present.sum()),
            'test_absent_n': int((~present).sum()),
            'mean_abs_SHAP': round(float(np.abs(values).mean()), 6),
            'mean_SHAP': round(mean_value, 6),
            'mean_abs_SHAP_present': round(float(np.abs(present_values).mean()), 6)
            if present_values.size else 0.0,
            'mean_abs_SHAP_absent': round(float(np.abs(absent_values).mean()), 6)
            if absent_values.size else 0.0,
            'mean_SHAP_present': round(mean_present, 6),
            'median_SHAP_present': round(float(np.median(present_values)), 6)
            if present_values.size else 0.0,
            'mean_SHAP_absent': round(float(absent_values.mean()), 6)
            if absent_values.size else 0.0,
            'median_SHAP_absent': round(float(np.median(absent_values)), 6)
            if absent_values.size else 0.0,
            'positive_SHAP_ratio_present': round(float((present_values > 0).mean()), 6)
            if present_values.size else 0.0,
            'negative_SHAP_ratio_present': round(float((present_values < 0).mean()), 6)
            if present_values.size else 0.0,
            'presence_direction': presence_direction(mean_present),
            'global_mean_SHAP_direction': global_direction(mean_value),
            '方向性': global_direction(mean_value),
            '低频标记': '低频（岗位数 < 30，解释需谨慎）' if jobs < 30 else '',
            '频率口径': frequency_basis,
            'SHAP presence 口径': f'原 test 子集 {test_rows:,} 行；present = 技能列 = 1 的岗位',
            '说明': NO_CAUSAL_NOTE,
        })
    table = (pd.DataFrame(rows).sort_values('mean_abs_SHAP', ascending=False)
             .reset_index(drop=True))
    table.insert(0, '排名', np.arange(1, len(table) + 1))
    return table


def key_skill_table(membership: pd.DataFrame, universe_ids, scopes, skill_shap: pd.DataFrame,
                    keys) -> pd.DataFrame:
    """12_关键技能SHAP：指定重点技能（含未进入模型特征者）的频率与 presence 解释。"""
    frame = membership[membership['match_scope'].isin(scopes)
                       & membership['intern_id'].isin(set(universe_ids))]
    frequency = frame.groupby('canonical_skill')['intern_id'].nunique()
    denominator = len(set(universe_ids))
    indexed = skill_shap.set_index('技能') if not skill_shap.empty else pd.DataFrame()
    rows = []
    for skill in keys:
        jobs = int(frequency.get(skill, 0))
        row = {
            '技能': skill, '模型样本岗位数': jobs,
            '模型样本岗位频率': round(jobs / denominator, 6) if denominator else 0.0,
            '是否进入模型特征': '是' if skill in indexed.index else '否（训练集频率 < 技能阈值）',
            'test_present_n': '', 'mean_SHAP_present': '', 'mean_abs_SHAP': '',
            'presence_direction': '未进入模型特征，无 SHAP 值',
            '说明': NO_CAUSAL_NOTE,
        }
        if skill in indexed.index:
            source = indexed.loc[skill]
            row.update({
                'test_present_n': int(source['test_present_n']),
                'mean_SHAP_present': float(source['mean_SHAP_present']),
                'mean_abs_SHAP': float(source['mean_abs_SHAP']),
                'presence_direction': source['presence_direction'],
            })
        rows.append(row)
    return pd.DataFrame(rows)


def group_split_leakage(splits: pd.DataFrame) -> dict:
    """公司 Group Split 的泄漏检查：同公司不得跨子集。"""
    frame = splits.dropna(subset=['company_group_split'])
    grouped = frame.groupby('company_entity_id')['company_group_split'].nunique()
    leaked = grouped[grouped > 1]
    return {
        '公司实体总数': int(grouped.shape[0]),
        '跨子集公司数': int(leaked.shape[0]),
        '是否泄漏': '否' if leaked.empty else '是',
        'train_公司数': int(frame.loc[frame['company_group_split'].eq('train'),
                                  'company_entity_id'].nunique()),
        'validation_公司数': int(frame.loc[frame['company_group_split'].eq('validation'),
                                   'company_entity_id'].nunique()),
        'test_公司数': int(frame.loc[frame['company_group_split'].eq('test'),
                                 'company_entity_id'].nunique()),
    }


def group_error_table(frame: pd.DataFrame, model_frame: pd.DataFrame,
                      label: str) -> pd.DataFrame:
    """09_分组误差：指定子集上的 城市 / 岗位大类 / 薪资分位段 MAE。"""
    merged = frame.merge(model_frame[[schema.ID_FIELD, '工作城市_规范', '岗位大类集合']],
                         on=schema.ID_FIELD, how='left')
    merged['绝对误差'] = merged['residual'].abs()
    merged['薪资分位段'] = pd.qcut(merged['y_true'], q=5,
                              labels=['Q1 最低', 'Q2', 'Q3', 'Q4', 'Q5 最高']).astype(str)
    rows = []
    for dimension in ['工作城市_规范', '薪资分位段']:
        grouped = merged.groupby(dimension)['绝对误差'].agg(['count', 'mean', 'median'])
        for value, record in grouped.iterrows():
            if int(record['count']) < 30:
                continue
            rows.append({'维度': dimension, '取值': value, '样本数': int(record['count']),
                         'MAE': round(float(record['mean']), 4),
                         '中位数绝对误差': round(float(record['median']), 4)})
    exploded = merged.explode('岗位大类集合').dropna(subset=['岗位大类集合'])
    grouped = exploded.groupby('岗位大类集合')['绝对误差'].agg(['count', 'mean', 'median'])
    for value, record in grouped.iterrows():
        if int(record['count']) < 30:
            continue
        rows.append({'维度': '岗位大类', '取值': value, '样本数': int(record['count']),
                     'MAE': round(float(record['mean']), 4),
                     '中位数绝对误差': round(float(record['median']), 4)})
    table = pd.DataFrame(rows).sort_values(['维度', 'MAE'], ascending=[True, False])
    table['数据子集'] = label
    table['说明'] = '分组误差诊断（跨公司泛化子集）'
    return table

# -*- coding: utf-8 -*-
"""Stage 14：薪资预测模型对比、验证集选模与一次性 test 评估。

流程（严格无泄漏）：
    1. 固定同一划分（train 70% / validation 15% / test 15%，random_state=42，按薪资分位分层）；
    2. 预处理（数值填补 + 缺失指示 + 类别 OneHot + 多值 multi-hot + 技能 multi-hot + 文本 SVD）
       全部只在 train 上 fit；
    3. 技能阈值（50/80/100/0.5% train）与文本维度（16/32/64）用参考模型在 validation MAE 上选择；
    4. 各模型小网格超参数用 validation MAE 选择；按 validation MAE 选出主模型；
    5. 锁定最终配置 → train+validation 重拟合 → **test 只评估一次**；
    6. 保存预测、完整 Pipeline 产出与审计表。

输入（封版）：data/processed/job_salary_model_dataset.parquet（14,883）
输出：
    data/processed/model_splits.parquet
    data/processed/model_predictions.parquet
    outputs/tables/30_model_comparison.xlsx
    outputs/models/salary_model/（pipeline + manifest + skill columns + encoder schema + reducer + params）
    outputs/figures/modeling/01_..05_...
    docs/records/21_salary_model_comparison_record.md

本轮禁止：SHAP、正式消融、用 Group Split 重新选主模型、根据 test 反复调参、git commit。

用法：
    python scripts/14_train_salary_model.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import (io_utils, model_training, plot_style,  # noqa: E402
                 project_paths, quality, schema, skill_eda)

STAGE = 'stage_14_model'
TITLE = 'Stage 14 薪资预测模型对比、验证集选模与一次性 test 评估'
EXPECTED_SAMPLE = 14883

MODEL_GRIDS = {
    'Ridge': [{'alpha': alpha} for alpha in (0.1, 1.0, 10.0, 100.0)],
    'RandomForest': [{'n_estimators': 300, 'max_depth': depth, 'min_samples_leaf': leaf,
                      'max_features': 'sqrt'}
                     for depth in (None, 20) for leaf in (1, 5)],
    'CatBoost': [{'depth': 6, 'learning_rate': 0.05, 'iterations': 500},
                 {'depth': 8, 'learning_rate': 0.05, 'iterations': 500}],
    'LightGBM': [{'n_estimators': 400, 'learning_rate': 0.05, 'num_leaves': 31},
                 {'n_estimators': 400, 'learning_rate': 0.05, 'num_leaves': 63},
                 {'n_estimators': 600, 'learning_rate': 0.03, 'num_leaves': 31}],
}
REFERENCE_MODEL = 'RandomForest'
REFERENCE_PARAMS = {'n_estimators': 300, 'max_depth': None, 'min_samples_leaf': 1,
                    'max_features': 'sqrt'}
SCALE_FOR = {'Ridge'}


def numeric_feature_columns() -> list:
    """A/B/C/D/E 中参与模型的数值字段（来自 Stage 12 Feature Manifest 口径）。"""
    return ['学历等级', '每周到岗天数', '实习月数', '岗位大类数量', '岗位细分类数量',
            '岗位标签数量', '岗位描述字符数', '岗位描述分词数', schema.JOB_DIRECTION_MATCH_FIELD,
            '核心版本数', '完整页面版本数', '是否多版本岗位', '是否直辖市',
            '映射置信度', '是否需人工复核', '是否跨地域', '是否存在同名跨地域歧义',
            '公司规模下限', '公司规模上限', '公司规模中点', '公司规模等级', '公司规模是否已知',
            schema.SCALE_SHIFT_FIELD, '公司标签数量', '公司简介字符数',
            schema.SKILL_COUNT_FIELD, '技术技能数', '工程工具数', '数据工具数', '业务能力数',
            '技术领域数', schema.AI_SKILL_COUNT_FIELD, schema.LLM_SKILL_COUNT_FIELD,
            schema.SKILL_LANGUAGE_COUNT_FIELD, schema.SKILL_DATABASE_COUNT_FIELD,
            schema.SKILL_SECURITY_COUNT_FIELD, schema.SKILL_OFFICE_COUNT_FIELD,
            model_training.__dict__.get('TEXT_EMPTY_FIELD', '文本是否为空'), '是否有技能',
            '文本向量是否可用']


def categorical_feature_columns() -> list:
    return ['学历要求', '每周到岗要求', '实习时长要求', schema.JOB_DIRECTION_FIELD,
            schema.JD_SPLIT_STATUS_FIELD, '工作城市', '工作城市_规范', '所属行业', '公司性质',
            '公司规模', schema.SKILL_SCOPE_FIELD]


def build_split_sheet(splits: pd.DataFrame, model_frame: pd.DataFrame,
                      group_split: pd.DataFrame) -> pd.DataFrame:
    salary = model_frame.set_index(schema.ID_FIELD)[schema.SALARY_MID_FIELD]
    rows = []
    for label in model_training.SPLIT_LABELS:
        ids = splits.loc[splits['split'].eq(label), schema.ID_FIELD]
        values = salary.reindex(ids)
        rows.append({
            'split': label, '岗位数': int(len(ids)),
            '占比': round(len(ids) / len(splits), 6),
            '薪资中点_中位数': round(float(values.median()), 2),
            '薪资中点_均值': round(float(values.mean()), 2),
            '薪资中点_IQR': round(float(values.quantile(0.75) - values.quantile(0.25)), 2),
            '薪资中点_最小': round(float(values.min()), 2),
            '薪资中点_最大': round(float(values.max()), 2),
        })
    table = pd.DataFrame(rows)
    group_counts = group_split.groupby('company_group_split')['company_entity_id'].nunique()
    rows = [{'split': label, '岗位数': int((group_split['company_group_split'] == label).sum()),
             '公司实体数': int(group_counts.get(label, 0)),
             '说明': '公司 Group Split：同一公司不同时出现在 train/test（Stage 15 稳健性使用，'
                     '本轮不用于选主模型）'} for label in model_training.SPLIT_LABELS]
    group_table = pd.DataFrame(rows)
    group_table.insert(0, '划分类型', 'company_group_split')
    table.insert(0, '划分类型', '主划分（modelling split）')
    return pd.concat([table, group_table], ignore_index=True, sort=False)


def build_dimension_sheet(assembler: model_training.SalaryFeatureAssembler,
                          threshold_table: pd.DataFrame,
                          dim_table: pd.DataFrame) -> pd.DataFrame:
    spec = assembler.schema
    rows = [
        {'特征块': '数值（含缺失指示）',
         '列数': len(spec.numeric_columns) + len(spec.numeric_indicator_columns),
         '说明': f'数值列 {len(spec.numeric_columns)}，缺失指示 {len(spec.numeric_indicator_columns)}'
                 f'（缺失列：{"、".join(spec.numeric_indicator_columns) or "无"}）'},
        {'特征块': '类别 OneHot',
         '列数': sum(len(levels) for levels in spec.category_levels.values()),
         '说明': f'{len(spec.category_levels)} 个类别字段；训练集频次 ≥ '
                 f'{model_training.MIN_CATEGORY_FREQUENCY} 的取值保留，'
                 f'其余并入「{model_training.OTHER_LABEL}」；缺失编码为「'
                 f'{model_training.UNKNOWN_LABEL}」'},
        {'特征块': '多值 multi-hot',
         '列数': sum(len(levels) for levels in spec.multi_value_levels.values()),
         '说明': '岗位大类 / 岗位细分类 / 公司认证标签（各自取训练集 Top'
                 f'{model_training.TOP_MULTI_VALUES}）'},
        {'特征块': '技能 multi-hot（D 组）', '列数': len(spec.skill_columns),
         '说明': f'阈值 = {spec.skill_threshold}（train 频率），口径 ALL_USABLE'},
        {'特征块': '文本语义 SVD（E 组）', '列数': spec.text_dim,
         '说明': 'model-safe BGE 向量在训练集上拟合 TruncatedSVD 后变换'},
        {'特征块': '合计', '列数': spec.dimension, '说明': '实际进入模型的特征维度'},
    ]
    table = pd.DataFrame(rows)
    table.insert(0, '块', '最终特征维度')
    return pd.concat([table,
                      threshold_table.assign(块='技能阈值选择'),
                      dim_table.assign(块='文本维度选择')],
                     ignore_index=True, sort=False)


def build_residual_sheet(predictions: pd.DataFrame, final_name: str) -> pd.DataFrame:
    frame = predictions[predictions['model_name'].eq(final_name)]
    rows = []
    for split in ('validation', 'test'):
        block = frame[frame['split'].eq(split)]
        if block.empty:
            continue
        stats = model_training.regression_metrics(block['y_true'], block['y_pred'])
        residual = block['residual']
        rows.append({'split': split, **stats,
                     '残差均值': round(float(residual.mean()), 4),
                     '残差标准差': round(float(residual.std(ddof=1)), 4),
                     '残差P10': round(float(residual.quantile(0.10)), 4),
                     '残差P50': round(float(residual.quantile(0.50)), 4),
                     '残差P90': round(float(residual.quantile(0.90)), 4),
                     '残差绝对值>100的占比': round(float((residual.abs() > 100).mean()), 6)})
    return pd.DataFrame(rows)


def build_error_group_sheet(model_frame: pd.DataFrame, frame_test: pd.DataFrame,
                            final_name: str) -> pd.DataFrame:
    """10_预测残差（分组误差诊断）：城市 / 岗位大类 / 学历 / 公司规模 / 薪资分位段。"""
    merged = frame_test.copy()
    merged = merged.merge(model_frame[[schema.ID_FIELD, '工作城市_规范', '学历要求', '公司规模',
                                       '岗位大类集合', schema.SALARY_MID_FIELD]],
                          on=schema.ID_FIELD, how='left')
    merged['绝对误差'] = merged['residual'].abs()
    merged['薪资分位段'] = pd.qcut(merged['y_true'], q=5,
                              labels=['Q1 最低', 'Q2', 'Q3', 'Q4', 'Q5 最高']).astype(str)
    rows = []
    for dimension in ['工作城市_规范', '学历要求', '公司规模', '薪资分位段']:
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
    table['说明'] = '正式主模型在 test 上的分组误差诊断（仅误差诊断，不含敏感属性分析）'
    return table


def figure_model_comparison(comparison: pd.DataFrame, registry: list) -> dict:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    frame = comparison.copy()
    labels = frame['模型'].tolist()
    positions = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.bar(positions - 0.2, frame['validation_MAE'], width=0.38, color=plot_style.MAIN_COLOR,
           edgecolor='black', linewidth=0.5, label='validation MAE')
    ax.bar(positions + 0.2, frame['test_MAE'], width=0.38, color=plot_style.ACCENT_COLOR,
           edgecolor='black', linewidth=0.5, label='test MAE')
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_xlabel('模型')
    ax.set_ylabel('MAE（元/天）')
    ax.set_ylim(0, float(np.nanmax([frame['validation_MAE'].max(), frame['test_MAE'].max()])) * 1.35)
    ax.legend(loc='upper right', frameon=False)
    plot_style.apply_sci_axis(ax, grid_axis='y')
    caption = '图11 薪资预测模型对比（主指标 MAE，validation 选模，test 仅一次评估）'
    plot_style.add_bottom_caption(fig, caption)
    fig.subplots_adjust(bottom=0.24)
    return _finish(fig, '01_model_comparison', caption, registry,
                   meta={'图表类型': '对比柱状图', '数据来源': '30 号模型对比表'})


def figure_prediction_scatter(frame_test: pd.DataFrame, final_name: str, registry: list) -> dict:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.scatter(frame_test['y_true'], frame_test['y_pred'], s=7, alpha=0.35,
               color=plot_style.MAIN_COLOR, edgecolor='none')
    limit = [float(frame_test['y_true'].min()), float(frame_test['y_true'].max())]
    ax.plot(limit, limit, color=plot_style.MUTED_COLOR, linestyle='--', linewidth=0.9)
    metrics = model_training.regression_metrics(frame_test['y_true'], frame_test['y_pred'])
    ax.text(0.04, 0.96, f"MAE = {metrics['MAE']:.1f}\nR² = {metrics['R2']:.3f}",
            transform=ax.transAxes, va='top', ha='left',
            fontsize=plot_style.FONT_SIZES['annotation'])
    ax.set_xlabel('真实薪资中点（元/天）')
    ax.set_ylabel('预测薪资中点（元/天）')
    plot_style.apply_sci_axis(ax)
    caption = f'图12 正式主模型（{final_name}）test 预测值 vs 真实值'
    plot_style.add_bottom_caption(fig, caption)
    fig.subplots_adjust(bottom=0.16)
    return _finish(fig, '02_prediction_scatter', caption, registry,
                   meta={'图表类型': '散点图', '数据来源': 'model_predictions.parquet'})


def figure_residuals(frame_test: pd.DataFrame, final_name: str, registry: list) -> dict:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    ax = axes[0]
    plot_style.hist_discrete(ax, frame_test['residual'], discrete=False, bins=50,
                             xlabel='残差（真实 − 预测，元/天）', ylabel='岗位数')
    plot_style.apply_sci_axis(ax, grid_axis='y')
    plot_style.add_subfigure_caption(ax, 'a', '残差分布直方图')
    ax2 = axes[1]
    ax2.scatter(frame_test['y_pred'], frame_test['residual'], s=7, alpha=0.3,
                color=plot_style.MAIN_COLOR, edgecolor='none')
    ax2.axhline(0, color=plot_style.MUTED_COLOR, linestyle='--', linewidth=0.9)
    ax2.set_xlabel('预测薪资中点（元/天）')
    ax2.set_ylabel('残差（元/天）')
    plot_style.apply_sci_axis(ax2)
    plot_style.add_subfigure_caption(ax2, 'b', '残差 vs 预测值')
    fig.subplots_adjust(wspace=0.3, bottom=0.3)
    caption = f'图13 正式主模型（{final_name}）test 残差诊断'
    plot_style.add_bottom_caption(fig, caption)
    return _finish(fig, '03_residual_diagnostics', caption, registry,
                   subfigures=[('a', '残差分布直方图', axes[0]), ('b', '残差 vs 预测值', axes[1])],
                   meta={'图表类型': '直方图 + 散点图', '数据来源': 'model_predictions.parquet'})


def figure_error_groups(error_table: pd.DataFrame, registry: list) -> dict:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    frame = error_table[error_table['维度'].isin(['学历要求', '公司规模', '薪资分位段'])]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.0))
    panels = [('a', '学历要求'), ('b', '公司规模'), ('c', '薪资分位段')]
    for (letter, dimension), ax in zip(panels, axes):
        block = frame[frame['维度'].eq(dimension)].sort_values('MAE')
        positions = np.arange(len(block))
        ax.barh(positions, block['MAE'], height=0.6, color=plot_style.MAIN_COLOR,
                edgecolor='black', linewidth=0.5)
        ax.set_yticks(positions)
        ax.set_yticklabels([f'{value}（n={int(count)}）' for value, count
                            in zip(block['取值'], block['样本数'])])
        ax.set_xlabel('MAE（元/天）')
        ax.set_ylabel(dimension)
        plot_style.apply_sci_axis(ax, grid_axis='x')
        plot_style.add_subfigure_caption(ax, letter, f'{dimension}分组 MAE')
    fig.subplots_adjust(wspace=0.62, bottom=0.3)
    caption = '图14 正式主模型 test 分组误差诊断（避免总体 MAE 掩盖主要群体偏差）'
    plot_style.add_bottom_caption(fig, caption)
    return _finish(fig, '04_error_groups', caption, registry,
                   subfigures=[(letter, f'{dimension}分组 MAE', ax)
                               for (letter, dimension), ax in zip(panels, axes)],
                   meta={'图表类型': '三联横向柱状图', '数据来源': '30 号分组误差表'})


def _finish(fig, stem: str, caption: str, registry: list, subfigures=None, meta=None) -> dict:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    diagnostics = plot_style.save_sci_figure(fig, stem, caption, subfigures=subfigures, meta=meta)
    plt.close(fig)
    registry.append(diagnostics)
    return diagnostics


def write_record(metrics: dict, audit: dict) -> Path:
    split_table = audit['01_数据划分']
    comparison = audit['08_Validation比较']
    test_table = audit['09_Test最终结果']
    lines = [
        '# 记录 21：Stage 14 薪资预测模型对比、验证集选模与一次性 test 评估',
        '',
        '> 本记录由 `scripts/14_train_salary_model.py` 自动生成，数字全部来自真实运行结果。',
        '> 本轮**不做 SHAP、不做正式消融、不使用公司 Group Split 重新选主模型、不据 test 反复调参**。',
        '',
        '## 1. 数据划分（所有模型复用同一划分）',
        '',
        '| 划分 | 岗位数 | 占比 | 薪资中点中位数 | IQR |',
        '| --- | --- | --- | --- | --- |',
    ]
    for row in split_table[split_table['划分类型'].str.startswith('主划分')].itertuples(index=False):
        lines.append(f'| {row.split} | {int(row.岗位数):,} | {row.占比:.2%} | '
                     f'{row.薪资中点_中位数} | {row.薪资中点_IQR} |')
    lines += [
        '',
        f"- 划分方式：按薪资中点 {metrics['split_bins']} 分位分箱分层，`random_state={metrics['random_state']}`，"
        f"`train/validation/test = 70%/15%/15%`；",
        '- 同一个划分文件 `data/processed/model_splits.parquet` 供全部模型复用，不重复随机分割；',
        '- 公司 Group Split（按 `company_entity_id` 分组，同一公司不同时出现在 train/test）'
        '已同时准备并写入同一文件，**本轮不用于选主模型**（留给 Stage 15 稳健性实验）。',
        '',
        '## 2. 特征组与维度',
        '',
        f"- 特征组：A+B+C+D+E（E 组使用 model-safe BGE 向量，{metrics['embedding_status']}）；",
        f"- 最终特征维度：**{metrics['final_dimension']}**（数值 / 类别 OneHot / 多值 multi-hot / "
        f'技能 multi-hot / 文本 SVD）；',
        f"- 技能 multi-hot 阈值 = **{metrics['skill_threshold']}**（train 频率），技能列数 "
        f"{metrics['skill_columns']}；文本维度 = **{metrics['text_dim']}**；",
        '- 所有预处理（填补、缺失指示、类别编码、Scaler、技能频率筛选与列集合、文本降维）'
        '**只在 train 上 fit**，再 transform validation/test。',
        '',
        '## 3. 技能阈值选择（只依据 validation MAE）',
        '',
        '| 阈值口径 | 阈值 | 技能列数 | 特征维度 | validation MAE |',
        '| --- | --- | --- | --- | --- |',
    ]
    for row in audit['12_技能阈值选择'].itertuples(index=False):
        lines.append(f'| {row.阈值口径} | {int(row.阈值)} | {int(row.技能列数)} | '
                     f'{int(row.特征维度)} | {row.validation_MAE} |')
    lines += [
        '',
        '| 文本维度 | 特征维度 | validation MAE | 说明 |',
        '| --- | --- | --- | --- |',
    ]
    for row in audit['13_文本维度选择'].itertuples(index=False):
        lines.append(f'| {row.文本维度} | {int(row.特征维度)} | {row.validation_MAE} | {row.说明} |')
    lines += [
        '',
        '## 4. 调参范围（小网格，全部以 validation MAE 选择）',
        '',
        '| 模型 | 候选网格 |',
        '| --- | --- |',
    ]
    for model_name, grid in MODEL_GRIDS.items():
        summary = '；'.join(str(item) for item in grid)
        lines.append(f'| {model_name} | {summary} |')
    lines += [
        '',
        f"- XGBoost：{metrics['xgboost_status']}（未为了凑模型数量安装依赖）；",
        '',
        '## 5. Validation 比较与主模型选择',
        '',
        '| 模型 | 最佳超参数 | validation MAE | validation RMSE | validation R² |',
        '| --- | --- | --- | --- | --- |',
    ]
    for row in comparison.itertuples(index=False):
        lines.append(f'| {row.模型} | {row.最佳超参数} | {row.validation_MAE} | '
                     f'{row.validation_RMSE} | {row.validation_R2} |')
    lines += [
        '',
        f"- 主模型按 **validation MAE 最小**选择：**{metrics['final_model']}**；",
        f"- 选择依据仅使用 train/validation，test 未参与任何选择；",
        '',
        '## 6. Test 评估（锁定配置后只评估一次）',
        '',
        '| 模型 | test MAE | test RMSE | test R² | n |',
        '| --- | --- | --- | --- | --- |',
    ]
    for row in test_table[test_table['test_MAE'].notna()].itertuples(index=False):
        lines.append(f'| {row.模型} | {row.test_MAE} | {row.test_RMSE} | {row.test_R2} | '
                     f'{int(row.n)} |')
    lines += [
        '',
        f"- 最终模型在 train+validation（{metrics['fit_rows_final']:,} 个岗位）上重拟合后，"
        f"对 test（{metrics['test_rows']:,} 个岗位）评估一次：",
        f"  **MAE = {metrics['final_test_MAE']} 元/天，RMSE = {metrics['final_test_RMSE']}，"
        f"R² = {metrics['final_test_R2']}**；",
        f"- 相对 Dummy（均值基线）的 MAE 改善：**{metrics['mae_improvement_vs_dummy']:.2%}**；",
        f"- test 评估次数：**{metrics['test_evaluation_count']}**（一次性，未据 test 调整任何配置）。",
        '',
        '## 7. 分组误差诊断（test）',
        '',
        '| 维度 | 取值 | 样本数 | MAE |',
        '| --- | --- | --- | --- |',
    ]
    error_rows = audit['10_预测残差']
    error_rows = error_rows[error_rows['维度'].notna()].head(15)
    for row in error_rows.to_dict('records'):
        lines.append(f"| {row['维度']} | {row['取值']} | {int(row['样本数'])} | {row['MAE']} |")
    lines += [
        '',
        '## 8. 产物',
        '',
        f"- 划分文件：`{metrics['splits_path']}`（仅 intern_id / split / company_group_split，"
        '不含目标值）；',
        f"- 预测文件：`{metrics['predictions_path']}`（intern_id / split / y_true / y_pred / "
        'residual / model_name）；',
        f"- 模型产出：`{metrics['artifact_dir']}`（pipeline + feature manifest + skill columns + "
        'category encoder schema + text reducer + model params，禁止只保存裸 estimator）；',
        f"- 审计表：`{metrics['audit_path']}`；图：`outputs/figures/modeling/`。",
        '',
        '## 9. 门禁',
        '',
        '| 门禁项 | 状态 | 说明 |',
        '| --- | --- | --- |',
    ]
    gate_detail = audit['11_门禁'].set_index('门禁项')['说明']
    for name, status in metrics['gates'].items():
        lines.append(f'| {name} | {status} | {gate_detail.get(name, "")} |')
    lines += [
        '',
        '## 10. 边界',
        '',
        '- 未执行 SHAP、未执行正式消融（Stage 15）；',
        '- 未使用公司 Group Split 选主模型；',
        '- 未根据 test 结果调整任何配置；',
        '- 本轮未执行 git commit。',
        '',
    ]
    return io_utils.write_markdown(
        project_paths.RECORDS_DIR / project_paths.RECORD_SALARY_MODEL, lines)


def main() -> int:
    quality.configure_logging()
    gates = quality.GateRegistry(STAGE)
    quality.stage_banner(STAGE, TITLE)

    model_frame = io_utils.read_parquet(project_paths.JOB_SALARY_MODEL_DATASET_PARQUET)
    analysis = io_utils.read_parquet(project_paths.JOB_ANALYSIS_DATASET_PARQUET,
                                     columns=[schema.ID_FIELD, schema.SKILL_SCOPE_FIELD])
    membership = io_utils.read_parquet(project_paths.JOB_SKILL_MEMBERSHIP_PARQUET)
    universe = skill_eda.load_scope_universe(analysis)
    print(f'建模样本 {len(model_frame):,}；技能关系 {len(membership):,} 行')

    # ---- 划分 ----
    splits = model_training.build_splits(model_frame)
    group_split = model_training.build_company_group_split(model_frame)
    splits_path = io_utils.write_parquet(
        splits.merge(group_split[[schema.ID_FIELD, 'company_group_split', 'company_entity_id']],
                     on=schema.ID_FIELD, how='left'),
        project_paths.MODEL_SPLITS_PARQUET)
    split_counts = splits['split'].value_counts().to_dict()
    train_ids = set(splits.loc[splits['split'].eq('train'), schema.ID_FIELD])
    valid_ids = set(splits.loc[splits['split'].eq('validation'), schema.ID_FIELD])
    test_ids = set(splits.loc[splits['split'].eq('test'), schema.ID_FIELD])
    gates.check('MODEL_SPLIT',
                len(splits) == EXPECTED_SAMPLE
                and set(splits['split']) == set(model_training.SPLIT_LABELS)
                and abs(split_counts['train'] / len(splits) - 0.70) < 0.01
                and abs(split_counts['validation'] / len(splits) - 0.15) < 0.01,
                f"train {split_counts['train']:,} / validation {split_counts['validation']:,} / "
                f"test {split_counts['test']:,}（random_state=42，按薪资分位分层）")
    gates.check('MODEL_SPLIT_DISJOINT',
                not (train_ids & valid_ids) and not (train_ids & test_ids)
                and not (valid_ids & test_ids)
                and len(train_ids) + len(valid_ids) + len(test_ids) == EXPECTED_SAMPLE,
                '三个子集互不相交且并集等于 14,883 个正式样本；公司 Group Split 已另存备用')

    # ---- 特征装配（train-only 拟合） ----
    train_frame = model_frame[model_frame[schema.ID_FIELD].isin(train_ids)].reset_index(drop=True)
    valid_frame = model_frame[model_frame[schema.ID_FIELD].isin(valid_ids)].reset_index(drop=True)
    test_frame = model_frame[model_frame[schema.ID_FIELD].isin(test_ids)].reset_index(drop=True)

    skill_map = model_training.build_skill_map(membership, skill_eda.ALL_USABLE_SCOPES)
    all_ids = model_frame[schema.ID_FIELD].tolist()
    text_matrix = model_training.load_text_matrix(
        all_ids, project_paths.FEATURES_DIR / 'job_text_embeddings.npz',
        project_paths.FEATURES_DIR / 'job_text_embedding_index.parquet')
    text_by_id = {job_id: position for position, job_id in enumerate(all_ids)}

    def text_for(frame):
        return text_matrix[[text_by_id[job_id] for job_id in frame[schema.ID_FIELD]]]

    numeric_columns = numeric_feature_columns()
    categorical_columns = categorical_feature_columns()
    y_train = train_frame[schema.SALARY_MID_FIELD].to_numpy(dtype='float64')
    y_valid = valid_frame[schema.SALARY_MID_FIELD].to_numpy(dtype='float64')
    y_test = test_frame[schema.SALARY_MID_FIELD].to_numpy(dtype='float64')

    def assemble(threshold: int, text_dim: int, scale: bool = False):
        cache_key = (threshold, text_dim, scale)
        if cache_key in assemble_cache:
            return assemble_cache[cache_key]
        assembler = model_training.SalaryFeatureAssembler(
            numeric_columns, categorical_columns, model_training.MULTI_VALUE_COLUMNS,
            skill_threshold=threshold, text_dim=text_dim, scale_numeric=scale)
        assembler.fit(train_frame, skill_map, text_for(train_frame))
        result = (assembler,
                  assembler.transform(train_frame, skill_map, text_for(train_frame)),
                  assembler.transform(valid_frame, skill_map, text_for(valid_frame)),
                  assembler.transform(test_frame, skill_map, text_for(test_frame)))
        assemble_cache[cache_key] = result
        return result

    assemble_cache = {}

    # ---- 技能阈值选择（仅 validation） ----
    train_rows = len(train_frame)
    threshold_rows = []
    for threshold in model_training.SKILL_THRESHOLD_CANDIDATES:
        resolved = (max(1, int(round(train_rows * threshold))) if threshold == 0.005
                    else int(threshold))
        label = (f'>= 0.5% × {train_rows}（= {resolved}）' if threshold == 0.005
                 else f'>= {int(threshold)}')
        assembler, matrix_train, matrix_valid, _ = assemble(resolved, text_dim=0)
        started = time.time()
        fit = model_training.fit_and_predict(matrix_train, y_train, matrix_valid, None,
                                            REFERENCE_MODEL, REFERENCE_PARAMS)
        metrics = model_training.regression_metrics(y_valid, fit['valid_pred'])
        threshold_rows.append({
            '阈值口径': label, '阈值': resolved, '技能列数': len(assembler.schema.skill_columns),
            '特征维度': assembler.schema.dimension, 'validation_MAE': metrics['MAE'],
            'validation_RMSE': metrics['RMSE'], 'validation_R2': metrics['R2'],
            '参考模型': f'{REFERENCE_MODEL}（固定参数）', '训练耗时秒': round(time.time() - started, 1),
            '说明': '技能频率仅从 train 统计；选择依据仅 validation MAE',
        })
        print(f'阈值 {label}: validation MAE {metrics["MAE"]}')
    threshold_table = pd.DataFrame(threshold_rows)
    best_threshold_row = threshold_table.sort_values(['validation_MAE', '技能列数']).iloc[0]
    skill_threshold = int(best_threshold_row['阈值'])

    # ---- 文本维度选择（仅 validation） ----
    dimension_rows = []
    for text_dim in model_training.TEXT_DIM_CANDIDATES:
        assembler, matrix_train, matrix_valid, _ = assemble(skill_threshold, text_dim)
        fit = model_training.fit_and_predict(matrix_train, y_train, matrix_valid, None,
                                            REFERENCE_MODEL, REFERENCE_PARAMS)
        metrics = model_training.regression_metrics(y_valid, fit['valid_pred'])
        dimension_rows.append({
            '文本维度': text_dim, '特征维度': assembler.schema.dimension,
            'validation_MAE': metrics['MAE'], 'validation_RMSE': metrics['RMSE'],
            'validation_R2': metrics['R2'], '参考模型': f'{REFERENCE_MODEL}（固定参数）',
            '说明': 'SVD 仅在 train 上拟合；选择依据仅 validation MAE',
        })
        print(f'文本维度 {text_dim}: validation MAE {metrics["MAE"]}')
    assembler_reference, matrix_train, matrix_valid, _ = assemble(skill_threshold, 0)
    fit = model_training.fit_and_predict(matrix_train, y_train, matrix_valid, None,
                                        REFERENCE_MODEL, REFERENCE_PARAMS)
    metrics = model_training.regression_metrics(y_valid, fit['valid_pred'])
    dimension_rows.append({
        '文本维度': '不使用 E 组（参考）', '特征维度': assembler_reference.schema.dimension,
        'validation_MAE': metrics['MAE'], 'validation_RMSE': metrics['RMSE'],
        'validation_R2': metrics['R2'], '参考模型': f'{REFERENCE_MODEL}（固定参数）',
        '说明': '仅作参考行，不参与候选选择（正式消融在 Stage 15）'})
    dimension_table = pd.DataFrame(dimension_rows)
    candidates = dimension_table[dimension_table['文本维度'].isin(model_training.TEXT_DIM_CANDIDATES)]
    best_text_dim = int(candidates.sort_values('validation_MAE').iloc[0]['文本维度'])
    print(f'选定技能阈值 = {skill_threshold}；文本维度 = {best_text_dim}')

    # ---- 模型网格（仅 validation） ----
    model_results = {}
    grid_details = {}
    prediction_frames = []
    for model_key, grid in MODEL_GRIDS.items():
        rows = []
        best = None
        for params in grid:
            scale = model_key in SCALE_FOR
            assembler, matrix_train, matrix_valid, matrix_test = assemble(
                skill_threshold, best_text_dim, scale=scale)
            try:
                fit = model_training.fit_and_predict(matrix_train, y_train, matrix_valid,
                                                    matrix_test, model_key, params)
            except ImportError as error:
                rows.append({'模型': model_key, '超参数': str(params), 'validation_MAE': None,
                             '状态': f'NOT_RUN（{error}）'})
                break
            metrics = model_training.regression_metrics(y_valid, fit['valid_pred'])
            rows.append({'模型': model_key, '超参数': str(params),
                         'validation_MAE': metrics['MAE'], 'validation_RMSE': metrics['RMSE'],
                         'validation_R2': metrics['R2'], '状态': 'OK'})
            if best is None or metrics['MAE'] < best['metrics']['MAE']:
                best = {'params': params, 'metrics': metrics, 'fit': fit,
                        'matrix': (matrix_train, matrix_valid, matrix_test),
                        'assembler': assembler}
        grid_details[model_key] = pd.DataFrame(rows)
        if best is not None:
            model_results[model_key] = best
            print(f'{model_key}: 最佳 validation MAE {best["metrics"]["MAE"]}（{best["params"]}）')

    # Dummy 基线（同一矩阵）
    assembler_base, matrix_train, matrix_valid, matrix_test = assemble(skill_threshold, best_text_dim)
    dummy_fit = model_training.fit_and_predict(matrix_train, y_train, matrix_valid, matrix_test,
                                              'Dummy', {})
    dummy_metrics = model_training.regression_metrics(y_valid, dummy_fit['valid_pred'])
    model_results['Dummy'] = {'params': {'strategy': 'mean'}, 'metrics': dummy_metrics,
                              'fit': dummy_fit, 'matrix': (matrix_train, matrix_valid, matrix_test),
                              'assembler': assembler_base}
    grid_details['Dummy'] = pd.DataFrame([{'模型': 'Dummy', '超参数': "{'strategy': 'mean'}",
                                           'validation_MAE': dummy_metrics['MAE'],
                                           'validation_RMSE': dummy_metrics['RMSE'],
                                           'validation_R2': dummy_metrics['R2'], '状态': 'OK'}])
    print(f"Dummy: validation MAE {dummy_metrics['MAE']}")

    # ---- Validation 比较 → 选主模型 ----
    comparison_rows = []
    for model_key, result in model_results.items():
        comparison_rows.append({
            '模型': model_key, '最佳超参数': str(result['params']),
            'validation_MAE': result['metrics']['MAE'],
            'validation_RMSE': result['metrics']['RMSE'],
            'validation_R2': result['metrics']['R2'],
            '特征维度': result['assembler'].schema.dimension,
            '是否基线': '是' if model_key in ('Dummy',) else '否',
            '说明': 'validation MAE 最小者为主模型',
        })
    comparison = pd.DataFrame(comparison_rows).sort_values('validation_MAE').reset_index(drop=True)
    non_dummy = comparison[comparison['模型'] != 'Dummy']
    final_model_key = non_dummy.iloc[0]['模型']
    final_params = model_results[final_model_key]['params']
    gates.check('MODEL_VALIDATION_SELECTION',
                final_model_key in model_results
                and comparison.iloc[0]['模型'] == final_model_key
                and {'Dummy', 'Ridge'} <= set(comparison['模型']),
                f'按 validation MAE 选出主模型 {final_model_key}'
                f"（{comparison.iloc[0]['validation_MAE']}）；"
                f"排序前 3：{'、'.join(comparison.head(3)['模型'].tolist())}；"
                '选择只用 train/validation')

    # ---- 锁定配置 → 预处理器保持 train-only 拟合，估计器在 train+validation 重拟合 → test 只评估一次 ----
    fit_ids = train_ids | valid_ids
    fit_frame = model_frame[model_frame[schema.ID_FIELD].isin(fit_ids)].reset_index(drop=True)
    # 预处理器沿用选模阶段在 train 上拟合的装配器（技能列、类别水平、中位数、SVD 均不接触 validation/test）
    final_assembler = model_results[final_model_key]['assembler']
    matrix_fit = final_assembler.transform(fit_frame, skill_map, text_for(fit_frame))
    matrix_test_final = final_assembler.transform(test_frame, skill_map, text_for(test_frame))
    final_model = model_training.make_model(final_model_key, final_params)
    final_model.fit(matrix_fit, fit_frame[schema.SALARY_MID_FIELD].to_numpy(dtype='float64'))
    test_pred = np.asarray(final_model.predict(matrix_test_final), dtype='float64')
    final_test = model_training.regression_metrics(y_test, test_pred)
    test_evaluation_count = 1
    print(f'最终模型 {final_model_key} test MAE = {final_test["MAE"]}（一次性评估）')

    gates.check('MODEL_TRAIN_ONLY_PREPROCESS',
                final_assembler.schema.fit_rows == len(train_frame)
                and final_assembler.schema.fit_scope == 'train'
                and final_assembler.schema.fit_rows + len(test_frame) < EXPECTED_SAMPLE,
                f'预处理器（填补 / 编码 / Scaler / 技能列 / SVD）只在 train（{len(train_frame):,} 行）上 fit；'
                f'估计器在 train+validation（{len(fit_frame):,} 行）上重拟合；'
                f'test（{len(test_frame):,} 行）从未参与拟合')
    train_frequency = model_training.skill_frequency_from_train(
        membership, train_ids, skill_eda.ALL_USABLE_SCOPES)
    expected_skills = model_training.selected_skill_columns(train_frequency, skill_threshold)
    gates.check('MODEL_SKILL_THRESHOLD_TRAIN_ONLY',
                expected_skills == sorted(final_assembler.schema.skill_columns)
                and len(expected_skills) >= 20,
                f'技能列集合 = train 频率 ≥ {skill_threshold} 的技能（{len(expected_skills)} 个），'
                f'与产出中 skill_columns 完全一致；验证/测试岗位未参与频率统计')
    gates.check('MODEL_TEST_SINGLE_EVAL',
                test_evaluation_count == 1
                and len(splits[splits['split'].eq('test')]) == len(test_frame),
                f'test 仅在锁定配置后评估一次（n = {len(test_frame):,}，MAE {final_test["MAE"]}），'
                '技能阈值/文本维度/超参数/模型族均在 train+validation 上确定，未据 test 调整')

    # ---- 预测结果 ----
    prediction_rows = []
    for model_key, result in model_results.items():
        valid_pred = result['fit']['valid_pred']
        test_pred_used = result['fit']['test_pred']
        for split_label, truth, pred in [('validation', y_valid, valid_pred),
                                        ('test', y_test, test_pred_used)]:
            if pred is None:
                continue
            frame = pd.DataFrame({
                schema.ID_FIELD: (valid_frame[schema.ID_FIELD].to_numpy()
                                  if split_label == 'validation'
                                  else test_frame[schema.ID_FIELD].to_numpy()),
                'split': split_label, 'y_true': truth, 'y_pred': pred,
                'residual': truth - pred, 'model_name': model_key})
            prediction_rows.append(frame)
    final_frame = pd.DataFrame({
        schema.ID_FIELD: test_frame[schema.ID_FIELD].to_numpy(), 'split': 'test',
        'y_true': y_test, 'y_pred': test_pred, 'residual': y_test - test_pred,
        'model_name': f'FINAL（{final_model_key}）'})
    predict_frame = pd.concat([*prediction_rows, final_frame], ignore_index=True)
    predictions_path = io_utils.write_parquet(predict_frame, project_paths.MODEL_PREDICTIONS_PARQUET)

    # ---- 模型产出 ----
    artifact_dir = project_paths.SALARY_MODEL_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({'assembler': final_assembler, 'model': final_model},
                artifact_dir / 'salary_model_pipeline.joblib')
    model_training.dump_json(artifact_dir / 'feature_manifest.json', {
        'numeric_columns': numeric_columns, 'categorical_columns': categorical_columns,
        'multi_value_columns': list(model_training.MULTI_VALUE_COLUMNS),
        'dimension': final_assembler.schema.dimension,
        'feature_groups': 'A+B+C+D+E', 'skill_scope': 'ALL_USABLE',
        'source': 'data/processed/job_salary_model_dataset.parquet'})
    model_training.dump_json(artifact_dir / 'skill_columns.json',
                             {'threshold': skill_threshold,
                              'threshold_basis': 'train frequency (>= 50/80/100/0.5% train)',
                              'columns': final_assembler.schema.skill_columns})
    model_training.dump_json(artifact_dir / 'category_encoder_schema.json', {
        'min_frequency': model_training.MIN_CATEGORY_FREQUENCY,
        'unknown_label': model_training.UNKNOWN_LABEL,
        'other_label': model_training.OTHER_LABEL,
        'levels': final_assembler.schema.category_levels,
        'multi_value_levels': final_assembler.schema.multi_value_levels,
        'numeric_medians': final_assembler.schema.numeric_medians,
        'numeric_indicator_columns': final_assembler.schema.numeric_indicator_columns,
        'scale_numeric': final_assembler.schema.scale_numeric,
        'numeric_mean': final_assembler.schema.numeric_mean,
        'numeric_std': final_assembler.schema.numeric_std})
    np.savez_compressed(artifact_dir / 'text_reducer.npz',
                        components=np.asarray(final_assembler.schema.text_components,
                                              dtype='float32'),
                        text_dim=np.array([final_assembler.schema.text_dim]))
    model_training.dump_json(artifact_dir / 'model_params.json', {
        'model_key': final_model_key, 'params': final_params,
        'main_metric': model_training.MAIN_METRIC,
        'selected_on': 'validation', 'test_evaluated_once': True,
        'preprocessor_fit_rows': int(final_assembler.schema.fit_rows),
        'preprocessor_fit_scope': final_assembler.schema.fit_scope,
        'estimator_fit_rows': int(len(fit_frame)),
        'test_rows_excluded_from_fit': int(len(test_frame)),
        'test_metrics': final_test,
        'random_state': model_training.SPLIT_RANDOM_STATE,
        'skill_threshold': skill_threshold, 'text_dim': best_text_dim})
    model_training.dump_json(artifact_dir / 'preprocessor_fit_scope.json', {
        'numeric_medians': final_assembler.schema.numeric_medians,
        'numeric_indicator_columns': final_assembler.schema.numeric_indicator_columns,
        'category_levels': {column: len(levels) for column, levels
                            in final_assembler.schema.category_levels.items()},
        'skill_columns': len(final_assembler.schema.skill_columns),
        'text_dim': final_assembler.schema.text_dim,
        'fit_rows': int(final_assembler.schema.fit_rows), 'fit_scope': 'train',
        'note': '预处理器统计量（中位数 / 类别水平 / 技能列 / SVD 分量）只在 train 上估计；'
                'validation/test 仅被 transform'})
    model_training.dump_json(artifact_dir / 'metrics_summary.json', {
        'validation': {key: value for key, value in
                       model_results[final_model_key]['metrics'].items()},
        'test': final_test, 'baseline': {'Dummy': model_results['Dummy']['metrics']},
        'comparison': comparison.to_dict('records')})
    artifact_files = sorted(path.name for path in artifact_dir.glob('*'))
    gates.check('MODEL_ARTIFACT_EXPORT',
                {'salary_model_pipeline.joblib', 'feature_manifest.json', 'skill_columns.json',
                 'category_encoder_schema.json', 'text_reducer.npz', 'model_params.json',
                 'metrics_summary.json'} <= set(artifact_files),
                f'模型产出目录 {project_paths.relative_to_root(artifact_dir)} 含 '
                f'{len(artifact_files)} 个文件：pipeline + feature manifest + skill columns + '
                f'category encoder schema + text reducer + model params（非裸 estimator）')

    # ---- 审计表 ----
    split_sheet = build_split_sheet(splits, model_frame, group_split)
    dimension_sheet = build_dimension_sheet(final_assembler, threshold_table, dimension_table)
    comparison_sheet = comparison.rename(columns={'validation_MAE': 'validation_MAE',
                                                  'validation_RMSE': 'validation_RMSE',
                                                  'validation_R2': 'validation_R2'})
    test_rows = []
    for model_key, result in model_results.items():
        pred = result['fit']['test_pred']
        if pred is None:
            test_rows.append({'模型': model_key, 'test_MAE': None, '状态': 'NOT_RUN'})
            continue
        metrics = model_training.regression_metrics(y_test, pred)
        test_rows.append({'模型': model_key, 'test_MAE': metrics['MAE'],
                          'test_RMSE': metrics['RMSE'], 'test_R2': metrics['R2'],
                          'n': metrics['n'], '状态': 'OK',
                          '说明': '该行用于对照；正式结果为 FINAL 行（train+validation 重拟合）'})
    test_rows.append({'模型': f'FINAL（{final_model_key}）', 'test_MAE': final_test['MAE'],
                      'test_RMSE': final_test['RMSE'], 'test_R2': final_test['R2'],
                      'n': final_test['n'], '状态': 'OK',
                      '说明': '锁定配置后在 train+validation 上重拟合，test 仅评估一次'})
    test_sheet = pd.DataFrame(test_rows)
    residual_sheet = build_residual_sheet(predict_frame, f'FINAL（{final_model_key}）')
    error_group_sheet = build_error_group_sheet(model_frame, final_frame, final_model_key)

    audit_sheets = {
        '01_数据划分': split_sheet,
        '02_特征维度': dimension_sheet,
        '03_Baseline': grid_details['Dummy'].assign(说明='均值基线（DummyRegressor）'),
        '04_Ridge': grid_details['Ridge'],
        '05_RandomForest': grid_details['RandomForest'],
        '06_CatBoost': grid_details.get('CatBoost', pd.DataFrame([{'模型': 'CatBoost',
                                                              '状态': 'NOT_RUN'}])),
        '07_Boosting': grid_details.get('LightGBM', pd.DataFrame([{'模型': 'LightGBM',
                                                              '状态': 'NOT_RUN'}])),
        '08_Validation比较': comparison_sheet,
        '09_Test最终结果': test_sheet,
        '10_预测残差': pd.concat([residual_sheet, error_group_sheet], ignore_index=True, sort=False),
        '12_技能阈值选择': threshold_table,
        '13_文本维度选择': dimension_table,
    }
    audit_sheets['11_门禁'] = pd.DataFrame([
        {'门禁项': name, '状态': item['status'], '说明': item['note']}
        for name, item in gates.results.items()])
    audit_path = project_paths.TABLES_DIR / project_paths.TABLE_MODEL_COMPARISON
    gates.check('MODEL_COMPARISON_EXPORT',
                len(audit_sheets) >= 11 and not comparison.empty and not test_sheet.empty,
                f'30 号审计表含 {len(audit_sheets)} 张子表（划分 / 特征维度 / 各模型网格 / '
                'validation 比较 / test 最终结果 / 预测残差 / 阈值与维度选择 / 门禁）')
    audit_path = io_utils.write_excel(audit_path, audit_sheets)

    # ---- 图 ----
    style_snapshot = plot_style.setup_sci_style()
    project_paths.MODELING_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_style.SCI_FIGURES_DIR = project_paths.MODELING_FIGURES_DIR
    registry: list = []
    figure_table = comparison_sheet.merge(
        test_sheet[['模型', 'test_MAE']].dropna(subset=['test_MAE']), on='模型', how='left')
    figure_model_comparison(figure_table, registry)
    figure_prediction_scatter(final_frame, final_model_key, registry)
    figure_residuals(final_frame, final_model_key, registry)
    figure_error_groups(error_group_sheet, registry)
    figure_checks = plot_style.check_figure_gates(registry, style_snapshot)

    dummy_mae = model_results['Dummy']['metrics']['MAE']
    metrics_payload = {
        'split_counts': {key: int(value) for key, value in split_counts.items()},
        'split_bins': model_training.SPLIT_BINS, 'random_state': model_training.SPLIT_RANDOM_STATE,
        'splits_path': project_paths.relative_to_root(splits_path),
        'predictions_path': project_paths.relative_to_root(predictions_path),
        'audit_path': project_paths.relative_to_root(audit_path),
        'artifact_dir': project_paths.relative_to_root(artifact_dir),
        'skill_threshold': skill_threshold, 'skill_columns': len(final_assembler.schema.skill_columns),
        'text_dim': best_text_dim, 'final_dimension': final_assembler.schema.dimension,
        'embedding_status': '可用（model-safe BGE 512 维 → 训练集拟合 SVD）',
        'xgboost_status': 'NOT_RUN（环境未安装 xgboost，按约定不强行安装；已使用 LightGBM）',
        'final_model': final_model_key, 'final_params': final_params,
        'final_test_MAE': final_test['MAE'], 'final_test_RMSE': final_test['RMSE'],
        'final_test_R2': final_test['R2'], 'test_rows': int(len(test_frame)),
        'fit_rows_final': int(len(fit_frame)),
        'mae_improvement_vs_dummy': round((dummy_mae - final_test['MAE']) / dummy_mae, 6),
        'test_evaluation_count': test_evaluation_count,
        'comparison': comparison_sheet.to_dict('records'),
        'threshold_selection': threshold_table.to_dict('records'),
        'dimension_selection': dimension_table.to_dict('records'),
        'error_groups': error_group_sheet.to_dict('records'),
        'figure_checks': figure_checks,
        'figures': registry,
        'gates': {name: item['status'] for name, item in gates.results.items()},
    }
    io_utils.write_json(project_paths.METRICS_DIR / f'{STAGE}.json', metrics_payload)

    record_path = write_record(metrics_payload, audit_sheets)
    print(f'阶段记录: {project_paths.relative_to_root(record_path)}')
    quality.print_gates(gates.results, order=quality.STAGE_GATE_MAP[STAGE])
    gates.save()
    return 0


if __name__ == '__main__':
    sys.exit(main())

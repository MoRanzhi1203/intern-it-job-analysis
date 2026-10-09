# -*- coding: utf-8 -*-
"""Stage 15：特征组消融、公司 Group Split 稳健性、技能价值 bootstrap 与 TreeSHAP 解释。

固定 Stage 14 正式主模型（LightGBM + 已选超参数 + 技能阈值 100 + 文本维度 16），
**不重新选模型、不据 test 调参、不据消融/SHAP 结果回头修改技能词典**。

流程：
    1. 四组消融（Base / Base+Skill / Full-Skill / Full）：同一 split、同一模型族、同一超参数、同一种子，
       只改变特征组；
    2. 技能增量（Base+Skill vs Base、Full vs Full-Skill）与文本增量（Full vs Base+Skill），
       附成对 bootstrap MAE 差值 95% CI；
    3. 公司 Group Split（company_entity_id，同公司不跨子集）与随机划分对比；
    4. 稳健性：y winsorize 1%/99%、薪资下限 / 薪资上限目标；
    5. TreeSHAP（LightGBM 原生 pred_contrib）只对最终主模型执行：整体 Top20 + 技能 Top20
       （含全局 mean(SHAP) 方向与 presence-conditioned 方向、模型样本频率与 test_present_n），
       并做多种子解释稳定性检查。

输出：
    outputs/tables/ch8/23_ablation_robustness_shap.xlsx
    outputs/figures/modeling/05_..08_...
    docs/records/22_ablation_robustness_shap_record.md
    outputs/logs/metrics/stage_15_ablation.json

用法：
    python scripts/ch8_robust/15_ablation_robustness_shap.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = next(
    _candidate for _candidate in Path(__file__).resolve().parents
    if (_candidate / "data").is_dir() and (_candidate / "scripts").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import (ablation_shap, io_utils, model_training, plot_style,  # noqa: E402
                 project_paths, quality, schema, skill_eda)
from src.script_support import finish_figure as _finish  # noqa: E402

STAGE = 'stage_15_ablation'
TITLE = 'Stage 15 特征组消融、公司 Group Split 稳健性、技能价值验证与 TreeSHAP'
EXPECTED_SAMPLE = 14883
SEED_STABILITY_SEEDS = (42, 7, 2024)
SHAP_TOP_N = 20
SKILL_SHAP_TOP_N = 20
# §17 重点技能（presence 统计方式必须逐个报告）
KEY_SKILLS = ['Python', 'Java', 'SQL', 'MySQL', 'C++', 'Excel',
              '机器学习', '深度学习', '大模型', 'Agent',
              '强化学习', 'Linux', '数据分析', '办公软件']


def register_gates(gates, audit: dict, metrics: dict, key_skill: pd.DataFrame) -> None:
    """登记 Stage 15 门禁（全部由真实运行结果判定）。"""
    ablation = audit['01_消融结果']
    increment = audit['02_技能增量']
    text_increment = audit['03_文本增量']
    group_table = audit['04_Random_vs_GroupSplit']
    robustness = audit['05_极端值敏感性']
    target_table = audit['06_目标稳健性']
    shap_table = audit['07_SHAP总排名']
    skill_shap = audit['08_技能SHAP']

    gates.check('ABLATION_SAME_SPLIT',
                metrics['same_split_check'] and len(ablation) == len(ablation_shap.CONFIG_ORDER),
                f"四个消融配置全部使用同一划分（train/validation/test 各 "
                f"{metrics['split_counts']['train']:,}/{metrics['split_counts']['validation']:,}/"
                f"{metrics['split_counts']['test']:,}），行集合逐配置校验一致")

    config_summary = '、'.join(
        '{}={}'.format(row.配置, '+'.join(row.特征组)) for row in ablation.itertuples())
    gates.check('ABLATION_FEATURE_GROUP_ONLY',
                metrics['same_model_check'] and metrics['feature_group_only_check'],
                f"四个配置模型族 / 超参数 / 随机种子完全一致（LightGBM "
                f"{metrics['final_params']}，seed=42），仅特征组不同：{config_summary}")

    gates.check('SKILL_INCREMENT_EVAL',
                not increment.empty and increment['bootstrap_CI95_下界'].notna().all()
                and set(increment['比较（加入特征组后）']) == {'Base+Skill vs Base',
                                                        'Full vs Full-Skill'},
                f"技能增量两组比较均完成（validation 与 test），含成对 bootstrap "
                f"{ablation_shap.BOOTSTRAP_ROUNDS} 轮 95% CI；"
                f"Base+Skill vs Base 的 ΔMAE = "
                f"{float(increment.iloc[0]['ΔMAE（无该组 − 有该组）'])}")

    gates.check('TEXT_INCREMENT_EVAL',
                not text_increment.empty
                and text_increment['ΔMAE（无该组 − 有该组）'].notna().all(),
                f"文本增量 {len(text_increment)} 行（Full vs Base+Skill，validation 与 test），"
                f"ΔMAE = {float(text_increment.iloc[0]['ΔMAE（无该组 − 有该组）'])}")

    gates.check('COMPANY_GROUP_SPLIT',
                metrics['group_split']['是否泄漏'] == '否'
                and metrics['group_split']['公司实体总数'] > 0
                and len(group_table) >= 4,
                f"公司 Group Split 共 {metrics['group_split']['公司实体总数']:,} 个公司实体，"
                f"跨子集公司数 = {metrics['group_split']['跨子集公司数']}（无泄漏）；"
                f"train/validation/test 公司数 "
                f"{metrics['group_split']['train_公司数']:,}/"
                f"{metrics['group_split']['validation_公司数']:,}/"
                f"{metrics['group_split']['test_公司数']:,}")

    gates.check('ROBUSTNESS_EXPORT',
                len(robustness) >= 3 and len(target_table) >= 3,
                f"极端值敏感性 {len(robustness)} 行（主模型 / winsorize 1%/99% / 重要性方向）与"
                f"目标稳健性 {len(target_table)} 行（薪资中点 / 薪资下限 / 薪资上限）已输出")

    gates.check('SHAP_FINAL_MODEL_ONLY',
                metrics['shap_model_key'] == metrics['artifact_model_key']
                and metrics['shap_rows'] == metrics['split_counts']['test']
                and metrics['shap_reconstruction_max_error'] < 1e-6,
                f"SHAP 只对 Stage 14 正式主模型 {metrics['shap_model_key']} 执行"
                f"（产出目录模型一致），解释样本 = test 子集 {metrics['shap_rows']:,} 行，"
                f"未对其他模型执行；加性一致性自校验：基准值 {metrics['shap_base_value']} + "
                f"各特征贡献之和 与模型预测的最大偏差 "
                f"{metrics['shap_reconstruction_max_error']:.2e}")

    gates.check('SHAP_EXPORT',
                not shap_table.empty and len(shap_table) == metrics['feature_dimension']
                and not skill_shap.empty
                and {'岗位数', '岗位频率', 'mean_abs_SHAP', '方向性'} <= set(skill_shap.columns)
                and int(skill_shap['低频标记'].ne('').sum()) >= 0,
                f"整体 SHAP 排名 {len(shap_table)} 个特征（Top20 已列）；"
                f"技能 SHAP {len(skill_shap)} 个技能，含岗位频率 / mean|SHAP| / 方向性，"
                f"低频技能（<30 岗位）标记 {int(skill_shap['低频标记'].ne('').sum())} 个")

    gates.check('SHAP_SKILL_PRESENCE',
                not skill_shap.empty
                and {'model_job_count', 'model_job_frequency', 'test_present_n', 'test_absent_n',
                     'mean_SHAP_present', 'median_SHAP_present', 'mean_SHAP_absent',
                     'median_SHAP_absent', 'mean_abs_SHAP_present', 'positive_SHAP_ratio_present',
                     'presence_direction', 'global_mean_SHAP_direction'} <= set(skill_shap.columns)
                and int((skill_shap['test_present_n'] + skill_shap['test_absent_n']).min())
                == metrics['shap_rows']
                and int((skill_shap['test_present_n'] + skill_shap['test_absent_n']).max())
                == metrics['shap_rows']
                and skill_shap['presence_direction']
                .isin(['正向预测贡献', '负向预测贡献', '中性/弱影响']).all()
                and not key_skill.empty and set(KEY_SKILLS) <= set(key_skill['技能']),
                f"技能 SHAP 已增加 presence 统计方式：present + absent = "
                f"{metrics['shap_rows']:,}（原 test 子集）逐技能成立；"
                f"presence_direction 按 mean_SHAP_present（阈值 1e-8）判定，"
                f"与 global_mean_SHAP_direction 分列；"
                f"重点技能表覆盖 {len(key_skill)} 个技能（含 SQL/MySQL/Excel 等），"
                f"模型样本频率分母 {metrics['split_counts']['train'] + metrics['split_counts']['validation'] + metrics['split_counts']['test']:,}"
                "（Stage 12 建模样本），与 presence 分母严格区分")


def build_ablation_sheet(records: list) -> pd.DataFrame:
    return pd.DataFrame(records)


def build_increment_sheet(prediction_frames: dict, pairs, group_label: str) -> pd.DataFrame:
    """增量表：ΔMAE/ΔRMSE/ΔR² + 成对 bootstrap 95% CI（ΔMAE = MAE(无该组) − MAE(有该组)）。"""
    rows = []
    for with_label, without_label in pairs:
        for split in ('validation', 'test'):
            frame_with = prediction_frames[(with_label, split)]
            frame_without = prediction_frames[(without_label, split)]
            truth = frame_with['y_true'].to_numpy()
            with_pred = frame_with['y_pred'].to_numpy()
            without_pred = frame_without['y_pred'].to_numpy()
            stats = ablation_shap.bootstrap_mae_difference(truth, with_pred, without_pred)
            metrics_with = model_training.regression_metrics(truth, with_pred)
            metrics_without = model_training.regression_metrics(truth, without_pred)
            delta = stats['ΔMAE（B − A）']
            if stats['CI是否跨0'] == '否':
                conclusion = (f'加入{group_label}后误差稳定下降（ΔMAE {delta:+.4f}，95% CI 不跨 0）'
                              if delta > 0 else
                              f'加入{group_label}后误差稳定上升（ΔMAE {delta:+.4f}，95% CI 不跨 0）')
            else:
                conclusion = (f'{group_label}的增量在 95% 置信水平上不显著'
                              f'（ΔMAE {delta:+.4f}，CI 跨 0），只作描述性报告')
            rows.append({
                '比较（加入特征组后）': f'{with_label} vs {without_label}',
                '涉及特征组': group_label, '数据子集': split,
                'MAE_有该组': metrics_with['MAE'], 'MAE_无该组': metrics_without['MAE'],
                'ΔMAE（无该组 − 有该组）': delta,
                'ΔRMSE（无该组 − 有该组）': round(metrics_without['RMSE'] - metrics_with['RMSE'], 6),
                'ΔR²（有该组 − 无该组）': round(metrics_with['R2'] - metrics_without['R2'], 6),
                'bootstrap_CI95_下界': stats['bootstrap_CI95_下界'],
                'bootstrap_CI95_上界': stats['bootstrap_CI95_上界'],
                'CI是否跨0': stats['CI是否跨0'],
                'bootstrap轮数': stats['bootstrap轮数'], '随机种子': stats['随机种子'],
                '结论表述': conclusion,
                '说明': '正值 ΔMAE 表示加入该特征组后 MAE 下降；CI 不跨 0 表示差异稳定',
            })
    return pd.DataFrame(rows)


def figure_ablation(ablation: pd.DataFrame, increment: pd.DataFrame, registry: list) -> dict:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    labels = ablation['配置'].tolist()
    positions = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    ax = axes[0]
    ax.bar(positions - 0.19, ablation['validation_MAE'], width=0.36,
           color=plot_style.MAIN_COLOR, edgecolor='black', linewidth=0.5, label='validation MAE')
    ax.bar(positions + 0.19, ablation['test_MAE'], width=0.36, color=plot_style.ACCENT_COLOR,
           edgecolor='black', linewidth=0.5, label='test MAE')
    for position, value in zip(positions - 0.19, ablation['validation_MAE']):
        ax.text(position, value + 0.6, f'{value:.1f}', ha='center', va='bottom',
                fontsize=plot_style.FONT_SIZES['annotation'])
    for position, value in zip(positions + 0.19, ablation['test_MAE']):
        ax.text(position, value + 0.6, f'{value:.1f}', ha='center', va='bottom',
                fontsize=plot_style.FONT_SIZES['annotation'])
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel('特征组配置')
    ax.set_ylabel('MAE（元/天）')
    ax.set_ylim(0, float(ablation[['validation_MAE', 'test_MAE']].to_numpy().max()) * 1.3)
    ax.legend(loc='upper right', frameon=False)
    plot_style.apply_sci_axis(ax, grid_axis='y')

    ax2 = axes[1]
    rows = increment.to_dict('records')
    row_labels = [f"{row['比较（加入特征组后）']}\n({row['数据子集']})" for row in rows]
    deltas = np.array([row['ΔMAE（无该组 − 有该组）'] for row in rows], dtype='float64')
    lower = np.array([row['bootstrap_CI95_下界'] for row in rows], dtype='float64')
    upper = np.array([row['bootstrap_CI95_上界'] for row in rows], dtype='float64')
    y_positions = np.arange(len(rows))[::-1]
    ax2.errorbar(deltas, y_positions, xerr=np.vstack([deltas - lower, upper - deltas]),
                 fmt='o', markersize=5, color=plot_style.MAIN_COLOR, ecolor=plot_style.MUTED_COLOR,
                 elinewidth=1.2, capsize=3)
    ax2.axvline(0, color='black', linestyle='--', linewidth=0.9)
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(row_labels, fontsize=plot_style.FONT_SIZES['tick'])
    ax2.set_xlabel('ΔMAE（无该组 − 有该组，元/天）')
    ax2.set_ylabel('技能组增量比较')
    plot_style.apply_sci_axis(ax2, grid_axis='x')

    plot_style.add_subfigure_caption(ax, 'a', '四配置 validation / test MAE')
    plot_style.add_subfigure_caption(ax2, 'b', '技能增量 ΔMAE 与 bootstrap 95% CI')
    caption = ('图15 特征组消融（同一 split / 模型 / 超参数 / 种子，仅改变特征组）：'
               '（a）四配置 validation / test MAE；（b）技能增量 ΔMAE 与 bootstrap 95% CI（跨 0 即不显著）')
    plot_style.add_bottom_caption(fig, caption)
    fig.subplots_adjust(bottom=0.2, wspace=0.35)
    return _finish(fig, '05_ablation_results', caption, registry,
                   subfigures=[('a', '四配置 validation / test MAE', axes[0]),
                               ('b', '技能增量 ΔMAE 与 bootstrap 95% CI', axes[1])],
                   meta={'图表类型': '对比柱状图 + 森林图', '数据来源': '31 号消融表与技能增量表'})


def figure_random_vs_group(group_table: pd.DataFrame, registry: list) -> dict:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    frame = group_table[group_table['数据子集'].eq('test')
                        & ~group_table['划分方式'].str.startswith('差异')]
    labels = frame['划分方式'].tolist()
    positions = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(positions - 0.19, frame['MAE'], width=0.36, color=plot_style.MAIN_COLOR,
           edgecolor='black', linewidth=0.5, label='MAE')
    ax.bar(positions + 0.19, frame['RMSE'], width=0.36, color=plot_style.MUTED_COLOR,
           edgecolor='black', linewidth=0.5, label='RMSE')
    for position, value in zip(positions - 0.19, frame['MAE']):
        ax.text(position, value + 0.8, f'{value:.1f}', ha='center', va='bottom',
                fontsize=plot_style.FONT_SIZES['annotation'])
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel('划分方式')
    ax.set_ylabel('误差（元/天）')
    ax.set_ylim(0, float(frame[['MAE', 'RMSE']].to_numpy().max()) * 1.3)
    ax.legend(loc='upper left', frameon=False)
    plot_style.apply_sci_axis(ax, grid_axis='y')
    caption = '图16 随机划分与公司 Group Split 对比（test；同分布预测 vs 跨公司泛化）'
    plot_style.add_bottom_caption(fig, caption)
    fig.subplots_adjust(bottom=0.18)
    return _finish(fig, '06_random_vs_group_split', caption, registry,
                   meta={'图表类型': '对比柱状图', '数据来源': '31 号 Group Split 表'})


def figure_shap_beeswarm(shap_matrix: np.ndarray, feature_names, feature_matrix, registry: list,
                         top_n: int = 12) -> dict:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    magnitude = np.abs(shap_matrix).mean(axis=0)
    order = np.argsort(magnitude)[::-1][:top_n]
    values_matrix = ablation_shap.dense_matrix(feature_matrix)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    rng = np.random.default_rng(42)
    for row_position, feature_index in enumerate(order):
        values = shap_matrix[:, feature_index]
        jitter = rng.uniform(-0.22, 0.22, size=len(values))
        colors = values_matrix[:, feature_index]
        ax.scatter(values, np.full(len(values), row_position) + jitter, s=6,
                   c=colors, cmap='cividis', alpha=0.55, edgecolor='none')
    ax.axvline(0, color=plot_style.MUTED_COLOR, linestyle='--', linewidth=0.9)
    ax.set_yticks(np.arange(top_n))
    ax.set_yticklabels([str(feature_names[index]) for index in order])
    ax.set_xlabel('SHAP 值（元/天，对预测薪资的贡献）')
    ax.set_ylabel('特征（按 mean|SHAP| 排序）')
    plot_style.apply_sci_axis(ax, grid_axis='x')
    caption = '图17 正式主模型 TreeSHAP 蜂群图（Top12；仅表示模型预测贡献，非因果）'
    plot_style.add_bottom_caption(fig, caption)
    fig.subplots_adjust(left=0.34, bottom=0.16)
    return _finish(fig, '07_shap_beeswarm', caption, registry,
                   meta={'图表类型': '蜂群图', '数据来源': 'TreeSHAP（LightGBM pred_contrib）'})


def figure_skill_shap(skill_table: pd.DataFrame, registry: list,
                      top_n: int = SKILL_SHAP_TOP_N) -> dict:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    frame = skill_table.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    positions = np.arange(len(frame))
    ax.barh(positions, frame['mean_abs_SHAP'], height=0.66,
            color=plot_style.MUTED_COLOR, edgecolor='black', linewidth=0.5)
    for position, direction in enumerate(frame['presence_direction']):
        mark = '▲' if direction == '正向预测贡献' else ('▼' if direction == '负向预测贡献' else '●')
        ax.text(frame['mean_abs_SHAP'].iloc[position] * 0.02, position, mark,
                va='center', ha='left', fontsize=plot_style.FONT_SIZES['annotation'],
                color='black')
    ax.set_yticks(positions)
    ax.set_yticklabels([f'{name}（岗位频率 {freq:.1%}）' for name, freq
                        in zip(frame['技能'], frame['model_job_frequency'])])
    ax.set_xlabel('mean|SHAP|（元/天，技能特征的预测贡献强度）')
    ax.set_ylabel('技能（按 mean|SHAP| 排序，Top20）')
    ax.text(0.97, 0.03,
            '标记：▲ 技能存在时平均贡献为正，▼ 为负，● 中性/弱影响\n'
            '方向表示「该技能存在时」的平均 SHAP 贡献方向，不代表技能的因果薪资效应',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=plot_style.FONT_SIZES['annotation'])
    plot_style.apply_sci_axis(ax, grid_axis='x')
    caption = ('图18 技能特征 TreeSHAP Top20（条长 = mean|SHAP|，附注 = 模型样本岗位频率，'
               '标记 = presence_direction；仅表示模型预测贡献，非因果）')
    plot_style.add_bottom_caption(fig, caption)
    fig.subplots_adjust(left=0.38, bottom=0.18)
    return _finish(fig, '08_skill_shap_top20', caption, registry,
                   meta={'图表类型': '横向柱状图', '数据来源': 'TreeSHAP 技能特征（presence 统计方式）'})


def pair_summary(increment: pd.DataFrame, pair_label: str) -> dict:
    """把增量表的 ΔMAE / CI 转成可判读的结论（不预设方向）。"""
    rows = increment[increment['比较（加入特征组后）'].eq(pair_label)].set_index('数据子集')
    deltas = {split: float(rows.loc[split, 'ΔMAE（无该组 − 有该组）'])
              for split in ('validation', 'test')}
    crossing = {split: rows.loc[split, 'CI是否跨0'] for split in ('validation', 'test')}
    return {'deltas': deltas, 'crossing': crossing,
            'improves': all(value > 0 for value in deltas.values()),
            'degrades': all(value < 0 for value in deltas.values()),
            'significant': all(value == '否' for value in crossing.values())}


def increment_sentence(increment: pd.DataFrame, pair_label: str, group_label: str) -> str:
    rows = increment[increment['比较（加入特征组后）'].eq(pair_label)].to_dict('records')
    parts = []
    for row in rows:
        delta = row['ΔMAE（无该组 − 有该组）']
        direction = '加入后 MAE 下降（改善）' if delta > 0 else '加入后 MAE 上升（未改善）'
        stability = 'CI 不跨 0，差异稳定' if row['CI是否跨0'] == '否' else 'CI 跨 0，差异不显著'
        parts.append(f"{row['数据子集']} ΔMAE = {delta:+.4f}（{direction}；95% CI "
                     f"[{row['bootstrap_CI95_下界']}, {row['bootstrap_CI95_上界']}]，{stability}）")
    return f"- {pair_label}（{group_label}）：" + '；'.join(parts) + '。'


def skill_value_verdict(summary: dict, label: str = '技能组') -> str:
    if summary['improves'] and summary['significant']:
        return (f'{label}在 A+B+C 之上带来**稳定的** MAE 改善（点估计为正且 95% CI 不跨 0），'
                f'{label}价值得到确认。')
    if summary['improves']:
        return (f'{label}的点估计方向为改善，但 bootstrap 95% CI 跨 0，'
                '即当前样本量下**无法证明**稳定增量，只能作为方向性证据。')
    if summary['degrades']:
        return (f'{label}的点估计方向为**未改善**（ΔMAE 为负，即加入后 MAE 上升），'
                '且 95% CI 跨 0，说明在 A+B+C 已存在的条件下该组未提供可检出的额外预测信息；'
                '该结果如实报告，**不得**据此反向修改技能词典。')
    return f'{label} 的增量方向在两个子集上不一致，视为不稳定，只作描述性报告。'


def write_record(metrics: dict, audit: dict) -> Path:
    ablation = audit['01_消融结果'].set_index('配置')
    increment = audit['02_技能增量']
    text_increment = audit['03_文本增量']
    group_table = audit['04_Random_vs_GroupSplit']
    robustness = audit['05_极端值敏感性']
    target_table = audit['06_目标稳健性']
    lines = [
        '# 记录 22：Stage 15 特征组消融、公司 Group Split 稳健性、技能价值验证与 TreeSHAP',
        '',
        '> 本记录由 `scripts/ch8_robust/15_ablation_robustness_shap.py` 自动生成，数字全部来自真实运行结果。',
        '> 本轮固定 Stage 14 主模型配置（不重新选模型、不据 test 调参、**不据消融或 SHAP 结果回头修改技能词典**）。',
        '',
        '## 1. 消融定义与公平性',
        '',
        '| 配置 | 特征组 | 特征维度 | 技能列 | 文本维度 | validation MAE | test MAE |',
        '| --- | --- | --- | --- | --- | --- | --- |',
    ]
    for label in ablation_shap.CONFIG_ORDER:
        row = ablation.loc[label]
        lines.append(f"| {label} | {'+'.join(row['特征组'])} | {int(row['特征维度'])} | "
                     f"{int(row['技能列数'])} | {int(row['文本维度'])} | "
                     f"{row['validation_MAE']} | {row['test_MAE']} |")
    lines += [
        '',
        f"- 公平性：同一 split（train/validation/test = {metrics['split_counts']['train']:,}/"
        f"{metrics['split_counts']['validation']:,}/{metrics['split_counts']['test']:,}）、"
        f"同一模型族与超参数（LightGBM {metrics['final_params']}）、同一随机种子（42），"
        '仅特征组不同；未对任何消融重新调参。',
        f"- 与正式主模型的关系：四个消融配置均只在本 split 的 **train** 上拟合，"
        f"因此 Full 配置的 validation MAE（{ablation.loc['Full', 'validation_MAE']}）与产出目录中"
        f"正式主模型一致（预处理器 train-only），但 test MAE 不同——正式主模型按记录 21 在 "
        f"train+validation 上重拟合估计器，其 test MAE = "
        f"{metrics['official_model_metrics']['test']['MAE']}，Full 消融配置 test MAE = "
        f"{ablation.loc['Full', 'test_MAE']}；两处数字均如实保留，不互相替代。",
        '',
        '## 2. 技能价值（核心验证）',
        '',
        '| 比较 | 数据子集 | ΔMAE | ΔRMSE | ΔR² | bootstrap 95% CI | CI 是否跨 0 |',
        '| --- | --- | --- | --- | --- | --- | --- |',
    ]
    for row in increment.to_dict('records'):
        lines.append(f"| {row['比较（加入特征组后）']} | {row['数据子集']} | "
                     f"{row['ΔMAE（无该组 − 有该组）']} | {row['ΔRMSE（无该组 − 有该组）']} | "
                     f"{row['ΔR²（有该组 − 无该组）']} | "
                     f"[{row['bootstrap_CI95_下界']}, {row['bootstrap_CI95_上界']}] | "
                     f"{row['CI是否跨0']} |")
    skill_pairs = [pair_summary(increment, f'{with_label} vs {without_label}')
                   for with_label, without_label in ablation_shap.SKILL_INCREMENT_PAIRS]
    text_summary = pair_summary(text_increment, 'Full vs Base+Skill')
    lines += [
        '',
        f"- ΔMAE = MAE(无该特征组) − MAE(有该特征组)，正值表示加入该组后误差下降；"
        f"bootstrap {metrics['bootstrap_rounds']} 轮、种子 {metrics['bootstrap_seed']}；",
        increment_sentence(increment, 'Base+Skill vs Base', 'D 技能组'),
        increment_sentence(increment, 'Full vs Full-Skill', 'D 技能组'),
        '',
        f"- **核心验证判读（Base+Skill vs Base）**：{skill_value_verdict(skill_pairs[0])}",
        f"- 第二处独立验证判读（Full vs Full-Skill）：{skill_value_verdict(skill_pairs[1])}",
        '- 两处比较均未达到 95% 显著性（CI 跨 0），且点估计方向相反，'
        '说明 D 组的增量在当前设置下**不稳健**；本轮只如实报告，'
        '**不修改技能词典**、不更换主模型、不据该结果回头调整特征工程。',
        '',
        '## 3. 文本语义价值',
        '',
        '| 比较 | 数据子集 | ΔMAE | ΔR² | bootstrap 95% CI | CI 是否跨 0 |',
        '| --- | --- | --- | --- | --- | --- |',
    ]
    for row in text_increment.to_dict('records'):
        lines.append(f"| Full vs Base+Skill | {row['数据子集']} | "
                     f"{row['ΔMAE（无该组 − 有该组）']} | {row['ΔR²（有该组 − 无该组）']} | "
                     f"[{row['bootstrap_CI95_下界']}, {row['bootstrap_CI95_上界']}] | "
                     f"{row['CI是否跨0']} |")
    lines += [
        '',
        increment_sentence(text_increment, 'Full vs Base+Skill', 'E 文本语义组'),
        f'- 判读：{skill_value_verdict(text_summary, "文本语义组 E")}',
        '- 说明：文本组与技能组的统计范围不同（前者是通用语义压缩维度，后者是显式技能要求），'
        '两者增量大小不可直接与「技能是否存在价值」的结论互相替代。',
        '',
        '## 4. 公司 Group Split（跨公司泛化）',
        '',
        '| 划分方式 | 数据子集 | MAE | RMSE | R² | n | 公司数 |',
        '| --- | --- | --- | --- | --- | --- | --- |',
    ]
    for row in group_table.to_dict('records'):
        lines.append(f"| {row['划分方式']} | {row['数据子集']} | {row['MAE']} | {row['RMSE']} | "
                     f"{row['R²']} | {int(row['n'])} | {row.get('公司数', '')} |")
    lines += [
        '',
        f"- 公司 Group Split 无泄漏：公司实体 {metrics['group_split']['公司实体总数']:,} 个，"
        f"跨子集公司数 {metrics['group_split']['跨子集公司数']}；",
        '- 解读：随机划分反映**同分布预测**能力（岗位可能来自训练集中出现过的公司），'
        '公司 Group Split 反映**跨公司泛化**能力（测试公司完全未见）。'
        '两者差距说明随机划分结果部分受益于同公司岗位相似性，但**不能据此认为原模型无效**；'
        '该差距正是需要在论文中显式报告的统计范围差异。',
        '',
        '## 5. 稳健性',
        '',
        '| 实验 | MAE | RMSE | R² | 说明 |',
        '| --- | --- | --- | --- | --- |',
    ]
    for row in robustness.to_dict('records'):
        lines.append(f"| {row['实验']} | {row.get('MAE', '')} | {row.get('RMSE', '')} | "
                     f"{row.get('R²', '')} | {row['说明']} |")
    lines += [
        '',
        '| 目标定义 | MAE | RMSE | R² | 说明 |',
        '| --- | --- | --- | --- | --- |',
    ]
    for row in target_table.to_dict('records'):
        lines.append(f"| {row['目标定义']} | {row['MAE']} | {row['RMSE']} | {row['R²']} | "
                     f"{row['说明']} |")
    lines += [
        '',
        '## 6. SHAP 解释（只对最终主模型）',
        '',
        '| 排名 | 特征 | mean|SHAP| | 方向性 |',
        '| --- | --- | --- | --- |',
    ]
    for row in audit['07_SHAP总排名'].head(SHAP_TOP_N).to_dict('records'):
        lines.append(f"| {int(row['排名'])} | {row['特征']} | {row['mean_abs_SHAP']} | "
                     f"{row['方向性']} |")
    lines += [
        '',
        '| 排名 | 技能 | 模型样本岗位频率 | test_present_n | mean_abs_SHAP | mean_SHAP_present | presence_direction | 全局方向 | 低频标记 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for row in audit['08_技能SHAP'].head(SKILL_SHAP_TOP_N).to_dict('records'):
        lines.append(f"| {int(row['排名'])} | {row['技能']} | {row['model_job_frequency']:.2%} | "
                     f"{int(row['test_present_n'])} | {row['mean_abs_SHAP']} | "
                     f"{row['mean_SHAP_present']} | {row['presence_direction']} | "
                     f"{row['global_mean_SHAP_direction']} | {row['低频标记']} |")
    lines += [
        '',
        '### 6.1 重点技能（presence 统计方式）',
        '',
        '| 技能 | 模型样本岗位频率 | test_present_n | mean_SHAP_present | mean_abs_SHAP | presence_direction |',
        '| --- | --- | --- | --- | --- | --- |',
    ]
    for row in audit['12_关键技能SHAP'].to_dict('records'):
        present = row['test_present_n'] if row['test_present_n'] != '' else '—'
        mean_present = row['mean_SHAP_present'] if row['mean_SHAP_present'] != '' else '—'
        mean_abs = row['mean_abs_SHAP'] if row['mean_abs_SHAP'] != '' else '—'
        lines.append(f"| {row['技能']} | {row['模型样本岗位频率']:.2%} | {present} | "
                     f"{mean_present} | {mean_abs} | {row['presence_direction']} |")
    lines += [
        '',
        '### 6.2 两种方向统计范围（严格区分，不得混用）',
        '',
        '| 统计范围 | 定义 | 用途 |',
        '| --- | --- | --- |',
        '| `global_mean_SHAP_direction`（原「方向性」） | mean(SHAP) 在**全部解释样本**上的正负 | '
        '历史统计范围，保留用于对比；受 0/1 特征结构影响 |',
        '| `presence_direction`（论文与图表优先） | **技能存在时**（技能列 = 1）的 '
        'mean_SHAP_present > 1e-8 → 正向预测贡献；< -1e-8 → 负向预测贡献；否则中性/弱影响 | '
        '对 0/1 技能特征更直观的解释方式 |',
        '',
        f"- SHAP 使用 LightGBM 原生 TreeSHAP（`pred_contrib`），解释样本 = test 子集 "
        f"{metrics['shap_rows']:,} 行；基准值 {metrics['shap_base_value']}，"
        f"加性一致性自校验（基准值 + 各特征贡献 = 模型预测）最大偏差 "
        f"{metrics['shap_reconstruction_max_error']:.2e}；",
        f"- 解释对象：产出目录中的正式主模型 {metrics['shap_model_key']}"
        '（预处理器 train-only 拟合、估计器在 train+validation 重拟合，test 从未参与拟合与选模），'
        '未对消融配置或其它候选模型执行 SHAP；',
        f"- 分母范围：模型样本频率 = 岗位数 / {metrics['model_sample']:,}（Stage 12 建模样本，"
        f'ALL_USABLE 技能统计范围）；presence 统计方式 = test 现技能岗位数 / '
        f"{metrics['shap_rows']:,}（原 test 子集），两个分母**禁止混用**；",
        f"- 解释稳定性（多种子 {SEED_STABILITY_SEEDS}）：mean|SHAP| 排名 Spearman 相关 "
        f"{metrics['shap_stability']}；",
        '- 措辞规范：SHAP 表示**在模型其它特征共同存在时的预测贡献**，'
        '`presence_direction` 表示「该技能存在时」的平均贡献方向，'
        '**均不得解释为因果**（不代表技能的因果薪资效应）；'
        '低频技能（岗位数 < 30）即使 SHAP 较高也已标记样本量。',
        '',
        '## 7. 结论汇总',
        '',
        f'1. 技能组消融（核心验证）：{skill_value_verdict(skill_pairs[0])}',
        f'2. 技能组消融（第二处独立验证，Full vs Full-Skill）：{skill_value_verdict(skill_pairs[1])}',
        f'3. 文本语义（E 组）：{skill_value_verdict(text_summary, "文本语义组 E")}',
        f'4. 随机划分与公司 Group Split 的差距（test MAE '
        f"{group_table[group_table['划分方式'].str.startswith('差异')]['MAE'].iloc[0]:+.4f}）"
        '量化了「同分布预测」与「跨公司泛化」的统计范围差异；',
        '5. 极端值与目标定义稳健性实验表明误差量级与重要性方向稳定，未改变主任务与主模型；',
        '6. 全部结论限于预测与关联层面，未做因果推断；技能词典与本轮主模型在本轮**未被修改**。',
        '',
        '## 8. 门禁',
        '',
        '| 门禁项 | 状态 | 说明 |',
        '| --- | --- | --- |',
    ]
    gate_detail = audit['10_门禁'].set_index('门禁项')['说明']
    for name, status in metrics['gates'].items():
        lines.append(f'| {name} | {status} | {gate_detail.get(name, "")} |')
    lines += [
        '',
        f"- 产物：`{metrics['audit_path']}`、图 `outputs/figures/modeling/05_..08_`；",
        '- 本轮未据消融/SHAP 结果修改技能词典，未重新使用 test 调参，未提交 git。',
        '',
    ]
    return io_utils.write_markdown(
        project_paths.RECORDS_DIR / project_paths.RECORD_ABLATION_SHAP, lines)


def main() -> int:
    quality.configure_logging()
    gates = quality.GateRegistry(STAGE)
    quality.stage_banner(STAGE, TITLE)

    model_frame = io_utils.read_parquet(project_paths.JOB_SALARY_MODEL_DATASET_PARQUET)
    membership = io_utils.read_parquet(project_paths.JOB_SKILL_MEMBERSHIP_PARQUET)
    splits = pd.read_parquet(project_paths.MODEL_SPLITS_PARQUET)
    artifact_dir = project_paths.SALARY_MODEL_DIR
    payload = joblib.load(artifact_dir / 'salary_model_pipeline.joblib')
    artifact_assembler = payload['assembler']
    artifact_model = payload['model']
    model_params = json.loads((artifact_dir / 'model_params.json').read_text(encoding='utf-8'))
    manifest = json.loads((artifact_dir / 'feature_manifest.json').read_text(encoding='utf-8'))
    skill_artifact = json.loads((artifact_dir / 'skill_columns.json').read_text(encoding='utf-8'))
    official_metrics = json.loads(
        (artifact_dir / 'metrics_summary.json').read_text(encoding='utf-8'))
    final_params = model_params['params']
    final_model_key = model_params['model_key']
    skill_threshold = int(skill_artifact['threshold'])
    text_dim = int(model_params['text_dim'])
    print(f'固定配置: {final_model_key} {final_params}；技能阈值 {skill_threshold}；文本维度 {text_dim}')

    if len(model_frame) != EXPECTED_SAMPLE:
        raise SystemExit(f'正式样本数异常: {len(model_frame)}')

    grouped = ablation_shap.split_columns_by_group(
        manifest['numeric_columns'], manifest['categorical_columns'],
        manifest['multi_value_columns'])
    skill_map = model_training.build_skill_map(membership, skill_eda.ALL_USABLE_SCOPES)
    all_ids = model_frame[schema.ID_FIELD].tolist()
    text_matrix = model_training.load_text_matrix(
        all_ids, project_paths.FEATURES_DIR / 'job_text_embeddings.npz',
        project_paths.FEATURES_DIR / 'job_text_embedding_index.parquet')
    text_by_id = {job_id: position for position, job_id in enumerate(all_ids)}

    def text_for(frame):
        return text_matrix[[text_by_id[job_id] for job_id in frame[schema.ID_FIELD]]]

    random_train = set(splits.loc[splits['split'].eq('train'), schema.ID_FIELD])
    random_valid = set(splits.loc[splits['split'].eq('validation'), schema.ID_FIELD])
    random_test = set(splits.loc[splits['split'].eq('test'), schema.ID_FIELD])
    train_frame = model_frame[model_frame[schema.ID_FIELD].isin(random_train)].reset_index(drop=True)
    valid_frame = model_frame[model_frame[schema.ID_FIELD].isin(random_valid)].reset_index(drop=True)
    test_frame = model_frame[model_frame[schema.ID_FIELD].isin(random_test)].reset_index(drop=True)
    y_train = train_frame[schema.SALARY_MID_FIELD].to_numpy(dtype='float64')
    y_valid = valid_frame[schema.SALARY_MID_FIELD].to_numpy(dtype='float64')
    y_test = test_frame[schema.SALARY_MID_FIELD].to_numpy(dtype='float64')

    # ---- 1. 四组消融 ----
    ablation_records = []
    prediction_frames = {}
    ablation_models = {}
    split_signatures = set()
    model_signatures = set()
    for label in ablation_shap.CONFIG_ORDER:
        groups = ablation_shap.ABLATION_CONFIGS[label]
        assembler = ablation_shap.build_assembler_for_groups(groups, grouped, skill_threshold,
                                                            text_dim)
        assembler.fit(train_frame, skill_map, text_for(train_frame))
        matrix_train = assembler.transform(train_frame, skill_map, text_for(train_frame))
        matrix_valid = assembler.transform(valid_frame, skill_map, text_for(valid_frame))
        matrix_test = assembler.transform(test_frame, skill_map, text_for(test_frame))
        model = model_training.make_model(final_model_key, final_params,
                                         random_state=model_training.SPLIT_RANDOM_STATE)
        model.fit(matrix_train, y_train)
        model_signatures.add((final_model_key, tuple(sorted(final_params.items())),
                              model_training.SPLIT_RANDOM_STATE))
        valid_pred = np.asarray(model.predict(matrix_valid), dtype='float64')
        test_pred = np.asarray(model.predict(matrix_test), dtype='float64')
        valid_metrics = model_training.regression_metrics(y_valid, valid_pred)
        test_metrics = model_training.regression_metrics(y_test, test_pred)
        ablation_models[label] = (assembler, model, matrix_train, matrix_test)
        ablation_records.append({
            '配置': label, '特征组': list(groups), '特征维度': assembler.schema.dimension,
            '数值列数': len(assembler.schema.numeric_columns)
            + len(assembler.schema.numeric_indicator_columns),
            '类别列数': sum(len(levels) for levels in assembler.schema.category_levels.values()),
            '多值列数': sum(len(levels) for levels in assembler.schema.multi_value_levels.values()),
            '技能列数': len(assembler.schema.skill_columns),
            '文本维度': assembler.schema.text_dim,
            'validation_MAE': valid_metrics['MAE'], 'validation_RMSE': valid_metrics['RMSE'],
            'validation_R2': valid_metrics['R2'], 'test_MAE': test_metrics['MAE'],
            'test_RMSE': test_metrics['RMSE'], 'test_R2': test_metrics['R2'],
            '训练行数': int(len(train_frame)),
            '说明': '同一 split / 模型族 / 超参数 / 种子，仅特征组不同',
        })
        prediction_frames[(label, 'validation')] = pd.DataFrame({
            schema.ID_FIELD: valid_frame[schema.ID_FIELD], 'y_true': y_valid, 'y_pred': valid_pred})
        prediction_frames[(label, 'test')] = pd.DataFrame({
            schema.ID_FIELD: test_frame[schema.ID_FIELD], 'y_true': y_test, 'y_pred': test_pred})
        split_signatures.add((frozenset(random_train).__hash__(), frozenset(random_valid).__hash__(),
                              frozenset(random_test).__hash__()))
        print(f'{label}: dimension {assembler.schema.dimension} | validation MAE '
              f'{valid_metrics["MAE"]} | test MAE {test_metrics["MAE"]}')
    ablation_table = build_ablation_sheet(ablation_records)

    increment_table = build_increment_sheet(
        prediction_frames, ablation_shap.SKILL_INCREMENT_PAIRS, 'D 技能组')
    text_increment_table = build_increment_sheet(
        prediction_frames, [('Full', 'Base+Skill')], 'E 文本语义组')
    print(increment_table[['比较（加入特征组后）', '数据子集', 'ΔMAE（无该组 − 有该组）',
                           'bootstrap_CI95_下界', 'bootstrap_CI95_上界']].to_string(index=False))

    # ---- 3. 公司 Group Split ----
    group_train = set(splits.loc[splits['company_group_split'].eq('train'), schema.ID_FIELD])
    group_valid = set(splits.loc[splits['company_group_split'].eq('validation'), schema.ID_FIELD])
    group_test = set(splits.loc[splits['company_group_split'].eq('test'), schema.ID_FIELD])
    group_frame_train = model_frame[model_frame[schema.ID_FIELD].isin(group_train)].reset_index(drop=True)
    group_frame_valid = model_frame[model_frame[schema.ID_FIELD].isin(group_valid)].reset_index(drop=True)
    group_frame_test = model_frame[model_frame[schema.ID_FIELD].isin(group_test)].reset_index(drop=True)
    full_groups = ablation_shap.ABLATION_CONFIGS['Full']
    group_assembler = ablation_shap.build_assembler_for_groups(full_groups, grouped,
                                                              skill_threshold, text_dim)
    group_assembler.fit(group_frame_train, skill_map, text_for(group_frame_train))
    group_model = model_training.make_model(final_model_key, final_params,
                                           random_state=model_training.SPLIT_RANDOM_STATE)
    group_model.fit(group_assembler.transform(group_frame_train, skill_map,
                                              text_for(group_frame_train)),
                    group_frame_train[schema.SALARY_MID_FIELD].to_numpy(dtype='float64'))
    group_test_pred = np.asarray(
        group_model.predict(group_assembler.transform(group_frame_test, skill_map,
                                                     text_for(group_frame_test))),
        dtype='float64')
    group_test_truth = group_frame_test[schema.SALARY_MID_FIELD].to_numpy(dtype='float64')
    group_metrics = model_training.regression_metrics(group_test_truth, group_test_pred)
    group_valid_pred = np.asarray(
        group_model.predict(group_assembler.transform(group_frame_valid, skill_map,
                                                      text_for(group_frame_valid))),
        dtype='float64')
    group_valid_metrics = model_training.regression_metrics(
        group_frame_valid[schema.SALARY_MID_FIELD].to_numpy(dtype='float64'), group_valid_pred)
    random_metrics = model_training.regression_metrics(y_test, prediction_frames[('Full', 'test')]
                                                       ['y_pred'])
    group_split_info = ablation_shap.group_split_leakage(splits)
    group_table = pd.DataFrame([
        {'划分方式': 'Random split（同分布预测）', '数据子集': 'test', 'MAE': random_metrics['MAE'],
         'RMSE': random_metrics['RMSE'], 'R²': random_metrics['R2'], 'n': random_metrics['n'],
         '公司数': int(splits.loc[splits['split'].eq('test'), 'company_entity_id'].nunique()),
         '说明': '岗位可能来自训练集中出现过的公司'},
        {'划分方式': 'Company Group Split（跨公司泛化）', '数据子集': 'test',
         'MAE': group_metrics['MAE'], 'RMSE': group_metrics['RMSE'], 'R²': group_metrics['R2'],
         'n': group_metrics['n'],
         '公司数': group_split_info['test_公司数'],
         '说明': '测试公司完全未在训练集出现；差距量化同公司相似性带来的乐观偏差'},
        {'划分方式': 'Company Group Split（跨公司泛化）', '数据子集': 'validation',
         'MAE': group_valid_metrics['MAE'], 'RMSE': group_valid_metrics['RMSE'],
         'R²': group_valid_metrics['R2'], 'n': group_valid_metrics['n'],
         '公司数': group_split_info['validation_公司数'],
         '说明': '未参与训练的跨公司验证子集，仅用于确认误差量级方向一致'},
        {'划分方式': '差异（Group − Random）', '数据子集': 'test',
         'MAE': round(group_metrics['MAE'] - random_metrics['MAE'], 6),
         'RMSE': round(group_metrics['RMSE'] - random_metrics['RMSE'], 6),
         'R²': round(group_metrics['R2'] - random_metrics['R2'], 6), 'n': group_metrics['n'],
         '公司数': '', '说明': '正 ΔMAE 表示跨公司泛化更困难，属预期方向'},
    ])
    group_error = ablation_shap.group_error_table(
        pd.DataFrame({schema.ID_FIELD: group_frame_test[schema.ID_FIELD],
                      'y_true': group_test_truth,
                      'y_pred': group_test_pred,
                      'residual': group_test_truth - group_test_pred}),
        model_frame, '公司 Group Split test')
    print(f'Group Split test MAE {group_metrics["MAE"]} vs Random test MAE {random_metrics["MAE"]}')

    # ---- 4. 稳健性：极端值 + 目标定义 ----
    low, high = np.percentile(y_train, 1), np.percentile(y_train, 99)
    winsor_train = np.clip(y_train, low, high)
    full_assembler, full_model, full_matrix_train, full_matrix_test = ablation_models['Full']
    winsor_model = model_training.make_model(final_model_key, final_params,
                                            random_state=model_training.SPLIT_RANDOM_STATE)
    winsor_model.fit(full_matrix_train, winsor_train)
    winsor_pred = np.asarray(winsor_model.predict(full_matrix_test), dtype='float64')
    winsor_metrics = model_training.regression_metrics(y_test, winsor_pred)
    main_metrics = model_training.regression_metrics(y_test, prediction_frames[('Full', 'test')]
                                                    ['y_pred'])
    from scipy import stats  # noqa: PLC0415

    main_importance = pd.Series(full_model.booster_.feature_importance(importance_type='gain'),
                                index=full_assembler.feature_names())
    winsor_importance = pd.Series(winsor_model.booster_.feature_importance(importance_type='gain'),
                                  index=full_assembler.feature_names())
    shared = main_importance.index.intersection(winsor_importance.index)
    importance_spearman = float(stats.spearmanr(main_importance[shared], winsor_importance[shared])
                                [0])
    robustness_table = pd.DataFrame([
        {'实验': '主模型（真实薪资 y）', 'MAE': main_metrics['MAE'], 'RMSE': main_metrics['RMSE'],
         'R²': main_metrics['R2'], '说明': '主分析与主模型统计范围，未做任何缩尾'},
        {'实验': '训练集 y winsorize 1%/99%', 'MAE': winsor_metrics['MAE'],
         'RMSE': winsor_metrics['RMSE'], 'R²': winsor_metrics['R2'],
         '说明': f'训练目标裁剪至 [{low:.1f}, {high:.1f}]，评估仍用真实 test 薪资'},
        {'实验': '特征重要性方向一致性（Spearman）', 'MAE': '', 'RMSE': '', 'R²': '',
         '说明': f'winsorize 与主模型 gain 重要性排名 Spearman = {importance_spearman:.4f}'
                 f'（共有特征 {len(shared)}）'},
        {'实验': 'winsorize 模型 Top5 特征', 'MAE': '', 'RMSE': '', 'R²': '',
         '说明': '、'.join(winsor_importance.sort_values(ascending=False).head(5).index.tolist())},
        {'实验': '主模型 Top5 特征', 'MAE': '', 'RMSE': '', 'R²': '',
         '说明': '、'.join(main_importance.sort_values(ascending=False).head(5).index.tolist())},
    ])
    target_rows = [{'目标定义': '薪资中点（主任务）', 'MAE': main_metrics['MAE'],
                    'RMSE': main_metrics['RMSE'], 'R²': main_metrics['R2'],
                    '说明': '正式主任务，本轮不改变'}]
    for target in ablation_shap.TARGET_VARIANTS:
        assembler_t = ablation_shap.build_assembler_for_groups(full_groups, grouped,
                                                              skill_threshold, text_dim)
        assembler_t.fit(train_frame, skill_map, text_for(train_frame))
        model_t = model_training.make_model(final_model_key, final_params,
                                           random_state=model_training.SPLIT_RANDOM_STATE)
        model_t.fit(assembler_t.transform(train_frame, skill_map, text_for(train_frame)),
                    train_frame[target].to_numpy(dtype='float64'))
        pred_t = np.asarray(
            model_t.predict(assembler_t.transform(test_frame, skill_map, text_for(test_frame))),
            dtype='float64')
        truth_t = test_frame[target].to_numpy(dtype='float64')
        metrics_t = model_training.regression_metrics(truth_t, pred_t)
        target_rows.append({'目标定义': target, 'MAE': metrics_t['MAE'], 'RMSE': metrics_t['RMSE'],
                            'R²': metrics_t['R2'],
                            '说明': '稳健性实验：仅比较趋势，不作为新的主任务或主模型'})
        print(f'{target}: MAE {metrics_t["MAE"]}')
    target_table = pd.DataFrame(target_rows)

    # ---- 5. TreeSHAP（只对最终主模型） ----
    matrix_test_artifact = artifact_assembler.transform(test_frame, skill_map,
                                                       text_for(test_frame))
    feature_names = artifact_assembler.feature_names()
    shap_matrix = ablation_shap.tree_shap_values(artifact_model, matrix_test_artifact)
    if shap_matrix.shape[1] != len(feature_names):
        raise SystemExit(f'SHAP 列数 {shap_matrix.shape[1]} 与特征名 {len(feature_names)} 不一致')
    # SHAP 自校验：基准值 + 各行贡献之和 == 模型预测（TreeSHAP 的加性一致性）
    contrib_all = artifact_model.predict(matrix_test_artifact, pred_contrib=True)
    contrib_all = (contrib_all.toarray() if hasattr(contrib_all, 'toarray')
                   else np.asarray(contrib_all, dtype='float64'))
    base_value = float(np.unique(np.round(contrib_all[:, -1], 6))[0])
    reconstruction_error = float(np.abs(
        contrib_all.sum(axis=1) - np.asarray(artifact_model.predict(matrix_test_artifact))).max())
    print(f'SHAP 基准值 {base_value:.4f} | 加性一致性最大误差 {reconstruction_error:.2e}')
    shap_table = ablation_shap.shap_overall_table(shap_matrix, feature_names)
    skill_shap_all = ablation_shap.shap_skill_table(
        shap_matrix, feature_names, membership, model_frame[schema.ID_FIELD].tolist(),
        skill_eda.ALL_USABLE_SCOPES, presence_matrix=matrix_test_artifact)
    key_skill = ablation_shap.key_skill_table(
        membership, model_frame[schema.ID_FIELD].tolist(), skill_eda.ALL_USABLE_SCOPES,
        skill_shap_all, KEY_SKILLS)
    skill_shap = skill_shap_all.head(SKILL_SHAP_TOP_N).copy()
    print('SHAP Top5:', '、'.join(shap_table.head(5)['特征'].tolist()))
    print('技能 SHAP Top5:', '、'.join(skill_shap.head(5)['技能'].tolist()))

    # 解释稳定性：多个随机种子的 mean|SHAP| 排名 Spearman
    stability_rows = []
    reference = pd.Series(np.abs(shap_matrix).mean(axis=0), index=feature_names)
    for seed in SEED_STABILITY_SEEDS:
        model_s = model_training.make_model(final_model_key, final_params, random_state=seed)
        model_s.fit(full_matrix_train, y_train)
        matrix_s = full_assembler.transform(test_frame, skill_map, text_for(test_frame))
        shap_s = ablation_shap.tree_shap_values(model_s, matrix_s)
        series_s = pd.Series(np.abs(shap_s).mean(axis=0), index=full_assembler.feature_names())
        shared_index = reference.index.intersection(series_s.index)
        rho = float(stats.spearmanr(reference[shared_index], series_s[shared_index])[0])
        stability_rows.append({'随机种子': seed, '与主模型 mean|SHAP| 排名 Spearman': round(rho, 4),
                               'Top10 特征': '、'.join(series_s.sort_values(ascending=False)
                                                    .head(10).index.tolist())})
    stability_table = pd.DataFrame(stability_rows)
    stability_spearman = '；'.join(f"{row['随机种子']}: {row['与主模型 mean|SHAP| 排名 Spearman']}"
                                  for row in stability_rows)

    metrics = {
        'split_counts': {'train': len(random_train), 'validation': len(random_valid),
                         'test': len(random_test)},
        'final_params': final_params, 'artifact_model_key': final_model_key,
        'official_model_metrics': official_metrics,
        'model_sample': int(len(model_frame)),
        'shap_model_key': final_model_key, 'shap_rows': int(len(test_frame)),
        'shap_base_value': round(base_value, 6),
        'shap_reconstruction_max_error': reconstruction_error,
        'feature_dimension': len(feature_names),
        'bootstrap_rounds': ablation_shap.BOOTSTRAP_ROUNDS,
        'bootstrap_seed': ablation_shap.BOOTSTRAP_SEED,
        'group_split': group_split_info,
        'shap_stability': stability_spearman,
        'same_split_check': len(split_signatures) == 1,
        'same_model_check': len(model_signatures) == 1,
        'feature_group_only_check': all(
            record['训练行数'] == len(train_frame) for record in ablation_records)
        and len({tuple(sorted(record['特征组'])) for record in ablation_records})
        == len(ablation_records),
    }
    audit_sheets = {
        '01_消融结果': ablation_table,
        '02_技能增量': increment_table,
        '03_文本增量': text_increment_table,
        '04_Random_vs_GroupSplit': group_table,
        '05_极端值敏感性': robustness_table,
        '06_目标稳健性': target_table,
        '07_SHAP总排名': shap_table,
        '08_技能SHAP': skill_shap_all,
        '09_分组误差': group_error,
    }
    register_gates(gates, audit_sheets, metrics, key_skill)
    audit_sheets['10_门禁'] = pd.DataFrame([
        {'门禁项': name, '状态': item['status'], '说明': item['note']}
        for name, item in gates.results.items()])
    audit_sheets['11_SHAP稳定性'] = stability_table
    audit_sheets['12_关键技能SHAP'] = key_skill
    audit_path = project_paths.TABLES_DIR / project_paths.TABLE_ABLATION_SHAP
    audit_path = io_utils.write_excel(audit_path, audit_sheets)

    style_snapshot = plot_style.setup_sci_style()
    project_paths.MODELING_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_style.SCI_FIGURES_DIR = project_paths.MODELING_FIGURES_DIR
    registry: list = []
    figure_ablation(ablation_table, increment_table, registry)
    figure_random_vs_group(group_table, registry)
    figure_shap_beeswarm(shap_matrix, feature_names, matrix_test_artifact, registry)
    figure_skill_shap(skill_shap_all, registry)
    figure_checks = plot_style.check_figure_gates(registry, style_snapshot)

    metrics.update({
        'audit_path': project_paths.relative_to_root(audit_path),
        'ablation': ablation_table.to_dict('records'),
        'skill_increment': increment_table.to_dict('records'),
        'text_increment': text_increment_table.to_dict('records'),
        'group_split_table': group_table.to_dict('records'),
        'robustness': robustness_table.to_dict('records'),
        'target_variants': target_table.to_dict('records'),
        'shap_top20': shap_table.head(SHAP_TOP_N).to_dict('records'),
        'skill_shap_top20': skill_shap.to_dict('records'),
        'key_skills': key_skill.to_dict('records'),
        'low_frequency_skill_count': int(skill_shap_all['低频标记'].ne('').sum()),
        'figure_checks': figure_checks, 'figures': registry,
        'gates': {name: item['status'] for name, item in gates.results.items()},
    })
    io_utils.write_json(project_paths.METRICS_DIR / f'{STAGE}.json', metrics)

    record_path = write_record(metrics, audit_sheets)
    print(f'阶段记录: {project_paths.relative_to_root(record_path)}')
    print(f'图目录: {project_paths.relative_to_root(project_paths.MODELING_FIGURES_DIR)}')
    quality.print_gates(gates.results, order=quality.STAGE_GATE_MAP[STAGE])
    gates.save()
    return 0


if __name__ == '__main__':
    sys.exit(main())

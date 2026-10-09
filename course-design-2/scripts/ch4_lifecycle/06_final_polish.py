# -*- coding: utf-8 -*-
"""Stage26.4：数据/模型修正与正文修订配套的必要重算（只新增文件）。

对应提示词 `docs/prompts/Trae_Stage26_4_终稿精简与学术化统一修订提示词.md`：

1. §13 / §15.2：用正式 **Strict Recruitment Episode**（周期层去重）重新生成图 4-4，
   输出到 `outputs/figures/supplementary/fig_s50_*.png/pdf`，图内文字极简、2×1 纵排；
2. §18.1 / §18.2：逐字段核查正式模型输入，把数据治理元数据（映射置信度、是否需人工复核、
   是否跨地域、是否存在同名跨地域歧义、技能提取范围、文本与方向一致性质量标志等）
   从未移出的字段中一并移出，并保持三个后验版本特征继续移出；
3. §18.1 / §19 / §15：移除后重训主模型（LightGBM 400/0.05/63，阈值 100，SVD 16，
   train-only 拟合），同步重算三种划分、消融五配置、模型比较、最终模型 test、
   多种子、公司分组 5 次、SHAP、稳健性与分组误差；Safe-F 只作扩展敏感性；
4. §20 / §21：重建多变量中位数回归——岗位大类与岗位细分类按 multi-hot 重叠变量处理
   （系数 = 命中该类别与未命中该类别的条件中位数差异），bootstrap 由 100 次提高到 1000 次，
   不再报告“61.5% 系数显著”一类全局判定；
5. 图 8-1 / 8-3 随新指标重绘为 `图S51` / `图S52`（不覆盖既有图件）。

运行::

    python scripts\\06_final_polish.py baseline
    python scripts\\06_final_polish.py compute
    python scripts\\06_final_polish.py textaudit

新增输出::

    outputs/tables/ch7/51_stage26_4_formal_feature_audit.xlsx
    outputs/tables/ch7/52_stage26_4_metrics_final.xlsx
    outputs/tables/ch5/53_stage26_4_median_regression.xlsx
    outputs/tables/ch4/54_stage26_4_text_revision_audit.xlsx
    outputs/figures/supplementary/fig_4_5_recruitment_duration_distribution.png/pdf
    outputs/figures/supplementary/fig_s51_feature_group_ablation_comparison.png/pdf
    outputs/figures/supplementary/fig_s52_generalization_scenario_comparison.png/pdf
    outputs/logs/metrics/stage_26_4_model_sync.json
    outputs/logs/metrics/stage_26_4_text_baseline.json
    outputs/logs/metrics/stage_26_4_text_revision.json
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = next(
    _candidate for _candidate in Path(__file__).resolve().parents
    if (_candidate / "data").is_dir() and (_candidate / "scripts").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import (ablation_shap, figure_finalize, io_utils, model_training,  # noqa: E402
                 plot_style, project_paths, schema, skill_eda)
from src.script_support import (build_assembler, dump_json, fit_eval,  # noqa: E402
                                       grouped_columns, section_length, sha256_of)

SEED = 42
SEEDS = (42, 52, 62, 72, 82)
SKILL_THRESHOLD = 100
TEXT_DIM = 16
LIGHTGBM_PARAMS = {'n_estimators': 400, 'learning_rate': 0.05, 'num_leaves': 63}
CATBOOST_PARAMS = {'depth': 8, 'learning_rate': 0.05, 'iterations': 500}
MODEL_FEATURE_GROUPS = ('A', 'B', 'C', 'D', 'E')
SAFE_F_STAGE26_1 = ['publish_month', 'publish_weekday', 'episode_no_strict',
                    'is_confirmed_reopen', 'historical_episode_count',
                    'previous_reopen_gap_days']
SAFE_F_FINAL = ['publish_month', 'publish_weekday']
BOOTSTRAP_ROUNDS = 1000
BOOTSTRAP_SEED = 42
MEDIAN_BOOTSTRAP_ROUNDS = 1000
MEDIAN_MIN_FREQ = 20
MEDIAN_TOP_MULTI = 30
MEDIAN_CATEGORICAL = ['工作城市', '学历要求', '每周到岗要求', '实习时长要求',
                      '公司规模', '公司性质', '所属行业', '公司认证类别']
MEDIAN_MULTI = ['岗位大类集合', '岗位细分类集合']

# ---- Stage26.3 已移出的字段（继续移出） ----
REMOVED_STAGE26_3 = {
    '核心版本数': '发布时点不可得的后验版本特征',
    '完整页面版本数': '发布时点不可得的后验版本特征',
    '是否多版本岗位': '发布时点不可得的后验版本特征',
    '岗位描述分词数': '与岗位描述字符数高度冗余（Spearman ρ = 0.976632）',
    '公司规模下限': '与公司规模等级高度冗余（ρ = 0.999994）',
    '公司规模上限': '与公司规模中点高度冗余（ρ = 1.000000），缺失率 49.01%',
    '公司规模中点': '与公司规模下限高度冗余（ρ = 0.999884），缺失率 49.01%',
    '公司规模是否已知': '近常量（取值最多的类别占 99.97%）',
    '公司规模槽位异常标志': '近常量（取值最多的类别占 99.97%）',
}
# ---- Stage26.4 新移出的数据治理元数据 ----
GOVERNANCE_FEATURES = {
    '映射置信度': ('公司实体映射治理', '公司实体映射的置信度，属数据治理审计元数据'),
    '是否需人工复核': ('公司实体映射治理', '映射复核流程标记，属数据治理审计元数据'),
    '是否跨地域': ('公司实体映射治理', '映射过程中的跨地域标记，属数据治理审计元数据'),
    '是否存在同名跨地域歧义': ('公司实体映射治理', '同名跨地域歧义标记，属数据治理审计元数据'),
    '岗位方向_与平台分类一致标志': ('数据来源质量', '表头统计范围与平台分类的一致性检查标记，属数据质量元数据'),
    '文本是否为空': ('数据来源质量', '岗位描述文本的可用性标记，属数据质量元数据'),
    '是否有技能': ('数据来源质量', '技能命中情况的数据可用性标记，属数据质量元数据'),
    '文本向量是否可用': ('数据来源质量', '文本向量可用性标记，属数据质量元数据'),
    '技能提取范围': ('技能匹配状态', '技能抽取时的匹配统计范围标记，属技能匹配状态元数据'),
    '岗位描述分段状态': ('文本清洗标记', '岗位描述分段处理的清洗状态标记，属清洗过程元数据'),
}
REMOVED_FINAL = list(REMOVED_STAGE26_3) + list(GOVERNANCE_FEATURES)

GROUP_LETTER_NAME = {'A': 'A 岗位基础', 'B': 'B 地域', 'C': 'C 公司',
                     'D': 'D 技能', 'E': 'E 文本语义', 'SafeF': 'Safe-F 发布时间位置'}
PURPOSE = {
    '学历等级': '学历要求的数值编码，刻画学历门槛高低',
    '每周到岗天数': '每周到岗要求的数值编码，刻画到岗强度要求',
    '实习月数': '实习时长要求的数值编码，刻画实习周期要求',
    '岗位大类数量': '岗位命中的岗位大类个数，刻画岗位职能的宽窄',
    '岗位细分类数量': '岗位命中的岗位细分类个数，刻画职能方向的集中程度',
    '岗位标签数量': '平台岗位标签个数，刻画页面标注丰富度',
    '岗位描述字符数': '岗位描述文本长度，刻画信息量与描述详尽度',
    '岗位描述分词数': '岗位描述分词计数，与字符数语义重复',
    '岗位方向_与平台分类一致标志': '岗位方向文本与平台分类是否一致的检查标记',
    '核心版本数': '岗位核心业务版本数量，发布时点不可得',
    '完整页面版本数': '岗位完整页面版本数量，发布时点不可得',
    '是否多版本岗位': '岗位是否出现多个完整页面版本，发布时点不可得',
    '是否直辖市': '工作城市是否直辖市的区位标记',
    '映射置信度': '公司实体映射的置信度，属数据治理过程记录',
    '是否需人工复核': '公司实体映射是否进入人工复核流程',
    '是否跨地域': '公司实体映射是否跨地域合并',
    '是否存在同名跨地域歧义': '同名公司是否存在跨地域歧义',
    '公司规模下限': '公司规模的区间下限数值',
    '公司规模上限': '公司规模的区间上限数值',
    '公司规模中点': '公司规模的区间中点数值',
    '公司规模等级': '公司规模区间的等级编码，按人数区间有序',
    '公司规模是否已知': '公司规模是否可解析的可用性标记',
    '公司规模槽位异常标志': '公司规模字段槽位异常的修复标记',
    '公司标签数量': '岗位命中的公司福利标签数量',
    '公司简介字符数': '公司简介文本长度，刻画企业信息详尽度',
    '技能数量': '岗位命中的技能条目总数',
    '技术技能数': '岗位命中的技术类技能数量',
    '工程工具数': '岗位命中的工程工具类技能数量',
    '数据工具数': '岗位命中的数据工具类技能数量',
    '业务能力数': '岗位命中的业务能力类技能数量',
    '技术领域数': '岗位命中的技术领域类技能数量',
    'AI技能数': '岗位命中的 AI 类技能数量',
    '大模型技能数': '岗位命中大模型相关技能的数量',
    '编程语言技能数': '岗位命中的编程语言类技能数量',
    '数据库技能数': '岗位命中的数据库类技能数量',
    '安全技能数': '岗位命中的安全类技能数量',
    '办公工具技能数': '岗位命中的办公工具类技能数量',
    '文本是否为空': '岗位描述文本是否为空的可用性标记',
    '是否有技能': '岗位是否命中任何技能的可用性标记',
    '文本向量是否可用': '岗位描述语义向量是否可用的标记',
    '学历要求': '岗位标注的学历门槛类别',
    '每周到岗要求': '岗位标注的每周到岗天数要求类别',
    '实习时长要求': '岗位标注的实习时长要求类别',
    '岗位方向_文本': '从岗位描述文本推断的岗位方向类别',
    '岗位描述分段状态': '岗位描述分段处理结果的状态标记',
    '工作城市': '岗位标注的工作城市（原始统计范围）',
    '工作城市_规范': '岗位工作城市的规范化取值',
    '所属行业': '公司所属行业类别',
    '公司性质': '公司性质类别',
    '公司规模': '公司规模区间类别',
    '技能提取范围': '技能抽取所用匹配统计范围的标记',
    '岗位大类集合': '岗位命中的平台岗位大类（多值）',
    '岗位细分类集合': '岗位命中的平台岗位细分类（多值）',
    '公司标签列表': '岗位命中的公司福利标签集合（高基数多值）',
}
TABLES = project_paths.TABLES_DIR
SUPP_DIR = project_paths.FIGURES_DIR / 'supplementary'
METRICS_DIR = project_paths.METRICS_DIR
SOURCE_DIR = PROJECT_ROOT / 'docs' / 'paper' / 'stage23'
TABLE_FEATURE_AUDIT = TABLES / 'ch7/51_stage26_4_formal_feature_audit.xlsx'
TABLE_METRICS = TABLES / 'ch7/52_stage26_4_metrics_final.xlsx'
TABLE_MEDIAN = TABLES / 'ch5/53_stage26_4_median_regression.xlsx'
TABLE_TEXT_AUDIT = TABLES / 'ch4/54_stage26_4_text_revision_audit.xlsx'
METRICS_PATH = METRICS_DIR / 'stage_26_4_model_sync.json'
TEXT_BASELINE_PATH = METRICS_DIR / 'stage_26_4_text_baseline.json'
TEXT_METRICS_PATH = METRICS_DIR / 'stage_26_4_text_revision.json'
FIG_DURATION = 'fig_4_5_recruitment_duration_distribution'
FIG_ABLATION = 'fig_s51_feature_group_ablation_comparison'
FIG_GENERALIZE = 'fig_s52_generalization_scenario_comparison'
EPISODE_PATH = project_paths.PROCESSED_DIR / 'job_strict_episode_26_1.parquet'
CHAPTERS = ['00_摘要与Abstract.md', '01_绪论.md', '02_相关理论与分析方法.md',
            '03_数据获取与预处理.md', '04_互联网IT实习岗位特征分析.md',
            '05_实习岗位薪资影响因素分析.md', '06_实习岗位技能需求分析.md',
            '07_薪资预测模型构建与结果分析.md', '08_模型稳健性与解释.md',
            '09_总结与展望.md']
NEW_FILES = [TABLE_FEATURE_AUDIT, TABLE_METRICS, TABLE_MEDIAN, TABLE_TEXT_AUDIT,
             SUPP_DIR / f'{FIG_DURATION}.png', SUPP_DIR / f'{FIG_DURATION}.pdf',
             SUPP_DIR / f'{FIG_ABLATION}.png', SUPP_DIR / f'{FIG_ABLATION}.pdf',
             SUPP_DIR / f'{FIG_GENERALIZE}.png', SUPP_DIR / f'{FIG_GENERALIZE}.pdf',
             METRICS_PATH, TEXT_BASELINE_PATH, TEXT_METRICS_PATH]
SKIP_DIRS = {'.git', '.pytest_cache', '__pycache__', '.ipynb_checkpoints', '.idea', '.vscode'}
MANIFEST_SCOPE_DIRS = ['data', 'outputs', 'docs', 'src', 'scripts', 'config',
                       'notebooks', 'tests']


# ============================================================================
# 通用工具
# ============================================================================
def project_manifest() -> dict:
    manifest: dict = {}
    new_resolved = {path.resolve() for path in NEW_FILES}
    for name in MANIFEST_SCOPE_DIRS:
        base = PROJECT_ROOT / name
        if not base.is_dir():
            continue
        for path in sorted(base.rglob('*')):
            if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
                continue
            if path.resolve() in new_resolved or path == Path(__file__).resolve():
                continue
            manifest[str(path.relative_to(PROJECT_ROOT)).replace('\\', '/')] = sha256_of(path)
    readme = PROJECT_ROOT / 'README.md'
    if readme.is_file():
        manifest['README.md'] = sha256_of(readme)
    return manifest


def write_excel(path: Path, sheets: dict) -> None:
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for name, frame in sheets.items():
            value = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
            value.to_excel(writer, sheet_name=name[:31], index=False)


def drop_features(grouped: dict, removed) -> dict:
    removed = set(removed)
    return {letter: {key: [column for column in spec[key] if column not in removed]
                     for key in ('numeric', 'categorical', 'multi')}
            for letter, spec in grouped.items()}


def paired_bootstrap(y_true, pred_without, pred_with, rounds: int = BOOTSTRAP_ROUNDS,
                     seed: int = BOOTSTRAP_SEED) -> dict:
    truth = np.asarray(y_true, dtype='float64')
    left = np.asarray(pred_without, dtype='float64')
    right = np.asarray(pred_with, dtype='float64')
    rng = np.random.default_rng(seed)
    size = len(truth)
    deltas = np.empty(rounds, dtype='float64')
    for index in range(rounds):
        sample = rng.integers(0, size, size)
        deltas[index] = (np.abs(truth[sample] - left[sample]).mean()
                         - np.abs(truth[sample] - right[sample]).mean())
    low, high = np.percentile(deltas, 2.5), np.percentile(deltas, 97.5)
    return {'ΔMAE': round(float(np.abs(truth - left).mean() - np.abs(truth - right).mean()), 6),
            'CI95下界': round(float(low), 6), 'CI95上界': round(float(high), 6),
            '是否跨0': '是' if low <= 0 <= high else '否'}


# ============================================================================
# 一、图 4-4：严格统计范围招聘周期持续时长（图S50）
# ============================================================================
def duration_block(values) -> dict:
    array = np.asarray(values, dtype='float64')
    array = array[array > 0]
    return {'n': int(array.size),
            'P10': float(np.percentile(array, 10)), 'P25': float(np.percentile(array, 25)),
            'Median': float(np.percentile(array, 50)), 'P75': float(np.percentile(array, 75)),
            'P90': float(np.percentile(array, 90)),
            'IQR': float(np.percentile(array, 75) - np.percentile(array, 25)),
            'max': float(array.max()), 'mean': float(array.mean())}


def figure_duration_strict(episodes: pd.DataFrame) -> dict:
    """图 4-4：严格统计范围招聘周期持续时长分布与累积分布（2×1，图内文字极简）。"""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    unique = episodes.drop_duplicates(subset=['intern_id', 'episode_id_strict'])
    duration = np.sort(unique.loc[unique['final_observed_planned_duration_days'] > 0,
                                  'final_observed_planned_duration_days']
                       .to_numpy('float64'))
    lower = float(np.quantile(duration, 0.005))
    upper = float(np.quantile(duration, 0.99))
    bins = np.linspace(lower, upper, 46)
    fig, axes = plt.subplots(2, 1, figsize=(5.85, 6.4))
    for ax in axes:
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontsize(plot_style.FONT_SIZES['tick'])

    ax = axes[0]
    ax.hist(duration, bins=bins, color=plot_style.MAIN_COLOR, edgecolor='black',
            linewidth=0.4, label='周期数直方图')
    median = float(np.median(duration))
    q1, q3 = float(np.quantile(duration, 0.25)), float(np.quantile(duration, 0.75))
    ax.axvspan(q1, q3, color=plot_style.ACCENT_COLOR, alpha=0.16, linewidth=0,
               label='IQR 区间')
    ax.axvline(median, color=plot_style.ACCENT_COLOR, linestyle='--', linewidth=1.1,
               label='中位数')
    ax.set_xlim(lower, upper)
    ax.set_xlabel('招聘周期计划持续天数')
    ax.set_ylabel('正式招聘周期数')
    ax.legend(loc='upper right', frameon=False, fontsize=plot_style.FONT_SIZES['legend'])
    plot_style.add_subfigure_caption(ax, 'a', '招聘周期计划持续时长分布',
                                     fontsize=plot_style.FONT_SIZES['legend'])
    plot_style.apply_sci_axis(ax, grid_axis='y')

    ax = axes[1]
    cumulative = np.arange(1, duration.size + 1) / duration.size
    ax.plot(duration, cumulative, color=plot_style.MAIN_COLOR, linewidth=1.8)
    for quantile, color, style, width in (
            (0.10, plot_style.MUTED_COLOR, '--', 1.2),
            (0.50, plot_style.ACCENT_COLOR, '-', 1.9),
            (0.90, plot_style.MUTED_COLOR, ':', 1.2)):
        value = float(np.quantile(duration, quantile))
        label = f'P{int(quantile * 100)} = {value:.0f} 天'
        ax.axvline(value, color=color, linestyle=style, linewidth=width, label=label)
    ax.set_xlim(lower, upper)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('招聘周期计划持续天数')
    ax.set_ylabel('累计比例')
    plot_style.format_percent_axis(ax, axis='y', decimals=0)
    ax.legend(loc='lower right', frameon=False, fontsize=plot_style.FONT_SIZES['legend'])
    plot_style.add_subfigure_caption(ax, 'b', '招聘周期计划持续时长经验累积分布',
                                     fontsize=plot_style.FONT_SIZES['legend'])
    plot_style.apply_sci_axis(ax, grid_axis='both')
    fig.subplots_adjust(left=0.155, right=0.975, bottom=0.135, top=0.985, hspace=0.30)
    diagnostics = figure_finalize.save_paper_figure(
        fig, SUPP_DIR, FIG_DURATION,
        subfigures=[('a', '招聘周期计划持续时长分布', axes[0]),
                    ('b', '招聘周期计划持续时长经验累积分布', axes[1])],
        meta={'数据来源': 'job_strict_episode_26_1.parquet（严格统计范围，周期层去重）',
              '统计范围': '计划持续天数 = 末次观测截止日 − 发布时间（日历差）+ 1；'
                      '属计划窗口，不等于实际招满所需时间',
              'seed': SEED,
              '用途': '第4章 图 4-4（与表 4-2 及正文严格范围一致）'})
    plt.close(fig)
    return diagnostics


# ============================================================================
# 二、图 8-1 / 图 8-3 重绘（随新指标）
# ============================================================================
def figure_ablation_new(ablation: pd.DataFrame) -> dict:
    """图 8-1：五配置消融的验证集与测试集 MAE（图内文字极简）。"""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(5.85, 3.7))
    labels = list(ablation['配置'])
    positions = np.arange(len(labels))
    width = 0.38
    valid = ablation['validation_MAE'].to_numpy('float64')
    test = ablation['test_MAE'].to_numpy('float64')
    ax.bar(positions - width / 2, valid, width, color=plot_style.MAIN_COLOR,
           edgecolor='black', linewidth=0.6, label='验证集 MAE')
    ax.bar(positions + width / 2, test, width, color=plot_style.ACCENT_COLOR,
           edgecolor='black', linewidth=0.6, label='测试集 MAE')
    for offset, values in ((-width / 2, valid), (width / 2, test)):
        for position, value in zip(positions + offset, values):
            ax.text(position, value + 0.12, f'{value:.2f}', ha='center', va='bottom',
                    fontsize=plot_style.FONT_SIZES['annotation'], rotation=90)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=12, ha='right')
    ax.set_xlabel('消融配置（特征组）')
    ax.set_ylabel('MAE（元/天）')
    ax.set_ylim(0, float(max(valid.max(), test.max())) * 1.28)
    ax.legend(loc='upper left', frameon=False, fontsize=plot_style.FONT_SIZES['legend'])
    ax.text(0.985, 0.955, '(a)', transform=ax.transAxes, ha='right', va='top',
            fontsize=plot_style.FONT_SIZES['legend'])
    plot_style.apply_sci_axis(ax, grid_axis='y')
    fig.subplots_adjust(left=0.135, right=0.975, bottom=0.275, top=0.975)
    diagnostics = figure_finalize.save_paper_figure(
        fig, SUPP_DIR, FIG_ABLATION, subfigures=[],
        meta={'数据来源': 'ch7/52_stage26_4_metrics_final.xlsx / 02_消融五配置',
              'seed': SEED, '用途': '第8章 图 8-1（随正式特征集更新重绘）'})
    plt.close(fig)
    return diagnostics


def figure_generalization_new(unified: pd.DataFrame) -> dict:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    labels = ['Random\nSplit', 'Company\nGroup Split', 'Retrospective\nTemporal Split']
    fig, axes = plt.subplots(2, 1, figsize=(5.85, 6.4))
    positions = np.arange(len(unified))
    ax = axes[0]
    ax.bar(positions - 0.19, unified['test MAE'], width=0.36, color=plot_style.MAIN_COLOR,
           edgecolor='black', linewidth=0.6, label='MAE')
    ax.bar(positions + 0.19, unified['test RMSE'], width=0.36, color=plot_style.MUTED_COLOR,
           edgecolor='black', linewidth=0.6, label='RMSE')
    span = float(unified[['test MAE', 'test RMSE']].to_numpy().max())
    for offset, column in ((-0.19, 'test MAE'), (0.19, 'test RMSE')):
        for position, value in zip(positions + offset, unified[column]):
            ax.text(position, value + span * 0.02, f'{value:.2f}', ha='center', va='bottom',
                    fontsize=plot_style.FONT_SIZES['annotation'])
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel('数据划分方式（同一协议，仅在各自训练集上拟合）')
    ax.set_ylabel('测试集误差（元/天）')
    ax.set_ylim(0, span * 1.34)
    ax.legend(loc='upper left', frameon=False, fontsize=plot_style.FONT_SIZES['legend'])
    ax.text(0.985, 0.955, '(a)', transform=ax.transAxes, ha='right', va='top',
            fontsize=plot_style.FONT_SIZES['legend'])
    plot_style.apply_sci_axis(ax, grid_axis='y')

    ax = axes[1]
    ax.bar(positions, unified['test R²'], width=0.5, color=plot_style.PALETTE[2],
           edgecolor='black', linewidth=0.6, label='R²')
    for position, value in zip(positions, unified['test R²']):
        ax.text(position, value + 0.012, f'{value:.3f}', ha='center', va='bottom',
                fontsize=plot_style.FONT_SIZES['annotation'])
    ax.set_xticks(positions)
    ax.set_xticklabels(['Random Split', 'Company Group Split', 'Temporal Split'])
    ax.set_xlabel('数据划分方式')
    ax.set_ylabel('测试集 R²')
    ax.set_ylim(0, float(unified['test R²'].max()) * 1.30)
    ax.legend(loc='upper right', frameon=False, fontsize=plot_style.FONT_SIZES['legend'])
    ax.text(0.013, 0.955, '(b)', transform=ax.transAxes, ha='left', va='top',
            fontsize=plot_style.FONT_SIZES['legend'])
    plot_style.apply_sci_axis(ax, grid_axis='y')
    fig.subplots_adjust(left=0.175, right=0.975, bottom=0.135, top=0.985, hspace=0.46)
    diagnostics = figure_finalize.save_paper_figure(
        fig, SUPP_DIR, FIG_GENERALIZE, subfigures=[],
        meta={'数据来源': 'ch7/52_stage26_4_metrics_final.xlsx / 02_三种划分统一协议',
              '用途': '第8章 图 8-3（随正式特征集更新重绘）'})
    plt.close(fig)
    return diagnostics


# ============================================================================
# 三、多变量中位数回归（multi-hot 统计范围 + 1000 次 bootstrap）
# ============================================================================
def build_median_design(frame: pd.DataFrame):
    """设计矩阵：单值互斥类别含参照类别；多值重叠类别按命中指示列全量进入。"""
    names = ['截距']
    blocks = [np.ones(len(frame), dtype='float64')]
    references = {}
    for column in MEDIAN_CATEGORICAL:
        series = frame[column].fillna('__未知__').astype(str)
        counts = series.value_counts()
        kept = [value for value, count in counts.items() if count >= MEDIAN_MIN_FREQ]
        reference = str(counts.index[0])
        references[column] = reference
        for level in kept:
            if level == reference:
                continue
            names.append(f'{column}={level}')
            blocks.append(series.eq(level).to_numpy('float64'))
        other = ~series.isin(kept)
        if bool(other.any()):
            names.append(f'{column}=其他')
            blocks.append(other.to_numpy('float64'))
    for column in MEDIAN_MULTI:
        exploded = frame[column].explode().dropna().astype(str)
        counts = exploded.value_counts()
        levels = [value for value, count in counts.items() if count >= MEDIAN_MIN_FREQ]
        levels = sorted(levels, key=lambda value: -counts[value])[:MEDIAN_TOP_MULTI]
        sets = [set(value) if value is not None else set() for value in frame[column]]
        for level in levels:
            names.append(f'{column}={level}')
            blocks.append(np.array([float(level in item) for item in sets]))
    return np.column_stack(blocks), names, references


def _median_fit(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import QuantileRegressor  # noqa: PLC0415

    model = QuantileRegressor(quantile=0.5, alpha=0.0, fit_intercept=False, solver='highs-ipm')
    model.fit(matrix, target)
    return np.asarray(model.coef_, dtype='float64')


def median_bootstrap(design: np.ndarray, target: np.ndarray, rounds: int,
                     seed: int) -> np.ndarray:
    """成对 bootstrap：把 rounds 轮分块并行（每块内串行），返回 rounds × p 的系数矩阵。"""
    from joblib import Parallel, delayed  # noqa: PLC0415

    workers = 8
    per_block = int(np.ceil(rounds / workers))
    blocks = []
    remaining = rounds
    start = 0
    while remaining > 0:
        size = min(per_block, remaining)
        blocks.append((start, size))
        start += size
        remaining -= size

    def run_chunk(offset: int, size: int):
        rng = np.random.default_rng(seed + 1000 + offset)
        holder = np.empty((size, design.shape[1]), dtype='float64')
        for index in range(size):
            sample = rng.integers(0, len(target), len(target))
            holder[index] = _median_fit(design[sample], target[sample])
        return offset, holder

    results = Parallel(n_jobs=workers, backend='loky')(
        delayed(run_chunk)(offset, size) for offset, size in blocks)
    holder = np.empty((rounds, design.shape[1]), dtype='float64')
    for offset, block in results:
        holder[offset:offset + len(block)] = block
    return holder


def representative_median_rows(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prefix in MEDIAN_CATEGORICAL:
        block = table[table['变量'].str.startswith(f'{prefix}=')]
        if len(block):
            rows.append(block.loc[block['系数'].abs().idxmax()])
    for prefix in MEDIAN_MULTI:
        block = table[table['变量'].str.startswith(f'{prefix}=')]
        if len(block):
            ordered = block.reindex(block['系数'].abs().sort_values(ascending=False).index)
            rows.extend([ordered.iloc[0]] + ([ordered.iloc[1]] if len(ordered) > 1 else []))
    return pd.DataFrame(rows).reset_index(drop=True)


# ============================================================================
# 主流程
# ============================================================================
def run_compute() -> int:  # noqa: C901
    started = time.time()
    print('=' * 96)
    print('Stage26.4 数据/模型修正重算（只新增文件）')
    print('=' * 96)
    manifest_before = project_manifest()
    print(f'运行前既有文件 SHA-256 清单：{len(manifest_before)} 个')
    stage26_3 = json.loads(
        (METRICS_DIR / 'stage_26_3_structure_finalize.json').read_text(encoding='utf-8'))

    model_frame = io_utils.read_parquet(project_paths.JOB_SALARY_MODEL_DATASET_PARQUET)
    splits = io_utils.read_parquet(project_paths.MODEL_SPLITS_PARQUET)
    membership = io_utils.read_parquet(project_paths.JOB_SKILL_MEMBERSHIP_PARQUET)
    analysis = io_utils.read_parquet(project_paths.JOB_ANALYSIS_DATASET_PARQUET)
    entity = pd.read_parquet(project_paths.PROCESSED_UNIQUE_PARQUET,
                             columns=[schema.ID_FIELD, '发布时间', schema.CERT_TAG_FIELD])
    entity['发布时间'] = pd.to_datetime(entity['发布时间'], errors='coerce')
    entity_publish = entity.set_index(schema.ID_FIELD)['发布时间']
    feature_manifest = json.loads(
        (project_paths.SALARY_MODEL_DIR / 'feature_manifest.json').read_text(encoding='utf-8'))
    assert len(model_frame) == 14883 and len(splits) == 14883

    job_ids = model_frame[schema.ID_FIELD]
    frame = model_frame.copy()
    publish = job_ids.map(entity_publish)
    frame['publish_month'] = publish.dt.month.to_numpy()
    frame['publish_weekday'] = publish.dt.weekday.to_numpy()
    random_labels = pd.Series(splits['split'].to_numpy(), index=job_ids)
    group_labels = pd.Series(splits['company_group_split'].to_numpy(), index=job_ids)

    temporal_frame = model_frame.assign(_business_date=publish.dt.normalize())
    date_counts = temporal_frame.groupby('_business_date').size().sort_index()
    cumulative = date_counts.cumsum()
    total = int(cumulative.iloc[-1])
    dates = cumulative.index
    index1 = min(max(int(np.searchsorted(cumulative.to_numpy(), 0.70 * total, side='left')), 1),
                 len(dates) - 2)
    index2 = min(max(int(np.searchsorted(cumulative.to_numpy(), 0.85 * total, side='left')),
                     index1 + 1), len(dates) - 1)
    temporal_values = np.where(
        temporal_frame['_business_date'].isin(dates[:index1]), 'train',
        np.where(temporal_frame['_business_date'].isin(dates[index1:index2]),
                 'validation', 'test'))
    temporal_labels = pd.Series(temporal_values, index=job_ids)
    split_definitions = {'Random Split': random_labels,
                         'Company Group Split': group_labels,
                         'Retrospective Temporal Split': temporal_labels}

    grouped = grouped_columns(feature_manifest, frame, SAFE_F_STAGE26_1, SAFE_F_FINAL)
    grouped_new = drop_features(grouped, REMOVED_FINAL)
    skill_map = model_training.build_skill_map(membership, skill_eda.ALL_USABLE_SCOPES)
    text_matrix = model_training.load_text_matrix(
        job_ids.tolist(), project_paths.FEATURES_DIR / 'job_text_embeddings.npz',
        project_paths.FEATURES_DIR / 'job_text_embedding_index.parquet')
    text_by_id = {job_id: position for position, job_id in enumerate(job_ids.tolist())}

    # ---------------------------------------------------------- A 字段核查审计
    audit_rows = []
    for column in (list(feature_manifest['numeric_columns'])
                   + list(feature_manifest['categorical_columns'])
                   + list(feature_manifest['multi_value_columns'])):
        letter = ablation_shap.column_group(column)
        if column in GOVERNANCE_FEATURES:
            category, reason = GOVERNANCE_FEATURES[column]
            audit_rows.append({'字段': column, '所属组': GROUP_LETTER_NAME.get(letter, letter),
                               '用途': PURPOSE.get(column, '—'), '是否进入正式模型': '否',
                               '决策': '移出（数据治理元数据）',
                               '理由': f'{category}：{reason}，不进入正式薪资预测模型'})
        elif column in REMOVED_STAGE26_3:
            audit_rows.append({'字段': column, '所属组': GROUP_LETTER_NAME.get(letter, letter),
                               '用途': PURPOSE.get(column, '—'), '是否进入正式模型': '否',
                               '决策': '移出（Stage26.3 已判定）',
                               '理由': REMOVED_STAGE26_3[column]})
        else:
            audit_rows.append({'字段': column, '所属组': GROUP_LETTER_NAME.get(letter, letter),
                               '用途': PURPOSE.get(column, '—'), '是否进入正式模型': '是',
                               '决策': '保留（发布时点可得）',
                               '理由': '岗位发布时点即可从页面字段得到，属正式预测特征'})
    audit_table = pd.DataFrame(audit_rows)

    dimension_rows = []
    for label, groups in (('A+B+C', ('A', 'B', 'C')), ('A+B+C+D', ('A', 'B', 'C', 'D')),
                          ('A+B+C+E', ('A', 'B', 'C', 'E')),
                          ('A+B+C+D+E', MODEL_FEATURE_GROUPS),
                          ('A+B+C+D+E+SafeF', MODEL_FEATURE_GROUPS + ('SafeF',))):
        new_numeric = sum(len(grouped_new[letter]['numeric']) for letter in groups)
        new_result = fit_eval(frame, random_labels, groups, grouped_new, skill_map,
                              text_matrix, text_by_id, SKILL_THRESHOLD, TEXT_DIM)
        old_dim = int(stage26_3['消融五配置'][list(
            pd.DataFrame(stage26_3['消融五配置'])['配置']).index(label)]['特征维度'])
        dimension_rows.append({'配置': label, '编码后特征维度（Stage26.3）': old_dim,
                               '编码后特征维度（Stage26.4）': new_result['特征维度'],
                               '维度变化': int(new_result['特征维度'] - old_dim),
                               '数值特征列数（Stage26.4）': int(new_numeric),
                               '技能列数': new_result['技能列数']})
        print(f"  维度对照 {label}: {old_dim} → {new_result['特征维度']}")
    dimension_table = pd.DataFrame(dimension_rows)

    # ---------------------------------------------------------- B 消融五配置
    ablation_rows, cache = [], {}
    ablation_order = ['A+B+C', 'A+B+C+D', 'A+B+C+E', 'A+B+C+D+E', 'A+B+C+D+E+SafeF']
    ablation_groups = {'A+B+C': ('A', 'B', 'C'), 'A+B+C+D': ('A', 'B', 'C', 'D'),
                       'A+B+C+E': ('A', 'B', 'C', 'E'), 'A+B+C+D+E': MODEL_FEATURE_GROUPS,
                       'A+B+C+D+E+SafeF': MODEL_FEATURE_GROUPS + ('SafeF',)}
    for label in ablation_order:
        groups = ablation_groups[label]
        result = fit_eval(frame, random_labels, groups, grouped_new, skill_map,
                          text_matrix, text_by_id, SKILL_THRESHOLD, TEXT_DIM)
        cache[label] = result
        ablation_rows.append({'配置': label, '特征维度': result['特征维度'],
                              'validation_MAE': result['validation']['MAE'],
                              'validation_RMSE': result['validation']['RMSE'],
                              'validation_R2': result['validation']['R2'],
                              'test_MAE': result['test']['MAE'],
                              'test_RMSE': result['test']['RMSE'],
                              'test_R2': result['test']['R2'],
                              'n_train': result['n_train'], 'n_validation': result['n_validation'],
                              'n_test': result['n_test']})
        print(f"  消融 {label}: dim {result['特征维度']} | validation MAE "
              f"{result['validation']['MAE']} | test MAE {result['test']['MAE']}")
    ablation_table = pd.DataFrame(ablation_rows)

    increment_rows = []
    for name, without, with_ in [
            ('扩展敏感性增量（Safe-F 发布时间位置）', 'A+B+C+D+E', 'A+B+C+D+E+SafeF'),
            ('技能增量（D）', 'A+B+C', 'A+B+C+D'),
            ('文本语义增量（E）', 'A+B+C', 'A+B+C+E'),
            ('技能与文本增量（D+E）', 'A+B+C', 'A+B+C+D+E')]:
        for key, pred_key, frame_key in (('validation', 'valid_pred', 'valid_frame'),
                                         ('test', 'test_pred', 'test_frame')):
            truth = cache[without][frame_key][schema.SALARY_MID_FIELD].to_numpy('float64')
            stats = paired_bootstrap(truth, cache[without][pred_key], cache[with_][pred_key])
            increment_rows.append({
                '增量检验': name, '比较': f'{with_} 对 {without}', '数据子集': key,
                '不含该组 MAE': round(float(np.abs(truth - cache[without][pred_key]).mean()), 6),
                '含该组 MAE': round(float(np.abs(truth - cache[with_][pred_key]).mean()), 6),
                'ΔMAE（不含 − 含）': stats['ΔMAE'],
                '95% 置信区间下界': stats['CI95下界'],
                '95% 置信区间上界': stats['CI95上界'], '是否跨 0': stats['是否跨0'],
                'bootstrap 次数': BOOTSTRAP_ROUNDS})
            print(f"  增量 {name} / {key}: ΔMAE {stats['ΔMAE']} "
                  f"[{stats['CI95下界']}, {stats['CI95上界']}] 跨0={stats['是否跨0']}")
    increment_table = pd.DataFrame(increment_rows)

    # ---------------------------------------------------------- C 三种划分统一协议
    unified_rows, unified_cache = [], {}
    for label, labels in split_definitions.items():
        result = fit_eval(frame, labels, MODEL_FEATURE_GROUPS, grouped_new, skill_map,
                          text_matrix, text_by_id, SKILL_THRESHOLD, TEXT_DIM)
        unified_cache[label] = result
        old_row = pd.DataFrame(stage26_3['三种划分统一协议'])
        old = old_row[old_row['划分方式'] == label].iloc[0]
        unified_rows.append({
            '划分方式': label, 'Train n': result['n_train'],
            'Validation n': result['n_validation'], 'Test n': result['n_test'],
            '特征维度': result['特征维度'], '技能列数': result['技能列数'],
            'test MAE': result['test']['MAE'], 'test RMSE': result['test']['RMSE'],
            'test R²': result['test']['R2'],
            'validation MAE': result['validation']['MAE'],
            'validation RMSE': result['validation']['RMSE'],
            'validation R²': result['validation']['R2'],
            'Stage26.3 test MAE': float(old['test MAE']),
            'Stage26.3 test RMSE': float(old['test RMSE']),
            'Stage26.3 test R²': float(old['test R²']),
            'Stage26.3 特征维度': int(old['特征维度'])})
        print(f"  统一协议 {label}: dim {result['特征维度']} / test MAE "
              f"{result['test']['MAE']} / RMSE {result['test']['RMSE']} / R² {result['test']['R2']}")
    unified_table = pd.DataFrame(unified_rows)

    # ---------------------------------------------------------- D 模型比较
    comparison_rows, best_params_by_model = [], {}
    grids = {
        'Ridge': [{'alpha': alpha} for alpha in (0.1, 1.0, 10.0, 100.0)],
        'RandomForest': [{'n_estimators': 300, 'max_depth': depth, 'min_samples_leaf': leaf,
                          'max_features': 'sqrt'}
                         for depth in (None, 20) for leaf in (1, 5)],
        'CatBoost': [{'depth': 6, 'learning_rate': 0.05, 'iterations': 500},
                     {'depth': 8, 'learning_rate': 0.05, 'iterations': 500}],
        'LightGBM': [{'n_estimators': 400, 'learning_rate': 0.05, 'num_leaves': 31},
                     {'n_estimators': 400, 'learning_rate': 0.05, 'num_leaves': 63},
                     {'n_estimators': 600, 'learning_rate': 0.03, 'num_leaves': 31}]}
    for model_key, grid in grids.items():
        best = None
        for params in grid:
            result = fit_eval(frame, random_labels, MODEL_FEATURE_GROUPS, grouped_new,
                              skill_map, text_matrix, text_by_id, SKILL_THRESHOLD, TEXT_DIM,
                              model_key=model_key, params=params,
                              scale_numeric=model_key == 'Ridge')
            row = {'模型': model_key, '超参数': str(params),
                   'validation_MAE': result['validation']['MAE'],
                   'validation_RMSE': result['validation']['RMSE'],
                   'validation_R2': result['validation']['R2'],
                   'test_MAE': result['test']['MAE'], 'test_RMSE': result['test']['RMSE'],
                   'test_R2': result['test']['R2']}
            if best is None or row['validation_MAE'] < best['validation_MAE']:
                best = row
                best_params_by_model[model_key] = dict(params)
        comparison_rows.append(best)
    for key, strategy in (('Dummy（median）', 'median'), ('Dummy（mean）', 'mean')):
        result = fit_eval(frame, random_labels, MODEL_FEATURE_GROUPS, grouped_new, skill_map,
                          text_matrix, text_by_id, SKILL_THRESHOLD, TEXT_DIM,
                          model_key='Dummy', params={'strategy': strategy})
        comparison_rows.append({'模型': key, '超参数': f'strategy = {strategy}',
                                'validation_MAE': result['validation']['MAE'],
                                'validation_RMSE': result['validation']['RMSE'],
                                'validation_R2': result['validation']['R2'],
                                'test_MAE': result['test']['MAE'],
                                'test_RMSE': result['test']['RMSE'],
                                'test_R2': result['test']['R2']})
    comparison_table = pd.DataFrame(comparison_rows).sort_values('validation_MAE')

    # ---------------------------------------------------------- E 最终模型
    best_params = best_params_by_model['LightGBM']
    print('  主模型超参数：', best_params)
    is_test = random_labels.reindex(frame[schema.ID_FIELD]).to_numpy() == 'test'
    train_valid = frame.loc[~is_test].reset_index(drop=True)
    test_frame = frame.loc[is_test].reset_index(drop=True)
    assembler = build_assembler(MODEL_FEATURE_GROUPS, grouped_new, SKILL_THRESHOLD, TEXT_DIM)
    assembler.fit(train_valid, skill_map,
                  text_matrix[[text_by_id[job_id] for job_id in train_valid[schema.ID_FIELD]]])
    matrix_fit = assembler.transform(
        train_valid, skill_map,
        text_matrix[[text_by_id[job_id] for job_id in train_valid[schema.ID_FIELD]]])
    matrix_test = assembler.transform(
        test_frame, skill_map,
        text_matrix[[text_by_id[job_id] for job_id in test_frame[schema.ID_FIELD]]])
    final_model = model_training.make_model('LightGBM', best_params, random_state=SEED)
    final_model.fit(matrix_fit, train_valid[schema.SALARY_MID_FIELD].to_numpy('float64'))
    final_pred = np.asarray(final_model.predict(matrix_test), dtype='float64')
    truth_test = test_frame[schema.SALARY_MID_FIELD].to_numpy('float64')
    final_metrics = model_training.regression_metrics(truth_test, final_pred)
    train_only = unified_cache['Random Split']
    final_table = pd.DataFrame([
        {'模型与配置': 'LightGBM（仅训练集拟合，用于对照）',
         'test MAE': train_only['test']['MAE'], 'test RMSE': train_only['test']['RMSE'],
         'test R²': train_only['test']['R2'], 'n': train_only['n_test'],
         '特征维度': train_only['特征维度']},
        {'模型与配置': 'LightGBM（训练集 + 验证集重拟合，最终模型）',
         'test MAE': final_metrics['MAE'], 'test RMSE': final_metrics['RMSE'],
         'test R²': final_metrics['R2'], 'n': final_metrics['n'],
         '特征维度': int(assembler.schema.dimension)}])
    print(final_table.to_string(index=False))

    # ---------------------------------------------------------- F SHAP
    shap_feature_rows, shap_skill_rows, shap_note = [], [], {}
    shap_skill_table = pd.DataFrame([{'说明': 'NOT_RUN'}])
    shap_status = 'NOT_RUN'
    try:
        raw_values = final_model.predict(matrix_test, pred_contrib=True)
        if hasattr(raw_values, 'toarray'):
            raw_values = raw_values.toarray()
        raw_values = np.asarray(raw_values, dtype='float64')
        base_value = float(raw_values[:, -1].mean())
        values = raw_values[:, :-1]
        additive_error = float(np.abs(values.sum(axis=1) + base_value - final_pred).max())
        mean_abs = np.abs(values).mean(axis=0)
        names = assembler.feature_names()
        if len(names) != values.shape[1]:
            raise ValueError(f'特征名与 SHAP 列数不一致：{len(names)} vs {values.shape[1]}')
        total = float(mean_abs.sum())
        dense_test = np.asarray(matrix_test.todense())
        for position in np.argsort(-mean_abs)[:15]:
            column = dense_test[:, position]
            present = column > 0
            shap_feature_rows.append({
                '特征': names[position], '平均绝对SHAP值': round(float(mean_abs[position]), 6),
                '重要性占比': round(float(mean_abs[position] / total), 6),
                '非零 / 命中样本数（测试集）': int(present.sum()),
                '非零样本平均SHAP': round(float(values[present, position].mean()), 6)
                if present.any() else None})
        skill_counts = {skill: sum(1 for job_id in job_ids
                                   if skill in skill_map.get(job_id, ()))
                        for skill in assembler.schema.skill_columns}
        for position, name in enumerate(names):
            if not name.startswith('技能='):
                continue
            skill = name.split('=', 1)[1]
            column = dense_test[:, position]
            present = column > 0
            shap_skill_rows.append({
                '技能': skill, '建模样本岗位数（频率）': int(skill_counts.get(skill, 0)),
                '测试集命中岗位数': int(present.sum()),
                '技能存在时平均SHAP': round(float(values[present, position].mean()), 6)
                if present.any() else None,
                '平均绝对SHAP值': round(float(mean_abs[position]), 6)})
        shap_skill_table = pd.DataFrame(shap_skill_rows).sort_values(
            '平均绝对SHAP值', ascending=False)
        shap_note = {'基准值（元/天）': round(base_value, 6),
                     '加性一致性最大误差': additive_error, '特征数': len(names),
                     '技能列数': len(assembler.schema.skill_columns)}
        shap_status = 'OK'
        print(f'  SHAP：基准值 {base_value:.6f} / 加性一致性最大误差 {additive_error:.3e}')
    except Exception as error:  # noqa: BLE001
        shap_skill_table = pd.DataFrame([{'说明': f'NOT_RUN（{type(error).__name__}: {error}）'}])
        print(f'  SHAP 解释未运行：{type(error).__name__}: {error}')

    # ---------------------------------------------------------- G 分组预测误差
    boundaries = [float(np.quantile(truth_test, value)) for value in (0.25, 0.5, 0.75)]
    error_rows = [{'维度': '总体（测试集）', '分组': '—', 'n': len(truth_test),
                   'MAE（元/天）': round(float(np.abs(truth_test - final_pred).mean()), 6),
                   'RMSE（元/天）': round(float(np.sqrt(np.mean(
                       (truth_test - final_pred) ** 2))), 6)}]
    quartile_labels = ['第一四分位组（最低四分位）', '第二四分位组', '第三四分位组',
                       '第四四分位组（最高四分位）']
    lower = float('-inf')
    for label, upper in zip(quartile_labels, boundaries + [float('inf')]):
        mask = (truth_test > lower) & (truth_test <= upper)
        if mask.sum():
            error_rows.append({
                '维度': '薪资区间（四分位）', '分组': label, 'n': int(mask.sum()),
                'MAE（元/天）': round(float(np.abs(truth_test[mask]
                                                - final_pred[mask]).mean()), 6),
                'RMSE（元/天）': round(float(np.sqrt(np.mean(
                    (truth_test[mask] - final_pred[mask]) ** 2))), 6)})
        lower = upper
    categories_test = np.array([set(items) if isinstance(items, (list, tuple, np.ndarray))
                                else set() for items in test_frame['岗位大类集合']], dtype=object)
    category_rows = []
    for category in sorted({item for items in categories_test for item in items}):
        mask = np.array([category in items for items in categories_test])
        if mask.sum():
            category_rows.append({
                '岗位大类': category, 'n': int(mask.sum()),
                'MAE（元/天）': round(float(np.abs(truth_test[mask]
                                                - final_pred[mask]).mean()), 6),
                'RMSE（元/天）': round(float(np.sqrt(np.mean(
                    (truth_test[mask] - final_pred[mask]) ** 2))), 6)})
    quartile_table = pd.DataFrame(error_rows)
    category_error_table = pd.DataFrame(category_rows).sort_values('MAE（元/天）')

    # ---------------------------------------------------------- H 多种子 + 公司分组
    seed_rows = []
    for seed in SEEDS:
        seed_labels = pd.Series(
            model_training.build_splits(frame, random_state=seed)['split'].to_numpy(),
            index=job_ids)
        for model_key, params in [('LightGBM', LIGHTGBM_PARAMS), ('CatBoost', CATBOOST_PARAMS)]:
            result = fit_eval(frame, seed_labels, MODEL_FEATURE_GROUPS, grouped_new,
                              skill_map, text_matrix, text_by_id, SKILL_THRESHOLD, TEXT_DIM,
                              model_key=model_key, params=params, random_state=seed)
            seed_rows.append({'随机种子': seed, '模型': model_key,
                              'validation MAE': result['validation']['MAE'],
                              'test MAE': result['test']['MAE'],
                              'test RMSE': result['test']['RMSE'],
                              'test R²': result['test']['R2']})
            print(f"  多种子 {seed} {model_key}: validation MAE {result['validation']['MAE']}")
    seed_table = pd.DataFrame(seed_rows)
    seed_summary = seed_table.groupby('模型').agg(
        validation_MAE均值=('validation MAE', 'mean'),
        validation_MAE标准差=('validation MAE', 'std'),
        test_MAE均值=('test MAE', 'mean'),
        test_MAE标准差=('test MAE', 'std')).round(6).reset_index()

    repeat_rows = []
    company_groups = splits.set_index(schema.ID_FIELD)['company_entity_id'].reindex(
        job_ids.to_numpy()).to_numpy()
    from sklearn.model_selection import GroupShuffleSplit  # noqa: PLC0415

    for seed in SEEDS:
        holder = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
        train_index, rest_index = next(holder.split(job_ids.to_numpy().reshape(-1, 1),
                                                    groups=company_groups))
        holder2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        valid_rel, test_rel = next(holder2.split(
            job_ids.to_numpy()[rest_index].reshape(-1, 1), groups=company_groups[rest_index]))
        values = np.empty(len(frame), dtype=object)
        values[train_index] = 'train'
        values[rest_index[valid_rel]] = 'validation'
        values[rest_index[test_rel]] = 'test'
        result = fit_eval(frame, pd.Series(values, index=job_ids), MODEL_FEATURE_GROUPS,
                          grouped_new, skill_map, text_matrix, text_by_id, SKILL_THRESHOLD,
                          TEXT_DIM)
        repeat_rows.append({'随机种子': seed, '特征维度': result['特征维度'],
                            'Train n': result['n_train'], 'Validation n': result['n_validation'],
                            'Test n': result['n_test'], 'test MAE': result['test']['MAE'],
                            'test RMSE': result['test']['RMSE'], 'test R²': result['test']['R2']})
        print(f"  公司分组第 {seed} 次划分: test MAE {result['test']['MAE']}")
    repeat_table = pd.DataFrame(repeat_rows)
    repeat_summary = pd.DataFrame([
        {'指标': 'test MAE 均值', '取值': round(float(repeat_table['test MAE'].mean()), 6)},
        {'指标': 'test MAE 标准差', '取值': round(float(repeat_table['test MAE'].std()), 6)},
        {'指标': 'test MAE 最小', '取值': round(float(repeat_table['test MAE'].min()), 6)},
        {'指标': 'test MAE 最大', '取值': round(float(repeat_table['test MAE'].max()), 6)},
        {'指标': 'test RMSE 均值', '取值': round(float(repeat_table['test RMSE'].mean()), 6)},
        {'指标': 'test RMSE 标准差', '取值': round(float(repeat_table['test RMSE'].std()), 6)},
        {'指标': 'test R² 均值', '取值': round(float(repeat_table['test R²'].mean()), 6)},
        {'指标': 'test R² 标准差', '取值': round(float(repeat_table['test R²'].std()), 6)}])

    # ---------------------------------------------------------- I 稳健性
    robustness_rows = []
    train_only_frame = frame.loc[random_labels.reindex(frame[schema.ID_FIELD]).to_numpy()
                                 == 'train'].reset_index(drop=True)
    base_assembler = build_assembler(MODEL_FEATURE_GROUPS, grouped_new, SKILL_THRESHOLD,
                                     TEXT_DIM)
    base_assembler.fit(train_only_frame, skill_map,
                       text_matrix[[text_by_id[job_id]
                                    for job_id in train_only_frame[schema.ID_FIELD]]])
    base_matrix_train = base_assembler.transform(
        train_only_frame, skill_map,
        text_matrix[[text_by_id[job_id] for job_id in train_only_frame[schema.ID_FIELD]]])
    base_matrix_test = base_assembler.transform(
        test_frame, skill_map,
        text_matrix[[text_by_id[job_id] for job_id in test_frame[schema.ID_FIELD]]])
    base_model = model_training.make_model('LightGBM', best_params, random_state=SEED)
    base_model.fit(base_matrix_train,
                   train_only_frame[schema.SALARY_MID_FIELD].to_numpy('float64'))
    base_pred = np.asarray(base_model.predict(base_matrix_test), dtype='float64')
    base_metrics = model_training.regression_metrics(truth_test, base_pred)
    robustness_rows.append({'检查项': '对照基准', '统计范围': '真实薪资，主统计范围未缩尾',
                            'MAE': base_metrics['MAE'], 'RMSE': base_metrics['RMSE'],
                            'R²': base_metrics['R2'],
                            '说明': 'A+B+C+D+E，仅用训练集拟合'})
    lower_bound = float(np.quantile(train_only_frame[schema.SALARY_MID_FIELD], 0.01))
    upper_bound = float(np.quantile(train_only_frame[schema.SALARY_MID_FIELD], 0.99))
    winsor_model = model_training.make_model('LightGBM', best_params, random_state=SEED)
    winsor_model.fit(base_matrix_train, train_only_frame[schema.SALARY_MID_FIELD]
                     .clip(lower_bound, upper_bound).to_numpy('float64'))
    winsor_pred = np.asarray(winsor_model.predict(base_matrix_test), dtype='float64')
    winsor_metrics = model_training.regression_metrics(truth_test, winsor_pred)
    robustness_rows.append({
        '检查项': '极端值敏感性', '统计范围': '训练目标 winsorize 1%/99%',
        'MAE': winsor_metrics['MAE'], 'RMSE': winsor_metrics['RMSE'],
        'R²': winsor_metrics['R2'],
        '说明': f'训练目标裁剪至 {lower_bound:.0f} 至 {upper_bound:.0f} 元/天，'
                f'评估仍用真实测试薪资'})
    for name, field in [('目标定义 薪资下限', schema.SALARY_MIN_FIELD),
                        ('目标定义 薪资上限', schema.SALARY_MAX_FIELD)]:
        model = model_training.make_model('LightGBM', best_params, random_state=SEED)
        model.fit(base_matrix_train, train_only_frame[field].to_numpy('float64'))
        metrics = model_training.regression_metrics(
            truth_test, np.asarray(model.predict(base_matrix_test), dtype='float64'))
        robustness_rows.append({'检查项': name, '统计范围': field, 'MAE': metrics['MAE'],
                                'RMSE': metrics['RMSE'], 'R²': metrics['R2'],
                                '说明': '仅比较趋势，不作为新的主任务'})
    gain_base = np.asarray(
        base_model.booster_.feature_importance(importance_type='gain')
        if hasattr(base_model, 'booster_') else base_model.feature_importances_, dtype='float64')
    gain_winsor = np.asarray(
        winsor_model.booster_.feature_importance(importance_type='gain')
        if hasattr(winsor_model, 'booster_') else winsor_model.feature_importances_,
        dtype='float64')
    from scipy import stats as scipy_stats  # noqa: PLC0415

    gain_spearman = float(scipy_stats.spearmanr(gain_base, gain_winsor).statistic)
    robustness_rows.append({
        '检查项': '特征重要性方向', '统计范围': 'winsorize 与主模型 gain 排名的 Spearman',
        'MAE': None, 'RMSE': None, 'R²': None,
        '说明': f'{gain_spearman:.6f}，共有特征 {len(base_assembler.feature_names())} 个'})
    shap_spearman = None
    try:
        base_shap = np.abs(ablation_shap.tree_shap_values(base_model,
                                                         base_matrix_test)).mean(axis=0)
        correlations = []
        for seed in (7, 2024):
            seed_labels = pd.Series(
                model_training.build_splits(frame, random_state=seed)['split'].to_numpy(),
                index=job_ids)
            result = fit_eval(frame, seed_labels, MODEL_FEATURE_GROUPS, grouped_new,
                              skill_map, text_matrix, text_by_id, SKILL_THRESHOLD,
                              TEXT_DIM, random_state=seed)
            names_other = result['assembler'].feature_names()
            matrix_other = result['assembler'].transform(
                result['test_frame'], skill_map,
                text_matrix[[text_by_id[job_id]
                             for job_id in result['test_frame'][schema.ID_FIELD]]])
            other_shap = np.abs(ablation_shap.tree_shap_values(
                result['model'], matrix_other)).mean(axis=0)
            position = {name: index for index, name in enumerate(names_other)}
            left, right = [], []
            for index, name in enumerate(base_assembler.feature_names()):
                if name in position:
                    left.append(base_shap[index])
                    right.append(other_shap[position[name]])
            correlations.append(float(scipy_stats.spearmanr(left, right).statistic))
        shap_spearman = correlations
        robustness_rows.append({
            '检查项': '解释稳定性',
            '统计范围': '三种随机种子 42、7、2024 与主模型平均绝对SHAP值排名的 Spearman',
            'MAE': None, 'RMSE': None, 'R²': None,
            '说明': '；'.join(f'{value:.6f}' for value in correlations)
                    + f'（共同特征 {len(base_assembler.feature_names())} 个）'})
    except Exception as error:  # noqa: BLE001
        print(f'  SHAP 稳定性未运行：{type(error).__name__}: {error}')
    table_robustness = pd.DataFrame(robustness_rows)

    # ---------------------------------------------------------- J 图件
    episodes = pd.read_parquet(EPISODE_PATH)
    duration_figure = figure_duration_strict(episodes)
    ablation_figure = figure_ablation_new(ablation_table)
    generalize_figure = figure_generalization_new(unified_table)
    duration_unique = episodes.drop_duplicates(subset=['intern_id', 'episode_id_strict'])
    duration_old = duration_block(episodes['final_observed_planned_duration_days'])
    duration_new = duration_block(
        duration_unique['final_observed_planned_duration_days'])

    # ---------------------------------------------------------- K 中位数回归
    from src import eda_analysis  # noqa: PLC0415

    analysis_cert = eda_analysis.attach_company_certification(
        analysis[[schema.ID_FIELD, '工作城市', '学历要求', '每周到岗要求', '实习时长要求',
                  '公司规模', '公司性质', '所属行业', '岗位大类集合', '岗位细分类集合']],
        entity)
    median_frame = analysis_cert.merge(
        model_frame[[schema.ID_FIELD, schema.SALARY_MID_FIELD]], on=schema.ID_FIELD, how='inner')
    design, design_names, references = build_median_design(median_frame)
    target = median_frame[schema.SALARY_MID_FIELD].to_numpy('float64')
    print(f'  中位数回归：n = {len(median_frame):,}，设计矩阵 {design.shape[1]} 列，'
          f'开始 {MEDIAN_BOOTSTRAP_ROUNDS} 轮 bootstrap')
    point = _median_fit(design, target)
    bootstrap = median_bootstrap(design, target, MEDIAN_BOOTSTRAP_ROUNDS, SEED)
    lower = np.percentile(bootstrap, 2.5, axis=0)
    upper = np.percentile(bootstrap, 97.5, axis=0)
    median_table = pd.DataFrame({
        '变量': design_names, '系数': np.round(point, 6),
        'bootstrap_CI95_下界': np.round(lower, 6),
        'bootstrap_CI95_上界': np.round(upper, 6),
        '方向': np.where(point > 0, '正向', '负向'),
        'bootstrap 区间是否跨 0': np.where((lower > 0) | (upper < 0), '否', '是')})
    median_table['变量类型'] = np.where(
        median_table['变量'].eq('截距'), '截距',
        np.where(median_table['变量'].str.split('=').str[0].isin(MEDIAN_CATEGORICAL),
                 '单值互斥类别（相对参照类别）', '多值重叠类别（命中该类别 − 未命中该类别）'))
    representative = representative_median_rows(median_table)
    median_meta = pd.DataFrame([
        {'项目': '因变量', '数值': f'{schema.SALARY_MID_FIELD}（元/天），分位数 0.5'},
        {'项目': '实现',
         '数值': 'sklearn.linear_model.QuantileRegressor（quantile=0.5、alpha=0.0、'
                 'fit_intercept=False、solver="highs-ipm"，设计矩阵首列显式包含截距）'},
        {'项目': '样本量', '数值': int(len(median_frame))},
        {'项目': '设计矩阵列数', '数值': int(design.shape[1])},
        {'项目': '单值互斥类别解释',
         '数值': '工作城市、学历要求、每周到岗要求、实习时长要求、公司规模、公司性质、'
                 '所属行业与公司认证类别为互斥类别，系数解释为相对参照类别的条件中位数差异；'
                 '参照类别为 ' + '；'.join(f'{k}={v}' for k, v in references.items())},
        {'项目': '多值重叠类别解释',
         '数值': '岗位大类集合与岗位细分类集合为多值重叠变量，各取值以命中指示列进入设计矩阵，'
                 '系数解释为命中该类别与未命中该类别的条件中位数差异，不设参照类别，'
                 '不与互斥类别的参照解释混用'},
        {'项目': 'bootstrap 次数', '数值': f'{MEDIAN_BOOTSTRAP_ROUNDS} 次（随机种子 {SEED}）'},
        {'项目': '区间作用',
         '数值': 'bootstrap 区间只作系数稳定性参考，不作为逐系数显著性判定；'
                 '本文不报告“系数显著项占比”一类全局判定，也不对全部系数做显著性排名'},
        {'项目': '纳入变量', '数值': '、'.join(MEDIAN_CATEGORICAL + MEDIAN_MULTI)},
        {'项目': '不纳入变量', '数值': '4,000 余个福利标签、全部技能特征、文本语义 SVD 成分'
                                 '与高维公司标识'},
        {'项目': '解释边界', '数值': '观察性多变量关联分析，不得表述为因果效应'},
    ])

    # ---------------------------------------------------------- L 输出
    removal_table = pd.DataFrame([
        {'特征': column,
         '类别': (GOVERNANCE_FEATURES[column][0] if column in GOVERNANCE_FEATURES
                else 'Stage26.3 已判定（后验版本 / 连续冗余 / 近常量）'),
         '决策': '移出正式模型',
         '理由': (GOVERNANCE_FEATURES[column][1] if column in GOVERNANCE_FEATURES
                else REMOVED_STAGE26_3[column])} for column in REMOVED_FINAL])
    write_excel(TABLE_FEATURE_AUDIT, {
        '01_正式模型输入字段审计': audit_table,
        '02_移除字段清单': removal_table,
        '03_特征组维度对照': dimension_table,
    })
    write_excel(TABLE_METRICS, {
        '01_三种划分统一协议': unified_table,
        '02_消融五配置': ablation_table,
        '03_消融增量bootstrap': increment_table,
        '04_模型比较': comparison_table,
        '05_最终模型测试集': final_table,
        '06_分组预测误差四分位': quartile_table,
        '07_岗位大类预测误差': category_error_table,
        '08_多种子明细': seed_table,
        '09_多种子汇总': seed_summary,
        '10_公司分组多次划分': repeat_table,
        '11_公司分组汇总': repeat_summary,
        '12_稳健性对照': table_robustness,
        '13_主模型SHAP特征': pd.DataFrame(shap_feature_rows),
        '14_技能SHAP': shap_skill_table,
        '15_图4-4新旧数据对照': pd.DataFrame([
            {'统计量': key, '旧图（候选段层，n = %d）' % duration_old['n']: duration_old[key],
             '重算（严格统计范围周期层，n = %d）' % duration_new['n']: duration_new[key]}
            for key in ('n', 'P10', 'P25', 'Median', 'P75', 'P90', 'IQR', 'max')]),
    })
    write_excel(TABLE_MEDIAN, {
        '01_代表性系数': representative,
        '02_全系数': median_table,
        '03_统计范围说明': median_meta,
    })

    payload = {
        '生成时间': time.strftime('%Y-%m-%d %H:%M:%S'),
        '输入': {'建模样本': int(len(model_frame)),
                 '移除的数据治理元数据': list(GOVERNANCE_FEATURES),
                 '移除的 Stage26.3 字段': list(REMOVED_STAGE26_3),
                 '正式模型': 'LightGBM 400/0.05/63', '技能频次阈值': SKILL_THRESHOLD,
                 '文本语义 SVD': TEXT_DIM, '随机种子': SEED,
                 '主特征体系': 'A+B+C+D+E', '扩展敏感性': 'Safe-F（publish_month、publish_weekday）'},
        '字段审计': audit_table.to_dict('records'),
        '特征组维度对照': dimension_table.to_dict('records'),
        '三种划分统一协议': unified_table.to_dict('records'),
        '消融五配置': ablation_table.to_dict('records'),
        '消融增量': increment_table.to_dict('records'),
        '模型比较': comparison_table.to_dict('records'),
        '最终模型测试集': final_table.to_dict('records'),
        '分组预测误差（四分位）': quartile_table.to_dict('records'),
        '岗位大类预测误差': category_error_table.to_dict('records'),
        '多种子汇总': seed_summary.to_dict('records'),
        '公司分组多次划分汇总': repeat_summary.to_dict('records'),
        '稳健性对照': table_robustness.to_dict('records'),
        '主模型SHAP特征': shap_feature_rows,
        '技能SHAP': shap_skill_table.to_dict('records'),
        'SHAP解释摘要': shap_note, 'SHAP稳定性': shap_status,
        'SHAP排名Spearman': shap_spearman,
        '中位数回归': {'统计范围': median_meta.to_dict('records'),
                   '代表性系数': representative.to_dict('records'),
                   'bootstrap 次数': MEDIAN_BOOTSTRAP_ROUNDS,
                   '设计矩阵列数': int(design.shape[1])},
        '图4-4数据对照': {'旧图（候选段层）': duration_old, '严格统计范围周期层': duration_new},
        '图件': {'图4-4': duration_figure.get('png_path'),
                 '图8-1': ablation_figure.get('png_path'),
                 '图8-3': generalize_figure.get('png_path')},
        '对照 Stage26.3': {
            '旧维度 A+B+C+D+E': 308, '旧维度 +SafeF': 310,
            '旧 Random test MAE': 36.451863, '旧 Company test MAE': 52.093534,
            '旧 Temporal test MAE': 32.132572,
            '旧最终模型': {'MAE': 35.28104, 'RMSE': 64.783455, 'R2': 0.582546}},
        '运行耗时秒': round(time.time() - started, 3),
    }
    dump_json(METRICS_PATH, payload)

    manifest_after = project_manifest()
    changed = sorted(key for key in manifest_before
                     if manifest_before.get(key) != manifest_after.get(key))
    added = sorted(key for key in manifest_after if key not in manifest_before)
    removed = sorted(key for key in manifest_before if key not in manifest_after)
    print(f'既有文件被修改：{len(changed)} 个；新增：{len(added)} 个；删除：{len(removed)} 个')
    for key in changed[:40]:
        print(f'  MODIFIED {key}')
    payload['文件完整性'] = {'既有文件被修改': changed, '新增文件': added, '删除文件': removed,
                        '既有文件被修改数': len(changed), '新增文件数': len(added)}
    dump_json(METRICS_PATH, payload)
    print('完成，耗时 %.1f s' % (time.time() - started))
    return 0


# ============================================================================
# 文本审计
# ============================================================================
AUDIT_PATTERNS = [
    ('研究问题 Q 简称', r'(?<![A-Za-z])Q[1-4](?![0-9A-Za-z])'),
    ('需要说明的是', r'需要说明的是'), ('需要强调的是', r'需要强调的是'),
    ('必须明确', r'必须明确'), ('值得注意的是', r'值得注意的是'),
    ('需要重申的是', r'需要重申的是'), ('这里需要指出', r'这里需要指出'),
    ('可以看到', r'可以看到'), ('可以发现', r'可以发现'),
    ('由此可以看出', r'由此可以看出'), ('综上可以看出', r'综上可以看出'),
    ('这意味着', r'这意味着'), ('不是……而是', r'不是[^。；\n]{0,40}而是'),
    ('并非……而是', r'并非[^。；\n]{0,40}而是'),
    ('中文引号 左', r'“'), ('中文引号 右', r'”'),
    ('中文括号 左', r'（'), ('中文括号 右', r'）'),
    ('破折号 ——', r'——'), ('短横 —', r'—(?![—])'),
    ('行首无序列表', r'(?m)^\s*[-*•]\s+'),
    ('开发语言 Stage2x', r'Stage2[0-9](\.[0-9])?'),
    ('加粗短句 方法目的', r'\*\*方法目的。\*\*'),
    ('加粗短句 变量说明', r'\*\*变量说明。\*\*'),
    ('加粗短句 本文中如何使用', r'\*\*本文中如何使用。\*\*'),
    ('加粗短句 方法机制', r'\*\*方法机制。\*\*'),
    ('加粗短句 参数说明', r'\*\*参数说明。\*\*'),
    ('章首机械总起段', r'(?m)^本章(?:说明|主要|首先|从|交代|先|以|给出|对比|在|完成|介绍|按)'),
    ('长英文变量', r'Duration_final|FinalDeadline|InitialDeadline|ReopenGap|'
                r'initial_observed_deadline|final_observed_deadline'),
    ('测试集一次评估表述', r'测试集只在最终配置锁定后评估一次'),
    ('61.5% 显著表述', r'61\.5%|61\.5 ?%'),
    ('旧维度 320/322', r'320 维|322 维'),
    ('保留特征名 mean\\|SHAP\\|', r'mean\|SHAP\|'),
    ('presence 统计方式', r'presence 统计方式'),
]
COUNTED_KEYS = [label for label, _ in AUDIT_PATTERNS
                if label not in ('章首机械总起段', '长英文变量', '测试集一次评估表述',
                                 '61.5% 显著表述', '旧维度 320/322',
                                 '保留特征名 mean|SHAP|', 'presence 统计方式')]


def strip_html_comments(text: str) -> str:
    return re.sub(r'<!--.*?-->', '', text, flags=re.S)


def han_count(text: str) -> int:
    return len(re.findall(r'[\u4e00-\u9fff]', text))


def visible_body(text: str) -> str:
    body = strip_html_comments(text)
    body = re.sub(r'(?m)^\s*>?\s*【插[图表][^\n]*】\s*$', '', body)
    body = re.sub(r'(?m)^\s*\$\$.*?\$\$\s*$', '', body)
    body = re.sub(r'(?m)^\s*[|>```].*$', '', body)
    return body


def chapter_summary_block() -> pd.DataFrame:
    rows = []
    for name in CHAPTERS:
        text = visible_body((SOURCE_DIR / name).read_text(encoding='utf-8'))
        lines = text.splitlines()
        index = None
        for position, line in enumerate(lines):
            if re.match(r'^#{2,4}\s*\d*\.?\d*\s*本章小结', line.strip()):
                index = position
                break
        if index is None:
            continue
        block = []
        for line in lines[index + 1:]:
            if re.match(r'^#{2,4}\s', line.strip()):
                break
            if line.strip():
                block.append(line.strip())
        joined = '\n'.join(block)
        sentences = [item for item in re.split(r'[。！？]', joined) if item.strip()]
        rows.append({'文件': name, '小结段落数': len(block), '小结句数': len(sentences),
                     '小结汉字数': han_count(joined)})
    return pd.DataFrame(rows)


def chapter_lead_paragraph() -> pd.DataFrame:
    rows = []
    for name in CHAPTERS:
        text = strip_html_comments((SOURCE_DIR / name).read_text(encoding='utf-8'))
        lines = text.splitlines()
        for position, line in enumerate(lines):
            if re.match(r'^##\s+\d+\s', line.strip()):
                for follow in lines[position + 1:]:
                    if not follow.strip():
                        continue
                    rows.append({'文件': name, '标题': line.strip(),
                                 '标题后首段': follow.strip()[:160]})
                    break
                break
    return pd.DataFrame(rows)


def mechanical_lead_count() -> int:
    leads = chapter_lead_paragraph()
    return int(sum(1 for value in leads['标题后首段']
                   if not str(value).strip().startswith('#')))


def audit_text() -> dict:
    counts, hits = {}, []
    for name in CHAPTERS:
        raw = (SOURCE_DIR / name).read_text(encoding='utf-8')
        body = visible_body(raw)
        for label, pattern in AUDIT_PATTERNS:
            for number, line in enumerate(raw.splitlines(), start=1):
                for match in re.finditer(pattern, line):
                    hits.append({'文件': name, '行号': number, '命中类型': label,
                                 '原文片段': line[max(match.start() - 30, 0):
                                              match.end() + 30].strip()})
            counts[label] = counts.get(label, 0) + len(re.findall(pattern, body))
    chapters = pd.DataFrame([
        {'文件': name,
         '汉字数（可见正文）': han_count(visible_body(
             (SOURCE_DIR / name).read_text(encoding='utf-8'))),
         '字节数': int((SOURCE_DIR / name).stat().st_size)} for name in CHAPTERS])
    return {'counts': counts, 'hits': pd.DataFrame(hits), 'chapters': chapters,
            '总计汉字数（可见正文）': int(chapters['汉字数（可见正文）'].sum())}


def table_shape(file_name: str, caption_keyword: str) -> dict:
    """定位含指定关键词的表题前一张表，返回行列数与表头列名。"""
    text = strip_html_comments((SOURCE_DIR / file_name).read_text(encoding='utf-8'))
    lines = text.splitlines()
    target = None
    for position, line in enumerate(lines):
        if caption_keyword in line and line.strip().startswith('>'):
            target = position
            break
    if target is None:
        return {'行列数': '—', '表头': '—'}
    rows = []
    for line in reversed(lines[:target]):
        if line.strip().startswith('|'):
            rows.append(line.strip())
        elif rows:
            break
    rows.reverse()
    if not rows:
        return {'行列数': '—', '表头': '—'}
    header = [cell.strip() for cell in rows[0].strip('|').split('|')]
    return {'行列数': f'{len(rows) - 2} 行 × {len(header)} 列', '表头': ' / '.join(header)}


TABLES_TO_TRACK = {
    '表 3-1': ('03_数据获取与预处理.md', '表 3-1'),
    '表 5-3': ('05_实习岗位薪资影响因素分析.md', '表 5-3'),
    '表 5-4': ('05_实习岗位薪资影响因素分析.md', '表 5-4'),
    '表 5-5': ('05_实习岗位薪资影响因素分析.md', '表 5-5'),
    '表 7-1': ('07_薪资预测模型构建与结果分析.md', '表 7-1'),
    '表 8-1': ('08_模型稳健性与解释.md', '表 8-1'),
    '表 8-5': ('08_模型稳健性与解释.md', '表 8-5'),
}


def run_baseline() -> int:
    audit = audit_text()
    payload = {'记录时间': time.strftime('%Y-%m-%d %H:%M:%S'),
               '命中计数': audit['counts'],
               '总计汉字数（可见正文）': audit['总计汉字数（可见正文）'],
               '各章汉字数': audit['chapters'].to_dict('records'),
               '各章标题后首段': chapter_lead_paragraph().to_dict('records'),
               '章首机械总起段数量': mechanical_lead_count(),
               '各章小结篇幅': chapter_summary_block().to_dict('records'),
               '3.2 数据采集方案': section_length(SOURCE_DIR, visible_body, han_count, '03_数据获取与预处理.md',
                                            r'^#{2,4}\s*3\.2', r'^#{2,4}\s*3\.3'),
               '重点表格改前': {key: table_shape(*value)
                            for key, value in TABLES_TO_TRACK.items()}}
    dump_json(TEXT_BASELINE_PATH, payload)
    print(json.dumps(payload['命中计数'], ensure_ascii=False, indent=2))
    return 0


def run_textaudit() -> int:
    audit = audit_text()
    baseline = json.loads(TEXT_BASELINE_PATH.read_text(encoding='utf-8')) \
        if TEXT_BASELINE_PATH.is_file() else {'命中计数': {}, '各章小结篇幅': [],
                                              '各章标题后首段': []}
    comparison = pd.DataFrame([
        {'项目': label, '改前': int(baseline['命中计数'].get(label, 0)),
         '改后': int(audit['counts'].get(label, 0)),
         '变化': int(audit['counts'].get(label, 0)) - int(baseline['命中计数'].get(label, 0))}
        for label in COUNTED_KEYS])
    comparison = pd.concat([comparison, pd.DataFrame([
        {'项目': '可见正文汉字数合计',
         '改前': int(baseline.get('总计汉字数（可见正文）', 0)),
         '改后': int(audit['总计汉字数（可见正文）']),
         '变化': int(audit['总计汉字数（可见正文）'])
                 - int(baseline.get('总计汉字数（可见正文）', 0))},
        {'项目': '章首机械总起段数量', '改前': int(baseline.get('章首机械总起段数量', 0)),
         '改后': int(mechanical_lead_count()),
         '变化': int(mechanical_lead_count()) - int(baseline.get('章首机械总起段数量', 0))},
        {'项目': '长英文变量（Duration_final 等）',
         '改前': int(baseline['命中计数'].get('长英文变量', 0)),
         '改后': int(audit['counts'].get('长英文变量', 0)),
         '变化': int(audit['counts'].get('长英文变量', 0))
                 - int(baseline['命中计数'].get('长英文变量', 0))},
        {'项目': '测试集一次评估表述',
         '改前': int(baseline['命中计数'].get('测试集一次评估表述', 0)),
         '改后': int(audit['counts'].get('测试集一次评估表述', 0)),
         '变化': int(audit['counts'].get('测试集一次评估表述', 0))
                 - int(baseline['命中计数'].get('测试集一次评估表述', 0))},
    ])], ignore_index=True)

    summary_before = pd.DataFrame(baseline.get('各章小结篇幅', []))
    summaries = chapter_summary_block()
    summary_compare = summaries.merge(
        summary_before if len(summary_before) else pd.DataFrame(columns=['文件']),
        on='文件', how='left', suffixes=('（改后）', '（改前）'))
    leads_after = chapter_lead_paragraph()
    leads_before = pd.DataFrame(baseline.get('各章标题后首段', []))
    leads_compare = leads_after.merge(
        leads_before[['文件', '标题后首段']] if len(leads_before) else
        pd.DataFrame(columns=['文件', '标题后首段']),
        on='文件', how='left', suffixes=('（改后）', '（改前）'))
    shape_rows = []
    for key, (file_name, keyword) in TABLES_TO_TRACK.items():
        before = baseline.get('重点表格改前', {}).get(key, {})
        after = table_shape(file_name, keyword)
        shape_rows.append({'表号': key, '改前行列数': before.get('行列数', '—'),
                           '改后行列数': after['行列数'],
                           '改前表头': before.get('表头', '—'), '改后表头': after['表头']})
    section = pd.DataFrame([
        {'项目': '3.2 数据采集方案（改前）', **baseline.get('3.2 数据采集方案', {})},
        {'项目': '3.2 数据采集方案（改后）',
         **section_length(SOURCE_DIR, visible_body, han_count, '03_数据获取与预处理.md', r'^#{2,4}\s*3\.2', r'^#{2,4}\s*3\.3')}])
    write_excel(TABLE_TEXT_AUDIT, {
        '01_文风命中数对照': comparison,
        '02_各章小结篇幅对照': summary_compare,
        '03_章首段落对照': leads_compare,
        '04_3.2压缩对照': section,
        '05_重点表格结构对照': pd.DataFrame(shape_rows),
        '06_命中明细': audit['hits'] if len(audit['hits']) else pd.DataFrame([{'说明': '无命中'}]),
        '07_各章汉字数': audit['chapters']})
    payload = {'生成时间': time.strftime('%Y-%m-%d %H:%M:%S'),
               '文风命中数对照': comparison.to_dict('records'),
               '各章小结篇幅（改后）': summaries.to_dict('records'),
               '重点表格结构对照': shape_rows,
               '3.2 压缩对照': section.to_dict('records'),
               '各章汉字数': audit['chapters'].to_dict('records'),
               '总计汉字数（可见正文）': audit['总计汉字数（可见正文）']}
    dump_json(TEXT_METRICS_PATH, payload)
    print(comparison.to_string(index=False))
    print(pd.DataFrame(shape_rows).to_string(index=False))
    return 0


def figure_split_compare(unified: pd.DataFrame) -> dict:
    """图 8-2：随机划分与按公司分组划分的测试集误差与 R² 对照。"""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    block = unified[unified['划分方式'].isin(['Random Split', 'Company Group Split'])]
    labels = ['随机划分', '按公司分组划分']
    positions = np.arange(len(block))
    fig, axes = plt.subplots(2, 1, figsize=(5.85, 6.0))
    ax = axes[0]
    width = 0.36
    ax.bar(positions - width / 2, block['test MAE'], width, color=plot_style.MAIN_COLOR,
           edgecolor='black', linewidth=0.6, label='MAE')
    ax.bar(positions + width / 2, block['test RMSE'], width, color=plot_style.MUTED_COLOR,
           edgecolor='black', linewidth=0.6, label='RMSE')
    span = float(block[['test MAE', 'test RMSE']].to_numpy().max())
    for offset, column in ((-width / 2, 'test MAE'), (width / 2, 'test RMSE')):
        for position, value in zip(positions + offset, block[column]):
            ax.text(position, value + span * 0.02, f'{value:.2f}', ha='center', va='bottom',
                    fontsize=plot_style.FONT_SIZES['annotation'])
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel('数据划分方式（同为 test 子集，样本量不同）')
    ax.set_ylabel('测试集误差（元/天）')
    ax.set_ylim(0, span * 1.30)
    ax.legend(loc='upper left', frameon=False, fontsize=plot_style.FONT_SIZES['legend'])
    ax.text(0.985, 0.955, '(a)', transform=ax.transAxes, ha='right', va='top',
            fontsize=plot_style.FONT_SIZES['legend'])
    plot_style.apply_sci_axis(ax, grid_axis='y')

    ax = axes[1]
    ax.bar(positions, block['test R²'], width=0.5, color=plot_style.PALETTE[2],
           edgecolor='black', linewidth=0.6, label='R²')
    for position, value in zip(positions, block['test R²']):
        ax.text(position, value + 0.015, f'{value:.3f}', ha='center', va='bottom',
                fontsize=plot_style.FONT_SIZES['annotation'])
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel('数据划分方式')
    ax.set_ylabel('测试集 R²')
    ax.set_ylim(0, float(block['test R²'].max()) * 1.32)
    ax.legend(loc='upper right', frameon=False, fontsize=plot_style.FONT_SIZES['legend'])
    ax.text(0.013, 0.955, '(b)', transform=ax.transAxes, ha='left', va='top',
            fontsize=plot_style.FONT_SIZES['legend'])
    plot_style.apply_sci_axis(ax, grid_axis='y')
    fig.subplots_adjust(left=0.165, right=0.975, bottom=0.165, top=0.985, hspace=0.42)
    diagnostics = figure_finalize.save_paper_figure(
        fig, SUPP_DIR, 'fig_s53_random_vs_group_split_comparison', subfigures=[],
        meta={'数据来源': 'ch7/52_stage26_4_metrics_final.xlsx / 01_三种划分统一协议',
              '用途': '第8章 图 8-2（随正式特征集更新重绘）'})
    plt.close(fig)
    return diagnostics


def figure_robustness(robust: pd.DataFrame) -> dict:
    """图 8-4：稳健性检查的 MAE 与 R² 对照。"""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    block = robust[robust['MAE'].notna()].reset_index(drop=True)
    labels = ['主统计范围', '极端值截断', '目标：薪资下限', '目标：薪资上限']
    positions = np.arange(len(block))
    fig, axes = plt.subplots(2, 1, figsize=(5.85, 6.0))
    ax = axes[0]
    bars = ax.bar(positions, block['MAE'], width=0.5, color=plot_style.MAIN_COLOR,
                  edgecolor='black', linewidth=0.6, label='MAE')
    span = float(block['MAE'].max())
    for bar, value in zip(bars, block['MAE']):
        ax.text(bar.get_x() + bar.get_width() / 2, value + span * 0.02, f'{value:.2f}',
                ha='center', va='bottom', fontsize=plot_style.FONT_SIZES['annotation'])
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=12, ha='right')
    ax.set_xlabel('稳健性检查统计范围')
    ax.set_ylabel('测试集 MAE（元/天）')
    ax.set_ylim(0, span * 1.28)
    ax.legend(loc='upper left', frameon=False, fontsize=plot_style.FONT_SIZES['legend'])
    ax.text(0.985, 0.955, '(a)', transform=ax.transAxes, ha='right', va='top',
            fontsize=plot_style.FONT_SIZES['legend'])
    plot_style.apply_sci_axis(ax, grid_axis='y')

    ax = axes[1]
    bars = ax.bar(positions, block['R²'], width=0.5, color=plot_style.PALETTE[2],
                  edgecolor='black', linewidth=0.6, label='R²')
    span = float(block['R²'].max())
    for bar, value in zip(bars, block['R²']):
        ax.text(bar.get_x() + bar.get_width() / 2, value + span * 0.02, f'{value:.3f}',
                ha='center', va='bottom', fontsize=plot_style.FONT_SIZES['annotation'])
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=12, ha='right')
    ax.set_xlabel('稳健性检查统计范围')
    ax.set_ylabel('测试集 R²')
    ax.set_ylim(0, span * 1.24)
    ax.legend(loc='upper right', frameon=False, fontsize=plot_style.FONT_SIZES['legend'])
    ax.text(0.013, 0.955, '(b)', transform=ax.transAxes, ha='left', va='top',
            fontsize=plot_style.FONT_SIZES['legend'])
    plot_style.apply_sci_axis(ax, grid_axis='y')
    fig.subplots_adjust(left=0.165, right=0.975, bottom=0.205, top=0.985, hspace=0.52)
    diagnostics = figure_finalize.save_paper_figure(
        fig, SUPP_DIR, 'fig_s54_robustness_check_comparison', subfigures=[],
        meta={'数据来源': 'ch7/52_stage26_4_metrics_final.xlsx / 12_稳健性对照',
              '统计范围': '下限与上限统计范围的预测目标与主任务不同，只比较趋势',
              '用途': '第8章 图 8-4（随正式特征集更新重绘）'})
    plt.close(fig)
    return diagnostics


def run_extra_figures() -> int:
    """按最新结果重绘图 8-2 与图 8-4（图S53 / 图S54）。"""
    metrics = json.loads(METRICS_PATH.read_text(encoding='utf-8'))
    unified = pd.DataFrame(metrics['三种划分统一协议'])
    robust = pd.DataFrame(metrics['稳健性对照'])
    for diagnostics in (figure_split_compare(unified), figure_robustness(robust)):
        print(json.dumps({key: diagnostics.get(key) for key in
                          ('stem', 'png_path', 'pdf_path', 'png_size_bytes',
                           'pdf_size_bytes', 'png_dpi')}, ensure_ascii=False))
    return 0


def main() -> int:
    plot_style.setup_sci_style()
    plot_style.FONT_SIZES.update({'axis_label': 12.0, 'tick': 11.0, 'legend': 11.0,
                                  'annotation': 10.5})
    mode = sys.argv[1] if len(sys.argv) > 1 else 'compute'
    if mode == 'compute':
        return run_compute()
    if mode == 'extra_figures':
        return run_extra_figures()
    if mode == 'baseline':
        return run_baseline()
    if mode == 'textaudit':
        return run_textaudit()
    print(f'未知模式：{mode}')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())

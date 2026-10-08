# -*- coding: utf-8 -*-
"""Stage26.2：方法修复与补充分析（只新增文件，不修改任何既有产物）。

本脚本严格只读既有产物（`data/**`、`outputs/**` 既有文件、`src/**`、`docs/**`、
既有 `scripts/**`，尤其是 Stage26 / Stage26.1 的产物），只做**新增**计算与**新增**文件输出。

覆盖提示词 §5~§15 的方法修复与补充分析部分：

1. 事实核实：三种划分的测试集样本量与构成、消融 MAE 真实跨度、T2「0 天」口径冲突、
   T1 阈值敏感性（3/5/7/14 天）、五层结构表述残留、Safe-F 退化证据；
2. Safe-F 精简为 `publish_month` + `publish_weekday` 两个发布时间位置特征；
3. 统一三种泛化实验协议（LightGBM / A+B+C+D+E / 技能阈值 100 / 文本 SVD 16）并重算；
4. LightGBM vs CatBoost 多随机种子稳定性（42/52/62/72/82）；
5. Company Group Split 5 次 GroupShuffleSplit 稳定性；
6. 多变量中位数回归（sklearn 分位数回归 + bootstrap 区间）；
7. 面议岗选择偏差结构对照与卡方检验；
8. 极端薪资源记录审计（最低 / 最高各 20 条）；
9. 新增图件 图S28~图S31。

新增输出（不覆盖任何既有文件）::

    outputs/tables/49_stage26_2_fact_verification.xlsx
    outputs/tables/50_stage26_2_unified_generalization.xlsx
    outputs/tables/51_stage26_2_median_regression.xlsx
    outputs/tables/52_stage26_2_negotiable_selection_bias.xlsx
    outputs/tables/53_stage26_2_extreme_salary_audit.xlsx
    outputs/tables/54_stage26_2_wording_consistency_audit.xlsx
    outputs/figures/supplementary/图S28_特征组消融对比.png/pdf
    outputs/figures/supplementary/图S29_招聘周期持续时长分布与累积分布.png/pdf
    outputs/figures/supplementary/图S30_样本计划招聘覆盖每日活跃与新增结束.png/pdf
    outputs/figures/supplementary/图S31_主要岗位大类计划覆盖.png/pdf
    outputs/figures/supplementary/_stage26_2_registry.json
    outputs/logs/metrics/stage_26_2_consolidation.json

运行::

    E:\\anaconda3\\envs\\reptile\\python.exe scripts\\26c_stage26_2_final_consolidation.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import (ablation_shap, eda_analysis, figure_finalize, io_utils,  # noqa: E402
                 model_training, plot_style, project_paths, schema, skill_eda)

# ============================================================================
# 常量与口径
# ============================================================================
SEED = 42
SEEDS = (42, 52, 62, 72, 82)
SKILL_THRESHOLD = 100
TEXT_DIM = 16
LIGHTGBM_PARAMS = {'n_estimators': 400, 'learning_rate': 0.05, 'num_leaves': 63}
CATBOOST_PARAMS = {'depth': 8, 'learning_rate': 0.05, 'iterations': 500}
THRESHOLD_GRID = (3, 5, 7, 14)
MAJOR_CATEGORIES = ['人工智能', '后端开发', '前端开发', '数据', '产品', '运营']
COLLECT_FIRST = pd.Timestamp('2026-03-22')
COLLECT_LAST = pd.Timestamp('2026-04-22')
MODEL_FEATURE_GROUPS = ('A', 'B', 'C', 'D', 'E')
SAFE_F_FINAL = ['publish_month', 'publish_weekday']
SAFE_F_STAGE26_1 = ['publish_month', 'publish_weekday', 'episode_no_strict',
                    'is_confirmed_reopen', 'historical_episode_count',
                    'previous_reopen_gap_days']
ABLATION_CONFIGS = {
    'A+B+C': ('A', 'B', 'C'),
    'A+B+C+D': ('A', 'B', 'C', 'D'),
    'A+B+C+E': ('A', 'B', 'C', 'E'),
    'A+B+C+D+E': ('A', 'B', 'C', 'D', 'E'),
    'A+B+C+D+E+SafeF': ('A', 'B', 'C', 'D', 'E', 'SafeF'),
}
ABLATION_ORDER = list(ABLATION_CONFIGS)
MEDIAN_CATEGORICAL = ['工作城市', '学历要求', '每周到岗要求', '实习时长要求',
                      '公司规模', '公司性质', '所属行业', '公司认证类别']
MEDIAN_MULTI = ['岗位大类集合', '岗位细分类集合']
MEDIAN_MIN_FREQ = 20
MEDIAN_TOP_MULTI = 30
MEDIAN_BOOTSTRAP_ROUNDS = 100

CALIBER_ACTIVE = ('按业务日期重构的样本活跃计划周期数量：由当前样本岗位的『发布时间 → 投递截止日期』'
                  '计划区间展开得到，曲线只反映本样本的计划招聘覆盖结构，'
                  '不等同于当日完整市场存量，也不构成市场时序。')

PROCESSED = project_paths.PROCESSED_DIR
TABLES = project_paths.TABLES_DIR
SUPP_DIR = project_paths.FIGURES_DIR / 'supplementary'
METRICS_PATH = project_paths.METRICS_DIR / 'stage_26_2_consolidation.json'
REGISTRY_PATH = SUPP_DIR / '_stage26_2_registry.json'
SEGMENT_PATH = PROCESSED / 'job_candidate_segment_26_1.parquet'
EPISODE_PATH = PROCESSED / 'job_strict_episode_26_1.parquet'
PANEL_PATH = PROCESSED / 'job_strict_daily_panel_26_1.parquet'
LIFECYCLE_TABLE = TABLES / '45_stage26_1_lifecycle_statistics.xlsx'

TABLE_FILES = [
    '49_stage26_2_fact_verification.xlsx',
    '50_stage26_2_unified_generalization.xlsx',
    '51_stage26_2_median_regression.xlsx',
    '52_stage26_2_negotiable_selection_bias.xlsx',
    '53_stage26_2_extreme_salary_audit.xlsx',
    '54_stage26_2_wording_consistency_audit.xlsx',
]
FIG_STEMS = [
    '图S28_特征组消融对比',
    '图S29_招聘周期持续时长分布与累积分布',
    '图S30_样本计划招聘覆盖每日活跃与新增结束',
    '图S31_主要岗位大类计划覆盖',
]
NEW_FILES = ([TABLES / name for name in TABLE_FILES]
             + [SUPP_DIR / f'{stem}{suffix}' for stem in FIG_STEMS
                for suffix in ('.png', '.pdf')]
             + [REGISTRY_PATH, METRICS_PATH])
SKIP_DIRS = {'.git', '.pytest_cache', '__pycache__', '.ipynb_checkpoints', '.idea', '.vscode'}
MANIFEST_SCOPE_DIRS = ['data', 'outputs', 'docs', 'src', 'scripts', 'config', 'notebooks', 'tests']

STAGE26_1_MOTHER = (PROJECT_ROOT / 'docs' / 'paper'
                    / '课程设计论文时序口径收紧修订版_Stage26.1.md')
WORDING_PATTERNS = [
    ('四层结构残留', r'四层'),
    ('三层结构残留', r'三层'),
    ('观测—版本—实体 三层简称', r'观测—版本—实体'),
    ('图 3-3 图题三层简称', r'观测 → 版本 → 实体'),
    ('T3 单独归并措辞', r'T3 的归并使'),
    ('T3 归并减少周期数行', r'T3 归并减少的周期数'),
    ('T2 间隔含 0 天', r'1 / 0 / 6'),
    ('3,410 减少量行', r'3,410'),
]


# ============================================================================
# 通用工具
# ============================================================================
def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


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
            if path.resolve() in new_resolved:
                continue
            manifest[str(path.relative_to(PROJECT_ROOT)).replace('\\', '/')] = sha256_of(path)
    readme = PROJECT_ROOT / 'README.md'
    if readme.is_file() and readme.resolve() not in new_resolved:
        manifest['README.md'] = sha256_of(readme)
    return manifest


def input_record(path: Path) -> dict:
    return {'路径': str(path.relative_to(PROJECT_ROOT)).replace('\\', '/'),
            '字节数': int(path.stat().st_size), 'SHA256': sha256_of(path)}


# ============================================================================
# T1 / T2 / T3 统一口径（gap 以精确时间差计算后转换为日历天；相差 0 天归入 T3）
# ============================================================================
def assign_episodes_calendar(segments: pd.DataFrame, threshold_days: int) -> pd.DataFrame:
    """按「日历天」统一口径重算三档转换与 Strict / Relaxed 周期归属。

    口径：gap = date(start) − date(当前周期结束日)（日历天，整数）。
    - `gap >= threshold_days` → T1_CONFIRMED，新建正式周期；
    - `0 < gap < threshold_days` → T2_SUSPECTED，严格口径归并（Relaxed 口径新建）；
    - `gap <= 0` → T3_CONFLICT，两口径均归并。

    与 Stage26.1 实现的差异：Stage26.1 用精确时间差（浮点天）判档、再用日历天报告，
    因而出现「T2 区间为 0 < gap < 7、但报告最小值为 0 天」的自相矛盾记录；
    本函数把判档口径与报告口径统一为日历天。
    """
    frame = segments.sort_values(['intern_id', 'segment_first_observed', 'segment_no'],
                                 kind='stable').reset_index(drop=True)
    job_ids = frame['intern_id'].to_numpy()
    starts = frame['segment_start'].to_numpy('datetime64[D]')
    ends = frame['segment_end'].to_numpy('datetime64[D]')
    count = len(frame)
    strict_no = np.zeros(count, dtype='int64')
    relax_no = np.zeros(count, dtype='int64')
    tier = np.empty(count, dtype=object)
    tier[:] = ''
    gap = np.full(count, np.nan)

    positions: dict = {}
    for position in range(count):
        positions.setdefault(job_ids[position], []).append(position)

    for _, indexes in positions.items():
        strict_counter = relax_counter = 1
        strict_end = ends[indexes[0]]
        relax_end = ends[indexes[0]]
        strict_no[indexes[0]] = relax_no[indexes[0]] = 1
        for position in indexes[1:]:
            start, end = starts[position], ends[position]
            value = float((start - strict_end) / np.timedelta64(1, 'D'))
            gap[position] = value
            if value >= threshold_days:
                strict_counter += 1
                tier[position] = 'T1_CONFIRMED'
            elif 0 < value < threshold_days:
                tier[position] = 'T2_SUSPECTED'
            else:
                tier[position] = 'T3_CONFLICT'
            strict_end = end
            strict_no[position] = strict_counter
            relax_value = float((start - relax_end) / np.timedelta64(1, 'D'))
            if relax_value >= threshold_days or 0 < relax_value < threshold_days:
                relax_counter += 1
            relax_end = end
            relax_no[position] = relax_counter

    frame['gap_days_calendar_unified'] = gap
    frame['transition_tier_unified'] = tier
    frame['episode_no_strict_unified'] = strict_no
    frame['episode_no_relaxed_unified'] = relax_no
    return frame


def threshold_block(frame: pd.DataFrame, threshold_days: int) -> list:
    strict = frame['transition_tier_unified'].eq('T1_CONFIRMED')
    relaxed_mask = frame['transition_tier_unified'].isin(['T1_CONFIRMED', 'T2_SUSPECTED'])
    return [
        {'口径': 'Strict（T1 新建；T2 / T3 归并）', '阈值（日历天）': threshold_days,
         'T1 事件数': int(strict.sum()),
         'T1 岗位数': int(frame.loc[strict, 'intern_id'].nunique()),
         '正式周期数': int(frame.groupby('intern_id')['episode_no_strict_unified'].max().sum())},
        {'口径': 'Relaxed（T1 + T2 新建；T3 归并）', '阈值（日历天）': threshold_days,
         'T1 事件数': int(relaxed_mask.sum()),
         'T1 岗位数': int(frame.loc[relaxed_mask, 'intern_id'].nunique()),
         '正式周期数': int(frame.groupby('intern_id')['episode_no_relaxed_unified'].max().sum())},
    ]


# ============================================================================
# 特征装配辅助
# ============================================================================
def grouped_columns_with_safe_f(manifest: dict, frame: pd.DataFrame) -> dict:
    grouped = ablation_shap.split_columns_by_group(
        manifest['numeric_columns'], manifest['categorical_columns'],
        manifest['multi_value_columns'])
    for _, spec in grouped.items():
        for key in spec:
            spec[key] = [column for column in spec[key] if column not in SAFE_F_STAGE26_1]
    grouped['SafeF'] = {'numeric': [column for column in SAFE_F_FINAL
                                    if column in frame.columns],
                        'categorical': [], 'multi': []}
    return grouped


def build_assembler(groups, grouped, skill_threshold: int, text_dim: int):
    numeric, categorical, multi = [], [], []
    for letter in groups:
        numeric += grouped[letter]['numeric']
        categorical += grouped[letter]['categorical']
        multi += grouped[letter]['multi']
    return model_training.SalaryFeatureAssembler(
        numeric, categorical, multi,
        skill_threshold=skill_threshold if 'D' in groups else 10 ** 9,
        text_dim=text_dim if 'E' in groups else 0)


def fit_eval(frame: pd.DataFrame, labels: pd.Series, groups, grouped, skill_map,
             text_matrix: np.ndarray, text_by_id: dict, skill_threshold: int,
             text_dim: int, model_key: str = 'LightGBM', params: dict | None = None,
             random_state: int = SEED) -> dict:
    """统一协议：预处理只在 train 拟合、模型只在 train 拟合，validation / test 只评估。"""
    values = frame[schema.ID_FIELD].map(labels).to_numpy()
    train_frame = frame[values == 'train'].reset_index(drop=True)
    valid_frame = frame[values == 'validation'].reset_index(drop=True)
    test_frame = frame[values == 'test'].reset_index(drop=True)

    def text_for(block):
        return text_matrix[[text_by_id[job_id] for job_id in block[schema.ID_FIELD]]]

    assembler = build_assembler(groups, grouped, skill_threshold, text_dim)
    assembler.fit(train_frame, skill_map, text_for(train_frame))
    matrix_train = assembler.transform(train_frame, skill_map, text_for(train_frame))
    matrix_valid = assembler.transform(valid_frame, skill_map, text_for(valid_frame))
    matrix_test = assembler.transform(test_frame, skill_map, text_for(test_frame))
    model = model_training.make_model(model_key, params or LIGHTGBM_PARAMS,
                                      random_state=random_state)
    model.fit(matrix_train, train_frame[schema.SALARY_MID_FIELD].to_numpy('float64'))
    valid_pred = np.asarray(model.predict(matrix_valid), dtype='float64')
    test_pred = np.asarray(model.predict(matrix_test), dtype='float64')
    return {
        '特征维度': int(assembler.schema.dimension),
        '技能列数': int(len(assembler.schema.skill_columns)),
        'n_train': int(len(train_frame)), 'n_validation': int(len(valid_frame)),
        'n_test': int(len(test_frame)),
        'validation': model_training.regression_metrics(
            valid_frame[schema.SALARY_MID_FIELD].to_numpy('float64'), valid_pred),
        'test': model_training.regression_metrics(
            test_frame[schema.SALARY_MID_FIELD].to_numpy('float64'), test_pred),
    }


# ============================================================================
# 中位数回归设计矩阵（显式参照类别 + 截距）
# ============================================================================
def build_median_design(frame: pd.DataFrame):
    names = ['截距']
    blocks = [np.ones(len(frame), dtype='float64')]
    references = {}
    for column in MEDIAN_CATEGORICAL:
        series = frame[column].fillna('__未知__').astype(str)
        counts = series.value_counts()
        kept = [value for value, count in counts.items() if count >= MEDIAN_MIN_FREQ]
        reference = counts.index[0]
        references[column] = str(reference)
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
        levels = exploded.value_counts().head(MEDIAN_TOP_MULTI).index.tolist()
        references[column] = str(levels[0])
        sets = [set(value) if value is not None else set() for value in frame[column]]
        for level in levels[1:]:
            names.append(f'{column}={level}')
            blocks.append(np.array([float(level in item) for item in sets]))
    return np.column_stack(blocks), names, references


def quantile_fit(matrix: np.ndarray, target: np.ndarray,
                 quantile: float = 0.5) -> np.ndarray:
    """分位数回归（Q0.5）系数；设计矩阵首列已显式包含截距，故 fit_intercept=False。"""
    from sklearn.linear_model import QuantileRegressor  # noqa: PLC0415

    model = QuantileRegressor(quantile=quantile, alpha=0.0, fit_intercept=False,
                              solver='highs-ipm')
    model.fit(matrix, target)
    return np.asarray(model.coef_, dtype='float64')


# ============================================================================
# 图件
# ============================================================================
def _window_marks(ax, date_min, date_max):
    from matplotlib.patches import Patch  # noqa: PLC0415

    ax.axvspan(date_min, COLLECT_FIRST, color='#d9d9d9', alpha=0.30, linewidth=0, zorder=0)
    ax.axvspan(COLLECT_FIRST, COLLECT_LAST, color='#b8ddb8', alpha=0.45, linewidth=0, zorder=0)
    ax.axvspan(COLLECT_LAST, date_max, color='#f7d9b8', alpha=0.45, linewidth=0, zorder=0)
    ax.axvline(COLLECT_FIRST, color='black', linestyle='--', linewidth=1.0, zorder=1)
    ax.axvline(COLLECT_LAST, color='black', linestyle='--', linewidth=1.0, zorder=1)
    return [Patch(facecolor='#d9d9d9', alpha=0.6, label='历史回溯区'),
            Patch(facecolor='#b8ddb8', alpha=0.7, label='实际采集窗口'),
            Patch(facecolor='#f7d9b8', alpha=0.7, label='计划未来覆盖区')]


def enlarge_fonts() -> None:
    """放大版图件字号（等效单栏宽 ≥ 15 cm、字号 ≥ 10.5 pt）。"""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plt.rcParams['font.size'] = 11.5
    plt.rcParams['axes.labelsize'] = 12.0
    plt.rcParams['xtick.labelsize'] = 11.0
    plt.rcParams['ytick.labelsize'] = 11.0
    plt.rcParams['legend.fontsize'] = 11.0
    plt.rcParams['axes.titlesize'] = 12.0


def figure_ablation(ablation: pd.DataFrame, registry: list) -> dict:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    enlarge_fonts()
    fig, ax = plt.subplots(figsize=(16.0, 6.6))
    labels = list(ablation['配置'])
    positions = np.arange(len(labels))
    width = 0.38
    valid = ablation['validation_MAE'].to_numpy('float64')
    test = ablation['test_MAE'].to_numpy('float64')
    bars_valid = ax.bar(positions - width / 2, valid, width, color=plot_style.MAIN_COLOR,
                        edgecolor='black', linewidth=0.6, label='validation MAE')
    bars_test = ax.bar(positions + width / 2, test, width, color=plot_style.ACCENT_COLOR,
                       edgecolor='black', linewidth=0.6, label='test MAE')
    span = float(test.max() - test.min())
    for bars in (bars_valid, bars_test):
        for bar in bars:
            ax.annotate(f'{bar.get_height():.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center',
                        va='bottom', fontsize=10.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=12, ha='right')
    ax.set_xlabel('消融配置（特征组）')
    ax.set_ylabel('MAE（元/天）')
    ax.set_ylim(0, float(max(valid.max(), test.max())) * 1.30)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.005), ncol=2, frameon=False)
    ax.text(0.985, 0.96,
            f'test 最大跨度 {span:.3f} 元/天\n'
            f'Random Split；LightGBM 400/0.05/63\nn = {int(ablation["n_test"].iloc[0]):,}',
            transform=ax.transAxes, ha='right', va='top', fontsize=10.5, linespacing=1.5)
    plot_style.apply_sci_axis(ax, grid_axis='y')
    plot_style.add_subfigure_caption(ax, 'a', '特征组消融的验证集与测试集 MAE（配置组数与表 8-1 一致）')
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.30, top=0.86)
    diagnostics = figure_finalize.save_paper_figure(
        fig, SUPP_DIR, FIG_STEMS[0],
        subfigures=[('a', '特征组消融的验证集与测试集 MAE（配置组数与表 8-1 一致）', ax)],
        meta={'数据来源': 'job_salary_model_dataset.parquet + model_splits.parquet',
              'seed': SEED, '用途': '第8章 图 8-1 重绘（含精简后 Safe-F 配置）'})
    plt.close(fig)
    registry.append(diagnostics)
    return diagnostics


def figure_duration(episodes: pd.DataFrame, registry: list) -> dict:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    enlarge_fonts()
    duration = np.sort(episodes.loc[
        episodes['final_observed_planned_duration_days'] > 0,
        'final_observed_planned_duration_days'].to_numpy('float64'))
    fig, axes = plt.subplots(1, 2, figsize=(16.4, 6.2))

    ax = axes[0]
    plot_style.hist_discrete(ax, duration, discrete=False, bins=60,
                             xlabel='招聘周期计划持续天数（日历日，含端点）',
                             ylabel='正式招聘周期数')
    ax.axvline(float(np.median(duration)), color=plot_style.ACCENT_COLOR,
               linestyle='--', linewidth=1.2)
    ax.set_xlim(0, float(np.quantile(duration, 0.99)))
    ax.text(0.985, 0.95, f'中位数 {np.median(duration):.0f} 天\n'
                         f'IQR {np.quantile(duration, .75) - np.quantile(duration, .25):.0f} 天\n'
                         f'n = {duration.size:,}',
            transform=ax.transAxes, ha='right', va='top', fontsize=10.5, linespacing=1.5)
    plot_style.apply_sci_axis(ax, grid_axis='y')
    plot_style.add_subfigure_caption(ax, 'a', '招聘周期计划持续天数分布（横轴截断至 P99）')

    ax = axes[1]
    cumulative = np.arange(1, duration.size + 1) / duration.size
    ax.plot(duration, cumulative, color=plot_style.MAIN_COLOR, linewidth=1.8)
    for quantile, label in [(0.10, 'P10'), (0.50, 'P50'), (0.90, 'P90')]:
        value = float(np.quantile(duration, quantile))
        ax.axvline(value, color=plot_style.MUTED_COLOR, linestyle=':', linewidth=1.0)
        ax.annotate(f'{label} = {value:.0f} 天', xy=(value, quantile),
                    xytext=(6, -14), textcoords='offset points', fontsize=10.5)
    ax.set_xlim(0, float(np.quantile(duration, 0.99)))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('招聘周期计划持续天数（日历日，含端点）')
    ax.set_ylabel('累计比例')
    plot_style.format_percent_axis(ax, axis='y', decimals=0)
    plot_style.apply_sci_axis(ax, grid_axis='both')
    plot_style.add_subfigure_caption(ax, 'b', '招聘周期计划持续天数的累积分布')
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.25, top=0.955, wspace=0.22)
    diagnostics = figure_finalize.save_paper_figure(
        fig, SUPP_DIR, FIG_STEMS[1],
        subfigures=[('a', '招聘周期计划持续天数分布（横轴截断至 P99）', axes[0]),
                    ('b', '招聘周期计划持续天数的累积分布', axes[1])],
        meta={'数据来源': 'job_strict_episode_26_1.parquet',
              '口径': ('计划持续天数 = 末次观测截止日 − 发布时间（日历差）+ 1；'
                       '属计划窗口，不等于实际招满所需时间'),
              'seed': SEED, '用途': '第4章 图 4-4 重绘（T1/T2/T3 数量改用表格）'})
    plt.close(fig)
    registry.append(diagnostics)
    return diagnostics


def figure_daily(daily: pd.DataFrame, registry: list) -> dict:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    enlarge_fonts()
    dates = pd.to_datetime(daily['date'])
    date_min, date_max = dates.min(), dates.max()
    panels = [
        ('a', 'N_t 样本活跃计划周期数量',
         [('N_t', 'Strict 活跃计划周期数（原始日序列）', plot_style.MUTED_COLOR, 0.7),
          ('N_t_7d', 'Strict 活跃计划周期数（7 日滚动中位数）', plot_style.MAIN_COLOR, 2.0),
          ('N_t_job_entity_7d', '去重到岗位实体（7 日滚动中位数）', plot_style.PALETTE[2], 1.2)]),
        ('b', 'O_t 每日新增 / C_t 每日结束周期数',
         [('O_t', '每日新增（原始日序列）', plot_style.MUTED_COLOR, 0.6),
          ('O_t_7d', '每日新增（7 日滚动中位数）', plot_style.MAIN_COLOR, 1.8),
          ('C_t', '每日结束（原始日序列）', plot_style.PALETTE[3], 0.6),
          ('C_t_7d', '每日结束（7 日滚动中位数）', plot_style.ACCENT_COLOR, 1.8)]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(17.0, 6.8))
    for (letter, title, series_list), ax in zip(panels, axes):
        handles = _window_marks(ax, date_min, date_max)
        for column, label, color, width in series_list:
            ax.plot(dates, daily[column], color=color, linewidth=width, label=label)
        ax.set_xlabel('业务日期（由岗位发布时间 / 投递截止日期重构）')
        ax.set_ylabel(title)
        handles = handles + [line for line in ax.get_lines()
                             if not str(line.get_label()).startswith('_')]
        ax.legend(handles, [h.get_label() for h in handles], loc='lower center',
                  bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False)
        plot_style.apply_sci_axis(ax, grid_axis='y')
        ax.text(0.985, 0.95, '曲线由当前样本岗位的业务日期重构，\n不等同于当日完整市场存量',
                transform=ax.transAxes, ha='right', va='top', fontsize=10.5, linespacing=1.5)
        plot_style.add_subfigure_caption(ax, letter, title)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.255, top=0.80, wspace=0.18)
    diagnostics = figure_finalize.save_paper_figure(
        fig, SUPP_DIR, FIG_STEMS[2],
        subfigures=[('a', panels[0][1], axes[0]), ('b', panels[1][1], axes[1])],
        meta={'数据来源': 'job_strict_daily_panel_26_1.parquet 的日级聚合',
              '口径': CALIBER_ACTIVE,
              '图注声明': '曲线由当前样本岗位的业务日期重构，不等同于当日完整市场存量',
              '采集窗口标识': f'首次采集日 {COLLECT_FIRST.date()} / 最后采集日 {COLLECT_LAST.date()}',
              'seed': SEED, '用途': '第4章 图 4-5 放大版（单独占整行）'})
    plt.close(fig)
    registry.append(diagnostics)
    return diagnostics


def figure_category(category_daily: dict, daily: pd.DataFrame, registry: list) -> dict:
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.ticker import MaxNLocator  # noqa: PLC0415

    enlarge_fonts()
    dates = pd.to_datetime(daily['date'])
    date_min, date_max = dates.min(), dates.max()
    fig, axes = plt.subplots(1, 2, figsize=(17.0, 7.0))
    for ax, smoothed, letter, title in [
            (axes[0], False, 'a', 'N(c,t) 样本活跃计划周期数（原始日序列）'),
            (axes[1], True, 'b', 'N(c,t) 样本活跃计划周期数（7 日滚动中位数）')]:
        handles = _window_marks(ax, date_min, date_max)
        lines = []
        for position, (category, series) in enumerate(category_daily.items()):
            values = series.reindex(dates).to_numpy('float64')
            values = (pd.Series(values).rolling(7, min_periods=1).median().to_numpy('float64')
                      if smoothed else values)
            line, = ax.plot(dates, values, linewidth=1.6,
                            color=plot_style.PALETTE[position % len(plot_style.PALETTE)],
                            label=category)
            lines.append(line)
        ax.set_xlabel('业务日期（由岗位发布时间 / 投递截止日期重构）')
        ax.set_ylabel(title)
        ax.xaxis.set_major_locator(MaxNLocator(6))
        handles = handles + lines
        ax.legend(handles, [h.get_label() for h in handles], loc='lower center',
                  bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False)
        plot_style.apply_sci_axis(ax, grid_axis='y')
        plot_style.add_subfigure_caption(ax, letter, title)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.25, top=0.775, wspace=0.16)
    diagnostics = figure_finalize.save_paper_figure(
        fig, SUPP_DIR, FIG_STEMS[3],
        subfigures=[('a', 'N(c,t) 样本活跃计划周期数（原始日序列）', axes[0]),
                    ('b', 'N(c,t) 样本活跃计划周期数（7 日滚动中位数）', axes[1])],
        meta={'数据来源': 'job_strict_daily_panel_26_1.parquet × 岗位大类集合',
              '口径': CALIBER_ACTIVE,
              '图注声明': '曲线由当前样本岗位的业务日期重构，不等同于当日完整市场存量',
              '采集窗口标识': f'首次采集日 {COLLECT_FIRST.date()} / 最后采集日 {COLLECT_LAST.date()}',
              'seed': SEED, '用途': '第4章 图 4-6 放大版（单独占整行）'})
    plt.close(fig)
    registry.append(diagnostics)
    return diagnostics


def build_category_daily() -> dict:
    """从 Stage26.1 严格日级面板按岗位大类汇总每日活跃计划周期数（向量化）。"""
    panel = io_utils.read_parquet(PANEL_PATH, columns=['date', 'intern_id'])
    analysis = io_utils.read_parquet(project_paths.JOB_ANALYSIS_DATASET_PARQUET,
                                     columns=[schema.ID_FIELD, '岗位大类集合'])
    mapping = dict(zip(analysis[schema.ID_FIELD], analysis['岗位大类集合']))
    dates = pd.to_datetime(panel['date'])
    day_numbers = dates.to_numpy().astype('datetime64[D]').astype('int64')
    day_index = day_numbers - day_numbers.min()
    size = int(day_index.max()) + 1
    codes, uniques = pd.factorize(panel['intern_id'].to_numpy())
    category_index = {name: position for position, name in enumerate(MAJOR_CATEGORIES)}
    mask_matrix = np.zeros((len(uniques), len(MAJOR_CATEGORIES)), dtype=bool)
    for row, job_id in enumerate(uniques):
        value = mapping.get(job_id)
        if value is None:
            continue
        for item in list(value):
            position = category_index.get(str(item))
            if position is not None:
                mask_matrix[row, position] = True
    output = {}
    for position, name in enumerate(MAJOR_CATEGORIES):
        selected = mask_matrix[codes, position]
        counts = np.bincount(day_index[selected], minlength=size)
        output[name] = pd.Series(counts, index=pd.date_range(
            dates.min().normalize(), periods=size, freq='D'))
    return output


def _chi2_with_positive_cells(table: pd.DataFrame):
    observed = table.to_numpy('float64')
    observed = observed[(observed.sum(axis=1) > 0) & (observed.min(axis=1) > 0)]
    if observed.shape[0] < 2:
        return float('nan'), float('nan'), 0, observed.shape[0]
    statistic, p_value, dof, _ = scipy_stats.chi2_contingency(observed)
    return float(statistic), float(p_value), int(dof), int(observed.shape[0])


def _chi2_2x2(observed: np.ndarray):
    if (observed.sum(axis=1) == 0).any() or (observed.sum(axis=0) == 0).any():
        return float('nan'), float('nan')
    statistic, p_value, _, _ = scipy_stats.chi2_contingency(observed, correction=False)
    return float(statistic), float(p_value)


def audit_verdict(row) -> tuple:
    raw = str(row[schema.SALARY_RAW_FIELD])
    if not re.search(r'元?\s*(/|每)\s*(天|日|月|时)', raw) and '/' not in raw:
        return '单位缺失，需人工复核', '单位'
    if re.search(r'(/月|每月|/时|每小时)', raw):
        return '疑似月薪 / 时薪误解析', '单位'
    low_value = float(row[schema.SALARY_MIN_FIELD])
    high_value = float(row[schema.SALARY_MAX_FIELD])
    if low_value <= 0 or high_value <= 0:
        return '非正数薪资，解析异常', '数值'
    if low_value > high_value:
        return '上下限倒置，解析异常', '数值'
    if high_value / max(low_value, 1e-9) > 20:
        return '上下限跨度异常大，需人工复核', '区间'
    if float(row[schema.SALARY_MID_FIELD]) <= 5:
        return '数值极低但格式合规（疑似象征性日薪）', '无'
    return '格式合规、解析正确', '无'


# ============================================================================
# 主流程
# ============================================================================
def main() -> int:  # noqa: C901
    started = time.time()
    print('=' * 96)
    print('Stage26.2 方法修复与补充分析（只新增文件）')
    print('=' * 96)
    manifest_before = project_manifest()
    print(f'运行前既有文件 SHA-256 清单：{len(manifest_before)} 个')

    # ---------------------------------------------------------------- 输入
    model_frame = io_utils.read_parquet(project_paths.JOB_SALARY_MODEL_DATASET_PARQUET)
    splits = io_utils.read_parquet(project_paths.MODEL_SPLITS_PARQUET)
    membership = io_utils.read_parquet(project_paths.JOB_SKILL_MEMBERSHIP_PARQUET)
    analysis = io_utils.read_parquet(project_paths.JOB_ANALYSIS_DATASET_PARQUET)
    entity = pd.read_parquet(project_paths.PROCESSED_UNIQUE_PARQUET,
                             columns=[schema.ID_FIELD, '发布时间', schema.CERT_TAG_FIELD])
    entity['发布时间'] = pd.to_datetime(entity['发布时间'], errors='coerce')
    entity_publish = entity.set_index(schema.ID_FIELD)['发布时间']
    segments = io_utils.read_parquet(SEGMENT_PATH)
    episodes = io_utils.read_parquet(EPISODE_PATH)
    salary_targets = io_utils.read_parquet(project_paths.SALARY_TARGETS_PARQUET)
    feature_manifest = json.loads(
        (project_paths.SALARY_MODEL_DIR / 'feature_manifest.json').read_text(encoding='utf-8'))
    assert len(model_frame) == 14883 and len(segments) == 20556
    job_ids = model_frame[schema.ID_FIELD]
    company_values = splits.set_index(schema.ID_FIELD)['company_entity_id'].reindex(
        job_ids.to_numpy()).to_numpy()
    episodes_unique = episodes.drop_duplicates(
        subset=['intern_id', 'episode_id_strict']).reset_index(drop=True)
    print(f'输入：建模样本 {len(model_frame):,} / 候选发布时间段 {len(segments):,} / '
          f'严格口径正式周期 {len(episodes_unique):,}（周期层导出一行 = 一个候选段）')

    # ---------------------------------------------------------------- A 事实核实
    frame_publish = job_ids.map(entity_publish)
    temporal_frame = model_frame.assign(_business_date=frame_publish.dt.normalize())
    date_counts = temporal_frame.groupby('_business_date').size().sort_index()
    cumulative = date_counts.cumsum()
    total = int(cumulative.iloc[-1])
    dates = cumulative.index
    index1 = min(max(int(np.searchsorted(cumulative.to_numpy(), 0.70 * total, side='left')), 1),
                 len(dates) - 2)
    index2 = min(max(int(np.searchsorted(cumulative.to_numpy(), 0.85 * total, side='left')),
                     index1 + 1), len(dates) - 1)
    temporal_values = np.where(
        temporal_frame['_business_date'].isin(dates[:index1]),
        'train', np.where(temporal_frame['_business_date'].isin(dates[index1:index2]),
                          'validation', 'test'))
    split_definitions = {
        'Random Split（同分布泛化）': pd.Series(splits['split'].to_numpy(), index=job_ids),
        'Company Group Split（跨公司泛化）': pd.Series(
            splits['company_group_split'].to_numpy(), index=job_ids),
        'Retrospective Temporal Split（跨发布时间区间泛化）': pd.Series(
            temporal_values, index=job_ids),
    }
    split_rows = []
    business_dates = temporal_frame['_business_date'].to_numpy()
    for label, labels in split_definitions.items():
        values = labels.to_numpy()
        for subset in ('train', 'validation', 'test'):
            mask = values == subset
            split_rows.append({
                '划分方式': label, '数据子集': subset, 'n': int(mask.sum()),
                '占比': round(float(mask.mean()), 6),
                '公司数': int(pd.unique(company_values[mask]).size),
                '唯一业务日数': int(pd.Series(business_dates[mask]).nunique())})
    split_table = pd.DataFrame(split_rows)
    print('三种划分样本量：')
    print(split_table.to_string(index=False))

    # ---- T2 gap 口径核实（Stage26.1 实测）
    tier_counts = segments['transition_tier'].value_counts()
    gap_rows = []
    for name, key in [('T1 确认重招', 'T1_CONFIRMED'), ('T2 疑似重开', 'T2_SUSPECTED'),
                      ('T3 时间冲突', 'T3_CONFLICT')]:
        block = segments[segments['transition_tier'].eq(key)]
        if not len(block):
            continue
        gap_rows.append({
            '转换类型': name, '事件数': int(len(block)),
            '涉及岗位数': int(block['intern_id'].nunique()),
            'gap（精确天）最小': round(float(block['gap_days_exact'].min()), 6),
            'gap（精确天）中位数': round(float(block['gap_days_exact'].median()), 6),
            'gap（精确天）最大': round(float(block['gap_days_exact'].max()), 6),
            'gap（日历天）最小': int(block['gap_days_calendar'].min()),
            'gap（日历天）中位数': int(block['gap_days_calendar'].median()),
            'gap（日历天）最大': int(block['gap_days_calendar'].max()),
            '精确天在 (0, 1) 的记录数': int(block['gap_days_exact'].between(
                0, 1, inclusive='neither').sum()),
            '日历天 == 0 的记录数': int(block['gap_days_calendar'].eq(0).sum())})
    gap_table = pd.DataFrame(gap_rows)
    t2_block = segments[segments['transition_tier'].eq('T2_SUSPECTED')]
    t2_calendar_zero = int(t2_block['gap_days_calendar'].eq(0).sum())
    t2_calendar_dist = (t2_block['gap_days_calendar'].value_counts().sort_index()
                        .rename_axis('gap（日历天）').reset_index(name='事件数'))
    print(f'T2 定义冲突证据：T2 事件 {len(t2_block)} 个，其中日历天 = 0 的记录 '
          f'{t2_calendar_zero} 个（精确天落在 (0,1)）')

    # ---- 阈值敏感性（统一日历天口径）
    threshold_frames = {}
    sensitivity_rows = []
    for threshold in THRESHOLD_GRID:
        block = assign_episodes_calendar(segments, threshold)
        threshold_frames[threshold] = block
        sensitivity_rows.extend(threshold_block(block, threshold))
    sensitivity_table = pd.DataFrame(sensitivity_rows)
    unified = threshold_frames[7]
    unified_counts = {
        'T1': int(unified['transition_tier_unified'].eq('T1_CONFIRMED').sum()),
        'T2': int(unified['transition_tier_unified'].eq('T2_SUSPECTED').sum()),
        'T3': int(unified['transition_tier_unified'].eq('T3_CONFLICT').sum())}
    unified_episodes = int(unified.groupby('intern_id')['episode_no_strict_unified'].max().sum())
    unified_relaxed = int(unified.groupby('intern_id')['episode_no_relaxed_unified'].max().sum())
    old_counts = {'T1': int(tier_counts.get('T1_CONFIRMED', 0)),
                  'T2': int(tier_counts.get('T2_SUSPECTED', 0)),
                  'T3': int(tier_counts.get('T3_CONFLICT', 0))}
    caliber_table = pd.DataFrame([
        {'项目': 'Stage26.1 判档依据', '内容': 'gap 用精确时间差（浮点天）判档，报告时改用日历天',
         'T1 事件数': old_counts['T1'], 'T2 事件数': old_counts['T2'],
         'T3 事件数': old_counts['T3'], 'Strict 周期数': int(len(episodes_unique)),
         'Relaxed 周期数': 17678},
        {'项目': 'Stage26.2 统一口径', '内容': 'gap 先转日历天再判档；相差 0 天归入 T3',
         'T1 事件数': unified_counts['T1'], 'T2 事件数': unified_counts['T2'],
         'T3 事件数': unified_counts['T3'], 'Strict 周期数': unified_episodes,
         'Relaxed 周期数': unified_relaxed},
        {'项目': '变化', '内容': f'日历天为 0 的 T2 记录 {t2_calendar_zero} 个改判为 T3',
         'T1 事件数': unified_counts['T1'] - old_counts['T1'],
         'T2 事件数': unified_counts['T2'] - old_counts['T2'],
         'T3 事件数': unified_counts['T3'] - old_counts['T3'],
         'Strict 周期数': unified_episodes - int(len(episodes_unique)),
         'Relaxed 周期数': unified_relaxed - 17678},
    ])
    print('阈值敏感性（统一日历天口径）：')
    print(sensitivity_table.to_string(index=False))

    # ---- Safe-F 退化证据与精简
    frame = model_frame.copy()
    frame['publish_month'] = frame_publish.dt.month.to_numpy()
    frame['publish_weekday'] = frame_publish.dt.weekday.to_numpy()
    rep_map = segments.loc[segments['segment_start'].eq(
        segments['intern_id'].map(entity_publish)),
        ['intern_id', 'episode_id_strict', 'episode_no_strict']].drop_duplicates('intern_id')
    rep_episodes = episodes_unique.merge(rep_map, on=['intern_id', 'episode_id_strict'],
                                         how='inner', suffixes=('', '_rep'))
    rep_episodes = rep_episodes.rename(columns={schema.ID_FIELD: 'intern_id'})
    t1_gap = segments.loc[segments['transition_tier'].eq('T1_CONFIRMED'),
                          ['intern_id', 'episode_id_strict', 'gap_days_exact']]
    rep_episodes = rep_episodes.merge(t1_gap, on=['intern_id', 'episode_id_strict'], how='left')
    rep_episodes = rep_episodes.rename(columns={'gap_days_exact': 'previous_reopen_gap_days'})
    rep_episodes['is_confirmed_reopen'] = rep_episodes['episode_no_strict_rep'].gt(1).astype(int)
    rep_episodes['historical_episode_count'] = rep_episodes['episode_no_strict_rep'] - 1
    safe_evidence = frame[[schema.ID_FIELD, 'publish_month', 'publish_weekday']].merge(
        rep_episodes[['intern_id', 'episode_no_strict_rep', 'is_confirmed_reopen',
                      'historical_episode_count', 'previous_reopen_gap_days']].rename(
            columns={'intern_id': schema.ID_FIELD,
                     'episode_no_strict_rep': 'episode_no_strict'}),
        on=schema.ID_FIELD, how='left')
    safe_rows = []
    for column, role in [('publish_month', '发布时间位置特征（保留）'),
                         ('publish_weekday', '发布时间位置特征（保留）'),
                         ('episode_no_strict', '生命周期历史变量（移出预测模型）'),
                         ('is_confirmed_reopen', '生命周期历史变量（移出预测模型）'),
                         ('historical_episode_count', '生命周期历史变量（移出预测模型）'),
                         ('previous_reopen_gap_days', '生命周期历史变量（移出预测模型）')]:
        series = safe_evidence[column]
        counts = series.value_counts(dropna=True)
        safe_rows.append({
            'Safe-F 特征': column, '处置': role,
            '缺失数（14,883）': int(series.isna().sum()),
            '非缺失唯一取值数': int(series.nunique(dropna=True)),
            '取值最多的取值': (float(counts.index[0]) if len(counts) else None),
            '取值最多的占比': (round(float(counts.iloc[0] / series.notna().sum()), 8)
                        if len(counts) else None)})
    safe_table = pd.DataFrame(safe_rows)
    multi_episode_jobs = int((rep_episodes['episode_no_strict_rep'] > 1).sum())
    safe_conclusion = pd.DataFrame([
        {'项目': 'Stage26.1 Safe-F 特征数', '数值': len(SAFE_F_STAGE26_1)},
        {'项目': 'Stage26.2 精简后 Safe-F 特征数', '数值': len(SAFE_F_FINAL)},
        {'项目': '精简后特征清单', '数值': '、'.join(SAFE_F_FINAL)},
        {'项目': '被移出预测模型的特征', '数值': '、'.join(SAFE_F_STAGE26_1[2:])},
        {'项目': '被移出特征的去向', '数值': '只保留在生命周期描述分析表（45 号表），不进入预测模型'},
        {'项目': '严格口径多周期岗位数（代表周期口径）', '数值': multi_episode_jobs},
        {'项目': 'previous_reopen_gap_days 缺失数', '数值': int(
            safe_evidence['previous_reopen_gap_days'].isna().sum())},
        {'项目': '正文口径要求', '数值': 'publish_month / publish_weekday 只解释为发布时间位置特征，'
                                 '不得解释为市场季节性'},
    ])

    # ---- 五层结构表述 / 3,410 措辞残留审计
    wording_rows = []
    if STAGE26_1_MOTHER.is_file():
        for number, line in enumerate(
                STAGE26_1_MOTHER.read_text(encoding='utf-8').splitlines(), start=1):
            for label, pattern in WORDING_PATTERNS:
                if re.search(pattern, line):
                    wording_rows.append({
                        '文件': str(STAGE26_1_MOTHER.relative_to(PROJECT_ROOT)).replace('\\', '/'),
                        '行号': number, '命中类型': label, '原文片段': line.strip()[:220]})
    wording_table = pd.DataFrame(wording_rows)
    print(f'五层 / 3,410 措辞残留命中：{len(wording_table)} 处')

    # ---------------------------------------------------------------- B 建模
    grouped = grouped_columns_with_safe_f(feature_manifest, frame)
    skill_map = model_training.build_skill_map(membership, skill_eda.ALL_USABLE_SCOPES)
    text_matrix = model_training.load_text_matrix(
        job_ids.tolist(), project_paths.FEATURES_DIR / 'job_text_embeddings.npz',
        project_paths.FEATURES_DIR / 'job_text_embedding_index.parquet')
    text_by_id = {job_id: position for position, job_id in enumerate(job_ids.tolist())}

    # ---- Safe-F 精简后的消融（Random Split）
    random_labels = split_definitions['Random Split（同分布泛化）']
    ablation_rows = []
    for label in ABLATION_ORDER:
        groups = ABLATION_CONFIGS[label]
        result = fit_eval(frame, random_labels, groups, grouped, skill_map, text_matrix,
                          text_by_id, SKILL_THRESHOLD, TEXT_DIM)
        ablation_rows.append({
            '配置': label, '特征组': '+'.join(groups), '特征维度': result['特征维度'],
            'validation_MAE': result['validation']['MAE'],
            'validation_RMSE': result['validation']['RMSE'],
            'validation_R2': result['validation']['R2'],
            'test_MAE': result['test']['MAE'], 'test_RMSE': result['test']['RMSE'],
            'test_R2': result['test']['R2'],
            'n_train': result['n_train'], 'n_validation': result['n_validation'],
            'n_test': result['n_test']})
        print(f"  消融 {label}: dim {result['特征维度']} | validation MAE "
              f"{result['validation']['MAE']} | test MAE {result['test']['MAE']}")
    ablation_table = pd.DataFrame(ablation_rows)
    test_mae = ablation_table['test_MAE']
    ablation_meta = pd.DataFrame([
        {'项目': 'test MAE 最小配置', '数值': ablation_table.loc[test_mae.idxmin(), '配置'],
         '取值': float(test_mae.min())},
        {'项目': 'test MAE 最大配置', '数值': ablation_table.loc[test_mae.idxmax(), '配置'],
         '取值': float(test_mae.max())},
        {'项目': 'test MAE 真实跨度', '数值': 'max − min',
         '取值': float(test_mae.max() - test_mae.min())},
        {'项目': 'validation MAE 真实跨度', '数值': 'max − min',
         '取值': float(ablation_table['validation_MAE'].max()
                     - ablation_table['validation_MAE'].min())},
        {'项目': 'A+B+C+D+E 编码后维度', '数值': '固定 A~E 特征组',
         '取值': int(ablation_table.set_index('配置').loc['A+B+C+D+E', '特征维度'])},
        {'项目': 'A+B+C+D+E+SafeF 编码后维度', '数值': '精简 Safe-F（2 个发布时间位置特征）',
         '取值': int(ablation_table.set_index('配置').loc['A+B+C+D+E+SafeF', '特征维度'])},
        {'项目': 'Safe-F 维度增量', '数值': '上两行之差', '取值': int(
            ablation_table.set_index('配置').loc['A+B+C+D+E+SafeF', '特征维度']
            - ablation_table.set_index('配置').loc['A+B+C+D+E', '特征维度'])},
    ])
    base_row = ablation_table.set_index('配置')
    delta_valid = float(base_row.loc['A+B+C+D+E', 'validation_MAE']
                        - base_row.loc['A+B+C+D+E+SafeF', 'validation_MAE'])
    delta_test = float(base_row.loc['A+B+C+D+E', 'test_MAE']
                       - base_row.loc['A+B+C+D+E+SafeF', 'test_MAE'])
    safe_increment = pd.DataFrame([
        {'数据子集': 'validation', 'MAE_无SafeF': float(base_row.loc['A+B+C+D+E', 'validation_MAE']),
         'MAE_有SafeF': float(base_row.loc['A+B+C+D+E+SafeF', 'validation_MAE']),
         'ΔMAE（无 − 有）': delta_valid,
         '方向': '正值表示加入精简 Safe-F 后误差下降' if delta_valid > 0 else '加入精简 Safe-F 后误差上升'},
        {'数据子集': 'test', 'MAE_无SafeF': float(base_row.loc['A+B+C+D+E', 'test_MAE']),
         'MAE_有SafeF': float(base_row.loc['A+B+C+D+E+SafeF', 'test_MAE']),
         'ΔMAE（无 − 有）': delta_test,
         '方向': '正值表示加入精简 Safe-F 后误差下降' if delta_test > 0 else '加入精简 Safe-F 后误差上升'},
    ])

    # ---- 统一协议：三种划分
    unified_rows = []
    protocol_cache = {}
    for label, labels, note in [
            ('Random Split（同分布泛化）', random_labels,
             '固定 model_splits 同一 Random Split；A+B+C+D+E；LightGBM 400/0.05/63；技能阈值 100；SVD 16'),
            ('Company Group Split（跨公司泛化）', split_definitions[
                'Company Group Split（跨公司泛化）'],
             '公司实体 GroupShuffleSplit（随机种子 42）；同公司不跨子集；其余与 Random 完全一致'),
            ('Retrospective Temporal Split（跨发布时间区间泛化）', split_definitions[
                'Retrospective Temporal Split（跨发布时间区间泛化）'],
             '按业务发布时间排序的回顾性划分（同一天不拆分，70%/15%/15%）；其余与 Random 完全一致')]:
        result = fit_eval(frame, labels, MODEL_FEATURE_GROUPS, grouped, skill_map,
                          text_matrix, text_by_id, SKILL_THRESHOLD, TEXT_DIM)
        protocol_cache[label] = result
        unified_rows.append({
            '划分方式': label, 'Train n': result['n_train'],
            'Validation n': result['n_validation'], 'Test n': result['n_test'],
            'MAE': result['test']['MAE'], 'RMSE': result['test']['RMSE'],
            'R²': result['test']['R2'],
            'validation MAE': result['validation']['MAE'],
            'validation RMSE': result['validation']['RMSE'],
            'validation R²': result['validation']['R2'],
            '特征维度': result['特征维度'], '技能列数': result['技能列数'], '说明': note})
        print(f"  统一协议 {label}: train {result['n_train']} / test {result['n_test']} | "
              f"test MAE {result['test']['MAE']} / RMSE {result['test']['RMSE']} / "
              f"R² {result['test']['R2']}")
    unified_table = pd.DataFrame(unified_rows)

    # ---- 旧两套 Company Group 指标来源差异
    group_threshold_rows = []
    for threshold in (50, 100):
        result = fit_eval(frame, split_definitions['Company Group Split（跨公司泛化）'],
                          MODEL_FEATURE_GROUPS, grouped, skill_map, text_matrix, text_by_id,
                          threshold, TEXT_DIM)
        group_threshold_rows.append({
            '技能频次阈值': threshold, '文本 SVD': TEXT_DIM,
            '技能列数': result['技能列数'], '特征维度': result['特征维度'],
            'validation MAE': result['validation']['MAE'],
            'test MAE': result['test']['MAE'], 'test RMSE': result['test']['RMSE'],
            'test R²': result['test']['R2'],
            '来源': ('Stage26.1 按 Temporal Split 验证集选定的阈值（旧两套中的 51.75）'
                   if threshold == 50 else
                   'Stage25 正式取值 / 本轮统一协议（旧两套中的 52.05）')})
    group_threshold_table = pd.DataFrame(group_threshold_rows)

    # ---- LightGBM vs CatBoost 多随机种子
    seed_rows = []
    for seed in SEEDS:
        seed_labels = pd.Series(
            model_training.build_splits(frame, random_state=seed)['split'].to_numpy(),
            index=job_ids)
        for model_key, params in [('LightGBM', LIGHTGBM_PARAMS), ('CatBoost', CATBOOST_PARAMS)]:
            result = fit_eval(frame, seed_labels, MODEL_FEATURE_GROUPS, grouped, skill_map,
                              text_matrix, text_by_id, SKILL_THRESHOLD, TEXT_DIM,
                              model_key=model_key, params=params, random_state=seed)
            seed_rows.append({
                '随机种子': seed, '模型': model_key,
                'validation_MAE': result['validation']['MAE'],
                'validation_RMSE': result['validation']['RMSE'],
                'validation_R2': result['validation']['R2'],
                'test_MAE': result['test']['MAE'], 'test_RMSE': result['test']['RMSE'],
                'test_R2': result['test']['R2'], 'n_validation': result['n_validation'],
                'n_test': result['n_test']})
            print(f'  种子 {seed} {model_key}: validation MAE {result["validation"]["MAE"]}')
    seed_table = pd.DataFrame(seed_rows)
    seed_summary_table = pd.DataFrame([
        {'模型': model_key, '随机种子数': int(len(SEEDS)),
         'validation_MAE_mean': round(float(block['validation_MAE'].mean()), 6),
         'validation_MAE_std': round(float(block['validation_MAE'].std(ddof=1)), 6),
         'validation_RMSE_mean': round(float(block['validation_RMSE'].mean()), 6),
         'validation_RMSE_std': round(float(block['validation_RMSE'].std(ddof=1)), 6),
         'validation_R2_mean': round(float(block['validation_R2'].mean()), 6),
         'validation_R2_std': round(float(block['validation_R2'].std(ddof=1)), 6),
         'test_MAE_mean': round(float(block['test_MAE'].mean()), 6),
         'test_MAE_std': round(float(block['test_MAE'].std(ddof=1)), 6)}
        for model_key, block in seed_table.groupby('模型', sort=False)])
    pivot = seed_table.pivot(index='随机种子', columns='模型', values='validation_MAE')
    lgb_wins = int((pivot['LightGBM'] < pivot['CatBoost']).sum())
    row_lgb = seed_summary_table.set_index('模型').loc['LightGBM']
    row_cat = seed_summary_table.set_index('模型').loc['CatBoost']
    seed_conclusion = pd.DataFrame([{
        '结论项': 'LightGBM vs CatBoost 多种子稳定性',
        'LightGBM validation MAE': f"{row_lgb['validation_MAE_mean']:.4f} ± "
                                   f"{row_lgb['validation_MAE_std']:.4f}",
        'CatBoost validation MAE': f"{row_cat['validation_MAE_mean']:.4f} ± "
                                   f"{row_cat['validation_MAE_std']:.4f}",
        'LightGBM 更优的种子数': f'{lgb_wins} / {len(SEEDS)}',
        '结论': ('LightGBM 在全部随机划分下均优于 CatBoost，主模型选择与多种子结果一致'
               if lgb_wins == len(SEEDS) else
               'LightGBM 与 CatBoost 表现接近，本文按预先设定的验证集 MAE 主指标选取 LightGBM 作为主模型')}])

    # ---- Company Group Split 5 次
    group_rows = []
    for seed in SEEDS:
        group_labels = pd.Series(model_training.build_company_group_split(
            frame, random_state=seed)['company_group_split'].to_numpy(), index=job_ids)
        result = fit_eval(frame, group_labels, MODEL_FEATURE_GROUPS, grouped, skill_map,
                          text_matrix, text_by_id, SKILL_THRESHOLD, TEXT_DIM,
                          random_state=SEED)
        group_rows.append({
            '随机种子': seed, 'Train n': result['n_train'], 'Validation n': result['n_validation'],
            'Test n': result['n_test'], 'test MAE': result['test']['MAE'],
            'test RMSE': result['test']['RMSE'], 'test R²': result['test']['R2'],
            'validation MAE': result['validation']['MAE']})
        print(f'  Company Group seed {seed}: test MAE {result["test"]["MAE"]}')
    group_table = pd.DataFrame(group_rows)
    random_reference = float(protocol_cache['Random Split（同分布泛化）']['test']['MAE'])
    group_summary = pd.DataFrame([{
        '项目': 'Company Group Split 5 次 GroupShuffleSplit',
        'test MAE mean ± std': f"{group_table['test MAE'].mean():.4f} ± "
                              f"{group_table['test MAE'].std(ddof=1):.4f}",
        'test RMSE mean ± std': f"{group_table['test RMSE'].mean():.4f} ± "
                               f"{group_table['test RMSE'].std(ddof=1):.4f}",
        'test R² mean ± std': f"{group_table['test R²'].mean():.6f} ± "
                             f"{group_table['test R²'].std(ddof=1):.6f}",
        '随机划分 test MAE（对照）': random_reference,
        'test MAE 取值范围': f"{group_table['test MAE'].min():.4f} ~ "
                          f"{group_table['test MAE'].max():.4f}",
        '结论': ('5 次公司分组划分的测试集 MAE 均高于随机划分，跨公司场景误差更高；'
               '但不同划分之间波动较大，正文以一次固定划分作为主结果，'
               '并同时报告多次划分的均值与波动'
               if group_table['test MAE'].min() > random_reference else
               '多次公司分组划分结果与随机划分接近，不足以支持跨公司误差更高的结论')}])

    # ---------------------------------------------------------------- C 补充分析
    # ---- 多变量中位数回归
    analysis_cert = eda_analysis.attach_company_certification(
        analysis[[schema.ID_FIELD, '工作城市', '学历要求', '每周到岗要求', '实习时长要求',
                  '公司规模', '公司性质', '所属行业', '岗位大类集合', '岗位细分类集合']],
        entity[[schema.ID_FIELD, schema.CERT_TAG_FIELD]])
    median_frame = analysis_cert.merge(
        model_frame[[schema.ID_FIELD, schema.SALARY_MID_FIELD]], on=schema.ID_FIELD, how='inner')
    design, design_names, references = build_median_design(median_frame)
    target = median_frame[schema.SALARY_MID_FIELD].to_numpy('float64')
    point = quantile_fit(design, target, 0.5)
    print(f'中位数回归：n = {len(median_frame):,}，设计矩阵 {design.shape[1]} 列，'
          f'参照类别截距 {point[0]:.4f}')

    rng = np.random.default_rng(SEED)
    bootstrap = np.empty((MEDIAN_BOOTSTRAP_ROUNDS, design.shape[1]), dtype='float64')
    for round_index in range(MEDIAN_BOOTSTRAP_ROUNDS):
        sample = rng.integers(0, len(target), len(target))
        bootstrap[round_index] = quantile_fit(design[sample], target[sample], 0.5)
    lower = np.percentile(bootstrap, 2.5, axis=0)
    upper = np.percentile(bootstrap, 97.5, axis=0)
    median_table = pd.DataFrame({
        '变量': design_names, '系数': np.round(point, 6),
        'bootstrap_CI95_下界': np.round(lower, 6), 'bootstrap_CI95_上界': np.round(upper, 6),
        '方向': np.where(point > 0, '正向', '负向'),
        '是否显著（CI 不跨 0）': np.where((lower > 0) | (upper < 0), '是', '否')})
    median_table['变量类型'] = np.where(
        median_table['变量'].eq('截距'), '截距',
        np.where(median_table['变量'].str.split('=').str[0].isin(MEDIAN_CATEGORICAL),
                 '单值类别（相对参照类别）', '多值类别（相对未命中）'))
    median_nonzero = median_table[~median_table['变量'].eq('截距')].copy()
    median_nonzero['绝对系数'] = median_nonzero['系数'].abs()
    median_top = median_nonzero.sort_values('绝对系数', ascending=False).head(20).drop(
        columns=['绝对系数']).reset_index(drop=True)
    significant_share = float(median_table['是否显著（CI 不跨 0）'].eq('是').mean())
    median_meta = pd.DataFrame([
        {'项目': '因变量', '数值': f'{schema.SALARY_MID_FIELD}（元/天），分位数 Q0.5'},
        {'项目': '实现', '数值': 'sklearn.linear_model.QuantileRegressor（quantile=0.5, '
                           'alpha=0.0, fit_intercept=False, solver="highs-ipm"，'
                           '设计矩阵首列显式包含截距）；statsmodels 未安装，未使用'},
        {'项目': '显著性口径', '数值': f'成对 bootstrap {MEDIAN_BOOTSTRAP_ROUNDS} 次（随机种子 '
                            f'{SEED}）的 2.5% / 97.5% 区间；区间不跨 0 记为显著'},
        {'项目': '纳入变量', '数值': '、'.join(MEDIAN_CATEGORICAL + MEDIAN_MULTI)},
        {'项目': '不纳入变量', '数值': '4,000+ 福利标签、全部技能、文本 SVD、高维公司 ID'},
        {'项目': '样本量', '数值': int(len(median_frame))},
        {'项目': '设计矩阵列数', '数值': int(design.shape[1])},
        {'项目': '分类变量参照类别', '数值': '；'.join(
            f'{key}={value}' for key, value in references.items())},
        {'项目': '系数显著项占比', '数值': round(significant_share, 6)},
        {'项目': '解释边界', '数值': '观察性多变量关联分析；不得表述为因果效应'},
    ])

    # ---- 面议岗选择偏差
    bias_frame = analysis[[schema.ID_FIELD, '岗位大类集合', '岗位细分类集合', '工作城市',
                           '公司规模', '公司性质', '所属行业', '学历要求']].merge(
        salary_targets[[schema.ID_FIELD, '是否面议', '薪资解析状态',
                        schema.SALARY_RAW_FIELD]], on=schema.ID_FIELD, how='left')
    public_ids = set(model_frame[schema.ID_FIELD])
    bias_frame['分组'] = np.where(
        bias_frame['是否面议'].eq(1), '面议',
        np.where(bias_frame[schema.ID_FIELD].isin(public_ids), '公开薪资', '解析异常剔除'))
    bias_compare = bias_frame[bias_frame['分组'].isin(['面议', '公开薪资'])].copy()
    bias_rows = []
    chi_rows = []
    for column in ['工作城市', '公司规模', '公司性质', '所属行业', '学历要求']:
        series = bias_compare[column].fillna('__未知__').astype(str)
        table = pd.crosstab(series, bias_compare['分组'])
        top_levels = table.sum(axis=1).sort_values(ascending=False).head(12).index
        trimmed = table.loc[top_levels]
        statistic, p_value, dof, used = _chi2_with_positive_cells(trimmed)
        chi_rows.append({'字段': column, '类型': '单值类别', '卡方统计量': round(statistic, 4),
                         '自由度': dof, 'p 值': p_value,
                         '是否显著（α = 0.05）': '是' if p_value < 0.05 else '否',
                         '参与检验的取值数': used, '高频取值总数': int(len(top_levels))})
        shares = trimmed.div(trimmed.sum(axis=0), axis=1)
        for level in top_levels:
            bias_rows.append({
                '字段': column, '取值': str(level),
                '面议占比': round(float(shares.loc[level, '面议']), 6),
                '公开薪资占比': round(float(shares.loc[level, '公开薪资']), 6),
                '占比之差（面议 − 公开）': round(float(shares.loc[level, '面议']
                                            - shares.loc[level, '公开薪资']), 6)})
    multi_bias_rows = []
    for column in ['岗位大类集合', '岗位细分类集合']:
        exploded = {group: bias_compare.loc[bias_compare['分组'].eq(group), column].explode()
                    .dropna().astype(str).value_counts()
                    for group in ('面议', '公开薪资')}
        levels = (exploded['面议'] / exploded['面议'].sum()).sort_values(
            ascending=False).head(12).index
        for level in levels:
            in_negotiable = float(exploded['面议'].get(level, 0) / exploded['面议'].sum())
            in_public = float(exploded['公开薪资'].get(level, 0) / exploded['公开薪资'].sum())
            observed = np.array([[exploded['面议'].get(level, 0),
                                  exploded['面议'].sum() - exploded['面议'].get(level, 0)],
                                 [exploded['公开薪资'].get(level, 0),
                                  exploded['公开薪资'].sum() - exploded['公开薪资'].get(level, 0)]])
            statistic, p_value = _chi2_2x2(observed)
            multi_bias_rows.append({
                '字段': column, '取值': str(level),
                '面议占比': round(in_negotiable, 6), '公开薪资占比': round(in_public, 6),
                '占比之差（面议 − 公开）': round(in_negotiable - in_public, 6),
                '卡方统计量': round(statistic, 4), 'p 值': p_value,
                '是否显著（α = 0.05）': '是' if p_value < 0.05 else '否'})
    multi_bias_table = pd.DataFrame(multi_bias_rows)
    if len(multi_bias_table):
        multi_bias_table['q 值（BH-FDR）'] = np.round(
            skill_eda.benjamini_hochberg(multi_bias_table['p 值'].to_numpy('float64')), 8)
    significant_single = sum(1 for row in chi_rows if row['是否显著（α = 0.05）'] == '是')
    significant_multi = int(multi_bias_table['q 值（BH-FDR）'].lt(0.05).sum())
    bias_summary = pd.DataFrame([
        {'项目': '面议岗位数', '数值': int(bias_frame['分组'].eq('面议').sum())},
        {'项目': '面议占全部岗位比例', '数值': round(float(bias_frame['分组'].eq('面议').mean()), 6)},
        {'项目': '公开薪资（建模样本）岗位数', '数值': int(bias_frame['分组'].eq('公开薪资').sum())},
        {'项目': '解析异常剔除岗位数', '数值': int(bias_frame['分组'].eq('解析异常剔除').sum())},
        {'项目': '单值字段卡方检验显著数', '数值': f'{significant_single} / {len(chi_rows)}'},
        {'项目': '多值字段显著取值数（BH-FDR 后 q < 0.05）', '数值': significant_multi},
        {'项目': '结论', '数值': '面议与公开薪资样本在岗位结构上存在系统性差异，'
                          '薪资建模样本存在由企业披露策略带来的选择性；'
                          '本轮不做面议薪资插补或预测'},
        {'项目': '实现口径', '数值': '多值岗位类别按 present / absent 逐取值做 2×2 卡方并做 BH-FDR；'
                           '单值字段按高频取值列联表做卡方检验'},
    ])

    # ---- 极端薪资源记录审计
    audit_columns = [schema.ID_FIELD, schema.SALARY_RAW_FIELD, schema.SALARY_MIN_FIELD,
                     schema.SALARY_MAX_FIELD, schema.SALARY_MID_FIELD,
                     schema.SALARY_UNIT_FIELD, schema.SALARY_ANOMALY_FIELD]
    audit_frame = salary_targets[audit_columns].merge(
        model_frame[[schema.ID_FIELD]], on=schema.ID_FIELD, how='inner')
    lowest = audit_frame.nsmallest(20, schema.SALARY_MID_FIELD).copy()
    highest = audit_frame.nlargest(20, schema.SALARY_MID_FIELD).copy()
    lowest['审计方向'] = '最低 20 条'
    highest['审计方向'] = '最高 20 条'
    extreme = pd.concat([lowest, highest], ignore_index=True)
    verdicts = extreme.apply(audit_verdict, axis=1, result_type='expand')
    extreme['解析判定'] = verdicts[0]
    extreme['异常类型'] = verdicts[1]
    anomaly_total = int(extreme['异常类型'].ne('无').sum())
    extreme_summary = pd.DataFrame([
        {'项目': '审计范围', '数值': '薪资建模样本（14,883）内最低 20 条与最高 20 条'},
        {'项目': '最小值', '数值': float(audit_frame[schema.SALARY_MID_FIELD].min())},
        {'项目': '最大值', '数值': float(audit_frame[schema.SALARY_MID_FIELD].max())},
        {'项目': '极端 40 条的单位分布', '数值': '；'.join(
            f'{key}: {value}' for key, value in extreme[
                schema.SALARY_UNIT_FIELD].value_counts().items())},
        {'项目': '极端样本中判定需复核 / 异常条数', '数值': anomaly_total},
        {'项目': '薪资目标层被标注逻辑异常的记录数（全量）', '数值': int(
            (~salary_targets[schema.SALARY_ANOMALY_FIELD].astype(str).str.strip().isin(
                ['', 'nan', 'None'])).sum())},
        {'项目': '是否发现解析异常', '数值': ('未发现单位误判、上下限倒置或月薪 / 时薪误解析'
                                if anomaly_total == 0 else
                                f'发现 {anomaly_total} 条需复核记录，需人工确认后再决定是否修正')},
        {'项目': '处理决定', '数值': '本轮不修改原始数据；若需修正，影响面仅限被标注的极少数记录，'
                              '并需同步重跑受影响的描述统计与模型结果（本轮未重算全链路）'},
    ])

    # ---------------------------------------------------------------- D 输出
    io_utils.write_excel(TABLES / TABLE_FILES[0], {
        '01_三种划分样本量': split_table,
        '02_T2口径与gap实测': gap_table,
        '03_T2日历天分布': t2_calendar_dist,
        '04_阈值敏感性_统一口径': sensitivity_table,
        '05_口径对比与变化': caliber_table,
        '06_SafeF退化与精简': pd.concat([safe_table, safe_conclusion], ignore_index=True,
                                   sort=False),
        '07_消融结果（精简SafeF）': ablation_table,
        '08_消融跨度与维度': pd.concat([ablation_meta, safe_increment], ignore_index=True,
                                 sort=False)})
    print(f'49 号表已写入；T2 日历天为 0 的记录 {t2_calendar_zero} 个；'
          f'test MAE 新跨度 {float(test_mae.max() - test_mae.min()):.6f}')

    io_utils.write_excel(TABLES / TABLE_FILES[1], {
        '01_统一协议三种划分': unified_table,
        '02_CompanyGroup阈值来源': group_threshold_table,
        '03_多种子明细': seed_table,
        '04_多种子汇总': seed_summary_table,
        '05_多种子结论': seed_conclusion,
        '06_CompanyGroup5次划分': group_table,
        '07_CompanyGroup汇总': group_summary})
    print('50 号表已写入')

    io_utils.write_excel(TABLES / TABLE_FILES[2], {
        '01_中位数回归全表': median_table, '02_主要变量Top20': median_top,
        '03_口径说明': median_meta})
    print('51 号表已写入')

    io_utils.write_excel(TABLES / TABLE_FILES[3], {
        '01_结构对照_单值': pd.DataFrame(bias_rows),
        '02_卡方检验_单值': pd.DataFrame(chi_rows),
        '03_结构对照_多值': multi_bias_table,
        '04_结论与口径': bias_summary})
    print('52 号表已写入')

    io_utils.write_excel(TABLES / TABLE_FILES[4], {
        '01_极端值审计': extreme[['审计方向', schema.ID_FIELD, schema.SALARY_RAW_FIELD,
                              schema.SALARY_MIN_FIELD, schema.SALARY_MAX_FIELD,
                              schema.SALARY_MID_FIELD, schema.SALARY_UNIT_FIELD,
                              schema.SALARY_ANOMALY_FIELD, '解析判定', '异常类型']],
        '02_审计结论': extreme_summary})
    print('53 号表已写入')

    io_utils.write_excel(TABLES / TABLE_FILES[5], {
        '01_表述残留命中': wording_table,
        '02_统一口径建议': pd.DataFrame([
            {'项目': '正式五层结构',
             '内容': '观测 → 完整页面版本 → 候选发布时间段 → 正式招聘周期 → 岗位实体'},
            {'项目': '图 3-3 建议图题',
             '内容': '岗位观测向唯一岗位实体的核心压缩路径（候选发布时间段与招聘周期'
                     '属生命周期分析的中间层）'},
            {'项目': '减少量正确口径',
             '内容': 'T2 与 T3 均在严格口径下并入当前周期，共减少 3,410 个候选发布时间段'},
            {'项目': 'T2 口径',
             '内容': 'gap 以精确时间差计算后转换为日历天；相差 0 天归入 T3，'
                     'T2 区间为 0 < gap < 7 日历天'}])})
    print('54 号表已写入')

    # ---------------------------------------------------------------- 图件
    plot_style.setup_sci_style()
    registry: list = []
    figure_ablation(ablation_table, registry)
    figure_duration(episodes_unique, registry)
    daily_table = pd.read_excel(LIFECYCLE_TABLE, sheet_name='08_日级指标明细')
    figure_daily(daily_table, registry)
    figure_category(build_category_daily(), daily_table, registry)
    for item in registry:
        item['failed_paper_gates'] = figure_finalize.failed_paper_gates(item)
        print(f"  图 {item['stem']}：{item['png_size_bytes']:,} 字节 / "
              f"{item['png_pixel_size']} px / 未通过门禁 {item['failed_paper_gates'] or '无'}")
    figure_finalize.write_registry(REGISTRY_PATH, {
        'stage': 'Stage26.2', 'seed': SEED,
        '图件': [{key: item[key] for key in
                 ('stem', 'png_path', 'pdf_path', 'png_pixel_size', 'png_dpi', 'png_size_bytes',
                  'pdf_size_bytes', 'no_infigure_caption', 'subfigure_caption_visible',
                  'failed_paper_gates')} for item in registry]})

    # ---------------------------------------------------------------- 指标 JSON
    manifest_after = project_manifest()
    changed = sorted(key for key in set(manifest_before) & set(manifest_after)
                     if manifest_before[key] != manifest_after[key])
    new_files = [input_record(path) for path in NEW_FILES
                 if path.exists() and path != METRICS_PATH]
    inputs = [project_paths.JOB_SALARY_MODEL_DATASET_PARQUET,
              project_paths.MODEL_SPLITS_PARQUET,
              project_paths.JOB_SKILL_MEMBERSHIP_PARQUET,
              project_paths.JOB_ANALYSIS_DATASET_PARQUET,
              project_paths.PROCESSED_UNIQUE_PARQUET,
              project_paths.SALARY_TARGETS_PARQUET, SEGMENT_PATH, EPISODE_PATH, PANEL_PATH,
              project_paths.FEATURES_DIR / 'job_text_embeddings.npz',
              project_paths.FEATURES_DIR / 'job_text_embedding_index.parquet',
              project_paths.SALARY_MODEL_DIR / 'feature_manifest.json',
              LIFECYCLE_TABLE, STAGE26_1_MOTHER]
    payload = {
        'stage': 'Stage26.2',
        '脚本路径': 'scripts/26c_stage26_2_final_consolidation.py',
        '运行命令': ('E:\\anaconda3\\envs\\reptile\\python.exe '
                 'scripts\\26c_stage26_2_final_consolidation.py'),
        '运行时间': {'开始': datetime.fromtimestamp(started).isoformat(timespec='seconds'),
                 '结束': datetime.now().isoformat(timespec='seconds')},
        '随机种子': {'固定': SEED, '多种子': list(SEEDS)},
        '统一协议': {'模型': 'LightGBM', '参数': LIGHTGBM_PARAMS,
                 '特征组': list(MODEL_FEATURE_GROUPS), '技能频次阈值': SKILL_THRESHOLD,
                 '文本SVD': TEXT_DIM, '拟合范围': '仅 train（预处理与估计器均只在 train 拟合）'},
        '事实核实': {
            '三种划分样本量': split_table.to_dict('records'),
            'T2口径实测': gap_table.to_dict('records'),
            'T2日历天为0的记录数': t2_calendar_zero,
            '统一口径三档计数': unified_counts,
            '统一口径Strict周期数': unified_episodes,
            '统一口径Relaxed周期数': unified_relaxed,
            '阈值敏感性': sensitivity_table.to_dict('records'),
            'SafeF精简': safe_conclusion.to_dict('records'),
            '消融新MAE跨度': float(test_mae.max() - test_mae.min()),
            '表述残留命中数': int(len(wording_table))},
        '消融结果': ablation_table.to_dict('records'),
        '统一协议结果': unified_table.to_dict('records'),
        'CompanyGroup阈值来源': group_threshold_table.to_dict('records'),
        '多种子': {'明细': seed_table.to_dict('records'),
                '汇总': seed_summary_table.to_dict('records'),
                '结论': seed_conclusion.to_dict('records')[0]},
        'CompanyGroup5次划分': {'明细': group_table.to_dict('records'),
                            '汇总': group_summary.to_dict('records')[0]},
        '中位数回归': {'口径': median_meta.to_dict('records'),
                   'Top20': median_top.to_dict('records')},
        '面议选择偏差': {'结论': bias_summary.to_dict('records'), '单值卡方': chi_rows},
        '极端值审计': {'结论': extreme_summary.to_dict('records'), '需复核条数': anomaly_total},
        '新增文件': new_files,
        '既有文件完整性': {'运行前清单文件数': len(manifest_before),
                    '运行后既有文件被修改数': len(changed), '被修改路径': changed},
        '输入文件SHA256': [input_record(path) for path in inputs if path.is_file()],
        '论文正文未修改': True,
    }
    io_utils.write_json(METRICS_PATH, payload)
    print(f'指标 JSON 已写入；既有文件被修改数 = {len(changed)}')
    print(f'Stage26.2 完成，用时 {time.time() - started:.1f} 秒')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

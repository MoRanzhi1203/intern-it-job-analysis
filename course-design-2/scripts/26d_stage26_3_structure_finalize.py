# -*- coding: utf-8 -*-
"""Stage26.3：正文结构精简配套的**必要重算**与文本审计（只新增文件）。

本脚本严格只读既有产物（`data/**`、`outputs/**` 既有文件、`src/**`、既有 `scripts/**`），
只做新增计算与新增文件输出，不覆盖任何既有文件。对应提示词
`docs/prompts/Trae_Stage26_3_结构精简与图文规范终修提示词.md` 的以下条款：

1. §19 时点可得性：把 `核心版本数`、`完整页面版本数`、`是否多版本岗位` 三个发布时点不可得的
   后验特征从正式预测特征 A 组中**移除**；
2. §20~§22 建模前诊断：缺失率、近常量、低频类别、稀有标签、连续数值特征 Spearman 冗余、
   Train / Validation / Test 基本分布对照；
3. §19 / §32 Step 5：移除后验特征后重跑统一协议下 Random / Company Group / Retrospective
   Temporal 三种划分的正式指标、特征组维度、消融五配置（含增量配对 bootstrap）、
   模型比较、最终模型测试集结果、分组预测误差、多种子稳定性与 Company Group 多次划分；
4. §11 正文段内公式化：生成逐文件逐行的**内联数学符号清单**（供后续构建环节实现，
   本环节不改造构建脚本）；
5. §5~§14 文本结构与文风审计：Q 简称、机械总起段、小结篇幅、加粗短句、开发过程语言、
   提示式前缀、引号 / 括号 / 破折号 / 无序列表的逐项命中数对照。

运行::

    python scripts\\26d_stage26_3_structure_finalize.py baseline
    python scripts\\26d_stage26_3_structure_finalize.py compute
    python scripts\\26d_stage26_3_structure_finalize.py inventory

新增输出::

    outputs/tables/47_stage26_3_inline_math_inventory.xlsx
    outputs/tables/48_stage26_3_pre_model_diagnostics.xlsx
    outputs/tables/49_stage26_3_metrics_after_feature_removal.xlsx
    outputs/tables/50_stage26_3_text_revision_audit.xlsx
    outputs/figures/supplementary/fig_s32_pre_model_diagnostics_subset_distribution.png/pdf
    outputs/logs/metrics/stage_26_3_structure_finalize.json
    outputs/logs/metrics/stage_26_3_text_revision.json
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import (ablation_shap, figure_finalize, io_utils, model_training,  # noqa: E402
                 plot_style, project_paths, schema, skill_eda)
from src.script_support import (build_assembler, dump_json, fit_eval,  # noqa: E402
                                       grouped_columns, section_length, sha256_of)

# ============================================================================
# 常量与口径（与 Stage26.2 统一协议保持一致，便于逐项对照）
# ============================================================================
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
POSTERIOR_FEATURES = ['核心版本数', '完整页面版本数', '是否多版本岗位']
REDUNDANT_FEATURES = ['岗位描述分词数', '公司规模下限', '公司规模上限', '公司规模中点']
NEAR_CONSTANT_FEATURES = ['公司规模是否已知', '公司规模槽位异常标志']
REMOVE_REASONS = {
    '核心版本数': '发布时点不可得：一个岗位最终有多少版本只有后续观测才能知道',
    '完整页面版本数': '发布时点不可得：完整页面版本数依赖后续页面变化',
    '是否多版本岗位': '发布时点不可得：是否多版本同样由后续页面变化决定',
    '岗位描述分词数': '连续数值冗余：与岗位描述字符数的 Spearman ρ = 0.976632，语义重复，'
                 '保留不受分词器影响的岗位描述字符数',
    '公司规模下限': '连续数值冗余：与公司规模等级的 Spearman ρ = 0.999994，保留公司规模等级',
    '公司规模上限': '连续数值冗余：与公司规模中点的 Spearman ρ = 1.000000，且缺失率 49.01%',
    '公司规模中点': '连续数值冗余：与公司规模下限的 Spearman ρ = 0.999884，且缺失率 49.01%',
    '公司规模是否已知': '近常量：14,883 个样本中取值最多的类别占比 99.97%，无判别信息',
    '公司规模槽位异常标志': '近常量：14,883 个样本中取值最多的类别占比 99.97%，无判别信息',
}
REMOVED_FEATURES = POSTERIOR_FEATURES + REDUNDANT_FEATURES + NEAR_CONSTANT_FEATURES
ABLATION_CONFIGS = {
    'A+B+C': ('A', 'B', 'C'),
    'A+B+C+D': ('A', 'B', 'C', 'D'),
    'A+B+C+E': ('A', 'B', 'C', 'E'),
    'A+B+C+D+E': ('A', 'B', 'C', 'D', 'E'),
    'A+B+C+D+E+SafeF': ('A', 'B', 'C', 'D', 'E', 'SafeF'),
}
ABLATION_ORDER = list(ABLATION_CONFIGS)
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
SCALE_FOR = {'Ridge'}
BOOTSTRAP_ROUNDS = 1000
BOOTSTRAP_SEED = 42
MIN_CATEGORY_FREQUENCY = model_training.MIN_CATEGORY_FREQUENCY
TIER1_CITIES = ['北京', '上海', '深圳', '广州']
CONTINUOUS_NUMERIC = [
    '岗位描述字符数', '岗位描述分词数', '技能数量', '岗位标签数量', '公司简介字符数',
    '岗位大类数量', '岗位细分类数量', '公司标签数量', '技术技能数', 'AI技能数',
    '大模型技能数', '编程语言技能数', '数据库技能数', '安全技能数', '工程工具数',
    '数据工具数', '业务能力数', '技术领域数', '办公工具技能数', '学历等级',
    '公司规模下限', '公司规模上限', '公司规模中点', '公司规模等级',
    '每周到岗天数', '实习月数',
]
REDUNDANCY_DECISIONS = {
    frozenset(('公司规模下限', '公司规模等级')): '语义重复（ρ = 0.999994），保留公司规模等级',
    frozenset(('公司规模上限', '公司规模中点')): '语义重复（ρ = 1.000000），两者均移出',
    frozenset(('公司规模下限', '公司规模上限')): '语义重复（ρ = 0.999884），两者均移出',
    frozenset(('公司规模下限', '公司规模中点')): '语义重复（ρ = 0.999884），两者均移出',
    frozenset(('公司规模上限', '公司规模等级')): '语义重复（ρ = 0.999844），移出公司规模上限',
    frozenset(('公司规模中点', '公司规模等级')): '语义重复（ρ = 0.999845），移出公司规模中点',
    frozenset(('岗位描述字符数', '岗位描述分词数')):
        '语义重复（ρ = 0.976632），保留岗位描述字符数，移出岗位描述分词数',
    frozenset(('数据工具数', '办公工具技能数')): '语义不同（数据工具与办公工具），两者均保留',
}

TABLES = project_paths.TABLES_DIR
SUPP_DIR = project_paths.FIGURES_DIR / 'supplementary'
METRICS_DIR = project_paths.METRICS_DIR
SOURCE_DIR = PROJECT_ROOT / 'docs' / 'paper' / 'stage23'
TABLE_INLINE_MATH = TABLES / '47_stage26_3_inline_math_inventory.xlsx'
TABLE_DIAGNOSTICS = TABLES / '48_stage26_3_pre_model_diagnostics.xlsx'
TABLE_METRICS = TABLES / '49_stage26_3_metrics_after_feature_removal.xlsx'
TABLE_TEXT_AUDIT = TABLES / '50_stage26_3_text_revision_audit.xlsx'
METRICS_PATH = METRICS_DIR / 'stage_26_3_structure_finalize.json'
TEXT_METRICS_PATH = METRICS_DIR / 'stage_26_3_text_revision.json'
TEXT_BASELINE_PATH = METRICS_DIR / 'stage_26_3_text_baseline.json'
FIG_STEM = 'fig_s32_pre_model_diagnostics_subset_distribution'
FIG_FLOW = 'fig_s33_salary_model_build_eval_flow'
FLOW_STEPS = [
    ('14,883 个正式薪资样本', '由 17,144 个唯一岗位实体中标注明确薪资的岗位构成'),
    ('A / B / C / D / E 特征组与 Safe-F 发布时间位置特征', '岗位基础、地域、公司、技能、文本语义与发布时间位置'),
    ('建模前特征诊断', '缺失率、近常量、低频类别、连续数值冗余与子集分布对照'),
    ('Train / Validation / Test 分层划分', '按薪资十分位分层，70% / 15% / 15%，随机种子 42'),
    ('仅用训练集拟合全部预处理', '类别编码、技能频次筛选与文本语义降维'),
    ('Ridge / RandomForest / CatBoost / LightGBM 与常数基线', '同一划分、同一特征、同一候选超参数网格'),
    ('按验证集平均绝对误差选定主模型', '锁定 LightGBM：400 棵树、学习率 0.05、叶子数 63'),
    ('训练集与验证集重拟合，测试集一次性评估', '测试集不参与主模型、超参数与特征配置的选择'),
    ('特征组消融、公司分组划分、回顾性时间划分与 SHAP 解释', '稳健性、跨公司与跨发布时间区间泛化、模型判断依据'),
]

CHAPTERS = ['00_摘要与Abstract.md', '01_绪论.md', '02_相关理论与分析方法.md',
            '03_数据获取与预处理.md', '04_互联网IT实习岗位特征分析.md',
            '05_实习岗位薪资影响因素分析.md', '06_实习岗位技能需求分析.md',
            '07_薪资预测模型构建与结果分析.md', '08_模型稳健性与解释.md',
            '09_总结与展望.md']

NEW_FILES = [TABLE_INLINE_MATH, TABLE_DIAGNOSTICS, TABLE_METRICS, TABLE_TEXT_AUDIT,
             SUPP_DIR / f'{FIG_STEM}.png', SUPP_DIR / f'{FIG_STEM}.pdf',
             SUPP_DIR / f'{FIG_FLOW}.png', SUPP_DIR / f'{FIG_FLOW}.pdf',
             METRICS_PATH, TEXT_METRICS_PATH, TEXT_BASELINE_PATH]
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
            if path.resolve() in new_resolved:
                continue
            if path == Path(__file__).resolve():
                continue
            manifest[str(path.relative_to(PROJECT_ROOT)).replace('\\', '/')] = sha256_of(path)
    readme = PROJECT_ROOT / 'README.md'
    if readme.is_file() and readme.resolve() not in new_resolved:
        manifest['README.md'] = sha256_of(readme)
    return manifest


def write_excel(path: Path, sheets: dict) -> None:
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for name, frame in sheets.items():
            value = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
            value.to_excel(writer, sheet_name=name[:31], index=False)


# ============================================================================
# 特征装配（移除后验特征）
# ============================================================================
def drop_posterior(grouped: dict) -> dict:
    """从全部特征组中移除发布时点不可得、连续数值冗余与近常量的特征。"""
    cleaned = {}
    for letter, spec in grouped.items():
        cleaned[letter] = {
            key: [column for column in spec[key] if column not in REMOVED_FEATURES]
            for key in ('numeric', 'categorical', 'multi')}
    return cleaned


def paired_bootstrap(y_true, pred_without, pred_with, rounds: int = BOOTSTRAP_ROUNDS,
                     seed: int = BOOTSTRAP_SEED) -> dict:
    """配对 bootstrap 的 ΔMAE（不含该组 − 含该组），正值表示加入后误差下降。"""
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
    return {
        'ΔMAE': round(float(np.abs(truth - left).mean() - np.abs(truth - right).mean()), 6),
        'CI95下界': round(float(low), 6), 'CI95上界': round(float(high), 6),
        '是否跨0': '是' if low <= 0 <= high else '否',
    }


# ============================================================================
# 建模前诊断（§20~§22）
# ============================================================================
def missing_rate(frame: pd.DataFrame, column: str) -> float:
    series = frame[column]
    values = series.map(lambda item: len(item) == 0 if isinstance(item, (list, tuple))
                        else (item is None or (isinstance(item, float) and np.isnan(item))
                              or (isinstance(item, str) and item.strip() == '')))
    return float(np.mean(values.to_numpy(dtype=bool)))


def build_diagnostics(frame: pd.DataFrame, feature_manifest: dict,
                      labels: pd.Series, membership: pd.DataFrame) -> dict:
    """建模前特征诊断（对进入模型的候选特征全量诊断，并标注最终处置）。"""
    numeric = list(feature_manifest['numeric_columns'])
    categorical = list(feature_manifest['categorical_columns'])
    multi = list(feature_manifest['multi_value_columns'])
    removed = set(REMOVED_FEATURES)

    missing_rows = []
    for column in numeric + categorical + multi:
        missing_rows.append({
            '特征': column, '类型': ('数值' if column in numeric
                                 else ('类别' if column in categorical else '多值')),
            '缺失率': round(missing_rate(frame, column), 6),
            '是否进入最终模型': '否' if column in removed else '是'})
    missing_table = pd.DataFrame(missing_rows).sort_values('缺失率', ascending=False)

    variance_rows = []
    for column in numeric:
        series = pd.to_numeric(frame[column], errors='coerce')
        counts = series.value_counts(dropna=True)
        share = float(counts.iloc[0] / max(series.notna().sum(), 1))
        variance_rows.append({
            '特征': column, '非缺失唯一取值数': int(series.nunique(dropna=True)),
            '取值最多的占比': round(share, 6), '标准差': round(float(series.std()), 6),
            '判定': ('近常量（取值最多的占比 ≥ 0.99）' if share >= 0.99 else '正常'),
            '是否进入最终模型': '否' if column in removed else '是'})
    variance_table = pd.DataFrame(variance_rows).sort_values('取值最多的占比', ascending=False)

    category_rows = []
    for column in categorical:
        series = frame[column].fillna('__未知__').astype(str)
        counts = series.value_counts()
        rare = counts[counts < MIN_CATEGORY_FREQUENCY]
        category_rows.append({
            '类别字段': column, '取值数': int(len(counts)),
            '低频取值数（< 20）': int(len(rare)),
            '低频取值样本占比': round(float(rare.sum() / len(series)), 6),
            '处理': '低频取值编码为其他类别' if len(rare) else '无需处理'})
    category_table = pd.DataFrame(category_rows).sort_values('低频取值数（< 20）',
                                                            ascending=False)

    tag_counts: dict = {}
    for items in frame['公司标签列表']:
        for item in (items if isinstance(items, (list, tuple, np.ndarray)) else []):
            tag_counts[str(item)] = tag_counts.get(str(item), 0) + 1
    tag_series = pd.Series(tag_counts).sort_values(ascending=False)
    tag_table = pd.DataFrame([
        {'检查项': '公司福利标签原子取值数', '数值': int(len(tag_series))},
        {'检查项': '出现次数 ≥ 50 的标签数', '数值': int((tag_series >= 50).sum())},
        {'检查项': '出现次数 < 5 的标签数（稀有标签）', '数值': int((tag_series < 5).sum())},
        {'检查项': '仅出现 1 次的标签数', '数值': int((tag_series == 1).sum())},
        {'检查项': '处理', '数值': '只保留高频取值展开为指示列，低频取值归入其他类别'}])

    train_ids = frame.loc[labels.reindex(frame[schema.ID_FIELD]).to_numpy() == 'train',
                          schema.ID_FIELD]
    train_membership = membership[membership['match_scope'].isin(skill_eda.ALL_USABLE_SCOPES)
                                  & membership['intern_id'].isin(set(train_ids))]
    frequency = train_membership.groupby('canonical_skill')['intern_id'].nunique()
    skill_table = pd.DataFrame([
        {'检查项': '技能主/扩展口径可命中技能条目数', '数值': int(frequency.size)},
        {'检查项': '训练集频次 ≥ 100（保留为模型列）', '数值': int((frequency >= 100).sum())},
        {'检查项': '训练集频次 < 100（不进入模型列）', '数值': int((frequency < 100).sum())},
        {'检查项': '训练集频次 < 10', '数值': int((frequency < 10).sum())},
        {'检查项': '阈值口径', '数值': '技能列只保留训练集频次 ≥ 100 的技能，'
                                 '该阈值只在训练集上计算'}])

    pairs = [column for column in CONTINUOUS_NUMERIC if column in frame.columns]
    matrix = frame[pairs].apply(pd.to_numeric, errors='coerce')
    correlation = matrix.corr(method='spearman')
    redundancy_rows = []
    for left_index, left in enumerate(pairs):
        for right in pairs[left_index + 1:]:
            value = float(correlation.loc[left, right])
            if abs(value) > 0.85:
                redundancy_rows.append({
                    '特征一': left, '特征二': right, 'Spearman ρ': round(value, 6),
                    '|ρ| > 0.85': '是',
                    '人工检查结论': REDUNDANCY_DECISIONS.get(
                        frozenset((left, right)), '需人工判断')})
    redundancy_table = pd.DataFrame(redundancy_rows).sort_values(
        'Spearman ρ', key=lambda series: series.abs(), ascending=False)
    correlation_table = correlation.round(6).reset_index().rename(columns={'index': '特征'})

    categories = [set(items) if isinstance(items, (list, tuple, np.ndarray)) else set()
                  for items in frame['岗位大类集合']]
    category_array = np.array(categories, dtype=object)
    metrics_by_subset = {}
    for subset in ('train', 'validation', 'test'):
        mask = (labels.reindex(frame[schema.ID_FIELD]).to_numpy() == subset)
        salary = frame.loc[mask, schema.SALARY_MID_FIELD]
        skills = frame.loc[mask, '技能数量']
        description = frame.loc[mask, '岗位描述字符数']
        city = frame.loc[mask, '工作城市_规范'].astype(str)
        tier1 = float(np.mean(city.isin(TIER1_CITIES).to_numpy()))
        ai = float(np.mean([('人工智能' in item) for item in category_array[mask]]))
        metrics_by_subset[subset] = {
            '样本量': int(mask.sum()),
            '薪资中位数': float(salary.median()),
            '薪资IQR': float(salary.quantile(0.75) - salary.quantile(0.25)),
            '平均技能数': round(float(skills.mean()), 4),
            '岗位描述字符数中位数': float(description.median()),
            '一线城市占比': round(tier1, 6),
            '人工智能岗位占比': round(ai, 6),
        }
    distribution_table = pd.DataFrame(
        [{'指标': key, 'Train': metrics_by_subset['train'][key],
          'Validation': metrics_by_subset['validation'][key],
          'Test': metrics_by_subset['test'][key]} for key in metrics_by_subset['train']])

    return {
        'missing': missing_table, 'variance': variance_table, 'category': category_table,
        'tag': tag_table, 'skill': skill_table, 'redundancy': redundancy_table,
        'correlation': correlation_table, 'correlation_matrix': correlation,
        'distribution': distribution_table, 'pairs': pairs,
        'metrics_by_subset': metrics_by_subset,
    }


# ============================================================================
# 图件：建模前特征诊断与子集分布对照
# ============================================================================
def figure_flow() -> dict:
    """图 7-1 薪资预测模型构建与评估流程（简洁学术流程图）。"""
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(11.0, 15.2))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis('off')
    count = len(FLOW_STEPS)
    height = 0.072
    gap = (0.96 - 0.03 - count * height) / (count - 1)
    for index, (title, detail) in enumerate(FLOW_STEPS):
        top = 0.96 - index * (height + gap)
        box = Rectangle((0.075, top - height), 0.85, height,
                        facecolor='#f2f2f2' if index % 2 == 0 else '#e8eef4',
                        edgecolor='black', linewidth=0.9)
        ax.add_patch(box)
        ax.text(0.5, top - height * 0.38, title, ha='center', va='center', fontsize=13.0)
        ax.text(0.5, top - height * 0.76, detail, ha='center', va='center',
                fontsize=10.5, color='#404040')
        if index < count - 1:
            arrow = FancyArrowPatch((0.5, top - height), (0.5, top - height - gap),
                                    arrowstyle='-|>', mutation_scale=16.0,
                                    linewidth=1.1, color='black')
            ax.add_patch(arrow)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    diagnostics = figure_finalize.save_paper_figure(
        fig, SUPP_DIR, FIG_FLOW,
        subfigures=[('a', '薪资预测模型构建与评估流程', ax)],
        meta={'数据来源': '本文第 7 章与第 8 章的建模流程',
              '口径': '流程图只描述正式流程，不含任何统计结果',
              'seed': SEED, '用途': '第7章 图 7-1（替代原模型验证集误差比较图）'})
    plt.close(fig)
    return diagnostics


def figure_diagnostics(diagnostics: dict) -> dict:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plt.rcParams['font.size'] = 11.5
    plt.rcParams['axes.labelsize'] = 12.0
    plt.rcParams['xtick.labelsize'] = 11.0
    plt.rcParams['ytick.labelsize'] = 11.0
    plt.rcParams['legend.fontsize'] = 11.0
    correlation = diagnostics['correlation_matrix']
    fig, axes = plt.subplots(2, 1, figsize=(12.6, 15.4),
                             gridspec_kw={'height_ratios': [1.0, 1.35]})

    ax = axes[0]
    distribution = diagnostics['distribution'].set_index('指标')
    values = distribution[['Train', 'Validation', 'Test']]
    labels = ['样本量', '薪资中位数', '薪资IQR', '平均技能数', '岗位描述字符数中位数',
              '一线城市占比', '人工智能岗位占比']
    values = values.loc[labels]
    scaled = values.div(values.max(axis=1), axis=0)
    positions = np.arange(len(labels))
    width = 0.26
    for offset, subset, color in [(-width, 'Train', plot_style.MAIN_COLOR),
                                  (0.0, 'Validation', plot_style.ACCENT_COLOR),
                                  (width, 'Test', plot_style.PALETTE[2])]:
        bars = ax.bar(positions + offset, scaled[subset], width, color=color,
                      edgecolor='black', linewidth=0.6, label=subset)
        for bar, raw in zip(bars, values[subset]):
            ax.annotate(f'{raw:,.4g}', xy=(bar.get_x() + bar.get_width() / 2,
                                           bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center',
                        va='bottom', fontsize=9.0, rotation=90)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=14, ha='right')
    ax.set_xlabel('诊断指标')
    ax.set_ylabel('子集内相对值（各组最大值归一）')
    ax.set_ylim(0, 1.42)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.004), ncol=3, frameon=False)
    plot_style.apply_sci_axis(ax, grid_axis='y')
    plot_style.add_subfigure_caption(ax, 'a', 'Train / Validation / Test 基本分布对照'
                                             '（柱顶为子集内原始取值）')

    ax = axes[1]
    matrix = correlation.to_numpy('float64')
    image = ax.imshow(matrix, cmap='RdBu_r', vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(correlation.columns)))
    ax.set_xticklabels(correlation.columns, rotation=90, fontsize=9.0)
    ax.set_yticks(np.arange(len(correlation.columns)))
    ax.set_yticklabels(correlation.columns, fontsize=9.0)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, f'{matrix[row, column]:.2f}', ha='center', va='center',
                    fontsize=6.0,
                    color='white' if abs(matrix[row, column]) > 0.6 else 'black')
    ax.set_title(f'连续数值特征的 Spearman 相关（{len(correlation.columns)} 个特征，'
                 f'共 {len(correlation.columns) * (len(correlation.columns) - 1) // 2} 对），'
                 f'不做全特征相关矩阵', fontsize=11.5)
    ax.set_xlabel('特征')
    ax.set_ylabel('特征')
    bar = fig.colorbar(image, ax=ax, fraction=0.026, pad=0.02)
    bar.set_label('Spearman ρ')
    caption_b = (f'连续数值特征间的 Spearman 冗余诊断（{len(correlation.columns)} 个特征、'
                 f'{len(correlation.columns) * (len(correlation.columns) - 1) // 2} 对，'
                 f'不做全特征相关矩阵）')
    plot_style.add_subfigure_caption(ax, 'b', caption_b)
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.185, top=0.945, hspace=0.42)
    diagnostics_figure = figure_finalize.save_paper_figure(
        fig, SUPP_DIR, FIG_STEM,
        subfigures=[('a', 'Train / Validation / Test 基本分布对照', axes[0]),
                    ('b', caption_b, axes[1])],
        meta={'数据来源': 'job_salary_model_dataset.parquet + model_splits.parquet',
              '口径': '移除发布时点不可得的后验特征后的正式特征集；'
                      '相关矩阵只覆盖有连续意义的数值特征，不做 320×320 全特征矩阵',
              'seed': SEED, '用途': '第7章 7.1 节 建模前特征诊断（配合表 7-1）'})
    plt.close(fig)
    return diagnostics_figure


# ============================================================================
# 文本审计（§5~§14 / §27）
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
    ('本轮', r'本轮'), ('旧F组', r'旧F组'), ('旧实现', r'旧实现'),
    ('前期固定图件', r'前期固定图件'), ('之前版本', r'之前版本'),
    ('本次修改', r'本次修改'), ('更早实现', r'更早实现'),
    ('加粗短句 方法目的', r'\*\*方法目的。\*\*'),
    ('加粗短句 变量说明', r'\*\*变量说明。\*\*'),
    ('加粗短句 本文中如何使用', r'\*\*本文中如何使用。\*\*'),
    ('加粗短句 方法机制', r'\*\*方法机制。\*\*'),
    ('加粗短句 参数说明', r'\*\*参数说明。\*\*'),
    ('章首机械总起段', r'(?m)^本章(?:说明|主要|首先|从|交代|先|以|给出|对比|在|完成|介绍|按)'),
]

COUNTED_KEYS = ['研究问题 Q 简称', '需要说明的是', '需要强调的是', '必须明确', '值得注意的是',
                '需要重申的是', '这里需要指出', '可以看到', '可以发现', '由此可以看出',
                '综上可以看出', '这意味着', '不是……而是', '并非……而是',
                '中文引号 左', '中文引号 右', '中文括号 左', '中文括号 右',
                '破折号 ——', '短横 —', '行首无序列表', '开发语言 Stage2x',
                '本轮', '旧F组', '旧实现', '前期固定图件', '之前版本', '本次修改',
                '更早实现', '加粗短句 方法目的', '加粗短句 变量说明',
                '加粗短句 本文中如何使用', '加粗短句 方法机制', '加粗短句 参数说明']


def strip_html_comments(text: str) -> str:
    return re.sub(r'<!--.*?-->', '', text, flags=re.S)


def han_count(text: str) -> int:
    return len(re.findall(r'[\u4e00-\u9fff]', text))


def sentence_count(text: str) -> int:
    """按句号、问号、叹号切分的句数（与小结压缩口径一致，不含分号）。"""
    return len([item for item in re.split(r'[。！？]', text) if item.strip()])


def visible_body(text: str) -> str:
    """剔除注释、插表 / 插图占位行与公式行后的可见正文。"""
    body = strip_html_comments(text)
    body = re.sub(r'(?m)^\s*>?\s*【插[图表][^\n]*】\s*$', '', body)
    body = re.sub(r'(?m)^\s*\$\$.*?\$\$\s*$', '', body)
    body = re.sub(r'(?m)^\s*[|>```].*$', '', body)
    return body


def summarize_chapters() -> pd.DataFrame:
    rows = []
    for name in CHAPTERS:
        path = SOURCE_DIR / name
        text = path.read_text(encoding='utf-8')
        body = visible_body(text)
        rows.append({'文件': name, '汉字数（可见正文）': han_count(body),
                     '字节数': int(path.stat().st_size)})
    return pd.DataFrame(rows)


def audit_text() -> dict:
    pattern_counts = {}
    hit_rows = []
    for name in CHAPTERS:
        path = SOURCE_DIR / name
        raw = path.read_text(encoding='utf-8')
        body = visible_body(raw)
        for label, pattern in AUDIT_PATTERNS:
            for number, line in enumerate(raw.splitlines(), start=1):
                for match in re.finditer(pattern, line):
                    hit_rows.append({'文件': name, '行号': number, '命中类型': label,
                                     '原文片段': line.strip()[:200]})
        for label, pattern in AUDIT_PATTERNS:
            pattern_counts[label] = pattern_counts.get(label, 0) + len(
                re.findall(pattern, body))
    summary = summarize_chapters()
    return {'counts': pattern_counts, 'hits': pd.DataFrame(hit_rows),
            'chapters': summary,
            '总计汉字数（可见正文）': int(summary['汉字数（可见正文）'].sum())}


def chapter_summary_block() -> pd.DataFrame:
    """各章小结的段落数、句数与汉字数。"""
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
        sentences = [item for item in re.split(r'[。；!？]', joined) if item.strip()]
        rows.append({'文件': name, '小结段落数': len(block),
                     '小结句数（含分号切分）': len(sentences),
                     '小结句数（按句号切分）': sentence_count(joined),
                     '小结汉字数': han_count(joined)})
    return pd.DataFrame(rows)


def chapter_lead_paragraph() -> pd.DataFrame:
    """各章大标题后的第一段（用于判断是否存在机械总起段）。"""
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
    """章标题后首段不是二级标题的章数（即存在机械总起段的章数）。"""
    leads = chapter_lead_paragraph()
    return int(sum(1 for value in leads['标题后首段']
                   if not str(value).strip().startswith('#')))


# ============================================================================
# 主流程
# ============================================================================
def run_compute() -> int:
    started = time.time()
    print('=' * 96)
    print('Stage26.3 必要重算与建模前诊断（只新增文件）')
    print('=' * 96)
    manifest_before = project_manifest()
    print(f'运行前既有文件 SHA-256 清单：{len(manifest_before)} 个')

    model_frame = io_utils.read_parquet(project_paths.JOB_SALARY_MODEL_DATASET_PARQUET)
    splits = io_utils.read_parquet(project_paths.MODEL_SPLITS_PARQUET)
    membership = io_utils.read_parquet(project_paths.JOB_SKILL_MEMBERSHIP_PARQUET)
    entity = pd.read_parquet(project_paths.PROCESSED_UNIQUE_PARQUET,
                             columns=[schema.ID_FIELD, '发布时间'])
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
    split_definitions = {
        'Random Split': random_labels,
        'Company Group Split': group_labels,
        'Retrospective Temporal Split': temporal_labels,
    }

    grouped = grouped_columns(feature_manifest, frame, SAFE_F_STAGE26_1, SAFE_F_FINAL)
    grouped_new = drop_posterior(grouped)
    skill_map = model_training.build_skill_map(membership, skill_eda.ALL_USABLE_SCOPES)
    text_matrix = model_training.load_text_matrix(
        job_ids.tolist(), project_paths.FEATURES_DIR / 'job_text_embeddings.npz',
        project_paths.FEATURES_DIR / 'job_text_embedding_index.parquet')
    text_by_id = {job_id: position for position, job_id in enumerate(job_ids.tolist())}

    # ---------------------------------------------------------- A 建模前诊断
    diagnostics = build_diagnostics(frame, feature_manifest,
                                    random_labels, membership)
    print('建模前诊断：缺失率表 %d 行 / 近常量 %d 行 / 高冗余对 %d 对'
          % (len(diagnostics['missing']), len(diagnostics['variance']),
             len(diagnostics['redundancy'])))
    print(diagnostics['distribution'].to_string(index=False))
    if len(diagnostics['redundancy']):
        print(diagnostics['redundancy'].to_string(index=False))

    # ---------------------------------------------------------- B 维度对照
    dimension_rows = []
    for label in ABLATION_ORDER:
        groups = ABLATION_CONFIGS[label]
        old_numeric = sum(len(grouped[letter]['numeric']) for letter in groups
                          if letter != 'SafeF')
        new_numeric = sum(len(grouped_new[letter]['numeric']) for letter in groups
                          if letter != 'SafeF')
        old_result = fit_eval(frame, random_labels, groups, grouped, skill_map, text_matrix,
                              text_by_id, SKILL_THRESHOLD, TEXT_DIM)
        new_result = fit_eval(frame, random_labels, groups, grouped_new, skill_map,
                              text_matrix, text_by_id, SKILL_THRESHOLD, TEXT_DIM)
        dimension_rows.append({
            '配置': label, '特征组': '+'.join(groups),
            '训练集内数值特征列数（改前）': int(old_numeric),
            '训练集内数值特征列数（改后）': int(new_numeric),
            '编码后特征维度（改前）': old_result['特征维度'],
            '编码后特征维度（改后）': new_result['特征维度'],
            '维度变化': int(new_result['特征维度'] - old_result['特征维度']),
            '技能列数（改前）': old_result['技能列数'],
            '技能列数（改后）': new_result['技能列数']})
        print(f"  维度对照 {label}: {old_result['特征维度']} → {new_result['特征维度']}")
    dimension_table = pd.DataFrame(dimension_rows)

    # ---------------------------------------------------------- C 消融五配置
    ablation_rows, predictions = [], {}
    for label in ABLATION_ORDER:
        groups = ABLATION_CONFIGS[label]
        result = fit_eval(frame, random_labels, groups, grouped_new, skill_map, text_matrix,
                          text_by_id, SKILL_THRESHOLD, TEXT_DIM)
        predictions[label] = result
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

    increment_rows = []
    for name, without, with_, subset in [
            ('Safe-F（发布时间位置）增量', 'A+B+C+D+E', 'A+B+C+D+E+SafeF', 'both'),
            ('技能增量（D）', 'A+B+C', 'A+B+C+D', 'both'),
            ('文本语义增量（E）', 'A+B+C', 'A+B+C+E', 'both'),
            ('技能与文本增量（D+E）', 'A+B+C', 'A+B+C+D+E', 'both')]:
        for part in (('validation', 'valid_pred', 'valid_frame'),
                     ('test', 'test_pred', 'test_frame')):
            key = part[0]
            truth = predictions[without][part[2]][schema.SALARY_MID_FIELD].to_numpy('float64')
            stats = paired_bootstrap(truth, predictions[without][part[1]],
                                     predictions[with_][part[1]])
            increment_rows.append({
                '增量检验': name, '比较': f'{with_} 对 {without}', '数据子集': key,
                '不含该组 MAE': round(float(np.abs(truth
                                              - predictions[without][part[1]]).mean()), 6),
                '含该组 MAE': round(float(np.abs(truth
                                            - predictions[with_][part[1]]).mean()), 6),
                'ΔMAE（不含 − 含）': stats['ΔMAE'], '95% 置信区间下界': stats['CI95下界'],
                '95% 置信区间上界': stats['CI95上界'], '是否跨 0': stats['是否跨0']})
            print(f"  增量 {name} / {key}: ΔMAE {stats['ΔMAE']} "
                  f"[{stats['CI95下界']}, {stats['CI95上界']}] 跨0={stats['是否跨0']}")
    increment_table = pd.DataFrame(increment_rows)

    # ---------------------------------------------------------- D 三种划分统一协议
    unified_rows = []
    unified_cache = {}
    for label, labels in split_definitions.items():
        result = fit_eval(frame, labels, MODEL_FEATURE_GROUPS, grouped_new, skill_map,
                          text_matrix, text_by_id, SKILL_THRESHOLD, TEXT_DIM)
        unified_cache[label] = result
        unified_rows.append({
            '划分方式': label, 'Train n': result['n_train'],
            'Validation n': result['n_validation'], 'Test n': result['n_test'],
            '特征维度': result['特征维度'], '技能列数': result['技能列数'],
            'test MAE': result['test']['MAE'], 'test RMSE': result['test']['RMSE'],
            'test R²': result['test']['R2'],
            'validation MAE': result['validation']['MAE'],
            'validation RMSE': result['validation']['RMSE'],
            'validation R²': result['validation']['R2']})
        print(f"  统一协议 {label}: dim {result['特征维度']} / test MAE "
              f"{result['test']['MAE']} / RMSE {result['test']['RMSE']} / "
              f"R² {result['test']['R2']}")
    unified_table = pd.DataFrame(unified_rows)

    # ---------------------------------------------------------- E 模型比较
    comparison_rows = []
    best_params_by_model = {}
    for model_key, grid in MODEL_GRIDS.items():
        best = None
        for params in grid:
            result = fit_eval(frame, random_labels, MODEL_FEATURE_GROUPS, grouped_new,
                              skill_map, text_matrix, text_by_id, SKILL_THRESHOLD, TEXT_DIM,
                              model_key=model_key, params=params,
                              scale_numeric=model_key in SCALE_FOR)
            row = {'模型': model_key, '超参数': str(params),
                   'validation_MAE': result['validation']['MAE'],
                   'validation_RMSE': result['validation']['RMSE'],
                   'validation_R2': result['validation']['R2'],
                   'test_MAE': result['test']['MAE'], 'test_RMSE': result['test']['RMSE'],
                   'test_R2': result['test']['R2']}
            if best is None or row['validation_MAE'] < best['validation_MAE']:
                best = row
                best_params_by_model[model_key] = dict(params)
            print(f"  模型比较 {model_key} {params}: validation MAE "
                  f"{row['validation_MAE']}")
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

    # ---------------------------------------------------------- F 最终模型（train + validation 重拟合）
    final_rows = []
    best_params = best_params_by_model['LightGBM']
    print('  主模型超参数：', best_params)
    train_valid = frame.loc[random_labels.reindex(frame[schema.ID_FIELD]).to_numpy()
                            != 'test'].reset_index(drop=True)
    test_frame = frame.loc[random_labels.reindex(frame[schema.ID_FIELD]).to_numpy()
                           == 'test'].reset_index(drop=True)
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
    final_metrics = model_training.regression_metrics(
        test_frame[schema.SALARY_MID_FIELD].to_numpy('float64'), final_pred)
    train_only = unified_cache['Random Split']
    final_rows.append({'模型与配置': 'LightGBM（仅训练集拟合，用于对照）',
                       'test MAE': train_only['test']['MAE'],
                       'test RMSE': train_only['test']['RMSE'],
                       'test R²': train_only['test']['R2'], 'n': train_only['n_test'],
                       '特征维度': train_only['特征维度']})
    final_rows.append({'模型与配置': 'LightGBM（训练集 + 验证集重拟合，最终模型）',
                       'test MAE': final_metrics['MAE'], 'test RMSE': final_metrics['RMSE'],
                       'test R²': final_metrics['R2'], 'n': final_metrics['n'],
                       '特征维度': int(assembler.schema.dimension)})
    final_table = pd.DataFrame(final_rows)
    print(final_table.to_string(index=False))

    # ---------------------------------------------------------- F2 SHAP 解释（新正式模型）
    shap_feature_rows, shap_skill_rows, shap_note = [], [], {}
    shap_skill_table = pd.DataFrame([{'说明': 'NOT_RUN'}])
    try:
        raw_values = final_model.predict(matrix_test, pred_contrib=True)
        if hasattr(raw_values, 'toarray'):
            raw_values = raw_values.toarray()
        raw_values = np.asarray(raw_values, dtype='float64')
        if raw_values.ndim == 1:
            raw_values = raw_values.reshape(len(final_pred), -1)
        base_value = float(raw_values[:, -1].mean())
        values = raw_values[:, :-1]
        if values.shape[0] != len(final_pred):
            raise ValueError(f'SHAP 输出形状异常：{values.shape}')
        additive_error = float(np.abs(values.sum(axis=1) + base_value - final_pred).max())
        mean_abs = np.abs(values).mean(axis=0)
        names = assembler.feature_names()
        if len(names) != values.shape[1]:
            raise ValueError(f'特征名与 SHAP 列数不一致：{len(names)} vs {values.shape[1]}')
        total = float(mean_abs.sum())
        order = np.argsort(-mean_abs)[:15]
        dense_test = np.asarray(matrix_test.todense())
        for position in order:
            column = dense_test[:, position]
            present = column > 0
            shap_feature_rows.append({
                '特征': names[position],
                'mean|SHAP|': round(float(mean_abs[position]), 6),
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
                '技能': skill,
                '建模样本岗位数（频率）': int(skill_counts.get(skill, 0)),
                '测试集命中岗位数': int(present.sum()),
                '技能存在时平均SHAP': round(float(values[present, position].mean()), 6)
                if present.any() else None,
                'mean|SHAP|': round(float(mean_abs[position]), 6)})
        shap_skill_table = pd.DataFrame(shap_skill_rows).sort_values(
            'mean|SHAP|', ascending=False)
        shap_note = {
            '基准值（元/天）': round(base_value, 6),
            '加性一致性最大误差': additive_error,
            '特征数': len(names),
            '技能列数': len(assembler.schema.skill_columns),
        }
        print(f'  SHAP：基准值 {base_value:.6f} / 加性一致性最大误差 {additive_error:.3e}')
    except Exception as error:  # noqa: BLE001
        shap_skill_table = pd.DataFrame([{'说明': f'NOT_RUN（{type(error).__name__}: {error}）'}])
        shap_feature_rows = []
        print(f'  SHAP 解释未运行：{type(error).__name__}: {error}')

    # ---------------------------------------------------------- G 分组预测误差（只保留四分位）
    truth_test = test_frame[schema.SALARY_MID_FIELD].to_numpy('float64')
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
                                else set() for items in test_frame['岗位大类集合']],
                               dtype=object)
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

    # ---------------------------------------------------------- H 多种子 + 公司分组多次划分
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
                              'validation RMSE': result['validation']['RMSE'],
                              'validation R²': result['validation']['R2'],
                              'test MAE': result['test']['MAE'],
                              'test RMSE': result['test']['RMSE'],
                              'test R²': result['test']['R2']})
            print(f"  多种子 {seed} {model_key}: validation MAE "
                  f"{result['validation']['MAE']}")
    seed_table = pd.DataFrame(seed_rows)
    seed_summary = seed_table.groupby('模型').agg(
        validation_MAE均值=('validation MAE', 'mean'),
        validation_MAE标准差=('validation MAE', 'std'),
        test_MAE均值=('test MAE', 'mean'), test_MAE标准差=('test MAE', 'std')).round(6).reset_index()

    repeat_rows = []
    company_groups = splits.set_index(schema.ID_FIELD)['company_entity_id'].reindex(
        job_ids.to_numpy()).to_numpy()
    for seed in SEEDS:
        from sklearn.model_selection import GroupShuffleSplit  # noqa: PLC0415

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
        labels = pd.Series(values, index=job_ids)
        result = fit_eval(frame, labels, MODEL_FEATURE_GROUPS, grouped_new, skill_map,
                          text_matrix, text_by_id, SKILL_THRESHOLD, TEXT_DIM)
        repeat_rows.append({'随机种子': seed, '特征维度': result['特征维度'],
                            '技能列数': result['技能列数'],
                            'Train n': result['n_train'],
                            'Validation n': result['n_validation'],
                            'Test n': result['n_test'],
                            'test MAE': result['test']['MAE'],
                            'test RMSE': result['test']['RMSE'],
                            'test R²': result['test']['R2']})
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

    # ---------------------------------------------------------- I 稳健性对照（仅训练集拟合）
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
    base_model.fit(base_matrix_train, train_only_frame[schema.SALARY_MID_FIELD]
                   .to_numpy('float64'))
    base_pred = np.asarray(base_model.predict(base_matrix_test), dtype='float64')
    base_metrics = model_training.regression_metrics(truth_test, base_pred)
    base_names = base_assembler.feature_names()
    robustness_rows.append({'检查项': '对照基准', '口径': '真实薪资，主口径未缩尾',
                            'MAE': base_metrics['MAE'], 'RMSE': base_metrics['RMSE'],
                            'R²': base_metrics['R2'],
                            '说明': 'A+B+C+D+E，仅用训练集拟合'})
    lower_bound = float(np.quantile(train_only_frame[schema.SALARY_MID_FIELD], 0.01))
    upper_bound = float(np.quantile(train_only_frame[schema.SALARY_MID_FIELD], 0.99))
    clipped = train_only_frame[schema.SALARY_MID_FIELD].clip(lower_bound, upper_bound)
    winsor_model = model_training.make_model('LightGBM', best_params, random_state=SEED)
    winsor_model.fit(base_matrix_train, clipped.to_numpy('float64'))
    winsor_pred = np.asarray(winsor_model.predict(base_matrix_test), dtype='float64')
    winsor_metrics = model_training.regression_metrics(truth_test, winsor_pred)
    robustness_rows.append({
        '检查项': '极端值敏感性', '口径': '训练目标 winsorize 1%/99%',
        'MAE': winsor_metrics['MAE'], 'RMSE': winsor_metrics['RMSE'],
        'R²': winsor_metrics['R2'],
        '说明': f'训练目标裁剪至 {lower_bound:.0f} 至 {upper_bound:.0f} 元/天，'
                f'评估仍用真实测试薪资'})
    for name, field in [('目标口径 薪资下限', schema.SALARY_MIN_FIELD),
                        ('目标口径 薪资上限', schema.SALARY_MAX_FIELD)]:
        model = model_training.make_model('LightGBM', best_params, random_state=SEED)
        model.fit(base_matrix_train, train_only_frame[field].to_numpy('float64'))
        prediction = np.asarray(model.predict(base_matrix_test), dtype='float64')
        metrics = model_training.regression_metrics(truth_test, prediction)
        robustness_rows.append({'检查项': name, '口径': field, 'MAE': metrics['MAE'],
                                'RMSE': metrics['RMSE'], 'R²': metrics['R2'],
                                '说明': '仅比较趋势，不作为新的主任务'})
    gain_base = np.asarray(
        base_model.booster_.feature_importance(importance_type='gain')
        if hasattr(base_model, 'booster_') else base_model.feature_importances_,
        dtype='float64')
    gain_winsor = np.asarray(
        winsor_model.booster_.feature_importance(importance_type='gain')
        if hasattr(winsor_model, 'booster_') else winsor_model.feature_importances_,
        dtype='float64')
    from scipy import stats as scipy_stats  # noqa: PLC0415

    gain_spearman = float(scipy_stats.spearmanr(gain_base, gain_winsor).statistic)
    robustness_rows.append({
        '检查项': '特征重要性方向', '口径': 'winsorize 与主模型 gain 排名的 Spearman',
        'MAE': None, 'RMSE': None, 'R²': None,
        '说明': f'{gain_spearman:.6f}，共有特征 {len(base_names)} 个'})

    shap_status = 'NOT_RUN'
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
            other_names = result['assembler'].feature_names()
            other_matrix = result['assembler'].transform(
                result['test_frame'], skill_map,
                text_matrix[[text_by_id[job_id]
                             for job_id in result['test_frame'][schema.ID_FIELD]]])
            other_shap = np.abs(ablation_shap.tree_shap_values(
                result['model'], other_matrix)).mean(axis=0)
            position = {name: index for index, name in enumerate(other_names)}
            left, right = [], []
            for index, name in enumerate(base_names):
                if name in position:
                    left.append(base_shap[index])
                    right.append(other_shap[position[name]])
            correlations.append(float(scipy_stats.spearmanr(left, right).statistic))
        shap_spearman = correlations
        shap_status = 'OK'
        robustness_rows.append({
            '检查项': '解释稳定性',
            '口径': '三种随机种子 42、7、2024 与主模型 mean|SHAP| 排名的 Spearman',
            'MAE': None, 'RMSE': None, 'R²': None,
            '说明': '；'.join(f'{value:.6f}' for value in correlations)
                    + f'（共同特征 {len(base_names)} 个）'})
    except Exception as error:  # noqa: BLE001
        shap_status = f'NOT_RUN（{type(error).__name__}: {error}）'
        print(f'  SHAP 稳定性未运行：{shap_status}')
    table_robustness = pd.DataFrame(robustness_rows)

    # ---------------------------------------------------------- J 图件
    figure_meta = figure_diagnostics(diagnostics)
    flow_meta = figure_flow()

    # ---------------------------------------------------------- K 输出
    ablation_meta = pd.DataFrame([
        {'项目': 'test MAE 最小配置',
         '配置': ablation_table.loc[ablation_table['test_MAE'].idxmin(), '配置'],
         '取值': float(ablation_table['test_MAE'].min())},
        {'项目': 'test MAE 最大配置',
         '配置': ablation_table.loc[ablation_table['test_MAE'].idxmax(), '配置'],
         '取值': float(ablation_table['test_MAE'].max())},
        {'项目': 'test MAE 真实跨度', '配置': 'max − min',
         '取值': float(ablation_table['test_MAE'].max()
                     - ablation_table['test_MAE'].min())},
        {'项目': 'validation MAE 真实跨度', '配置': 'max − min',
         '取值': float(ablation_table['validation_MAE'].max()
                     - ablation_table['validation_MAE'].min())},
        {'项目': 'A+B+C+D+E 编码后维度', '配置': '改后',
         '取值': int(ablation_table.set_index('配置').loc['A+B+C+D+E', '特征维度'])},
        {'项目': 'A+B+C+D+E+SafeF 编码后维度', '配置': '改后',
         '取值': int(ablation_table.set_index('配置')
                     .loc['A+B+C+D+E+SafeF', '特征维度'])},
    ])

    removal_table = pd.DataFrame([
        {'特征': column, '类别': ('发布时点不可得' if column in POSTERIOR_FEATURES
                              else ('连续数值冗余' if column in REDUNDANT_FEATURES
                                    else '近常量')),
         '移出依据': REMOVE_REASONS[column]} for column in REMOVED_FEATURES])

    write_excel(TABLE_DIAGNOSTICS, {
        '01_缺失率': diagnostics['missing'],
        '02_近常量与低方差': diagnostics['variance'],
        '03_低频类别': diagnostics['category'],
        '04_稀有标签与低频技能': pd.concat([diagnostics['tag'], diagnostics['skill']],
                                    ignore_index=True),
        '05_连续数值特征冗余对': diagnostics['redundancy'] if len(
            diagnostics['redundancy']) else pd.DataFrame(
            [{'说明': '连续数值特征之间不存在 |Spearman ρ| > 0.85 的特征对'}]),
        '06_Spearman相关矩阵': diagnostics['correlation'],
        '07_三子集分布对照': diagnostics['distribution'],
        '08_移除特征清单': removal_table,
    })
    write_excel(TABLE_METRICS, {
        '01_特征组维度对照': dimension_table,
        '02_三种划分统一协议': unified_table,
        '03_消融五配置': ablation_table,
        '04_消融增量bootstrap': increment_table,
        '05_模型比较': comparison_table,
        '06_最终模型测试集': final_table,
        '07_分组预测误差四分位': quartile_table,
        '08_岗位大类预测误差': category_error_table,
        '09_多种子明细': seed_table,
        '10_多种子汇总': seed_summary,
        '11_公司分组多次划分': repeat_table,
        '12_公司分组汇总': repeat_summary,
        '13_稳健性对照': table_robustness,
        '14_主模型SHAP特征': pd.DataFrame(shap_feature_rows),
        '15_技能SHAP': shap_skill_table,
    })

    payload = {
        '生成时间': time.strftime('%Y-%m-%d %H:%M:%S'),
        '输入': {
            '建模样本': int(len(model_frame)), '建模划分': int(len(splits)),
            '移除的后验特征': POSTERIOR_FEATURES,
            '移除的冗余与近常量特征': REDUNDANT_FEATURES + NEAR_CONSTANT_FEATURES,
            '移除特征总清单': REMOVED_FEATURES,
            '移除依据': REMOVE_REASONS,
            '统一协议': {'模型': 'LightGBM', '参数': LIGHTGBM_PARAMS,
                     '特征组': 'A+B+C+D+E', '技能频次阈值': SKILL_THRESHOLD,
                     '文本语义 SVD': TEXT_DIM, '随机种子': SEED},
        },
        '特征组维度对照': dimension_table.to_dict('records'),
        '三种划分统一协议': unified_table.to_dict('records'),
        '消融五配置': ablation_table.to_dict('records'),
        '消融跨度摘要': ablation_meta.to_dict('records'),
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
        'SHAP解释摘要': shap_note,
        'SHAP稳定性': shap_status,
        'SHAP排名Spearman': shap_spearman,
        '建模前诊断': {
            '三子集分布对照': diagnostics['distribution'].to_dict('records'),
            '缺失率最大取值': float(diagnostics['missing']['缺失率'].max()),
            '缺失率非零字段数': int((diagnostics['missing']['缺失率'] > 0).sum()),
            '近常量数值特征': diagnostics['variance'].loc[
                diagnostics['variance']['判定'].ne('正常'), '特征'].tolist(),
            '连续数值特征高冗余对': diagnostics['redundancy'].to_dict('records'),
            '低频类别字段': diagnostics['category'].to_dict('records'),
            '稀有标签与低频技能': pd.concat([diagnostics['tag'], diagnostics['skill']],
                                     ignore_index=True).to_dict('records'),
        },
        '图件': figure_meta.get('file', str(SUPP_DIR / f'{FIG_STEM}.png')),
        '流程图形文件': flow_meta.get('stem', FIG_FLOW),
        '对照 Stage26.2': {
            'Random Split': {'旧 test MAE': 36.074723},
            'Company Group Split': {'旧 test MAE': 52.049937},
            'Retrospective Temporal Split': {'旧 test MAE': 31.725791},
            '旧 A+B+C+D+E 维度': 320, '旧 A+B+C+D+E+SafeF 维度': 322,
        },
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
    payload['文件完整性'] = {'既有文件被修改': changed, '新增文件': added,
                        '删除文件': removed,
                        '既有文件被修改数': len(changed), '新增文件数': len(added)}
    dump_json(METRICS_PATH, payload)
    print('完成，耗时 %.1f s' % (time.time() - started))
    return 0


# ============================================================================
# 文本审计
# ============================================================================
def run_baseline() -> int:
    """记录改写前的文本审计基线（供改写后逐项对照）。"""
    audit = audit_text()
    leads = chapter_lead_paragraph()
    summaries = chapter_summary_block()
    payload = {
        '记录时间': time.strftime('%Y-%m-%d %H:%M:%S'),
        '命中计数': audit['counts'],
        '总计汉字数（可见正文）': audit['总计汉字数（可见正文）'],
        '各章汉字数': audit['chapters'].to_dict('records'),
        '各章标题后首段': leads.to_dict('records'),
        '章首机械总起段数量': mechanical_lead_count(),
        '各章小结篇幅': summaries.to_dict('records'),
        '3.2 数据采集方案': section_length(SOURCE_DIR, visible_body, han_count, '03_数据获取与预处理.md',
                                      r'^#{2,4}\s*3\.2', r'^#{2,4}\s*3\.3'),
    }
    dump_json(TEXT_BASELINE_PATH, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2)[:4000])
    return 0


def run_inventory() -> int:
    """生成内联数学符号清单（表 55）与改写后文本审计（表 58）。"""
    inventory_rows = scan_inline_math()
    inventory = pd.DataFrame(inventory_rows)
    write_excel(TABLE_INLINE_MATH, {'01_内联数学符号清单': inventory,
                                    '02_按文件计数': (inventory.groupby('文件').size()
                                                  .reset_index(name='条目数')
                                                  if len(inventory) else
                                                  pd.DataFrame([{'文件': '—', '条目数': 0}]))})

    audit = audit_text()
    baseline = json.loads(TEXT_BASELINE_PATH.read_text(encoding='utf-8')) \
        if TEXT_BASELINE_PATH.is_file() else {'命中计数': {}, '各章小结篇幅': [],
                                             '各章标题后首段': []}
    comparison_rows = []
    for label in COUNTED_KEYS:
        before = int(baseline['命中计数'].get(label, 0))
        after = int(audit['counts'].get(label, 0))
        comparison_rows.append({'项目': label, '改前': before, '改后': after,
                                '变化': after - before})
    comparison_rows.append({'项目': '可见正文汉字数合计',
                            '改前': int(baseline.get('总计汉字数（可见正文）', 0)),
                            '改后': int(audit['总计汉字数（可见正文）']),
                            '变化': int(audit['总计汉字数（可见正文）'])
                            - int(baseline.get('总计汉字数（可见正文）', 0))})
    comparison_rows.append({'项目': '章首机械总起段数量（按标题后首段判定）',
                            '改前': int(baseline.get('章首机械总起段数量', 0)),
                            '改后': int(mechanical_lead_count()),
                            '变化': int(mechanical_lead_count())
                            - int(baseline.get('章首机械总起段数量', 0))})
    comparison = pd.DataFrame(comparison_rows)

    summary_before = pd.DataFrame(baseline.get('各章小结篇幅', []))
    summaries = chapter_summary_block()
    summary_cols = [column for column in summary_before.columns if column != '文件']
    summary_compare = summaries.merge(
        summary_before[['文件'] + summary_cols] if len(summary_before) else
        pd.DataFrame(columns=['文件']),
        on='文件', how='left', suffixes=('（改后）', '（改前）'))

    leads_before = pd.DataFrame(baseline.get('各章标题后首段', []))
    leads_after = chapter_lead_paragraph()
    leads_compare = leads_after.merge(
        leads_before[['文件', '标题后首段']] if len(leads_before) else
        pd.DataFrame(columns=['文件', '标题后首段']),
        on='文件', how='left', suffixes=('（改后）', '（改前）'))

    section = pd.DataFrame([
        {'项目': '3.2 数据采集方案（改前）', **baseline.get('3.2 数据采集方案', {})},
        {'项目': '3.2 数据采集方案（改后）',
         **section_length(SOURCE_DIR, visible_body, han_count, '03_数据获取与预处理.md', r'^#{2,4}\s*3\.2', r'^#{2,4}\s*3\.3')},
    ])

    write_excel(TABLE_TEXT_AUDIT, {
        '01_文风命中数对照': comparison,
        '02_各章小结篇幅对照': summary_compare,
        '03_章首段落对照': leads_compare,
        '04_3.2压缩对照': section,
        '05_命中明细': audit['hits'] if len(audit['hits']) else
        pd.DataFrame([{'说明': '无命中'}]),
        '06_各章汉字数': audit['chapters'],
    })
    payload = {
        '生成时间': time.strftime('%Y-%m-%d %H:%M:%S'),
        '内联数学符号清单条目数': int(len(inventory)),
        '按文件计数': (inventory.groupby('文件').size().to_dict()
                   if len(inventory) else {}),
        '文风命中数对照': comparison.to_dict('records'),
        '3.2 压缩对照': section.to_dict('records'),
        '各章汉字数': audit['chapters'].to_dict('records'),
        '总计汉字数（可见正文）': audit['总计汉字数（可见正文）'],
    }
    dump_json(TEXT_METRICS_PATH, payload)
    print(comparison.to_string(index=False))
    print(f'内联数学符号清单条目数：{len(inventory)}')
    return 0


# ---------------------------------------------------------------------------
# 内联数学符号扫描
# ---------------------------------------------------------------------------
MATH_TOKEN_PATTERNS = [
    ('希腊字母与统计量',
     r'ε²|ε\^2|δ|ρ|α|β|λ|ΔMAE|Δ|φ_[0-9a-zA-Z]|\|φ_[0-9a-zA-Z]\||R²|R\^2'),
    ('带下标数学量',
     r'(?<![A-Za-z0-9_])[A-Za-zRrNnOoStUuqpGgDdCcHhkKmXxYy]_\{?[a-zA-Z0-9,+\-]+\}?'
     r'(?![A-Za-z0-9])'),
    ('帽符号与均值符号', r'ŷ_[a-zA-Z0-9]+|ŷ|ȳ|R̄_[a-zA-Z0-9]+|R̄'),
    ('单字母数学变量',
     r'(?<![A-Za-z0-9_])(?:N|k|H|R|X|Y|M|B|F|T|n|m|i|j|p|q|x|y|b|e|s|d|t|G|D|O|C)'
     r'(?=\s*(?:为|表示|与|和|的|值|取|大于|小于|[=<>]))'),
    ('绝对值与阈值比较',
     r'\|δ\|\s*(?:小于|大于|不小于|不大于|小于等于|大于等于|[<>=])\s*[0-9.]+|'
     r'(?:q|p)\s*(?:值)?\s*(?:小于|大于|[<>=])\s*0\.05|'
     r'\|ρ\|\s*[<>=]\s*[0-9.]+'),
    ('长英文变量',
     r'Duration[_\^][a-zA-Z]+|FinalDeadline|InitialDeadline|ReopenGap|'
     r'Start|End\b|CurrentStart|PreviousEnd'),
]


def scan_inline_math() -> list:
    rows = []
    for name in CHAPTERS:
        path = SOURCE_DIR / name
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(),
                                      start=1):
            stripped = line.strip()
            if stripped.startswith('$$') or stripped.startswith('|'):
                continue
            for label, pattern in MATH_TOKEN_PATTERNS:
                for match in re.finditer(pattern, line):
                    token = match.group(0)
                    context_start = max(match.start() - 24, 0)
                    context_end = min(match.end() + 24, len(line))
                    rows.append({
                        '文件': name, '行号': number, '类型': label,
                        '原文片段': line[max(context_start, 0):context_end].strip(),
                        '命中文本': token,
                        '建议 LaTeX / UnicodeMath': suggest_math(token)})
    return rows


SUGGESTION = {
    'ε²': r'\varepsilon^{2}', 'δ': r'\delta', 'ρ': r'\rho', 'α': r'\alpha',
    'β': r'\boldsymbol{\beta}', 'λ': r'\lambda', 'R²': r'R^{2}',
    'ŷ': r'\hat{y}', 'ȳ': r'\bar{y}', 'R̄': r'\bar{R}',
    'Start': 's', 'End': 'd', 'FinalDeadline': "d^{(f)}",
    'InitialDeadline': "d^{(0)}", 'ReopenGap': 'G',
}


def suggest_math(token: str) -> str:
    if token in SUGGESTION:
        return SUGGESTION[token]
    plain = token.replace('_', '').replace('^', '')
    match = re.match(r'^([A-Za-zRrNnOoStUuqpGgDdCcHhkKmXxYy])_\{?([A-Za-z0-9,+\-]+)\}?$', token)
    if match:
        return f'{match.group(1)}_{{{match.group(2)}}}'
    if token.startswith('Duration'):
        suffix = token.split('^')[-1] if '^' in token else token.split('_')[-1]
        return f'D_{{{suffix}}}'
    return plain


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else 'compute'
    if mode == 'compute':
        return run_compute()
    if mode == 'baseline':
        return run_baseline()
    if mode == 'inventory':
        return run_inventory()
    if mode == 'flow':
        print(json.dumps(figure_flow(), ensure_ascii=False, indent=2, default=str))
        return 0
    print(f'未知模式：{mode}')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())

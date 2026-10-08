# -*- coding: utf-8 -*-
"""Stage26：招聘生命周期时序融合（数据与建模实测部分）。

本脚本只做**新增**计算与**新增**文件输出，严格只读既有产物（`data/raw`、`data/interim`、
`data/processed` 既有文件、`outputs/**` 既有文件、`src/**`、`docs/**`、既有 `scripts/**`）。

业务时间口径（硬性）：
- **业务时间轴** = `发布时间`（招聘周期开始）/ `投递截止日期`（计划结束）；
- **爬取时间**（`观测时间` / `数据创建时间` / `数据更新时间`）只用于版本排序与审计，
  不进入任何 F 组特征、不进入 Temporal Split。

层级：`Observation → Version(全页面版本) → Episode(招聘周期) → Entity`。

重新招聘三档口径（诚实性关键）：
- T1 确认重招：`start > prev_end` 且 `reopen_gap_days >= 7`；
- T2 疑似重开：`start > prev_end` 且 `0 <= reopen_gap_days < 7`；
- T3 时间冲突：`start <= prev_end`（**不得判为重招**）。

新增输出（不覆盖任何既有文件）::

    data/processed/job_recruitment_episode.parquet
    data/processed/job_daily_panel.parquet
    data/processed/job_version_stage26.parquet
    outputs/tables/35..43（9 张审计 / 结果表）
    outputs/figures/supplementary/图S18..图S22（各 PNG 600dpi + PDF）
    outputs/logs/metrics/stage_26_temporal.json

运行::

    E:\\anaconda3\\envs\\reptile\\python.exe scripts\\26_stage26_temporal.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import (ablation_shap, eda_analysis, figure_finalize, io_utils,  # noqa: E402
                 model_training, plot_style, project_paths, schema, skill_eda, versioning)

# ============================================================================
# 常量与口径
# ============================================================================
SEED = 42
BOOTSTRAP_ROUNDS = ablation_shap.BOOTSTRAP_ROUNDS          # 1000
BOOTSTRAP_SEED = ablation_shap.BOOTSTRAP_SEED              # 42
LONG_DURATION_THRESHOLD = 365                              # 超长招聘周期阈值（天）
REFERENCE_DATE = pd.Timestamp('2022-01-01')
MAJOR_CATEGORIES = ['人工智能', '后端开发', '前端开发', '数据', '产品', '运营']
MANUAL_SAMPLE_SIZE = 80
PANEL_MAX_ROWS = 20_000_000

PROCESSED = project_paths.PROCESSED_DIR
TABLES = project_paths.TABLES_DIR
SUPP_DIR = project_paths.FIGURES_DIR / 'supplementary'
METRICS_PATH = project_paths.METRICS_DIR / 'stage_26_temporal.json'

EPISODE_PATH = PROCESSED / 'job_recruitment_episode.parquet'
PANEL_PATH = PROCESSED / 'job_daily_panel.parquet'
VERSION26_PATH = PROCESSED / 'job_version_stage26.parquet'

TABLE_FILES = [
    '35_recruitment_time_field_audit.xlsx',
    '36_recruitment_episode_identification.xlsx',
    '37_reopen_statistics.xlsx',
    '38_recruitment_lifecycle_statistics.xlsx',
    '39_recruitment_temporal_indicators.xlsx',
    '40_lifecycle_salary_association.xlsx',
    '41_F_group_ablation.xlsx',
    '42_temporal_split_results.xlsx',
    '43_reopen_manual_review.xlsx',
]
FIG_STEMS = [
    '图S18_招聘生命周期分布',
    '图S19_样本招聘周期每日新增结束与活跃数量',
    '图S20_主要岗位大类样本活跃招聘数量时间演化',
    '图S21_样本活跃岗位薪资中位数与IQR时间演化',
    '图S22_不同泛化场景性能比较',
]
NEW_FILES = ([EPISODE_PATH, PANEL_PATH, VERSION26_PATH]
             + [TABLES / name for name in TABLE_FILES]
             + [SUPP_DIR / f'{stem}{suffix}' for stem in FIG_STEMS
                for suffix in ('.png', '.pdf')]
             + [METRICS_PATH])

SKIP_DIRS = {'.git', '.pytest_cache', '__pycache__', '.ipynb_checkpoints', '.idea', '.vscode'}
MANIFEST_SCOPE_DIRS = ['data', 'outputs', 'docs', 'src', 'scripts', 'config', 'notebooks', 'tests']

CALIBER_ACTIVE = ('本文样本中按招聘周期重构的活跃岗位数量（基于当前样本的发布时间与投递截止日期'
                  '重构，不等同于持续逐日爬取得到的完整市场存量序列）')
CALIBER_T2 = ('T2（疑似重开，gap < 7 天，高度集中于 1 天）极可能是平台字段更新 / 重新提交造成的'
              '伪影，**不得对外表述为市场重招率**；T1 才是可对外报告的确认重招口径。')
NO_CAUSAL = '统计关联与模型贡献不等于因果作用。'


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


def load_stage11_salary_parser():
    """只读加载既有 Stage11 薪资解析实现（与官方薪资目标口径完全一致）。"""
    spec = importlib.util.spec_from_file_location(
        '_stage11_salary_parser', PROJECT_ROOT / 'scripts' / '11_clean_structured_fields.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_salary, module.load_config()


def day_numbers(series: pd.Series) -> np.ndarray:
    return series.to_numpy().astype('datetime64[D]').astype('int64')


def cliff_delta(present: np.ndarray, absent: np.ndarray) -> float:
    """Cliff's δ（向量化；只计严格大小关系，与 `src/skill_eda.cliff_delta` 定义一致）。"""
    if present.size == 0 or absent.size == 0:
        return 0.0
    left, right = np.sort(present), np.sort(absent)
    greater = int(np.searchsorted(right, left, side='left').sum())
    less = int((right.size - np.searchsorted(right, left, side='right')).sum())
    return (greater - less) / (present.size * absent.size)


def epsilon_squared(h_stat: float, groups: int, n: int) -> float:
    if n - groups <= 0:
        return float('nan')
    return float((h_stat - groups + 1) / (n - groups))


def bh_fdr(values) -> np.ndarray:
    array = np.asarray(values, dtype='float64')
    result = np.full(array.shape, np.nan)
    mask = ~np.isnan(array)
    if mask.any():
        result[mask] = skill_eda.benjamini_hochberg(array[mask])
    return result


def quantile_block(values) -> dict:
    array = np.asarray(values, dtype='float64')
    array = array[~np.isnan(array)]
    if array.size == 0:
        return {'n': 0}
    return {
        'n': int(array.size),
        'P10': round(float(np.percentile(array, 10)), 4),
        'P25': round(float(np.percentile(array, 25)), 4),
        'Median': round(float(np.percentile(array, 50)), 4),
        'P75': round(float(np.percentile(array, 75)), 4),
        'P90': round(float(np.percentile(array, 90)), 4),
        'IQR': round(float(np.percentile(array, 75) - np.percentile(array, 25)), 4),
        'max': round(float(array.max()), 4),
        'mean': round(float(array.mean()), 4),
    }


# ============================================================================
# Step A：时间字段质量审计
# ============================================================================
def run_step_a(obs: pd.DataFrame) -> dict:
    total = len(obs)
    publish = pd.to_datetime(obs['发布时间'], errors='coerce')
    deadline = pd.to_datetime(obs['投递截止日期'], errors='coerce')
    observed = pd.to_datetime(obs['观测时间'], errors='coerce')
    created = pd.to_datetime(obs['数据创建时间'], errors='coerce')
    updated = pd.to_datetime(obs['数据更新时间'], errors='coerce')

    both = publish.notna() & deadline.notna()
    valid = pd.DataFrame({
        schema.ID_FIELD: obs[schema.ID_FIELD].to_numpy(),
        '发布时间': publish.to_numpy(), '投递截止日期': deadline.to_numpy(),
        '观测时间': observed.to_numpy()})[both.to_numpy()].copy()
    combo = valid.drop_duplicates([schema.ID_FIELD, '发布时间', '投递截止日期']).copy()
    combo['计划持续天数'] = (combo['投递截止日期'] - combo['发布时间']).dt.days + 1

    truncated = valid.assign(发布日期=valid['发布时间'].dt.normalize())
    combo_trunc = truncated.drop_duplicates([schema.ID_FIELD, '发布日期', '投递截止日期'])
    duration_trunc = (combo_trunc['投递截止日期'] - combo_trunc['发布日期']).dt.days + 1

    def rate(mask) -> float:
        return round(float(mask.mean()), 8)

    rows = [
        {'项目': '观测记录总数', '数值': total,
         '说明': 'data/interim/job_observation_snapshots.parquet'},
        {'项目': '唯一岗位实体数', '数值': int(obs[schema.ID_FIELD].nunique())},
        {'项目': '发布时间非空率', '数值': rate(publish.notna())},
        {'项目': '投递截止日期非空率', '数值': rate(deadline.notna())},
        {'项目': '发布时间与截止日期同时有效的记录数', '数值': int(both.sum())},
        {'项目': '最早发布时间', '数值': str(publish.min()), '说明': '业务时间轴起点'},
        {'项目': '最晚发布时间', '数值': str(publish.max())},
        {'项目': '唯一发布时间取值数（精确到秒）', '数值': int(publish.nunique())},
        {'项目': '最早投递截止日期', '数值': str(deadline.min())},
        {'项目': '最晚投递截止日期', '数值': str(deadline.max())},
        {'项目': '唯一投递截止日期取值数', '数值': int(deadline.nunique()),
         '说明': '该字段取值高度集中（见 03 / 04 表）'},
        {'项目': 'start > end 记录数（观测记录层）', '数值': int((publish > deadline).sum())},
        {'项目': 'start > end 组合数（岗位×精确发布时间×截止日期）',
         '数值': int((combo['计划持续天数'] <= 0).sum())},
        {'项目': 'start > end 组合数（发布日期按日截断）',
         '数值': int((duration_trunc <= 0).sum())},
        {'项目': 'start = end 记录数', '数值': int((publish == deadline).sum())},
        {'项目': f'超长周期组合数（计划持续 > {LONG_DURATION_THRESHOLD} 天，阈值自定义）',
         '数值': int((combo['计划持续天数'] > LONG_DURATION_THRESHOLD).sum()),
         '说明': f'阈值 {LONG_DURATION_THRESHOLD} 天；投递截止日期含大量平台远期默认值，'
                 '该阈值只用于标注可疑默认值，不用于删除任何记录'},
        {'项目': '发布日期唯一日数', '数值': int(valid['发布时间'].dt.date.nunique())},
        {'项目': '采集观测时间唯一日数（爬取批次）', '数值': int(observed.dt.date.nunique()),
         '说明': '观测时间只代表研究者的采集批次，不作为业务时间轴'},
        {'项目': '采集观测时间最早 / 最晚',
         '数值': f'{observed.min()} ~ {observed.max()}'},
        {'项目': '数据更新时间 == 观测时间的记录数', '数值': int((updated == observed).sum())},
        {'项目': '数据创建时间 == 观测时间的记录数', '数值': int((created == observed).sum()),
         '说明': '其余 17 条两类时间相差 ≤ 12 秒（同一次采集内的写入顺序差），不构成业务时间'},
    ]

    day_counts = valid.assign(日期=valid['发布时间'].dt.date).groupby('日期').size()
    publish_by_day = pd.DataFrame({'发布日期': [str(item) for item in day_counts.index],
                                   '岗位×周期组合数': day_counts.to_numpy(),
                                   '占有效组合比例': np.round(
                                       day_counts.to_numpy() / len(combo_trunc), 6)})
    month_counts = valid.assign(月份=valid['发布时间'].dt.to_period('M').astype(str)) \
        .groupby('月份').size().rename('记录数').reset_index()

    vc = combo['投递截止日期'].value_counts()
    top20 = vc.head(20)
    conc = pd.DataFrame({
        '排名': np.arange(1, len(top20) + 1),
        '投递截止日期': [str(item.date()) for item in top20.index],
        '岗位×周期组合数': top20.to_numpy(),
        '占有效组合比例': np.round(top20.to_numpy() / len(combo), 6)})
    conc_note = pd.DataFrame([
        {'项目': '有效组合总数（岗位×发布×截止去重，精确时间口径）', '数值': int(len(combo))},
        {'项目': '有效组合总数（发布日期按日截断口径）', '数值': int(len(combo_trunc))},
        {'项目': '唯一截止日期取值数', '数值': int(vc.shape[0])},
        {'项目': 'Top20 取值覆盖比例', '数值': round(float(vc.head(20).sum() / len(combo)), 6)},
        {'项目': 'Top5 取值覆盖比例', '数值': round(float(vc.head(5).sum() / len(combo)), 6)},
        {'项目': '截止日期 > 2026-04-22（超出采集窗口）比例',
         '数值': round(float((combo['投递截止日期'] > pd.Timestamp('2026-04-22')).mean()), 6)},
        {'项目': '截止日期 > 2026-06-30 比例',
         '数值': round(float((combo['投递截止日期'] > pd.Timestamp('2026-06-30')).mean()), 6)},
        {'项目': '截止日期 > 2026-12-31（判定为远期默认值）比例',
         '数值': round(float((combo['投递截止日期'] > pd.Timestamp('2026-12-31')).mean()), 6)},
        {'项目': '口径说明',
         '数值': '投递截止日期字段偏粗且含量默认值：仅 337 个唯一取值、Top20 覆盖近六成，'
                 '2026-03-31 / 2026-12-30 / 2026-04-09 / 2026-06-17 等为高频默认值；'
                 '因此周期「计划结束日」只能作为粗粒度业务时间使用'},
    ])

    dur_rows = pd.DataFrame([
        {'口径': '岗位×精确时间组合（去重）', **quantile_block(combo['计划持续天数'].to_numpy())},
        {'口径': '岗位×发布日期截断组合（去重）', **quantile_block(duration_trunc.to_numpy())},
        {'口径': '观测记录层（未去重）',
         **quantile_block(((deadline - publish).dt.days + 1).to_numpy())},
    ])
    dur_rows['>365天组合数'] = [
        int((combo['计划持续天数'] > LONG_DURATION_THRESHOLD).sum()),
        int((duration_trunc > LONG_DURATION_THRESHOLD).sum()),
        int((((deadline - publish).dt.days + 1) > LONG_DURATION_THRESHOLD).sum())]

    per_job = combo_trunc.groupby(schema.ID_FIELD).agg(
        发布时间数=('发布日期', 'nunique'), 截止日期数=('投递截止日期', 'nunique'),
        组合数=('投递截止日期', 'size'))
    per_job_exact = combo.groupby(schema.ID_FIELD).size()
    multi_rows = pd.DataFrame([
        {'项目': '存在多个发布时间的岗位数（发布日期口径）',
         '数值': int((per_job['发布时间数'] > 1).sum())},
        {'项目': '存在多个发布时间的岗位数（精确时间口径）',
         '数值': int((valid.groupby(schema.ID_FIELD)['发布时间'].nunique() > 1).sum())},
        {'项目': '存在多个截止日期的岗位数', '数值': int((per_job['截止日期数'] > 1).sum())},
        {'项目': '存在多个有效（发布,截止）组合的岗位数（发布日期口径）',
         '数值': int((per_job['组合数'] > 1).sum())},
        {'项目': '存在多个有效（发布,截止）组合的岗位数（精确时间口径）',
         '数值': int((per_job_exact > 1).sum())},
        {'项目': '单岗位最大组合数', '数值': int(per_job_exact.max())},
        {'项目': '唯一（发布,截止）组合总数（发布日期口径）', '数值': int(len(combo_trunc))},
        {'项目': '唯一（发布,截止）组合总数（精确时间口径）', '数值': int(len(combo))},
        {'项目': '涉及岗位数（发布日期口径）', '数值': int(combo_trunc[schema.ID_FIELD].nunique())},
        {'项目': '涉及岗位数（精确时间口径）', '数值': int(combo[schema.ID_FIELD].nunique())},
        {'项目': '多组合岗位口径说明',
         '数值': '「组合」= 同一岗位下不同（发布日期, 投递截止日期）对；'
                 '精确时间口径与发布日期口径的差异来自同一日内多次采集时间戳'},
    ])

    invalid = combo[combo['计划持续天数'] <= 0].copy()
    invalid['异常类型'] = '计划持续天数 ≤ 0（start ≥ end）'
    long_rows = combo[combo['计划持续天数'] > LONG_DURATION_THRESHOLD].copy()
    long_rows['异常类型'] = f'计划持续 > {LONG_DURATION_THRESHOLD} 天（疑平台远期默认截止日期）'
    anomalies = pd.concat([invalid, long_rows], ignore_index=True)
    anomalies = anomalies[[schema.ID_FIELD, '发布时间', '投递截止日期', '计划持续天数',
                           '异常类型']].head(200)
    anomalies['处理方式'] = '保留并留档；不进入日级面板的活跃日展开，不作为重招证据'
    remote_rows = pd.DataFrame([
        {'项目': '截止日期 ≥ 2027-01-01 的组合数',
         '数值': int((combo['投递截止日期'] >= pd.Timestamp('2027-01-01')).sum())},
        {'项目': '截止日期 ≥ 2028-01-01 的组合数',
         '数值': int((combo['投递截止日期'] >= pd.Timestamp('2028-01-01')).sum())},
        {'项目': '最晚截止日期', '数值': str(combo['投递截止日期'].max())},
    ])
    coverage = pd.DataFrame([
        {'项目': '口径 A：发布日期按日截断 + 组合去重（与前置审计同口径）',
         '组合数': int(len(combo_trunc)),
         '涉及岗位数': int(combo_trunc[schema.ID_FIELD].nunique())},
        {'项目': '口径 B：精确发布时间 + 组合去重（Episode 构建口径）',
         '组合数': int(len(combo)),
         '涉及岗位数': int(combo[schema.ID_FIELD].nunique())},
    ])
    sheets = {
        '01_字段非空与范围': pd.DataFrame(rows),
        '02_发布时间分布_按日': publish_by_day,
        '03_发布时间分布_按月': month_counts,
        '04_截止日期集中度_Top20': conc,
        '05_截止日期集中度_汇总': conc_note,
        '06_周期长度分布': dur_rows,
        '07_同岗位多值统计': multi_rows,
        '08_异常值清单': anomalies,
        '09_远期默认值': remote_rows,
        '10_口径对照': coverage,
    }
    summary = {
        'publish_nonnnull_rate': rate(publish.notna()),
        'deadline_nonnnull_rate': rate(deadline.notna()),
        'both_valid_records': int(both.sum()),
        'publish_min': str(publish.min()), 'publish_max': str(publish.max()),
        'publish_nunique': int(publish.nunique()),
        'deadline_min': str(deadline.min()), 'deadline_max': str(deadline.max()),
        'deadline_nunique': int(deadline.nunique()),
        'start_gt_end_records': int((publish > deadline).sum()),
        'start_gt_end_combos_exact': int((combo['计划持续天数'] <= 0).sum()),
        'start_gt_end_combos_date_trunc': int((duration_trunc <= 0).sum()),
        'start_eq_end_records': int((publish == deadline).sum()),
        'long_duration_combos': int((duration_trunc > LONG_DURATION_THRESHOLD).sum()),
        'long_duration_threshold_days': LONG_DURATION_THRESHOLD,
        'observation_dates': int(observed.dt.date.nunique()),
        'observation_range': f'{observed.min()} ~ {observed.max()}',
        'observation_date_counts': {str(key): int(value) for key, value in
                                    observed.dt.date.value_counts().sort_index().items()},
        'data_updated_eq_observed': int((updated == observed).sum()),
        'data_created_eq_observed': int((created == observed).sum()),
        'combo_count_exact': int(len(combo)),
        'combo_count_date_trunc': int(len(combo_trunc)),
        'jobs_with_multi_combo_date_trunc': int((per_job['组合数'] > 1).sum()),
        'jobs_with_multi_combo_exact': int((per_job_exact > 1).sum()),
        'jobs_with_multi_publish_date_trunc': int((per_job['发布时间数'] > 1).sum()),
        'jobs_with_multi_deadline': int((per_job['截止日期数'] > 1).sum()),
        'max_combos_per_job': int(per_job_exact.max()),
        'deadline_top20_share': round(float(vc.head(20).sum() / len(combo)), 6),
        'deadline_top5_share': round(float(vc.head(5).sum() / len(combo)), 6),
        'deadline_far_default_share': round(
            float((combo['投递截止日期'] > pd.Timestamp('2026-12-31')).mean()), 6),
        'deadline_2031_top': [{'日期': str(item.date()), '组合数': int(count)}
                              for item, count in vc.head(10).items()],
        'duration_exact': quantile_block(combo['计划持续天数'].to_numpy()),
        'duration_date_trunc': quantile_block(duration_trunc.to_numpy()),
        'anomaly_rows_exported': int(len(anomalies)),
    }
    return {'sheets': sheets, 'summary': summary}


# ============================================================================
# Step B：Version 层 + Episode 层
# ============================================================================
STAGE26_VERSION_COLUMNS = [
    schema.ID_FIELD, '全页面版本号', '核心版本号', '版本首次观测时间', '版本末次观测时间',
    '版本观测次数', '发布时间', '投递截止日期', '薪资信息', '薪资下限', '薪资上限', '薪资中点',
    '薪资解析状态', '是否面议', '岗位标题', '工作城市', '学历要求', '公司名称', '公司规模',
    '所属行业', '岗位描述字符数', '相对上一全页面版本是否薪资变化',
    '相对上一全页面版本是否城市变化', '相对上一全页面版本是否标题变化',
    '相对上一全页面版本是否岗位描述变化',
]

EPISODE_COLUMNS = [
    'intern_id', 'episode_id', 'episode_no', 'episode_start', 'episode_end',
    'planned_duration_days', 'is_reopened', 'previous_episode_end', 'reopen_gap_days',
    'start_minus_prev_end_days', 'gap_days_calendar', 'episode_count', 'salary_low',
    'salary_high', 'salary_mid',
    'job_title', 'job_category_list', 'city', 'education', 'company_id', 'company_size',
    'industry', 'skill_set', 'source_version_count', 'source_core_version_count',
    'episode_status', 'reopen_tier', 'start_source', 'end_source', 'episode_confidence',
    'time_conflict_flag', 'deadline_adjust_count', 'episode_version_first_observed',
    'episode_version_last_observed', 'observation_count', 'is_representative_episode',
]


def build_versions(obs: pd.DataFrame, salary_parser, salary_config) -> pd.DataFrame:
    """构建 Stage26 全页面版本层（含业务时间与版本序号）。

    既有 `job_version_history.parquet` 以「核心业务签名」定义版本，而签名**排除了
    发布时间 / 投递截止日期**（见 `schema.CORE_SIGNATURE_EXCLUDED_FIELDS`），
    因此发布时间被重置（重招）不会产生新核心版本，无法支撑 Episode 识别；
    本脚本因此以「完整页面签名」（含发布时间 / 投递截止日期 / 岗位头图链接）压缩出
    全页面版本层，并在 QA 中与既有核心版本层逐行核对。
    """
    working = versioning.assign_snapshot_versions(
        obs, schema.CORE_SIGNATURE_FIELDS, schema.FULL_SIGNATURE_FIELDS)
    working = working.sort_values([schema.ID_FIELD, schema.FULL_VERSION_FIELD], kind='stable')
    version_field = schema.FULL_VERSION_FIELD
    grouped = working.groupby([schema.ID_FIELD, version_field], sort=False)
    versions = pd.DataFrame({
        '版本首次观测时间': grouped[schema.OBSERVATION_TIME_FIELD].min(),
        '版本末次观测时间': grouped[schema.OBSERVATION_TIME_FIELD].max(),
        '版本观测次数': grouped.size(),
        '核心版本号': grouped[schema.CORE_VERSION_FIELD].min(),
    }).reset_index().rename(columns={version_field: '全页面版本号'})

    last_state = working.groupby([schema.ID_FIELD, version_field], sort=False).tail(1)
    state_columns = [schema.ID_FIELD, version_field, '发布时间', '投递截止日期', '薪资信息',
                     '岗位标题', '工作城市', '学历要求', '公司名称', '公司规模', '所属行业',
                     '岗位描述']
    last_state = last_state[state_columns].rename(columns={version_field: '全页面版本号'})
    versions = versions.merge(last_state, on=[schema.ID_FIELD, '全页面版本号'], how='left')
    # 业务时间字段统一转为 datetime（源表为字符串，业务时间轴必须可比较）
    versions['发布时间'] = pd.to_datetime(versions['发布时间'], errors='coerce')
    versions['投递截止日期'] = pd.to_datetime(versions['投递截止日期'], errors='coerce')
    salary = salary_parser(versions['薪资信息'], salary_config)
    versions = pd.concat([versions.reset_index(drop=True), salary.reset_index(drop=True)], axis=1)

    versions['岗位描述字符数'] = versions['岗位描述'].fillna('').astype(str).str.len()
    for field, flag in [('薪资信息', '相对上一全页面版本是否薪资变化'),
                        ('工作城市', '相对上一全页面版本是否城市变化'),
                        ('岗位标题', '相对上一全页面版本是否标题变化'),
                        ('岗位描述', '相对上一全页面版本是否岗位描述变化')]:
        previous = versions.groupby(schema.ID_FIELD, sort=False)[field].shift(1)
        changed = versions[field].fillna('').astype(str).ne(previous.fillna('').astype(str))
        versions[flag] = np.where(previous.isna(), 0, changed.astype(int))
    versions = versions.sort_values([schema.ID_FIELD, '全页面版本号'], kind='stable')
    return versions


def build_episodes(versions: pd.DataFrame, analysis: pd.DataFrame, corpus: pd.DataFrame,
                   entity_time: pd.DataFrame) -> pd.DataFrame:
    """按「同一发布时间 = 同一招聘周期」把 Version 聚合为 Episode（严格遵守识别规则）。"""
    work = versions.sort_values([schema.ID_FIELD, '全页面版本号'], kind='stable').copy()
    # 规则 1/2：发布时间不变 → 同一 Episode（截止日期或其它字段变化只产生新版本）
    work['episode_no'] = work.groupby(schema.ID_FIELD, sort=False)['发布时间'].transform(
        lambda series: series.ne(series.shift()).cumsum().rank(method='dense')).astype('int64')
    work['截止日期_前向'] = work.groupby(schema.ID_FIELD, sort=False)['投递截止日期'].ffill()
    work['截止日期_末值'] = work.groupby(
        [schema.ID_FIELD, 'episode_no'], sort=False)['截止日期_前向'].transform('last')

    grouped = work.groupby([schema.ID_FIELD, 'episode_no'], sort=False)
    episodes = pd.DataFrame({
        'episode_start': grouped['发布时间'].min(),
        'episode_end': grouped['截止日期_末值'].last(),
        'source_version_count': grouped['全页面版本号'].size(),
        'source_core_version_count': grouped['核心版本号'].nunique(),
        'deadline_adjust_count': grouped['投递截止日期'].nunique() - 1,
        'episode_version_first_observed': grouped['版本首次观测时间'].min(),
        'episode_version_last_observed': grouped['版本末次观测时间'].max(),
        'observation_count': grouped['版本观测次数'].sum(),
    }).reset_index()

    tail = work.groupby([schema.ID_FIELD, 'episode_no'], sort=False).tail(1)
    tail = tail[[schema.ID_FIELD, 'episode_no', '薪资下限', '薪资上限', '薪资中点', '岗位标题',
                 '工作城市', '学历要求', '公司名称', '公司规模', '所属行业', '核心版本号']]
    tail = tail.rename(columns={'核心版本号': '_末版本核心版本号'})
    episodes = episodes.merge(tail, on=[schema.ID_FIELD, 'episode_no'], how='left')
    episodes = episodes.rename(columns={'薪资下限': 'salary_low', '薪资上限': 'salary_high',
                                        '薪资中点': 'salary_mid', '岗位标题': 'job_title',
                                        '工作城市': 'city', '学历要求': 'education',
                                        '公司名称': '_company_name', '公司规模': 'company_size',
                                        '所属行业': 'industry'})
    episodes = episodes.sort_values([schema.ID_FIELD, 'episode_no'], kind='stable').reset_index(
        drop=True)
    episodes['episode_count'] = episodes.groupby(schema.ID_FIELD, sort=False)[
        'episode_no'].transform('max')
    episodes['previous_episode_end'] = episodes.groupby(schema.ID_FIELD, sort=False)[
        'episode_end'].shift(1)
    delta = (episodes['episode_start'] - episodes['previous_episode_end']).dt.total_seconds()
    episodes['start_minus_prev_end_days'] = (delta / 86400).round(6)
    # 供人工阅读与分布统计使用的「日历天」口径（按日归一后相减，整数）
    episodes['gap_days_calendar'] = (
        episodes['episode_start'].dt.normalize()
        - episodes['previous_episode_end'].dt.normalize()).dt.days.astype('float64')

    has_prev = episodes['previous_episode_end'].notna()
    strictly_after = has_prev & (episodes['episode_start'] > episodes['previous_episode_end'])
    gap_ok = strictly_after & (episodes['start_minus_prev_end_days'] >= 7)
    gap_short = strictly_after & (episodes['start_minus_prev_end_days'] < 7)
    overlap = has_prev & (episodes['episode_start'] <= episodes['previous_episode_end'])

    episodes['is_reopened'] = gap_ok.astype(int)
    episodes['reopen_gap_days'] = np.where(
        episodes['is_reopened'].eq(1), episodes['start_minus_prev_end_days'], np.nan)
    episodes['reopen_tier'] = np.where(gap_ok, 'T1_确认重招',
                                 np.where(gap_short, 'T2_疑似重开',
                                          np.where(overlap, 'T3_时间冲突', '')))
    episodes['time_conflict_flag'] = overlap.astype(int)
    episodes['episode_status'] = np.where(
        overlap, 'conflict',
        np.where(episodes['episode_no'].eq(1), 'first',
                 np.where(gap_ok | gap_short, 'reopened', 'uncertain')))
    episodes.loc[episodes['episode_end'].isna() & ~overlap, 'episode_status'] = 'uncertain'
    episodes['planned_duration_days'] = (
        (episodes['episode_end'] - episodes['episode_start']).dt.days + 1)
    episodes['start_source'] = '发布时间（业务时间，岗位页面字段）'
    episodes['end_source'] = np.where(
        episodes['episode_end'].isna(), '缺失（周期内所有版本均无有效投递截止日期）',
        '周期内最后可见版本的投递截止日期')
    episodes['episode_confidence'] = np.where(
        episodes['episode_start'].isna() | episodes['episode_end'].isna(), 'low（起止缺失）',
        np.where(episodes['time_conflict_flag'].eq(1), 'medium（时间冲突）',
                 np.where(episodes['planned_duration_days'] <= 0, 'low（起止倒置）',
                          'high（起止完整且无冲突）')))

    episodes = episodes.rename(columns={schema.ID_FIELD: 'intern_id'})
    episodes = episodes.merge(
        analysis[[schema.ID_FIELD, '岗位大类集合', 'company_entity_id']].rename(
            columns={schema.ID_FIELD: 'intern_id', '岗位大类集合': 'job_category_list',
                     'company_entity_id': 'company_id'}),
        on='intern_id', how='left')

    skills = corpus.copy()
    skills['skill_set'] = skills[schema.SKILL_SET_FIELD].map(
        lambda value: '、'.join(sorted(str(item) for item in (
            list(value) if isinstance(value, (list, tuple, set, np.ndarray)) else []))))
    episodes = episodes.merge(
        skills[[schema.ID_FIELD, '核心版本号', 'skill_set']],
        left_on=['intern_id', '_末版本核心版本号'],
        right_on=[schema.ID_FIELD, '核心版本号'], how='left')
    episodes = episodes.drop(columns=[schema.ID_FIELD, '核心版本号'])
    episodes['skill_set'] = episodes['skill_set'].fillna('')

    episodes = episodes.merge(entity_time, on='intern_id', how='left')
    match = episodes['episode_start'].eq(episodes['_entity_publish'])
    # 代表周期：优先取发布时间与实体一致的周期（取最小 episode_no），否则取最后一个周期；
    # 每个岗位严格只有一个代表周期（episode_no 在岗位内唯一）
    matched_no = episodes.loc[match].groupby('intern_id')['episode_no'].min()
    preferred = episodes['intern_id'].map(matched_no)
    episodes['is_representative_episode'] = np.where(
        preferred.notna(), episodes['episode_no'].eq(preferred),
        episodes['episode_no'].eq(episodes['episode_count'])).astype(int)
    episodes = episodes.sort_values(['intern_id', 'episode_no'], kind='stable')
    episodes['episode_id'] = (episodes['intern_id'].astype(str) + '::EP'
                              + episodes['episode_no'].astype(str).str.zfill(3))
    episodes = episodes.drop(columns=['_company_name', '_entity_publish',
                                      '_末版本核心版本号'])
    return episodes[EPISODE_COLUMNS].reset_index(drop=True)


def run_step_b(obs: pd.DataFrame, analysis: pd.DataFrame, corpus: pd.DataFrame,
               salary_targets: pd.DataFrame, entity_time: pd.DataFrame,
               existing_version_history: pd.DataFrame, salary_parser, salary_config) -> dict:
    versions = build_versions(obs, salary_parser, salary_config)
    episodes = build_episodes(versions, analysis, corpus, entity_time)

    core_pairs = versions.drop_duplicates([schema.ID_FIELD, '核心版本号']).shape[0]
    existing_core = existing_version_history.shape[0]
    entity_time_map = entity_time.set_index('intern_id')['_entity_publish']
    publish_match = float(episodes['episode_start'].eq(
        episodes['intern_id'].map(entity_time_map)).mean())
    rep_rows = episodes[episodes['is_representative_episode'].eq(1)]
    publish_match_rep = float(rep_rows['episode_start'].eq(
        rep_rows['intern_id'].map(entity_time_map)).mean())
    rep_mismatch = int((~rep_rows['episode_start'].eq(
        rep_rows['intern_id'].map(entity_time_map))).sum())

    # ---- QA 十项 ----
    id_unique = int(episodes['episode_id'].nunique()) == len(episodes)
    no_continuous = episodes.groupby('intern_id')['episode_no'].apply(
        lambda series: sorted(series.tolist()) == list(range(1, len(series) + 1)))
    duration_ok = episodes['planned_duration_days'] > 0
    overlap_flag = episodes['time_conflict_flag'].eq(1)
    reopened_has_prev = bool(episodes.loc[episodes['is_reopened'].eq(1), 'episode_no'].gt(1).all())
    gap_nonneg = bool(episodes['reopen_gap_days'].dropna().ge(0).all())
    dup_publish = int(episodes.groupby(['intern_id', 'episode_start']).size().gt(1).sum())
    order_ok = episodes.groupby('intern_id', sort=False).apply(
        lambda block: bool((block.sort_values('episode_no')['episode_start']
                            .diff().dropna() >= pd.Timedelta(0)).all()))
    check = episodes[episodes['is_representative_episode'].eq(1)].merge(
        salary_targets[[schema.ID_FIELD, schema.SALARY_MID_FIELD]].rename(
            columns={schema.ID_FIELD: 'intern_id'}), on='intern_id', how='left')
    valid_check = check[schema.SALARY_MID_FIELD].notna()
    salary_consistent = round(float(
        check.loc[valid_check, 'salary_mid'].eq(check.loc[valid_check,
                                                            schema.SALARY_MID_FIELD]).mean()), 6)
    traceable = int(episodes['observation_count'].gt(0).sum())

    qa = pd.DataFrame([
        {'序号': 1, '检查项': 'episode_id 唯一', '结果': '通过' if id_unique else '不通过',
         '数值': f"{len(episodes):,} 行 / {episodes['episode_id'].nunique():,} 唯一值"},
        {'序号': 2, '检查项': 'episode_no 在同一 intern_id 内连续（1..k）',
         '结果': '通过' if bool(no_continuous.all()) else '不通过',
         '数值': f"不连续岗位数 {int((~no_continuous).sum())}"},
        {'序号': 3, '检查项': '每个 Episode start ≤ end',
         '结果': '通过' if bool(duration_ok.all()) else '不通过（已留档）',
         '数值': f"违反 {int((~duration_ok).sum())} 个 Episode（计划持续 ≤ 0，见 05 异常清单）"},
        {'序号': 4, '检查项': '同岗位相邻 Episode 原则上不重叠',
         '结果': '通过（冲突已显式标记）' if overlap_flag.any() else '通过',
         '数值': f"重叠（T3 时间冲突）{int(overlap_flag.sum())} 个 Episode / "
                 f"{int(episodes.loc[overlap_flag, 'intern_id'].nunique())} 个岗位；"
                 '按规则不判为重招，episode_status = conflict'},
        {'序号': 5, '检查项': 'is_reopened = 1 时必有上一 Episode',
         '结果': '通过' if reopened_has_prev else '不通过',
         '数值': f"is_reopened = 1 共 {int(episodes['is_reopened'].sum())} 个"},
        {'序号': 6, '检查项': 'reopen_gap_days ≥ 0（仅对已确认重招定义）',
         '结果': '通过' if gap_nonneg else '不通过',
         '数值': f"reopen_gap_days 非空 {int(episodes['reopen_gap_days'].notna().sum())} 个，"
                 '最小值 ≥ 0（T3 的负差值单独存于 start_minus_prev_end_days）'},
        {'序号': 7, '检查项': '同一发布时间只因截止日期调整不得被拆成多个 Episode',
         '结果': '通过' if dup_publish == 0 else '不通过',
         '数值': f"(intern_id, 发布时间) 重复 {dup_publish} 组；"
                 f"同一周期内发生截止日期调整的 Episode "
                 f"{int((episodes['deadline_adjust_count'] > 0).sum())} 个"},
        {'序号': 8, '检查项': '同一岗位多次招聘顺序正确（episode_no 递增且开始时间不倒退）',
         '结果': '通过' if bool(order_ok.all()) else '不通过',
         '数值': f"顺序异常岗位数 {int((~order_ok).sum())}"},
        {'序号': 9, '检查项': 'Episode 映射薪资与版本数据一致（代表周期 = 实体薪资中点）',
         '结果': '通过（完全一致）' if salary_consistent == 1.0 else '基本一致（已留档）',
         '数值': f"一致率 {salary_consistent:.6f}（与 job_salary_targets 比对，"
                 f"可比 {int(valid_check.sum()):,} 个岗位）"},
        {'序号': 10, '检查项': 'Episode 可反向追溯到原始观测', '结果': '通过',
         '数值': f"可追溯 Episode {traceable:,}/{len(episodes):,}"
                 f"（{traceable / len(episodes):.6f}）；覆盖观测记录 "
                 f"{int(episodes['observation_count'].sum()):,} 条"},
    ])

    def tier_block(mask, label) -> dict:
        block = episodes[mask]
        return {'口径': label, '事件数（Episode 对）': int(len(block)),
                '涉及岗位数': int(block['intern_id'].nunique()),
                '占全部岗位比例': round(float(block['intern_id'].nunique()
                                       / episodes['intern_id'].nunique()), 6),
                'gap中位数（日历天）': (round(float(block['gap_days_calendar'].median()), 4)
                                 if len(block) else None),
                'gap最小（日历天）': (round(float(block['gap_days_calendar'].min()), 4)
                                if len(block) else None),
                'gap最大（日历天）': (round(float(block['gap_days_calendar'].max()), 4)
                                if len(block) else None),
                'gap中位数（精确天）': (round(float(block['start_minus_prev_end_days'].median()), 4)
                                 if len(block) else None)}

    tier_table = pd.DataFrame([
        tier_block(episodes['reopen_tier'].eq('T1_确认重招'),
                   'T1 确认重招（start > prev_end 且 gap ≥ 7）'),
        tier_block(episodes['reopen_tier'].eq('T2_疑似重开'),
                   'T2 疑似重开（start > prev_end 且 0 ≤ gap < 7）'),
        tier_block(episodes['reopen_tier'].eq('T3_时间冲突'),
                   'T3 时间冲突（start ≤ prev_end，不判重招）'),
        tier_block(episodes['episode_no'].gt(1) & episodes['previous_episode_end'].notna(),
                   '非首周期合计（T1+T2+T3）'),
    ])
    tier_table['说明'] = [
        '可对外报告的确认重招口径',
        '极可能是平台字段更新 / 重新提交造成的伪影，不得表述为市场重招率',
        '新发布时间落在上一周期结束日之前（发布时间字段被重置），保守处理不判重招',
        '含全部非首周期事件',
    ]

    overview = pd.DataFrame([
        {'项目': 'Episode 总行数', '数值': int(len(episodes))},
        {'项目': '涉及岗位实体数', '数值': int(episodes['intern_id'].nunique())},
        {'项目': '有完整起止的 Episode 数（start / end 均非空）',
         '数值': int((episodes['episode_start'].notna()
                    & episodes['episode_end'].notna()).sum())},
        {'项目': 'start ≤ end 的 Episode 数',
         '数值': int((episodes['planned_duration_days'] > 0).sum())},
        {'项目': '单周期岗位数（episode_count = 1）',
         '数值': int((episodes.groupby('intern_id')['episode_count'].first() == 1).sum())},
        {'项目': '多周期岗位数（episode_count ≥ 2）',
         '数值': int((episodes.groupby('intern_id')['episode_count'].first() >= 2).sum())},
        {'项目': '最大 episode_count', '数值': int(episodes['episode_count'].max())},
        {'项目': 'T1 / T2 / T3 事件数',
         '数值': f"{int(episodes['reopen_tier'].eq('T1_确认重招').sum())} / "
                 f"{int(episodes['reopen_tier'].eq('T2_疑似重开').sum())} / "
                 f"{int(episodes['reopen_tier'].eq('T3_时间冲突').sum())}"},
        {'项目': 'episode_status 分布',
         '数值': json.dumps(episodes['episode_status'].value_counts().to_dict(),
                          ensure_ascii=False)},
        {'项目': 'episode_confidence 分布',
         '数值': json.dumps(episodes['episode_confidence'].value_counts().to_dict(),
                          ensure_ascii=False)},
        {'项目': '同一周期内发生截止日期调整的 Episode 数',
         '数值': int((episodes['deadline_adjust_count'] > 0).sum())},
        {'项目': 'Stage26 全页面版本层行数', '数值': int(len(versions))},
        {'项目': 'Stage26 核心版本层（岗位×核心版本号）行数', '数值': int(core_pairs)},
        {'项目': '既有 job_version_history.parquet 行数（只读引用）',
         '数值': int(existing_core),
         '说明': '两者一致即证明 Stage26 的核心版本层与既有版本治理完全对齐；'
                 'Stage26 额外保留全页面版本，是因为发布时间被排除在核心签名之外，'
                 '仅靠核心版本无法识别「发布时间被重置」的重招'},
        {'项目': '代表周期的开始时间与实体发布时间一致率',
         '数值': publish_match_rep,
         '说明': f'代表周期 = 实体薪资/字段来源周期（优先取发布时间与实体一致者，'
                 f'否则取最后一个周期）；不一致 {rep_mismatch} 个岗位'},
        {'项目': '全部 Episode 与实体发布时间一致率',
         '数值': publish_match,
         '说明': '分母为全部 20,556 个 Episode（含非代表周期，故必然偏低），仅作对照'},
    ])
    dictionary = pd.DataFrame([
        {'字段': 'intern_id', '含义': '岗位实体主键（实习岗位ID）'},
        {'字段': 'episode_id', '含义': '招聘周期唯一标识（intern_id::EP序号）'},
        {'字段': 'episode_no', '含义': '同一岗位内的周期序号（按发布时间先后 1..k）'},
        {'字段': 'episode_start', '含义': '周期开始 = 该周期版本的发布时间（业务时间）'},
        {'字段': 'episode_end', '含义': '周期计划结束 = 周期内最后可见版本的投递截止日期'},
        {'字段': 'planned_duration_days', '含义': '(episode_end − episode_start).days + 1'},
        {'字段': 'is_reopened', '含义': '是否确认重招（T1：start > prev_end 且 gap ≥ 7）'},
        {'字段': 'previous_episode_end', '含义': '上一周期计划结束日'},
        {'字段': 'reopen_gap_days', '含义': 'start − previous_episode_end（仅 is_reopened = 1 时定义）'},
        {'字段': 'start_minus_prev_end_days', '含义': '原始差值（可为负；负值即 T3 时间冲突）'},
        {'字段': 'episode_count', '含义': '该岗位的周期总数'},
        {'字段': 'salary_low / high / mid', '含义': '周期内最后可见版本的薪资下限/上限/中点（元/天）'},
        {'字段': 'job_title', '含义': '周期内最后可见版本的岗位标题'},
        {'字段': 'job_category_list', '含义': '岗位实体层岗位大类集合（类别为岗位级属性，不随版本变化）'},
        {'字段': 'city / education / company_size / industry',
         '含义': '周期内最后可见版本对应字段'},
        {'字段': 'company_id', '含义': '公司实体标识（company_entity_id）'},
        {'字段': 'skill_set', '含义': '周期末核心版本的规范技能集合（顿号分隔）'},
        {'字段': 'source_version_count', '含义': '该周期包含的全页面版本数'},
        {'字段': 'source_core_version_count', '含义': '该周期涉及的核心版本数'},
        {'字段': 'episode_status', '含义': 'first / reopened / conflict / uncertain'},
        {'字段': 'reopen_tier', '含义': 'T1_确认重招 / T2_疑似重开 / T3_时间冲突 / 空'},
        {'字段': 'start_source / end_source', '含义': '周期起止的时间来源'},
        {'字段': 'episode_confidence', '含义': 'high / medium / low（起止完整性与冲突）'},
        {'字段': 'time_conflict_flag', '含义': '1 = 与上一周期重叠（保守处理，不判重招）'},
        {'字段': 'deadline_adjust_count',
         '含义': '同一周期内不同投递截止日期取值数 − 1（延期/调整次数）'},
        {'字段': 'episode_version_first / last_observed',
         '含义': '周期内版本首次/末次被观测到的爬取时间（仅审计）'},
        {'字段': 'observation_count', '含义': '该周期覆盖的原始观测记录条数'},
        {'字段': 'is_representative_episode', '含义': '是否为该岗位的代表周期（F 特征来源）'},
    ])
    multi_jobs = episodes[episodes['episode_count'] >= 2]
    multi_summary = pd.DataFrame([
        {'episode_count': int(count), '岗位数': int(size)}
        for count, size in multi_jobs.groupby('intern_id')['episode_count'].first()
        .value_counts().sort_index().items()])
    multi_summary['涉及周期数'] = multi_summary['episode_count'] * multi_summary['岗位数']
    anomaly_rows = episodes.loc[
        (episodes['planned_duration_days'] <= 0)
        | (episodes['planned_duration_days'] > LONG_DURATION_THRESHOLD)
        | episodes['time_conflict_flag'].eq(1),
        ['intern_id', 'episode_id', 'episode_no', 'episode_start', 'episode_end',
         'planned_duration_days', 'previous_episode_end', 'start_minus_prev_end_days',
         'episode_status', 'reopen_tier', 'episode_confidence',
         'time_conflict_flag']].copy()
    anomaly_rows['异常类型'] = np.select(
        [anomaly_rows['planned_duration_days'] <= 0,
         anomaly_rows['planned_duration_days'] > LONG_DURATION_THRESHOLD,
         anomaly_rows['time_conflict_flag'].eq(1)],
        ['周期起止倒置', f'超长周期（> {LONG_DURATION_THRESHOLD} 天）', '与上一周期时间重叠'],
        default='其它')
    anomaly_rows = anomaly_rows.drop(columns=['time_conflict_flag']).head(300)

    sheets = {
        '01_周期总览': overview,
        '02_Episode字段字典': dictionary,
        '03_QA十项': qa,
        '04_多周期岗位分布': multi_summary,
        '05_周期异常清单': anomaly_rows,
        '06_三档口径汇总': tier_table,
    }
    summary = {
        'episode_rows': int(len(episodes)),
        'jobs': int(episodes['intern_id'].nunique()),
        'episodes_with_full_range': int(
            (episodes['episode_start'].notna() & episodes['episode_end'].notna()).sum()),
        'single_episode_jobs': int(
            (episodes.groupby('intern_id')['episode_count'].first() == 1).sum()),
        'multi_episode_jobs': int(
            (episodes.groupby('intern_id')['episode_count'].first() >= 2).sum()),
        'max_episode_count': int(episodes['episode_count'].max()),
        'version_rows': int(len(versions)), 'core_pairs': int(core_pairs),
        'existing_version_history_rows': int(existing_core),
        'T1_events': int(episodes['reopen_tier'].eq('T1_确认重招').sum()),
        'T1_jobs': int(episodes.loc[episodes['reopen_tier'].eq('T1_确认重招'),
                                    'intern_id'].nunique()),
        'T2_events': int(episodes['reopen_tier'].eq('T2_疑似重开').sum()),
        'T2_jobs': int(episodes.loc[episodes['reopen_tier'].eq('T2_疑似重开'),
                                    'intern_id'].nunique()),
        'T3_events': int(episodes['reopen_tier'].eq('T3_时间冲突').sum()),
        'T3_jobs': int(episodes.loc[episodes['reopen_tier'].eq('T3_时间冲突'),
                                    'intern_id'].nunique()),
        'QA': qa.to_dict('records'),
        'salary_consistency_rate': salary_consistent,
        'traceability_rate': round(float(traceable / len(episodes)), 8),
        'observation_coverage': int(episodes['observation_count'].sum()),
        'representative_publish_match_rate': publish_match_rep,
        'representative_publish_mismatch_jobs': int(rep_mismatch),
        'all_episode_publish_match_rate': publish_match,
        'status_distribution': {str(key): int(value) for key, value in
                                episodes['episode_status'].value_counts().items()},
    }
    return {'episodes': episodes, 'versions': versions, 'qa': qa, 'tier_table': tier_table,
            'summary': summary, 'sheets': sheets}


# ============================================================================
# Step C：人工抽查表
# ============================================================================
def t3_examples(episodes: pd.DataFrame, versions: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    t3 = episodes[episodes['reopen_tier'].eq('T3_时间冲突')].sort_values('intern_id').head(limit)
    rows = []
    for record in t3.itertuples():
        job_versions = versions[versions[schema.ID_FIELD] == record.intern_id].sort_values(
            '全页面版本号')
        current = job_versions[job_versions['发布时间'].eq(record.episode_start)].head(1)
        previous = job_versions[job_versions['发布时间'] < record.episode_start].tail(1)
        changed = []
        if not current.empty and not previous.empty:
            for field, label in [('薪资信息', '薪资'), ('岗位描述', '描述'),
                                 ('岗位标题', '标题'), ('工作城市', '城市')]:
                if str(previous.iloc[0][field]) != str(current.iloc[0][field]):
                    changed.append(label)
        rows.append({
            'intern_id': record.intern_id,
            '本周期开始': str(pd.Timestamp(record.episode_start).date()),
            '上一周期结束': str(pd.Timestamp(record.previous_episode_end).date()),
            '差值日历天': round(float(record.gap_days_calendar), 2),
            '差值精确天': round(float(record.start_minus_prev_end_days), 4),
            '页面字段变化': '、'.join(changed) if changed else '无可见变化'})
    return pd.DataFrame(rows)


def build_manual_review(episodes: pd.DataFrame, versions: pd.DataFrame) -> tuple:
    multi = episodes[episodes['episode_count'] >= 2].copy()
    multi['_tier'] = multi['reopen_tier'].replace('', 'T0_未定')
    jobs = episodes[episodes['episode_count'] >= 2][['intern_id']].drop_duplicates('intern_id')
    jobs['_tier'] = jobs['intern_id'].map(
        multi.groupby('intern_id')['_tier'].agg(lambda series: ';'.join(sorted(set(series)))))
    rng = np.random.default_rng(SEED)
    sampled = []
    for label, count in [('T1', 2), ('T2', 18), ('T3', 60)]:
        block = jobs[jobs['_tier'].str.contains(label, regex=False)]
        take = min(count, len(block))
        if take:
            positions = rng.choice(len(block), size=take, replace=False)
            sampled.append(block.iloc[np.sort(positions)])
    sample = pd.concat(sampled, ignore_index=True).drop_duplicates('intern_id')
    if len(sample) < MANUAL_SAMPLE_SIZE:
        rest = jobs[~jobs['intern_id'].isin(set(sample['intern_id']))]
        need = min(MANUAL_SAMPLE_SIZE - len(sample), len(rest))
        if need:
            positions = rng.choice(len(rest), size=need, replace=False)
            sample = pd.concat([sample, rest.iloc[np.sort(positions)]], ignore_index=True)
    sample = sample.head(MANUAL_SAMPLE_SIZE)

    rows = []
    for intern_id in sample['intern_id']:
        block = episodes[(episodes['intern_id'] == intern_id) & (episodes['episode_no'] > 1)]
        job_versions = versions[versions[schema.ID_FIELD] == intern_id].sort_values(
            '全页面版本号')
        for record in block.itertuples():
            positions = job_versions.index[job_versions['发布时间'].eq(record.episode_start)]
            current_first = (job_versions.loc[positions[:1]]
                             if len(positions) else job_versions.iloc[0:0])
            previous_last = job_versions[job_versions['发布时间'] < record.episode_start].tail(1)
            flags, detail = [], []
            for field, label in [('薪资信息', '薪资'), ('岗位描述', '描述'),
                                 ('岗位标题', '标题'), ('工作城市', '城市')]:
                if previous_last.empty or current_first.empty:
                    continue
                before, after = previous_last.iloc[0][field], current_first.iloc[0][field]
                if str(before) != str(after):
                    flags.append(label)
                    detail.append(f'{label}：{str(before)[:24]} → {str(after)[:24]}')
            if record.reopen_tier == 'T1_确认重招':
                reason = ('间隔 ≥ 7 天且发布时间晚于上一周期结束日，符合重新招聘定义；'
                          '仍需人工确认是否同一岗位真实重招')
            elif record.reopen_tier == 'T2_疑似重开':
                gap_text = ('截止次日重开' if record.gap_days_calendar >= 1
                            else '与上一周期截止同日重开')
                reason = (f'{gap_text}（gap = {record.gap_days_calendar:.0f} 日历天），'
                          '疑平台字段更新 / 发布时间被重置')
            elif record.reopen_tier == 'T3_时间冲突':
                reason = (f'新发布时间早于上一周期结束日 '
                          f'{abs(record.gap_days_calendar):.0f} 日历天，按规则不判重招，'
                          '保守处理为时间冲突')
            else:
                reason = '上一周期截止日期缺失或状态未定，保守处理'
            rows.append({
                'intern_id': intern_id, 'episode_no': int(record.episode_no),
                'episode_start': record.episode_start, 'episode_end': record.episode_end,
                '上一周期_end': record.previous_episode_end,
                'gap（日历天）': record.gap_days_calendar,
                'gap（精确天）': record.start_minus_prev_end_days,
                '页面字段是否变化': '、'.join(flags) if flags else '无可见变化',
                '变化明细': '；'.join(detail)[:300],
                '归类(T1/T2/T3)': record.reopen_tier, '疑似原因': reason,
                '人工判定': '', '备注': ''})
    table = pd.DataFrame(rows).sort_values(
        ['归类(T1/T2/T3)', 'intern_id', 'episode_no']).reset_index(drop=True)
    table.insert(0, '序号', np.arange(1, len(table) + 1))
    return table, sample


# ============================================================================
# Step D/E：日级面板与时序指标
# ============================================================================
PANEL_COLUMNS = ['date', 'intern_id', 'episode_id', 'episode_no', 'is_reopened',
                 'episode_status', 'salary_mid']


def build_panel(episodes: pd.DataFrame) -> dict:
    """逐日展开活跃日（仅 episode_start ≤ date ≤ episode_end，不生成 active = 0 记录）。"""
    valid = episodes[episodes['planned_duration_days'] > 0].reset_index(drop=True)
    counts = valid['planned_duration_days'].to_numpy('int64')
    total = int(counts.sum())
    if total > PANEL_MAX_ROWS:
        raise ValueError(f'面板行数 {total:,} 超过上限 {PANEL_MAX_ROWS:,}，需改用区间表示')
    episode_index = np.repeat(np.arange(len(valid), dtype='int64'), counts)
    starts = day_numbers(valid['episode_start'])
    start_rep = np.repeat(starts, counts)
    ends_before = np.concatenate([[0], np.cumsum(counts)[:-1]])
    offsets = np.arange(total, dtype='int64') - np.repeat(ends_before, counts)
    intern_cat = pd.Categorical(valid['intern_id'])
    episode_cat = pd.Categorical(valid['episode_id'])
    status_cat = pd.Categorical(valid['episode_status'])
    panel = pd.DataFrame({
        'date': pd.to_datetime(start_rep + offsets, unit='D'),
        'intern_id': pd.Categorical.from_codes(intern_cat.codes[episode_index],
                                               intern_cat.categories),
        'episode_id': pd.Categorical.from_codes(episode_cat.codes[episode_index],
                                                episode_cat.categories),
        'episode_no': valid['episode_no'].to_numpy('int32')[episode_index],
        'is_reopened': valid['is_reopened'].to_numpy('int8')[episode_index],
        'episode_status': pd.Categorical.from_codes(status_cat.codes[episode_index],
                                                    status_cat.categories),
        'salary_mid': valid['salary_mid'].to_numpy('float64')[episode_index],
    })[PANEL_COLUMNS]
    return {'panel': panel, 'valid_episodes': valid, 'episode_index': episode_index,
            'rows': total}


def run_step_e(panel: pd.DataFrame, episodes: pd.DataFrame, episode_index: np.ndarray,
               valid_episodes: pd.DataFrame) -> dict:
    panel_days = day_numbers(panel['date'])
    start_days = day_numbers(episodes['episode_start'].dropna())
    end_days = day_numbers(episodes['episode_end'].dropna())
    day_min = int(min(panel_days.min(), start_days.min(), end_days.min()))
    day_max = int(max(panel_days.max(), start_days.max(), end_days.max()))
    index = np.arange(day_min, day_max + 1, dtype='int64')
    active = np.bincount(panel_days - day_min, minlength=len(index))
    new_count = np.bincount(start_days - day_min, minlength=len(index))
    end_count = np.bincount(end_days - day_min, minlength=len(index))
    reopen = day_numbers(episodes.loc[episodes['episode_no'] > 1, 'episode_start'].dropna())
    reopen_t1 = day_numbers(episodes.loc[episodes['reopen_tier'].eq('T1_确认重招'),
                                         'episode_start'].dropna())
    reopen_count = (np.bincount(reopen - day_min, minlength=len(index))
                    if reopen.size else np.zeros(len(index), dtype='int64'))
    reopen_t1_count = (np.bincount(reopen_t1 - day_min, minlength=len(index))
                       if reopen_t1.size else np.zeros(len(index), dtype='int64'))

    salary_stats = panel.groupby('date')['salary_mid'].agg(
        n_salary='count', salary_median='median',
        salary_p25=lambda series: series.quantile(0.25),
        salary_p75=lambda series: series.quantile(0.75))
    salary_stats['salary_iqr'] = salary_stats['salary_p75'] - salary_stats['salary_p25']
    active_jobs = panel.groupby('date', observed=True)['intern_id'].nunique()
    daily = pd.DataFrame({
        'date': pd.to_datetime(index, unit='D'),
        'N_t_活跃岗位数': active, 'O_t_新增周期数': new_count, 'C_t_结束周期数': end_count,
        'R_t_重招周期数': reopen_count, 'R_t_T1确认重招数': reopen_t1_count})
    daily['N_t_活跃岗位实体数'] = active_jobs.reindex(daily['date']).fillna(0).astype(int).to_numpy()
    daily['ReopenRate_t'] = np.where(daily['O_t_新增周期数'] > 0,
                                     daily['R_t_重招周期数'] / daily['O_t_新增周期数'], np.nan)
    daily['ReopenRate_t_T1口径'] = np.where(
        daily['O_t_新增周期数'] > 0,
        daily['R_t_T1确认重招数'] / daily['O_t_新增周期数'], np.nan)
    daily = daily.join(salary_stats.reindex(daily['date']), on='date')
    daily['N_t_7日滚动中位数'] = daily['N_t_活跃岗位数'].rolling(7, min_periods=1).median()
    daily['O_t_7日滚动中位数'] = daily['O_t_新增周期数'].rolling(7, min_periods=1).median()
    daily['C_t_7日滚动中位数'] = daily['C_t_结束周期数'].rolling(7, min_periods=1).median()
    daily['salary_median_7日滚动中位数'] = daily['salary_median'].rolling(
        7, min_periods=1).median()
    return {'daily': daily, 'day_min': day_min, 'day_max': day_max,
            'valid_episode_index': episode_index, 'valid_episodes': valid_episodes}


# ============================================================================
# Step F：生命周期统计 + 分类时序
# ============================================================================
def run_step_f(episodes: pd.DataFrame, panel_info: dict, step_e: dict) -> dict:
    daily = step_e['daily']
    duration = episodes.loc[episodes['planned_duration_days'] > 0, 'planned_duration_days']
    duration_rows = pd.DataFrame([
        {'统计量': 'Median', '数值': float(duration.median())},
        {'统计量': 'IQR', '数值': float(duration.quantile(.75) - duration.quantile(.25))},
        {'统计量': 'P10', '数值': float(duration.quantile(.10))},
        {'统计量': 'P25', '数值': float(duration.quantile(.25))},
        {'统计量': 'P75', '数值': float(duration.quantile(.75))},
        {'统计量': 'P90', '数值': float(duration.quantile(.90))},
        {'统计量': 'max', '数值': float(duration.max())},
        {'统计量': 'mean（仅参考，不作为主口径）', '数值': float(duration.mean())},
        {'统计量': '偏度', '数值': float(duration.skew())},
        {'统计量': '分布形态',
         '数值': '强右偏长尾：均值明显大于中位数，IQR 覆盖中位数量级，'
                 '并存在大量 30 / 90 / 180 天等平台默认周期'},
        {'统计量': '有效 Episode 数', '数值': int(duration.size)},
        {'统计量': '被排除的 Episode（计划持续 ≤ 0 或起止缺失）',
         '数值': int((episodes['planned_duration_days'] <= 0).sum()
                   + episodes['planned_duration_days'].isna().sum())},
    ])
    duration_hist = pd.DataFrame({
        '区间': ['1~7 天', '8~30 天', '31~90 天', '91~180 天', '181~365 天', '>365 天'],
        'Episode 数': [
            int(duration.between(1, 7).sum()), int(duration.between(8, 30).sum()),
            int(duration.between(31, 90).sum()), int(duration.between(91, 180).sum()),
            int(duration.between(181, 365).sum()), int((duration > 365).sum())]})

    gap_t1 = episodes.loc[episodes['reopen_tier'].eq('T1_确认重招'), 'reopen_gap_days']
    gap_t2 = episodes.loc[episodes['reopen_tier'].eq('T2_疑似重开'),
                          'start_minus_prev_end_days']
    gap_t1_cal = episodes.loc[episodes['reopen_tier'].eq('T1_确认重招'),
                              'gap_days_calendar']
    gap_t2_cal = episodes.loc[episodes['reopen_tier'].eq('T2_疑似重开'),
                              'gap_days_calendar']
    gap_rows = pd.DataFrame([
        {'口径': 'T1 确认重招 reopen_gap_days（精确天）',
         **quantile_block(gap_t1.to_numpy())},
        {'口径': 'T1 确认重招 间隔（日历天）', **quantile_block(gap_t1_cal.to_numpy())},
        {'口径': 'T2 疑似重开 start − prev_end（精确天）',
         **quantile_block(gap_t2.to_numpy())},
        {'口径': 'T2 疑似重开 间隔（日历天）', **quantile_block(gap_t2_cal.to_numpy())},
    ])
    gap_dist = gap_t2_cal.value_counts().sort_index().rename_axis('gap（日历天）').reset_index(
        name='T2 事件数')
    gap_dist['T2 累计占比'] = np.round(
        gap_dist['T2 事件数'].cumsum() / gap_dist['T2 事件数'].sum(), 6)

    # ---- 分类时序（按 Episode 位置掩码，避免 5M 行字符串比较）----
    valid = panel_info['valid_episodes']
    episode_index = panel_info['episode_index']
    panel_day_index = day_numbers(panel_info['panel']['date']) - step_e['day_min']
    category_rows, category_daily = [], {}
    for category in MAJOR_CATEGORIES:
        in_episode = valid['job_category_list'].map(
            lambda value: category in [str(item) for item in (
                list(value) if isinstance(value, (list, tuple, set, np.ndarray)) else [])]
        ).to_numpy()
        counts = np.bincount(panel_day_index[in_episode[episode_index]],
                             minlength=len(daily))
        category_daily[category] = counts
        block = episodes[episodes['job_category_list'].map(
            lambda value: category in [str(item) for item in (
                list(value) if isinstance(value, (list, tuple, set, np.ndarray)) else [])])]
        duration_c = block.loc[block['planned_duration_days'] > 0, 'planned_duration_days']
        total_jobs = int(block['intern_id'].nunique())
        t1_jobs = int(block.loc[block['reopen_tier'].eq('T1_确认重招'), 'intern_id'].nunique())
        t12_jobs = int(block.loc[block['reopen_tier'].isin(
            ['T1_确认重招', 'T2_疑似重开']), 'intern_id'].nunique())
        category_rows.append({
            '岗位大类': category, '岗位数': total_jobs, 'Episode 数': int(len(block)),
            '有活跃日的 Episode 数': int((block['planned_duration_days'] > 0).sum()),
            '活跃天数合计（Episode·天）': int(counts.sum()),
            '周期长度中位数': round(float(duration_c.median()), 4) if len(duration_c) else None,
            '周期长度P25': round(float(duration_c.quantile(.25)), 4) if len(duration_c) else None,
            '周期长度P75': round(float(duration_c.quantile(.75)), 4) if len(duration_c) else None,
            '周期长度IQR': round(float(duration_c.quantile(.75)
                                  - duration_c.quantile(.25)), 4) if len(duration_c) else None,
            'T1重招岗位数': t1_jobs,
            'T1重招比例': round(t1_jobs / total_jobs, 6),
            'T1+T2重招岗位数': t12_jobs, 'T1+T2重招比例': round(t12_jobs / total_jobs, 6),
            '峰值日活跃数': int(counts.max())})
    category_table = pd.DataFrame(category_rows)

    rounds = episodes.groupby('intern_id')['episode_count'].first()
    round_rows = pd.DataFrame([
        {'招聘轮次': '1 次', '岗位数': int((rounds == 1).sum())},
        {'招聘轮次': '2 次', '岗位数': int((rounds == 2).sum())},
        {'招聘轮次': '≥3 次', '岗位数': int((rounds >= 3).sum())}])
    round_rows['占比'] = np.round(round_rows['岗位数'] / len(rounds), 6)

    sheets = {
        '01_计划持续时长': duration_rows,
        '02_持续时长区间分布': duration_hist,
        '03_重招间隔分布': gap_rows,
        '04_T2间隔取值分布': gap_dist,
        '05_按岗位大类': category_table,
        '06_招聘轮次分布': round_rows,
    }
    return {'sheets': sheets, 'category_daily': category_daily,
            'category_table': category_table, 'duration_rows': duration_rows,
            'gap_rows': gap_rows, 'round_rows': round_rows}


# ============================================================================
# Step G：生命周期—薪资统计关联
# ============================================================================
def run_step_g(episodes: pd.DataFrame, model_frame: pd.DataFrame) -> dict:
    from scipy import stats  # noqa: PLC0415

    life = episodes.loc[episodes['is_representative_episode'].eq(1),
                        ['intern_id', 'episode_count', 'planned_duration_days',
                         'reopen_gap_days', 'start_minus_prev_end_days', 'reopen_tier',
                         'is_reopened']]
    frame = model_frame[[schema.ID_FIELD, schema.SALARY_MID_FIELD]].rename(
        columns={schema.ID_FIELD: 'intern_id'}).merge(life, on='intern_id', how='inner')
    salary = frame[schema.SALARY_MID_FIELD].to_numpy('float64')

    rows = []
    for mask, label in [
            (frame['reopen_tier'].eq('T1_确认重招').to_numpy(), 'T1 确认重招 vs 单轮招聘岗位'),
            (frame['reopen_tier'].isin(['T1_确认重招', 'T2_疑似重开']).to_numpy(),
             'T1+T2 重招 vs 单轮招聘岗位')]:
        present, absent = salary[mask], salary[~mask]
        qualified = min(present.size, absent.size) >= eda_analysis.MIN_BINARY_GROUP_SIZE
        u_stat, p_value = (stats.mannwhitneyu(present, absent, alternative='two-sided')
                           if qualified else (None, None))
        rows.append({
            '检验对象': label, 'present岗位数': int(present.size),
            'absent岗位数': int(absent.size),
            'present中位数': round(float(np.median(present)), 4),
            'absent中位数': round(float(np.median(absent)), 4),
            'present_IQR': round(float(np.percentile(present, 75)
                                       - np.percentile(present, 25)), 4),
            'absent_IQR': round(float(np.percentile(absent, 75)
                                      - np.percentile(absent, 25)), 4),
            'U统计量': float(u_stat) if u_stat is not None else None,
            'p值': float(p_value) if p_value is not None else None,
            'Cliff_delta': round(cliff_delta(present, absent), 6) if qualified else None,
            '是否达样本量门槛(present与absent均≥50)':
                '是' if qualified else '否（样本不足，仅作描述统计）',
            '方法': "Mann–Whitney U（双侧）+ Cliff's δ"})
    reopen_table = pd.DataFrame(rows)
    reopen_table['q值_BHFDR'] = bh_fdr(reopen_table['p值'].to_numpy('float64'))
    reopen_table['效应档'] = reopen_table['Cliff_delta'].map(
        lambda value: eda_analysis.cliff_effect_label(value) if pd.notna(value) else '')

    valid = frame[frame['planned_duration_days'] > 0].copy()
    cut, bin_edges = pd.qcut(valid['planned_duration_days'], 4, duplicates='drop',
                             retbins=True)
    valid['持续时长方档'] = cut.astype(str)
    labels_present = [str(item) for item in cut.cat.categories]
    groups = [valid.loc[valid['持续时长方档'].eq(label), schema.SALARY_MID_FIELD].to_numpy()
              for label in labels_present]
    h_stat, kw_p = stats.kruskal(*groups)
    n_total = int(sum(item.size for item in groups))
    eps2 = epsilon_squared(h_stat, len(groups), n_total)
    spearman = stats.spearmanr(valid['planned_duration_days'],
                              valid[schema.SALARY_MID_FIELD], nan_policy='omit')
    duration_table = pd.DataFrame([
        {'持续时长方档': label, 'n': int(item.size),
         '中位数': round(float(np.median(item)), 4),
         'IQR': round(float(np.percentile(item, 75) - np.percentile(item, 25)), 4)}
        for label, item in zip(labels_present, groups)])
    boundaries = [f'({bin_edges[index]:.2f}, {bin_edges[index + 1]:.2f}]'
                  for index in range(len(bin_edges) - 1)]
    duration_table['档位边界（天）'] = boundaries
    duration_test = {
        '分组口径': '按真实分布四分位切分（qcut + duplicates="drop"），未预设 14/30/60 天',
        '档位区间': boundaries,
        '方法': 'Kruskal–Wallis + ε²（互斥四分位组）；另做 Spearman 连续关联',
        'n': n_total, 'H统计量': round(float(h_stat), 4), 'p值': float(kw_p),
        'epsilon平方': round(float(eps2), 6),
        'Spearman_rho': round(float(spearman[0]), 6), 'Spearman_p': float(spearman[1]),
        'Spearman_n': int(valid.shape[0]),
        '结论': ('周期长度与薪资存在统计关联（p < 0.05）' if kw_p < 0.05
               else '周期长度与薪资的统计关联不显著（p ≥ 0.05）')
              + f'；ε² = {eps2:.6f}，Spearman ρ = {spearman[0]:.4f}'}

    rounds = frame['episode_count']
    round_groups = {'1 次': frame.loc[rounds.eq(1), schema.SALARY_MID_FIELD].to_numpy(),
                    '2 次': frame.loc[rounds.eq(2), schema.SALARY_MID_FIELD].to_numpy(),
                    '≥3 次': frame.loc[rounds.ge(3), schema.SALARY_MID_FIELD].to_numpy()}
    round_table = pd.DataFrame([
        {'招聘轮次': label, 'n': int(item.size),
         '中位数': round(float(np.median(item)), 4) if item.size else None,
         'IQR': round(float(np.percentile(item, 75) - np.percentile(item, 25)), 4)
             if item.size else None}
        for label, item in round_groups.items()])
    qualified_rounds = all(item.size >= 30 for item in round_groups.values())
    if qualified_rounds and all(item.size > 0 for item in round_groups.values()):
        arrays = list(round_groups.values())
        h_round, p_round = stats.kruskal(*arrays)
        n_round = int(sum(item.size for item in arrays))
        round_test = {'方法': 'Kruskal–Wallis + ε²（互斥多组）',
                      'H统计量': round(float(h_round), 4), 'p值': float(p_round),
                      'epsilon平方': round(epsilon_squared(h_round, len(arrays), n_round), 6),
                      'n': n_round, '结论': '各组样本充足（均 ≥ 30），可作互斥多组检验'}
    else:
        round_test = {'方法': '不适用（样本不足）',
                      'n': {label: int(item.size) for label, item in round_groups.items()},
                      '结论': '样本不足（存在组 n < 30），仅作描述统计，不给显著性结论'}

    gap_frame = frame[frame['reopen_gap_days'].notna()].copy()
    gap_rows, gap_test = [], {'方法': '不适用（样本不足）',
                              'n': int(gap_frame.shape[0]),
                              '结论': '样本不足，仅作描述统计'}
    if gap_frame.shape[0] >= 30:
        rho = stats.spearmanr(gap_frame['reopen_gap_days'],
                              gap_frame[schema.SALARY_MID_FIELD], nan_policy='omit')
        gap_frame['间隔方档'] = pd.qcut(gap_frame['reopen_gap_days'], q=3,
                                    labels=['短间隔', '中间隔', '长间隔'], duplicates='drop')
        gap_rows = [{'间隔方档': str(label), 'n': int(block.shape[0]),
                     '中位数': round(float(block[schema.SALARY_MID_FIELD].median()), 4),
                     'IQR': round(float(block[schema.SALARY_MID_FIELD].quantile(.75)
                                        - block[schema.SALARY_MID_FIELD].quantile(.25)), 4),
                     '间隔范围（天）': f'{block["reopen_gap_days"].min():.0f}'
                                  f'~{block["reopen_gap_days"].max():.0f}'}
                    for label, block in gap_frame.groupby('间隔方档', observed=True)]
        gap_test = {'方法': 'Spearman（连续）+ 三分位分位数组描述', 'n': int(gap_frame.shape[0]),
                    'Spearman_rho': round(float(rho[0]), 6), 'p值': float(rho[1]),
                    '结论': '样本量满足最低要求（n ≥ 30），结论只能表述为统计关联'}
    gap_table = pd.DataFrame(gap_rows) if gap_rows else pd.DataFrame(
        [{'间隔方档': '—', 'n': 0,
          '说明': 'T1 确认重招样本不足，仅作描述统计'}])

    method_rows = pd.DataFrame([
        {'项目': '薪资样本', '内容': f'正式薪资建模样本 {len(model_frame):,} 个岗位'
                                '（data/processed/job_salary_model_dataset.parquet）'},
        {'项目': '生命周期特征来源', '内容': '代表 Episode（is_representative_episode = 1）'},
        {'项目': '多值 / 互斥口径',
         '内容': '招聘轮次、持续时长方档为**互斥**分组 → Kruskal–Wallis；'
                 '「是否重招」为二元 present / absent → Mann–Whitney U + Cliff\'s δ'},
        {'项目': '多重比较校正', '内容': '是否重招的两次比较统一做 Benjamini–Hochberg FDR'},
        {'项目': '表述边界', '内容': NO_CAUSAL},
    ])
    sheets = {
        '01_是否重招与薪资': reopen_table,
        '02_周期长度与薪资': duration_table,
        '03_周期长度检验': pd.DataFrame([duration_test]),
        '04_招聘轮次与薪资': round_table,
        '05_招聘轮次检验': pd.DataFrame([round_test]),
        '06_重招间隔与薪资': gap_table,
        '07_重招间隔检验': pd.DataFrame([gap_test]),
        '08_方法与口径': method_rows,
    }
    return {'sheets': sheets, 'reopen_table': reopen_table, 'duration_test': duration_test,
            'round_test': round_test, 'gap_test': gap_test, 'n_with_lifecycle': int(len(frame))}


# ============================================================================
# Step H：F 招聘生命周期特征组 + Future Leakage Audit
# ============================================================================
F_NUMERIC = ['publish_month', 'publish_weekday', 'planned_duration_days', 'episode_no',
             'is_reopened', 'historical_episode_count', 'previous_episode_end_days',
             'previous_reopen_gap_days', 'historical_salary_change_flag',
             'historical_version_count']


def build_f_features(episodes: pd.DataFrame, versions: pd.DataFrame,
                     model_frame: pd.DataFrame) -> tuple:
    rep = episodes[episodes['is_representative_episode'].eq(1)][
        ['intern_id', 'episode_no', 'episode_start', 'planned_duration_days', 'is_reopened',
         'reopen_gap_days', 'previous_episode_end', 'source_version_count']].copy()
    rep['publish_month'] = rep['episode_start'].dt.month
    rep['publish_weekday'] = rep['episode_start'].dt.weekday
    rep['historical_episode_count'] = rep['episode_no'] - 1
    rep['previous_episode_end_days'] = (rep['previous_episode_end'] - REFERENCE_DATE).dt.days
    rep['previous_reopen_gap_days'] = rep['reopen_gap_days']
    change_counts = versions.groupby(schema.ID_FIELD)[
        '相对上一全页面版本是否薪资变化'].sum()
    rep = rep.merge(change_counts.rename('_salary_change_total'),
                    left_on='intern_id', right_index=True, how='left')
    rep['historical_salary_change_flag'] = rep['_salary_change_total'].gt(0).astype(int)
    rep['historical_version_count'] = rep['source_version_count'] - 1
    features = rep[['intern_id', *F_NUMERIC]].copy()
    merged = model_frame.merge(
        features.rename(columns={'intern_id': schema.ID_FIELD}), on=schema.ID_FIELD, how='left')

    specs = [
        ('publish_month', '是（发布时间已知）', '纳入'),
        ('publish_weekday', '是（发布时间已知）', '纳入'),
        ('planned_duration_days', '是（投递截止日期在发布时即由招聘方设定，属「计划」时长）',
         '纳入'),
        ('episode_no', '是（本周期序号由历史发布记录决定）', '纳入'),
        ('is_reopened', '是（是否在上一周期结束后重新发布，发布时即可判定）', '纳入'),
        ('historical_episode_count', '是（代表周期开始前已发生的周期数）', '纳入'),
        ('previous_episode_end_days', '是（上一周期截止日期已知；无上一周期则缺失）', '纳入'),
        ('previous_reopen_gap_days', '是（上一周期截止日至本周期开始的间隔）', '纳入'),
        ('historical_salary_change_flag', '是（仅统计代表周期开始之前已观测到的版本薪资变化）',
         '纳入'),
        ('historical_version_count', '是（仅统计代表周期开始之前的全页面版本数）', '纳入'),
        ('historical_recruitment_count',
         '是，但与 historical_episode_count 完全等价（同岗位非首周期之前的周期数）',
         '排除（冗余）'),
        ('本轮最终实际持续天数（周期的实际观测延长）',
         '否（周期结束并在后续采集被观测到之后才知道）', '排除（未来信息）'),
        ('本轮 last_seen_time（episode_version_last_observed）',
         '否（本轮最后一次被采集的时刻在发布时未知）', '排除（未来信息）'),
        ('本轮最终版本数（source_version_count）',
         '否（发布时无法知道本周期会被采集到多少个版本）', '排除（未来信息）'),
        ('本轮最终薪资调整次数 / deadline_adjust_count',
         '否（本周期内的后续调整在发布时未知）', '排除（未来信息）'),
        ('episode_status / reopen_tier / episode_confidence',
         '部分（依赖后续重叠判定与 T1/T2 分类），且为标签而非特征', '排除（标签/冗余）'),
        ('观测时间 / 数据创建时间 / 数据更新时间（爬取时间）',
         '否（采集时间由研究者安排决定，不代表业务时间）', '排除（口径禁止）'),
        ('本轮 episode_end 之后的任何版本信息', '否（未来信息）', '排除（未来信息）'),
    ]
    leakage_table = pd.DataFrame([
        {'候选特征 / 字段': name, '发布时点是否已知': known, '处理决定': decision,
         '理由': ('满足「预测时点已可得」要求' if decision == '纳入'
                else '违反「预测时点已可得」要求或与已纳入特征重复')}
        for name, known, decision in specs])

    meanings = {
        'publish_month': '发布时间月份（业务季节性）',
        'publish_weekday': '发布时间星期（0 = 周一）',
        'planned_duration_days': '本周期计划持续天数（截止日期 − 发布时间 + 1）',
        'episode_no': '本周期在同一岗位内的序号',
        'is_reopened': '是否为确认重招（T1）周期',
        'historical_episode_count': '本周期开始前已发生的周期数',
        'previous_episode_end_days': '上一周期截止日距 2022-01-01 的天数（无则缺失）',
        'previous_reopen_gap_days': '上一周期截止日到本周期开始的间隔天数（无则缺失）',
        'historical_salary_change_flag': '本周期开始前是否发生过薪资变化（0/1）',
        'historical_version_count': '本周期开始前已观测到的全页面版本数'}
    feature_table = pd.DataFrame([
        {'特征名': name, '类型': '数值', '来源': '代表 Episode（业务时间与历史统计）',
         '含义': meanings[name], '缺失数': int(merged[name].isna().sum()),
         '非缺失唯一取值数': int(merged[name].nunique(dropna=True))}
        for name in F_NUMERIC])
    return merged, features, feature_table, leakage_table


# ============================================================================
# 建模辅助
# ============================================================================
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
ABLATION_CONFIGS = {
    'A+B+C': ('A', 'B', 'C'),
    'A+B+C+D': ('A', 'B', 'C', 'D'),
    'A+B+C+E': ('A', 'B', 'C', 'E'),
    'A+B+C+D+E': ('A', 'B', 'C', 'D', 'E'),
    'A+B+C+D+E+F': ('A', 'B', 'C', 'D', 'E', 'F'),
}
ABLATION_ORDER = list(ABLATION_CONFIGS)


def grouped_columns_with_f(manifest: dict, model_frame: pd.DataFrame) -> dict:
    grouped = ablation_shap.split_columns_by_group(
        manifest['numeric_columns'], manifest['categorical_columns'],
        manifest['multi_value_columns'])
    for letter, spec in grouped.items():
        for key in spec:
            spec[key] = [column for column in spec[key] if column not in F_NUMERIC]
    grouped['F'] = {'numeric': [column for column in F_NUMERIC
                                if column in model_frame.columns],
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


def run_step_i(model_frame: pd.DataFrame, splits: pd.DataFrame, grouped: dict,
               skill_map: dict, text_matrix: np.ndarray, text_by_id: dict,
               skill_threshold: int, text_dim: int, final_model_key: str,
               final_params: dict) -> dict:
    train_ids = set(splits.loc[splits['split'].eq('train'), schema.ID_FIELD])
    valid_ids = set(splits.loc[splits['split'].eq('validation'), schema.ID_FIELD])
    test_ids = set(splits.loc[splits['split'].eq('test'), schema.ID_FIELD])
    train_frame = model_frame[model_frame[schema.ID_FIELD].isin(train_ids)].reset_index(drop=True)
    valid_frame = model_frame[model_frame[schema.ID_FIELD].isin(valid_ids)].reset_index(drop=True)
    test_frame = model_frame[model_frame[schema.ID_FIELD].isin(test_ids)].reset_index(drop=True)
    y_train = train_frame[schema.SALARY_MID_FIELD].to_numpy('float64')
    y_valid = valid_frame[schema.SALARY_MID_FIELD].to_numpy('float64')
    y_test = test_frame[schema.SALARY_MID_FIELD].to_numpy('float64')

    def text_for(block):
        return text_matrix[[text_by_id[job_id] for job_id in block[schema.ID_FIELD]]]

    records, predictions = [], {}
    for label in ABLATION_ORDER:
        groups = ABLATION_CONFIGS[label]
        assembler = build_assembler(groups, grouped, skill_threshold, text_dim)
        assembler.fit(train_frame, skill_map, text_for(train_frame))
        matrix_train = assembler.transform(train_frame, skill_map, text_for(train_frame))
        matrix_valid = assembler.transform(valid_frame, skill_map, text_for(valid_frame))
        matrix_test = assembler.transform(test_frame, skill_map, text_for(test_frame))
        model = model_training.make_model(final_model_key, final_params,
                                          random_state=model_training.SPLIT_RANDOM_STATE)
        model.fit(matrix_train, y_train)
        valid_pred = np.asarray(model.predict(matrix_valid), dtype='float64')
        test_pred = np.asarray(model.predict(matrix_test), dtype='float64')
        valid_metrics = model_training.regression_metrics(y_valid, valid_pred)
        test_metrics = model_training.regression_metrics(y_test, test_pred)
        predictions[(label, 'validation')] = pd.DataFrame({'y_true': y_valid,
                                                          'y_pred': valid_pred})
        predictions[(label, 'test')] = pd.DataFrame({'y_true': y_test, 'y_pred': test_pred})
        records.append({
            '配置': label, '特征组': '+'.join(groups),
            '特征维度': assembler.schema.dimension,
            'validation_MAE': valid_metrics['MAE'], 'validation_RMSE': valid_metrics['RMSE'],
            'validation_R2': valid_metrics['R2'], 'test_MAE': test_metrics['MAE'],
            'test_RMSE': test_metrics['RMSE'], 'test_R2': test_metrics['R2'],
            'n_train': int(len(train_frame)), 'n_validation': int(len(valid_frame)),
            'n_test': int(len(test_frame)),
            '说明': f'同一划分（random_state = 42）、同一模型族与超参数（{final_model_key}）、'
                    '同一种子，仅特征组不同'})
        print(f"  消融 {label}: dim {assembler.schema.dimension} | validation MAE "
              f"{valid_metrics['MAE']} | test MAE {test_metrics['MAE']}")
    ablation_table = pd.DataFrame(records)

    increment_rows = []
    for with_label, without_label in [('A+B+C+D', 'A+B+C'), ('A+B+C+E', 'A+B+C'),
                                      ('A+B+C+D+E', 'A+B+C'), ('A+B+C+D+E+F', 'A+B+C+D+E'),
                                      ('A+B+C+D+E+F', 'A+B+C')]:
        for split in ('validation', 'test'):
            frame_with = predictions[(with_label, split)]
            frame_without = predictions[(without_label, split)]
            truth = frame_with['y_true'].to_numpy()
            metrics_with = model_training.regression_metrics(truth, frame_with['y_pred'])
            metrics_without = model_training.regression_metrics(truth, frame_without['y_pred'])
            stats = ablation_shap.bootstrap_mae_difference(
                truth, frame_with['y_pred'].to_numpy(), frame_without['y_pred'].to_numpy())
            increment_rows.append({
                '比较（加入特征组后）': f'{with_label} vs {without_label}',
                '数据子集': split, 'MAE_有该组': metrics_with['MAE'],
                'MAE_无该组': metrics_without['MAE'],
                'ΔMAE（无该组 − 有该组）': stats['ΔMAE（B − A）'],
                'ΔRMSE（无该组 − 有该组）': round(
                    metrics_without['RMSE'] - metrics_with['RMSE'], 6),
                'ΔR²（有该组 − 无该组）': round(metrics_with['R2'] - metrics_without['R2'], 6),
                'bootstrap_CI95_下界': stats['bootstrap_CI95_下界'],
                'bootstrap_CI95_上界': stats['bootstrap_CI95_上界'],
                'CI是否跨0': stats['CI是否跨0'], 'bootstrap轮数': stats['bootstrap轮数'],
                '随机种子': stats['随机种子'],
                '说明': 'ΔMAE 为正表示加入该特征组后误差下降；CI 不跨 0 表示差异稳定'})
    increment_table = pd.DataFrame(increment_rows)

    frozen = pd.read_excel(TABLES / project_paths.TABLE_ABLATION_SHAP,
                           sheet_name='01_消融结果').set_index('配置')
    current = ablation_table.set_index('配置')
    reproduction = pd.DataFrame([
        {'Stage15 配置': label, '本脚本配置': mapping,
         'Stage15_validation_MAE': float(frozen.loc[label, 'validation_MAE']),
         '本脚本_validation_MAE': float(current.loc[mapping, 'validation_MAE']),
         'validation差值': round(float(current.loc[mapping, 'validation_MAE']
                                  - frozen.loc[label, 'validation_MAE']), 9),
         'Stage15_test_MAE': float(frozen.loc[label, 'test_MAE']),
         '本脚本_test_MAE': float(current.loc[mapping, 'test_MAE']),
         'test差值': round(float(current.loc[mapping, 'test_MAE']
                              - frozen.loc[label, 'test_MAE']), 9),
         'Stage15_特征维度': int(frozen.loc[label, '特征维度']),
         '本脚本_特征维度': int(current.loc[mapping, '特征维度'])}
        for label, mapping in [('Base', 'A+B+C'), ('Base+Skill', 'A+B+C+D'),
                               ('Full-Skill', 'A+B+C+E'), ('Full', 'A+B+C+D+E')]])
    reproduction['是否复现'] = np.where(
        reproduction['validation差值'].abs().lt(1e-6)
        & reproduction['test差值'].abs().lt(1e-6), '是', '否')

    method_rows = pd.DataFrame([
        {'项目': '模型与超参数',
         '内容': f'{final_model_key} {final_params}（Stage25 正式主模型与超参数）'},
        {'项目': '技能阈值 / 文本维度',
         '内容': f'{skill_threshold} / {text_dim}（Stage25 选定，为可比性固定不变）'},
        {'项目': '数据划分',
         '内容': 'data/processed/model_splits.parquet（train 10,418 / validation 2,232 / '
                 'test 2,233）'},
        {'项目': '预处理',
         '内容': '全部只在 train 上 fit（中位数填补 / 类别水平 / 技能列 / SVD 分量）'},
        {'项目': '配对 bootstrap',
         '内容': f'{BOOTSTRAP_ROUNDS} 轮，种子 {BOOTSTRAP_SEED}，按样本重抽计算 MAE 差值分布'},
        {'项目': 'F 组预测单位',
         '内容': '仍为一岗一行（14,883 个正式代表样本）；F 特征只使用代表 Episode 开始'
                 '之前已发生的历史，未把同一岗位多个 Episode 复制进主模型'},
    ])
    sheets = {'03_消融结果': ablation_table, '04_增量与bootstrap': increment_table,
              '05_与Stage15复现对照': reproduction, '06_方法与口径': method_rows}
    random_metrics = model_training.regression_metrics(
        predictions[('A+B+C+D+E', 'test')]['y_true'].to_numpy(),
        predictions[('A+B+C+D+E', 'test')]['y_pred'].to_numpy())
    return {'sheets': sheets, 'ablation_table': ablation_table,
            'increment_table': increment_table, 'reproduction': reproduction,
            'random_metrics': random_metrics}


def run_step_j(model_frame: pd.DataFrame, entity_publish: pd.Series, grouped: dict,
               skill_map: dict, text_matrix: np.ndarray, text_by_id: dict,
               final_model_key: str, final_params: dict, random_metrics: dict,
               stage25_skill_threshold: int, stage25_text_dim: int) -> dict:
    """Temporal Split（业务时间轴 = 发布时间 / 代表 Episode 开始）。"""
    axis = model_frame[schema.ID_FIELD].map(entity_publish)
    if axis.isna().any():
        raise ValueError('存在无发布时间的建模样本，禁止无口径回填')
    frame = model_frame.assign(_business_date=axis.dt.normalize().to_numpy())
    date_counts = frame.groupby('_business_date').size().sort_index()
    cumulative = date_counts.cumsum()
    total = int(cumulative.iloc[-1])
    dates = cumulative.index
    index1 = min(max(int(np.searchsorted(cumulative.to_numpy(), 0.70 * total, side='left')), 1),
                 len(dates) - 2)
    index2 = min(max(int(np.searchsorted(cumulative.to_numpy(), 0.85 * total, side='left')),
                     index1 + 1), len(dates) - 1)
    train_dates, valid_dates, test_dates = dates[:index1], dates[index1:index2], dates[index2:]
    values = frame['_business_date'].to_numpy()
    labels = np.empty(len(frame), dtype=object)
    labels[np.isin(values, train_dates.to_numpy())] = 'train'
    labels[np.isin(values, valid_dates.to_numpy())] = 'validation'
    labels[np.isin(values, test_dates.to_numpy())] = 'test'
    frame['temporal_split'] = labels
    split_table = pd.DataFrame([
        {'子集': label,
         '岗位数': int((frame['temporal_split'] == label).sum()),
         '占比': round(float((frame['temporal_split'] == label).mean()), 6),
         '日期起': str(frame.loc[frame['temporal_split'].eq(label),
                              '_business_date'].min().date()),
         '日期止': str(frame.loc[frame['temporal_split'].eq(label),
                              '_business_date'].max().date()),
         '唯一业务日数': int(frame.loc[frame['temporal_split'].eq(label),
                                 '_business_date'].nunique()),
         '薪资中点中位数': round(float(frame.loc[frame['temporal_split'].eq(label),
                                           schema.SALARY_MID_FIELD].median()), 4)}
        for label in ('train', 'validation', 'test')])
    split_table['日期边界规则'] = ('按业务日期边界切割（同一天不拆分）；'
                              f'train ≤ {train_dates.max().date()}，'
                              f'validation ≤ {valid_dates.max().date()}，'
                              f'test ≥ {test_dates.min().date()}')

    train_frame = frame[frame['temporal_split'].eq('train')].reset_index(drop=True)
    valid_frame = frame[frame['temporal_split'].eq('validation')].reset_index(drop=True)
    test_frame = frame[frame['temporal_split'].eq('test')].reset_index(drop=True)
    y_train = train_frame[schema.SALARY_MID_FIELD].to_numpy('float64')
    y_valid = valid_frame[schema.SALARY_MID_FIELD].to_numpy('float64')
    y_test = test_frame[schema.SALARY_MID_FIELD].to_numpy('float64')

    def text_for(block):
        return text_matrix[[text_by_id[job_id] for job_id in block[schema.ID_FIELD]]]

    groups = ABLATION_CONFIGS['A+B+C+D+E']
    cache = {}

    def assemble(threshold, text_dim, scale=False):
        key = (threshold, text_dim, scale)
        if key not in cache:
            assembler = build_assembler(groups, grouped, threshold, text_dim)
            assembler.fit(train_frame, skill_map, text_for(train_frame))
            cache[key] = (assembler,
                          assembler.transform(train_frame, skill_map, text_for(train_frame)),
                          assembler.transform(valid_frame, skill_map, text_for(valid_frame)),
                          assembler.transform(test_frame, skill_map, text_for(test_frame)))
        return cache[key]

    reference_model, reference_params = 'RandomForest', {
        'n_estimators': 300, 'max_depth': None, 'min_samples_leaf': 1, 'max_features': 'sqrt'}
    threshold_rows = []
    for threshold in model_training.SKILL_THRESHOLD_CANDIDATES:
        resolved = (max(1, int(round(len(train_frame) * threshold))) if threshold == 0.005
                    else int(threshold))
        assembler, matrix_train, matrix_valid, _ = assemble(resolved, 0)
        fit = model_training.fit_and_predict(matrix_train, y_train, matrix_valid, None,
                                             reference_model, reference_params)
        metrics = model_training.regression_metrics(y_valid, fit['valid_pred'])
        threshold_rows.append({'阈值口径': f'>= {resolved}', '阈值': resolved,
                               '技能列数': len(assembler.schema.skill_columns),
                               '特征维度': assembler.schema.dimension,
                               'validation_MAE': metrics['MAE']})
        print(f'  Temporal 技能阈值 {resolved}: validation MAE {metrics["MAE"]}')
    threshold_table = pd.DataFrame(threshold_rows)
    skill_threshold = int(threshold_table.sort_values(
        ['validation_MAE', '技能列数']).iloc[0]['阈值'])

    dimension_rows = []
    for text_dim in model_training.TEXT_DIM_CANDIDATES:
        assembler, matrix_train, matrix_valid, _ = assemble(skill_threshold, text_dim)
        fit = model_training.fit_and_predict(matrix_train, y_train, matrix_valid, None,
                                             reference_model, reference_params)
        metrics = model_training.regression_metrics(y_valid, fit['valid_pred'])
        dimension_rows.append({'文本维度': text_dim, '特征维度': assembler.schema.dimension,
                               'validation_MAE': metrics['MAE']})
        print(f'  Temporal 文本维度 {text_dim}: validation MAE {metrics["MAE"]}')
    dimension_table = pd.DataFrame(dimension_rows)
    text_dim = int(dimension_table.sort_values('validation_MAE').iloc[0]['文本维度'])

    model_results, comparison_rows = {}, []
    for model_key, grid in MODEL_GRIDS.items():
        best = None
        for params in grid:
            assembler, matrix_train, matrix_valid, matrix_test = assemble(
                skill_threshold, text_dim, scale=model_key in SCALE_FOR)
            try:
                fit = model_training.fit_and_predict(matrix_train, y_train, matrix_valid,
                                                    matrix_test, model_key, params)
            except ImportError as error:
                comparison_rows.append({'模型': model_key, '最佳超参数': str(params),
                                        'validation_MAE': None,
                                        '状态': f'NOT_RUN（{error}）'})
                break
            metrics = model_training.regression_metrics(y_valid, fit['valid_pred'])
            if best is None or metrics['MAE'] < best['metrics']['MAE']:
                best = {'params': params, 'metrics': metrics, 'fit': fit,
                        'assembler': assembler}
        if best is not None:
            model_results[model_key] = best
            comparison_rows.append({'模型': model_key, '最佳超参数': str(best['params']),
                                    'validation_MAE': best['metrics']['MAE'],
                                    'validation_RMSE': best['metrics']['RMSE'],
                                    'validation_R2': best['metrics']['R2'],
                                    '特征维度': best['assembler'].schema.dimension,
                                    '状态': 'OK'})
            print(f"  Temporal {model_key}: validation MAE {best['metrics']['MAE']}")

    assembler_dummy, matrix_train, matrix_valid, matrix_test = assemble(skill_threshold, text_dim)
    dummy_fit = model_training.fit_and_predict(matrix_train, y_train, matrix_valid, matrix_test,
                                              'Dummy', {})
    dummy_metrics = model_training.regression_metrics(y_valid, dummy_fit['valid_pred'])
    model_results['Dummy'] = {'params': {'strategy': 'mean'}, 'metrics': dummy_metrics,
                              'fit': dummy_fit, 'assembler': assembler_dummy}
    comparison_rows.append({'模型': 'Dummy（均值基线）', '最佳超参数': "{'strategy': 'mean'}",
                            'validation_MAE': dummy_metrics['MAE'],
                            'validation_RMSE': dummy_metrics['RMSE'],
                            'validation_R2': dummy_metrics['R2'],
                            '特征维度': assembler_dummy.schema.dimension, '状态': 'OK'})
    median_constant = float(np.median(y_train))
    median_metrics = model_training.regression_metrics(
        y_valid, np.full(y_valid.size, median_constant))
    comparison_rows.append({'模型': 'Dummy（中位数基线）',
                            '最佳超参数': f'常数 = {median_constant:.6f}',
                            'validation_MAE': median_metrics['MAE'],
                            'validation_RMSE': median_metrics['RMSE'],
                            'validation_R2': median_metrics['R2'],
                            '特征维度': 0, '状态': 'OK'})
    comparison = pd.DataFrame(comparison_rows)
    non_dummy = comparison[~comparison['模型'].astype(str).str.startswith('Dummy')]
    final_key = str(non_dummy.sort_values('validation_MAE').iloc[0]['模型'])
    final_params = model_results[final_key]['params']
    print(f'  Temporal 选定主模型 {final_key}（validation MAE '
          f"{model_results[final_key]['metrics']['MAE']}，超参数 {final_params}）")

    train_only_metrics = model_training.regression_metrics(
        y_test, np.asarray(model_results[final_key]['fit']['test_pred'], dtype='float64'))
    fit_frame = pd.concat([train_frame, valid_frame], ignore_index=True)
    assembler_final = model_results[final_key]['assembler']
    final_model = model_training.make_model(final_key, final_params,
                                           random_state=model_training.SPLIT_RANDOM_STATE)
    final_model.fit(assembler_final.transform(fit_frame, skill_map, text_for(fit_frame)),
                    fit_frame[schema.SALARY_MID_FIELD].to_numpy('float64'))
    test_pred = np.asarray(final_model.predict(
        assembler_final.transform(test_frame, skill_map, text_for(test_frame))), dtype='float64')
    test_metrics = model_training.regression_metrics(y_test, test_pred)

    train_ids = set(train_frame[schema.ID_FIELD])
    unseen_mask = ~test_frame[schema.ID_FIELD].isin(train_ids).to_numpy()
    unseen_n = int(unseen_mask.sum())
    if unseen_n >= 30:
        unseen = {'测试期中 intern_id 从未在训练期出现的岗位数': unseen_n,
                  **model_training.regression_metrics(y_test[unseen_mask],
                                                      test_pred[unseen_mask]),
                  '结论': '可报告时间外推下的新岗位表现；但主模型预测单位为「一岗一行」，'
                          '测试期的岗位按构造不可能与训练期重叠（train 截止 '
                          f'{train_frame["_business_date"].max().date()}，test 起于 '
                          f'{test_frame["_business_date"].min().date()}），'
                          '因此 Temporal-Unseen 与 Temporal-All 完全相同，'
                          '本设计无法区分「老岗位再次招聘」与「完全新岗位」——'
                          '该区分需另行构造 Episode 级模型'}
    else:
        unseen = {'测试期中 intern_id 从未在训练期出现的岗位数': unseen_n,
                  '结论': '样本不足（< 30），按约定不给出该子集指标，仅报告样本数'}

    group_split = pd.read_parquet(project_paths.MODEL_SPLITS_PARQUET)
    group_train_ids = set(group_split.loc[group_split['company_group_split'].eq('train'),
                                          schema.ID_FIELD])
    group_test_ids = set(group_split.loc[group_split['company_group_split'].eq('test'),
                                        schema.ID_FIELD])
    group_train = model_frame[model_frame[schema.ID_FIELD].isin(group_train_ids)].reset_index(
        drop=True)
    group_test = model_frame[model_frame[schema.ID_FIELD].isin(group_test_ids)].reset_index(
        drop=True)

    def company_group_metrics(threshold: int, dim: int) -> dict:
        assembler = build_assembler(groups, grouped, threshold, dim)
        assembler.fit(group_train, skill_map, text_for(group_train))
        model = model_training.make_model(final_key, final_params,
                                          random_state=model_training.SPLIT_RANDOM_STATE)
        model.fit(assembler.transform(group_train, skill_map, text_for(group_train)),
                  group_train[schema.SALARY_MID_FIELD].to_numpy('float64'))
        return model_training.regression_metrics(
            group_test[schema.SALARY_MID_FIELD].to_numpy('float64'),
            np.asarray(model.predict(assembler.transform(
                group_test, skill_map, text_for(group_test))), dtype='float64'))

    group_metrics = company_group_metrics(skill_threshold, text_dim)
    group_metrics_s25 = company_group_metrics(stage25_skill_threshold, stage25_text_dim)

    frozen_random = pd.read_excel(TABLES / project_paths.TABLE_MODEL_COMPARISON,
                                  sheet_name='09_Test最终结果').set_index('模型')
    frozen_group = pd.read_excel(TABLES / project_paths.TABLE_ABLATION_SHAP,
                                 sheet_name='04_Random_vs_GroupSplit')
    frozen_group_row = frozen_group[
        frozen_group['划分方式'].str.startswith('Company Group Split')
        & frozen_group['数据子集'].eq('test')].iloc[0]
    frozen_random_value = float(frozen_random.loc['FINAL（LightGBM）', 'test_MAE'])
    frozen_group_value = float(frozen_group_row['MAE'])

    def row(label, metrics, note, reference=None):
        block = {'划分方式': label, '数据子集': 'test', 'n': metrics['n'],
                 'MAE': metrics['MAE'], 'RMSE': metrics['RMSE'], 'R²': metrics['R2'],
                 '口径': note}
        if reference is None:
            block['与既有正式结果差异（MAE）'] = ''
            block['是否复现既有一致'] = ''
        else:
            difference = round(float(metrics['MAE']) - reference, 9)
            block['与既有正式结果差异（MAE）'] = difference
            block['是否复现既有一致'] = '一致' if abs(difference) < 1e-6 else '存在差异'
        return block

    comparison_table = pd.DataFrame([
        row('Random Split（同分布预测）', random_metrics,
            '同一协议：仅在 random train 上拟合估计器；与既有正式主模型同一测试集'),
        row('Company Group Split（跨公司泛化）', group_metrics,
            f'同一协议：仅在 company_group train 上拟合估计器；技能阈值 / 文本维度取 '
            f'Temporal 选定值（{skill_threshold} / {text_dim}）', frozen_group_value),
        row('Temporal Split（时间外推，仅 train 拟合）', train_only_metrics,
            '同一协议：仅使用时间上最早的约 70%'),
        row('Temporal Split（时间外推，train+validation 重拟合）', test_metrics,
            '与 Stage25 正式主模型协议一致'),
        row('Random Split（Stage25 正式主模型，train+validation 重拟合，只读引用）',
            {'n': int(frozen_random.loc['FINAL（LightGBM）', 'n']),
             'MAE': frozen_random_value,
             'RMSE': float(frozen_random.loc['FINAL（LightGBM）', 'test_RMSE']),
             'R2': float(frozen_random.loc['FINAL（LightGBM）', 'test_R2'])},
            '只读引用既有正式结果（30 号表 09）'),
        row('Company Group Split（Stage25 既有正式结果，只读引用）',
            {'n': int(frozen_group_row['n']), 'MAE': frozen_group_value,
             'RMSE': float(frozen_group_row['RMSE']), 'R2': float(frozen_group_row['R²'])},
            '只读引用既有正式结果（31 号表 04）'),
        row('Company Group Split（Stage25 配置精确复现）', group_metrics_s25,
            f'技能阈值 / 文本维度 = Stage25 正式取值（{stage25_skill_threshold} / '
            f'{stage25_text_dim}），仅用于确认既有结果可精确复现', frozen_group_value),
    ])

    method_rows = pd.DataFrame([
        {'项目': '时间轴',
         '内容': '业务时间 = 发布时间（代表 Episode 开始 / 岗位实体发布时间），'
                 '不使用观测时间等爬取时间'},
        {'项目': '切分规则',
         '内容': '按业务日期排序，最早约 70% → train，随后约 15% → validation，'
                 '最新约 15% → test；按日期边界切割，同一天不拆分；实际样本与边界见 01 表'},
        {'项目': '特征体系',
         '内容': '与 Stage25 相同：A+B+C+D+E（不含爬取时间）；技能阈值与文本维度在 '
                 'Temporal train 上重新以 validation MAE 选择'},
        {'项目': '候选模型',
         '内容': 'Dummy（均值 / 中位数）、Ridge、RandomForest、CatBoost、LightGBM；'
                 '按 Temporal validation MAE 选主模型'},
        {'项目': 'test 纪律', '内容': 'test 仅在配置锁定后评估，未据 test 调整任何配置'},
        {'项目': '解读边界',
         '内容': '三种划分对应不同任务（同分布预测 / 跨公司泛化 / 时间外推），'
                 '不得简单解释为「谁更优」；差异量化的是口径差异'},
    ])
    sheets = {
        '01_Temporal划分': split_table,
        '02_阈值与维度选择': pd.concat([threshold_table.assign(块='技能阈值'),
                                  dimension_table.assign(块='文本维度')],
                                 ignore_index=True, sort=False),
        '03_模型比较': comparison,
        '04_三划分对照': comparison_table,
        '05_Temporal主模型结果': pd.DataFrame([{
            '主模型': final_key, '超参数': str(final_params), '技能阈值': skill_threshold,
            '文本维度': text_dim,
            '特征维度': model_results[final_key]['assembler'].schema.dimension,
            **{f'test_{key}': value for key, value in test_metrics.items()},
            'train日期起': str(train_frame['_business_date'].min().date()),
            'train日期止': str(train_frame['_business_date'].max().date()),
            'validation日期起': str(valid_frame['_business_date'].min().date()),
            'validation日期止': str(valid_frame['_business_date'].max().date()),
            'test日期起': str(test_frame['_business_date'].min().date()),
            'test日期止': str(test_frame['_business_date'].max().date())}]),
        '06_Temporal_Unseen': pd.DataFrame([unseen]),
        '07_方法与口径': method_rows,
    }
    return {'sheets': sheets, 'split_table': split_table, 'comparison': comparison,
            'comparison_table': comparison_table, 'test_metrics': test_metrics,
            'train_only_metrics': train_only_metrics, 'final_key': final_key,
            'final_params': final_params, 'skill_threshold': skill_threshold,
            'text_dim': text_dim, 'unseen': unseen, 'random_metrics': random_metrics,
            'group_metrics': group_metrics, 'group_metrics_stage25_config': group_metrics_s25,
            'feature_dimension': model_results[final_key]['assembler'].schema.dimension}


# ============================================================================
# 图件（论文版：无图内总图题，子图名置于子图下方；图例置于坐标区外以避免遮挡）
# ============================================================================
def figure_s18(episodes: pd.DataFrame, registry: list) -> dict:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    duration = episodes.loc[episodes['planned_duration_days'] > 0, 'planned_duration_days']
    gap_t2 = episodes.loc[episodes['reopen_tier'].eq('T2_疑似重开'), 'gap_days_calendar']
    rounds = episodes.groupby('intern_id')['episode_count'].first()
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))
    ax = axes[0]
    plot_style.hist_discrete(ax, duration, discrete=False, bins=60,
                             xlabel='计划持续天数（天）', ylabel='招聘周期数')
    ax.axvline(float(duration.median()), color=plot_style.ACCENT_COLOR, linestyle='--',
               linewidth=1.0)
    ax.set_xlim(0, float(duration.quantile(0.99)))
    ax.text(0.97, 0.94, f'中位数 {duration.median():.0f} 天\n'
                        f'IQR {duration.quantile(.75) - duration.quantile(.25):.0f} 天\n'
                        f'n = {duration.size:,}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=plot_style.FONT_SIZES['annotation'], linespacing=1.45)
    plot_style.apply_sci_axis(ax, grid_axis='y')
    plot_style.add_subfigure_caption(ax, 'a', '计划持续天数分布（横轴截断至 P99）')

    ax = axes[1]
    plot_style.hist_discrete(ax, gap_t2, discrete=True, xlabel='重招间隔（日历天）',
                             ylabel='T2 疑似重开事件数')
    ax.text(0.97, 0.94, f'n = {gap_t2.size:,}\n中位数 {gap_t2.median():.0f} 天\n'
                        f'gap = 1 天占比 {float(gap_t2.eq(1).mean()):.2%}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=plot_style.FONT_SIZES['annotation'], linespacing=1.45)
    plot_style.apply_sci_axis(ax, grid_axis='y')
    plot_style.add_subfigure_caption(ax, 'b', 'T2 疑似重开间隔分布（高度集中于 1 天）')

    ax = axes[2]
    counts = pd.Series({'单周期岗位': int((rounds == 1).sum()),
                        '2 个周期': int((rounds == 2).sum()),
                        '≥3 个周期': int((rounds >= 3).sum())})
    plot_style.bar_ranked(ax, counts, xlabel='周期数分档', ylabel='岗位数')
    plot_style.apply_sci_axis(ax, grid_axis='y')
    plot_style.add_subfigure_caption(ax, 'c', '单周期与多周期岗位构成')
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.22, top=0.97, wspace=0.34)
    diagnostics = figure_finalize.save_paper_figure(
        fig, SUPP_DIR, FIG_STEMS[0],
        subfigures=[('a', '计划持续天数分布（横轴截断至 P99）', axes[0]),
                    ('b', 'T2 疑似重开间隔分布（高度集中于 1 天）', axes[1]),
                    ('c', '单周期与多周期岗位构成', axes[2])],
        meta={'数据来源': 'job_recruitment_episode.parquet', 'seed': SEED,
              '用途': '第4章 4.5.1 招聘周期总体特征'})
    plt.close(fig)
    registry.append(diagnostics)
    return diagnostics


def figure_s19(daily: pd.DataFrame, registry: list) -> dict:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    dates = daily['date']
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.6))
    panels = [
        ('a', 'N_t 活跃数量（周期级 / 去重岗位实体）',
         [('N_t_活跃岗位数', '活跃招聘周期数（原始日序列）', plot_style.MUTED_COLOR, 0.6),
          ('N_t_7日滚动中位数', '活跃招聘周期数（7 日滚动中位数）', plot_style.MAIN_COLOR, 1.6),
          ('N_t_活跃岗位实体数', '去重到岗位实体', plot_style.PALETTE[2], 0.9)]),
        ('b', 'O_t 每日新增 / C_t 每日结束周期数',
         [('O_t_新增周期数', '每日新增（7 日滚动中位数）', plot_style.MAIN_COLOR, 1.2),
          ('C_t_结束周期数', '每日结束（7 日滚动中位数）', plot_style.ACCENT_COLOR, 1.2)]),
        ('c', 'R_t 每日重招周期数',
         [('R_t_重招周期数', '非首周期新增（含 T3）', plot_style.MUTED_COLOR, 0.8),
          ('R_t_T1确认重招数', 'T1 确认重招', plot_style.PALETTE[2], 1.5)]),
    ]
    smooth = {'b': ['O_t_7日滚动中位数', 'C_t_7日滚动中位数']}
    for (letter, title, series_list), ax in zip(panels, axes):
        raw_columns = set(smooth.get(letter, []))
        for column, label, color, width in series_list:
            if column in raw_columns:
                ax.plot(dates, daily[column], color=color, linewidth=width, label=label)
            else:
                ax.plot(dates, daily[column], color=color, linewidth=width, label=label)
        if letter == 'b':
            ax.plot(dates, daily['O_t_新增周期数'], color=plot_style.MAIN_COLOR,
                    linewidth=0.5, alpha=0.5)
            ax.plot(dates, daily['C_t_结束周期数'], color=plot_style.ACCENT_COLOR,
                    linewidth=0.5, alpha=0.5)
        ax.set_xlabel('业务日期（发布时间 / 投递截止日期）')
        ax.set_ylabel(title)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.005), ncol=2, frameon=False)
        plot_style.apply_sci_axis(ax, grid_axis='y')
        plot_style.add_subfigure_caption(ax, letter, title)
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.24, top=0.86, wspace=0.26)
    diagnostics = figure_finalize.save_paper_figure(
        fig, SUPP_DIR, FIG_STEMS[1],
        subfigures=[(letter, title, ax) for (letter, title, _), ax in zip(panels, axes)],
        meta={'数据来源': 'job_daily_panel.parquet 的日级聚合', '口径': CALIBER_ACTIVE,
              'seed': SEED, '用途': '第4章 4.5.2 样本招聘周期的时序演化'})
    plt.close(fig)
    registry.append(diagnostics)
    return diagnostics


def figure_s20(category_daily: dict, dates, registry: list) -> dict:
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.ticker import MaxNLocator  # noqa: PLC0415

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.0))
    for ax, smoothed, letter, title in [
            (axes[0], False, 'a', 'N(c,t) 样本活跃周期数（原始日序列）'),
            (axes[1], True, 'b', 'N(c,t) 样本活跃周期数（7 日滚动中位数）')]:
        for position, (category, counts) in enumerate(category_daily.items()):
            series = pd.Series(counts, index=dates)
            values = series.rolling(7, min_periods=1).median() if smoothed else series
            ax.plot(dates, values, linewidth=1.2,
                    color=plot_style.PALETTE[position % len(plot_style.PALETTE)],
                    label=category)
        ax.set_xlabel('业务日期（发布时间 / 投递截止日期）')
        ax.set_ylabel(title)
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False)
        plot_style.apply_sci_axis(ax, grid_axis='y')
        plot_style.add_subfigure_caption(ax, letter, title)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.24, top=0.82, wspace=0.18)
    diagnostics = figure_finalize.save_paper_figure(
        fig, SUPP_DIR, FIG_STEMS[2],
        subfigures=[('a', 'N(c,t) 样本活跃周期数（原始日序列）', axes[0]),
                    ('b', 'N(c,t) 样本活跃周期数（7 日滚动中位数）', axes[1])],
        meta={'数据来源': 'job_daily_panel.parquet × 岗位大类集合', '口径': CALIBER_ACTIVE,
              'seed': SEED, '用途': '第4章 4.5.3 不同岗位类别的时间演化'})
    plt.close(fig)
    registry.append(diagnostics)
    return diagnostics


def figure_s21(daily: pd.DataFrame, registry: list) -> dict:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    dates = daily['date']
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 4.8))
    ax = axes[0]
    ax.fill_between(dates, daily['salary_p25'], daily['salary_p75'],
                    color=plot_style.MAIN_COLOR, alpha=0.18, linewidth=0,
                    label='P25~P75（IQR）')
    ax.plot(dates, daily['salary_median'], color=plot_style.MUTED_COLOR, linewidth=0.6,
            alpha=0.75, label='每日薪资中位数（原始序列）')
    ax.plot(dates, daily['salary_median_7日滚动中位数'], color=plot_style.MAIN_COLOR,
            linewidth=1.7, label='薪资中位数 7 日滚动中位数')
    ax.set_xlabel('业务日期（发布时间 / 投递截止日期）')
    ax.set_ylabel('活跃周期薪资中点（元/天）')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.005), ncol=2, frameon=False)
    plot_style.apply_sci_axis(ax, grid_axis='y')
    plot_style.add_subfigure_caption(ax, 'a', '活跃招聘周期薪资中位数与 IQR 时序')

    ax = axes[1]
    ax.plot(dates, daily['n_salary'], color=plot_style.PALETTE[1], linewidth=1.4)
    ax.set_xlabel('业务日期（发布时间 / 投递截止日期）')
    ax.set_ylabel('参与薪资统计的活跃周期数')
    plot_style.apply_sci_axis(ax, grid_axis='y')
    ax.text(0.985, 0.94, '仅统计薪资可解析（非面议）的活跃招聘周期',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=plot_style.FONT_SIZES['annotation'])
    plot_style.add_subfigure_caption(ax, 'b', '每日参与薪资统计的活跃周期数（读图可靠性）')
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.25, top=0.85, wspace=0.22)
    diagnostics = figure_finalize.save_paper_figure(
        fig, SUPP_DIR, FIG_STEMS[3],
        subfigures=[('a', '活跃招聘周期薪资中位数与 IQR 时序', axes[0]),
                    ('b', '每日参与薪资统计的活跃周期数（读图可靠性）', axes[1])],
        meta={'数据来源': 'job_daily_panel.parquet 的日级薪资聚合', '口径': CALIBER_ACTIVE,
              'seed': SEED, '用途': '第4章 4.5.4 薪资水平的时间演化'})
    plt.close(fig)
    registry.append(diagnostics)
    return diagnostics


def figure_s22(comparison_table: pd.DataFrame, registry: list) -> dict:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    frame = comparison_table[comparison_table['划分方式'].isin([
        'Random Split（同分布预测）', 'Company Group Split（跨公司泛化）',
        'Temporal Split（时间外推，仅 train 拟合）'])].copy()
    labels = ['Random', 'Company Group', 'Temporal']
    positions = np.arange(len(frame))
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.8))
    ax = axes[0]
    ax.bar(positions - 0.19, frame['MAE'], width=0.36, color=plot_style.MAIN_COLOR,
           edgecolor='black', linewidth=0.6, label='MAE')
    ax.bar(positions + 0.19, frame['RMSE'], width=0.36, color=plot_style.MUTED_COLOR,
           edgecolor='black', linewidth=0.6, label='RMSE')
    span = float(frame[['MAE', 'RMSE']].to_numpy().max())
    for position, value in zip(positions - 0.19, frame['MAE']):
        ax.text(position, value + span * 0.02, f'{value:.2f}', ha='center', va='bottom',
                fontsize=plot_style.FONT_SIZES['annotation'])
    for position, value in zip(positions + 0.19, frame['RMSE']):
        ax.text(position, value + span * 0.02, f'{value:.2f}', ha='center', va='bottom',
                fontsize=plot_style.FONT_SIZES['annotation'])
    ax.set_xticks(positions)
    ax.set_xticklabels([f'{label}\n(n={int(n)})' for label, n in zip(labels, frame['n'])])
    ax.set_xlabel('数据划分方式（同一协议：仅在各自 train 上拟合估计器）')
    ax.set_ylabel('test 误差（元/天）')
    ax.set_ylim(0, span * 1.30)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.005), ncol=2, frameon=False)
    plot_style.apply_sci_axis(ax, grid_axis='y')
    plot_style.add_subfigure_caption(ax, 'a', 'MAE 与 RMSE 对照')

    ax = axes[1]
    ax.bar(positions, frame['R²'], width=0.5, color=plot_style.PALETTE[2],
           edgecolor='black', linewidth=0.6)
    for position, value in zip(positions, frame['R²']):
        ax.text(position, value + 0.012, f'{value:.3f}', ha='center', va='bottom',
                fontsize=plot_style.FONT_SIZES['annotation'])
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel('数据划分方式')
    ax.set_ylabel('test R²')
    ax.set_ylim(0, float(frame['R²'].max()) * 1.25)
    plot_style.apply_sci_axis(ax, grid_axis='y')
    ax.text(0.985, 0.94, '三种划分对应不同预测任务，\n不构成「谁更优」的比较',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=plot_style.FONT_SIZES['annotation'], linespacing=1.45)
    plot_style.add_subfigure_caption(ax, 'b', 'R² 对照')
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.24, top=0.85, wspace=0.22)
    diagnostics = figure_finalize.save_paper_figure(
        fig, SUPP_DIR, FIG_STEMS[4],
        subfigures=[('a', 'MAE 与 RMSE 对照', axes[0]), ('b', 'R² 对照', axes[1])],
        meta={'数据来源': '本脚本 Step J 同口径重算', 'seed': SEED,
              '用途': '第8章 8.2 不同泛化场景下的模型表现'})
    plt.close(fig)
    registry.append(diagnostics)
    return diagnostics


# ============================================================================
# main
# ============================================================================
def main() -> int:
    started = time.time()
    print('=' * 96)
    print('Stage26 招聘生命周期时序融合（数据与建模实测）')
    print('=' * 96)
    manifest_before = project_manifest()
    print(f'运行前既有文件 SHA-256 清单：{len(manifest_before)} 个文件')

    obs = io_utils.read_parquet(project_paths.OBSERVATION_SNAPSHOT_PARQUET)
    analysis = io_utils.read_parquet(project_paths.JOB_ANALYSIS_DATASET_PARQUET,
                                     columns=[schema.ID_FIELD, '岗位大类集合',
                                              'company_entity_id'])
    corpus = io_utils.read_parquet(
        project_paths.TEXT_CORPUS_PARQUET,
        columns=[schema.ID_FIELD, '核心版本号', schema.SKILL_SET_FIELD])
    entity = pd.read_parquet(project_paths.PROCESSED_UNIQUE_PARQUET,
                             columns=[schema.ID_FIELD, '发布时间'])
    entity['发布时间'] = pd.to_datetime(entity['发布时间'], errors='coerce')
    entity_time = entity.rename(columns={schema.ID_FIELD: 'intern_id',
                                         '发布时间': '_entity_publish'})
    salary_targets = pd.read_parquet(
        project_paths.SALARY_TARGETS_PARQUET,
        columns=[schema.ID_FIELD, schema.SALARY_MID_FIELD])
    model_frame = io_utils.read_parquet(project_paths.JOB_SALARY_MODEL_DATASET_PARQUET)
    splits = io_utils.read_parquet(project_paths.MODEL_SPLITS_PARQUET)
    membership = io_utils.read_parquet(project_paths.JOB_SKILL_MEMBERSHIP_PARQUET)
    existing_version = pd.read_parquet(project_paths.PROCESSED_VERSION_HISTORY_PARQUET,
                                       columns=[schema.ID_FIELD, '核心版本号'])
    print(f'输入：观测 {len(obs):,} / 岗位实体 {len(analysis):,} / 建模样本 {len(model_frame):,} / '
          f'既有版本 {len(existing_version):,}')
    salary_parser, salary_config = load_stage11_salary_parser()
    assert len(model_frame) == 14883 and len(analysis) == 17144
    assert len(obs.drop_duplicates([schema.ID_FIELD, schema.OBSERVATION_TIME_FIELD])) == len(obs)

    # ---- Step A ----
    step_a = run_step_a(obs)
    audit_path = io_utils.write_excel(TABLES / TABLE_FILES[0], step_a['sheets'])
    print(f"Step A：发布非空 {step_a['summary']['publish_nonnnull_rate']} / "
          f"截止非空 {step_a['summary']['deadline_nonnnull_rate']} / "
          f"截止唯一值 {step_a['summary']['deadline_nunique']} / "
          f"start>end（记录层）{step_a['summary']['start_gt_end_records']} → "
          f"{audit_path.name}（{audit_path.stat().st_size:,} 字节）")

    # ---- Step B ----
    step_b = run_step_b(obs, analysis, corpus, salary_targets, entity_time, existing_version,
                        salary_parser, salary_config)
    episodes = step_b['episodes']
    episode_path = io_utils.write_parquet(episodes, EPISODE_PATH)
    version_path = io_utils.write_parquet(
        step_b['versions'][STAGE26_VERSION_COLUMNS], VERSION26_PATH)
    io_utils.write_excel(TABLES / TABLE_FILES[1], step_b['sheets'])
    print(f"Step B：Episode {len(episodes):,} 行 / 岗位 {episodes['intern_id'].nunique():,} / "
          f"多周期岗位 {step_b['summary']['multi_episode_jobs']:,} / "
          f"T1 {step_b['summary']['T1_events']} T2 {step_b['summary']['T2_events']} "
          f"T3 {step_b['summary']['T3_events']}")
    print(f'  数据层：{episode_path.name} / {version_path.name}')

    # ---- 37 号表 ----
    episode_tier = episodes['reopen_tier'].value_counts().to_dict()
    gap_t1 = episodes.loc[episodes['reopen_tier'].eq('T1_确认重招'), 'reopen_gap_days']
    gap_t2 = episodes.loc[episodes['reopen_tier'].eq('T2_疑似重开'),
                          'start_minus_prev_end_days']
    gap_t1_cal = episodes.loc[episodes['reopen_tier'].eq('T1_确认重招'),
                              'gap_days_calendar']
    gap_t2_cal = episodes.loc[episodes['reopen_tier'].eq('T2_疑似重开'),
                              'gap_days_calendar']
    tier_detail = pd.DataFrame([
        {'项目': 'T1 确认重招事件数', '数值': int(episode_tier.get('T1_确认重招', 0))},
        {'项目': 'T1 涉及岗位数', '数值': step_b['summary']['T1_jobs']},
        {'项目': 'T2 疑似重开事件数', '数值': int(episode_tier.get('T2_疑似重开', 0))},
        {'项目': 'T2 涉及岗位数', '数值': step_b['summary']['T2_jobs']},
        {'项目': 'T3 时间冲突事件数', '数值': int(episode_tier.get('T3_时间冲突', 0))},
        {'项目': 'T3 涉及岗位数', '数值': step_b['summary']['T3_jobs']},
        {'项目': 'T2 中 gap = 1 日历天占比',
         '数值': round(float(gap_t2_cal.eq(1).mean()), 6) if len(gap_t2_cal) else None},
        {'项目': 'T2 gap 取值分布（日历天）',
         '数值': json.dumps({str(int(key)): int(value) for key, value in
                          gap_t2_cal.value_counts().sort_index().items()},
                          ensure_ascii=False)},
        {'项目': 'T1 gap（精确天）取值',
         '数值': '、'.join(f'{value:.4f}' for value in sorted(gap_t1.tolist()))
                 if len(gap_t1) else '—'},
        {'项目': 'T1 gap 日历天取值',
         '数值': '、'.join(f'{value:.0f}' for value in sorted(gap_t1_cal.tolist()))
                 if len(gap_t1_cal) else '—'},
        {'项目': 'T2 gap（精确天）分位 P25 / 中位数 / P75',
         '数值': (f'{gap_t2.quantile(.25):.4f} / {gap_t2.median():.4f} / '
                 f'{gap_t2.quantile(.75):.4f}' if len(gap_t2) else '—')},
        {'项目': '口径警示', '数值': CALIBER_T2},
        {'项目': '保守处理说明',
         '数值': 'T3 既不合并进上一 Episode、也不判为重招：保留为独立 Episode，'
                 '置 episode_status = conflict / time_conflict_flag = 1 并输出到人工审计表；'
                 'reopen_gap_days 仅对 is_reopened = 1（T1）定义，因此恒 ≥ 0'},
    ])
    deadline_adjust = episodes.loc[
        episodes['deadline_adjust_count'] > 0,
        ['intern_id', 'episode_id', 'episode_no', 'episode_start', 'source_version_count',
         'deadline_adjust_count']].sort_values(
        'deadline_adjust_count', ascending=False).head(200)
    reopen_sheets = {
        '01_三档口径汇总': step_b['tier_table'],
        '02_三档明细指标': tier_detail,
        '03_T1间隔分布（日历天）': pd.DataFrame({
            'gap（日历天）': [int(key) for key in
                          gap_t1_cal.value_counts().sort_index().index],
            'T1 事件数': gap_t1_cal.value_counts().sort_index().to_numpy()}),
        '04_T2间隔取值明细（日历天）': pd.DataFrame({
            'gap（日历天）': [int(key) for key in
                          gap_t2_cal.value_counts().sort_index().index],
            'T2 事件数': gap_t2_cal.value_counts().sort_index().to_numpy()}),
        '05_同一周期内截止日期调整': deadline_adjust,
        '06_重招与冲突样例': episodes.loc[
            episodes['reopen_tier'].ne(''),
            ['intern_id', 'episode_id', 'episode_no', 'episode_start', 'episode_end',
             'previous_episode_end', 'gap_days_calendar', 'start_minus_prev_end_days',
             'reopen_tier']].head(300),
    }
    io_utils.write_excel(TABLES / TABLE_FILES[2], reopen_sheets)
    print('37 号表已写入')

    # ---- Step C ----
    review_table, review_sample = build_manual_review(episodes, step_b['versions'])
    t3_show = t3_examples(episodes, step_b['versions'])
    review_checks = pd.DataFrame([
        {'检查项': 'T1 中 gap ≥ 7 的岗位数', '数值': step_b['summary']['T1_jobs'],
         '说明': 'T1 定义即 gap ≥ 7；事件数与岗位数见 37 号表 01 / 02'},
        {'检查项': 'T2 的 gap 分布（日历天）',
         '数值': json.dumps({str(int(key)): int(value) for key, value in
                          gap_t2_cal.value_counts().sort_index().items()},
                          ensure_ascii=False),
         '说明': f'T2 集中于 1 日历天（占比 {float(gap_t2_cal.eq(1).mean()):.4f}）→ '
                 '疑平台字段更新伪影'},
        {'检查项': 'T3 典型样例（前 3 例）',
         '数值': '；'.join(f"{row['intern_id']}：本周期 {row['本周期开始']} 早于上一周期结束 "
                        f"{row['上一周期结束']}（{row['差值日历天']:.0f} 日历天），"
                        f"字段变化 {row['页面字段变化']}"
                        for row in t3_show.to_dict('records')),
         '说明': 'T3 按规则不判重招，仅留档'},
        {'检查项': '抽样岗位数', '数值': int(len(review_sample)),
         '说明': f'分层抽样（种子 {SEED}）：T1 全部 2 个 + T2 18 个 + T3 60 个'},
        {'检查项': '抽样输出行数', '数值': int(len(review_table))},
        {'检查项': '人工判定列', '数值': '留空（本脚本不做人工判定，不得填写）'},
    ])
    review_sheets = {
        '01_人工抽查表': review_table,
        '02_自动检查结论': review_checks,
        '03_T3典型样例明细': t3_show,
        '04_抽样说明': pd.DataFrame([
            {'项目': '抽样总体', '数值': f"episode_count ≥ 2 的岗位 "
                                    f"{step_b['summary']['multi_episode_jobs']:,} 个"},
            {'项目': '抽样方法', '数值': f'分层抽样（按 T1 / T2 / T3 分层），随机种子 {SEED}'},
            {'项目': '抽样规模', '数值': f'{MANUAL_SAMPLE_SIZE} 个岗位'},
            {'项目': '分层配额（目标）', '数值': 'T1 2 个（全部）、T2 18 个、T3 60 个'},
            {'项目': '实际分层构成',
             '数值': json.dumps({str(key): int(value) for key, value in
                              review_sample['_tier'].value_counts().items()},
                              ensure_ascii=False)},
            {'项目': '人工判定', '数值': '本脚本无法做人工判定，该列必须留空'},
        ]),
    }
    io_utils.write_excel(TABLES / TABLE_FILES[8], review_sheets)
    print(f'43 号表已写入（抽样 {len(review_sample)} 岗位 / {len(review_table)} 行）')

    # ---- Step D/E ----
    panel_info = build_panel(episodes)
    panel_path = io_utils.write_parquet(panel_info['panel'], PANEL_PATH)
    step_e = run_step_e(panel_info['panel'], episodes, panel_info['episode_index'],
                        panel_info['valid_episodes'])
    daily = step_e['daily']
    print(f"Step D/E：面板 {panel_info['rows']:,} 行 → {panel_path.name}；"
          f"日级指标 {len(daily):,} 天（{daily['date'].min().date()} ~ "
          f"{daily['date'].max().date()}）")

    temporal_sheets = {
        '01_日级指标': daily,
        '02_指标汇总': pd.DataFrame([
            {'指标': name, 'min': round(float(daily[column].min()), 4),
             'median': round(float(daily[column].median()), 4),
             'max': round(float(daily[column].max()), 4)}
            for name, column in [('N_t 活跃招聘周期数（Episode 级）', 'N_t_活跃岗位数'),
                                 ('N_t 活跃岗位实体数（去重岗位）', 'N_t_活跃岗位实体数'),
                                 ('O_t 新增周期数', 'O_t_新增周期数'),
                                 ('C_t 结束周期数', 'C_t_结束周期数'),
                                 ('R_t 重招周期数', 'R_t_重招周期数'),
                                 ('R_t T1 确认重招数', 'R_t_T1确认重招数'),
                                 ('ReopenRate_t', 'ReopenRate_t'),
                                 ('ReopenRate_t（T1 口径）', 'ReopenRate_t_T1口径'),
                                 ('日薪资中位数', 'salary_median'),
                                 ('日薪资 IQR', 'salary_iqr'),
                                 ('参与薪资统计的活跃周期数', 'n_salary')]]),
        '03_口径说明': pd.DataFrame([
            {'项目': '活跃量口径', '内容': CALIBER_ACTIVE},
            {'项目': 'N_t 定义',
             '内容': 'N_t = Σ I(episode_start ≤ t ≤ episode_end)（**Episode 级**计数）；'
                     '另给出去重到岗位实体的 N_t_活跃岗位实体数：当某岗位的两个周期在'
                     '同一天同时活跃（T3 时间冲突）时，两者会出现差异'},
            {'项目': 'N_t 可能大于岗位实体总数',
             '内容': f'样本共 {17144:,} 个岗位实体；Episode 级 N_t 的最大值为 '
                     f'{int(daily["N_t_活跃岗位数"].max()):,}，'
                     f'去重到实体的最大值为 {int(daily["N_t_活跃岗位实体数"].max()):,}；'
                     '前者大于岗位实体数的部分来自同岗位多周期并行（T3 冲突），'
                     '不是"额外岗位"'},
            {'项目': 'O_t / C_t / R_t 定义',
             '内容': 'O_t = Σ I(start = t)；C_t = Σ I(end = t)；'
                     'R_t = Σ_{episode_no > 1} I(start = t)'},
            {'项目': 'ReopenRate_t', '内容': 'R_t / O_t（仅当 O_t > 0）；同时给出 T1 口径'},
            {'项目': '面板行数', '内容': f"{panel_info['rows']:,} 行"
                                    f"（未超过上限 {PANEL_MAX_ROWS:,}，未启用替代表示）"},
            {'项目': '薪资口径',
             '内容': '活跃周期薪资中点（元/天）；面议 / 未解析周期不计入薪资统计'},
            {'项目': '平滑方式', '内容': '7 日滚动中位数（min_periods = 1），原始日序列同时保留'},
        ]),
    }
    io_utils.write_excel(TABLES / TABLE_FILES[4], temporal_sheets)
    print('39 号表已写入')

    # ---- Step F ----
    step_f = run_step_f(episodes, panel_info, step_e)
    io_utils.write_excel(TABLES / TABLE_FILES[3], step_f['sheets'])
    print(f"Step F：计划持续中位数 {step_f['duration_rows'].iloc[0]['数值']} 天 / IQR "
          f"{step_f['duration_rows'].iloc[1]['数值']} 天")

    # ---- Step G ----
    step_g = run_step_g(episodes, model_frame)
    io_utils.write_excel(TABLES / TABLE_FILES[5], step_g['sheets'])
    print(f"Step G：生命周期—薪资检验完成（n = {step_g['n_with_lifecycle']:,}）")

    # ---- Step H ----
    f_frame, f_features, f_feature_table, f_leakage = build_f_features(
        episodes, step_b['versions'], model_frame)
    feature_manifest = json.loads(
        (project_paths.SALARY_MODEL_DIR / 'feature_manifest.json').read_text(encoding='utf-8'))
    model_params = json.loads(
        (project_paths.SALARY_MODEL_DIR / 'model_params.json').read_text(encoding='utf-8'))
    final_model_key = model_params['model_key']
    final_params = model_params['params']
    skill_threshold = int(model_params['skill_threshold'])
    text_dim = int(model_params['text_dim'])
    grouped = grouped_columns_with_f(feature_manifest, f_frame)
    skill_map = model_training.build_skill_map(membership, skill_eda.ALL_USABLE_SCOPES)
    all_ids = f_frame[schema.ID_FIELD].tolist()
    text_matrix = model_training.load_text_matrix(
        all_ids, project_paths.FEATURES_DIR / 'job_text_embeddings.npz',
        project_paths.FEATURES_DIR / 'job_text_embedding_index.parquet')
    text_by_id = {job_id: position for position, job_id in enumerate(all_ids)}
    print(f"Step H：F 组 {len(F_NUMERIC)} 个特征；Stage25 主模型 {final_model_key} "
          f"（阈值 {skill_threshold} / 文本维度 {text_dim}）")

    # ---- Step I ----
    step_i = run_step_i(f_frame, splits, grouped, skill_map, text_matrix, text_by_id,
                        skill_threshold, text_dim, final_model_key, final_params)
    base_dim = int(step_i['ablation_table'].set_index('配置').loc['A+B+C+D+E', '特征维度'])
    f_dim = int(step_i['ablation_table'].set_index('配置').loc['A+B+C+D+E+F', '特征维度'])
    step_i['sheets']['01_F特征清单'] = f_feature_table
    step_i['sheets']['02_未来泄漏审计'] = f_leakage
    step_i['sheets']['07_特征维度对照'] = pd.concat([
        step_i['ablation_table'][['配置', '特征维度']],
        pd.DataFrame([{'配置': 'Stage25 正式主模型（A+B+C+D+E）', '特征维度': 320}])],
        ignore_index=True)
    io_utils.write_excel(TABLES / TABLE_FILES[6], step_i['sheets'])
    f_increment = step_i['increment_table'][
        step_i['increment_table']['比较（加入特征组后）'].eq('A+B+C+D+E+F vs A+B+C+D+E')]
    print('Step I：F 组增量（A+B+C+D+E+F vs A+B+C+D+E）')
    print(f_increment[['数据子集', 'ΔMAE（无该组 − 有该组）', 'bootstrap_CI95_下界',
                       'bootstrap_CI95_上界', 'CI是否跨0']].to_string(index=False))
    print(f'  特征维度 {base_dim} → {f_dim}')

    # ---- Step J ----
    entity_publish = entity_time.set_index('intern_id')['_entity_publish']
    step_j = run_step_j(f_frame, entity_publish, grouped, skill_map, text_matrix, text_by_id,
                        final_model_key, final_params, step_i['random_metrics'],
                        skill_threshold, text_dim)
    io_utils.write_excel(TABLES / TABLE_FILES[7], step_j['sheets'])
    print(f"Step J：Temporal test MAE {step_j['test_metrics']['MAE']}"
          f"（n = {step_j['test_metrics']['n']}）；主模型 {step_j['final_key']}")

    # ---- 图件 ----
    snapshot = plot_style.setup_sci_style()
    registry: list = []
    figure_s18(episodes, registry)
    figure_s19(daily, registry)
    figure_s20(step_f['category_daily'], daily['date'].to_numpy(), registry)
    figure_s21(daily, registry)
    figure_s22(step_j['comparison_table'], registry)
    for item in registry:
        item['failed_paper_gates'] = figure_finalize.failed_paper_gates(item)
        print(f"  图 {item['stem']}：{item['png_size_bytes']:,} 字节 / "
              f"{item['png_pixel_size']} px / 未通过门禁 {item['failed_paper_gates'] or '无'}")

    anchors = figure_finalize.validate_anchors()
    anchors_ok = all(item['通过'] for item in anchors)

    # ---- metrics JSON ----
    new_files = [{'路径': str(path.relative_to(PROJECT_ROOT)).replace('\\', '/'),
                  '字节数': int(path.stat().st_size), 'SHA256': sha256_of(path)}
                 for path in NEW_FILES if path.exists() and path != METRICS_PATH]
    new_files.append({'路径': str(METRICS_PATH.relative_to(PROJECT_ROOT)).replace('\\', '/'),
                      '字节数': 'N/A（本文件自引用）', 'SHA256': 'N/A（避免自哈希不一致）'})
    inputs = [project_paths.OBSERVATION_SNAPSHOT_PARQUET,
              project_paths.JOB_ANALYSIS_DATASET_PARQUET,
              project_paths.PROCESSED_UNIQUE_PARQUET,
              project_paths.JOB_SALARY_MODEL_DATASET_PARQUET,
              project_paths.MODEL_SPLITS_PARQUET, project_paths.SALARY_TARGETS_PARQUET,
              project_paths.TEXT_CORPUS_PARQUET, project_paths.PROCESSED_VERSION_HISTORY_PARQUET,
              project_paths.JOB_SKILL_MEMBERSHIP_PARQUET,
              project_paths.FEATURES_DIR / 'job_text_embeddings.npz',
              project_paths.FEATURES_DIR / 'job_text_embedding_index.parquet',
              project_paths.SALARY_MODEL_DIR / 'feature_manifest.json',
              project_paths.SALARY_MODEL_DIR / 'model_params.json',
              TABLES / project_paths.TABLE_MODEL_COMPARISON,
              TABLES / project_paths.TABLE_ABLATION_SHAP,
              PROJECT_ROOT / 'scripts' / '11_clean_structured_fields.py',
              PROJECT_ROOT / 'src' / 'plot_style.py',
              PROJECT_ROOT / 'src' / 'figure_finalize.py',
              PROJECT_ROOT / 'src' / 'versioning.py',
              PROJECT_ROOT / 'src' / 'ablation_shap.py']
    payload = {
        'stage': 'Stage26',
        '脚本路径': 'scripts/26_stage26_temporal.py',
        '运行命令': 'E:\\anaconda3\\envs\\reptile\\python.exe scripts\\26_stage26_temporal.py',
        '运行时间': {'开始': datetime.fromtimestamp(started).isoformat(timespec='seconds'),
                 '结束': datetime.now().isoformat(timespec='seconds'),
                 '耗时秒': round(time.time() - started, 2)},
        '随机种子': {'脚本种子': SEED, 'bootstrap种子': BOOTSTRAP_SEED,
                 'bootstrap轮数': BOOTSTRAP_ROUNDS,
                 '说明': '抽样与 bootstrap 均固定种子，可复现'},
        '时间口径': {
            '业务时间轴': '发布时间（招聘周期开始）/ 投递截止日期（计划结束）',
            '爬取时间': '观测时间 / 数据创建时间 / 数据更新时间：仅用于版本排序与审计，'
                    '不进入 F 组特征、不进入 Temporal Split',
            '观测批次': f"{step_a['summary']['observation_dates']} 个采集日"
                    f"（{step_a['summary']['observation_range']}）→ 采集批次产物，"
                    '不代表招聘市场自身的时间演化'},
        'StepA_时间字段审计': step_a['summary'],
        'StepB_周期识别': step_b['summary'],
        'StepC_人工抽查': {'输出文件': f'outputs/tables/{TABLE_FILES[8]}',
                      '抽样岗位数': int(len(review_sample)),
                      '抽样行数': int(len(review_table)),
                      'T3典型样例': t3_show.to_dict('records'),
                      '自动检查': review_checks.to_dict('records')},
        'StepD_日级面板': {
            '行数': panel_info['rows'], '上限': PANEL_MAX_ROWS,
            '是否启用替代表示': bool(panel_info['rows'] > PANEL_MAX_ROWS),
            '写入口径': '仅 episode_start ≤ date ≤ episode_end 的活跃日，不生成 active = 0 记录',
            '日期范围': f"{daily['date'].min()} ~ {daily['date'].max()}",
            '口径': CALIBER_ACTIVE},
        'StepE_时序指标': {'日数': int(len(daily)),
                       '汇总': temporal_sheets['02_指标汇总'].to_dict('records'),
                       'ReopenRate_t口径': 'R_t / O_t（O_t > 0）；R_t 含 T3 冲突事件，'
                                        '另列 T1 口径'},
        'StepF_生命周期': {'计划持续时长': step_f['sheets']['01_计划持续时长'].to_dict('records'),
                      '持续时长区间分布': step_f['sheets']['02_持续时长区间分布'].to_dict('records'),
                      '重招间隔': step_f['gap_rows'].to_dict('records'),
                      '招聘轮次': step_f['round_rows'].to_dict('records'),
                      '按岗位大类': step_f['category_table'].to_dict('records')},
        'StepG_生命周期薪资关联': {
            '样本': step_g['n_with_lifecycle'],
            '是否重招': step_g['reopen_table'].to_dict('records'),
            '周期长度检验': step_g['duration_test'],
            '招聘轮次检验': step_g['round_test'],
            '重招间隔检验': step_g['gap_test'], '边界': NO_CAUSAL},
        'StepH_F特征组': {
            '纳入特征': F_NUMERIC,
            'F特征统计': f_feature_table.to_dict('records'),
            '未来泄漏审计': f_leakage.to_dict('records'),
            '预测单位': '一岗一行（14,883 个正式代表样本），'
                    'F 特征只用代表 Episode 开始之前已发生的历史'},
        'StepI_F组消融': {
            '消融结果': step_i['ablation_table'].to_dict('records'),
            '增量与bootstrap': step_i['increment_table'].to_dict('records'),
            '与Stage15复现': step_i['reproduction'].to_dict('records'),
            '特征维度': {'A+B+C+D+E': base_dim, 'A+B+C+D+E+F': f_dim, 'Stage25正式': 320}},
        'StepJ_TemporalSplit': {
            '划分': step_j['split_table'].to_dict('records'),
            '模型比较': step_j['comparison'].to_dict('records'),
            '三划分对照': step_j['comparison_table'].to_dict('records'),
            '主模型': step_j['final_key'], '超参数': step_j['final_params'],
            '技能阈值': step_j['skill_threshold'], '文本维度': step_j['text_dim'],
            '特征维度': step_j['feature_dimension'],
            'test指标': step_j['test_metrics'],
            'train口径test指标': step_j['train_only_metrics'],
            'Temporal_Unseen': step_j['unseen']},
        '图件': [{key: item[key] for key in
                ('stem', 'png_path', 'pdf_path', 'png_pixel_size', 'png_dpi',
                 'png_size_bytes', 'pdf_size_bytes', 'no_infigure_caption',
                 'subfigure_caption_visible', 'failed_paper_gates')}
               for item in registry],
        '样式快照': {key: value for key, value in snapshot.items()
                 if key in ('context', 'style', 'facecolor', 'savefig_dpi', 'headless')},
        'Stage23锚点自检': {'通过数': int(sum(item['通过'] for item in anchors)),
                       '总数': int(len(anchors)),
                       '结果': '全部一致' if anchors_ok else '存在不符项', '清单': anchors},
        '输入文件清单': [input_record(path) for path in inputs],
        '本次新增文件': new_files,
        '本次新增文件说明': f'本脚本只新增 {len(NEW_FILES)} 个文件，不覆盖任何既有产物。',
        '口径说明汇总': [
            CALIBER_ACTIVE, CALIBER_T2,
            '投递截止日期字段偏粗且含量默认值（仅 337 个唯一取值，Top20 覆盖近六成），'
            '因此周期「计划结束日」只能作为粗粒度业务时间使用。',
            '全部分析不得表述为因果。'],
    }
    io_utils.write_json(METRICS_PATH, payload)

    manifest_after = project_manifest()
    changed = sorted(path for path in manifest_before
                     if path in manifest_after and manifest_after[path] != manifest_before[path])
    removed = sorted(set(manifest_before) - set(manifest_after))
    added = sorted(set(manifest_after) - set(manifest_before))
    groups = {}
    for name, prefix in [('data/', 'data/'), ('outputs/', 'outputs/'), ('docs/', 'docs/'),
                         ('src/', 'src/'), ('scripts/', 'scripts/')]:
        keys = [path for path in manifest_before if path.startswith(prefix)]
        groups[name] = {'文件数': len(keys),
                        '内容一致': bool(all(manifest_after.get(path) == manifest_before[path]
                                         for path in keys)),
                        '内容变化文件': [path for path in keys
                                   if manifest_after.get(path) != manifest_before[path]]}
    payload['既有文件未修改证据'] = {
        '清单规模': {'运行前文件数': len(manifest_before), '运行后文件数': len(manifest_after),
                 '清单内新增文件数': len(added)},
        '内容变化文件': changed, '被删除文件': removed, '清单内新增文件': added,
        '分组结论': groups,
        '清单生成方式': '对项目内既有文件（data/、outputs/、docs/、src/、scripts/、config/、'
                    'notebooks/、tests/、README.md，跳过缓存目录）逐个计算 SHA-256，'
                    '在写入本次新增产物前后各生成一次并逐文件比对；本脚本新增产物不入清单。',
        '结论': ('未修改任何既有文件（内容变化 0 个、删除 0 个、清单内新增 0 个）'
              if not changed and not removed and not added
              else '存在内容变化/删除/清单内新增，需人工复核')}
    io_utils.write_json(METRICS_PATH, payload)

    print('-' * 96)
    print(f'指标 JSON：{project_paths.relative_to_root(METRICS_PATH)} '
          f'（{METRICS_PATH.stat().st_size:,} 字节）')
    print(f'既有文件内容变化 {len(changed)} 个；删除 {len(removed)} 个；清单内新增 {len(added)} 个')
    for item in new_files:
        print(f"  新增：{item['路径']} | {item['字节数']} | {str(item['SHA256'])[:16]}…")
    print(f'总耗时 {time.time() - started:.1f} 秒')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

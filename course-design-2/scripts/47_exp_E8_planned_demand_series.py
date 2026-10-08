# -*- coding: utf-8 -*-
"""E8：计划招聘需求时间序列（发布—截止日窗口的日级展开）。

定义（与论文 4.5 节统一口径一致）：
    W_e = [p_e, d_e]，p_e 为该重构招聘周期的发布时间，d_e 为投递截止日期，端点均包含；
    N_t = Σ_e I(p_e ≤ t ≤ d_e) 为业务日期 t 的计划招聘需求窗口覆盖量，单位是「岗位周期数」；
    A_t = Σ_e I(p_e = t) 为流入，C_t = Σ_e I(d_e = t − 1) 为计划退出（截止日当天仍计入窗口）。

口径与边界：
    1. 输入为 3.4.3 节识别的重构招聘周期（7 个日历天阈值下的严格口径）；
    2. 时间轴只用业务日期（发布时间、投递截止日期），不使用任何采集时间；
    3. 缺失日期、d_e < p_e 的记录先出审计表，不静默填补或删除；
    4. 异常长窗口保留原值，另做「剔除窗口长度最高 1%」的敏感性对照；
    5. N_t 只表示样本中可重构的计划招聘需求窗口覆盖量，不是实际市场需求、招聘人数或预测。

用法：
    python scripts\\47_exp_E8_planned_demand_series.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src import figure_finalize, plot_style, project_paths, schema  # noqa: E402

EPISODE = PROJECT / 'data' / 'processed' / 'job_strict_episode_26_1.parquet'
DAILY_PANEL = PROJECT / 'data' / 'processed' / 'job_strict_daily_panel_26_1.parquet'
CATEGORY = PROJECT / 'data' / 'processed' / 'job_category_membership.parquet'
RESULTS = project_paths.OUTPUTS_RESULTS_DIR
FIGURES = project_paths.FIGURES_DIR
TOP_CATEGORIES = 5


def build_series(starts, ends, timeline):
    """由窗口端点构造 N_t / A_t / C_t。"""
    position = {value: index for index, value in enumerate(timeline)}
    delta = np.zeros(len(timeline) + 1, dtype='int64')
    inflow = np.zeros(len(timeline), dtype='int64')
    exit_series = np.zeros(len(timeline), dtype='int64')
    for begin, finish in zip(starts, ends):
        delta[position[begin]] += 1
        delta[position[finish] + 1] -= 1
        inflow[position[begin]] += 1
        # 截止日当天仍计入窗口，因此计划退出计在 d_e + 1 当日
        next_day = position[finish] + 1
        if next_day < len(timeline):
            exit_series[next_day] += 1
    coverage = np.cumsum(delta[:-1])
    frame = pd.DataFrame({
        'date': timeline,
        'planned_demand_coverage_N': coverage,
        'new_window_A': inflow,
        'planned_exit_C': exit_series,
    })
    frame['net_change'] = frame['planned_demand_coverage_N'].diff()
    frame.loc[0, 'net_change'] = frame.loc[0, 'planned_demand_coverage_N']
    frame['net_change'] = frame['net_change'].astype('int64')
    return frame


def main() -> int:
    plot_style.setup_sci_style()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    episode = pd.read_parquet(EPISODE)
    unique = episode.drop_duplicates(subset=['episode_id_strict']).copy()
    unique['p_date'] = pd.to_datetime(unique['episode_start']).dt.normalize()
    unique['d_date'] = pd.to_datetime(unique['final_observed_deadline']).dt.normalize()
    print('严格口径招聘周期（去重后）:', len(unique))

    missing = unique['p_date'].isna() | unique['d_date'].isna()
    reversed_order = (~missing) & (unique['d_date'] < unique['p_date'])
    expandable = (~missing) & (~reversed_order)
    window = (unique.loc[expandable, 'd_date'] - unique.loc[expandable, 'p_date']).dt.days + 1
    print('日期缺失 %d；逆序 %d；可展开 %d' % (missing.sum(), reversed_order.sum(),
                                        expandable.sum()))

    audit_rows = [
        {'项目': '输入周期总数（严格口径）', '取值': int(len(unique))},
        {'项目': '日期缺失周期数', '取值': int(missing.sum())},
        {'项目': '截止日期早于发布日期的逆序周期数', '取值': int(reversed_order.sum())},
        {'项目': '可展开周期数', '取值': int(expandable.sum())},
        {'项目': '窗口长度 P50（日历天）', '取值': float(window.median())},
        {'项目': '窗口长度 P90（日历天）', '取值': float(window.quantile(0.90))},
        {'项目': '窗口长度 P99（日历天）', '取值': float(window.quantile(0.99))},
        {'项目': '最长窗口（日历天）', '取值': int(window.max())},
        {'项目': '最短窗口（日历天）', '取值': int(window.min())},
        {'项目': '时间轴起始日期', '取值': str(unique.loc[expandable, 'p_date'].min().date())},
        {'项目': '时间轴结束日期', '取值': str(unique.loc[expandable, 'd_date'].max().date())},
        {'项目': '时间轴天数', '取值': 0},
        {'项目': '窗口日数合计（= Σ N_t）', '取值': 0},
        {'项目': '冻结日级面板行数（交叉核对）', '取值': int(len(pd.read_parquet(
            DAILY_PANEL, columns=['date'])))},
    ]

    timeline = pd.date_range(unique.loc[expandable, 'p_date'].min(),
                             unique.loc[expandable, 'd_date'].max(), freq='D')
    series = build_series(unique.loc[expandable, 'p_date'].tolist(),
                          unique.loc[expandable, 'd_date'].tolist(), timeline)
    audit_rows[11]['取值'] = int(len(timeline))
    audit_rows[12]['取值'] = int(series['planned_demand_coverage_N'].sum())
    identity = (series['planned_demand_coverage_N']
                - series['planned_demand_coverage_N'].shift(1).fillna(0)
                - series['new_window_A'] + series['planned_exit_C'])
    audit_rows.append({'项目': '恒等式 N_t = N_(t−1) + A_t − C_t 的最大偏离',
                       '取值': int(np.abs(identity).max())})
    audit_rows.append({'项目': 'N_t 最小值', '取值': int(series['planned_demand_coverage_N'].min())})
    audit_rows.append({'项目': 'N_t 最大值', '取值': int(series['planned_demand_coverage_N'].max())})
    audit_rows.append({'项目': 'N_t 中位数', '取值': float(series['planned_demand_coverage_N'].median())})
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(project_paths.RESULTS_E8 / 'planned_recruitment_demand_audit.csv', index=False,
                 encoding='utf-8-sig')
    print(audit.to_string(index=False))

    series.to_csv(project_paths.RESULTS_E8 / 'planned_recruitment_demand_daily.csv', index=False,
                  encoding='utf-8-sig')

    # ---- 敏感性：剔除窗口长度最高 1% 的周期 ----
    cutoff = float(window.quantile(0.99))
    kept_index = window[window <= cutoff].index
    trimmed = unique.loc[kept_index]
    trimmed_series = build_series(trimmed['p_date'].tolist(), trimmed['d_date'].tolist(),
                                  timeline)
    merged = series[['date', 'planned_demand_coverage_N']].merge(
        trimmed_series[['date', 'planned_demand_coverage_N']], on='date',
        suffixes=('_主口径', '_剔除最高1%'))
    spearman = float(merged['planned_demand_coverage_N_主口径'].corr(
        merged['planned_demand_coverage_N_剔除最高1%'], method='spearman'))
    sensitivity = pd.DataFrame([
        {'指标': '周期数', '主口径': int(len(unique)), '剔除窗口长度最高 1%': int(len(trimmed))},
        {'指标': 'Σ N_t（窗口日数合计）', '主口径': int(series['planned_demand_coverage_N'].sum()),
         '剔除窗口长度最高 1%': int(trimmed_series['planned_demand_coverage_N'].sum())},
        {'指标': 'N_t 均值', '主口径': float(series['planned_demand_coverage_N'].mean()),
         '剔除窗口长度最高 1%': float(trimmed_series['planned_demand_coverage_N'].mean())},
        {'指标': 'N_t 中位数', '主口径': float(series['planned_demand_coverage_N'].median()),
         '剔除窗口长度最高 1%': float(trimmed_series['planned_demand_coverage_N'].median())},
        {'指标': 'N_t 最大值', '主口径': int(series['planned_demand_coverage_N'].max()),
         '剔除窗口长度最高 1%': int(trimmed_series['planned_demand_coverage_N'].max())},
        {'指标': '两序列 Spearman 相关系数', '主口径': round(spearman, 6),
         '剔除窗口长度最高 1%': '—'},
        {'指标': '窗口长度 99% 分位阈值（日历天）', '主口径': round(cutoff, 1),
         '剔除窗口长度最高 1%': '—'},
    ])
    sensitivity.to_csv(project_paths.RESULTS_E8 / 'planned_recruitment_demand_sensitivity.csv', index=False,
                       encoding='utf-8-sig')
    print(sensitivity.to_string(index=False))

    # ---- 图 4-6：样本内可重构的计划招聘需求窗口覆盖量（单面板 + 敏感性对照） ----
    # 只保留覆盖量面板：日级流入 63.8%、日级计划退出 85.8% 的日期取值为 0，任何时间聚合
    # 都仍留下大量空桶（周级 49% / 79%），故流量不单独入图，只在 CSV 与审计表交付。
    # 为让图承担正文没有的信息量，叠加「剔除窗口长度最高 1%」子样本的 N_t 作稳健性对照。
    # 不再画 7 日滚动均值：日级 N_t 本身平滑（日级净变化 P90 = 6 个周期 ≈ 0.6 px），
    # 滚动均值与它完全重合，只会把日级线盖住（实测仅 1,374 px 可见），属冗余元素。
    with plt.rc_context({'font.size': 11.0, 'axes.labelsize': 12.0,
                         'xtick.labelsize': 11.0, 'ytick.labelsize': 11.0,
                         'legend.fontsize': 11.0}):
        fig, ax = plt.subplots(figsize=(15.5 / 2.54 * 11 / 12, 3.4))
        ax.plot(series['date'], series['planned_demand_coverage_N'],
                color='#1F4E79', linewidth=1.5, label='日级覆盖量 N_t（主口径）')
        ax.plot(trimmed_series['date'], trimmed_series['planned_demand_coverage_N'],
                color=plot_style.ACCENT_COLOR, linestyle='--', linewidth=1.2,
                label='N_t（剔除窗口长度最高 1%）')
        ax.set_xlabel('业务日期')
        ax.set_ylabel('计划招聘需求窗口覆盖量\n（岗位周期数）')
        ax.legend(loc='upper left', frameon=False)
        fig.subplots_adjust(left=0.16, right=0.975, bottom=0.17, top=0.965)
        diagnostics = figure_finalize.save_paper_figure(
            fig, FIGURES, 'fig_4_6_planned_recruitment_demand_time_series',
            subfigures=[],
            meta={'数据来源': 'outputs/results/E8/planned_recruitment_demand_daily.csv；'
                             '敏感性对照取自 planned_recruitment_demand_sensitivity.csv',
                  '口径': 'W_e = [p_e, d_e] 端点均包含；N_t = Σ I(p_e ≤ t ≤ d_e)；'
                          '剔除窗口长度最高 1%% 后两序列 Spearman = %.4f' % spearman,
                  '解释边界': 'N_t 只表示样本中可重构的计划招聘需求窗口覆盖量，'
                              '不是历史市场真实岗位存量、招聘人数或实际新增岗位数，'
                              '也不用于需求预测'})
        plt.close(fig)
    print('图 4-6 门禁未通过项:', figure_finalize.failed_paper_gates(diagnostics) or '无')

    # ---- 图 4-7：按岗位大类的覆盖量（最多 5 个方向 + 其他） ----
    membership = pd.read_parquet(CATEGORY)
    weight = membership.groupby([schema.ID_FIELD, '岗位大类'])['原始命中次数'].sum().reset_index()
    primary = weight.sort_values(['实习岗位ID', '原始命中次数'],
                                 ascending=[True, False]).drop_duplicates('实习岗位ID')
    primary = primary.set_index(schema.ID_FIELD)['岗位大类']
    unique['岗位大类'] = unique['intern_id'].map(primary)
    counts = unique.loc[expandable, '岗位大类'].value_counts()
    top = list(counts.index[:TOP_CATEGORIES])
    unique['方向分档'] = np.where(unique['岗位大类'].isin(top), unique['岗位大类'], '其他')
    rows = []
    for label in top + ['其他']:
        block = unique.loc[expandable & unique['方向分档'].eq(label)]
        if not len(block):
            continue
        block_series = build_series(block['p_date'].tolist(), block['d_date'].tolist(), timeline)
        rows.append((label, len(block), block_series))
    # ---- 图 4-7：按岗位大类分解的覆盖量（论文版式：15.5 cm 版心宽、无图内标题） ----
    palette = ['#1F4E79', '#2E75B6', '#8FB4D9', '#C0504D', '#E0A177', '#9A9A9A']
    with plt.rc_context({'font.size': 11.0, 'axes.labelsize': 12.0,
                         'xtick.labelsize': 11.0, 'ytick.labelsize': 11.0,
                         'legend.fontsize': 11.0}):
        fig, ax = plt.subplots(figsize=(15.5 / 2.54 * 11 / 12, 4.0))
        for (label, size, block_series), color in zip(rows, palette):
            ax.plot(block_series['date'],
                    block_series['planned_demand_coverage_N'].rolling(14, min_periods=1).mean(),
                    color=color, linewidth=1.4,
                    label='%s（%s 个周期）' % (label, f'{size:,}'))
        ax.set_xlabel('业务日期')
        # 纵轴只留量名；「14 日滚动均值」这个图级口径参数按论文惯例移到图题
        ax.set_ylabel('计划招聘需求窗口覆盖量')
        # 图例放到坐标区上方：6 条标签在区内会占掉 91% 宽度、24% 高度并压住运营的峰值
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=2,
                  frameon=False, fontsize=10.0)
        fig.subplots_adjust(left=0.155, right=0.985, bottom=0.165, top=0.845)
        diagnostics = figure_finalize.save_paper_figure(
            fig, FIGURES, 'fig_4_7_planned_demand_by_job_category',
            subfigures=[],
            meta={'数据来源': 'job_strict_episode_26_1.parquet × 岗位大类集合',
                  '口径': '各方向 N_t 的 14 日滚动均值；分档 = 周期数前 5 个大类 + 其他',
                  '方向分档': {label: int(size) for label, size, _ in rows},
                  '解释边界': 'N_t 只表示样本中可重构的计划招聘需求窗口覆盖量，'
                              '不是历史市场真实岗位存量、招聘人数或实际新增岗位数，'
                              '也不用于需求预测'})
        plt.close(fig)
    print('图 4-7 门禁未通过项:', figure_finalize.failed_paper_gates(diagnostics) or '无')

    meta = {
        '输入': 'data/processed/job_strict_episode_26_1.parquet（7 个日历天阈值严格口径）',
        '口径': 'W_e = [p_e, d_e] 端点均包含；N_t = Σ I(p_e ≤ t ≤ d_e)；'
                'A_t = Σ I(p_e = t)；C_t = Σ I(d_e = t − 1)',
        '时间轴': '%s ~ %s（%d 天）' % (timeline[0].date(), timeline[-1].date(), len(timeline)),
        '可展开周期数': int(expandable.sum()),
        '两序列 Spearman': spearman,
        '方向分档': {label: int(size) for label, size, _ in rows},
        '解释边界': 'N_t 只表示样本中可重构的计划招聘需求窗口覆盖量，'
                    '不是历史市场真实岗位存量、招聘人数或实际新增岗位数，也不用于需求预测',
    }
    with (project_paths.RESULTS_E8 / 'planned_recruitment_demand_meta.json').open('w', encoding='utf-8') as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=1)
    print('E8 完成：', json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())

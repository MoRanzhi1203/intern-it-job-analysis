# -*- coding: utf-8 -*-
"""重绘第 4 章业务时间维度两张图（outputs/figures/time/02、03）。

只做版式与可读性优化，不改动任何统计范围与数值：

- 图「发布时间队列的薪资中位数及四分位区间」（``fig_4_9_salary_by_publish_time``）：
  数据源为冻结结果表 ``outputs/tables/ch4/59_business_time_dimension_analysis.xlsx`` 的
  ``03_发布时间月度薪资分布`` 工作表，参考线取自 29 号表 ``02_薪资描述统计`` 的中位数；
- 图「核心技能需求的发布时间队列变化」（``fig_4_10_skill_or_category_time_structure``）：
  数据源为同表的 ``04_发布时间技能或岗位结构`` 工作表（命中率列），横轴沿用原图统计范围，
  只排布主统计范围岗位数不少于 30、即给出命中率的窗口；
- 版式为论文版：无图内总图题、600 dpi PNG + 矢量 PDF、四边框内向刻度、浅虚线网格，
  与全文其他重绘图件（如 26j 的图S69~图S73）保持一致；
- 所有读取均为只读，不重算、不写回任何冻结数据表。

绘制与门禁逻辑集中在 :func:`save_salary_figure` 与 :func:`save_skill_figure`，
33 号流水线脚本复用同一函数出图。

用法：:

    python scripts\\49_time_cohort_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = next(
    _candidate for _candidate in Path(__file__).resolve().parents
    if (_candidate / "data").is_dir() and (_candidate / "scripts").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import figure_finalize, plot_style, project_paths  # noqa: E402

TABLES = project_paths.TABLES_DIR
FIGDIR = project_paths.FIGURES_DIR / 'time'
METRICS = project_paths.METRICS_DIR
TABLE33 = project_paths.TABLE_BUSINESS_TIME_DIMENSION

STEM_SALARY = 'fig_4_9_salary_by_publish_time'
STEM_SKILL = 'fig_4_10_skill_or_category_time_structure'
MIN_WINDOW_N = 30
SKILLS = ['Python', 'SQL', '人工智能', '大模型']
# 与全文其他图件一致的色盲友好配色（plot_style.PALETTE 前四位）
SKILL_COLORS = ['#0173B2', '#DE8F05', '#029E73', '#D55E00']

CM = 1.0 / 2.54
FONT_SCALE = 11.0 / 12.0
FONTS = {'axis_label': 12.0, 'tick': 11.0, 'legend': 11.0, 'annotation': 10.5}
# 两图共用的图例参数：置于坐标区上方、单/多行排布，宽度不得超过坐标区宽度
LEGEND_KW = dict(frameon=False, fontsize=FONTS['legend'], columnspacing=0.9,
                 handletextpad=0.5, handlelength=1.4, borderaxespad=0.0)


def _w(width_cm: float) -> float:
    """按论文版心宽度（cm）换算绘图物理宽度（英寸）。"""
    return width_cm * CM * FONT_SCALE


def _with_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """把同名索引还原成列，并按键升序排列（兼容流水线与冻结表的两种入参形态）。"""
    frame = frame.reset_index()
    if column not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: column})
    return frame.sort_values(column).reset_index(drop=True)


def _apply_month_axis(ax, labels: list, step: int, xlabel: str,
                      minor_ticks: bool = True):
    """横轴：每 step 个窗口标注一个刻度；minor_ticks 为真时逐窗口给短刻度。"""
    positions = np.arange(len(labels), dtype='float64')
    tick_pos = [int(index) for index in positions[::step]]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([str(labels[index]) for index in tick_pos],
                       rotation=45, ha='right', rotation_mode='anchor')
    if minor_ticks:
        ax.set_xticks(positions, minor=True)
    ax.set_xlim(-0.7, len(labels) - 0.3)
    ax.set_xlabel(xlabel, fontsize=FONTS['axis_label'])


def read_monthly_table() -> pd.DataFrame:
    """只读冻结表 33 的月度薪资分布（按月份升序）。"""
    frame = pd.read_excel(TABLES / TABLE33, sheet_name='03_发布时间月度薪资分布')
    return _with_column(frame, '月份')


def read_structure_table() -> pd.DataFrame:
    """只读冻结表 33 的月度技能命中结构（按月份升序）。"""
    frame = pd.read_excel(TABLES / TABLE33, sheet_name='04_发布时间技能或岗位结构')
    return _with_column(frame, 'period')


def read_full_median() -> float:
    """只读 29 号表给出的全样本薪资中点中位数。"""
    stats = pd.read_excel(TABLES / 'ch4/21_eda_statistical_analysis.xlsx',
                          sheet_name='02_薪资描述统计').set_index('指标')['数值']
    return float(stats['中位数'])


def build_salary_figure(monthly: pd.DataFrame, full_median: float):
    """绘制薪资中位数及四分位区间图，返回 ``(fig, frame)``。"""
    import matplotlib.pyplot as plt

    frame = _with_column(monthly, '月份')
    positions = np.arange(len(frame), dtype='float64')
    median = frame['薪资中位数 Median'].to_numpy('float64')
    p25 = frame['P25'].to_numpy('float64')
    p75 = frame['P75'].to_numpy('float64')
    sizes = frame['有效薪资岗位数 n'].to_numpy('float64')
    has_salary = ~np.isnan(median)
    sparse = sizes < MIN_WINDOW_N
    dense = has_salary & ~sparse
    thin = has_salary & sparse

    fig, ax = plt.subplots(figsize=(_w(15.5), 4.1))
    # n < 30 的窗口：浅色底纹，对应「仅展示、不作趋势解释」的窗口
    for position in positions[sparse]:
        ax.axvspan(position - 0.5, position + 0.5, facecolor=plot_style.MUTED_COLOR,
                   alpha=0.13, edgecolor='none', zorder=0.4)
    band_handle = plt.Rectangle((0, 0), 1, 1, facecolor=plot_style.MUTED_COLOR,
                                alpha=0.13, edgecolor='none')
    dense_bar = ax.errorbar(
        positions[dense], median[dense],
        yerr=[median[dense] - p25[dense], p75[dense] - median[dense]],
        fmt='o', color=plot_style.MAIN_COLOR, ecolor=plot_style.MAIN_COLOR,
        elinewidth=1.0, capsize=2.4, capthick=0.9, markersize=3.4, zorder=3)
    thin_bar = ax.errorbar(
        positions[thin], median[thin],
        yerr=[median[thin] - p25[thin], p75[thin] - median[thin]],
        fmt='o', color=plot_style.MUTED_COLOR, ecolor=plot_style.MUTED_COLOR,
        elinewidth=1.0, capsize=2.4, capthick=0.9, markersize=3.4, zorder=3)
    reference = ax.axhline(full_median, color=plot_style.MUTED_COLOR, linewidth=0.8,
                           linestyle='--', zorder=2)

    _apply_month_axis(ax, list(frame['月份'].astype(str)), step=6,
                      xlabel='岗位发布时间')
    ax.set_yticks(np.arange(100, 251, 25))
    ax.set_ylim(100, 280)
    ax.set_ylabel('薪资中点（元/天）', fontsize=FONTS['axis_label'])
    ax.legend(handles=[dense_bar, thin_bar, band_handle, reference],
              labels=['中位数与四分位区间', '小样本窗口（n < 30）',
                      '小样本窗口底纹', '全样本中位数（%.0f 元/天）' % full_median],
              loc='lower center', bbox_to_anchor=(0.5, 1.005), ncol=2, **LEGEND_KW)
    plot_style.apply_sci_axis(ax, grid_axis='y')
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.235, top=0.80)
    return fig, frame


def build_skill_figure(structure: pd.DataFrame):
    """绘制核心技能需求的发布时间队列变化图，返回 ``(fig, windows)``。

    横轴沿用原图统计范围：只排布给出命中率的窗口（主统计范围岗位数不少于 30），等距首尾相接，
    不补画未达阈值的窗口。
    """
    import matplotlib.pyplot as plt

    frame = _with_column(structure, 'period')
    eligible = frame['主统计范围岗位数'].to_numpy('float64') >= MIN_WINDOW_N
    windows = frame.loc[eligible].reset_index(drop=True)
    positions = np.arange(len(windows), dtype='float64')

    fig, ax = plt.subplots(figsize=(_w(15.5), 4.1))
    lines = []
    for color, skill in zip(SKILL_COLORS, SKILLS):
        values = windows['%s命中率' % skill].to_numpy('float64') * 100.0
        lines.append(ax.plot(positions, values, color=color, linewidth=1.2, marker='o',
                             markersize=2.8, markeredgewidth=0, label=skill, zorder=3)[0])

    _apply_month_axis(ax, list(windows['period'].astype(str)), step=4,
                      xlabel='发布时间窗口', minor_ticks=False)
    ax.set_yticks(np.arange(0, 31, 5))
    ax.set_ylim(-1.5, 32)
    ax.set_ylabel('要求段落命中率（%）', fontsize=FONTS['axis_label'])
    ax.legend(handles=lines, labels=SKILLS, loc='lower center',
              bbox_to_anchor=(0.5, 1.005), ncol=4, **LEGEND_KW)
    plot_style.apply_sci_axis(ax, grid_axis='y')
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.235, top=0.80)
    return fig, windows


def save_salary_figure(monthly: pd.DataFrame, full_median: float,
                       out_dir=FIGDIR, stem: str = STEM_SALARY) -> dict:
    """按论文版式出薪资图并做视觉门禁校验，返回诊断字典。"""
    import matplotlib.pyplot as plt

    plot_style.setup_sci_style()
    plot_style.FONT_SIZES.update(FONTS)
    fig, frame = build_salary_figure(monthly, full_median)
    diagnostics = figure_finalize.save_paper_figure(
        fig, out_dir, stem, subfigures=[],
        meta={'数据来源': '33 号表 03_发布时间月度薪资分布（岗位发布时间聚合到自然月）',
              '统计范围': '点为该发布月窗口内有效明确薪资岗位的薪资中点中位数，误差线为 P25~P75 真实'
                      '四分位区间（下误差 = 中位数 − P25，上误差 = P75 − 中位数）；'
                      'n < 30 的窗口加浅色底纹，仅展示、不作趋势解释；'
                      '参考线为全样本薪资中点中位数 %.0f 元/天；逐月对照只描述本样本，'
                      '不代表市场岗位存量' % full_median,
              '窗口数': int(frame.shape[0]),
              'n<30 窗口数': int((frame['有效薪资岗位数 n'] < MIN_WINDOW_N).sum()),
              '用途': '第4章「发布时间队列的薪资中位数及四分位区间」重绘（图题由 Word 构建脚本生成）'})
    plt.close(fig)
    return diagnostics


def save_skill_figure(structure: pd.DataFrame,
                      out_dir=FIGDIR, stem: str = STEM_SKILL) -> dict:
    """按论文版式出技能图并做视觉门禁校验，返回诊断字典。"""
    import matplotlib.pyplot as plt

    plot_style.setup_sci_style()
    plot_style.FONT_SIZES.update(FONTS)
    fig, windows = build_skill_figure(structure)
    diagnostics = figure_finalize.save_paper_figure(
        fig, out_dir, stem, subfigures=[],
        meta={'数据来源': '33 号表 04_发布时间技能或岗位结构（要求段落主统计范围命中率列）',
              '统计范围': '命中率 = 该发布月窗口内命中该技能的岗位数 ÷ 该窗口主统计范围岗位数；'
                      '只在主统计范围岗位数不少于 30 的窗口给出命中率，未达阈值的窗口不进入本图；'
                      '四项技能分属具体技术技能（Python、SQL、大模型）与技术领域（人工智能），'
                      '只比较各自随发布队列的变化，不做跨层级效应量比较',
              '横轴': '只排布给出命中率的 %d 个窗口，等距首尾相接' % windows.shape[0],
              '窗口数': int(windows.shape[0]),
              '阈值（主统计范围岗位数）': MIN_WINDOW_N,
              '用途': '第4章「核心技能需求的发布时间队列变化」重绘（图题由 Word 构建脚本生成）'})
    plt.close(fig)
    return diagnostics


def main() -> int:
    monthly = read_monthly_table()
    structure = read_structure_table()
    salary = save_salary_figure(monthly, read_full_median())
    skill = save_skill_figure(structure)
    results = {'salary': salary, 'skill': skill}
    failed = {key: figure_finalize.failed_paper_gates(value)
              for key, value in results.items()}

    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / 'stage_26_7_figure_redraw.json').write_text(
        json.dumps({key: {'png': value['png_path'], 'pdf': value['pdf_path'],
                          '论文版门禁未通过项': failed[key]}
                    for key, value in results.items()},
                   ensure_ascii=False, indent=2), encoding='utf-8')

    print('=' * 78)
    for key, value in results.items():
        print('图件：', value['png_path'])
        print('  像素：', value['png_pixel_size'], '| 门禁未通过项：', failed[key] or '无')
    print('=' * 78)
    return 0 if not any(failed.values()) else 1


if __name__ == '__main__':
    sys.exit(main())

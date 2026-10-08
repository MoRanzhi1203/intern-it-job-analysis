# -*- coding: utf-8 -*-
"""重绘图「代表性中位数回归系数的岗位级与公司级 bootstrap 95% 区间对照」。

沿用 46 号实验（E7）的冻结结果，只重排呈现方式，不重跑 bootstrap、不改动任何数值：

- 只读 ``outputs/results/quantile_regression_cluster_bootstrap.csv``（12 个代表性变量）；
- 单面板森林图：隔行浅底纹 + 带端帽的两类 95% 区间（岗位级在上、公司级在下）+ 菱形点估计；
- 横轴用 symlog：``|系数| ≤ 200`` 元/天 段保持线性（**100~200 不折叠**），更大值段按对数
  压缩，`每周到岗要求=7天／周` 的公司级区间宽 767 元/天，线性轴会把它之外的变量挤成一条线；
  压缩起点用点线标出，200 / 400 / 700 刻度在图上被压缩排布，折叠效果可直接读出，
  且不截断、不隐藏任何数值；
- 版式按项目论文版：无图内总图题、600 dpi PNG + 矢量 PDF、四边框内向刻度、
  Times New Roman + 微软雅黑；配色与图例用全文统一的色盲友好色板；
- y 轴标签按「因素」与「取值」两行排布，避免长变量名占满画幅宽度。

用法：:

    python scripts\\53_redraw_median_regression_bootstrap.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter  # noqa: E402

from src import figure_finalize, plot_style, project_paths  # noqa: E402

STEM = 'fig_5_7_median_regression_cluster_bootstrap'
CSV = project_paths.PROJECT_ROOT / 'outputs' / 'results' / 'quantile_regression_cluster_bootstrap.csv'
FIGDIR = project_paths.FIGURES_DIR
PRINT_WIDTH_CM = 16.0
PRINT_HEIGHT_CM = 16.5
# 横轴对称对数压缩的线性段阈值（元/天）与刻度：100~200 之间保持线性不折叠，
# 只有「每周到岗要求=7天／周」的公司级上界 754.6（以及 201.3）落在压缩段，
# 因此 200 / 400 / 700 三个刻度在图上被压缩排布，可直观看出折叠
SYMLOG_LINTHRESH = 200.0
TICKS = [-100, -50, 0, 50, 100, 200, 400, 700]
CM = 1.0 / 2.54
FONT_SCALE = 11.0 / 12.0
FONTS = {'axis_label': 12.0, 'tick': 11.0, 'legend': 11.0, 'annotation': 10.5}


def _w(width_cm: float) -> float:
    """按目标打印宽度（cm）换算绘图物理宽度（英寸）。"""
    return width_cm * CM * FONT_SCALE


def _label(name: str) -> str:
    """变量名拆成「因素 / 取值」两行，避免 16 字长名占满左侧。"""
    factor, _, value = str(name).partition('=')
    return '%s\n%s' % (factor, value) if value else str(name)


def load_representative() -> pd.DataFrame:
    """只读冻结结果，取代表性变量并按公司级区间宽度降序（与摘要表一致）。"""
    frame = pd.read_csv(CSV)
    frame = frame[frame['是否代表性变量'].astype(str).str.lower().isin(['true', '1'])]
    frame = frame.reindex(frame['公司级区间宽度'].sort_values(ascending=False).index)
    return frame.reset_index(drop=True)


def build_figure(frame: pd.DataFrame):
    """单面板森林图：行底纹 + 带端帽的两类 95% 区间 + 菱形点估计，横轴 symlog。"""
    height = PRINT_HEIGHT_CM / PRINT_WIDTH_CM * _w(PRINT_WIDTH_CM)
    fig, ax = plt.subplots(figsize=(_w(PRINT_WIDTH_CM), height))

    y = np.arange(len(frame), dtype='float64')
    job_lower = frame['岗位级 CI95 下界'].to_numpy('float64')
    job_upper = frame['岗位级 CI95 上界'].to_numpy('float64')
    cluster_lower = frame['公司级 CI95 下界'].to_numpy('float64')
    cluster_upper = frame['公司级 CI95 上界'].to_numpy('float64')
    coefficient = frame['系数'].to_numpy('float64')

    # 隔行浅底纹：帮助视线在同一行的两条区间之间横向对齐
    for row in y[::2]:
        ax.axhspan(row - 0.5, row + 0.5, facecolor='#f2f2f2', edgecolor='none', zorder=0.4)

    ax.errorbar(coefficient, y + 0.16,
                xerr=np.vstack([coefficient - job_lower, job_upper - coefficient]),
                fmt='none', ecolor=plot_style.MAIN_COLOR, elinewidth=1.6, capsize=2.6,
                capthick=1.4, zorder=3, label='岗位级 bootstrap 95% 区间')
    ax.errorbar(coefficient, y - 0.16,
                xerr=np.vstack([coefficient - cluster_lower, cluster_upper - coefficient]),
                fmt='none', ecolor=plot_style.ACCENT_COLOR, elinewidth=1.6, capsize=2.6,
                capthick=1.4, zorder=3, label='公司级 cluster bootstrap 95% 区间')
    ax.scatter(coefficient, y, marker='D', s=15, color='#333333', zorder=4)
    ax.axvline(0, color=plot_style.MUTED_COLOR, linestyle='--', linewidth=0.9, zorder=2)
    # 压缩起点：虚线以上按对数排布，直观标出「折叠」发生的位置
    ax.axvline(SYMLOG_LINTHRESH, color=plot_style.MUTED_COLOR, linestyle=':',
               linewidth=0.8, zorder=2)

    # 横轴对称对数压缩：|系数| ≤ 200 段保持线性（100~200 不折叠），更大值段压缩
    ax.set_xscale('symlog', linthresh=SYMLOG_LINTHRESH)
    ax.xaxis.set_major_locator(FixedLocator(TICKS))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _pos: ('%g' % value).replace('-', '\u2212')))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_yticks(y)
    ax.set_yticklabels([_label(name) for name in frame['变量']], linespacing=1.05)
    ax.set_ylim(-0.7, len(frame) - 0.3)
    ax.invert_yaxis()
    ax.set_xlabel('中位数回归系数（元/天）', fontsize=FONTS['axis_label'])
    ax.set_ylabel('代表性变量', fontsize=FONTS['axis_label'])
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.005), ncol=2, frameon=False,
              fontsize=FONTS['legend'], columnspacing=1.8, handletextpad=0.6,
              handlelength=1.8, borderaxespad=0.0)
    plot_style.apply_sci_axis(ax, grid=False)
    fig.subplots_adjust(left=0.30, right=0.985, bottom=0.105, top=0.925)
    return fig, ax


def main() -> int:
    plot_style.setup_sci_style()
    plot_style.FONT_SIZES.update(FONTS)
    plt.rcParams.update({'axes.labelsize': FONTS['axis_label'],
                         'xtick.labelsize': FONTS['tick'],
                         'ytick.labelsize': FONTS['tick']})

    frame = load_representative()
    fig, _ = build_figure(frame)
    diagnostics = figure_finalize.save_paper_figure(
        fig, FIGDIR, STEM, subfigures=[],
        meta={'数据来源': 'outputs/results/quantile_regression_cluster_bootstrap.csv（E7 冻结结果）',
              '口径': '只展示已定义的代表性变量，不做显著性筛选；岗位级 bootstrap 以岗位为重抽样单位，'
                      '公司级 cluster bootstrap 以公司为重抽样单位并带入该公司全部岗位，各 1,000 轮',
              '呈现': (f'单面板森林图：隔行浅底纹 + 带端帽的两类 95% 区间 + 菱形点估计；'
                       f'横轴为 symlog（|系数| ≤ {SYMLOG_LINTHRESH:.0f} 元/天 段线性，'
                       '更大值段对数压缩），极端区间与小区间可同时辨认'),
              '用途': '实验补全报告 E7 图件重绘（图题由载体文档给出）'})

    failed = figure_finalize.failed_paper_gates(diagnostics)
    metrics = project_paths.METRICS_DIR / 'stage_exp_e7_figure_redraw.json'
    metrics.write_text(json.dumps(
        {'stem': STEM, '代表性变量数': int(frame.shape[0]),
         '论文版门禁未通过项': failed,
         'png_pixel_size': diagnostics['png_pixel_size'],
         'png': diagnostics['png_path'], 'pdf': diagnostics['pdf_path']},
        ensure_ascii=False, indent=2), encoding='utf-8')

    print('=' * 78)
    print('图件：', diagnostics['png_path'])
    print('像素：', diagnostics['png_pixel_size'], '| 门禁未通过项：', failed or '无')
    print('打印尺寸：%.2f × %.2f cm' % (PRINT_WIDTH_CM, PRINT_HEIGHT_CM))
    print('=' * 78)
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())

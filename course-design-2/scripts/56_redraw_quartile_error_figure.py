# -*- coding: utf-8 -*-
"""重绘图「测试集按薪资四分位分组的绝对误差与相对误差」（outputs/figures/fig_7_8_...）。

沿用 45 号实验（E3）的冻结结果，只改版式与呈现规范，不改动任何数值：

- 只读 ``outputs/results/E1_E3_E4_E5/test_error_by_salary_quartile.csv``（总体行不进入柱图）；
- 面板构成保持 1×2：左＝绝对误差 MAE，右＝相对误差 nMAE；
- 去掉原图的 ``fig.suptitle`` 与两个 ``ax.set_title``（项目禁用图内标题），
  改为 (a)(b) 子图名置于对应子图下方，总图题交给载体文档；
- 按载体版心 15.5 cm 出图并把字号换算到打印 11~12 pt（原图宽 9.5 in，插到 15.5 cm
  后 9 pt 字号只剩约 5.8 pt）；
- 两面板统一到全文色盲友好色板，并补齐论文版规范：600 dpi PNG ＋ 矢量 PDF、
  四边框、内向刻度、浅虚线网格。

不加总体参照线、柱顶数值、百分比刻度等增强项（按本轮选择保持原呈现）。

用法：:

    python scripts\\56_redraw_quartile_error_figure.py
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

from src import figure_finalize, plot_style, project_paths  # noqa: E402

STEM = 'fig_7_8_error_by_salary_quartile'
CSV = project_paths.RESULTS_E1_E3_E4_E5 / 'test_error_by_salary_quartile.csv'
FIGDIR = project_paths.FIGURES_DIR
CAPTIONS = [('a', '绝对误差 MAE'), ('b', '相对误差 nMAE')]
PRINT_WIDTH_CM = 15.5
PRINT_HEIGHT_CM = 7.4
CM = 1.0 / 2.54
FONT_SCALE = 11.0 / 12.0
FONTS = {'axis_label': 12.0, 'tick': 11.0, 'legend': 11.0, 'annotation': 10.5}
# 冻结值锚点（45 号实验 E3 输出，用于防止读入漂移）
ANCHORS = {'MAE': [35.4336, 22.7183, 29.4076, 63.3567],
           'nMAE': [0.2953, 0.1377, 0.1368, 0.2112]}


def _w(width_cm: float) -> float:
    return width_cm * CM * FONT_SCALE


def load_quartiles() -> pd.DataFrame:
    """读四分位分组误差，并逐项断言与冻结值一致（不含总体行）。"""
    frame = pd.read_csv(CSV)
    body = frame[~frame['分组'].str.startswith('总体')].reset_index(drop=True)
    if len(body) != 4:
        raise AssertionError('四分位分组数不是 4：%d' % len(body))
    for column, expected in ANCHORS.items():
        actual = [round(float(value), 4) for value in body[column]]
        if actual != expected:
            raise AssertionError('%s 与冻结值不符：%s ≠ %s' % (column, actual, expected))
    return body


def build_figure(body: pd.DataFrame):
    """1×2 柱图：左 MAE、右 nMAE，统一色板与论文版坐标区规范。"""
    height = PRINT_HEIGHT_CM / PRINT_WIDTH_CM * _w(PRINT_WIDTH_CM)
    fig, axes = plt.subplots(1, 2, figsize=(_w(PRINT_WIDTH_CM), height))

    positions = np.arange(len(body), dtype='float64')
    panels = [('MAE（元/天）', 'MAE', plot_style.MAIN_COLOR, np.arange(0.0, 80.1, 20.0), 80.0),
              # (b) 的 y 轴名精简为 nMAE：口径（MAE / 组内薪资中位数）已在子图名与载体正文给出
              ('nMAE', 'nMAE', plot_style.ACCENT_COLOR, np.arange(0.0, 0.31, 0.1), 0.36)]
    for ax, (ylabel, column, color, ticks, top) in zip(axes, panels):
        values = body[column].to_numpy('float64')
        ax.bar(positions, values, width=0.62, color=color, edgecolor='black', linewidth=0.5,
               zorder=3)
        ax.set_xticks(positions)
        ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
        ax.set_xlim(-0.62, len(body) - 0.38)
        # 显式给定上限与刻度：留出柱顶余量，又不产生超出坐标区的空刻度
        ax.set_yticks(ticks)
        ax.set_ylim(0.0, top)
        ax.set_xlabel('薪资四分位组', fontsize=FONTS['axis_label'])
        ax.set_ylabel(ylabel, fontsize=FONTS['axis_label'])
        plot_style.apply_sci_axis(ax, grid_axis='y')
    fig.subplots_adjust(left=0.102, right=0.985, bottom=0.26, top=0.955, wspace=0.34)
    return fig, axes


def add_subcaptions(fig, axes) -> None:
    """(a)(b) 子图名置于「横轴刻度标签与横轴名」下沿之下。"""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, (letter, text) in zip(axes, CAPTIONS):
        box = ax.get_window_extent(renderer)
        bottom = box.y0
        for artist in [ax.xaxis.label, *ax.get_xticklabels()]:
            try:
                artist_box = artist.get_window_extent(renderer)
            except Exception:                                     # noqa: BLE001
                continue
            bottom = min(bottom, artist_box.y0)
        y = -((box.y0 - bottom) + 8.0) / max(box.height, 1.0)
        plot_style.add_subfigure_caption(ax, letter, text, y=y)


def main() -> int:
    plot_style.setup_sci_style()
    plot_style.FONT_SIZES.update(FONTS)
    plt.rcParams.update({'axes.labelsize': FONTS['axis_label'],
                         'xtick.labelsize': FONTS['tick'],
                         'ytick.labelsize': FONTS['tick']})

    body = load_quartiles()
    fig, axes = build_figure(body)
    add_subcaptions(fig, axes)
    diagnostics = figure_finalize.save_paper_figure(
        fig, FIGDIR, STEM,
        subfigures=[(letter, text, ax) for (letter, text), ax in zip(CAPTIONS, axes)],
        meta={'数据来源': 'outputs/results/E1_E3_E4_E5/test_error_by_salary_quartile.csv（45 号实验 E3 冻结结果）',
              '口径': '测试集按真实薪资四分位分组；nMAE = 该组 MAE / 组内薪资中位数；'
                      '总体（n = 2,233）不进入柱图，只在组内比较',
              '版式': '去掉图内总标题与子图标题，改为 (a)(b) 子图名下置；'
                      '按版心 %.1f cm 出图，字号换算到打印 11~12 pt；两面板统一全文色板'
                      % PRINT_WIDTH_CM,
              '用途': '实验补全报告 E3 图件重绘（图题由载体文档给出）'})
    failed = figure_finalize.failed_paper_gates(diagnostics)

    metrics = project_paths.METRICS_DIR / 'stage_exp_e3_figure_redraw.json'
    metrics.write_text(json.dumps(
        {'stem': STEM, '分组数': int(body.shape[0]),
         'MAE': [round(v, 4) for v in body['MAE']],
         'nMAE': [round(v, 4) for v in body['nMAE']],
         '论文版门禁未通过项': failed,
         'png_pixel_size': diagnostics['png_pixel_size'],
         'png': diagnostics['png_path'], 'pdf': diagnostics['pdf_path']},
        ensure_ascii=False, indent=2), encoding='utf-8')

    print('=' * 78)
    print('图件：', diagnostics['png_path'])
    print('像素：', diagnostics['png_pixel_size'], '| 门禁未通过项：', failed or '无')
    print('打印尺寸目标：%.1f × %.1f cm' % (PRINT_WIDTH_CM, PRINT_HEIGHT_CM))
    print('=' * 78)
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())

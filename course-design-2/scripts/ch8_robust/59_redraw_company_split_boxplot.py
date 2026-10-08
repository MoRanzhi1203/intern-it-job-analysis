# -*- coding: utf-8 -*-
"""重绘图 8-10「重复公司分组划分下的测试集 MAE 分布」。

只改版式与呈现，不动数值（两条数据序列均与冻结结果逐项断言）：

- 移除图内标题 ``ax.set_title``；
- 移除图内占比式统计文字（「公司分组划分 5 次：均值…标准差…范围…」）——这些数字在
  表 8-6 与正文中已有，图内重复属于冗余；
- 虚线基准线改为一条图例说明（原先没有任何说明，读者无法判断这条线是什么）；
- 箱线图不再重复画离群点（散点已画出全部 5 次划分的取值，原先同一取值会被画两次）；
- 按载体版心 15.5 cm 出图并把字号换算到打印 11~12 pt（原图 6.4 in 宽插到 15.5 cm 后
  9 pt 只剩约 8.6 pt）；
- 配色统一到全文色板，补 600 dpi PNG ＋ 矢量 PDF、四边框、内向刻度、浅虚线网格。

用法：:

    python scripts\\59_redraw_company_split_boxplot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matplotlib import pyplot as plt  # noqa: E402

from src import figure_finalize, plot_style, project_paths  # noqa: E402

STEM = 'fig_8_6_company_split_mae_distribution'
REPEATS_CSV = project_paths.RESULTS_E2_E7 / 'company_group_split_repeats.csv'
SUMMARY_CSV = project_paths.RESULTS_E2_E7 / 'company_group_split_summary.csv'
METRICS_XLSX = project_paths.TABLES_DIR / 'ch7/52_stage26_4_metrics_final.xlsx'
PRINT_WIDTH_CM = 15.5
PRINT_HEIGHT_CM = 8.2
CM = 1.0 / 2.54
FONT_SCALE = 11.0 / 12.0
FONTS = {'axis_label': 12.0, 'tick': 11.0, 'legend': 9.5, 'annotation': 10.5}
CATEGORIES = ['公司分组划分', '随机划分']
# 冻结锚点（company_group_split_repeats.csv / 08_多种子明细）
ANCHOR_COMPANY = [51.983919, 81.463150, 38.831912, 49.441596, 45.737550]
ANCHOR_RANDOM = [35.996, 35.330, 35.884, 35.101, 36.314]
ANCHOR_BASELINE = 35.996481


def _w(width_cm: float) -> float:
    return width_cm * CM * FONT_SCALE


def load_data() -> tuple[np.ndarray, np.ndarray, float]:
    """读两类划分的测试集 MAE 与基准线取值，并逐项断言与冻结值一致。"""
    company = pd.read_csv(REPEATS_CSV)['test MAE'].to_numpy('float64')
    summary = pd.read_csv(SUMMARY_CSV).set_index('指标')['取值']
    # 该行名是冻结 CSV 的取值键，不进入图件文字
    baseline = float(summary.loc['随机划分同口径 MAE 基准'])
    seeds = pd.read_excel(METRICS_XLSX, sheet_name='08_多种子明细')
    random_mae = seeds[seeds['模型'] == 'LightGBM']['test MAE'].to_numpy('float64')
    if [round(float(v), 3) for v in company] != [round(v, 3) for v in ANCHOR_COMPANY]:
        raise AssertionError('公司分组划分取值与冻结值不符：%s' % [round(float(v), 3) for v in company])
    if [round(float(v), 3) for v in random_mae] != [round(v, 3) for v in ANCHOR_RANDOM]:
        raise AssertionError('随机划分取值与冻结值不符：%s' % [round(float(v), 3) for v in random_mae])
    if abs(baseline - ANCHOR_BASELINE) > 1e-6:
        raise AssertionError('基准线与冻结值不符：%.6f' % baseline)
    return company, random_mae, baseline


def build_figure(company: np.ndarray, random_mae: np.ndarray, baseline: float):
    """箱线图 ＋ 全量散点 ＋ 基准线（仅基准线进图例，箱体含义由横轴刻度承担）。"""
    fig, ax = plt.subplots(figsize=(_w(PRINT_WIDTH_CM),
                                    PRINT_HEIGHT_CM / PRINT_WIDTH_CM * _w(PRINT_WIDTH_CM)))
    box = ax.boxplot([company, random_mae], widths=0.42, patch_artist=True,
                     tick_labels=CATEGORIES, showfliers=False,
                     medianprops={'color': 'black', 'linewidth': 1.1},
                     boxprops={'edgecolor': 'black', 'linewidth': 0.7},
                     whiskerprops={'color': 'black', 'linewidth': 0.9},
                     capprops={'color': 'black', 'linewidth': 0.9})
    for patch, color in zip(box['boxes'], [plot_style.MAIN_COLOR, plot_style.ACCENT_COLOR]):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    for index, values in enumerate([company, random_mae], start=1):
        jitter = np.linspace(-0.09, 0.09, len(values))
        ax.scatter(np.full(len(values), index, dtype='float64') + jitter, values, s=22,
                   color='black', linewidths=0.0, zorder=3)
    ax.axhline(baseline, color=plot_style.MUTED_COLOR, linestyle='--', linewidth=1.1, zorder=2)
    ax.set_xlabel('数据划分方式', fontsize=FONTS['axis_label'])
    ax.set_ylabel('测试集 MAE（元/天）', fontsize=FONTS['axis_label'])
    ax.set_xlim(0.45, 2.55)
    ax.legend([plt.Line2D([], [], color=plot_style.MUTED_COLOR, linestyle='--', linewidth=1.1)],
              ['随机划分基准 %.2f' % baseline],
              loc='upper right', frameon=True, framealpha=0.94, facecolor='white',
              edgecolor='#8C8C8C', fancybox=True, fontsize=FONTS['legend'],
              handlelength=1.6, handletextpad=0.6, borderpad=0.45, borderaxespad=0.7)
    plot_style.apply_sci_axis(ax, grid_axis='y')
    fig.subplots_adjust(left=0.098, right=0.985, bottom=0.145, top=0.965)
    return fig, ax


def main() -> int:
    plot_style.setup_sci_style()
    plot_style.FONT_SIZES.update(FONTS)
    plt.rcParams.update({'axes.labelsize': FONTS['axis_label'],
                         'xtick.labelsize': FONTS['tick'],
                         'ytick.labelsize': FONTS['tick']})

    company, random_mae, baseline = load_data()
    fig, ax = build_figure(company, random_mae, baseline)
    diagnostics = figure_finalize.save_paper_figure(
        fig, project_paths.FIGURES_DIR, STEM, subfigures=[],
        meta={'数据来源': 'company_group_split_repeats.csv（5 次公司分组划分）＋ '
                          'ch7/52_stage26_4_metrics_final.xlsx / 08_多种子明细（5 个随机种子）；'
                          '基准线取 company_group_split_summary.csv 的「随机划分同口径 MAE 基准」行'
                          '（该行名沿用冻结 CSV 原样，仅作取值键，不出现在图中）',
              '计算说明': '公司分组划分按公司实体整体划分（训练 2,453 / 验证 526 / 测试 526 家公司），'
                          '随机划分按薪资十分位分层随机划分；两者均为 A+B+C+D+E（288 维）、'
                          '预处理只在训练集拟合、LightGBM 400/0.05/63',
              '版式': '移除图内标题与图内统计文字（均值/标准差/范围见表 8-6）；'
                      '基准线由图例说明；箱线图不重复画离群点（散点已含全部取值）；'
                      '按版心 %.1f cm 出图换算字号' % PRINT_WIDTH_CM,
              '用途': '实验补全报告 E2 图件重绘（图题由载体文档给出）'})
    failed = figure_finalize.failed_paper_gates(diagnostics)

    metrics = project_paths.METRICS_DIR / 'stage_exp_e2_figure_redraw.json'
    metrics.write_text(json.dumps(
        {'stem': STEM,
         '公司分组划分 test MAE': [round(float(v), 3) for v in company],
         '随机划分 test MAE': [round(float(v), 3) for v in random_mae],
         '基准线': round(baseline, 6),
         '图内文字数': len([t for t in ax.texts if t.get_text()]),
         '论文版门禁未通过项': failed, 'png_pixel_size': diagnostics['png_pixel_size'],
         'png': diagnostics['png_path'], 'pdf': diagnostics['pdf_path']},
        ensure_ascii=False, indent=2), encoding='utf-8')

    print('=' * 78)
    print('公司分组划分 MAE：%s' % [round(float(v), 2) for v in company])
    print('随机划分 MAE：%s；基准线 %.4f' % ([round(float(v), 2) for v in random_mae], baseline))
    print('图件：', diagnostics['png_path'])
    print('像素：', diagnostics['png_pixel_size'], '| 门禁未通过项：', failed or '无')
    print('=' * 78)
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())

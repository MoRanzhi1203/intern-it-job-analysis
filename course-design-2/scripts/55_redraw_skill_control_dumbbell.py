# -*- coding: utf-8 -*-
"""重绘图「控制岗位细分类前后的技能薪资差异变化」（outputs/figures/supplementary/图S09）。

改为**哑铃箭头图**：纵轴 6 个技能（按控制后差异降序），横轴为「有技能 − 无技能」的
薪资中点中位数差（元/天），每个技能一条从「未控制」（空心点）指向「控制后」（实心点）
的水平箭头，箭头方向即增减方向。

相对原斜率图的变化：

- 按 docx 实际插入宽度 10.5 cm 出图并把字号换算到打印 11~12 pt（原图物理宽 7.76 in，
  插到 10.5 cm 后字号只剩约 5.1 pt）；
- 删除图内说明框（口径说明移入图注/正文）；
- 删除「技能：+50 → +125.0」式图例，改为右端数值列，避免图例与折线来回对照；
- 两类口径的读数统一保留 1 位小数，行序改为按控制后差异降序；
- 横轴名去掉与纵轴重复的「对比口径」字样与括号。

数据仍取自 29 号表 ``08_技能薪资`` 与 ``09_岗位内技能薪资``：控制后取值按「细分类内
有技能岗位数」加权平均，与原图口径完全一致，并把 6 个技能的两类取值与冻结值逐项断言。

用法：:

    E:\\anaconda3\\envs\\reptile\\python.exe scripts\\55_redraw_skill_control_dumbbell.py
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

STEM = '图S09_控制细分类前后技能薪资差异'
TABLE29 = project_paths.TABLES_DIR / '29_eda_statistical_analysis.xlsx'
OUT_DIR = project_paths.FIGURES_DIR / 'supplementary'
# docx 侧图 6-3 不在 FIG_WIDTH_CM 名单，按默认 10.5 cm 插入
PRINT_WIDTH_CM = 10.5
PRINT_HEIGHT_CM = 8.6
CM = 1.0 / 2.54
FONT_SCALE = 11.0 / 12.0
FONTS = {'axis_label': 12.0, 'tick': 11.0, 'legend': 11.0, 'annotation': 10.5}
# 冻结值锚点（原图 V['skill_salary_map'] 与 V['skill_in_cat_agg']，用于防止重算漂移）
ANCHORS = {'Excel': (-10.0, 0.0), 'Agent': (50.0, 51.9), 'SQL': (50.0, 60.2),
           'Java': (75.0, 93.4), 'Python': (60.0, 122.8), '大模型': (50.0, 125.0)}


def _w(width_cm: float) -> float:
    return width_cm * CM * FONT_SCALE


def load_differences() -> pd.DataFrame:
    """读两类口径的中位数差，并逐项断言与冻结值一致。"""
    overall = pd.read_excel(TABLE29, sheet_name='08_技能薪资').set_index('技能标准名')
    within = pd.read_excel(TABLE29, sheet_name='09_岗位内技能薪资')
    rows = []
    for skill in dict.fromkeys(within['技能标准名'].tolist()):
        block = within[within['技能标准名'] == skill]
        after = float(np.average(block['中位数差'].to_numpy('float64'),
                                 weights=block['有技能岗位数'].to_numpy('float64')))
        before = float(overall.loc[skill, '中位数差'])
        expected = ANCHORS[skill]
        if not (abs(before - expected[0]) < 0.05 and abs(after - expected[1]) < 0.05):
            raise AssertionError('%s 取值与冻结值不符：%.2f/%.2f ≠ %s' % (skill, before, after, expected))
        rows.append({'技能': skill, '未控制': before, '控制后': after})
    frame = pd.DataFrame(rows)
    return frame.sort_values('控制后', ascending=False).reset_index(drop=True)


def build_figure(frame: pd.DataFrame):
    """哑铃箭头图：空心点＝未控制，实心点＝控制后，右端给出两类读数。"""
    height = PRINT_HEIGHT_CM / PRINT_WIDTH_CM * _w(PRINT_WIDTH_CM)
    fig, ax = plt.subplots(figsize=(_w(PRINT_WIDTH_CM), height))

    y = np.arange(len(frame), dtype='float64')
    for row, (_, item) in zip(y, frame.iterrows()):
        before, after = float(item['未控制']), float(item['控制后'])
        ax.annotate('', xy=(after, row), xytext=(before, row),
                    arrowprops={'arrowstyle': '-|>', 'color': plot_style.MAIN_COLOR,
                                'linewidth': 1.6, 'shrinkA': 0.0, 'shrinkB': 0.0},
                    zorder=3)
        ax.plot([before], [row], marker='o', markersize=7.0, markerfacecolor='white',
                markeredgecolor=plot_style.MAIN_COLOR, markeredgewidth=1.4, zorder=4)
        # 控制后用小实心点：变化极小的行（如 Agent +50.0 → +51.9）两点几乎重合，
        # 小点嵌在大空心环里仍能看出「几乎没变」，而不是糊成一个点
        ax.plot([after], [row], marker='o', markersize=4.2, color=plot_style.MAIN_COLOR,
                markeredgecolor='black', markeredgewidth=0.45, zorder=5)
        # 右端数值列：两类读数统一 1 位小数，列对齐在坐标区之外
        ax.text(1.03, row, '%+.1f → %+.1f' % (before, after), transform=ax.get_yaxis_transform(),
                va='center', ha='left', fontsize=FONTS['annotation'], clip_on=False)

    ax.axvline(0, color=plot_style.MUTED_COLOR, linestyle='--', linewidth=0.9, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(frame['技能'].tolist())
    ax.set_xlim(-20, 135)
    ax.set_ylim(-0.7, len(frame) - 0.3)
    ax.invert_yaxis()
    ax.set_xlabel('有技能 − 无技能的薪资中点中位数差（元/天）', fontsize=FONTS['axis_label'])
    ax.set_ylabel('技能', fontsize=FONTS['axis_label'])
    plot_style.apply_sci_axis(ax, grid_axis='x')
    handles = [plt.Line2D([], [], marker='o', linestyle='none', markersize=7.0,
                          markerfacecolor='white', markeredgecolor=plot_style.MAIN_COLOR,
                          markeredgewidth=1.4),
               plt.Line2D([], [], marker='o', linestyle='none', markersize=4.2,
                          color=plot_style.MAIN_COLOR, markeredgecolor='black',
                          markeredgewidth=0.45)]
    ax.legend(handles, ['未控制岗位细分类', '控制岗位细分类后'],
              loc='lower center', bbox_to_anchor=(0.5, 1.005), ncol=2, frameon=False,
              fontsize=FONTS['legend'], columnspacing=1.8, handletextpad=0.6,
              borderaxespad=0.0)
    fig.subplots_adjust(left=0.245, right=0.72, bottom=0.135, top=0.86)
    return fig, frame


def main() -> int:
    plot_style.setup_sci_style()
    plot_style.FONT_SIZES.update(FONTS)
    plt.rcParams.update({'axes.labelsize': FONTS['axis_label'],
                         'xtick.labelsize': FONTS['tick'],
                         'ytick.labelsize': FONTS['tick']})

    frame = load_differences()
    fig, _ = build_figure(frame)
    diagnostics = figure_finalize.save_paper_figure(
        fig, OUT_DIR, STEM, subfigures=[],
        meta={'数据来源': '29_eda_statistical_analysis.xlsx / 08_技能薪资、09_岗位内技能薪资',
              '口径': '控制后取值为同一岗位细分类内部对比、并按细分类内有技能岗位数加权平均；'
                      '虚线为零差异参考线；未控制口径即表 6-1 的整体中位数差',
              '呈现': '哑铃箭头图：空心点＝未控制岗位细分类，实心点＝控制岗位细分类后，'
                      '箭头方向为增减方向；右端数值列给出两类读数（统一 1 位小数）',
              '版式': '按 docx 插入宽度 %.1f cm 出图，字号换算到打印 11~12 pt；'
                      '已删除原图内说明框与「技能：+50 → +125.0」式图例'
                      % PRINT_WIDTH_CM,
              '用途': '第6章「控制岗位细分类前后的技能薪资差异变化」重绘（图题由 Word 构建脚本生成）'})
    failed = figure_finalize.failed_paper_gates(diagnostics)

    metrics = project_paths.METRICS_DIR / 'stage_23_figure_redraw_s09.json'
    metrics.write_text(json.dumps(
        {'stem': STEM, '技能数': int(frame.shape[0]),
         '对照': frame.assign(变化=frame['控制后'] - frame['未控制']).round(1).to_dict('records'),
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

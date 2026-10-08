# -*- coding: utf-8 -*-
"""重绘图 S44「岗位细分类技能命中率热力图」：移除图内「主口径」字样。

S44 由 ``scripts/26e_stage26_3_figure_refresh.py`` 在 2026-09-19 生成，其 x 轴标题写作
「具体技术技能（主口径命中率前 20）」；其后 ``scripts/18a_redraw_eda_modeling_figures.py``
的 x 轴标题已改为「具体技术技能（命中率前 20）」，但 S44 未随之重绘，因此图内残留「主口径」。

本脚本复用 26e 的版式重映射（``_apply_mode`` / ``save``）与 18a 的绘图函数，按**与 S44 完全
相同的版式参数**（single，打印宽 15.5 cm，面板高 5.2，left 0.36 / right 0.98 / bottom 0.30 /
top 0.97）重绘，只改变 x 轴标题文本，不改动数据、行列选择、配色与色标。

用法：E:\\anaconda3\\envs\\reptile\\python.exe scripts\\65_redraw_figure_s44_heatmap.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
import numpy as np  # noqa: E402

from src import plot_style  # noqa: E402

STEM = '图S44_岗位细分类技能命中率热力图'
# 打印宽 15.5 cm（与论文版心一致）；面板高由 5.2 in 拉长到 6.0 in（图更高）
DISPLAY_CM = 15.5
PANEL_HEIGHT_IN = 6.0
MODE_ARGS = ('single', DISPLAY_CM, PANEL_HEIGHT_IN,
             {'left': 0.36, 'right': 0.98, 'bottom': 0.30, 'top': 0.97})
# x 轴刻度标签（技能名）旋转角度：与 18a / 图 S63 一致
XTICK_ROTATION = 60
# 色条名称（去掉原「（比例）」后缀）与刻度步长
COLORBAR_LABEL = '细分类内部技能命中率'
COLORBAR_STEP = 0.05


def _stretch_colorbar(fig) -> dict:
    """把色条高度拉到与热力图坐标区等高（保留其原有宽度与水平间距）。"""
    heat = next(axis for axis in fig.axes if axis.get_label() != '<colorbar>')
    colorbar = next(axis for axis in fig.axes if axis.get_label() == '<colorbar>')
    box = heat.get_position()
    cbox = colorbar.get_position()
    gap = max(cbox.x0 - box.x1, 0.0)
    width = cbox.width
    colorbar.set_box_aspect(None)
    colorbar.set_aspect('auto')
    colorbar.set_position([box.x1 + gap, box.y0, width, box.height])
    return {'色条高度': round(box.height, 6), '色条宽度': round(width, 6),
            '色条与坐标区间距': round(gap, 6)}


def _percent_colorbar(fig) -> dict:
    """色条刻度改为百分号刻度，并把名称里的「（比例）」去掉。"""
    heat = next(axis for axis in fig.axes if axis.get_label() != '<colorbar>')
    colorbar = next(axis for axis in fig.axes if axis.get_label() == '<colorbar>')
    upper = float(heat.images[0].get_clim()[1])
    count = int(np.floor(upper / COLORBAR_STEP + 1e-9))
    ticks = [index * COLORBAR_STEP for index in range(count + 1)]
    colorbar.set_yticks(ticks)
    colorbar.set_yticklabels(['%.0f%%' % (value * 100) for value in ticks])
    colorbar.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    colorbar.set_ylabel(COLORBAR_LABEL, fontsize=plot_style.FONT_SIZES['axis_label'])
    return {'色条名': COLORBAR_LABEL, '色条上限（数据最大值）': round(upper, 6),
            '色标刻度': ['%.0f%%' % (value * 100) for value in ticks]}


def _load(alias: str, relative: str):
    spec = importlib.util.spec_from_file_location(alias, str(PROJECT_ROOT / relative))
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    # 26e 在导入时安装版式重映射（plt.subplots / Figure.subplots_adjust）
    g = _load('_s65_26e', 'scripts/26e_stage26_3_figure_refresh.py')
    a18 = _load('_s65_18a', 'scripts/18a_redraw_eda_modeling_figures.py')

    plot_style.setup_sci_style()
    plot_style.FONT_SIZES.update(g.TARGET_FONTS)
    plt.rcParams.update({
        'font.size': g.TARGET_FONTS['tick'],
        'axes.labelsize': g.TARGET_FONTS['axis_label'],
        'axes.titlesize': g.TARGET_FONTS['axis_label'],
        'xtick.labelsize': g.TARGET_FONTS['tick'],
        'ytick.labelsize': g.TARGET_FONTS['tick'],
        'legend.fontsize': g.TARGET_FONTS['legend'],
    })
    plot_style.SCI_FIGURES_DIR = g.SUPP_DIR
    g.SUPP_DIR.mkdir(parents=True, exist_ok=True)

    g._apply_mode(*MODE_ARGS)
    fig, subfigures, meta = a18.build_07_category_skill_heatmap({})

    # ---- x 轴刻度标签（技能名，60° 竖排）向右移动：改用 anchor 旋转模式 ----
    # 默认旋转模式下 matplotlib 是把「旋转后包围盒的右沿」对齐到刻度，竖排标签因此整体左移；
    # rotation_mode='anchor' 改为绕标签末端旋转，使标签末端落在各自列的刻度上（与图 S63 一致）。
    ax = fig.axes[0]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tick_boxes = [text.get_window_extent(renderer) for text in ax.get_xticklabels()]
    labels = [str(text.get_text()) for text in ax.get_xticklabels()]
    before_x = [box.x1 for box in tick_boxes]
    ax.set_xticklabels(labels, rotation=XTICK_ROTATION, ha='right', rotation_mode='anchor')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    after_x = [text.get_window_extent(renderer).x1 for text in ax.get_xticklabels()]
    shifts = [after - before for after, before in zip(after_x, before_x)]
    print('x 轴刻度标签数：%d；右移像素（120 dpi 画布）：最小 %.1f 最大 %.1f 平均 %.1f'
          % (len(labels), min(shifts), max(shifts),
             sum(shifts) / len(shifts)))

    # ---- 色条：名称去「（比例）」、高度与热力图齐平、刻度用百分号 ----
    colorbar_box = _stretch_colorbar(fig)
    colorbar_info = _percent_colorbar(fig)
    heat = next(axis for axis in fig.axes if axis.get_label() != '<colorbar>')
    bar = next(axis for axis in fig.axes if axis.get_label() == '<colorbar>')
    print('色条：高度 %.4f（与热力图 %.4f 齐平=%s）；名称「%s」；刻度 %s'
          % (bar.get_position().height, heat.get_position().height,
             abs(bar.get_position().height - heat.get_position().height) < 1e-9,
             colorbar_info['色条名'], '、'.join(colorbar_info['色标刻度'])))

    diagnostics = g.save(STEM, fig, subfigures, {
        **meta,
        'x 轴刻度标签': '技能名 %d° 竖排，rotation_mode="anchor"，使标签末端锚定在各自列上'
                        '（原为旋转后按包围盒右沿对齐，标签整体偏左）' % XTICK_ROTATION,
        '色标': '名称「%s」（去掉原「（比例）」后缀）；刻度为百分号刻度 %s；'
                '色条高度与热力图坐标区齐平（%s）'
                % (colorbar_info['色条名'], '、'.join(colorbar_info['色标刻度']), colorbar_box),
        '用途': '第6章 图 6-2 放大版（版式同前，仅移除图内「主口径」字样后重绘）'})
    print('PNG:', diagnostics['png_path'])
    print('PDF:', diagnostics['pdf_path'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

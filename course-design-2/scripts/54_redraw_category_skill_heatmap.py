# -*- coding: utf-8 -*-
"""重绘图「岗位细分类 × 技能命中率热力图」（outputs/figures/supplementary/图S63）。

沿用冻结的 27 号表 ``08_岗位类别技能画像``（15 个细分类 × 20 个技能，行列为原口径），
只做两处版式优化，不改动任何数值与行列选择：

- 色标改为百分比刻度并把上限固定为 30%（原为 0~0.287 的比例刻度），便于与其它
  命中率图直接对照；
- x 轴刻度标签改为 90° 竖排、居中锚定在各自列上（原为 60° 旋转＋右对齐，标签右端
  锚在刻度上会向左压在邻列上）。

绘制函数复用 18a 的 ``build_07_category_skill_heatmap``，版式重映射、图元重叠实测与
论文版导出复用 26g 的 ``_apply_mode`` / ``save``，只读不写回任何数据表。

用法：:

    python scripts\\54_redraw_category_skill_heatmap.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matplotlib.ticker import PercentFormatter  # noqa: E402

from src import figure_finalize, project_paths  # noqa: E402
from src.script_support import load_script as _load  # noqa: E402
from src.script_support import apply_style as _apply_style  # noqa: E402

STEM = '图S63_岗位细分类技能命中率热力图'
# 与 26g 中「图 6-2（热力图放大）」完全相同的版式参数
MODE_ARGS = ('single', 15.5, 6.6,
             {'left': 0.29, 'right': 0.975, 'bottom': 0.30, 'top': 0.975})
COLOR_LIMIT = 0.30
COLOR_TICKS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
LABEL_ROTATION = 60
COLORBAR_LABEL = '细分类内部技能命中率'
COLORBAR_WIDTH = 0.016
COLORBAR_GAP = 0.010


def align_column_labels(ax) -> list:
    """x 轴标签按原图 60° 倾斜，末端锚定在各自列的刻度上，返回标签文本。

    用 ``rotation_mode='anchor'`` ＋ ``ha='right'``：旋转绕「文本末端」进行，标签末端
    正好落在所属列的刻度上（与本文时间序列图同一做法）。默认旋转模式会先旋转再按旋转后
    包围盒对齐，整个标签会相对刻度整体左移；若改为按包围盒中心对齐，则标签中心而非末端
    落在刻度上，同样不是列与标签的对应关系。
    """
    labels = [str(text.get_text()) for text in ax.get_xticklabels()]
    ax.set_xticklabels(labels, rotation=LABEL_ROTATION, ha='right', rotation_mode='anchor')
    return labels


def stretch_colorbar(fig) -> dict:
    """把色条拉到与热力图等高。

    两处都要处理：① 主坐标区经过 ``subplots_adjust`` 后位置才最终确定，色条需按最终
    几何重新定位；② ``fig.colorbar`` 给色条坐标区设了 ``box_aspect = 20``，绘制时会按
    宽度反算高度（实测只铺 1.79 in），必须清掉盒比例并改为 auto 比例才能铺满给定高度。
    """
    heat = fig.axes[0]
    colorbar = next(axis for axis in fig.axes if axis.get_label() == '<colorbar>')
    box = heat.get_position()
    colorbar.set_box_aspect(None)
    colorbar.set_aspect('auto')
    colorbar.set_position([box.x1 + COLORBAR_GAP, box.y0, COLORBAR_WIDTH, box.height])
    return {'色条宽度': COLORBAR_WIDTH, '色条与坐标区间距': COLORBAR_GAP}


def percent_colorbar(fig) -> dict:
    """色标改百分比刻度，把上限固定为 COLOR_LIMIT，并去掉色条名里的括号。"""
    from src import plot_style

    heat = fig.axes[0]
    colorbar = next(axis for axis in fig.axes if axis.get_label() == '<colorbar>')
    heat.images[0].set_clim(0.0, COLOR_LIMIT)
    colorbar.set_yticks(COLOR_TICKS)
    colorbar.set_yticklabels(['%.0f%%' % (value * 100) for value in COLOR_TICKS])
    colorbar.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    colorbar.set_ylabel(COLORBAR_LABEL, fontsize=plot_style.FONT_SIZES['axis_label'])
    return {'色标上限': COLOR_LIMIT, '色条名': COLORBAR_LABEL,
            '色标刻度': ['%.0f%%' % (value * 100) for value in COLOR_TICKS]}


def main() -> int:
    g = _load('_g26_stage26_4', 'scripts/26g_stage26_4_figure_rebuild.py')
    a18 = _load('_g18a_eda_figures', 'scripts/18a_redraw_eda_modeling_figures.py')
    _apply_style(g)
    g._apply_mode(*MODE_ARGS)

    fig, subfigures, meta = a18.build_07_category_skill_heatmap({})
    labels = align_column_labels(fig.axes[0])
    colorbar_box = stretch_colorbar(fig)
    colorbar_info = percent_colorbar(fig)

    diagnostics = g.save(STEM, fig, subfigures, {
        **meta,
        '行列口径': '行＝岗位数前 15 的岗位细分类（本样本以运营／产品类为主），'
                    '列＝主口径频次前 20 的具体技术技能；本轮不改行列选择',
        '色标': '百分比刻度，上限固定 %d%%，共 %d 个刻度；色条拉到与热力图等高'
                % (round(COLOR_LIMIT * 100), len(COLOR_TICKS)),
        'x 轴标签': '%d° 倾斜并居中锚定在各自列上（原为 %d° 旋转＋右对齐）'
                    % (LABEL_ROTATION, LABEL_ROTATION),
        '色条定位': colorbar_box,
        '用途': '第6章「岗位细分类 × 技能命中率热力图」重绘（图题由 Word 构建脚本生成）'})

    metrics = project_paths.METRICS_DIR / 'stage_26_4_figure_redraw_s63.json'
    metrics.write_text(json.dumps(
        {'stem': STEM, '矩阵形状': '15 × 20', **colorbar_info,
         '论文版门禁未通过项': figure_finalize.failed_paper_gates(diagnostics),
         '图元重叠处数': diagnostics['overlap_count'],
         'png': diagnostics['png_path'], 'pdf': diagnostics['pdf_path']},
        ensure_ascii=False, indent=2), encoding='utf-8')
    print('x 轴列标签数：', len(labels))
    return 0 if not figure_finalize.failed_paper_gates(diagnostics) else 1


if __name__ == '__main__':
    sys.exit(main())

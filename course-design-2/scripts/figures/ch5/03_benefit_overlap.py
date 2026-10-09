# -*- coding: utf-8 -*-
"""重绘图「高效应福利标签的共现结构」（outputs/figures/supplementary/图S62）。

只做版式优化，数据、图形元素与数值与 Stage26.4 版本一致：

- 两个子图之间的竖向间距缩小（hspace 0.70 → 0.45），版面更紧凑、坐标区更高；
- (b) 图保持完整 5×5 对称矩阵（含对角与上三角的 25 个 Jaccard 值），不做三角隐藏。

数据仍来自 29 号表 ``11_统计检验（标签块）`` 的锚点取值：绘制函数复用 02_supplementary_figures 的
``tag_overlap_bars`` / ``tag_jaccard_heatmap``，版式重映射、图元重叠实测与论文版导出
复用 07_figure_rebuild 的 ``_apply_mode`` / ``save``，只读不写回任何数据表。

用法：:

    python scripts\\03_benefit_overlap.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = next(
    _candidate for _candidate in Path(__file__).resolve().parents
    if (_candidate / "data").is_dir() and (_candidate / "scripts").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import figure_finalize, project_paths  # noqa: E402
from src.script_support import load_script as _load  # noqa: E402
from src.script_support import apply_style as _apply_style  # noqa: E402

STEM = 'fig_5_6_high_effect_benefit_label_cluster'
CAPTIONS = [('a', '高效应福利标签、交集与并集的岗位规模'),
            ('b', '高效应福利标签两两 Jaccard 相似度')]
# 版式与 07_figure_rebuild 一致，仅把子图间距 hspace 由 0.70 收紧到 0.45
MODE_ARGS = ('stack', 15.5, 2.9,
             {'left': 0.235, 'right': 0.975, 'bottom': 0.135, 'top': 0.925, 'hspace': 0.45})


def heatmap_cells(ax) -> int:
    """返回 (b) 图当前展示的 Jaccard 格子数。"""
    return len([text for text in ax.texts
                if re.match(r'^\d\.\d{3}$', str(text.get_text()))])


def main() -> int:
    from matplotlib import pyplot as plt

    from src import plot_style

    g = _load('_g26_stage26_4', 'scripts/ch4_lifecycle/07_figure_rebuild.py')
    b18 = _load('_b18_supplementary', 'scripts/figures/base/02_supplementary_figures.py')
    _apply_style(g)
    g.setup_supplementary_figures(b18)
    # 与 07_figure_rebuild 一致：最大两两 Jaccard 说明框已移入正文，此处不再绘制
    b18.note = lambda *args, **kwargs: None
    g._apply_mode(*MODE_ARGS)

    fig, axes = plt.subplots(2, 1, figsize=(5.85, 5.8))
    b18.tag_overlap_bars(axes[0])
    b18.tag_jaccard_heatmap(axes[1])
    for (letter, caption), ax in zip(CAPTIONS, axes):
        plot_style.add_subfigure_caption(ax, letter, caption)
    fig.subplots_adjust()

    cells = heatmap_cells(axes[1])
    diagnostics = g.save(STEM, fig, [(letter, caption, ax)
                                     for (letter, caption), ax in zip(CAPTIONS, axes)],
                         {'数据来源': 'ch4/21_eda_statistical_analysis.xlsx / 11_统计检验（标签块）',
                          '图内文字': '已删除最大两两 Jaccard 说明框',
                          '热力图': '完整 5×5 对称矩阵，共 %d 个 Jaccard 值' % cells,
                          '版式': '子图间距 hspace 0.70 → 0.45',
                          '用途': '第5章「高效应福利标签的共现结构」重绘（图题由 Word 构建脚本生成）'})

    metrics = project_paths.METRICS_DIR / 'stage_26_4_figure_redraw_s62.json'
    metrics.write_text(json.dumps(
        {'stem': STEM, '热力图格子数': cells,
         '论文版门禁未通过项': figure_finalize.failed_paper_gates(diagnostics),
         '图元重叠处数': diagnostics['overlap_count'],
         'png': diagnostics['png_path'], 'pdf': diagnostics['pdf_path']},
        ensure_ascii=False, indent=2), encoding='utf-8')
    return 0 if not figure_finalize.failed_paper_gates(diagnostics) else 1


if __name__ == '__main__':
    sys.exit(main())

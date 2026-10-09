# -*- coding: utf-8 -*-
"""重绘图「公司认证状态的薪资分布与组间比较」（outputs/figures/supplementary/图S60）。

只做括号与冗余文字清理，图形元素、数值与版式与 Stage26.4 版本完全一致：

- (b) 图横轴名由 ``|Cliff's δ|（效应量）`` 改为 ``组间效应量 |Cliff's δ|``；
- (b) 图图例由 ``校正后显著（q < 0.05）`` 改为 ``校正后显著 q < 0.05``；
- (a) 图横轴刻度由 ``无认证`` + ``(n = 9,104)`` 改为 ``无认证`` + ``n = 9,104``；
- 单位括号（``薪资中点（元/天）``）按约定保留。

数据仍来自冻结的统计结果（18b 内的锚点取值，来源为 29 号表的
``06_公司因素薪资`` 与 ``11_统计检验``）：绘制函数复用 18b 的
:func:`fig_s04`，版式重映射、图元重叠实测与论文版导出复用 26g 的
``_apply_mode`` / ``save``，只读不写回任何数据表。

用法：:

    python scripts\\51_redraw_certification_salary_figure.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import figure_finalize, project_paths  # noqa: E402
from src.script_support import load_script as _load  # noqa: E402
from src.script_support import apply_style as _apply_style  # noqa: E402

STEM = 'fig_5_5_certification_salary_distribution'
# 与 26g 中「图 5-2（1×2 → 2×1，删除统计注释框）」完全相同的版式参数
MODE_ARGS = ('stack', 15.5, 2.85,
             {'left': 0.30, 'right': 0.975, 'bottom': 0.145, 'top': 0.90, 'hspace': 0.72})


def clean_labels(fig) -> list:
    """去掉三处括号写法，返回改动清单（只改文字，不动任何图形元素）。

    (a) 图刻度标签由 ``set_xticklabels`` 设置，刻度的 FixedFormatter 会在每次绘制时
    按存储的字符串重新生成标签，因此必须整体回写刻度标签列表，不能只改 Text 对象。
    """
    ax_a, ax_b = fig.axes[0], fig.axes[1]
    changed = []

    labels = []
    for label in ax_a.get_xticklabels():
        content = str(label.get_text())
        replaced = re.sub(r'\(n = ([\d,]+)\)', r'n = \1', content)
        labels.append(replaced)
        if replaced != content:
            changed.append('%s → %s' % (content.replace('\n', ' / '),
                                        replaced.replace('\n', ' / ')))
    ax_a.set_xticklabels(labels)

    content = str(ax_b.get_xlabel())
    ax_b.set_xlabel("组间效应量 |Cliff's δ|")
    changed.append('%s → %s' % (content, ax_b.get_xlabel()))

    # (b) 图横轴收紧到数据与数值标签刚好容纳的范围，去掉右侧无对应数据的空白
    delta_max = max(float(line.get_xdata()[0]) for line in ax_b.lines)
    previous_limit = float(ax_b.get_xlim()[1])
    ax_b.set_xlim(0.0, delta_max * 1.25)
    changed.append('(b) 横轴上限 %.3f → %.3f' % (previous_limit, ax_b.get_xlim()[1]))

    legend = ax_b.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            content = str(text.get_text())
            replaced = content.replace('（q < 0.05）', ' q < 0.05')
            if replaced != content:
                text.set_text(replaced)
                changed.append('%s → %s' % (content, replaced))
    return changed


def main() -> int:
    g = _load('_g26_stage26_4', 'scripts/ch4_lifecycle/26g_stage26_4_figure_rebuild.py')
    b18 = _load('_b18_supplementary', 'scripts/figures_base/18b_supplementary_figures.py')
    _apply_style(g)
    g.setup_18b(b18)
    # 与 26g 一致：ε² 统计注释框已移入正文，此处不再绘制
    b18.note = lambda *args, **kwargs: None
    g._apply_mode(*MODE_ARGS)

    fig, subfigures = b18.fig_s04(0)
    g.round_labels(fig, 3)
    changed = clean_labels(fig)
    diagnostics = g.save(STEM, fig, subfigures, {
        '数据来源': 'ch4/21_eda_statistical_analysis.xlsx / 06_公司因素薪资、11_统计检验',
        '图内文字': '已删除 ε² 统计注释框（统计范围与数值移入正文表 5-1）',
        '文字清理': changed,
        '用途': '第5章「公司认证状态的薪资分布与组间比较」重绘（图题由 Word 构建脚本生成）'})

    metrics = project_paths.METRICS_DIR / 'stage_26_4_figure_redraw_s60.json'
    metrics.write_text(json.dumps(
        {'stem': STEM, '文字清理': changed,
         '论文版门禁未通过项': figure_finalize.failed_paper_gates(diagnostics),
         '图元重叠处数': diagnostics['overlap_count'],
         'png': diagnostics['png_path'], 'pdf': diagnostics['pdf_path']},
        ensure_ascii=False, indent=2), encoding='utf-8')
    return 0 if not figure_finalize.failed_paper_gates(diagnostics) else 1


if __name__ == '__main__':
    sys.exit(main())

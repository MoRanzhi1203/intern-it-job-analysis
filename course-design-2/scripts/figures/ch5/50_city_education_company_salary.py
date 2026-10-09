# -*- coding: utf-8 -*-
"""重绘图「城市、学历与公司规模的薪资中点中位数」（outputs/figures/supplementary/图S59）。

只做两处调整，图形元素与数值与 Stage26.4 版本完全一致：

1. 清理括号与冗余文字：纵轴名由「城市（取值）」改为「城市」，子图名由
   「(a) 城市薪资中点中位数」改为「(a) 城市」——「薪资中点中位数」已在总图题中给出；
2. 三个子图统一横轴量程，柱长可跨子图比较（原为各子图各自取 (0, 本子图最大中位数 × 1.22)）。

数据仍来自 29 号表的 ``04_城市薪资``、``05_学历薪资``、``06_公司因素薪资``：绘制函数复用
18a 的 :func:`build_04_structured_factor_salary`，版式重映射、图元重叠实测与论文版导出
复用 26g 的 ``_apply_mode`` / ``save``，只读不写回任何数据表。

用法：:

    python scripts\\50_city_education_company_salary.py
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
from src.script_support import apply_style as _apply_style  # noqa: E402
from src.script_support import load_script as _load  # noqa: E402

STEM = 'fig_5_4_city_education_company_size_salary'
PANELS = ['城市', '学历要求', '公司规模']
# 与 26g 中「图 5-1（1×3 → 3×1）」完全相同的版式参数
MODE_ARGS = ('stack', 10.5, 2.15,
             {'left': 0.20, 'right': 0.975, 'bottom': 0.115, 'top': 0.955, 'hspace': 0.72})


def clean_labels_and_unify_axis(fig) -> float:
    """清理纵轴名与子图名中的括号与冗余文字，并把三个子图统一到同一横轴量程。"""
    limits = [float(max(patch.get_width() for patch in ax.patches)) for ax in fig.axes]
    span = max(limits) * 1.22
    for ax, label in zip(fig.axes, PANELS):
        ax.set_xlim(0.0, span)
        ax.set_ylabel(label)
        for text in ax.texts:
            matched = re.match(r'^\(([a-z])\)\s', str(text.get_text()).strip())
            if matched:
                text.set_text('(%s) %s' % (matched.group(1), label))
    return span


def main() -> int:
    g = _load('_g26_stage26_4', 'scripts/ch4_lifecycle/26g_figure_rebuild.py')
    a18 = _load('_g18a_eda_figures', 'scripts/figures/base/18a_eda_modeling_figures.py')
    _apply_style(g)
    g._apply_mode(*MODE_ARGS)

    fig, _, meta = a18.build_04_structured_factor_salary({})
    span = clean_labels_and_unify_axis(fig)
    subfigures = [(chr(ord('a') + index), label, ax)
                  for index, (label, ax) in enumerate(zip(PANELS, fig.axes))]
    diagnostics = g.save(STEM, fig, subfigures, {
        **meta,
        '纵轴名': '已去掉「（取值）」后缀，只保留因素名',
        '子图名': '已去掉总图题中已有的「薪资中点中位数」重复文字',
        '横轴量程': '三个子图统一为 0~%.1f 元/天，柱长可跨子图比较' % span,
        '用途': '第5章「城市、学历与公司规模的薪资中点中位数」重绘（图题由 Word 构建脚本生成）'})

    metrics = project_paths.METRICS_DIR / 'stage_26_4_figure_redraw_s59.json'
    metrics.write_text(json.dumps(
        {'stem': STEM, '横轴量程上限': round(span, 2),
         '论文版门禁未通过项': figure_finalize.failed_paper_gates(diagnostics),
         '图元重叠处数': diagnostics['overlap_count'],
         'png': diagnostics['png_path'], 'pdf': diagnostics['pdf_path']},
        ensure_ascii=False, indent=2), encoding='utf-8')
    return 0 if not figure_finalize.failed_paper_gates(diagnostics) else 1


if __name__ == '__main__':
    sys.exit(main())

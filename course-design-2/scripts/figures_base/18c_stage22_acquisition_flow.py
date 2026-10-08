# -*- coding: utf-8 -*-
"""图 S16：数据采集总体流程（论文正文图 3-1，Stage23 收口版）。

定位与约束
----------
- 只新增一张流程示意图，不改动任何统计结果、模型与既有图件；
- 绘图风格唯一来源 ``src/plot_style.py``（本脚本只读该模块，不修改）；
- 纯流程框图：关闭坐标轴，图名由底部总图题机制负责
  （禁止 ``ax.set_title`` / ``fig.suptitle``）；
- **双输出路径（Stage23）**：论文版 ``fig_3_3_acquisition_flow.png`` 不含图内总图题
  （总图题交给 Word Caption）；展示版 ``fig_s16_acquisition_flow_display.png`` 底部带
  Stage23 正式图题「图 3-1 数据采集总体流程」；两者均为 600 dpi PNG + 矢量 PDF；
- 原始观测规模从只读原始数据文件的元信息读取并校验，不硬编码；
- 只写 ``outputs/figures/supplementary/``。

运行::

    python scripts\\18c_stage22_acquisition_flow.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import figure_finalize  # noqa: E402
from src import plot_style  # noqa: E402
from src import project_paths  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

SUPP_DIR = project_paths.FIGURES_DIR / 'supplementary'
STEM = 'fig_3_3_acquisition_flow'
CAPTION = '图 3-1 数据采集总体流程'
EXPECTED_RAW_ROWS = 172063

PALETTE = plot_style.PALETTE
FS = plot_style.FONT_SIZES
BLUE, ORANGE, GREEN = PALETTE[0], PALETTE[1], PALETTE[2]

# ---- 版面参数（数据坐标，xlim 0~10 / ylim 0.9~7.0）----
COL_CENTERS = (1.55, 5.0, 8.45)
ROW_CENTERS = (5.90, 4.00, 2.10)
NODE_W, NODE_H = 2.70, 1.15
ARROW_COLOR = '#4d4d4d'

# 节点按阅读顺序排布（蛇形：第 1 行左→右，第 2 行右→左，第 3 行左→右）
NODES = [
    ('实习僧“互联网 IT”菜单', BLUE),
    ('菜单任务构建', BLUE),
    ('标准化搜索 URL', BLUE),
    ('列表页抓取', ORANGE),
    ('详情页抓取', ORANGE),
    ('字段解析', ORANGE),
    ('URL 规范化与\n联合键去重', GREEN),
    ('MySQL 结构化存储', GREEN),
    ('原始岗位观测\n（{rows} 条）', GREEN),
]

LEGEND_ITEMS = [
    ('采集入口与任务构建', BLUE),
    ('页面抓取与字段解析', ORANGE),
    ('去重、存储与原始观测形成', GREEN),
]

# 蛇形顺序下每个节点的网格坐标（行, 列）
GRID = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0), (2, 0), (2, 1), (2, 2)]
# 相邻节点之间的箭头走向：> 向右、< 向左、v 向下
MOVES = ['>', '>', 'v', '<', '<', 'v', '>', '>']


def raw_row_count() -> int:
    """从只读原始数据文件的元信息读取观测行数（不读取数据内容）。"""
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(project_paths.RAW_PARQUET).metadata.num_rows)


def node_xy(row: int, col: int) -> tuple:
    return COL_CENTERS[col], ROW_CENTERS[row]


def draw_flow(ax, rows_label: str) -> None:
    for index, (label, color) in enumerate(NODES):
        row, col = GRID[index]
        x, y = node_xy(row, col)
        box = FancyBboxPatch(
            (x - NODE_W / 2, y - NODE_H / 2), NODE_W, NODE_H,
            boxstyle='round,pad=0.02,rounding_size=0.14',
            linewidth=0.8, edgecolor='black', facecolor=color, alpha=0.88,
            mutation_aspect=1.0, zorder=2)
        ax.add_patch(box)
        ax.text(x, y, label.format(rows=rows_label), ha='center', va='center',
                fontsize=FS['tick'] + 0.4, linespacing=1.45, zorder=3)

    for index, move in enumerate(MOVES):
        row, col = GRID[index]
        nxt_row, nxt_col = GRID[index + 1]
        x0, y0 = node_xy(row, col)
        x1, y1 = node_xy(nxt_row, nxt_col)
        if move == '>':
            start, end = (x0 + NODE_W / 2, y0), (x1 - NODE_W / 2, y1)
        elif move == '<':
            start, end = (x0 - NODE_W / 2, y0), (x1 + NODE_W / 2, y1)
        else:
            start, end = (x0, y0 - NODE_H / 2), (x1, y1 + NODE_H / 2)
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle='-|>', mutation_scale=9, linewidth=0.9,
            color=ARROW_COLOR, shrinkA=0.0, shrinkB=0.0, zorder=1))


def build_figure(rows_label: str):
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.115, top=0.845)
    draw_flow(ax, rows_label)
    ax.set_xlim(0, 10)
    ax.set_ylim(0.9, 7.0)
    ax.set_axis_off()
    handles = [Patch(facecolor=color, edgecolor='black', linewidth=0.8, alpha=0.88,
                     label=text) for text, color in LEGEND_ITEMS]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.005),
              ncol=3, frameon=False, fontsize=FS['legend'], handlelength=1.5,
              columnspacing=1.4, borderaxespad=0.0)
    return fig


    plot_style.setup_sci_style()
    plot_style.setup_sci_style()
    plot_style.SCI_FIGURES_DIR = SUPP_DIR
    SUPP_DIR.mkdir(parents=True, exist_ok=True)
    plot_style.FIGURE_REGISTRY.clear()

    anchors = figure_finalize.validate_anchors()
    if not figure_finalize.print_anchor_validation(anchors):
        print('Stage23 锚点校验未通过：按约束停止出图。')
        return 2

    rows = raw_row_count()
    assert rows == EXPECTED_RAW_ROWS, f'原始观测行数与正式口径不一致：{rows}'

    fig = build_figure(f'{rows:,}')
    paper = figure_finalize.save_paper_figure(
        fig, SUPP_DIR, STEM, subfigures=[],
        meta={'图型': '流程图', '所属章节': '3.2 数据采集方案',
              'Stage23正式图题': CAPTION, '论文去向': '正文 图 3-1（3.2 节）',
              'variant': 'paper',
              '数据来源': '数据采集机制清单（采集入口、菜单任务构建、标准化搜索 URL、'
                          '列表页与详情页抓取、多进程协同、双标签页、URL 规范化、联合键去重、'
                          '进度记录、断点续爬、异常页面处理、结构化存储）',
              '规模锚点': {'原始岗位观测': rows}})
    plot_style.add_bottom_caption(fig, CAPTION)
    diagnostics = plot_style.save_sci_figure(
        fig, f'{STEM}_display', CAPTION, figure_id='图S16',
        meta={'图内图题': CAPTION, '图型': '流程图', '所属章节': '3.2 数据采集方案',
              'variant': 'display'})
    plt.close(fig)

    gate_keys = ('caption_structure_ok', 'caption_ink_visible', 'caption_not_cropped',
                 'no_top_title', 'png_600dpi', 'pdf_valid', 'legend_no_overlap',
                 'axis_labels_present')
    failed = [key for key in gate_keys if not diagnostics.get(key)]
    failed_paper = figure_finalize.failed_paper_gates(paper)

    print('=' * 78)
    print('图 S16（正文图 3-1）Stage23 双版本执行结果')
    print('=' * 78)
    print(f'原始观测行数（只读元信息）：{rows}')
    print(f'论文版（无图内总图题）：{paper["png_path"]}（{paper["png_size_bytes"]:,} B）'
          f' ｜ 展示版：{diagnostics["png_path"]}'
          f'（{figure_finalize.png_size(diagnostics["png_path"]):,} B）')
    print(f'论文版已移除图内总图题：{paper["no_infigure_caption"]}')
    print(f'论文版门禁未通过项：{failed_paper if failed_paper else "无"}')
    for key in gate_keys:
        print(f'  展示版 {key}: {diagnostics.get(key)}')
    print(f'  展示版 legend_overlap_ratio: {diagnostics.get("legend_overlap_ratio")}')
    print(f'  展示版 png: {diagnostics.get("png_path")}')
    print(f'  展示版 pdf: {diagnostics.get("pdf_path")}')
    print(f'展示版未通过项：{failed if failed else "无"}')
    return 0 if (not failed and not failed_paper) else 1


if __name__ == '__main__':
    raise SystemExit(main())

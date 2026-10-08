# -*- coding: utf-8 -*-
"""重绘图 8-11「福利标签簇整体置换与单标签置换的 MAE 增量」。

原图（45 号实验 E5 段）的问题与改法：

- 图内含 ``ax.set_title`` 与 3 行统计说明框（基准 MAE／置换次数／误差线口径），
  与全文「图片内不放正式图题与说明框」的规范冲突 → 图题交给载体文档，
  口径（误差线为标准差、置换次数）也一并交给载体文档，图内不再保留任何说明文字；
- 原图 7.6 in 宽、字号 9 pt，插到 15.5 cm 版心后只剩约 7.2 pt → 按版心出图并把
  字号换算到打印 11~12 pt；
- 六根柱里后四根（0.745 / 0.005 / 0.001 / 0.000 元/天）在线性轴上几乎不可见，
  原图又没有数据标签 → 柱顶补 3 位小数数值标签，「簇置换 ≈ 最强单列 ≫ 其余」
  的量级可以直接读出来；近零标签的原因（三列 one-hot 在测试集逐行相同）由载体正文承担；
- 只保存 PNG、缺矢量版 → 本轮补 600 dpi PNG ＋ 矢量 PDF，四边框、内向刻度、
  浅虚线网格、统一色板（簇用主色、单列用中性灰）；
- 横轴 6 个中文标签实测在水平排布下相邻会相撞（10 pt 时「免费健身设施」与
  「就近租房补贴」重叠 208 px²）→ 标签按 30° 倾斜：相邻标签成为平行线，
  平行间距（列距 × sin30° ≈ 1.14 cm）远大于标签厚度（约 0.43 cm），
  同时水平投影收缩到 2.05 cm 以内、小于列距 2.27 cm。（原图 18° 正是同一考虑。）
- 按用户要求删除图内文字：**整条图例**（白底框与两行说明「福利标签簇 = 5 列同步置换」
  「误差线为标准差（200 次置换）」）与**横轴名「置换对象」**；
  后者使论文版门禁的 ``axis_labels_present`` 判不过，在脚本中登记为「主动豁免项」，
  不计入阻断项（横轴含义由 6 个刻度标签承担，置换口径由载体表与正文承担）。

数值只读 ``outputs/results/benefit_cluster_permutation.csv``，6 行 × 3 列统计与
基准 MAE／置换次数逐项对齐冻结值，任一不符即中止，不产出图件。

用法：:

    python scripts\\61_redraw_benefit_cluster_permutation.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matplotlib import pyplot as plt  # noqa: E402

from src import figure_finalize, plot_style, project_paths  # noqa: E402

STEM = 'fig_8_11_benefit_cluster_permutation'
PERM_CSV = PROJECT_ROOT / 'outputs' / 'results' / 'benefit_cluster_permutation.csv'
PRINT_WIDTH_CM = 15.5
PRINT_HEIGHT_CM = 7.6
CM = 1.0 / 2.54
FONT_SCALE = 11.0 / 12.0
FONTS = {'axis_label': 12.0, 'tick': 11.0, 'legend': 9.5, 'annotation': 10.5}
CLUSTER_ROW = '福利标签簇（5 列同步置换）'
CLUSTER_TICK = '福利标签簇'        # 刻度标签只用短名，「5 列同步置换」由图例一条说明承担
LABEL_ROTATION = 30.0             # 横轴标签倾角：水平排布时相邻标签会相撞
XLABEL_FONTSIZE = 9.5             # 标签字号：倾角下水平投影需小于列距
LINE_SPACING = 1.2                # matplotlib 默认行距系数（标签厚度实测用）
CAP_HALF = 0.15                   # 误差线帽的半个宽度（数据单位，柱宽 0.62）

# 冻结锚点：benefit_cluster_permutation.csv 的 6 行 × 3 列统计（保留 4 位小数）
ANCHORS = {
    '福利标签簇（5 列同步置换）': (14.1503, 0.4552, 14.8669),
    '单列：免费健身设施': (12.2478, 0.4087, 12.8852),
    '单列：就近租房补贴': (0.0000, 0.0000, 0.0000),
    '单列：餐饮': (0.0048, 0.0120, 0.0236),
    '单列：节日礼品': (0.7451, 0.0932, 0.8987),
    '单列：弹性工作制': (0.0008, 0.0042, 0.0070),
}
ANCHOR_BASE_MAE = 35.476021
ANCHOR_ROUNDS = 200
# 按用户要求删除横轴名「置换对象」：论文版门禁的 axis_labels_present 因此判不过，
# 属主动豁免项（横轴含义由 6 个刻度标签与载体文档承担），不视为图件缺陷
WAIVED_GATES = {'axis_labels_present': '横轴名按用户要求删除'}


def _w(width_cm: float) -> float:
    return width_cm * CM * FONT_SCALE


def read_permutation() -> pd.DataFrame:
    """读冻结的置换结果，并逐项断言 6 行 × 3 列统计与基准 MAE／置换次数。"""
    frame = pd.read_csv(PERM_CSV)
    if list(frame['置换对象']) != list(ANCHORS):
        raise AssertionError('置换对象与冻结顺序不符：%s' % list(frame['置换对象']))
    columns = ['MAE 增量均值', 'MAE 增量标准差', 'MAE 增量 95% 分位']
    for _, row in frame.iterrows():
        actual = tuple(round(float(row[column]), 4) for column in columns)
        expected = ANCHORS[row['置换对象']]
        if actual != expected:
            raise AssertionError('%s 统计与冻结值不符：%s ≠ %s'
                                 % (row['置换对象'], actual, expected))
    base_mae = float(frame['基准 MAE'].iloc[0])
    rounds = int(frame['置换次数'].iloc[0])
    if abs(base_mae - ANCHOR_BASE_MAE) > 5e-7 or rounds != ANCHOR_ROUNDS:
        raise AssertionError('基准 MAE／置换次数与冻结值不符：%.6f / %d' % (base_mae, rounds))
    return frame


def build_figure(frame: pd.DataFrame):
    """6 根柱：簇用主色、单列用中性灰；误差线为标准差；柱顶标 3 位小数。"""
    labels = [CLUSTER_TICK if name == CLUSTER_ROW else str(name).replace('单列：', '')
              for name in frame['置换对象']]
    values = frame['MAE 增量均值'].to_numpy('float64')
    errors = frame['MAE 增量标准差'].to_numpy('float64')
    colors = [plot_style.MAIN_COLOR] + [plot_style.MUTED_COLOR] * (len(frame) - 1)

    fig, ax = plt.subplots(figsize=(_w(PRINT_WIDTH_CM),
                                    PRINT_HEIGHT_CM / PRINT_WIDTH_CM * _w(PRINT_WIDTH_CM)))
    positions = np.arange(len(frame), dtype='float64')
    ax.bar(positions, values, width=0.62, color=colors, edgecolor='black', linewidth=0.5,
           zorder=3)
    # 误差线逐柱单独画，不用 bar(yerr=...)：后者把 6 根柱的帽线合并成一条跨全宽的
    # Line2D，其包围盒覆盖整个坐标区，会让「图例与数据图元重叠率」门禁误报
    for position, value, error in zip(positions, values, errors):
        ax.plot([position, position], [value - error, value + error], color='black',
                linewidth=0.8, zorder=4)
        for edge in (value - error, value + error):
            ax.plot([position - CAP_HALF, position + CAP_HALF], [edge, edge],
                    color='black', linewidth=0.8, zorder=4)

    top = float(np.max(values + errors))
    pad = 0.035 * top
    for position, (value, error) in enumerate(zip(values, errors)):
        ax.text(position, value + error + pad, '%.3f' % value, ha='center', va='bottom',
                fontsize=FONTS['annotation'], zorder=4)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=LABEL_ROTATION, rotation_mode='anchor',
                       ha='right', va='top', fontsize=XLABEL_FONTSIZE)
    ax.set_xlim(-0.55, len(frame) - 0.45)
    ax.set_ylim(0.0, top * 1.16)
    # 横轴名与整条图例均按用户要求删除：横轴含义由 6 个刻度标签承担，
    # 「簇 = 5 列同步置换」「误差线 = 标准差」「置换 200 次」由载体文档（表与正文）承担
    ax.set_ylabel('置换后 MAE 增量（元/天）', fontsize=FONTS['axis_label'])
    plot_style.apply_sci_axis(ax, grid_axis='y')
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.19, top=0.96)
    return fig, ax


def label_geometry_audit(fig, ax) -> dict:
    """倾斜标签的几何自检（单位换成 pt，与最终缩放无关）。

    - 相邻标签是平行线，其**平行间距** ＝ 列距 × sin(倾角)；
    - 标签的**厚度** ＝ 行数 × 行距系数 × 字号；
    - 只有平行间距大于厚度，相邻倾斜标签才不会叠字（这是判定条件）；
    - 另报「标签水平投影最大值」，作为版式松紧的参考（平行线投影可互压而不叠字）。
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_pt = 72.0 / fig.dpi
    ticks = [tick for tick in ax.get_xticklabels() if tick.get_text()]
    positions = sorted(ax.transData.transform((tick.get_position()[0], 0.0))[0]
                       for tick in ticks)
    pitch_pt = min(np.diff(positions)) * to_pt
    angle = math.radians(LABEL_ROTATION)
    font_pt = max(tick.get_fontsize() for tick in ticks)
    lines = max(len(tick.get_text().split('\n')) for tick in ticks)
    thickness = font_pt * LINE_SPACING * lines
    gap = pitch_pt * math.sin(angle)
    footprint = max(tick.get_window_extent(renderer).width * to_pt for tick in ticks)
    return {'倾角': LABEL_ROTATION,
            '列距pt': round(pitch_pt, 2),
            '平行间距pt': round(gap, 2),
            '标签厚度pt': round(thickness, 2),
            '平行间距大于标签厚度': bool(gap > thickness),
            '标签行数': lines,
            '标签水平投影最大值pt': round(footprint, 2),
            '水平投影小于列距（参考）': bool(footprint < pitch_pt)}


def main() -> int:
    plot_style.setup_sci_style()
    plot_style.FONT_SIZES.update(FONTS)
    plt.rcParams.update({'axes.labelsize': FONTS['axis_label'],
                         'xtick.labelsize': FONTS['tick'],
                         'ytick.labelsize': FONTS['tick']})

    frame = read_permutation()
    fig, ax = build_figure(frame)
    geometry = label_geometry_audit(fig, ax)
    diagnostics = figure_finalize.save_paper_figure(
        fig, project_paths.FIGURES_DIR, STEM, subfigures=[],
        meta={'数据来源': 'outputs/results/benefit_cluster_permutation.csv（45 号实验 E5 段冻结结果）',
              '口径': '对测试集特征矩阵中对应的 one-hot 列做行置换；簇置换 = 5 列用同一次行置换，'
                      '单列置换 = 每次只置换 1 列；每组 %d 次，误差线为标准差'
                      % int(frame['置换次数'].iloc[0]),
              '校验': '6 行 × 3 列统计与基准 MAE／置换次数逐项对齐冻结值：基准 MAE %.6f 元/天'
                      % float(frame['基准 MAE'].iloc[0]),
              '版式': '图内不留任何说明文字：去掉图内标题、3 行统计说明框、整条图例'
                      '（白底框与「簇=5 列同步置换」「误差线=标准差」两行说明），'
                      '并删除横轴名「置换对象」（门禁 axis_labels_present 主动豁免）；'
                      '仅保留柱顶 3 位小数数值标签，使 0.745／0.005／0.001／0.000 等近零柱可读；'
                      '横轴标签 %.0f° 倾斜（水平排布实测相邻相撞），'
                      '按版心 %.1f cm 出图换算字号，补矢量 PDF' % (LABEL_ROTATION, PRINT_WIDTH_CM),
              '用途': '实验补全报告 E5 图件重绘（图题由载体文档给出）'})
    failed = figure_finalize.failed_paper_gates(diagnostics)
    waived = {gate: WAIVED_GATES[gate] for gate in failed if gate in WAIVED_GATES}
    blocking = [gate for gate in failed if gate not in WAIVED_GATES]

    metrics = project_paths.METRICS_DIR / 'stage_exp_e5_figure_redraw.json'
    metrics.write_text(json.dumps(
        {'stem': STEM,
         '基准 MAE': round(float(frame['基准 MAE'].iloc[0]), 6),
         '置换次数': int(frame['置换次数'].iloc[0]),
         'MAE 增量均值': {str(row['置换对象']): round(float(row['MAE 增量均值']), 4)
                         for _, row in frame.iterrows()},
         '横轴标签几何': geometry,
         '论文版门禁未通过项': failed,
         '主动豁免项': waived,
         'png_pixel_size': diagnostics['png_pixel_size'],
         'png': diagnostics['png_path'], 'pdf': diagnostics['pdf_path']},
        ensure_ascii=False, indent=2), encoding='utf-8')

    print('=' * 78)
    print(frame.round(4).to_string(index=False))
    print('图件：', diagnostics['png_path'])
    print('像素：', diagnostics['png_pixel_size'])
    print('论文版门禁未通过项：', failed or '无')
    print('主动豁免项：', waived or '无')
    print('阻断项：', blocking or '无')
    print('横轴标签几何（设计尺寸下的 pt）：')
    for key, value in geometry.items():
        print('        %s = %s' % (key, value))
    print('=' * 78)
    return 0 if not blocking and geometry['平行间距大于标签厚度'] else 1


if __name__ == '__main__':
    sys.exit(main())

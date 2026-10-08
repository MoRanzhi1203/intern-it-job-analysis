# -*- coding: utf-8 -*-
"""图 8-1（特征组消融实验的验证集与测试集 MAE）四种候选呈现，供对比选择。

问题背景：现有柱图 y 轴从 0 起、上限 46.6，而验证集各配置 MAE 只跨 0.476 元/天
（占轴长 1.0%）、测试集跨 1.683（3.6%），肉眼是 10 根等高柱，正文引用的
「0.48 / 1.68 元/天」「测试集最低 34.71」在图上读不出来。

四种候选（均只读冻结结果，不改数值）：

- A 哑铃点图：y＝5 个配置、x＝MAE（截断 34~37），每行两点（验证集/测试集）加连线；
- B 柱图＋ΔMAE 区间双面板：左＝原柱图（展示差异量级很小），右＝4 个增量的 ΔMAE 与
  95% 置信区间；
- C 单面板 ΔMAE 区间点图（同 B 右面板，放大到整幅）；
- D 保守小改：柱图保留，y 轴上限收紧 + 数值标注改水平 + x 轴名去括号。

输出到 ``outputs/figures/_compare_fig_8_1/``，不覆盖正式图件。

用法：:

    python scripts\\57_fig8_1_variant_compare.py
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

OUT_DIR = project_paths.FIGURES_DIR / '_compare_fig_8_1'
MODEL_JSON = project_paths.METRICS_DIR / 'stage_26_4_model_sync.json'
METRICS_XLSX = project_paths.TABLES_DIR / '61_stage26_4_metrics_final.xlsx'
PRINT_WIDTH_CM = 15.5
CM = 1.0 / 2.54
FONT_SCALE = 11.0 / 12.0
FONTS = {'axis_label': 12.0, 'tick': 11.0, 'legend': 11.0, 'annotation': 10.5}
X_TICK_FONT = 9.8                            # 候选 D 的 x 刻度字号（单行标签需要）
CONFIG_LABELS = {'A+B+C': '基础', 'A+B+C+D': '基础+技能', 'A+B+C+E': '基础+文本',
                 'A+B+C+D+E': '完整模型', 'A+B+C+D+E+SafeF': '完整+时间位置'}
INCREMENT_LABELS = {'技能增量（D）': '技能', '文本语义增量（E）': '文本',
                    '技能与文本增量（D+E）': '技能+文本',
                    '扩展敏感性增量（Safe-F 发布时间位置）': '发布时间位置'}
SPLIT_LABELS = {'validation': '验证集', 'test': '测试集'}
# 冻结值锚点（stage_26_4_model_sync.json，用于防止读入漂移）
ANCHOR_VALID = [36.4500, 36.4914, 36.5149, 36.0390, 36.1261]
ANCHOR_TEST = [34.7127, 35.4574, 36.2249, 35.9965, 36.3960]


def _w(width_cm: float) -> float:
    return width_cm * CM * FONT_SCALE


def _height(print_width_cm: float, print_height_cm: float) -> float:
    return print_height_cm / print_width_cm * _w(print_width_cm)


def apply_fonts() -> None:
    plot_style.setup_sci_style()
    plot_style.FONT_SIZES.update(FONTS)
    plt.rcParams.update({'axes.labelsize': FONTS['axis_label'],
                         'xtick.labelsize': FONTS['tick'],
                         'ytick.labelsize': FONTS['tick']})


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = json.loads(MODEL_JSON.read_text(encoding='utf-8'))
    frame = pd.DataFrame(metrics['消融五配置'])
    frame['配置名'] = [CONFIG_LABELS[str(item)] for item in frame['配置']]
    valid = [round(float(value), 4) for value in frame['validation_MAE']]
    test = [round(float(value), 4) for value in frame['test_MAE']]
    if valid != ANCHOR_VALID or test != ANCHOR_TEST:
        raise AssertionError('消融 MAE 与冻结值不符：%s / %s' % (valid, test))
    increments = pd.read_excel(METRICS_XLSX, sheet_name='03_消融增量bootstrap')
    increments['增量名'] = [INCREMENT_LABELS[str(item)] for item in increments['增量检验']]
    increments['子集名'] = [SPLIT_LABELS[str(item)] for item in increments['数据子集']]
    return frame, increments


def variant_a(frame: pd.DataFrame):
    """候选 A：哑铃点图（每行两点＝验证集/测试集，连线＝两子集差距）。"""
    fig, ax = plt.subplots(figsize=(_w(PRINT_WIDTH_CM), _height(PRINT_WIDTH_CM, 7.4)))
    rows = np.arange(len(frame), dtype='float64')
    valid = frame['validation_MAE'].to_numpy('float64')
    test = frame['test_MAE'].to_numpy('float64')
    for row, low, high in zip(rows, valid, test):
        ax.plot([min(low, high), max(low, high)], [row, row], color=plot_style.MUTED_COLOR,
                linewidth=1.6, zorder=2, solid_capstyle='round')
    ax.plot(valid, rows, marker='o', linestyle='none', markersize=7.0, markerfacecolor='white',
            markeredgecolor=plot_style.MAIN_COLOR, markeredgewidth=1.5, zorder=4,
            label='验证集 MAE')
    ax.plot(test, rows, marker='o', linestyle='none', markersize=4.6, color=plot_style.ACCENT_COLOR,
            markeredgecolor='black', markeredgewidth=0.45, zorder=5, label='测试集 MAE')
    ax.set_yticks(rows)
    ax.set_yticklabels(frame['配置名'].tolist())
    ax.set_ylim(-0.6, len(frame) - 0.4)
    ax.invert_yaxis()
    # 截断轴：点图不要求零起点，量程收到数据区间附近才看得出差异
    ax.set_xlim(34.0, 37.0)
    ax.set_xticks(np.arange(34.0, 37.01, 0.5))
    ax.set_xlabel('MAE（元/天）', fontsize=FONTS['axis_label'])
    ax.set_ylabel('特征组配置', fontsize=FONTS['axis_label'])
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
              fontsize=FONTS['legend'], columnspacing=2.0, handletextpad=0.5,
              borderaxespad=0.0)
    plot_style.apply_sci_axis(ax, grid_axis='x')
    fig.subplots_adjust(left=0.245, right=0.985, bottom=0.135, top=0.885)
    return fig, [], frame


def _draw_increment_panel(ax, increments: pd.DataFrame, show_legend: bool) -> None:
    """ΔMAE 与 95% 区间：每行一个增量，验证集/测试集两条错开的区间。"""
    names = list(dict.fromkeys(increments['增量名'].tolist()))
    rows = np.arange(len(names), dtype='float64')
    for index, name in enumerate(names):
        block = increments[increments['增量名'] == name]
        for split, offset, color, filled in (('验证集', 0.15, plot_style.MAIN_COLOR, False),
                                             ('测试集', -0.15, plot_style.ACCENT_COLOR, True)):
            item = block[block['子集名'] == split].iloc[0]
            low, high = float(item['95% 置信区间下界']), float(item['95% 置信区间上界'])
            estimate = float(item['ΔMAE（不含 − 含）'])
            y = rows[index] + offset
            ax.plot([low, high], [y, y], color=color, linewidth=1.6,
                    solid_capstyle='butt', zorder=3)
            for edge in (low, high):
                ax.plot([edge, edge], [y - 0.055, y + 0.055], color=color, linewidth=1.2,
                        zorder=3)
            ax.plot([estimate], [y], marker='o', linestyle='none',
                    markersize=5.2 if filled else 6.4,
                    color=color if filled else 'white', markerfacecolor=color if filled else 'white',
                    markeredgecolor=color, markeredgewidth=1.4, zorder=4)
    ax.axvline(0.0, color='black', linewidth=1.0, linestyle='-', zorder=2)
    ax.set_yticks(rows)
    ax.set_yticklabels(names)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel('ΔMAE（元/天），正值＝加入该组后误差下降', fontsize=FONTS['axis_label'])
    ax.set_ylabel('特征组增量', fontsize=FONTS['axis_label'])
    plot_style.apply_sci_axis(ax, grid_axis='x')
    if show_legend:
        handles = [plt.Line2D([], [], color=plot_style.MAIN_COLOR, linewidth=1.6, marker='o',
                              markerfacecolor='white', markeredgecolor=plot_style.MAIN_COLOR,
                              markersize=6.4, label='验证集'),
                   plt.Line2D([], [], color=plot_style.ACCENT_COLOR, linewidth=1.6, marker='o',
                              markersize=5.2, label='测试集')]
        # 图例文字只用两个子集名：口径（ΔMAE 与 95% 置信区间）由子图名与图注承担，
        # 以免图例宽于坐标区、把 tight bbox 撑大导致打印字号被动变小
        ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2,
                  frameon=False, fontsize=FONTS['legend'], columnspacing=1.6,
                  handletextpad=0.5, borderaxespad=0.0)


def add_subcaptions(fig, panels) -> None:
    """(a)(b) 子图名置于「横轴刻度标签与横轴名」下沿之下（与 26g/55/56 号脚本同规则）。"""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, (letter, text) in panels:
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


def variant_b(frame: pd.DataFrame, increments: pd.DataFrame):
    """候选 B：左＝柱图（0 起点），右＝ΔMAE 与 95% 区间。"""
    fig, axes = plt.subplots(1, 2, figsize=(_w(PRINT_WIDTH_CM), _height(PRINT_WIDTH_CM, 8.8)))
    ax = axes[0]
    positions = np.arange(len(frame), dtype='float64')
    width = 0.38
    valid = frame['validation_MAE'].to_numpy('float64')
    test = frame['test_MAE'].to_numpy('float64')
    ax.bar(positions - width / 2, valid, width, color=plot_style.MAIN_COLOR, edgecolor='black',
           linewidth=0.6, label='验证集 MAE', zorder=3)
    ax.bar(positions + width / 2, test, width, color=plot_style.ACCENT_COLOR, edgecolor='black',
           linewidth=0.6, label='测试集 MAE', zorder=3)
    for offset, values in ((-width / 2, valid), (width / 2, test)):
        for position, value in zip(positions + offset, values):
            ax.text(position, value + 0.25, '%.2f' % value, ha='center', va='bottom',
                    fontsize=FONTS['annotation'])
    ax.set_xticks(positions)
    ax.set_xticklabels(frame['配置名'].tolist(), rotation=20, ha='right')
    ax.set_ylim(0.0, float(max(valid.max(), test.max())) * 1.12)
    ax.set_xlabel('特征组配置', fontsize=FONTS['axis_label'])
    ax.set_ylabel('MAE（元/天）', fontsize=FONTS['axis_label'])
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
              fontsize=FONTS['legend'], columnspacing=1.6, handletextpad=0.5,
              borderaxespad=0.0)
    plot_style.apply_sci_axis(ax, grid_axis='y')
    _draw_increment_panel(axes[1], increments, show_legend=True)
    panels = [('a', '各配置的验证集与测试集 MAE'), ('b', '各特征组增量的 ΔMAE 与 95% 置信区间')]
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.30, top=0.845, wspace=0.40)
    add_subcaptions(fig, list(zip(axes, panels)))
    return fig, [(letter, text, ax) for (letter, text), ax in zip(panels, axes)], frame


def variant_c(increments: pd.DataFrame):
    """候选 C：单面板 ΔMAE 与 95% 区间。"""
    fig, ax = plt.subplots(figsize=(_w(PRINT_WIDTH_CM), _height(PRINT_WIDTH_CM, 7.6)))
    _draw_increment_panel(ax, increments, show_legend=True)
    fig.subplots_adjust(left=0.255, right=0.985, bottom=0.185, top=0.845)
    return fig, [], increments


def variant_d(frame: pd.DataFrame):
    """候选 D：保守小改（柱图 + y 轴收紧 + 数值标注水平 + x 轴名去括号）。

    高度取 10.2 cm（原 7.0 cm 时宽高比 2.15:1，整体过扁；柱宽不变而纵向拉长后，
    柱体不再显得被压平）。
    """
    fig, ax = plt.subplots(figsize=(_w(PRINT_WIDTH_CM), _height(PRINT_WIDTH_CM, 10.2)))
    positions = np.arange(len(frame), dtype='float64')
    width = 0.38
    valid = frame['validation_MAE'].to_numpy('float64')
    test = frame['test_MAE'].to_numpy('float64')
    ax.bar(positions - width / 2, valid, width, color=plot_style.MAIN_COLOR, edgecolor='black',
           linewidth=0.6, label='验证集 MAE', zorder=3)
    ax.bar(positions + width / 2, test, width, color=plot_style.ACCENT_COLOR, edgecolor='black',
           linewidth=0.6, label='测试集 MAE', zorder=3)
    for offset, values in ((-width / 2, valid), (width / 2, test)):
        for position, value in zip(positions + offset, values):
            ax.text(position, value + 0.35, '%.2f' % value, ha='center', va='bottom',
                    fontsize=FONTS['annotation'])
    ax.set_xticks(positions)
    # x 轴标签不倾斜、居中于各自的分组，且保持单行：最长的「完整+时间位置」在 11 pt 下
    # 宽 2.79 cm、组距 2.78 cm（会互相顶住），因此把 x 刻度字号收到 9.8 pt（打印约 10.7 pt）
    ax.set_xticklabels(frame['配置名'].tolist())
    ax.set_yticks(np.arange(0.0, 40.1, 10.0))
    ax.set_ylim(0.0, 43.0)                      # 轴顶 43：柱顶（36.51）以下留出数值标注空间
    ax.set_xlabel('特征组配置', fontsize=FONTS['axis_label'])
    ax.set_ylabel('MAE（元/天）', fontsize=FONTS['axis_label'])
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
              fontsize=FONTS['legend'], columnspacing=1.6, handletextpad=0.5,
              borderaxespad=0.0)
    plot_style.apply_sci_axis(ax, grid_axis='y')
    # apply_sci_axis 会按 rcParams 重设刻度字号，故在其之后单独收小 x 刻度
    for label in ax.get_xticklabels():
        label.set_fontsize(X_TICK_FONT)
    fig.subplots_adjust(left=0.088, right=0.985, bottom=0.245, top=0.855)
    return fig, [], frame


def main() -> int:
    apply_fonts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame, increments = load_data()
    summary = []
    variants = [('候选A_哑铃点图', variant_a(frame),
                 '截断轴点图：差异与两子集差距可见；不下调零起点'),
                ('候选B_柱图加ΔMAE区间', variant_b(frame, increments),
                 '左柱图保留「差异量级很小」的直观，右面板承担增量与显著性'),
                ('候选C_ΔMAE区间点图', variant_c(increments),
                 '只保留增量与 95% 区间，最聚焦（需改图题与正文引用）'),
                ('候选D_保守小改', variant_d(frame),
                 '柱图结构不变，仅收紧 y 轴上留白、数值标注改水平、x 轴名去括号')]
    for stem, (fig, subfigures, _), note in variants:
        diagnostics = figure_finalize.save_paper_figure(
            fig, OUT_DIR, stem, subfigures=subfigures,
            meta={'数据来源': 'stage_26_4_model_sync.json 消融五配置；'
                              '61_stage26_4_metrics_final.xlsx / 03_消融增量bootstrap',
                  '口径': 'ΔMAE = 不含该特征组的 MAE − 含该特征组的 MAE，正值表示加入后误差下降；'
                          '区间为配对 bootstrap 1,000 次、随机种子固定',
                  '用途': '图 8-1 呈现方式候选择一（对比稿，不覆盖正式图件）'})
        failed = figure_finalize.failed_paper_gates(diagnostics)
        pixels = diagnostics['png_pixel_size']
        summary.append({'候选': stem, '说明': note, '门禁未通过项': failed,
                        'png_pixel_size': pixels,
                        '打印高度_cm': round(PRINT_WIDTH_CM * pixels[1] / pixels[0], 2),
                        'png': diagnostics['png_path']})
        print('  %-24s 未通过项=%-4s 打印 %.1f × %.1f cm' % (
            stem, failed or '无', PRINT_WIDTH_CM, summary[-1]['打印高度_cm']))
        plt.close(fig)

    (project_paths.METRICS_DIR / 'stage_exp_fig_8_1_variants.json').write_text(
        json.dumps({'说明': '图 8-1 四种候选呈现的对比记录（正式图件未改动）',
                    '对比': summary}, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())

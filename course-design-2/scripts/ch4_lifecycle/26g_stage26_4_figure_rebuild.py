# -*- coding: utf-8 -*-
"""Stage26.4 图件重构（只新增文件，不覆盖任何既有图件）。

对应提示词 `docs/prompts/Trae_Stage26_4_终稿精简与学术化统一修订提示词.md`
§12～§18、§31：把正文中仍存在元素重叠、图内文字过多或与旧模型不一致的图件
**实际重绘**为 `outputs/figures/supplementary/图S55..图S65`（PNG 600 dpi + 矢量 PDF），
既有 `图S01..图S54` 与 `eda/**`、`modeling/**` 一律不动。

逐图处置
--------
| 新文件 | 正文图号 | 改前问题 | 改后方案 |
|---|---|---|---|
| 图S55 | 4-3 | 图内统计注释框（Median/IQR/P90 多行），1×2 横排 | 2×1 纵排，删除统计注释框，只留 P50/P90 参考线与必要数据标签 |
| 图S56 | 4-5 | 时间区域说明色块 + 三条窗口图例，图例与横轴挤压 | 2×1 纵排，删除时间区域色块与说明，图例移至坐标区上方并加大上下间距 |
| 图S57 | 4-6 | 图内两行统计范围说明框、时间区域色块 | 单面板 7 日滚动中位数，删除图内说明框与色块并放大 |
| 图S58 | 4-7 | 原始逐日序列与 7 日滚动序列高度重复，图内色块 | 只保留 IQR 带 + 7 日滚动中位数（删除重复原始序列面板，统计范围说明移正文） |
| 图S59 | 5-1 | 1×3 横排，子图名与横轴、纵轴标签挤压 | 3×1 纵排，加大 hspace，统一总图题 |
| 图S60 | 5-2 | 1×2 横排 + 图内 ε² 统计注释框 | 2×1 纵排，删除统计注释框，效应量标签由 6 位小数收紧为 3 位 |
| 图S61 | 5-3 | 与共现结构挤在同一组合图，图内 4 行统计说明框 | 拆分为独立图「福利标签薪资关联效应」，删除统计说明框 |
| 图S62 | 5-4 | 同上（拆分前为 5-3 的 (b) 面板） | 拆分为独立图「高效应福利标签共现结构」，2×1 纵排 |
| 图S63 | 6-2 | 热力图偏小、标签拥挤 | 按版心最大宽度放大并增加图高，色标与标签清晰 |
| 图S64 | 8-5 | 旧模型 SHAP 产物，与表 8-5 不一致 | 按新模型（正式特征体系 A+B+C+D+E）重算 TreeSHAP 重绘 |
| 图S65 | 8-6 | 旧模型技能 SHAP 产物 | 按新模型技能 SHAP 重绘 |

约束
----
- 只读既有产物（`outputs/tables/**`、`data/**`、既有 scripts、`src/**`）；
- 只写 `outputs/figures/supplementary/图S55..图S65`（PNG + PDF），
  以及 `outputs/logs/metrics/stage_26_4_figure_rebuild.json`；
- 复用既有绘图函数时通过内存内版式重映射实现，不写回磁盘、不修改任何既有脚本；
- 每张图导出后做**图元重叠实测**（子图名 / 图例 / 坐标轴名 / 色标两两包围盒求交）。

运行::

    python scripts\\26g_stage26_4_figure_rebuild.py
    python scripts\\26g_stage26_4_figure_rebuild.py shap
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src import figure_finalize, io_utils, model_training, plot_style, project_paths, schema  # noqa: E402
from src.script_support import (load_script as _load, make_patched_adjust,  # noqa: E402
                                   make_patched_subplots)

TABLES_DIR = project_paths.TABLES_DIR
SUPP_DIR = project_paths.FIGURES_DIR / 'supplementary'
METRICS_PATH = project_paths.METRICS_DIR / 'stage_26_4_figure_rebuild.json'

TARGET_FONTS = {
    'axis_label': 12.0, 'tick': 11.0, 'legend': 11.0, 'annotation': 10.5,
    'subfigure_caption': 11.0, 'figure_caption': 11.0,
}
CM = 1.0 / 2.54
FONT_SCALE = 11.0 / 12.0

MODE: dict = {'kind': 'single', 'w': 3.96, 'panel_h': 2.8, 'adjust': {}}
_ORIG_SUBPLOTS = plt.subplots
_ORIG_ADJUST = Figure.subplots_adjust


def _apply_mode(kind: str, display_cm: float, panel_h: float, adjust: dict) -> None:
    MODE.update(kind=kind, w=display_cm * CM * FONT_SCALE, panel_h=panel_h, adjust=adjust)


plt.subplots = make_patched_subplots(lambda: MODE, _ORIG_SUBPLOTS)
Figure.subplots_adjust = make_patched_adjust(lambda: MODE, _ORIG_ADJUST)

# --------------------------------------------------------------------------- #
# 子图名定位：按「横轴刻度标签 + 横轴名的实际下沿」下移，与坐标轴及其刻度不重叠
# （§13 / §15 要求）。
# --------------------------------------------------------------------------- #
_ORIG_SUBFIGCAP = plot_style.add_subfigure_caption


def _patched_subfigcap(ax, letter, text, y=None, fontsize=None):
    if y is None:
        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        box = ax.get_window_extent(renderer)
        bottom = box.y0
        for artist in [ax.xaxis.label, *ax.get_xticklabels()]:
            try:
                artist_box = artist.get_window_extent(renderer)
            except Exception:                                 # noqa: BLE001
                continue
            bottom = min(bottom, artist_box.y0)
        y = -((box.y0 - bottom) + 7.0) / max(box.height, 1.0)
    return _ORIG_SUBFIGCAP(ax, letter, text, y=y, fontsize=fontsize)


plot_style.add_subfigure_caption = _patched_subfigcap


# --------------------------------------------------------------------------- #
# 通用工具
# --------------------------------------------------------------------------- #
def read_table(name: str, sheet: str) -> pd.DataFrame:
    return pd.read_excel(TABLES_DIR / name, sheet_name=sheet)


def overlap_audit(fig, extra=()) -> list:
    """图元重叠实测：子图名 / 图例 / 坐标轴名 / 色标标签两两包围盒求交。"""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    items = []
    for index, ax in enumerate(fig.axes):
        if ax.get_label() == '<colorbar>':
            items.append((f'色标标签{index}', ax.yaxis.label))
        else:
            items.append((f'横轴名{index}', ax.xaxis.label))
            items.append((f'纵轴名{index}', ax.yaxis.label))
        legend = ax.get_legend()
        if legend is not None:
            items.append((f'图例{index}', legend))
        for text in ax.texts:
            content = str(text.get_text()).strip()
            if content:
                items.append((f'子图名{content[:14]}', text))
    items.extend(extra)
    boxes = []
    for name, artist in items:
        try:
            box = artist.get_window_extent(renderer)
        except Exception:                                     # noqa: BLE001
            continue
        if box.width <= 0 or box.height <= 0:
            continue
        boxes.append((name, box))
    hits = []
    for first in range(len(boxes)):
        for second in range(first + 1, len(boxes)):
            a, b = boxes[first][1], boxes[second][1]
            width = min(a.x1, b.x1) - max(a.x0, b.x0)
            height = min(a.y1, b.y1) - max(a.y0, b.y0)
            if width > 1.0 and height > 1.0:
                hits.append({'元素A': boxes[first][0], '元素B': boxes[second][0],
                             '重叠面积px²': round(float(width * height), 1)})
    sys.stdout.flush()
    return hits


def reposition_captions(fig, gap_px: float = 10.0) -> int:
    """按**最终版式**重排子图名：置于「横轴刻度标签与横轴名」下沿之下 gap_px 像素处。

    子图名以 axes 分数坐标定位，而 axes 高度在 `subplots_adjust` 后会变化，
    须在版式固定后用实际像素几何重新锚定，以消除子图标题与坐标轴 / 刻度重叠。
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    moved = 0
    for ax in fig.axes:
        if ax.get_label() == '<colorbar>':
            continue
        box = ax.get_window_extent(renderer)
        bottom = box.y0
        candidates = [ax.xaxis.label, *ax.get_xticklabels()]
        for artist in candidates:
            try:
                artist_box = artist.get_window_extent(renderer)
            except Exception:                                     # noqa: BLE001
                continue
            bottom = min(bottom, artist_box.y0)
        y = -((box.y0 - bottom) + gap_px) / max(box.height, 1.0)
        for text in ax.texts:
            if re.match(r'^\([a-z]\)\s', str(text.get_text()).strip()):
                text.set_position((0.5, y))
                moved += 1
    return moved


def save(stem: str, fig, subfigures, meta: dict) -> dict:
    reposition_captions(fig)
    hits = overlap_audit(fig)
    diagnostics = figure_finalize.save_paper_figure(
        fig, SUPP_DIR, stem, subfigures=subfigures, meta=meta)
    failed = figure_finalize.failed_paper_gates(diagnostics)
    plt.close(fig)
    width_in = diagnostics['png_pixel_size'][0] / 600.0
    height_in = diagnostics['png_pixel_size'][1] / 600.0
    display_in = MODE['w'] / FONT_SCALE
    print(f"  [{'通过' if not failed else '未通过'}] {stem}  未通过项={failed or '无'}  "
          f"像素={diagnostics['png_pixel_size']}  图元重叠={len(hits)} 处  "
          f"打印宽={display_in / CM:.2f}cm  打印高={height_in * display_in / width_in / CM:.2f}cm  "
          f"最小字号等效={TARGET_FONTS['tick'] * display_in / width_in:.2f}pt")
    for hit in hits:
        print(f"        ! 重叠 {hit['元素A']} × {hit['元素B']}  {hit['重叠面积px²']} px²")
    diagnostics['overlap_hits'] = hits
    diagnostics['overlap_count'] = len(hits)
    return diagnostics


def round_labels(fig, ndigits: int = 3) -> None:
    """把图内 4 位以上小数的数据标签统一收紧为 ndigits 位（§26 精度统一）。"""
    pattern = re.compile(r'(?<![\d.])0\.(\d{4,})(?![\d.])')
    for ax in fig.axes:
        for text in ax.texts:
            content = str(text.get_text())
            replaced = pattern.sub(lambda m: f'0.{m.group(1)[:ndigits]}', content)
            if replaced != content:
                text.set_text(replaced)


# =========================================================================== #
# 复用既有绘图函数重绘（内存内版式重映射）
# =========================================================================== #
def setup_18b(b18) -> None:
    """按 18b.main 的同一顺序载入冻结锚点（只读，不写回任何文件）。

    与 scripts/ch4_lifecycle/26e_stage26_3_figure_refresh.py 的装载块一致，
    使图内数值与 Stage23 冻结锚点一一对应。
    """
    b18.DATA = b18.load_data()
    b18.JOB = b18.build_job_frame()
    b18.V = b18.build_anchor_values(b18.DATA, b18.JOB)
    values = b18.V
    values['stats_table'] = b18.DATA['stats']
    values['label_block'] = b18.DATA['stats'][
        b18.DATA['stats']['检验块'] == '单标签二元比较（present vs absent；Mann–Whitney + BH-FDR）'].copy()
    values['kw_table'] = b18.DATA['stats'][
        b18.DATA['stats']['检验块'] == '多组比较（Kruskal–Wallis）'].copy()
    values['robust_extreme'] = b18.DATA['robust_extreme']
    values['robust_target'] = b18.DATA['robust_target']
    values['ablation'] = b18.DATA['ablation']
    values['manifest'] = b18.DATA['manifest']
    values['skill_threshold'] = b18.DATA['skill_threshold']
    values['text_dim'] = b18.DATA['text_dim']
    tag_sets = b18.JOB['公司标签列表'].map(b18.as_tag_set)
    top5 = b18.TOP5_TAGS
    masks = {tag: tag_sets.map(lambda items, tag=tag: tag in items) for tag in top5}
    values['top5_counts'] = {tag: int(masks[tag].sum()) for tag in top5}
    values['top5_intersection'] = int(tag_sets.map(
        lambda items: all(tag in items for tag in top5)).sum())
    values['top5_union'] = int(tag_sets.map(
        lambda items: any(tag in items for tag in top5)).sum())
    jaccard = np.ones((len(top5), len(top5)))
    for first in range(len(top5)):
        for second in range(len(top5)):
            if first != second:
                both = int((masks[top5[first]] & masks[top5[second]]).sum())
                either = int((masks[top5[first]] | masks[top5[second]]).sum())
                jaccard[first, second] = both / either if either else 0.0
    values['top5_jaccard'] = jaccard


def run_reused() -> list:
    a18 = _load('_s26g_18a', 'scripts/figures_base/18a_redraw_eda_modeling_figures.py')
    b18 = _load('_s26g_18b', 'scripts/figures_base/18b_supplementary_figures.py')
    c26 = _load('_s26g_26c', 'scripts/ch4_lifecycle/26c_stage26_2_final_consolidation.py')
    setup_18b(b18)

    # 图 5-2 的 ε² 统计注释框属于「图内说明框」，按 §12 整体移入正文
    b18.note = lambda *args, **kwargs: None
    # 图 4-3 的 annotate_stats 同样是多行统计注释框
    plot_style.annotate_stats = lambda *args, **kwargs: None

    results = []

    # ---------------- 图 4-3（1×2 → 2×1，删除统计注释框） ---------------- #
    _apply_mode('stack', 10.5, 2.75,
                {'left': 0.17, 'right': 0.975, 'bottom': 0.135, 'top': 0.955,
                 'hspace': 0.42})
    fig, subs, meta = a18.build_02_salary_distribution({})
    results.append(save('fig_4_2_salary_midpoint_distribution', fig, subs,
                        {**meta, '图内文字': '已删除 Median/IQR 统计注释框，只保留 P50/P90 参考线',
                         '用途': '第4章 图 4-3 重制（1×2 横排 → 2×1 纵排）'}))

    # ---------------- 图 4-5（2×1，删除时间区域色块与说明） ---------------- #
    daily = read_table('ch4/37_stage26_1_lifecycle_statistics.xlsx', '08_日级指标明细')
    dates = pd.to_datetime(daily['date'])
    _apply_mode('stack', 15.5, 3.0,
                {'left': 0.115, 'right': 0.985, 'bottom': 0.145, 'top': 0.90,
                 'hspace': 0.80})
    fig, axes = plt.subplots(2, 1, figsize=(5.85, 6.0))
    panels = [
        ('a', '样本活跃计划周期数量',
         [('N_t', '每日活跃（原始日序列）', plot_style.MUTED_COLOR, 0.7),
          ('N_t_7d', '每日活跃（7 日滚动中位数）', plot_style.MAIN_COLOR, 2.0)]),
        ('b', '每日新增与每日结束数量',
         [('O_t_7d', '每日新增（7 日滚动中位数）', plot_style.MAIN_COLOR, 1.8),
          ('C_t_7d', '每日结束（7 日滚动中位数）', plot_style.ACCENT_COLOR, 1.8)]),
    ]
    for (letter, title, series_list), ax in zip(panels, axes):
        for column, label, color, width in series_list:
            ax.plot(dates, daily[column], color=color, linewidth=width, label=label)
        ax.set_xlabel('业务日期（由岗位发布时间与投递截止日期重构）')
        ax.set_ylabel(title)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.005), ncol=2,
                  frameon=False, fontsize=plot_style.FONT_SIZES['legend'])
        plot_style.apply_sci_axis(ax, grid_axis='y')
        plot_style.add_subfigure_caption(ax, letter, title)
    fig.subplots_adjust()
    results.append(save('fig_s56_planned_coverage_daily_active', fig,
                        [('a', panels[0][1], axes[0]), ('b', panels[1][1], axes[1])],
                        {'数据来源': 'ch4/37_stage26_1_lifecycle_statistics.xlsx / 08_日级指标明细',
                         '图内文字': '已删除历史回溯区 / 实际采集窗口 / 计划未来覆盖区色块与说明',
                         '用途': '第4章 图 4-5 重制（图例移至坐标区上方，上下间距加大）'}))

    # ---------------- 图 5-1（1×3 → 3×1） ---------------- #
    _apply_mode('stack', 10.5, 2.15,
                {'left': 0.20, 'right': 0.975, 'bottom': 0.115, 'top': 0.955,
                 'hspace': 0.72})
    fig, subs, meta = a18.build_04_structured_factor_salary({})
    results.append(save('fig_5_4_city_education_company_size_salary', fig, subs,
                        {**meta, '用途': '第5章 图 5-1 重制（1×3 横排 → 3×1 纵排）'}))

    # ---------------- 图 5-2（1×2 → 2×1，删除统计注释框） ---------------- #
    _apply_mode('stack', 15.5, 2.85,
                {'left': 0.30, 'right': 0.975, 'bottom': 0.145, 'top': 0.90,
                 'hspace': 0.72})
    fig, subs = b18.fig_s04(0)
    round_labels(fig, 3)
    results.append(save('fig_5_5_certification_salary_distribution', fig, subs,
                        {'数据来源': 'ch4/21_eda_statistical_analysis.xlsx / 11_统计检验（成对比较）',
                         '图内文字': '已删除 ε² 统计注释框（统计范围与数值移入正文表 5-1）',
                         '用途': '第5章 图 5-2 重制（1×2 横排 → 2×1 纵排）'}))

    # ---------------- 图 5-3（拆分：福利标签薪资关联效应） ---------------- #
    _apply_mode('single', 15.5, 4.6,
                {'left': 0.115, 'right': 0.975, 'bottom': 0.155, 'top': 0.86})
    fig, subs = b18.fig_s05(0)
    results.append(save('fig_s61_benefit_label_salary_effect', fig, subs,
                        {'数据来源': 'ch4/21_eda_statistical_analysis.xlsx / 11_统计检验（标签块）',
                         '图内文字': '已删除进入比较数 / 显著数 / 上限与阈值说明框',
                         '用途': '第5章 图 5-3（由原 5-3 组合图拆分而来）'}))

    # ---------------- 图 5-4（拆分：高效应福利标签共现结构） ---------------- #
    _apply_mode('stack', 15.5, 2.9,
                {'left': 0.235, 'right': 0.975, 'bottom': 0.135, 'top': 0.925,
                 'hspace': 0.70})
    fig, axes = plt.subplots(2, 1, figsize=(5.85, 5.8))
    b18.tag_overlap_bars(axes[0])
    b18.tag_jaccard_heatmap(axes[1])
    captions = [('a', '高效应福利标签与集合统计范围的岗位规模'),
                ('b', '高效应福利标签两两 Jaccard 相似度')]
    for (letter, caption), ax in zip(captions, axes):
        plot_style.add_subfigure_caption(ax, letter, caption)
    fig.subplots_adjust()
    results.append(save('fig_5_6_high_effect_benefit_label_cluster', fig,
                        [(letter, caption, ax)
                         for (letter, caption), ax in zip(captions, axes)],
                        {'数据来源': 'ch4/21_eda_statistical_analysis.xlsx / 11_统计检验（标签块）',
                         '图内文字': '已删除最大两两 Jaccard 说明框',
                         '用途': '第5章 图 5-4（由原 5-3 组合图拆分而来）'}))

    # ---------------- 图 6-2（热力图放大） ---------------- #
    _apply_mode('single', 15.5, 6.6,
                {'left': 0.29, 'right': 0.975, 'bottom': 0.30, 'top': 0.975})
    fig, subs, meta = a18.build_07_category_skill_heatmap({})
    results.append(save('fig_6_3_category_skill_hit_heatmap', fig, subs,
                        {**meta, '用途': '第6章 图 6-2 继续放大（版心最大宽度 + 增加图高）'}))

    # ---------------- 图 4-6 与图 4-7（自定义：删除重复面板与色块） ---------------- #
    results.extend(run_lifecycle_figures(c26, daily, dates))

    # ---------------- 图 8-6（技能 SHAP，按新模型结果重绘） ---------------- #
    results.append(run_skill_shap())
    return results


def run_lifecycle_figures(c26, daily, dates) -> list:
    """图 4-6 / 图 4-7：删除重复面板与时间区域色块，只保留有解释价值的内容。"""
    results = []
    category_daily = c26.build_category_daily()

    _apply_mode('single', 15.5, 4.3,
                {'left': 0.115, 'right': 0.975, 'bottom': 0.235, 'top': 0.80})
    fig, ax = plt.subplots(figsize=(5.85, 4.3))
    for position, (category, series) in enumerate(category_daily.items()):
        values = series.reindex(dates).to_numpy('float64')
        values = pd.Series(values).rolling(7, min_periods=1).median().to_numpy('float64')
        ax.plot(dates, values, linewidth=1.7,
                color=plot_style.PALETTE[position % len(plot_style.PALETTE)],
                label=category)
    ax.set_xlabel('业务日期（由岗位发布时间与投递截止日期重构）')
    ax.set_ylabel('活跃计划周期数（7 日滚动中位数）')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False,
              fontsize=plot_style.FONT_SIZES['legend'])
    plot_style.apply_sci_axis(ax, grid_axis='y')
    plot_style.format_integer_axis(ax, axis='y')
    caption = '主要岗位大类的活跃计划周期数'
    plot_style.add_subfigure_caption(ax, 'a', caption)
    fig.subplots_adjust()
    results.append(save('fig_s57_main_category_planned_coverage', fig, [('a', caption, ax)],
                        {'数据来源': 'job_strict_daily_panel_26_1.parquet × 岗位大类集合',
                         '图内文字': '已删除样本统计范围说明框与历史回溯区 / 采集窗口色块',
                         '用途': '第4章 图 4-6（只保留 7 日滚动中位数并放大）'}))

    _apply_mode('single', 15.5, 4.3,
                {'left': 0.155, 'right': 0.975, 'bottom': 0.235, 'top': 0.80})
    fig, ax = plt.subplots(figsize=(5.85, 4.3))
    ax.fill_between(dates, daily['salary_p25'], daily['salary_p75'],
                    color=plot_style.MAIN_COLOR, alpha=0.18, linewidth=0,
                    label='P25~P75（IQR）')
    ax.plot(dates, daily['salary_median_7d'], color=plot_style.MAIN_COLOR,
            linewidth=1.8, label='薪资中位数（7 日滚动中位数）')
    ax.set_xlabel('业务日期（由岗位发布时间与投递截止日期重构）')
    ax.set_ylabel('活跃计划周期薪资中点（元/天）')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.005), ncol=2, frameon=False,
              fontsize=plot_style.FONT_SIZES['legend'])
    plot_style.apply_sci_axis(ax, grid_axis='y')
    plot_style.format_integer_axis(ax, axis='y')
    caption = '活跃计划周期薪资中位数与四分位距'
    plot_style.add_subfigure_caption(ax, 'a', caption)
    fig.subplots_adjust()
    results.append(save('fig_s58_active_cycle_salary_median_iqr', fig, [('a', caption, ax)],
                        {'数据来源': 'job_strict_daily_panel_26_1.parquet 的日级薪资聚合',
                         '图内文字': '已删除与滚动中位数高度重复的原始逐日序列与时间区域色块',
                         '用途': '第4章 图 4-7（只保留 IQR 带与 7 日滚动中位数并放大）'}))
    return results


# =========================================================================== #
# 图 8-5 / 图 8-6：按 Stage26.4 正式模型（A+B+C+D+E，288 维）重算 SHAP
# =========================================================================== #
def build_final_shap() -> dict:
    """复现 Stage26.4 最终模型（训练集 + 验证集重拟合）并计算测试集 TreeSHAP。"""
    f26 = _load('_s26g_26f', 'scripts/ch4_lifecycle/26f_stage26_4_final_polish.py')
    from src import skill_eda  # noqa: PLC0415

    model_frame = io_utils.read_parquet(project_paths.JOB_SALARY_MODEL_DATASET_PARQUET)
    splits = io_utils.read_parquet(project_paths.MODEL_SPLITS_PARQUET)
    membership = io_utils.read_parquet(project_paths.JOB_SKILL_MEMBERSHIP_PARQUET)
    entity = pd.read_parquet(project_paths.PROCESSED_UNIQUE_PARQUET,
                             columns=[schema.ID_FIELD, '发布时间'])
    entity['发布时间'] = pd.to_datetime(entity['发布时间'], errors='coerce')
    publish = entity.set_index(schema.ID_FIELD)['发布时间']
    feature_manifest = json.loads(
        (project_paths.SALARY_MODEL_DIR / 'feature_manifest.json').read_text(encoding='utf-8'))

    job_ids = model_frame[schema.ID_FIELD]
    frame = model_frame.copy()
    frame['publish_month'] = job_ids.map(publish).dt.month.to_numpy()
    frame['publish_weekday'] = job_ids.map(publish).dt.weekday.to_numpy()
    random_labels = pd.Series(splits['split'].to_numpy(), index=job_ids)

    grouped = f26.grouped_columns(feature_manifest, frame)
    grouped_new = f26.drop_features(grouped, f26.REMOVED_FINAL)
    skill_map = model_training.build_skill_map(membership, skill_eda.ALL_USABLE_SCOPES)
    text_matrix = model_training.load_text_matrix(
        job_ids.tolist(), project_paths.FEATURES_DIR / 'job_text_embeddings.npz',
        project_paths.FEATURES_DIR / 'job_text_embedding_index.parquet')
    text_by_id = {job_id: position for position, job_id in enumerate(job_ids.tolist())}

    is_test = random_labels.reindex(frame[schema.ID_FIELD]).to_numpy() == 'test'
    train_valid = frame.loc[~is_test].reset_index(drop=True)
    test_frame = frame.loc[is_test].reset_index(drop=True)

    def text_for(block):
        return text_matrix[[text_by_id[job_id] for job_id in block[schema.ID_FIELD]]]

    assembler = f26.build_assembler(f26.MODEL_FEATURE_GROUPS, grouped_new,
                                    f26.SKILL_THRESHOLD, f26.TEXT_DIM)
    assembler.fit(train_valid, skill_map, text_for(train_valid))
    matrix_fit = assembler.transform(train_valid, skill_map, text_for(train_valid))
    matrix_test = assembler.transform(test_frame, skill_map, text_for(test_frame))
    model = model_training.make_model('LightGBM', f26.LIGHTGBM_PARAMS,
                                      random_state=f26.SEED)
    model.fit(matrix_fit, train_valid[schema.SALARY_MID_FIELD].to_numpy('float64'))
    prediction = np.asarray(model.predict(matrix_test), dtype='float64')
    truth = test_frame[schema.SALARY_MID_FIELD].to_numpy('float64')
    metrics = model_training.regression_metrics(truth, prediction)

    raw = model.predict(matrix_test, pred_contrib=True)
    if hasattr(raw, 'toarray'):
        raw = raw.toarray()
    raw = np.asarray(raw, dtype='float64')
    base_value = float(raw[:, -1].mean())
    values = raw[:, :-1]
    names = assembler.feature_names()
    dense = np.asarray(matrix_test.todense())
    return {'特征维度': int(assembler.schema.dimension), '特征名': names,
            '测试集SHAP': values, '测试集矩阵': dense, '基准值': base_value,
            '测试集指标': metrics, 'n_test': int(len(test_frame)),
            '技能列数': int(len(assembler.schema.skill_columns))}


def run_skill_shap() -> dict:
    """图 8-6：技能特征 SHAP Top20（条长 = 平均绝对 SHAP 值，标记 = 方向）。"""
    table = read_table('ch7/52_stage26_4_metrics_final.xlsx', '14_技能SHAP')
    frame = table.head(20).iloc[::-1].reset_index(drop=True)
    positions = np.arange(len(frame))

    _apply_mode('single', 15.5, 7.6,
                {'left': 0.30, 'right': 0.975, 'bottom': 0.115, 'top': 0.93})
    fig, ax = plt.subplots(figsize=(5.85, 7.6))
    ax.barh(positions, frame['平均绝对SHAP值'], height=0.62,
            color=plot_style.MUTED_COLOR, edgecolor='black', linewidth=0.5)
    for position, (value, direction) in enumerate(
            zip(frame['平均绝对SHAP值'], frame['技能存在时平均SHAP'])):
        marker = '^' if float(direction) > 0 else 'v'
        ax.plot(value * 0.04, position, marker=marker, markersize=5.2, color='black',
                zorder=3)
    ax.set_yticks(positions)
    ax.set_yticklabels([f'{name}（{int(freq):,}）' for name, freq
                        in zip(frame['技能'], frame['建模样本岗位数（频率）'])])
    ax.set_xlim(0, float(frame['平均绝对SHAP值'].max()) * 1.22)
    ax.set_xlabel('平均绝对 SHAP 值（元/天）')
    ax.set_ylabel('技能（括号内为建模样本岗位数）')
    ax.legend(handles=[Line2D([], [], marker='^', linestyle='none', color='black',
                              markersize=5.2),
                       Line2D([], [], marker='v', linestyle='none', color='black',
                              markersize=5.2)],
              labels=['技能存在时平均贡献为正', '技能存在时平均贡献为负'],
              loc='lower right', frameon=False, fontsize=plot_style.FONT_SIZES['legend'])
    plot_style.apply_sci_axis(ax, grid_axis='x')
    fig.subplots_adjust()
    values = frame['平均绝对SHAP值']
    expected = {'机器学习': 1.726, '强化学习': 0.609}
    for name, value in expected.items():
        actual = float(values[frame['技能'].eq(name)].iloc[0])
        assert abs(actual - value) < 5e-4, f'{name} 平均绝对 SHAP 与表 8-5 不一致：{actual}'
    return save('fig_s65_skill_shap_contribution', fig, [],
                {'数据来源': 'ch7/52_stage26_4_metrics_final.xlsx / 14_技能SHAP',
                 '统计范围': '技能出现统计范围；条长 = 平均绝对 SHAP 值，标记 = 技能存在时平均贡献方向',
                 '用途': '第8章 图 8-6（按 Stage26.4 正式模型 288 维重绘）'})


def run_beeswarm(block: dict) -> dict:
    """图 8-5：正式主模型 SHAP 蜂群图（Top12 特征）。"""
    names = block['特征名']
    values = block['测试集SHAP']
    dense = block['测试集矩阵']
    magnitude = np.abs(values).mean(axis=0)
    order = np.argsort(magnitude)[::-1][:12]

    _apply_mode('single', 15.5, 6.4,
                {'left': 0.315, 'right': 0.90, 'bottom': 0.115, 'top': 0.975})
    fig, ax = plt.subplots(figsize=(5.85, 6.4))
    rng = np.random.default_rng(42)
    for row_position, feature_index in enumerate(order):
        column = values[:, feature_index]
        raw = dense[:, feature_index]
        span = float(raw.max() - raw.min())
        scaled = (raw - raw.min()) / span if span > 0 else np.zeros_like(raw)
        jitter = rng.uniform(-0.22, 0.22, size=len(column))
        scatter = ax.scatter(column, np.full(len(column), row_position) + jitter, s=6,
                             c=scaled, cmap='cividis', vmin=0.0, vmax=1.0,
                             alpha=0.6, edgecolor='none')
    ax.axvline(0, color=plot_style.MUTED_COLOR, linestyle='--', linewidth=0.9)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([str(names[index]) for index in order])
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.set_xlabel('SHAP 值（元/天）')
    ax.set_ylabel('特征（按平均绝对 SHAP 值排序，Top12）')
    plot_style.apply_sci_axis(ax, grid_axis='x')
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.034, pad=0.02, ticks=[0.0, 0.5, 1.0])
    colorbar.set_label('特征取值（各特征内归一化）',
                       fontsize=plot_style.FONT_SIZES['axis_label'])
    colorbar.set_ticklabels(['低', '中', '高'])
    fig.subplots_adjust()

    expected = {'公司标签列表=免费健身设施': 14.88, '技术技能数': 7.42, '文本SVD3': 5.96}
    for name, value in expected.items():
        actual = float(magnitude[names.index(name)])
        assert abs(actual - value) < 0.02, f'{name} 平均绝对 SHAP 与表 8-5 不一致：{actual}'
    return save('fig_8_9_shap_beeswarm', fig, [],
                {'数据来源': 'Stage26.4 正式主模型（正式特征体系 A+B+C+D+E；'
                             '划分协议下编码后 288 维，'
                             '训练集 + 验证集重拟合后编码 %d 维）在测试集上的 TreeSHAP'
                             % block['特征维度'],
                 '统计范围': '加性一致性最大误差随模型重算校验；只表示模型预测贡献，非因果',
                 '测试集 MAE': round(block['测试集指标']['MAE'], 6),
                 '用途': '第8章 图 8-5（按 Stage26.4 正式模型重绘）'})


def main() -> int:
    started = time.time()
    snapshot = plot_style.setup_sci_style()
    plot_style.FONT_SIZES.update(TARGET_FONTS)
    plt.rcParams.update({
        'font.size': TARGET_FONTS['tick'],
        'axes.labelsize': TARGET_FONTS['axis_label'],
        'axes.titlesize': TARGET_FONTS['axis_label'],
        'xtick.labelsize': TARGET_FONTS['tick'],
        'ytick.labelsize': TARGET_FONTS['tick'],
        'legend.fontsize': TARGET_FONTS['legend'],
    })
    plt.rcParams['axes.unicode_minus'] = False
    SUPP_DIR.mkdir(parents=True, exist_ok=True)
    print('=' * 96)
    print('Stage26.4 图件重构：图S55..图S65（PNG 600 dpi + 矢量 PDF，只新增文件）')
    print(f"无头模式：backend={snapshot['backend']} headless={snapshot['headless']}")
    print('=' * 96)
    print('--- 一、复用既有绘图函数（内存内版式重映射） ---')
    results = run_reused()
    print('--- 二、图 8-5 蜂群图（按正式模型 SHAP 重算） ---')
    block = build_final_shap()
    results.append(run_beeswarm(block))
    print('    模型复核：特征维度=%d，测试集 MAE=%.6f / RMSE=%.6f / R²=%.6f'
          % (block['特征维度'], block['测试集指标']['MAE'], block['测试集指标']['RMSE'],
             block['测试集指标']['R2']))
    payload = {
        '生成时间': time.strftime('%Y-%m-%d %H:%M:%S'),
        '字体目标': TARGET_FONTS,
        '图件': [{key: value for key, value in item.items()
                  if key in ('stem', 'png_path', 'pdf_path', 'png_pixel_size',
                             'png_dpi', 'png_600dpi', 'pdf_valid', 'overlap_count',
                             'overlap_hits', 'removed_infigure_captions',
                             'no_infigure_caption', 'no_top_title')} for item in results],
        '图元重叠合计': int(sum(item['overlap_count'] for item in results)),
        '模型复核': {'特征维度': block['特征维度'], '技能列数': block['技能列数'],
                     'n_test': block['n_test'], '基准值（元/天）': round(block['基准值'], 6),
                     **{key: round(float(value), 6)
                        for key, value in block['测试集指标'].items()}},
        '运行耗时秒': round(time.time() - started, 3),
    }
    METRICS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                            encoding='utf-8')
    print('=' * 96)
    print(f'图元重叠实测合计：{payload["图元重叠合计"]} 处（应为 0）')
    print(f'仍打开的画布数（应为 0）：{plt.get_fignums()}')
    print(f'元数据：{METRICS_PATH}')
    print('=' * 96)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

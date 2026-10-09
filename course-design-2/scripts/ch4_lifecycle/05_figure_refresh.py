# -*- coding: utf-8 -*-
"""Stage26.3 图件重制（只新增文件，不覆盖任何既有图件）。

对应提示词 `docs/prompts/Trae_Stage26_3_结构精简与图文规范终修提示词.md`
第十五～十七条：把正文中"打印尺寸过小、图内字号不可读"的图件按
**论文实际打印宽度**重新生成，使图内最小字号不低于 10.5 pt 等效值。

判定依据（Stage26.2 成品实测，图片按版心宽度打印）
---------------------------------------------------
| 图号 | 原文件 | 打印宽 | 图内基础字号 | 打印后等效字号 |
|---|---|---|---|---|
| 4-1  | eda/01_sample_structure        | 10.5 cm | 8.6 pt | ≈5.1 pt |
| 4-3  | eda/02_salary_distribution     | 10.5 cm | 8.6 pt | ≈4.9 pt |
| 4-4  | fig_s29_recruitment_duration_distribution | 10.5 cm | 11.5 pt | ≈3.0 pt |
| 4-5  | fig_s30_planned_coverage_daily_active | 15.5 cm | 11.5 pt | ≈4.2 pt |
| 5-1  | eda/04_structured_factor_salary | 10.5 cm | 8.6 pt | ≈3.5 pt |
| 5-2  | fig_s04_company_certification_four_class_salary | 10.5 cm | 8.6 pt | ≈4.2 pt |
| 5-3  | 图S05S06_公司福利标签薪资关联与共现 | 10.5 cm | 8.6 pt | ≈4.1 pt |
| 5-4  | fig_s17_salary_factor_evidence_multi_category | 10.5 cm | 8.6 pt | ≈5.2 pt |
| 6-2  | eda/07_category_skill_heatmap  | 10.5 cm | 8.6 pt | ≈3.8 pt |
| 8-1  | fig_s28_feature_group_ablation_comparison           | 10.5 cm | 11.5 pt | ≈3.1 pt |
| 8-3  | fig_s22_generalization_scenario_comparison     | 10.5 cm | 8.6 pt | ≈4.5 pt |
| 8-4  | fig_s10_robustness_check_comparison           | 10.5 cm | 8.6 pt | ≈4.3 pt |

做法：把每个图件的**画布物理宽度**设为与其论文打印宽度一致（约 1:1），
并统一放大图内字号到 11～12 pt，同时把横排 1×2 / 1×3 子图改为
2×1 / 3×1 纵排；对内容重复的面板按提示词要求删除后单独重绘。

约束
----
- 只读既有产物（冻结 xlsx / parquet / 既有 scripts）；
- 只写 ``outputs/figures/supplementary/图S34..图S49``（PNG 600 dpi + 矢量 PDF），
  **不覆盖任何既有图件**，不修改任何既有脚本文件；
- 复用既有脚本的绘图函数时通过内存内版式重映射实现，不写回磁盘。

运行::

    python scripts\\05_figure_refresh.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = next(
    _candidate for _candidate in Path(__file__).resolve().parents
    if (_candidate / "data").is_dir() and (_candidate / "scripts").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from src import figure_finalize, plot_style, project_paths  # noqa: E402
from src.script_support import (load_script as _load, make_patched_adjust,  # noqa: E402
                                   make_patched_subplots)

TABLES_DIR = project_paths.TABLES_DIR
SUPP_DIR = project_paths.FIGURES_DIR / 'supplementary'

# --------------------------------------------------------------------------- #
# 图内字号：打印后等效 ≥ 10.5 pt（画布宽度≈打印宽度，故 1 pt ≈ 1 pt）
# --------------------------------------------------------------------------- #
TARGET_FONTS = {
    'axis_label': 12.0, 'tick': 11.0, 'legend': 11.0, 'annotation': 10.5,
    'subfigure_caption': 11.0, 'figure_caption': 11.0,
}
CM = 1.0 / 2.54
FONT_SCALE = 11.0 / 12.0      # 画布略窄于打印宽度，等效字号再放大 ~4%

MODE: dict = {'kind': 'single', 'w': 3.96, 'panel_h': 2.8, 'adjust': {}}
_ORIG_SUBPLOTS = plt.subplots
_ORIG_ADJUST = Figure.subplots_adjust


def _apply_mode(kind: str, display_cm: float, panel_h: float, adjust: dict) -> None:
    MODE.update(kind=kind, w=display_cm * CM * FONT_SCALE, panel_h=panel_h, adjust=adjust)


plt.subplots = make_patched_subplots(lambda: MODE, _ORIG_SUBPLOTS)
Figure.subplots_adjust = make_patched_adjust(lambda: MODE, _ORIG_ADJUST)

# --------------------------------------------------------------------------- #
# 既有脚本（只读导入；不修改磁盘上的文件）
# --------------------------------------------------------------------------- #
def read_table(name: str, sheet: str) -> pd.DataFrame:
    return pd.read_excel(TABLES_DIR / name, sheet_name=sheet)


def report(stem: str, display_cm: float) -> None:
    """打印既有绘图函数内部保存的图件尺寸与等效字号（这些图不再经 save() 包装）。"""
    from PIL import Image
    with Image.open(SUPP_DIR / f'{stem}.png') as image:
        width_px, height_px = image.size
    display_in = display_cm * CM
    natural_in = width_px / 600.0
    print(f"  [已保存] {stem}  像素={width_px}x{height_px}  打印宽={display_cm:.2f}cm  "
          f"打印高={height_px * display_in / width_px / CM:.2f}cm  "
          f"最小字号等效={TARGET_FONTS['tick'] * display_in / natural_in:.2f}pt")


def save(stem: str, fig, subfigures, meta: dict):
    diagnostics = figure_finalize.save_paper_figure(
        fig, SUPP_DIR, stem, subfigures=subfigures, meta=meta)
    failed = figure_finalize.failed_paper_gates(diagnostics)
    plt.close(fig)
    width_in = diagnostics['png_pixel_size'][0] / 600.0
    height_in = diagnostics['png_pixel_size'][1] / 600.0
    display_in = MODE['w'] / FONT_SCALE
    print(f"  [{ '通过' if not failed else '未通过'}] {stem}  未通过项={failed or '无'}  "
          f"像素={diagnostics['png_pixel_size']}  打印宽={display_in / CM:.2f}cm  "
          f"打印高={height_in * display_in / width_in / CM:.2f}cm  "
          f"最小字号等效={TARGET_FONTS['tick'] * display_in / width_in:.2f}pt")
    return diagnostics


# =========================================================================== #
# 一、复用既有绘图函数（已在别处冻结的数据与逻辑）
# =========================================================================== #
def run_reused() -> None:
    eda_figures = _load('_eda_figures', 'scripts/figures/base/01_eda_modeling_figures.py')
    supplementary_figures = _load('_supplementary_figures', 'scripts/figures/base/02_supplementary_figures.py')
    consolidation = _load('_consolidation', 'scripts/ch4_lifecycle/03_final_consolidation.py')

    # ---- 02_supplementary_figures 的全局锚点与只读宽表（与 02_supplementary_figures.main 完全一致） ---- #
    supplementary_figures.DATA = supplementary_figures.load_data()
    supplementary_figures.JOB = supplementary_figures.build_job_frame()
    supplementary_figures.V = supplementary_figures.build_anchor_values(supplementary_figures.DATA, supplementary_figures.JOB)
    V = supplementary_figures.V
    V['stats_table'] = supplementary_figures.DATA['stats']
    V['label_block'] = supplementary_figures.DATA['stats'][
        supplementary_figures.DATA['stats']['检验块'] == '单标签二元比较（present vs absent；Mann–Whitney + BH-FDR）'].copy()
    kw = supplementary_figures.DATA['stats'][supplementary_figures.DATA['stats']['检验块'] == '多组比较（Kruskal–Wallis）'].copy()
    V['kw_table'] = kw
    V['robust_extreme'] = supplementary_figures.DATA['robust_extreme']
    V['robust_target'] = supplementary_figures.DATA['robust_target']
    V['ablation'] = supplementary_figures.DATA['ablation']
    V['manifest'] = supplementary_figures.DATA['manifest']
    V['leak_blacklist'] = supplementary_figures.DATA['leak_blacklist']
    V['skill_threshold'] = supplementary_figures.DATA['skill_threshold']
    V['text_dim'] = supplementary_figures.DATA['text_dim']
    tag_sets = supplementary_figures.JOB['公司标签列表'].map(supplementary_figures.as_tag_set)
    TOP5 = supplementary_figures.TOP5_TAGS
    masks = {tag: tag_sets.map(lambda values, tag=tag: tag in values) for tag in TOP5}
    V['top5_counts'] = {tag: int(masks[tag].sum()) for tag in TOP5}
    V['top5_intersection'] = int(tag_sets.map(
        lambda values: all(tag in values for tag in TOP5)).sum())
    V['top5_union'] = int(tag_sets.map(
        lambda values: any(tag in values for tag in TOP5)).sum())
    jaccard = np.ones((len(TOP5), len(TOP5)))
    for i, first in enumerate(TOP5):
        for j, second in enumerate(TOP5):
            if i != j:
                both = int((masks[first] & masks[second]).sum())
                either = int((masks[first] | masks[second]).sum())
                jaccard[i, j] = both / either if either else 0.0
    V['top5_jaccard'] = jaccard

    # ---- 时间序列图件所需的冻结数据（与 03_final_consolidation.main 完全一致：冻结日级表） ---- #
    episodes = pd.read_parquet(consolidation.EPISODE_PATH)
    daily = read_table('ch4/37_stage26_1_lifecycle_statistics.xlsx', '08_日级指标明细')
    dates = pd.to_datetime(daily['date'])
    date_min, date_max = dates.min(), dates.max()

    flat = {'left': 0.34, 'right': 0.975, 'bottom': 0.15, 'top': 0.90, 'hspace': 0.62}
    wide_left = {'left': 0.44, 'right': 0.975, 'bottom': 0.15, 'top': 0.90, 'hspace': 0.62}

    # ---------------- 图 4-1 ----------------
    _apply_mode('stack', 15.5, 2.8, flat)
    fig, subs, meta = eda_figures.build_01_sample_structure({})
    meta = {**meta, '用途': '第4章 图 4-1 重制（1×2 横排 → 2×1 纵排）'}
    save('fig_s34_formal_sample_skill_extraction_structure', fig, subs, meta)

    # ---------------- 图 4-3 ----------------
    _apply_mode('stack', 10.5, 2.8, flat)
    fig, subs, meta = eda_figures.build_02_salary_distribution({})
    meta = {**meta, '用途': '第4章 图 4-3 重制（1×2 横排 → 2×1 纵排）'}
    save('fig_s35_salary_midpoint_distribution', fig, subs, meta)

    # ---------------- 图 4-4（复用 03_final_consolidation.figure_duration，只改输出文件名） ---------------- #
    consolidation.FIG_STEMS[1] = 'fig_s36_recruitment_duration_distribution'
    _apply_mode('stack', 10.5, 2.8, flat)
    consolidation.figure_duration(episodes, [])
    report('fig_s36_recruitment_duration_distribution', 10.5)
    # ---------------- 图 4-5（2×1 纵排，图例置于坐标区上方） ---------------- #
    _apply_mode('stack', 15.5, 3.0,
                {'left': 0.11, 'right': 0.985, 'bottom': 0.17, 'top': 0.83, 'hspace': 0.72})
    fig, axes = plt.subplots(2, 1, figsize=(5.85, 6.0))
    panels = [
        ('a', '样本活跃计划周期数量 N_t',
         [('N_t', '原始日序列', plot_style.MUTED_COLOR, 0.7),
          ('N_t_7d', '7 日滚动中位数', plot_style.MAIN_COLOR, 2.0),
          ('N_t_job_entity_7d', '去重到岗位实体', plot_style.PALETTE[2], 1.3)]),
        ('b', '每日新增 O_t / 每日结束 C_t',
         [('O_t', '每日新增（原始）', plot_style.MUTED_COLOR, 0.6),
          ('O_t_7d', '每日新增（7 日滚动）', plot_style.MAIN_COLOR, 1.8),
          ('C_t', '每日结束（原始）', plot_style.PALETTE[3], 0.6),
          ('C_t_7d', '每日结束（7 日滚动）', plot_style.ACCENT_COLOR, 1.8)]),
    ]
    for (letter, title, series_list), ax in zip(panels, axes):
        handles = consolidation._window_marks(ax, date_min, date_max)
        for column, label, color, width in series_list:
            ax.plot(dates, daily[column], color=color, linewidth=width, label=label)
        ax.set_xlabel('业务日期（由岗位发布时间 / 投递截止日期重构）')
        ax.set_ylabel(title)
        lines = [line for line in ax.get_lines()
                 if not str(line.get_label()).startswith('_')]
        ax.legend(handles + lines, [h.get_label() for h in handles + lines],
                  loc='lower center', bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False)
        plot_style.apply_sci_axis(ax, grid_axis='y')
        plot_style.add_subfigure_caption(ax, letter, title)
    fig.subplots_adjust()
    save('fig_s37_planned_coverage_daily_active', fig,
         [('a', panels[0][1], axes[0]), ('b', panels[1][1], axes[1])],
         {'数据来源': 'ch4/37_stage26_1_lifecycle_statistics.xlsx / 08_日级指标明细',
          '图注声明': '曲线由当前样本岗位的业务日期重构，不等同于当日完整市场存量',
          '用途': '第4章 图 4-5 重制（1×2 横排 → 2×1 纵排）'})

    # ---------------- 图 5-1（1×3 横排 → 3×1 纵排） ---------------- #
    _apply_mode('stack', 10.5, 2.15, flat)
    fig, subs, meta = eda_figures.build_04_structured_factor_salary({})
    meta = {**meta, '用途': '第5章 图 5-1 重制（1×3 横排 → 3×1 纵排）'}
    save('fig_s40_city_education_company_size_salary', fig, subs, meta)

    # ---------------- 图 5-2 ----------------
    _apply_mode('stack', 15.5, 2.8, wide_left)
    fig, subs = supplementary_figures.fig_s04(0)
    save('fig_s41_certification_salary_distribution', fig, subs,
         {'数据来源': 'ch4/21_eda_statistical_analysis.xlsx / 11_统计检验（成对比较）',
          '用途': '第5章 图 5-2 重制（1×2 横排 → 2×1 纵排）'})

    # ---------------- 图 5-3 ----------------
    _apply_mode('stack', 15.5, 2.8, wide_left)
    fig, subs = supplementary_figures.fig_s05s06(0)
    save('fig_s42_benefit_label_salary_association', fig, subs,
         {'数据来源': 'ch4/21_eda_statistical_analysis.xlsx / 11_统计检验（标签块）',
          '用途': '第5章 图 5-3 重制（1×2 横排 → 2×1 纵排）'})

    # ---------------- 图 5-4（复用 ch5_factors/03_factor_revision.figure_s17） ---------------- #
    factor_revision = _load('_factor_revision', 'scripts/ch5_factors/03_factor_revision.py')
    excl = pd.read_excel(TABLES_DIR / project_paths.TABLE_STAGE25_FACTOR_REVISION,
                         sheet_name='01_岗位大类_二元检验')
    sub = pd.read_excel(TABLES_DIR / project_paths.TABLE_STAGE25_FACTOR_REVISION,
                        sheet_name='02_岗位细分类_二元检验')
    top_binary = pd.concat([
        excl.head(10)[['类别', 'present岗位数', 'q值_BHFDR', 'Cliff_delta']].assign(字段='岗位大类'),
        sub.head(10)[['类别', 'present岗位数', 'q值_BHFDR', 'Cliff_delta']].assign(字段='岗位细分类')],
        ignore_index=True)
    top_binary['abs_delta'] = top_binary['Cliff_delta'].abs()
    factor_revision.FIG_STEM = 'fig_s43_salary_factor_evidence_multi_category'
    _apply_mode('stack', 15.5, 2.8, wide_left)
    factor_revision.figure_s17(factor_revision.exclusive_factor_table(), top_binary)
    report('fig_s43_salary_factor_evidence_multi_category', 15.5)

    # ---------------- 图 6-2（热力图明显放大） ---------------- #
    _apply_mode('single', 15.5, 5.2,
                {'left': 0.36, 'right': 0.98, 'bottom': 0.30, 'top': 0.97})
    fig, subs, meta = eda_figures.build_07_category_skill_heatmap({})
    meta = {**meta, '用途': '第6章 图 6-2 放大（接近版心最大宽度并增加图高）'}
    save('fig_6_3_category_skill_hit_heatmap', fig, subs, meta)

    # ---------------- 图 8-1（复用 03_final_consolidation.figure_ablation） ---------------- #
    ablation = read_table('ch7/49_stage26_3_metrics_after_feature_removal.xlsx', '03_消融五配置')
    consolidation.FIG_STEMS[0] = 'fig_s45_feature_group_ablation_comparison'
    _apply_mode('single', 15.5, 4.2,
                {'left': 0.10, 'right': 0.98, 'bottom': 0.34, 'top': 0.84})
    consolidation.figure_ablation(ablation, [])
    report('fig_s45_feature_group_ablation_comparison', 15.5)

    # ---------------- 图 8-4 ----------------
    _apply_mode('stack', 10.5, 2.8, wide_left)
    fig, subs = supplementary_figures.fig_s10(0)
    save('fig_s47_robustness_check_comparison', fig, subs,
         {'数据来源': 'ch8/23_ablation_robustness_shap.xlsx / 05_极端值敏感性、06_目标稳健性',
          '用途': '第8章 图 8-4 重制（1×2 横排 → 2×1 纵排）'})


# =========================================================================== #
# 二、按提示词删除重复面板后单独重绘的图件
# =========================================================================== #
def run_custom() -> None:
    consolidation = _load('_consolidation_b', 'scripts/ch4_lifecycle/03_final_consolidation.py')
    daily = read_table('ch4/37_stage26_1_lifecycle_statistics.xlsx', '08_日级指标明细')
    category_daily = consolidation.build_category_daily()
    dates = pd.to_datetime(daily['date'])
    date_min, date_max = dates.min(), dates.max()

    # ---------------- 图 4-6：只保留滚动/平滑结果（删除原始日序列面板） ---------------- #
    _apply_mode('single', 15.5, 3.7,
                {'left': 0.14, 'right': 0.98, 'bottom': 0.28, 'top': 0.80})
    fig, ax = plt.subplots(figsize=(5.85, 3.7))
    handles = consolidation._window_marks(ax, date_min, date_max)
    lines = []
    for position, (category, series) in enumerate(category_daily.items()):
        values = series.reindex(dates).to_numpy('float64')
        values = pd.Series(values).rolling(7, min_periods=1).median().to_numpy('float64')
        line, = ax.plot(dates, values, linewidth=1.6,
                        color=plot_style.PALETTE[position % len(plot_style.PALETTE)],
                        label=category)
        lines.append(line)
    ax.set_xlabel('业务日期（由岗位发布时间 / 投递截止日期重构）')
    ax.set_ylabel('样本活跃计划周期数（7 日滚动中位数）')
    ax.legend(handles + lines, [h.get_label() for h in handles + lines],
              loc='lower center', bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False)
    plot_style.apply_sci_axis(ax, grid_axis='y')
    ax.text(0.985, 0.03, '曲线由当前样本岗位的业务日期重构，\n不等同于当日完整市场存量',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=plot_style.FONT_SIZES['annotation'], linespacing=1.5)
    caption = '主要岗位大类样本活跃计划周期数（7 日滚动中位数）'
    plot_style.add_subfigure_caption(ax, 'a', caption)
    fig.subplots_adjust()
    save('fig_s38_main_category_planned_coverage', fig, [('a', caption, ax)],
         {'数据来源': 'job_strict_daily_panel_26_1.parquet × 岗位大类集合',
          '用途': '第4章 图 4-6（删除与原始日序列重复的面板，只保留滚动中位数）'})

    # ---------------- 图 4-7：只保留薪资中位数与 IQR（删除活跃数量面板） ---------------- #
    _apply_mode('single', 15.5, 3.7,
                {'left': 0.15, 'right': 0.98, 'bottom': 0.28, 'top': 0.80})
    fig, ax = plt.subplots(figsize=(5.85, 3.7))
    handles = consolidation._window_marks(ax, date_min, date_max)
    band = ax.fill_between(dates, daily['salary_p25'], daily['salary_p75'],
                           color=plot_style.MAIN_COLOR, alpha=0.18, linewidth=0,
                           label='P25~P75（IQR）')
    line_raw, = ax.plot(dates, daily['salary_median'], color=plot_style.MUTED_COLOR,
                        linewidth=0.7, alpha=0.75, label='每日薪资中位数（原始序列）')
    line_smooth, = ax.plot(dates, daily['salary_median_7d'], color=plot_style.MAIN_COLOR,
                           linewidth=1.8, label='薪资中位数 7 日滚动中位数')
    ax.set_xlabel('业务日期（由岗位发布时间 / 投递截止日期重构）')
    ax.set_ylabel('活跃计划周期薪资中点（元/天）')
    ax.legend(handles + [band, line_raw, line_smooth],
              [h.get_label() for h in handles + [band, line_raw, line_smooth]],
              loc='lower center', bbox_to_anchor=(0.5, 1.005), ncol=2, frameon=False)
    plot_style.apply_sci_axis(ax, grid_axis='y')
    caption = '活跃计划周期薪资中位数与 IQR（Strict 统计范围）'
    plot_style.add_subfigure_caption(ax, 'a', caption)
    fig.subplots_adjust()
    save('fig_s39_active_cycle_salary_median_iqr', fig, [('a', caption, ax)],
         {'数据来源': 'job_strict_daily_panel_26_1.parquet 的日级薪资聚合',
          '用途': '第4章 图 4-7（删除与图 4-5 重复的活跃数量面板并放大）'})

    # ---------------- 图 7-1：薪资预测模型构建与评估流程 ---------------- #
    structure_finalize = _load('_structure_finalize', 'scripts/ch4_lifecycle/04_structure_finalize.py')
    steps = [
        ('14,883 个正式薪资样本', '来自 17,144 个唯一岗位实体中薪资可解析的岗位'),
        ('A/B/C/D/E 特征组与 Safe-F', '岗位基础、地域、公司、技能与文本语义，加发布时间位置'),
        ('建模前特征诊断', '缺失率、近常量、低频类别、数值冗余与子集分布对照'),
        ('Train / Validation / Test 划分', '按薪资十分位分层，70% / 15% / 15%'),
        ('仅用训练集拟合全部预处理', '类别编码、技能频次筛选与文本语义降维'),
        ('Ridge / RandomForest / CatBoost / LightGBM', '同一划分、同一特征、同一候选参数网格'),
        ('按验证集 MAE 选定主模型', '锁定 LightGBM：400 棵树、学习率 0.05、叶子数 63'),
        ('训练集与验证集重拟合', '测试集一次性评估，不参与模型与参数选择'),
        ('消融、分组划分、时间划分与 SHAP', '稳健性、跨公司与跨发布时间区间泛化'),
    ]
    assert len(steps) == len(structure_finalize.FLOW_STEPS)
    _apply_mode('single', 10.5, 6.5,
                {'left': 0.02, 'right': 0.98, 'bottom': 0.02, 'top': 0.98})
    from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: PLC0415
    fig, ax = plt.subplots(figsize=(3.96, 6.5))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis('off')
    count = len(steps)
    height = 0.075
    gap = (0.965 - 0.02 - count * height) / (count - 1)
    for index, (title, detail) in enumerate(steps):
        top = 0.965 - index * (height + gap)
        box = Rectangle((0.045, top - height), 0.91, height,
                        facecolor='#f2f2f2' if index % 2 == 0 else '#e8eef4',
                        edgecolor='black', linewidth=0.9)
        ax.add_patch(box)
        ax.text(0.5, top - height * 0.36, title, ha='center', va='center', fontsize=12.0)
        ax.text(0.5, top - height * 0.74, detail, ha='center', va='center',
                fontsize=10.6, color='#404040')
        if index < count - 1:
            arrow = FancyArrowPatch((0.5, top - height), (0.5, top - height - gap),
                                    arrowstyle='-|>', mutation_scale=14.0,
                                    linewidth=1.0, color='black')
            ax.add_patch(arrow)
    fig.subplots_adjust()
    save('fig_s48_salary_model_build_eval_flow', fig,
         [],
         {'数据来源': '本文第 7 章与第 8 章的正式建模流程',
          '统计范围': '流程图只描述正式流程，不含任何统计结果',
          '用途': '第7章 图 7-1（替代原模型验证集 MAE / RMSE 比较图）'})

    # ---------------- 图 7-2：建模前特征诊断与子集分布对照 ---------------- #
    dist = read_table('ch7/48_stage26_3_pre_model_diagnostics.xlsx', '07_三子集分布对照')
    redundant = read_table('ch7/48_stage26_3_pre_model_diagnostics.xlsx', '05_连续数值特征冗余对')
    _apply_mode('stack', 15.5, 2.9,
                {'left': 0.40, 'right': 0.975, 'bottom': 0.16, 'top': 0.90, 'hspace': 0.70})
    fig, axes = plt.subplots(2, 1, figsize=(5.85, 5.8))

    ax = axes[0]
    values = dist.set_index('指标')[['Train', 'Validation', 'Test']]
    order = ['样本量', '薪资中位数', '薪资IQR', '平均技能数', '岗位描述字符数中位数',
             '一线城市占比', '人工智能岗位占比']
    labels = ['样本量', '薪资中位数', '薪资IQR', '平均技能数', '描述字符数中位数',
              '一线城市占比', 'AI 岗位占比']
    values = values.loc[order]
    scaled = values.div(values.max(axis=1), axis=0)
    positions = np.arange(len(order))
    width = 0.26
    for offset, subset, color in [(-width, 'Train', plot_style.MAIN_COLOR),
                                  (0.0, 'Validation', plot_style.ACCENT_COLOR),
                                  (width, 'Test', plot_style.PALETTE[2])]:
        bars = ax.bar(positions + offset, scaled[subset], width, color=color,
                      edgecolor='black', linewidth=0.6, label=subset)
        for bar, raw in zip(bars, values[subset]):
            ax.annotate(f'{raw:,.4g}', xy=(bar.get_x() + bar.get_width() / 2,
                                           bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center',
                        va='bottom', fontsize=9.0, rotation=90)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=16, ha='right')
    ax.set_xlabel('诊断指标')
    ax.set_ylabel('子集内相对值（各指标最大值归一）')
    ax.set_ylim(0, 1.46)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.004), ncol=3, frameon=False)
    plot_style.apply_sci_axis(ax, grid_axis='y')
    caption_a = 'Train / Validation / Test 基本分布对照（柱顶为子集内原始取值）'
    plot_style.add_subfigure_caption(ax, 'a', caption_a)

    ax = axes[1]
    frame = redundant[redundant['|ρ| > 0.85'].astype(str).str.strip().isin(['是', 'True', '1'])].copy()
    frame['|ρ|'] = frame['Spearman ρ'].abs()
    frame = frame.sort_values('|ρ|')
    positions = np.arange(len(frame))
    colors = [plot_style.ACCENT_COLOR if value >= 0.99 else plot_style.MAIN_COLOR
              for value in frame['|ρ|']]
    ax.barh(positions, frame['|ρ|'], height=0.62, color=colors,
            edgecolor='black', linewidth=0.5)
    for position, value in zip(positions, frame['|ρ|']):
        ax.text(value + 0.004, position, f'{value:.6f}', va='center', ha='left',
                fontsize=plot_style.FONT_SIZES['annotation'])
    ax.axvline(0.85, color='black', linestyle='--', linewidth=0.9)
    ax.set_yticks(positions)
    ax.set_yticklabels([f'{a} × {b}' for a, b in zip(frame['特征一'], frame['特征二'])])
    ax.set_xlim(0.80, 1.13)
    ax.set_xlabel('|Spearman ρ|')
    ax.set_ylabel(f'冗余特征对（|ρ| > 0.85，共 {len(frame)} 对）')
    plot_style.apply_sci_axis(ax, grid_axis='x')
    ax.text(0.985, 0.06, '虚线 = 0.85 人工检查阈值\n只对有连续意义的数值特征计算',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=plot_style.FONT_SIZES['annotation'], linespacing=1.5)
    caption_b = f'连续数值特征冗余诊断（|ρ| > 0.85 的 {len(frame)} 对，不做全特征相关矩阵）'
    plot_style.add_subfigure_caption(ax, 'b', caption_b)
    fig.subplots_adjust()
    save('fig_s49_pre_model_diagnostics_subset_distribution', fig,
         [('a', caption_a, axes[0]), ('b', caption_b, axes[1])],
         {'数据来源': 'ch7/48_stage26_3_pre_model_diagnostics.xlsx / 05、07 子表',
          '统计范围': '移除发布时点不可得的后验特征后的正式特征集；'
                  '冗余诊断只覆盖有连续意义的数值特征，不做 320×320 全特征矩阵',
          '用途': '第7章 7.1 节 图 7-2 建模前特征诊断（配合表 7-1、表 7-3）'})

    # ---------------- 图 8-3：三类泛化场景对照 ---------------- #
    unified = read_table('ch7/49_stage26_3_metrics_after_feature_removal.xlsx', '02_三种划分统一协议')
    labels = ['Random\nSplit', 'Company\nGroup Split', 'Retrospective\nTemporal Split']
    _apply_mode('stack', 10.5, 2.8,
                {'left': 0.22, 'right': 0.975, 'bottom': 0.20, 'top': 0.86, 'hspace': 0.62})
    fig, axes = plt.subplots(2, 1, figsize=(3.96, 5.6))
    positions = np.arange(len(unified))
    ax = axes[0]
    ax.bar(positions - 0.19, unified['test MAE'], width=0.36, color=plot_style.MAIN_COLOR,
           edgecolor='black', linewidth=0.6, label='MAE')
    ax.bar(positions + 0.19, unified['test RMSE'], width=0.36, color=plot_style.MUTED_COLOR,
           edgecolor='black', linewidth=0.6, label='RMSE')
    span = float(unified[['test MAE', 'test RMSE']].to_numpy().max())
    for offset, column in ((-0.19, 'test MAE'), (0.19, 'test RMSE')):
        for position, value in zip(positions + offset, unified[column]):
            ax.text(position, value + span * 0.02, f'{value:.2f}', ha='center', va='bottom',
                    fontsize=plot_style.FONT_SIZES['annotation'])
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel('数据划分方式（同一协议：仅在各自 train 上拟合）')
    ax.set_ylabel('test 误差（元/天）')
    ax.set_ylim(0, span * 1.32)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.005), ncol=2, frameon=False)
    plot_style.apply_sci_axis(ax, grid_axis='y')
    caption_a = 'MAE 与 RMSE 对照'
    plot_style.add_subfigure_caption(ax, 'a', caption_a)

    ax = axes[1]
    ax.bar(positions, unified['test R²'], width=0.5, color=plot_style.PALETTE[2],
           edgecolor='black', linewidth=0.6)
    for position, value in zip(positions, unified['test R²']):
        ax.text(position, value + 0.012, f'{value:.3f}', ha='center', va='bottom',
                fontsize=plot_style.FONT_SIZES['annotation'])
    ax.set_xticks(positions)
    ax.set_xticklabels(['Random Split', 'Company Group Split', 'Temporal Split'])
    ax.set_xlabel('数据划分方式')
    ax.set_ylabel('test R²')
    ax.set_ylim(0, float(unified['test R²'].max()) * 1.28)
    plot_style.apply_sci_axis(ax, grid_axis='y')
    ax.text(0.985, 0.95, '三种划分对应不同预测任务，\n不构成「谁更优」的比较',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=plot_style.FONT_SIZES['annotation'], linespacing=1.5)
    caption_b = 'R² 对照'
    plot_style.add_subfigure_caption(ax, 'b', caption_b)
    fig.subplots_adjust()
    save('fig_s46_generalization_scenario_comparison', fig,
         [('a', caption_a, axes[0]), ('b', caption_b, axes[1])],
         {'数据来源': 'ch7/49_stage26_3_metrics_after_feature_removal.xlsx / 02 子表（统一协议三种划分）',
          '用途': '第8章 图 8-3 重制（1×2 横排 → 2×1 纵排）'})


def main() -> int:
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
    plot_style.SCI_FIGURES_DIR = SUPP_DIR
    SUPP_DIR.mkdir(parents=True, exist_ok=True)
    print('=' * 96)
    print('Stage26.3 图件重制：按论文打印宽度重绘（目标等效字号 ≥ 10.5 pt）')
    print(f"无头模式：backend={snapshot['backend']} headless={snapshot['headless']}")
    print('=' * 96)
    print('--- 复用既有绘图函数（内存内版式重映射，不写回任何既有文件） ---')
    run_reused()
    print('--- 按提示词删除重复面板后单独重绘 ---')
    run_custom()
    print('=' * 96)
    print(f'出图后仍打开的画布数（应为 0）：{plt.get_fignums()}')
    print('=' * 96)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

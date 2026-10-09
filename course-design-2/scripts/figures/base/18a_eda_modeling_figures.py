# -*- coding: utf-8 -*-
"""Stage23 收口重绘：eda 10 张 + modeling 7 张正式图件（不重算任何统计 / 模型 / SHAP）。

双输出路径（Stage23 硬要求）：
- **论文版**：覆盖同名主文件（``eda/01_sample_structure.png`` 等），图片内部**不含**
  「图 X-X」正式总图题（总图题交给 Word Caption），保留绘图区 / 坐标轴 / 图例 / 位于
  各子图下方的 ``(a)(b)(c)(d)`` 子图名；600 dpi PNG + 矢量 PDF；
- **独立展示版**：``<stem>_display.png|pdf``，底部带 Stage23 正式图题（用于 PPT / 单独查看）。

约束（代码层面强制）：
- 只读：``outputs/tables/*.xlsx``、``outputs/logs/metrics/*.json``、``data/processed/*.parquet``；
- 只写：``outputs/figures/eda/*.png|pdf``、``outputs/figures/modeling/*.png|pdf``
  与 ``outputs/registries/_redraw_registry.json``；
- 绘图前先跑 Stage23 冻结锚点校验（``src/figure_finalize.validate_anchors``），任一不符即中止；
- 不重跑 ``scripts/ch4_lifecycle/13_run_eda.py``、``scripts/ch7_model/14_train_salary_model.py``、
  ``scripts/ch8_robust/15_ablation_robustness_shap.py``，不重跑 notebook，不重算 SHAP；
- 不重绘 ``modeling/07_shap_beeswarm``（逐样本 SHAP 矩阵未落盘，保留原文件不动）；
- 视觉规范唯一来源 ``src/plot_style.py``（本脚本不修改该模块）。

运行：
    python scripts/figures/base/18a_eda_modeling_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = next(
    _candidate for _candidate in Path(__file__).resolve().parents
    if (_candidate / "data").is_dir() and (_candidate / "scripts").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import figure_finalize, plot_style, project_paths  # noqa: E402

TABLES_DIR = project_paths.TABLES_DIR
EDA_DIR = project_paths.EDA_FIGURES_DIR
MODELING_DIR = project_paths.MODELING_FIGURES_DIR
REGISTRY_PATH = project_paths.REGISTRIES_DIR / '_redraw_registry.json'

TABLE_27 = 'ch6/19_skill_eda_scope_audit.xlsx'
TABLE_29 = 'ch4/21_eda_statistical_analysis.xlsx'
TABLE_30 = 'ch7/22_model_comparison.xlsx'
TABLE_31 = 'ch8/23_ablation_robustness_shap.xlsx'

SALARY_LABEL = '薪资中点（元/天）'
FINAL_MODEL = 'FINAL（LightGBM）'
SALARY_COLUMN = '薪资中点'

# 逐图门禁（与 save_sci_figure 实际写回的键一致，全部必须为 True）
GATE_KEYS = (
    'inward_ticks', 'light_grid', 'caption_structure_ok', 'subfigure_caption_visible',
    'subfigure_label_sequence_ok', 'png_600dpi', 'pdf_valid', 'four_spines_visible',
    'no_top_title', 'axis_labels_present', 'legend_no_overlap',
)

# 重绘版式梯度：首轮用默认版式；若不达标则放大画布 / 加大下边距 / 隐藏图例 / 缩短标签
RETRY_LAYOUTS = (
    {'tag': '默认版式'},
    {'tag': '放大画布+加大下边距', 'hscale': 1.18, 'vscale': 1.24,
     'bottom_extra': 0.08, 'wspace_extra': 0.06},
    {'tag': '进一步放大+隐藏图例+缩短标签', 'hscale': 1.36, 'vscale': 1.40,
     'bottom_extra': 0.14, 'wspace_extra': 0.12, 'legend_on': False, 'short_labels': True},
)

_TABLE_CACHE: dict = {}


def table(name: str, sheet: str) -> pd.DataFrame:
    """只读取冻结 xlsx 子表（带缓存）。"""
    key = (name, sheet)
    if key not in _TABLE_CACHE:
        _TABLE_CACHE[key] = pd.read_excel(TABLES_DIR / name, sheet_name=sheet)
    return _TABLE_CACHE[key]


def _size(layout: dict, width: float, height: float) -> tuple:
    return (width * layout.get('hscale', 1.0), height * layout.get('vscale', 1.0))


def _adjust(fig, layout: dict, *, left=None, right=None, top=None, bottom=0.18,
            wspace=None, hspace=None) -> None:
    kwargs = {}
    if left is not None:
        kwargs['left'] = left
    if right is not None:
        kwargs['right'] = right
    if top is not None:
        kwargs['top'] = top
    if hspace is not None:
        kwargs['hspace'] = hspace
    if wspace is not None:
        kwargs['wspace'] = wspace + layout.get('wspace_extra', 0.0)
    kwargs['bottom'] = bottom + layout.get('bottom_extra', 0.0)
    fig.subplots_adjust(**kwargs)


def _short(layout: dict) -> bool:
    return bool(layout.get('short_labels', False))


def _legend_on(layout: dict) -> bool:
    return bool(layout.get('legend_on', True))


def run_anchor_validation() -> list:
    """Stage23 锚点统一校验（实现见 ``src/figure_finalize.validate_anchors``，只读冻结产物）。"""
    return figure_finalize.validate_anchors()


def print_anchor_validation(records: list) -> bool:
    return figure_finalize.print_anchor_validation(records)


def read_predictions() -> pd.DataFrame:
    frame = pd.read_parquet(project_paths.PROCESSED_DIR / 'model_predictions.parquet',
                            columns=['split', 'y_true', 'y_pred', 'residual', 'model_name'])
    return frame[frame['split'].eq('test') & frame['model_name'].eq(FINAL_MODEL)].copy()


def final_test_metrics(predictions: pd.DataFrame) -> dict:
    truth = predictions['y_true'].to_numpy(dtype='float64')
    pred = predictions['y_pred'].to_numpy(dtype='float64')
    return {
        'n': int(len(predictions)),
        'MAE': float(np.mean(np.abs(truth - pred))),
        'RMSE': float(np.sqrt(np.mean((truth - pred) ** 2))),
        'R2': float(1.0 - np.sum((truth - pred) ** 2) / np.sum((truth - truth.mean()) ** 2)),
    }


def build_01_sample_structure(layout: dict):
    """图 4-1 正式分析样本与技能提取统计范围结构。"""
    overview = table(TABLE_29, '01_样本概况').set_index('指标')['数值']
    scope = table(TABLE_27, '01_样本统计范围').set_index('统计范围')
    full_jobs = int(overview['全量岗位数（EDA 分析单元）'])
    salary_n = int(overview['正式薪资分析样本'])
    anomaly = int(overview['薪资逻辑异常岗位数'])
    negotiable = int(overview['薪资面议岗位数'])
    layer_values = [int(scope.loc['REQUIREMENT_SECTION', '岗位数']),
                    int(scope.loc['FULL_TEXT_FALLBACK', '岗位数']),
                    int(scope.loc['EMPTY_TEXT', '岗位数'])]

    fig, axes = plt.subplots(1, 2, figsize=_size(layout, 7.8, 4.4))
    ax = axes[0]
    labels = ['全量岗位实体', '薪资可解析', '正式薪资样本']
    values = [full_jobs, salary_n + anomaly, salary_n]
    positions = np.arange(len(labels))[::-1]
    ax.barh(positions, values, height=0.56, color=plot_style.MAIN_COLOR,
            edgecolor='black', linewidth=0.5)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    span = float(max(values))
    for position, value in zip(positions, values):
        ax.text(value + span * 0.02, position, f'{value:,}', va='center', ha='left',
                fontsize=plot_style.FONT_SIZES['annotation'])
    ax.set_xlim(0, span * 1.58)
    ax.set_xlabel('岗位数（个）')
    ax.set_ylabel('样本统计范围层级')
    ax.text(0.98, 0.10, f'未进入正式薪资样本：\n薪资面议 {negotiable:,} 个\n逻辑异常 {anomaly} 个',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=plot_style.FONT_SIZES['annotation'], linespacing=1.6)
    plot_style.apply_sci_axis(ax, grid_axis='x')
    plot_style.format_integer_axis(ax, axis='x')

    ax2 = axes[1]
    labels2 = ['主统计范围：明确要求段落', '扩展统计范围：全文回退', '无可用技能文本']
    positions2 = np.arange(len(labels2))[::-1]
    ax2.barh(positions2, layer_values, height=0.56, color=plot_style.ACCENT_COLOR,
             edgecolor='black', linewidth=0.5)
    ax2.set_yticks(positions2)
    ax2.set_yticklabels(labels2)
    span2 = float(max(layer_values))
    for position, value in zip(positions2, layer_values):
        ax2.text(value + span2 * 0.02, position, f'{value:,}', va='center', ha='left',
                 fontsize=plot_style.FONT_SIZES['annotation'])
    ax2.set_xlim(0, span2 * 1.32)
    ax2.set_xlabel('岗位数（个）')
    ax2.set_ylabel('技能提取统计范围')
    plot_style.apply_sci_axis(ax2, grid_axis='x')
    plot_style.format_integer_axis(ax2, axis='x')

    plot_style.add_subfigure_caption(ax, 'a', '样本统计范围层级（单位：岗位）')
    plot_style.add_subfigure_caption(ax2, 'b', '技能提取三类统计范围（单位：岗位）')
    _adjust(fig, layout, left=0.16, wspace=0.62, bottom=0.32)
    return fig, [('a', '样本统计范围层级（单位：岗位）', ax),
                 ('b', '技能提取三类统计范围（单位：岗位）', ax2)], {
        '图表类型': '横向柱状图（双面板）', '数据来源': '29 号表 01_样本概况；27 号表 01_样本统计范围'}


def build_02_salary_distribution(layout: dict):
    """图 4-3 薪资中点分布与经验累积分布（n = 14,883）。"""
    stats = table(TABLE_29, '02_薪资描述统计').set_index('指标')['数值']
    median = float(stats['中位数'])
    p90 = float(stats['P90'])
    values = pd.read_parquet(project_paths.PROCESSED_DIR / 'job_salary_model_dataset.parquet',
                             columns=[SALARY_COLUMN])[SALARY_COLUMN].dropna().to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=_size(layout, 8.2, 4.2))
    ax = axes[0]
    plot_style.hist_discrete(ax, values, discrete=False, bins=40,
                             xlabel=SALARY_LABEL, ylabel='岗位数')
    ax.axvline(median, color=plot_style.MUTED_COLOR, linestyle='--', linewidth=0.9)
    ax.axvline(p90, color=plot_style.MUTED_COLOR, linestyle=':', linewidth=0.9)
    plot_style.apply_sci_axis(ax, grid_axis='y')
    plot_style.format_integer_axis(ax, axis='y')
    plot_style.annotate_stats(ax, values, y=0.60)
    top = ax.get_ylim()[1]
    ax.text(p90 * 1.03, top * 0.90, f'P90 = {p90:,.0f}', ha='left', va='top',
            fontsize=plot_style.FONT_SIZES['annotation'])
    plot_style.add_subfigure_caption(ax, 'a', '薪资中点分布（虚线为 P50 / P90）')

    ax2 = axes[1]
    ordered = np.sort(values)
    ecdf = np.arange(1, len(ordered) + 1) / len(ordered)
    ax2.plot(ordered, ecdf, color=plot_style.MAIN_COLOR, linewidth=1.1)
    ax2.axhline(0.5, color=plot_style.MUTED_COLOR, linestyle='--', linewidth=0.9)
    ax2.axvline(median, color=plot_style.MUTED_COLOR, linestyle='--', linewidth=0.9)
    ax2.text(median * 1.06, 0.52, f'中位数 = {median:,.0f}', ha='left', va='bottom',
             fontsize=plot_style.FONT_SIZES['annotation'])
    ax2.set_xlabel(SALARY_LABEL)
    ax2.set_ylabel('累计占比')
    ax2.set_ylim(0, 1.02)
    plot_style.apply_sci_axis(ax2)
    plot_style.format_percent_axis(ax2, axis='y')
    plot_style.add_subfigure_caption(ax2, 'b', '薪资中点经验累积分布')
    _adjust(fig, layout, left=0.11, wspace=0.30, bottom=0.32)
    return fig, [('a', '薪资中点分布（虚线为 P50 / P90）', ax),
                 ('b', '薪资中点经验累积分布', ax2)], {
        '图表类型': '直方图 + 经验累积分布',
        '数据来源': '29 号表 02_薪资描述统计；job_salary_model_dataset（薪资中点）'}


def build_03_category_salary(layout: dict):
    """图 4-2 主要岗位细分类薪资中点中位数（误差线为 IQR/2）。"""
    frame = table(TABLE_29, '03_岗位因素薪资')
    frame = frame[frame['因素'].eq('岗位细分类')]
    frame = frame.sort_values('中位数', ascending=False).head(12).sort_values('中位数')

    fig, ax = plt.subplots(figsize=_size(layout, 7.4, 5.2))
    positions = np.arange(len(frame))
    ax.barh(positions, frame['中位数'], height=0.62, color=plot_style.MAIN_COLOR,
            edgecolor='black', linewidth=0.5, xerr=frame['IQR'] / 2,
            error_kw={'elinewidth': 0.8, 'capsize': 2.5})
    ax.set_yticks(positions)
    if _short(layout):
        ax.set_yticklabels([str(item) for item in frame['取值']])
    else:
        ax.set_yticklabels([f'{name}（n={int(count):,}）' for name, count
                            in zip(frame['取值'], frame['样本数'])])
    ax.set_xlim(0, float(frame['中位数'].max()) * 1.35)
    ax.set_xlabel(SALARY_LABEL)
    ax.set_ylabel('岗位细分类（按样本数降序，自上而下）')
    plot_style.apply_sci_axis(ax, grid_axis='x')
    _adjust(fig, layout, left=0.30, bottom=0.20)
    return fig, [], {'图表类型': '横向柱状图 + IQR/2 误差线',
                     '数据来源': f'{TABLE_29} / 03_岗位因素薪资（岗位细分类）'}


def build_04_structured_factor_salary(layout: dict):
    """图 5-1 城市、学历与公司规模的薪资中点中位数。"""
    city = table(TABLE_29, '04_城市薪资')
    education = table(TABLE_29, '05_学历薪资')
    company = table(TABLE_29, '06_公司因素薪资')
    panels = [('a', '工作城市', '城市', city, 8), ('b', '学历要求', '学历要求', education, 8),
              ('c', '公司规模', '公司规模', company, 8)]

    fig, axes = plt.subplots(1, 3, figsize=_size(layout, 11.4, 4.4))
    subfigures = []
    for (letter, factor, label, frame, top_n), ax in zip(panels, axes):
        block = frame[frame['因素'].eq(factor)].head(top_n).iloc[::-1]
        positions = np.arange(len(block))
        ax.barh(positions, block['中位数'], height=0.60, color=plot_style.MAIN_COLOR,
                edgecolor='black', linewidth=0.5)
        ax.set_yticks(positions)
        ax.set_yticklabels([str(item) for item in block['取值']])
        ax.set_xlim(0, float(block['中位数'].max()) * 1.22)
        ax.set_xlabel(SALARY_LABEL)
        ax.set_ylabel(f'{label}（取值）')
        for position, value in zip(positions, block['中位数']):
            ax.text(value + float(block['中位数'].max()) * 0.02, position, f'{value:,.0f}',
                    va='center', ha='left', fontsize=plot_style.FONT_SIZES['annotation'])
        plot_style.apply_sci_axis(ax, grid_axis='x')
        plot_style.format_integer_axis(ax, axis='x')
        caption = f'{label}薪资中点中位数'
        plot_style.add_subfigure_caption(ax, letter, caption)
        subfigures.append((letter, caption, ax))
    _adjust(fig, layout, left=0.07, wspace=0.72, bottom=0.32)
    return fig, subfigures, {'图表类型': '三联横向柱状图',
                            '数据来源': f'{TABLE_29} / 04_城市薪资、05_学历薪资、06_公司因素薪资'}


def build_fig_6_1_tech_skill_top20(layout: dict):
    """图 6-1 核心技术技能需求 Top20（技能主统计范围，分母 8,822）。"""
    demand = table(TABLE_29, '07_技能需求')
    frame = demand[demand['榜单'].str.startswith('核心技术技能')].head(20)
    series = frame.set_index('技能标准名')['岗位数']
    fig, ax = plt.subplots(figsize=_size(layout, 7.4, 5.6))
    plot_style.barh_ranked(ax, series, xlabel='明确要求该技能的岗位数（个）',
                           ylabel='具体技术技能')
    ax.set_xlim(0, float(series.max()) * 1.20)
    plot_style.apply_sci_axis(ax, grid_axis='x')
    plot_style.format_integer_axis(ax, axis='x')
    _adjust(fig, layout, left=0.30, bottom=0.20)
    return fig, [], {'图表类型': '横向柱状图',
                     '数据来源': f'{TABLE_29} / 07_技能需求（具体技术技能）'}


def build_06_skill_layer_structure(layout: dict):
    """附图 A-1 技能需求的分层结构。"""
    demand = table(TABLE_29, '07_技能需求')
    panels = [('a', '技术领域', '技术领域 Top15', 12),
              ('b', '业务能力', '业务能力', 8),
              ('c', '办公工具', '办公工具', 8)]
    fig, axes = plt.subplots(1, 3, figsize=_size(layout, 11.4, 4.6))
    subfigures = []
    for (letter, factor, prefix, top_n), ax in zip(panels, axes):
        frame = demand[demand['榜单'].str.startswith(prefix)].head(top_n)
        series = frame.set_index('技能标准名')['岗位数']
        plot_style.barh_ranked(ax, series, color=plot_style.ACCENT_COLOR,
                               xlabel='岗位数（个，主统计范围）', ylabel=factor)
        ax.set_xlim(0, float(series.max()) * 1.28)
        plot_style.apply_sci_axis(ax, grid_axis='x')
        plot_style.format_integer_axis(ax, axis='x')
        caption = f'{factor}需求岗位数'
        plot_style.add_subfigure_caption(ax, letter, caption)
        subfigures.append((letter, caption, ax))
    _adjust(fig, layout, left=0.06, wspace=0.74, bottom=0.32)
    return fig, subfigures, {'图表类型': '三联横向柱状图',
                            '数据来源': f'{TABLE_29} / 07_技能需求（技术领域 / 业务能力 / 办公工具）'}


def build_07_category_skill_heatmap(layout: dict):
    """图 6-2 岗位细分类 × 技能命中率热力图。"""
    matrix = table(TABLE_27, '08_岗位类别技能画像')
    counts = matrix['细分类岗位数'].to_numpy()
    values = matrix.drop(columns=['细分类岗位数']).set_index('岗位细分类')
    data = values.to_numpy(dtype='float64')

    fig, ax = plt.subplots(figsize=_size(layout, 10.2, 5.6))
    image = ax.imshow(data, aspect='auto', cmap='cividis', vmin=0.0, vmax=float(np.nanmax(data)))
    ax.set_xticks(np.arange(values.shape[1]))
    ax.set_xticklabels([str(item) for item in values.columns], rotation=60, ha='right')
    ax.set_yticks(np.arange(values.shape[0]))
    ax.set_yticklabels([f'{name}（n={int(count):,}）' for name, count
                        in zip(values.index, counts)])
    ax.set_xlabel('具体技术技能（命中率前 20）')
    ax.set_ylabel('岗位细分类（按细分类岗位数降序）')
    plot_style.apply_sci_axis(ax, grid=False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.026, pad=0.02)
    colorbar.set_label('细分类内部技能命中率（比例）',
                       fontsize=plot_style.FONT_SIZES['axis_label'])
    _adjust(fig, layout, left=0.20, bottom=0.34)
    return fig, [], {'图表类型': '热力图', '数据来源': f'{TABLE_27} / 08_岗位类别技能画像'}


def build_08_skill_cooccurrence(layout: dict):
    """图 6-3 技能共现（共现岗位数 ≥ 30 的技能对）。"""
    cooc = table(TABLE_29, '10_技能共现')
    skills: list = []
    for row in cooc.head(40).itertuples(index=False):
        for skill in (row.技能A, row.技能B):
            if skill not in skills:
                skills.append(skill)
    skills = skills[:15]
    matrix = pd.DataFrame(np.nan, index=skills, columns=skills)
    for row in cooc.itertuples(index=False):
        if row.技能A in skills and row.技能B in skills:
            matrix.loc[row.技能A, row.技能B] = row.Lift
            matrix.loc[row.技能B, row.技能A] = row.Lift
    np.fill_diagonal(matrix.values, 1.0)
    data = matrix.to_numpy(dtype='float64')

    fig, ax = plt.subplots(figsize=_size(layout, 8.8, 6.2))
    image = ax.imshow(data, cmap='magma', vmin=1.0, vmax=float(np.nanmax(data)))
    ax.set_xticks(np.arange(len(skills)))
    ax.set_xticklabels(skills, rotation=60, ha='right')
    ax.set_yticks(np.arange(len(skills)))
    ax.set_yticklabels(skills)
    ax.set_xlabel('具体技术技能（按共现 Lift 排序取前 15）')
    ax.set_ylabel('具体技术技能')
    plot_style.apply_sci_axis(ax, grid=False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label('共现 Lift', fontsize=plot_style.FONT_SIZES['axis_label'])
    _adjust(fig, layout, left=0.20, bottom=0.28)
    return fig, [], {'图表类型': '热力图',
                     '数据来源': f'{TABLE_29} / 10_技能共现（共现岗位数 ≥ 30）'}


def build_09_skill_salary(layout: dict):
    """附图 A-6 技能与薪资的描述性关联（中位数差，非因果）。"""
    per_skill = table(TABLE_29, '08_技能薪资')
    frame = per_skill.sort_values('中位数差', ascending=False).head(12).iloc[::-1]
    positions = np.arange(len(frame))
    colors = [plot_style.MAIN_COLOR if value >= 0 else plot_style.MUTED_COLOR
              for value in frame['中位数差']]

    fig, axes = plt.subplots(1, 2, figsize=_size(layout, 11.2, 5.0))
    ax = axes[0]
    ax.barh(positions, frame['中位数差'], height=0.60, color=colors,
            edgecolor='black', linewidth=0.5)
    ax.set_yticks(positions)
    if _short(layout):
        ax.set_yticklabels([str(item) for item in frame['技能标准名']])
    else:
        ax.set_yticklabels([f'{name}（n={int(count):,}）' for name, count
                            in zip(frame['技能标准名'], frame['有技能岗位数'])])
    span = float(frame['中位数差'].abs().max())
    for position, value in zip(positions, frame['中位数差']):
        offset = span * 0.03
        ax.text(value + (offset if value >= 0 else -offset), position, f'{value:,.0f}',
                va='center', ha='left' if value >= 0 else 'right',
                fontsize=plot_style.FONT_SIZES['annotation'])
    ax.set_xlim(-span * 1.35, span * 1.35)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('中位数差（有技能 − 无技能，元/天）')
    ax.set_ylabel('技能（按中位数差降序，自上而下）')
    plot_style.apply_sci_axis(ax, grid_axis='x')
    plot_style.format_integer_axis(ax, axis='x')
    plot_style.add_subfigure_caption(ax, 'a', '技能有/无的薪资中位数差排序')

    ax2 = axes[1]
    ax2.barh(positions, frame["Cliff's delta"], height=0.60, color=plot_style.ACCENT_COLOR,
             edgecolor='black', linewidth=0.5)
    ax2.set_yticks(positions)
    ax2.set_yticklabels([str(item) for item in frame['技能标准名']])
    ax2.axvline(0, color='black', linewidth=0.8)
    span2 = float(frame["Cliff's delta"].abs().max())
    ax2.set_xlim(0, span2 * 1.30)
    for position, value in zip(positions, frame["Cliff's delta"]):
        ax2.text(value + span2 * 0.03, position, f'{value:.3f}', va='center', ha='left',
                 fontsize=plot_style.FONT_SIZES['annotation'])
    significant = int(frame['FDR显著'].eq('是').sum())
    ax2.set_xlabel("Cliff's δ（效应量，正值表示有技能组薪资更靠前）")
    ax2.set_ylabel('技能（与左图同序）')
    ax2.text(0.98, 0.03, f'BH-FDR q < 0.05：{significant}/{len(frame)}；'
                         '描述性关联，非因果',
             transform=ax2.transAxes, ha='right', va='bottom',
             fontsize=plot_style.FONT_SIZES['annotation'])
    plot_style.apply_sci_axis(ax2, grid_axis='x')
    plot_style.add_subfigure_caption(ax2, 'b', "Cliff's δ 效应量与 FDR 显著性")
    _adjust(fig, layout, left=0.17, wspace=0.62, bottom=0.30)
    return fig, [('a', '技能有/无的薪资中位数差排序', ax),
                 ('b', "Cliff's δ 效应量与 FDR 显著性", ax2)], {
        '图表类型': '双面板横向柱状图',
        '数据来源': f'{TABLE_29} / 08_技能薪资（中位数差 Top12）'}


def build_10_scope_robustness(layout: dict):
    """附图 A-2 技能提取两种统计范围稳健性。"""
    summary_27 = table(TABLE_27, '11_两种统计范围稳健性')
    summary_27 = summary_27[summary_27['分析块'].eq('稳健性汇总')].set_index('技能标准名或指标')['数值']
    summary_29 = table(TABLE_29, '12_稳健性')
    summary_29 = summary_29[summary_29['项目'].isin(
        ['共有技能数', 'Top10 overlap', 'Top10 overlap 比例', 'Top20 overlap',
         'Top20 overlap 比例', 'Spearman 排名相关'])].set_index('项目')['数值']
    detail = table(TABLE_29, '12_稳健性')
    detail = detail[detail['项目'].eq('Top50 技能排名对照')]

    top10 = float(summary_27['Top10 overlap 比例'])
    top20 = float(summary_27['Top20 overlap 比例'])
    spearman = float(summary_27['Spearman 排名相关'])
    shared_skills = int(summary_27['共有技能数'])
    cross_check = (abs(float(summary_29['Top10 overlap 比例']) - top10) < 1e-9
                   and abs(float(summary_29['Top20 overlap 比例']) - top20) < 1e-9
                   and abs(float(summary_29['Spearman 排名相关']) - spearman) < 1e-9)
    print(f"[交叉核对] 附图 A-2 两种统计范围稳健性：29 号表 12_稳健性 与 27 号表 11_两种统计范围稳健性 "
          f"汇总值一致 = {cross_check}（Top10 {top10:.2f} / Top20 {top20:.2f} / "
          f"Spearman {spearman:.6f} / 共有技能 {shared_skills}）")

    fig, axes = plt.subplots(1, 2, figsize=_size(layout, 10.4, 4.4))
    ax = axes[0]
    xs = detail['主统计范围排名'].to_numpy(dtype='float64')
    ys = detail['扩展统计范围排名'].to_numpy(dtype='float64')
    ax.scatter(xs, ys, s=14, color=plot_style.MAIN_COLOR, alpha=0.75, edgecolor='none')
    limit = float(max(xs.max(), ys.max())) * 1.05
    ax.plot([0, limit], [0, limit], color=plot_style.MUTED_COLOR, linestyle='--', linewidth=0.9)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel('主统计范围排名（明确要求段落）')
    ax.set_ylabel('扩展统计范围排名（全文回退统计范围）')
    ax.text(0.97, 0.06, f'Spearman ρ = {spearman:.3f}\n共有技能 {shared_skills} 个',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=plot_style.FONT_SIZES['annotation'], linespacing=1.5)
    plot_style.apply_sci_axis(ax)
    plot_style.format_integer_axis(ax)
    plot_style.add_subfigure_caption(ax, 'a', '两种统计范围技能排名对照（Top50）')

    ax2 = axes[1]
    labels = ['Top10 重合率', 'Top20 重合率']
    values = [top10, top20]
    positions = np.arange(len(labels))
    ax2.bar(positions, values, width=0.48, color=plot_style.ACCENT_COLOR,
            edgecolor='black', linewidth=0.5)
    for position, value in zip(positions, values):
        ax2.text(position, value + 0.02, f'{value:.0%}', ha='center', va='bottom',
                 fontsize=plot_style.FONT_SIZES['annotation'])
    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 1.12)
    ax2.axhline(spearman, color=plot_style.MUTED_COLOR, linestyle='--', linewidth=0.9)
    ax2.text(0.98, 0.86, f'Spearman ρ = {spearman:.3f}（虚线）', transform=ax2.transAxes,
             ha='right', va='top', fontsize=plot_style.FONT_SIZES['annotation'])
    ax2.set_xlabel('一致性指标')
    ax2.set_ylabel('重合比例')
    plot_style.apply_sci_axis(ax2, grid_axis='y')
    plot_style.format_percent_axis(ax2, axis='y')
    plot_style.add_subfigure_caption(ax2, 'b', 'Top-K 重合率与排名相关')
    _adjust(fig, layout, left=0.10, wspace=0.34, bottom=0.30)
    return fig, [('a', '两种统计范围技能排名对照（Top50）', ax),
                 ('b', 'Top-K 重合率与排名相关', ax2)], {
        '图表类型': '散点图 + 柱状图',
        '数据来源': f'{TABLE_29} / 12_稳健性；{TABLE_27} / 11_两种统计范围稳健性'}


def build_11_model_comparison(layout: dict):
    """图 7-1 模型验证集 MAE / RMSE 比较。"""
    validation = table(TABLE_30, '08_Validation比较')
    test = table(TABLE_30, '09_Test最终结果')
    comparison = validation[['模型', 'validation_MAE', 'validation_RMSE']].merge(
        test.loc[~test['模型'].eq(FINAL_MODEL), ['模型', 'test_MAE', 'test_RMSE']],
        on='模型', how='left')
    final_row = test[test['模型'].eq(FINAL_MODEL)].iloc[0]
    labels = comparison['模型'].tolist()
    positions = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=_size(layout, 11.0, 4.6))
    ax = axes[0]
    ax.bar(positions, comparison['validation_MAE'], width=0.56, color=plot_style.MAIN_COLOR,
           edgecolor='black', linewidth=0.5)
    ceiling = float(comparison['validation_MAE'].max())
    for position, value in zip(positions, comparison['validation_MAE']):
        ax.text(position, value + ceiling * 0.015, f'{value:.2f}', ha='center', va='bottom',
                fontsize=plot_style.FONT_SIZES['annotation'])
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylim(0, ceiling * 1.28)
    ax.set_xlabel('模型')
    ax.set_ylabel('MAE（元/天）')
    plot_style.apply_sci_axis(ax, grid_axis='y')
    plot_style.add_subfigure_caption(ax, 'a', '验证集 MAE（选模依据）')

    ax2 = axes[1]
    test_values = comparison['test_MAE'].tolist() + [float(final_row['test_MAE'])]
    test_labels = labels + [f'{FINAL_MODEL}\n（训练集重拟合）']
    test_positions = np.arange(len(test_values))
    colors = [plot_style.MAIN_COLOR] * len(labels) + [plot_style.ACCENT_COLOR]
    ax2.bar(test_positions, test_values, width=0.58, color=colors,
            edgecolor='black', linewidth=0.5)
    ceiling2 = float(max(test_values))
    for position, value in zip(test_positions, test_values):
        ax2.text(position, value + ceiling2 * 0.015, f'{value:.2f}', ha='center', va='bottom',
                 fontsize=plot_style.FONT_SIZES['annotation'])
    ax2.set_xticks(test_positions)
    ax2.set_xticklabels(test_labels, rotation=15, ha='right')
    ax2.set_ylim(0, ceiling2 * 1.30)
    ax2.set_xlabel('模型')
    ax2.set_ylabel('MAE（元/天）')
    if _legend_on(layout):
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=plot_style.MAIN_COLOR,
                                 edgecolor='black', linewidth=0.5, label='调参对照模型'),
                   plt.Rectangle((0, 0), 1, 1, facecolor=plot_style.ACCENT_COLOR,
                                 edgecolor='black', linewidth=0.5, label='正式主模型')]
        ax2.legend(handles=handles, loc='upper left', frameon=False)
    plot_style.apply_sci_axis(ax2, grid_axis='y')
    plot_style.add_subfigure_caption(ax2, 'b', '测试集 MAE（主模型为最终结果）')
    _adjust(fig, layout, left=0.08, wspace=0.28, bottom=0.34)
    return fig, [('a', '验证集 MAE（选模依据）', ax),
                 ('b', '测试集 MAE（主模型为最终结果）', ax2)], {
        '图表类型': '双面板柱状图',
        '数据来源': f'{TABLE_30} / 08_Validation比较、09_Test最终结果'}


def build_12_prediction_scatter(layout: dict):
    """附图 A-3 正式主模型 test 预测值 vs 真实值。"""
    predictions = read_predictions()
    metrics = final_test_metrics(predictions)
    truth = predictions['y_true'].to_numpy(dtype='float64')
    pred = predictions['y_pred'].to_numpy(dtype='float64')

    fig, ax = plt.subplots(figsize=_size(layout, 5.8, 5.2))
    ax.scatter(truth, pred, s=7, alpha=0.35, color=plot_style.MAIN_COLOR, edgecolor='none')
    low = float(min(truth.min(), pred.min()))
    high = float(max(truth.max(), pred.max()))
    ax.plot([low, high], [low, high], color=plot_style.MUTED_COLOR, linestyle='--',
            linewidth=0.9)
    ax.set_xlim(low * 0.95, high * 1.02)
    ax.set_ylim(low * 0.95, high * 1.02)
    ax.text(0.04, 0.96, f"MAE = {metrics['MAE']:.2f} 元/天\nR² = {metrics['R2']:.3f}\n"
                        f"n = {metrics['n']:,}",
            transform=ax.transAxes, va='top', ha='left',
            fontsize=plot_style.FONT_SIZES['annotation'], linespacing=1.5)
    ax.set_xlabel('真实薪资中点（元/天）')
    ax.set_ylabel('预测薪资中点（元/天）')
    plot_style.apply_sci_axis(ax)
    plot_style.format_integer_axis(ax)
    _adjust(fig, layout, left=0.18, bottom=0.20)
    return fig, [], {'图表类型': '散点图 + y = x 参考线',
                     '数据来源': 'model_predictions.parquet（split = test，正式主模型）'}


def build_13_residual_diagnostics(layout: dict):
    """附图 A-4 正式主模型 test 残差诊断。"""
    predictions = read_predictions()
    residual = predictions['residual'].to_numpy(dtype='float64')
    pred = predictions['y_pred'].to_numpy(dtype='float64')

    fig, axes = plt.subplots(1, 2, figsize=_size(layout, 10.2, 4.2))
    ax = axes[0]
    plot_style.hist_discrete(ax, residual, discrete=False, bins=50,
                             xlabel='残差（真实 − 预测，元/天）', ylabel='岗位数')
    plot_style.apply_sci_axis(ax, grid_axis='y')
    plot_style.format_integer_axis(ax, axis='y')
    ax.axvline(float(np.median(residual)), color=plot_style.MUTED_COLOR, linestyle='--',
               linewidth=0.9)
    plot_style.add_subfigure_caption(ax, 'a', '残差分布（虚线为中位数）')

    ax2 = axes[1]
    ax2.scatter(pred, residual, s=7, alpha=0.30, color=plot_style.MAIN_COLOR, edgecolor='none')
    ax2.axhline(0, color=plot_style.MUTED_COLOR, linestyle='--', linewidth=0.9)
    ax2.set_xlabel('预测薪资中点（元/天）')
    ax2.set_ylabel('残差（元/天）')
    plot_style.apply_sci_axis(ax2)
    plot_style.format_integer_axis(ax2, axis='x')
    plot_style.add_subfigure_caption(ax2, 'b', '残差 vs 预测值')
    _adjust(fig, layout, left=0.09, wspace=0.28, bottom=0.32)
    return fig, [('a', '残差分布（虚线为中位数）', ax), ('b', '残差 vs 预测值', ax2)], {
        '图表类型': '直方图 + 散点图',
        '数据来源': 'model_predictions.parquet（split = test，正式主模型）'}


def build_14_error_groups(layout: dict):
    """附图 A-5 正式主模型 test 分组误差诊断。"""
    residual_table = table(TABLE_30, '10_预测残差')
    residual_table = residual_table[residual_table['维度'].notna()]
    group_table = table(TABLE_31, '09_分组误差')
    panels = [('a', '学历要求', residual_table, '学历要求分组 MAE'),
              ('b', '公司规模', residual_table, '公司规模分组 MAE'),
              ('c', '岗位大类', group_table, '跨公司子集分组 MAE')]

    fig, axes = plt.subplots(1, 3, figsize=_size(layout, 11.6, 4.4))
    subfigures = []
    for (letter, dimension, source, caption), ax in zip(panels, axes):
        block = source[source['维度'].eq(dimension)].sort_values('MAE')
        positions = np.arange(len(block))
        ax.barh(positions, block['MAE'], height=0.60, color=plot_style.MAIN_COLOR,
                edgecolor='black', linewidth=0.5)
        ax.set_yticks(positions)
        if _short(layout):
            ax.set_yticklabels([str(item) for item in block['取值']])
        else:
            ax.set_yticklabels([f'{name}（n={int(count):,}）' for name, count
                                in zip(block['取值'], block['样本数'])])
        ax.set_xlim(0, float(block['MAE'].max()) * 1.24)
        ax.set_xlabel('MAE（元/天）')
        ax.set_ylabel({'a': '学历要求', 'b': '公司规模', 'c': '岗位大类（跨公司子集）'}[letter])
        plot_style.apply_sci_axis(ax, grid_axis='x')
        plot_style.add_subfigure_caption(ax, letter, caption)
        subfigures.append((letter, caption, ax))
    _adjust(fig, layout, left=0.06, wspace=0.78, bottom=0.32)
    return fig, subfigures, {'图表类型': '三联横向柱状图',
                            '数据来源': f'{TABLE_30} / 10_预测残差；{TABLE_31} / 09_分组误差'}


def build_15_ablation_results(layout: dict):
    """图 8-1 特征组消融（四配置 MAE 与技能增量 bootstrap 95% 置信区间）。"""
    ablation = table(TABLE_31, '01_消融结果')
    increment = table(TABLE_31, '02_技能增量')
    labels = ablation['配置'].tolist()
    positions = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=_size(layout, 11.8, 4.8))
    ax = axes[0]
    ax.bar(positions - 0.19, ablation['validation_MAE'], width=0.36,
           color=plot_style.MAIN_COLOR, edgecolor='black', linewidth=0.5,
           label='validation MAE')
    ax.bar(positions + 0.19, ablation['test_MAE'], width=0.36, color=plot_style.ACCENT_COLOR,
           edgecolor='black', linewidth=0.5, label='test MAE')
    ceiling = float(ablation[['validation_MAE', 'test_MAE']].to_numpy().max())
    for position, value in zip(positions - 0.19, ablation['validation_MAE']):
        ax.text(position, value + ceiling * 0.015, f'{value:.2f}', ha='center', va='bottom',
                fontsize=plot_style.FONT_SIZES['annotation'])
    for position, value in zip(positions + 0.19, ablation['test_MAE']):
        ax.text(position, value + ceiling * 0.015, f'{value:.2f}', ha='center', va='bottom',
                fontsize=plot_style.FONT_SIZES['annotation'])
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, ceiling * 1.30)
    ax.set_xlabel('特征组配置')
    ax.set_ylabel('MAE（元/天）')
    if _legend_on(layout):
        ax.legend(loc='upper left', frameon=False)
    plot_style.apply_sci_axis(ax, grid_axis='y')
    plot_style.add_subfigure_caption(ax, 'a', '四配置 validation / test MAE')

    ax2 = axes[1]
    rows = increment.to_dict('records')
    row_labels = [f"{row['比较（加入特征组后）']}\n（{row['数据子集']}）" for row in rows]
    deltas = np.array([row['ΔMAE（无该组 − 有该组）'] for row in rows], dtype='float64')
    lower = np.array([row['bootstrap_CI95_下界'] for row in rows], dtype='float64')
    upper = np.array([row['bootstrap_CI95_上界'] for row in rows], dtype='float64')
    y_positions = np.arange(len(rows))[::-1]
    ax2.errorbar(deltas, y_positions, xerr=np.vstack([deltas - lower, upper - deltas]),
                 fmt='o', markersize=5, color=plot_style.MAIN_COLOR,
                 ecolor=plot_style.MUTED_COLOR, elinewidth=1.2, capsize=3)
    ax2.axvline(0, color='black', linestyle='--', linewidth=0.9)
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(row_labels, fontsize=plot_style.FONT_SIZES['tick'])
    span = float(max(abs(lower).max(), abs(upper).max())) * 1.45
    ax2.set_xlim(-span, span)
    ax2.set_xlabel('ΔMAE（无该组 − 有该组，元/天）')
    ax2.set_ylabel('技能组增量（无该组 − 有该组）')
    plot_style.apply_sci_axis(ax2, grid_axis='x')
    plot_style.add_subfigure_caption(ax2, 'b', '技能增量 ΔMAE 与 bootstrap 95% 置信区间')
    _adjust(fig, layout, left=0.07, wspace=0.42, bottom=0.32)
    return fig, [('a', '四配置 validation / test MAE', ax),
                 ('b', '技能增量 ΔMAE 与 bootstrap 95% 置信区间', ax2)], {
        '图表类型': '柱状图 + 森林图',
        '数据来源': f'{TABLE_31} / 01_消融结果、02_技能增量'}


def build_16_random_vs_group_split(layout: dict):
    """图 8-2 随机划分与按公司分组划分的预测误差比较（test）。"""
    group_table = table(TABLE_31, '04_Random_vs_GroupSplit')
    frame = group_table[group_table['数据子集'].eq('test')
                        & ~group_table['划分方式'].str.startswith('差异')]
    labels = frame['划分方式'].tolist()
    positions = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=_size(layout, 9.0, 4.4))
    ax = axes[0]
    ax.bar(positions, frame['MAE'], width=0.52, color=plot_style.MAIN_COLOR,
           edgecolor='black', linewidth=0.5)
    ceiling = float(frame['MAE'].max())
    for position, value in zip(positions, frame['MAE']):
        ax.text(position, value + ceiling * 0.02, f'{value:.2f}', ha='center', va='bottom',
                fontsize=plot_style.FONT_SIZES['annotation'])
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0, ceiling * 1.26)
    ax.set_xlabel('数据划分方式（test 子集）')
    ax.set_ylabel('MAE（元/天）')
    plot_style.apply_sci_axis(ax, grid_axis='y')
    plot_style.add_subfigure_caption(ax, 'a', 'MAE 对比（跨公司泛化更难）')

    ax2 = axes[1]
    ax2.bar(positions, frame['R²'], width=0.52, color=plot_style.ACCENT_COLOR,
            edgecolor='black', linewidth=0.5)
    ceiling2 = float(frame['R²'].max())
    for position, value in zip(positions, frame['R²']):
        ax2.text(position, value + ceiling2 * 0.03, f'{value:.3f}', ha='center', va='bottom',
                 fontsize=plot_style.FONT_SIZES['annotation'])
    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, ceiling2 * 1.30)
    ax2.set_xlabel('数据划分方式（test 子集）')
    ax2.set_ylabel('R²')
    plot_style.apply_sci_axis(ax2, grid_axis='y')
    plot_style.add_subfigure_caption(ax2, 'b', 'R² 对比（同分布 vs 跨公司）')
    _adjust(fig, layout, left=0.09, wspace=0.30, bottom=0.34)
    return fig, [('a', 'MAE 对比（跨公司泛化更难）', ax),
                 ('b', 'R² 对比（同分布 vs 跨公司）', ax2)], {
        '图表类型': '双面板柱状图',
        '数据来源': f'{TABLE_31} / 04_Random_vs_GroupSplit（test）'}


def build_17_skill_shap_top20(layout: dict):
    """图 8-4 技能特征 SHAP Top20（条长 = mean|SHAP|，标记 = presence_direction）。"""
    skill_table = table(TABLE_31, '08_技能SHAP')
    frame = skill_table.head(20).iloc[::-1]
    positions = np.arange(len(frame))

    fig, ax = plt.subplots(figsize=_size(layout, 8.8, 5.8))
    ax.barh(positions, frame['mean_abs_SHAP'], height=0.66, color=plot_style.MUTED_COLOR,
            edgecolor='black', linewidth=0.5)
    for position, (value, direction) in enumerate(zip(frame['mean_abs_SHAP'],
                                                      frame['presence_direction'])):
        marker = '^' if str(direction).startswith('正向') else (
            'v' if str(direction).startswith('负向') else 'o')
        ax.scatter([value * 0.03], [position], marker=marker, s=26, color='black', zorder=3)
    ax.set_yticks(positions)
    if _short(layout):
        ax.set_yticklabels([str(item) for item in frame['技能']])
    else:
        ax.set_yticklabels([f'{name}（岗位频率 {freq:.1%}）' for name, freq
                            in zip(frame['技能'], frame['model_job_frequency'])])
    ax.set_xlim(0, float(frame['mean_abs_SHAP'].max()) * 1.22)
    ax.set_xlabel('mean|SHAP|（元/天，技能特征的平均绝对贡献）')
    ax.set_ylabel('技能（按 mean|SHAP| 排序，Top20）')
    ax.text(0.97, 0.03, '标记：▲ 该技能存在时平均贡献为正，▼ 为负\n'
                        '方向仅为模型预测贡献方向，不代表因果效应',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=plot_style.FONT_SIZES['annotation'], linespacing=1.5)
    plot_style.apply_sci_axis(ax, grid_axis='x')
    _adjust(fig, layout, left=0.38, bottom=0.20)
    return fig, [], {'图表类型': '横向柱状图（含方向标记）',
                     '数据来源': f'{TABLE_31} / 08_技能SHAP（必要时并用 12_关键技能SHAP）'}


# （组别, 文件名, Stage23 正式图题（展示版底部图题）, 绘图函数, 回答的问题, 正文去向）
figure_specs = [
    ('eda', '01_sample_structure', '图 4-1 正式分析样本与技能提取统计范围结构',
     build_01_sample_structure, '正式分析样本与技能提取统计范围的结构', '正文 图 4-1（4.1 节）'),
    ('eda', '02_salary_distribution', '图 4-3 薪资中点分布与经验累积分布（n = 14,883）',
     build_02_salary_distribution, '薪资中点分布形态与累积结构', '正文 图 4-3（4.4 节）'),
    ('eda', '03_category_salary', '图 4-2 主要岗位细分类薪资中点中位数（误差线为 IQR/2）',
     build_03_category_salary, '主要岗位细分类的薪资中点中位数', '正文 图 4-2（4.2 节）'),
    ('eda', '04_structured_factor_salary', '图 5-1 城市、学历与公司规模的薪资中点中位数',
     build_04_structured_factor_salary, '结构化因素的薪资中点中位数差异', '正文 图 5-1（5.3 节）'),
    ('eda', 'fig_6_1_tech_skill_top20', '图 6-1 核心技术技能需求 Top20（技能主统计范围，分母 8,822）',
     build_fig_6_1_tech_skill_top20, '企业明确要求的核心技术技能', '正文 图 6-1（6.2 节）'),
    ('eda', '06_skill_layer_structure', '附图 A-1 技能需求的分层结构',
     build_06_skill_layer_structure, '技术领域 / 业务能力 / 办公工具的分层需求', '附录'),
    ('eda', '07_category_skill_heatmap', '图 6-2 岗位细分类 × 技能命中率热力图',
     build_07_category_skill_heatmap, '岗位细分类与技能命中率的对应关系', '正文 图 6-2（6.3 节）'),
    ('eda', '08_skill_cooccurrence', '图 6-3 技能共现（共现岗位数 ≥ 30 的技能对）',
     build_08_skill_cooccurrence, '具体技术技能的共现结构', '正文 图 6-3（6.4 节）'),
    ('eda', '09_skill_salary', '附图 A-6 技能与薪资的描述性关联（未控制岗位类别）',
     build_09_skill_salary, '技能与薪资的描述性关联强度', '附录（已从正文移入附录）'),
    ('eda', '10_scope_robustness', '附图 A-2 技能提取两种统计范围稳健性',
     build_10_scope_robustness, '技能结论对文本提取统计范围是否稳健', '附录'),
    ('modeling', '01_model_comparison', '图 7-1 模型验证集 MAE / RMSE 比较',
     build_11_model_comparison, '模型验证集与测试集的误差比较', '正文 图 7-1（7.4 节）'),
    ('modeling', '02_prediction_scatter', '附图 A-3 正式主模型 test 预测值 vs 真实值',
     build_12_prediction_scatter, '正式主模型在测试集上的拟合质量', '附录'),
    ('modeling', '03_residual_diagnostics', '附图 A-4 正式主模型 test 残差诊断',
     build_13_residual_diagnostics, '正式主模型的残差结构', '附录'),
    ('modeling', '04_error_groups', '附图 A-5 正式主模型 test 分组误差诊断',
     build_14_error_groups, '分组误差是否被总体 MAE 掩盖', '附录'),
    ('modeling', '05_ablation_results',
     '图 8-1 特征组消融：四配置 MAE 与技能增量 bootstrap 95% 置信区间',
     build_15_ablation_results, '特征组消融与技能增量显著性', '正文 图 8-1（8.1 节）'),
    ('modeling', '06_random_vs_group_split',
     '图 8-2 随机划分与按公司分组划分的预测误差比较（test）',
     build_16_random_vs_group_split, '同分布预测与跨公司泛化的差距', '正文 图 8-2（8.2 节）'),
    ('modeling', '08_skill_shap_top20',
     '图 8-4 技能特征 SHAP Top20（条长 = mean|SHAP|，标记 = presence_direction）',
     build_17_skill_shap_top20, '技能特征的预测贡献强度与方向', '正文 图 8-4（8.4.2 节）'),
]


def render(group: str, stem: str, caption: str, builder, question: str,
           disposition: str = '') -> tuple:
    """双版本出图 + 门禁：论文版（覆盖主文件、无图内总图题）+ 展示版（``_display`` 带图题）。"""
    out_dir = EDA_DIR if group == 'eda' else MODELING_DIR
    record = {'组别': group, '文件名': stem, 'Stage23正式图题': caption, '论文去向': disposition,
              '论文版路径': '', '展示版路径': '', '论文版门禁': {}, '展示版门禁': {},
              '论文版未通过': [], '展示版未通过': [], '已移除图内总图题': False,
              '重绘次数': 0, '最终是否通过': False, '尝试记录': []}
    paper = display = None
    for index, layout in enumerate(RETRY_LAYOUTS):
        fig, subfigures, meta = builder(layout)
        meta = {**meta, '组别': group, '回答的问题': question,
                'Stage23正式图题': caption, '论文去向': disposition}
        paper = figure_finalize.save_paper_figure(
            fig, out_dir, stem, subfigures=subfigures, meta=meta)
        plot_style.add_bottom_caption(fig, caption)
        plot_style.SCI_FIGURES_DIR = out_dir
        display = plot_style.save_sci_figure(
            fig, f'{stem}_display', caption, subfigures=subfigures, figure_id=stem,
            meta={**meta, 'variant': 'display'})
        plt.close(fig)
        failed_paper = figure_finalize.failed_paper_gates(paper)
        failed_display = [key for key in GATE_KEYS if not display.get(key)]
        if (failed_paper or failed_display) and index < len(RETRY_LAYOUTS) - 1:
            plot_style.FIGURE_REGISTRY.pop()
        paper['重绘轮次'] = display['重绘轮次'] = index + 1
        display['版式说明'] = paper['版式说明'] = layout['tag']
        record['尝试记录'].append({'轮次': index + 1, '版式': layout['tag'],
                                   '论文版未通过': failed_paper,
                                   '展示版未通过': failed_display})
        record['重绘次数'] = index
        record['论文版路径'] = paper['png_path']
        record['展示版路径'] = display['png_path']
        record['论文版门禁'] = figure_finalize.paper_gates(paper)
        record['展示版门禁'] = {key: bool(display.get(key)) for key in GATE_KEYS}
        record['论文版未通过'] = failed_paper
        record['展示版未通过'] = failed_display
        record['已移除图内总图题'] = bool(paper['no_infigure_caption'])
        record['最终是否通过'] = not (failed_paper or failed_display)
        if record['最终是否通过']:
            break
    status = '通过' if record['最终是否通过'] else '不通过'
    print(f"[{status}] {group}/{stem:<32} 重绘 {record['重绘次数']} 次 "
          f"（{record['尝试记录'][-1]['版式']}）"
          + ('' if record['最终是否通过']
             else f" 论文版未通过 {record['论文版未通过']} / 展示版未通过 {record['展示版未通过']}"))
    print(f"        论文版（无图内总图题）：{record['论文版路径']}"
          f"（{paper['png_size_bytes']:,} B）｜展示版：{record['展示版路径']}"
          f"（{figure_finalize.png_size(record['展示版路径']):,} B）")
    print(f"        Stage23 正式图题（展示版底部）：{caption}")
    return record, {'paper': paper, 'display': display}


def main() -> int:
    print('=' * 96)
    print('Stage23 收口重绘：eda 10 张 + modeling 7 张（论文版覆盖主文件 + _display 展示版）')
    print('=' * 96)
    print('禁止运行：scripts/13、scripts/14、scripts/15、notebooks/*；'
          '不重绘 modeling/07_shap_beeswarm（逐样本 SHAP 矩阵未落盘）')
    style_snapshot = plot_style.setup_sci_style()
    print(f"无头模式：backend={style_snapshot['backend']} headless={style_snapshot['headless']} "
          f"字体={style_snapshot['font_family']} 刻度={style_snapshot['xtick_direction']}")

    anchors = run_anchor_validation()
    if not print_anchor_validation(anchors):
        print('锚点校验未通过：按约束停止出图，不修改任何数值。')
        return 2

    print('=' * 96)
    print('说明：图 4-2 按任务规定取「样本数最多的前 12 个岗位细分类」；'
          '其余图的数值统计范围与原冻结图件一致（只改版式与图内图题）。')
    print('=' * 96)
    records, final_diagnostics = [], []
    for spec in figure_specs:
        record, diagnostics = render(*spec)
        records.append(record)
        final_diagnostics.append(diagnostics)

    print('=' * 96)
    print('逐图结果（论文版 = 覆盖主文件、无图内总图题；展示版 = _display、含 Stage23 正式图题）')
    print('=' * 96)
    print(f"{'序号':<4}{'文件':<40}{'论文版':<12}{'展示版':<12}{'图内总图题':<14}{'重绘次数'}")
    for index, record in enumerate(records, start=1):
        print(f"{index:<4}{record['组别'] + '/' + record['文件名']:<40}"
              f"{('已移除' if record['已移除图内总图题'] else '仍存在'):<12}"
              f"{('已写入' if record['展示版路径'] else '缺失'):<12}"
              f"{('无' if record['已移除图内总图题'] else '有'):<14}{record['重绘次数']}")

    open_figures = plt.get_fignums()
    print(f'出图后仍打开的画布数（应为 0）：{list(open_figures)}')
    print(f'plt.show / Figure.show 调用次数（应为 0）：'
          f"{plot_style.DISPLAY_TRACKER['show_calls']} / "
          f"{plot_style.DISPLAY_TRACKER['figure_show_calls']}")

    overall = plot_style.check_figure_gates(
        [item['display'] for item in final_diagnostics], style_snapshot)
    print('-' * 96)
    print('17 张展示版合并门禁（check_figure_gates）')
    for key, value in overall.items():
        if key != '_detail':
            print(f'  {key:<28}{value}')
    print(f"  _detail: {overall['_detail']}")

    registry_path = figure_finalize.write_registry(REGISTRY_PATH, {
        'style_snapshot': style_snapshot,
        'anchor_log': anchors,
        'gate_checks': {key: value for key, value in overall.items() if key != '_detail'},
        'gate_detail': overall['_detail'],
        'figures': records,
        'summary': {
            'total': len(records),
            'paper_pass': sum(not row['论文版未通过'] for row in records),
            'display_pass': sum(not row['展示版未通过'] for row in records),
            'final_pass': sum(row['最终是否通过'] for row in records),
            'in_figure_caption_removed': sum(row['已移除图内总图题'] for row in records),
        },
    })
    print('-' * 96)
    print(f'图件注册表已落盘：{project_paths.relative_to_root(registry_path)}（含双版本路径与逐项门禁）')
    print(f"图片目录：{project_paths.relative_to_root(EDA_DIR)} / "
          f"{project_paths.relative_to_root(MODELING_DIR)}")

    final_pass = sum(row['最终是否通过'] for row in records)
    failed_anchors = [row for row in anchors if not row['通过']]
    print('=' * 96)
    print(f"汇总：17 张图中双版本门禁全部通过 {final_pass} 张、仍不达标 "
          f"{len(records) - final_pass} 张；锚点不一致 {len(failed_anchors)} 项。")
    print('=' * 96)
    return 0 if (final_pass == len(records) and not failed_anchors) else 1


if __name__ == '__main__':
    sys.exit(main())

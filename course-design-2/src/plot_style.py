# -*- coding: utf-8 -*-
"""SCI 投稿风统一绘图模块。

全项目所有正式图片统一走本模块，禁止在各图重复写整套样式代码：

- 白底、低饱和、四边框、刻度朝内、网格淡化；
- 整图正式图名固定放在图片下方中央（禁止 ax.set_title / plt.title / fig.suptitle）；
- 子图编号统一 ``(a)(b)(c)(d)``，且「编号 + 子图名」放在对应子图下方；
- 每个正式图同时导出 PNG 600 dpi 与 PDF 矢量版；
- 保存时真实校验视觉规范并把诊断写入 ``FIGURE_REGISTRY``，供门禁汇总。

**无头保存模式（强制）**：

- 在导入 ``pyplot`` 之前先 ``matplotlib.use("Agg")``，不弹出任何图形窗口；
- 禁止 ``plt.show()`` / ``fig.show()`` / ``display(fig)``；``plt.show`` 与
  ``Figure.show`` 被替换为「只记录调用次数、不做任何显示」的守卫函数；
- 统一流程：绘制 → 保存 PNG / PDF → ``plt.close(fig)``；
- 不通过弹窗预览检查结果，全部直接保存到 ``outputs/figures/sci/``。

用法::

    from src import plot_style
    plot_style.setup_sci_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ...
    plot_style.apply_sci_axis(ax)
    plot_style.add_bottom_caption(fig, "图02 xxx")
    plot_style.save_sci_figure(fig, "fig_02_xxx", "图02 xxx")
    plt.close(fig)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

# 必须在导入 pyplot 之前设置无头后端（Agg），避免任何图形窗口弹出
matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib import _pylab_helpers  # noqa: E402
from matplotlib.collections import PathCollection  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from PIL import Image  # noqa: E402

from . import project_paths  # noqa: E402

# ---- 输出与登记 ----
SCI_FIGURES_DIR = project_paths.FIGURES_DIR / 'sci'
SCI_NOTEBOOK_PATH = project_paths.PROJECT_ROOT / 'notebooks' / '01_data_governance_sci_visualization.ipynb'
PNG_DPI = 600
FIGURE_REGISTRY: list = []

# ---- 无头模式守卫：记录任何被禁止的显示调用 ----
NON_GUI_BACKENDS = {'agg', 'cairo', 'pdf', 'pgf', 'svg', 'template', 'ps'}
FORBIDDEN_DISPLAY_CALLS = ('plt.show(', 'fig.show(', 'display(')
DISPLAY_TRACKER = {'show_calls': 0, 'figure_show_calls': 0}

# ---- 颜色与字号 ----
PALETTE = list(sns.color_palette('colorblind'))
MAIN_COLOR = PALETTE[0]
ACCENT_COLOR = PALETTE[3]
MUTED_COLOR = '#8c8c8c'
FONT_SIZES = {
    'axis_label': 9.6,
    'tick': 8.6,
    'legend': 8.6,
    'annotation': 8.0,
    'subfigure_caption': 9.2,
    'figure_caption': 10.5,
}

# ---- 规范取值（门禁核对用） ----
SPINE_WIDTH = 0.8
TICK_LENGTH = 4.0
TICK_WIDTH = 0.8
GRID_STYLE = {'linestyle': '--', 'linewidth': 0.45, 'alpha': 0.18}
CAPTION_Y = 0.018
SUBCAPTION_Y = -0.18


def enforce_headless_mode() -> dict:
    """强制无头保存模式：非 GUI 后端 + 关闭交互模式 + 拦截 show 类调用。

    被拦截的调用不会弹出任何窗口，但会被计数，供 ``NO_PLT_SHOW`` 门禁判定。
    """
    if matplotlib.get_backend().lower() not in NON_GUI_BACKENDS:
        matplotlib.use('Agg', force=True)
    plt.ioff()

    if not getattr(plt.show, '_sci_headless_guarded', False):
        def _guarded_show(*args, **kwargs):
            DISPLAY_TRACKER['show_calls'] += 1
            return None

        _guarded_show._sci_headless_guarded = True
        plt.show = _guarded_show

    if not getattr(Figure.show, '_sci_headless_guarded', False):
        def _guarded_figure_show(self, *args, **kwargs):
            DISPLAY_TRACKER['figure_show_calls'] += 1
            return None

        _guarded_figure_show._sci_headless_guarded = True
        Figure.show = _guarded_figure_show

    return {
        'backend': matplotlib.get_backend(),
        'interactive': bool(matplotlib.is_interactive()),
        'headless': matplotlib.get_backend().lower() in NON_GUI_BACKENDS,
        'show_guard_installed': bool(getattr(plt.show, '_sci_headless_guarded', False)
                                     and getattr(Figure.show, '_sci_headless_guarded', False)),
    }


def setup_sci_style() -> dict:
    """设置全局 SCI 投稿风样式并进入无头保存模式，返回关键样式快照（供门禁核对）。"""
    headless = enforce_headless_mode()
    sns.set_theme(context='paper', style='ticks', font_scale=1.0, palette=PALETTE)
    plt.rcParams.update({
        'font.family': ['Times New Roman', 'Microsoft YaHei'],
        'font.size': FONT_SIZES['tick'],
        'axes.unicode_minus': False,
        'axes.titlesize': FONT_SIZES['tick'],
        'axes.labelsize': FONT_SIZES['axis_label'],
        'xtick.labelsize': FONT_SIZES['tick'],
        'ytick.labelsize': FONT_SIZES['tick'],
        'legend.fontsize': FONT_SIZES['legend'],
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.grid': False,
        'axes.edgecolor': 'black',
        'axes.linewidth': SPINE_WIDTH,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'figure.dpi': 120,
        'savefig.dpi': PNG_DPI,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    SCI_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return {
        'context': 'paper',
        'style': 'ticks',
        'facecolor': plt.rcParams['figure.facecolor'],
        'font_family': list(plt.rcParams['font.family']),
        'xtick_direction': plt.rcParams['xtick.direction'],
        'ytick_direction': plt.rcParams['ytick.direction'],
        'grid_alpha': GRID_STYLE['alpha'],
        'savefig_dpi': plt.rcParams['savefig.dpi'],
        'xaxis_below': plt.rcParams.get('axes.axisbelow', True),
        **headless,
    }


def apply_sci_axis(ax, grid_axis: str = 'both', grid: bool = True) -> None:
    """统一坐标轴：四边框保留、刻度朝内、网格淡化。"""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_color('black')
    ax.tick_params(axis='both', which='major', direction='in', top=True, right=True,
                   length=TICK_LENGTH, width=TICK_WIDTH, labelsize=FONT_SIZES['tick'])
    ax.tick_params(axis='both', which='minor', direction='in', top=True, right=True,
                   length=2.2, width=0.6)
    if grid:
        ax.grid(True, axis=grid_axis, **GRID_STYLE)
    ax.set_axisbelow(True)


def add_bottom_caption(fig, text: str, y: float = CAPTION_Y,
                       fontsize: float | None = None):
    """整图正式图名：放在整幅图片下方中央（禁止使用 title / suptitle）。"""
    return fig.text(0.5, y, text, ha='center', va='bottom',
                    fontsize=fontsize or FONT_SIZES['figure_caption'])


def add_subfigure_caption(ax, letter: str, text: str, y: float = SUBCAPTION_Y,
                         fontsize: float | None = None):
    """子图编号 + 子图名：放在对应子图下方（禁止放在左上角）。"""
    return ax.text(0.5, y, f'({letter}) {text}', transform=ax.transAxes,
                   ha='center', va='top', fontsize=fontsize or FONT_SIZES['subfigure_caption'],
                   clip_on=False)


def format_integer_axis(ax, axis: str = 'both') -> None:
    """整数刻度：千位分隔，禁止科学计数法。"""
    formatter = FuncFormatter(lambda value, _pos: f'{value:,.0f}')
    if axis in ('both', 'x'):
        ax.xaxis.set_major_formatter(formatter)
    if axis in ('both', 'y'):
        ax.yaxis.set_major_formatter(formatter)


def format_percent_axis(ax, axis: str = 'both', decimals: int = 1) -> None:
    """比例刻度：按百分比显示。"""
    formatter = FuncFormatter(lambda value, _pos: f'{value * 100:.{decimals}f}%')
    if axis in ('both', 'x'):
        ax.xaxis.set_major_formatter(formatter)
    if axis in ('both', 'y'):
        ax.yaxis.set_major_formatter(formatter)


def barh_ranked(ax, series, color=None, value_fmt: str = '{:,.0f}',
                xlabel: str = '', ylabel: str = '', label_pad_ratio: float = 0.012):
    """降序横向柱状图（类别数 >= 6 时的默认图型），柱末标数值。"""
    data = series.sort_values(ascending=True)
    positions = np.arange(len(data))
    ax.barh(positions, data.to_numpy(), height=0.68,
            color=color or MAIN_COLOR, edgecolor='black', linewidth=0.5)
    ax.set_yticks(positions)
    ax.set_yticklabels([str(label) for label in data.index])
    span = float(data.max()) if len(data) else 1.0
    for position, value in zip(positions, data.to_numpy()):
        ax.text(value + span * label_pad_ratio, position, value_fmt.format(value),
                va='center', ha='left', fontsize=FONT_SIZES['annotation'])
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


def bar_ranked(ax, series, color=None, value_fmt: str = '{:,.0f}',
               xlabel: str = '', ylabel: str = '', label_pad_ratio: float = 0.012):
    """纵向柱状图（类别数 < 6 时的默认图型），柱顶标数值。"""
    positions = np.arange(len(series))
    ax.bar(positions, series.to_numpy(), width=0.62,
           color=color or MAIN_COLOR, edgecolor='black', linewidth=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(label) for label in series.index])
    span = float(series.max()) if len(series) else 1.0
    for position, value in zip(positions, series.to_numpy()):
        ax.text(position, value + span * label_pad_ratio, value_fmt.format(value),
                ha='center', va='bottom', fontsize=FONT_SIZES['annotation'])
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


def hist_discrete(ax, values, color=None, xlabel: str = '', ylabel: str = '岗位数量',
                  discrete: bool = True, bins=None):
    """直方图：离散整数变量使用 discrete=True，连续变量可指定 bins。"""
    series = np.asarray(values, dtype='float64')
    series = series[~np.isnan(series)]
    sns.histplot(x=series, discrete=discrete, bins=bins, ax=ax,
                 color=color or MAIN_COLOR, edgecolor='black', linewidth=0.5)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


def annotate_stats(ax, values, x: float = 0.97, y: float = 0.94, prefix: str = ''):
    """在子图内标注 P50 / P95 / Max（仅必要统计信息）。"""
    array = np.asarray(values, dtype='float64')
    array = array[~np.isnan(array)]
    text = (f'{prefix}P50 = {np.percentile(array, 50):.0f}\n'
            f'P95 = {np.percentile(array, 95):.0f}\n'
            f'Max = {array.max():.0f}')
    ax.text(x, y, text, transform=ax.transAxes, ha='right', va='top',
            fontsize=FONT_SIZES['annotation'], linespacing=1.45)
    return text


# ---------------------------------------------------------------- 保存与校验

def _band_has_ink(image: np.ndarray, y0: float, y1: float,
                  x0: float = 0.10, x1: float = 0.90) -> bool:
    """检查图片自下而上 [y0, y1] 比例区间内（水平 x0~x1）是否存在墨迹。"""
    height, width = image.shape[:2]
    top = int(height * (1 - y1))
    bottom = int(height * (1 - y0))
    left, right = int(width * x0), int(width * x1)
    strip = image[max(top, 0):max(bottom, 1), max(left, 0):max(right, 1), :3]
    if strip.size == 0:
        return False
    return bool((strip.min(axis=-1) < 0.85).any())


def _margins_blank(image: np.ndarray, y0: float, y1: float,
                   margin: float = 0.004) -> bool:
    """检查图名所在条带的左右边缘是否留白（用于判断是否被裁切）。"""
    height, width = image.shape[:2]
    top = int(height * (1 - y1))
    bottom = int(height * (1 - y0))
    edge = max(int(width * margin), 1)
    strip = image[max(top, 0):max(bottom, 1), :, :3]
    if strip.size == 0:
        return False
    left = strip[:, :edge].min(axis=-1)
    right = strip[:, -edge:].min(axis=-1)
    return bool((left >= 0.85).all() and (right >= 0.85).all())


def _text_band_in_image(fig, text, renderer, pad_inches: float = 0.1) -> tuple:
    """把某段文本的窗口范围换算为「导出图片」中的归一化条带。

    导出使用 ``bbox_inches='tight'``，因此必须用 tight bbox 做坐标换算，
    才能真实判断图名是否出现在最终 PNG 中且未被裁切。
    """
    bbox_px = text.get_window_extent(renderer)
    tight = fig.get_tightbbox(renderer).padded(pad_inches)
    dpi = fig.dpi
    x0 = (bbox_px.x0 / dpi - tight.x0) / tight.width
    x1 = (bbox_px.x1 / dpi - tight.x0) / tight.width
    y0 = (bbox_px.y0 / dpi - tight.y0) / tight.height
    y1 = (bbox_px.y1 / dpi - tight.y0) / tight.height
    return (float(np.clip(x0, 0, 1)), float(np.clip(x1, 0, 1)),
            float(np.clip(y0, 0, 1)), float(np.clip(y1, 0, 1)))


def _collection_overlap_ratio(ax, artist, legend_box) -> float:
    """散点等集合图元：按「点」逐个判断是否落入图例区域。"""
    offsets = artist.get_offsets()
    if offsets is None or len(offsets) == 0:
        return 0.0
    points = ax.transData.transform(np.asarray(offsets, dtype='float64'))
    sizes = artist.get_sizes()
    radius_pt = float(np.sqrt(max(sizes))) / 2.0 if len(sizes) else 3.0
    radius_px = radius_pt * ax.figure.dpi / 72.0
    x_overlap = np.clip(np.minimum(points[:, 0] + radius_px, legend_box.x1)
                        - np.maximum(points[:, 0] - radius_px, legend_box.x0), 0.0, None)
    y_overlap = np.clip(np.minimum(points[:, 1] + radius_px, legend_box.y1)
                        - np.maximum(points[:, 1] - radius_px, legend_box.y0), 0.0, None)
    covered = float((x_overlap * y_overlap).sum())
    legend_area = max(legend_box.width * legend_box.height, 1.0)
    return float(min(covered / legend_area, 1.0))


def _legend_overlap_ratio(fig) -> float:
    """图例与数据图元的窗口重叠比例上限（0 表示无重叠）。

    网格线与坐标轴背景不计入；散点按逐点覆盖面积统计。
    """
    fig.canvas.draw()
    worst = 0.0
    for ax in fig.axes:
        if ax.get_label() == '<colorbar>':
            continue
        legend = ax.get_legend()
        if legend is None or not legend.get_visible():
            continue
        legend_box = legend.get_window_extent()
        legend_area = max(legend_box.width * legend_box.height, 1.0)
        gridlines = set(ax.get_xgridlines()) | set(ax.get_ygridlines())
        for artist in ax.get_children():
            if not artist.get_visible() or artist in gridlines:
                continue
            if isinstance(artist, PathCollection):
                worst = max(worst, _collection_overlap_ratio(ax, artist, legend_box))
                continue
            if not isinstance(artist, (Line2D, Patch)):
                continue
            if isinstance(artist, Rectangle) and artist is ax.patch:
                continue
            try:
                box = artist.get_window_extent()
            except (RuntimeError, ValueError):
                continue
            if box.width <= 0 or box.height <= 0:
                continue
            x_overlap = max(0.0, min(legend_box.x1, box.x1) - max(legend_box.x0, box.x0))
            y_overlap = max(0.0, min(legend_box.y1, box.y1) - max(legend_box.y0, box.y0))
            worst = max(worst, x_overlap * y_overlap / legend_area)
    return round(float(worst), 4)


def save_sci_figure(fig, stem: str, caption: str, subfigures=None, figure_id: str = '',
                    meta: dict | None = None):
    """按规范保存 PNG 600 dpi + PDF，并做真实视觉校验后登记。

    ``subfigures`` 形如 ``[('a', '子图名', ax), ...]``，用于校验子图下置图名；
    ``meta`` 为图片元信息（图表类型、数据来源、推荐等级、建议章节等），写入登记表。
    """
    subfigures = list(subfigures or [])
    SCI_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = SCI_FIGURES_DIR / f'{stem}.png'
    pdf_path = SCI_FIGURES_DIR / f'{stem}.pdf'

    diagnostics: dict = {'figure_id': figure_id or stem.split('_')[0], 'stem': stem,
                         'caption': caption}
    if meta:
        diagnostics.update(meta)

    caption_text = next((text for text in fig.texts if text.get_text() == caption), None)
    diagnostics['caption_structure_ok'] = bool(
        caption_text is not None
        and abs(caption_text.get_position()[0] - 0.5) < 0.02
        and caption_text.get_position()[1] < 0.15
        and caption_text.get_ha() == 'center')
    diagnostics['no_top_title'] = (not any(ax.get_title() for ax in fig.axes)
                                   and getattr(fig, '_suptitle', None) is None)

    spines_ok = True
    inward_ok = True
    grid_ok = True
    labels_ok = True
    # 关闭坐标轴的示意图与颜色条轴不参与坐标轴规范检查
    active_axes = [ax for ax in fig.axes
                   if ax.axison and ax.get_label() != '<colorbar>']
    for ax in active_axes:
        spines_ok &= all(spine.get_visible() for spine in ax.spines.values())
        for axis in (ax.xaxis, ax.yaxis):
            for tick in axis.get_major_ticks():
                if getattr(tick, '_tickdir', None) != 'in':
                    inward_ok = False
        for line in ax.get_xgridlines() + ax.get_ygridlines():
            if line.get_visible():
                grid_ok &= (line.get_linestyle() == '--'
                            and line.get_linewidth() <= 0.45 + 1e-9
                            and line.get_alpha() is not None
                            and line.get_alpha() <= 0.18 + 1e-9)
        labels_ok &= bool(ax.get_xlabel()) and bool(ax.get_ylabel())
    diagnostics.update({
        'four_spines_visible': bool(spines_ok),
        'inward_ticks': bool(inward_ok),
        'light_grid': bool(grid_ok),
        'axis_labels_present': bool(labels_ok),
        'legend_overlap_ratio': _legend_overlap_ratio(fig),
    })
    diagnostics['legend_no_overlap'] = diagnostics['legend_overlap_ratio'] <= 0.05

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    caption_band = (_text_band_in_image(fig, caption_text, renderer)
                    if caption_text is not None else None)
    sub_bands = []
    for letter, text, ax in subfigures:
        child = next((item for item in ax.texts
                      if item.get_text() == f'({letter}) {text}'), None)
        sub_bands.append((letter, text, ax, child,
                          _text_band_in_image(fig, child, renderer) if child is not None else None))

    fig.savefig(png_path, dpi=PNG_DPI, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')

    with Image.open(png_path) as image:
        dpi = image.info.get('dpi', (0, 0))
        array = np.asarray(image.convert('RGB'), dtype='float64') / 255.0
    diagnostics['png_dpi'] = [float(dpi[0]), float(dpi[1])] if dpi else [0.0, 0.0]
    diagnostics['png_600dpi'] = bool(diagnostics['png_dpi'][0] >= 599.0)
    diagnostics['pdf_valid'] = png_path.exists() and pdf_path.exists() and \
        pdf_path.read_bytes()[:4] == b'%PDF'

    if caption_band is None:
        diagnostics['caption_ink_visible'] = False
        diagnostics['caption_not_cropped'] = False
    else:
        x0, x1, y0, y1 = caption_band
        diagnostics['caption_ink_visible'] = _band_has_ink(
            array, max(y0 - 0.004, 0.0), min(y1 + 0.004, 1.0), x0, x1)
        diagnostics['caption_not_cropped'] = _margins_blank(
            array, max(y0 - 0.004, 0.0), min(y1 + 0.004, 1.0))

    sub_ok = True
    sub_records = []
    for letter, text, ax, child, band in sub_bands:
        below_axes = bool(child is not None
                          and child.get_position()[1] < 0
                          and child.get_clip_on() is False)
        if band is None:
            visible = False
        else:
            x0, x1, y0, y1 = band
            visible = _band_has_ink(array, max(y0 - 0.004, 0.0), min(y1 + 0.004, 1.0), x0, x1)
        sub_ok &= bool(below_axes and visible)
        sub_records.append({'letter': letter, 'text': text,
                            'position_below': below_axes, 'ink_visible': bool(visible)})
    diagnostics['subfigures'] = sub_records
    diagnostics['subfigure_caption_visible'] = bool(sub_ok)
    diagnostics['subfigure_label_sequence_ok'] = [
        record['letter'] for record in sub_records
    ] == [chr(ord('a') + index) for index in range(len(sub_records))]

    diagnostics['png_path'] = str(png_path.relative_to(project_paths.PROJECT_ROOT))
    diagnostics['pdf_path'] = str(pdf_path.relative_to(project_paths.PROJECT_ROOT))
    FIGURE_REGISTRY.append(diagnostics)
    return diagnostics


def notebook_forbidden_calls(path: Path | None = None) -> list:
    """扫描正式 Notebook 的代码单元，返回被禁止的显示调用命中列表。"""
    path = Path(path or SCI_NOTEBOOK_PATH)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding='utf-8'))
    hits = []
    for index, cell in enumerate(payload.get('cells', [])):
        if cell.get('cell_type') != 'code':
            continue
        source = ''.join(cell.get('source', []))
        for pattern in FORBIDDEN_DISPLAY_CALLS:
            if pattern in source:
                hits.append(f'cell{index}:{pattern}')
    return hits


def check_figure_gates(registry=None, style_snapshot: dict | None = None) -> dict:
    """汇总视觉门禁：全部由真实文件与图形属性判定，禁止硬编码 PASS。"""
    registry = FIGURE_REGISTRY if registry is None else registry
    style_snapshot = style_snapshot or {}
    total = len(registry)
    if total == 0:
        raise ValueError('尚未生成任何正式图片，无法进行视觉门禁检查')

    subfigures = [record for record in registry if record['subfigures']]

    def _all(key: str) -> bool:
        return all(record.get(key) for record in registry)

    manager_windows = [getattr(manager, 'window', None)
                       for manager in _pylab_helpers.Gcf.get_all_fig_managers()]
    forbidden_in_notebook = notebook_forbidden_calls()
    open_figures = plt.get_fignums()
    saved_files_ok = all(
        (project_paths.PROJECT_ROOT / record['png_path']).exists()
        and (project_paths.PROJECT_ROOT / record['pdf_path']).exists()
        for record in registry)

    checks = {
        'SCI_STYLE': (style_snapshot.get('context') == 'paper'
                      and style_snapshot.get('style') == 'ticks'
                      and style_snapshot.get('facecolor') == 'white'
                      and style_snapshot.get('grid_alpha', 1) <= 0.2),
        'SNS_PLT_ONLY': not any(module in sys.modules for module in
                                ('plotly', 'bokeh', 'altair', 'pyecharts')),
        'INWARD_TICKS': _all('inward_ticks'),
        'FOUR_SPINES_VISIBLE': _all('four_spines_visible'),
        'LIGHT_GRID': _all('light_grid'),
        'CHINESE_FONT': bool(style_snapshot.get('font_family')
                             and 'Microsoft YaHei' in style_snapshot['font_family']),
        'BOTTOM_CAPTION_VISIBLE': _all('caption_structure_ok') and _all('caption_ink_visible')
                                  and _all('caption_not_cropped'),
        'SUBFIGURE_CAPTION_VISIBLE': all(record['subfigure_caption_visible']
                                         for record in subfigures) and len(subfigures) > 0,
        'SUBFIGURE_LABEL_SEQUENCE': all(record['subfigure_label_sequence_ok']
                                        for record in subfigures),
        'PNG_600DPI': _all('png_600dpi'),
        'PDF_EXPORT': _all('pdf_valid'),
        'LEGEND_NO_OVERLAP': _all('legend_no_overlap'),
        'AXIS_LABEL_VISIBLE': _all('axis_labels_present'),
        'NO_TITLE_ON_TOP': _all('no_top_title'),
        'NO_FIGURE_POPUP': bool(not any(window is not None for window in manager_windows)
                                and style_snapshot.get('headless')),
        'NO_PLT_SHOW': bool(DISPLAY_TRACKER['show_calls'] == 0
                            and DISPLAY_TRACKER['figure_show_calls'] == 0
                            and not forbidden_in_notebook
                            and not style_snapshot.get('interactive')),
        'HEADLESS_SAVE_MODE': bool(style_snapshot.get('headless')
                                   and style_snapshot.get('show_guard_installed')
                                   and not matplotlib.is_interactive()),
        'AUTO_CLOSE_FIGURE': bool(not open_figures and saved_files_ok),
    }
    checks['_detail'] = {
        'figure_total': total,
        'subfigure_figures': len(subfigures),
        'worst_legend_overlap': max(record['legend_overlap_ratio'] for record in registry),
        'backend': matplotlib.get_backend(),
        'interactive': bool(matplotlib.is_interactive()),
        'show_calls': DISPLAY_TRACKER['show_calls'],
        'figure_show_calls': DISPLAY_TRACKER['figure_show_calls'],
        'forbidden_in_notebook': forbidden_in_notebook,
        'open_figures': list(open_figures),
        'failed': [key for key, value in checks.items()
                   if key != '_detail' and not value],
    }
    return checks

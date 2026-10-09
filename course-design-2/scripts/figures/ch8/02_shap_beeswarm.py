# -*- coding: utf-8 -*-
"""重绘图 8-7「正式主模型 SHAP 蜂群图（Top12 特征）」：整图横置 ＋ 密度分层蜂群。

本轮只改呈现，不改任何数值（逐项断言平均绝对 SHAP 与表 8-5 同源）：

1. **整图横置**：12 个特征排在横轴（特征名倾斜 45°），SHAP 值排在纵轴；
   色条随之移到坐标区右侧，长度等于坐标区高度（约 4.4 cm），
   不再横跨整幅图高。
2. **密度分层蜂群**：同一 SHAP 邻域内的样本按局部密度左右对称堆叠，点的横向厚度
   即局部密度，替代原先的均匀随机抖动。
3. 点径 s = 14、不透明度 0.70；零值虚线置于散点之下，不再压在点上。
4. **名称括号只删冗余**：横轴名「特征（按平均绝对 SHAP 值排序）」删去与图题重复的
   「Top12」；纵轴「SHAP 值（元/天）」与色条「特征取值（各特征内归一化）」的括号
   属读图必需（单位、归一化统计范围），保留。
5. **设计宽度 12.5 cm**（即 Word 插入宽度），打印字号等效约 11 pt；
   并对倾斜标签做两式实测：相邻标签的**平行间距**（＝列距 × sin 倾角）必须大于
   标签的**垂直厚度**（＝行数 × 行距 × 字号），以及最深标签的下沿不得碰到横轴名。
   倾角为 45° 时标签必须单行，因此特征名不再折行。
6. **特征名简写**：只删冗余限定词与工程后缀——公司标签列表=免费健身设施 → 福利标签=免费健身设施
   （术语表禁止单用「公司标签」）、是否直辖市 → 直辖市、岗位细分类集合=人工智能 → 岗位细分类=人工智能；
   其余 9 个保留特征清单全名（技术技能数 / 学历等级 与 技能数量 / 学历要求 是两个不同字段，不可再简）。
   显示名与字段全名的对照写入图件登记，保持可追溯。

数值来源只读：复用 ``scripts/ch4_lifecycle/07_figure_rebuild.py`` 的
``build_final_shap()``（重拟合正式主模型 A+B+C+D+E，重算测试集 TreeSHAP）。

用法：:

    python scripts\\02_shap_beeswarm.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = next(
    _candidate for _candidate in Path(__file__).resolve().parents
    if (_candidate / "data").is_dir() and (_candidate / "scripts").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from src import figure_finalize, plot_style, project_paths  # noqa: E402

STEM = 'fig_8_9_shap_beeswarm'
SUPP_DIR = project_paths.FIGURES_DIR / 'supplementary'
METRICS_PATH = project_paths.METRICS_DIR / 'stage_60_shap_beeswarm_redraw.json'

PRINT_WIDTH_CM = 12.5                 # 设计宽度 = Word 插入宽度
FIG_HEIGHT_IN = 3.62
CM = 1.0 / 2.54
FONT_SCALE = 11.0 / 12.0
FONTS = {'axis_label': 11.5, 'tick': 10.5, 'legend': 10.0, 'annotation': 10.0}
ADJUST = {'left': 0.115, 'right': 0.880, 'bottom': 0.500, 'top': 0.975}
TOP_N = 12
DOT_SIZE = 14.0
DOT_ALPHA = 0.70
SWARM_HALF_WIDTH = 0.42               # 蜂群在「列间距 = 1」下的最大半宽
N_BINS = 100                          # 密度分层用的 SHAP 邻域分箱数
LAYER_SEED = 42
LABEL_ROTATION = 45.0                 # 横轴特征名倾斜角度（度）
LABEL_LINESPACING = 1.2               # matplotlib 默认行距系数

XLABEL = '特征（按平均绝对 SHAP 值排序）'
YLABEL = 'SHAP 值（元/天）'
CBAR_LABEL = '特征取值（各特征内归一化）'

# 显示名简写只删冗余限定词与工程后缀，对照一并写入图件登记
LABEL_ALIASES = {
    # 统一用「福利标签」，「列表」只是字段后缀（图 5-6、表 5-2 同统计范围）
    '公司标签列表=免费健身设施': '福利标签=免费健身设施',
    # 二值指示列，「是否」不含信息量，无同名混淆字段
    '是否直辖市': '直辖市',
    # 「集合」是承载字段的后缀，去掉不改变语义
    '岗位细分类集合=人工智能': '岗位细分类=人工智能',
}

# 冻结锚点（平均绝对 SHAP，元/天；与表 8-5 技能 SHAP 明细同源）
ANCHORS = {'公司标签列表=免费健身设施': 14.88, '技术技能数': 7.42, '文本SVD3': 5.96}


def load_figure_rebuild():
    """载入 07_figure_rebuild 模块并还原它在导入时打的内存补丁（只复用其 SHAP 计算）。"""
    spec = importlib.util.spec_from_file_location(
        '_s60_26g', str(PROJECT_ROOT / 'scripts' / 'ch4_lifecycle' / '07_figure_rebuild.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    plt.subplots = module._ORIG_SUBPLOTS
    Figure.subplots_adjust = module._ORIG_ADJUST
    plot_style.add_subfigure_caption = module._ORIG_SUBFIGCAP
    return module


def swarm_offsets(values: np.ndarray) -> np.ndarray:
    """密度分层蜂群：返回每个样本相对「特征列中心线」的左右偏移。

    与 shap 蜂群图同一做法：先按取值分箱，同一箱内的样本交替向两侧堆叠
    （0, +1, -1, +2, -2, …），再整体归一到 ``SWARM_HALF_WIDTH`` 以内，
    因此某处的横向厚度与该处的局部密度成正比。
    """
    values = np.asarray(values, dtype='float64')
    span = float(values.max() - values.min())
    if span <= 0:
        quant = np.zeros(len(values), dtype='int64')
    else:
        quant = np.round(N_BINS * (values - values.min()) / (span + 1e-8)).astype('int64')
    # 同一取值内的先后次序随机打散（固定种子），使堆叠层号不被原始行序决定
    rng = np.random.default_rng(LAYER_SEED)
    order = np.argsort(quant + rng.normal(0.0, 1e-6, len(values)), kind='stable')

    offsets = np.zeros(len(values), dtype='float64')
    layer = 0
    last_bin = -1
    for index in order:
        if quant[index] != last_bin:
            layer = 0
        offsets[index] = np.ceil(layer / 2.0) * ((layer % 2) * 2 - 1)
        layer += 1
        last_bin = quant[index]
    largest = float(offsets.max())
    if largest <= 0:
        return offsets
    return offsets * (0.9 * SWARM_HALF_WIDTH / (largest + 1.0))


def label_spacing_audit(fig, ax) -> dict:
    """倾斜标签的几何自检（单位一律换成 pt，故与最终缩放无关）。

    - 相邻标签是平行线，其**平行间距** ＝ 列距 × sin(倾角)；
    - 标签的**垂直厚度** ＝ 行数 × 行距系数 × 字号；
    - 只有平行间距大于垂直厚度，相邻倾斜标签才不会叠字；
    - 另测最深标签的下沿与横轴名上沿的净空。
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
    parallel_gap = pitch_pt * math.sin(angle)
    thickness = font_pt * LABEL_LINESPACING * lines
    lowest = min(tick.get_window_extent(renderer).y0 for tick in ticks) * to_pt
    xlabel_top = ax.xaxis.label.get_window_extent(renderer).y1 * to_pt
    return {'倾角': LABEL_ROTATION,
            '列距pt': round(pitch_pt, 2),
            '平行间距pt': round(parallel_gap, 2),
            '标签厚度pt': round(thickness, 2),
            '平行间距大于标签厚度': bool(parallel_gap > thickness),
            '标签行数': lines,
            '标签最深下沿至横轴名上沿净空pt': round(lowest - xlabel_top, 2),
            '标签与横轴名不重叠': bool(lowest >= xlabel_top)}


def build_figure(block: dict) -> tuple:
    names = block['特征名']
    values = block['测试集SHAP']
    dense = block['测试集矩阵']
    magnitude = np.abs(values).mean(axis=0)
    rank = np.argsort(magnitude)[::-1][:TOP_N]
    missing = [key for key in LABEL_ALIASES if key not in names]
    assert not missing, f'简写表里的字段名不在特征清单中：{missing}'

    fig, ax = plt.subplots(figsize=(PRINT_WIDTH_CM * CM * FONT_SCALE, FIG_HEIGHT_IN))
    scatter = None
    columns = []
    for column_position, feature_index in enumerate(rank):
        shap = values[:, feature_index]
        raw = dense[:, feature_index]
        span = float(raw.max() - raw.min())
        scaled = (raw - raw.min()) / span if span > 0 else np.zeros_like(raw)
        offsets = swarm_offsets(shap)
        scatter = ax.scatter(column_position + offsets, shap, s=DOT_SIZE, c=scaled,
                             cmap='cividis', vmin=0.0, vmax=1.0, alpha=DOT_ALPHA,
                             edgecolor='none', zorder=3)
        columns.append({'显示标签': LABEL_ALIASES.get(str(names[feature_index]),
                                                     str(names[feature_index])),
                        '字段全名': str(names[feature_index]),
                        '平均绝对SHAP': round(float(magnitude[feature_index]), 3),
                        '最大堆叠半宽（列间距=1）': round(float(np.abs(offsets).max()), 3),
                        '样本数': int(len(shap))})

    ax.axhline(0, color=plot_style.MUTED_COLOR, linestyle='--', linewidth=0.9, zorder=1)
    ax.set_xticks(np.arange(len(rank)))
    ax.set_xticklabels([item['显示标签'] for item in columns], rotation=LABEL_ROTATION,
                       rotation_mode='anchor', ha='right', va='top')
    ax.set_xlim(-0.6, len(rank) - 0.4)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    plot_style.apply_sci_axis(ax, grid_axis='y')
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.030, pad=0.016,
                            ticks=[0.0, 0.5, 1.0])
    colorbar.set_label(CBAR_LABEL, fontsize=plot_style.FONT_SIZES['axis_label'])
    colorbar.set_ticklabels(['低', '中', '高'])
    fig.subplots_adjust(**ADJUST)

    for name, expected in ANCHORS.items():
        actual = float(magnitude[names.index(name)])
        assert abs(actual - expected) < 0.02, f'{name} 平均绝对 SHAP 与表 8-5 不一致：{actual}'
    return fig, ax, columns


def main() -> int:
    started = time.time()
    snapshot = plot_style.setup_sci_style()
    plot_style.FONT_SIZES.update(FONTS)
    plt.rcParams.update({'font.size': FONTS['tick'],
                         'axes.labelsize': FONTS['axis_label'],
                         'axes.titlesize': FONTS['axis_label'],
                         'xtick.labelsize': FONTS['tick'],
                         'ytick.labelsize': FONTS['tick'],
                         'legend.fontsize': FONTS['legend'],
                         'axes.unicode_minus': False})

    g26 = load_figure_rebuild()
    print('=' * 92)
    print('重绘图 8-7 SHAP 蜂群图：整图横置（特征在横轴，色条随坐标区自然缩短）')
    print(f"无头模式：backend={snapshot['backend']} headless={snapshot['headless']}")
    print('=' * 92)
    block = g26.build_final_shap()
    print('模型复核：特征维度=%d，测试集 MAE=%.6f / RMSE=%.6f / R²=%.6f'
          % (block['特征维度'], block['测试集指标']['MAE'], block['测试集指标']['RMSE'],
             block['测试集指标']['R2']))

    fig, ax, columns = build_figure(block)
    geometry = label_spacing_audit(fig, ax)
    diagnostics = figure_finalize.save_paper_figure(
        fig, SUPP_DIR, STEM, subfigures=[],
        meta={'数据来源': 'Stage26.4 正式主模型（正式特征体系 A+B+C+D+E，'
                          '训练集 + 验证集重拟合后编码 %d 维）在测试集 %d 个岗位上的 TreeSHAP'
                          % (block['特征维度'], block['n_test']),
              '统计范围': '加性一致性最大误差随模型重算校验；只表示模型预测贡献，非因果',
              '测试集 MAE': round(float(block['测试集指标']['MAE']), 6),
              '基准值（元/天）': round(float(block['基准值']), 6),
              '版式': '整图横置（特征在横轴、SHAP 值在纵轴，色条随坐标区高度自然缩短）；'
                      '横轴特征名倾斜 %.0f°（单行，倾角下相邻标签为平行线，实测平行间距大于标签厚度）；'
                      '密度分层蜂群（同一 SHAP 邻域内按局部密度左右对称堆叠，厚度即局部密度）；'
                      '点径 s=%g、不透明度 %.2f，零值虚线置于散点之下；'
                      '名称括号只删冗余：横轴名删去与图题重复的 Top12，纵轴与色条的括号为读图必需而保留；'
                      '特征名按术语统计范围简写 3 处（公司标签列表→福利标签、是否直辖市→直辖市、'
                      '岗位细分类集合→岗位细分类），其余保留字段全名，对照见登记；'
                      '设计宽度 %.1f cm（= Word 插入宽度）'
                      % (LABEL_ROTATION, DOT_SIZE, DOT_ALPHA, PRINT_WIDTH_CM),
              '用途': '第8章 图 8-7（散点可读性优化 ＋ 整图横置，图题由载体文档给出）'})
    plt.close(fig)
    failed = figure_finalize.failed_paper_gates(diagnostics)

    pixel = diagnostics['png_pixel_size']
    png_width_cm = pixel[0] / 600.0 / CM
    png_height_cm = pixel[1] / 600.0 / CM
    printed_tick = FONTS['tick'] * PRINT_WIDTH_CM / png_width_cm
    printed_label = FONTS['axis_label'] * PRINT_WIDTH_CM / png_width_cm

    payload = {'stem': STEM,
               '生成时间': time.strftime('%Y-%m-%d %H:%M:%S'),
               '版式': {'方向': '横置（特征在横轴）', '设计宽度cm': PRINT_WIDTH_CM,
                        '横轴标签倾角': LABEL_ROTATION,
                        'PNG尺寸cm': [round(png_width_cm, 3), round(png_height_cm, 3)],
                        '打印字号等效pt': {'刻度': round(printed_tick, 2),
                                           '坐标轴名': round(printed_label, 2)},
                        '色条长度占坐标区': '与坐标区等高',
                        '名称': {'横轴': XLABEL, '纵轴': YLABEL, '色条': CBAR_LABEL}},
               '倾斜标签几何': geometry,
               '标签简写对照': {item['显示标签']: item['字段全名'] for item in columns
                                if item['显示标签'] != item['字段全名']},
               '散点画法': {'分层': '密度分层蜂群（shap 式交替堆叠）', '点径s': DOT_SIZE,
                            '不透明度': DOT_ALPHA, '分箱数': N_BINS,
                            '最大半宽（列间距=1）': SWARM_HALF_WIDTH, '堆叠种子': LAYER_SEED},
               'Top12 特征': columns,
               '模型复核': {'特征维度': block['特征维度'], 'n_test': block['n_test'],
                            '基准值（元/天）': round(float(block['基准值']), 6),
                            **{key: round(float(value), 6)
                               for key, value in block['测试集指标'].items()}},
               '论文版门禁未通过项': failed,
               'png_pixel_size': pixel,
               'png_size_bytes': diagnostics['png_size_bytes'],
               'pdf_size_bytes': diagnostics['pdf_size_bytes'],
               'png': diagnostics['png_path'], 'pdf': diagnostics['pdf_path']}
    METRICS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    print('-' * 92)
    for item in columns:
        print('  %-14s （字段全名 %-16s）平均绝对SHAP=%6.3f  最大半宽=%.3f  n=%d'
              % (item['显示标签'], item['字段全名'], item['平均绝对SHAP'],
                 item['最大堆叠半宽（列间距=1）'], item['样本数']))
    print('-' * 92)
    print(f"图件：{diagnostics['png_path']}")
    print(f"像素：{pixel}＝{png_width_cm:.2f}×{png_height_cm:.2f} cm（按 {PRINT_WIDTH_CM} cm "
          f"插入时等效字号：坐标轴名 {printed_label:.2f} pt / 刻度 {printed_tick:.2f} pt）")
    print('倾斜标签几何（设计尺寸下的 pt）：')
    for key, value in geometry.items():
        print(f'        {key} = {value}')
    print(f"论文版门禁未通过项：{failed or '无'}")
    print(f"元数据：{METRICS_PATH}  耗时 {time.time() - started:.1f} s")
    print('=' * 92)
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())

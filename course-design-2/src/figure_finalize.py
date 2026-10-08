# -*- coding: utf-8 -*-
"""Stage23 图表最终收口支持模块：冻结锚点校验 + 论文版（无图内总图题）导出。

只服务于 Stage23 的图件收口脚本（``scripts/18a`` / ``18b`` / ``18c``）：

- :func:`validate_anchors` / :func:`print_anchor_validation`：绘图前只读
  ``outputs/tables/*.xlsx`` 与 ``outputs/logs/metrics/*.json``，逐项断言 Stage23 冻结锚点，
  任一不符由调用方终止出图；本模块不写入、不修改任何冻结产物。
- :func:`save_paper_figure`：**论文版**导出——600 dpi PNG + 矢量 PDF，图片内部不含
  正式「图 X-X」总图题（总图题交给 Word Caption），但仍保留绘图区、坐标轴、图例与
  位于各子图下方的 ``(a)(b)(c)(d)`` 子图名；导出后按像素与图元真实校验门禁。
- :func:`strip_infigure_captions`：清除绘图函数内可能遗留的底部总图题（展示版随后按
  Stage23 正式图题重新绘制）。

视觉规范的唯一来源仍是 ``src/plot_style.py``：本模块只读取其样式常量与校验函数，
**不修改该模块**。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import plot_style, project_paths

# ---- 图片内部正式总图题识别模式（图 X-X / 附图 A-x / 图 Sxx） ----
CAPTION_PATTERN = re.compile(r'^\s*(附\s*)?图\s*(\d+\s*[-－–]\s*\d+|[AS]\s*\d+)')

# ---- 论文版门禁（必须全部为 True；展示版门禁由 plot_style 负责） ----
PAPER_GATE_KEYS = (
    'no_infigure_caption', 'no_top_title', 'subfigure_caption_visible',
    'subfigure_label_sequence_ok', 'inward_ticks', 'light_grid',
    'four_spines_visible', 'axis_labels_present', 'legend_no_overlap',
    'png_600dpi', 'pdf_valid',
)

# ---- Stage23 冻结锚点（值、来源与容差） ----
ANCHOR_ATOL = 1e-6
_TABLE_CACHE: dict = {}
_METRICS_CACHE: dict = {}


def _sheet(name: str, sheet: str) -> pd.DataFrame:
    key = (name, sheet)
    if key not in _TABLE_CACHE:
        _TABLE_CACHE[key] = pd.read_excel(project_paths.TABLES_DIR / name, sheet_name=sheet)
    return _TABLE_CACHE[key]


def _metrics(name: str) -> dict:
    if name not in _METRICS_CACHE:
        _METRICS_CACHE[name] = json.loads(
            (project_paths.METRICS_DIR / name).read_text(encoding='utf-8'))
    return _METRICS_CACHE[name]


def _record(records: list, label: str, expected, actual, source: str,
            atol: float = ANCHOR_ATOL) -> None:
    ok = bool(np.isclose(float(actual), float(expected), rtol=0.0, atol=max(atol, 1e-9)))
    records.append({'锚点': label, '期望值': expected, '实际读取值': actual,
                    '来源': source, '通过': ok})


def validate_anchors() -> list:
    """读取冻结产物并逐项校验 Stage23 锚点，返回逐项记录（只读，不修改任何文件）。"""
    table_13 = 'ch3/07_observation_snapshot_audit.xlsx'
    table_21 = 'ch3/15_structured_field_salary_audit.xlsx'
    table_29 = 'ch4/21_eda_statistical_analysis.xlsx'
    table_30 = 'ch7/22_model_comparison.xlsx'
    table_31 = 'ch8/23_ablation_robustness_shap.xlsx'
    table_32 = 'ch5/24_company_field_semantic_audit.xlsx'

    sample = _sheet(table_29, '01_样本概况').set_index('指标')['数值']
    salary = _sheet(table_29, '02_薪资描述统计').set_index('指标')['数值']
    stats = _sheet(table_29, '11_统计检验')
    observations = _sheet(table_13, '01_总体统计').set_index('指标')['数值']
    parse = _sheet(table_21, '02_薪资解析状态').set_index('薪资解析状态')['岗位数']
    wide = _sheet(table_32, '02_宽表字段统计')
    test_table = _sheet(table_30, '09_Test最终结果').set_index('模型')
    ablation = _sheet(table_31, '01_消融结果').set_index('配置')
    group_split = _sheet(table_31, '04_Random_vs_GroupSplit')
    eda_metrics = _metrics('stage_13_eda.json')

    label_block = stats[stats['检验块'].eq(
        '单标签二元比较（present vs absent；Mann–Whitney + BH-FDR）')]
    cert_kw = stats[stats['检验块'].eq('多组比较（Kruskal–Wallis）')
                    & stats['检验对象'].eq('公司认证')].iloc[0]
    overlap = eda_metrics['company_tag_analysis']['top_cliff_overlap']
    tag_atoms = wide[wide['数据表'].str.startswith('job_salary_model_dataset')]
    final_row = test_table.loc['FINAL（LightGBM）']
    group_row = group_split[group_split['划分方式'].str.startswith('Company Group Split')
                            & group_split['数据子集'].eq('test')]

    records: list = []
    _record(records, '原始岗位观测', 172063, int(observations['原始记录数']),
            f'{table_13} / 01_总体统计')
    _record(records, '唯一岗位实体', 17144, int(sample['全量岗位数（EDA 分析单元）']),
            f'{table_29} / 01_样本概况')
    _record(records, '明确薪资可解析', 14899, int(parse['已解析']),
            f'{table_21} / 02_薪资解析状态')
    _record(records, '薪资逻辑异常', 16, int(sample['薪资逻辑异常岗位数']),
            f'{table_29} / 01_样本概况')
    _record(records, '正式薪资样本', 14883, int(sample['正式薪资分析样本']),
            f'{table_29} / 01_样本概况')
    _record(records, '薪资面议岗位', 2245, int(sample['薪资面议岗位数']),
            f'{table_29} / 01_样本概况')
    _record(records, '薪资中点中位数', 175.0, float(salary['中位数']),
            f'{table_29} / 02_薪资描述统计')
    _record(records, '薪资中点均值', 192.87, float(salary['均值']),
            f'{table_29} / 02_薪资描述统计', atol=0.005)
    _record(records, '薪资中点 IQR', 90.0, float(salary['IQR']),
            f'{table_29} / 02_薪资描述统计')
    _record(records, '薪资中点 P10', 110.0, float(salary['P10']),
            f'{table_29} / 02_薪资描述统计')
    _record(records, '薪资中点 P90', 300.0, float(salary['P90']),
            f'{table_29} / 02_薪资描述统计')
    _record(records, '公司认证 ε²', 0.197545, float(cert_kw['效应量']),
            f'{table_29} / 11_统计检验（公司认证 KW）')
    _record(records, '福利标签原子标签数', 4328,
            int(tag_atoms['explode后唯一原子标签数'].dropna().iloc[0]),
            f'{table_32} / 02_宽表字段统计')
    _record(records, '福利标签进入二元比较', 183, int(len(label_block)),
            f'{table_29} / 11_统计检验（单标签二元比较）')
    _record(records, '福利标签 BH-FDR 显著', 114,
            int(label_block['FDR显著'].eq('是').sum()),
            f'{table_29} / 11_统计检验（单标签二元比较）')
    _record(records, '前5高效应标签交集岗位', 731, int(overlap['全部交集岗位数']),
            'outputs/logs/metrics/stage_13_eda.json: top_cliff_overlap')
    _record(records, '前5高效应标签并集岗位', 784, int(overlap['全部并集岗位数']),
            'outputs/logs/metrics/stage_13_eda.json: top_cliff_overlap')
    _record(records, '前5高效应标签最大两两 Jaccard', 1.0,
            float(overlap['两两最大Jaccard']),
            'outputs/logs/metrics/stage_13_eda.json: top_cliff_overlap')
    _record(records, '主模型 test MAE', 35.335506, float(final_row['test_MAE']),
            f'{table_30} / 09_Test最终结果')
    _record(records, '消融 Full test MAE', 36.074723, float(ablation.loc['Full', 'test_MAE']),
            f'{table_31} / 01_消融结果')
    _record(records, '按公司分组划分 test MAE', 52.049937,
            float(group_row['MAE'].iloc[0]), f'{table_31} / 04_Random_vs_GroupSplit')
    return records


def print_anchor_validation(records: list) -> bool:
    """打印锚点校验明细，返回是否全部通过。"""
    print('=' * 96)
    print('Stage23 冻结数值锚点校验（只读 outputs/tables 与 outputs/logs/metrics，禁止改动）')
    print('=' * 96)
    for item in records:
        flag = '  OK  ' if item['通过'] else ' FAIL '
        print(f"[{flag}] {item['锚点']:<26} 期望 {item['期望值']:<12} "
              f"实际 {item['实际读取值']:<16} 来源 {item['来源']}")
    passed = all(item['通过'] for item in records)
    print(f"锚点校验结论：{'全部通过' if passed else '存在不符项'}，"
          f"通过 {sum(item['通过'] for item in records)}/{len(records)}")
    return passed


# ---------------------------------------------------------------- 论文版导出


def strip_infigure_captions(fig) -> list:
    """移除图形级文本中遗留的正式总图题（图 X-X / 附图 A-x / 图 Sxx）。"""
    removed = []
    for text in list(fig.texts):
        content = text.get_text()
        if CAPTION_PATTERN.match(content):
            removed.append(content)
            text.remove()
    return removed


def save_paper_figure(fig, out_dir, stem: str, subfigures=None, meta: dict | None = None) -> dict:
    """论文版导出：无图内总图题，600 dpi PNG + 矢量 PDF，并真实校验视觉门禁。"""
    subfigures = list(subfigures or [])
    out_dir = project_paths.figure_chapter_dir(stem, Path(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f'{stem}.png'
    pdf_path = out_dir / f'{stem}.pdf'

    diagnostics: dict = {'stem': stem, 'variant': 'paper'}
    if meta:
        diagnostics.update(meta)

    diagnostics['removed_infigure_captions'] = strip_infigure_captions(fig)

    figure_texts = [text.get_text() for text in fig.texts]
    axis_texts = [text.get_text() for ax in fig.axes for text in ax.texts]
    offending = [text for text in [*figure_texts, *axis_texts]
                 if CAPTION_PATTERN.match(text)]
    diagnostics['figure_level_texts'] = figure_texts
    diagnostics['offending_infigure_captions'] = offending
    diagnostics['no_infigure_caption'] = bool(not figure_texts and not offending)
    diagnostics['no_top_title'] = bool(not any(ax.get_title() for ax in fig.axes)
                                       and getattr(fig, '_suptitle', None) is None)

    spines_ok = True
    inward_ok = True
    grid_ok = True
    labels_ok = True
    active_axes = [ax for ax in fig.axes if ax.axison and ax.get_label() != '<colorbar>']
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
    diagnostics['four_spines_visible'] = bool(spines_ok)
    diagnostics['inward_ticks'] = bool(inward_ok)
    diagnostics['light_grid'] = bool(grid_ok)
    diagnostics['axis_labels_present'] = bool(labels_ok)

    # 复用 plot_style 的图例重叠检测（只读调用，不修改该模块）
    diagnostics['legend_overlap_ratio'] = plot_style._legend_overlap_ratio(fig)
    diagnostics['legend_no_overlap'] = diagnostics['legend_overlap_ratio'] <= 0.05

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    sub_bands = []
    for letter, text, ax in subfigures:
        child = next((item for item in ax.texts
                      if item.get_text() == f'({letter}) {text}'), None)
        band = plot_style._text_band_in_image(fig, child, renderer) if child is not None else None
        sub_bands.append((letter, text, ax, child, band))

    fig.savefig(png_path, dpi=plot_style.PNG_DPI, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')

    with Image.open(png_path) as image:
        size = image.size
        dpi = image.info.get('dpi', (0, 0))
        array = np.asarray(image.convert('RGB'), dtype='float64') / 255.0
    diagnostics['png_pixel_size'] = [int(size[0]), int(size[1])]
    diagnostics['png_dpi'] = [float(dpi[0]), float(dpi[1])] if dpi else [0.0, 0.0]
    diagnostics['png_600dpi'] = bool(diagnostics['png_dpi'][0] >= 599.0)
    diagnostics['png_size_bytes'] = int(png_path.stat().st_size)
    diagnostics['pdf_size_bytes'] = int(pdf_path.stat().st_size)
    diagnostics['pdf_valid'] = bool(pdf_path.read_bytes()[:4] == b'%PDF')

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
            visible = plot_style._band_has_ink(
                array, max(y0 - 0.004, 0.0), min(y1 + 0.004, 1.0), x0, x1)
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
    return diagnostics


def paper_gates(diagnostics: dict) -> dict:
    """抽取论文版门禁布尔值。"""
    return {key: bool(diagnostics.get(key)) for key in PAPER_GATE_KEYS}


def failed_paper_gates(diagnostics: dict) -> list:
    return [key for key, value in paper_gates(diagnostics).items() if not value]


def png_size(relative_path) -> int:
    """按相对项目根的路径读取 PNG 字节数（展示版由 plot_style 保存，无字节数字段）。"""
    if not relative_path:
        return 0
    return int((project_paths.PROJECT_ROOT / str(relative_path)).stat().st_size)


def write_registry(path, payload: dict) -> Path:
    """写出 Stage23 图件登记表（仅图件日志，不含任何统计结果）。"""
    path = Path(path)
    payload = {'stage': 'Stage23', **payload}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                    encoding='utf-8')
    return path

# -*- coding: utf-8 -*-
"""图 S01~S16 补图脚本（Stage23 收口版）：只新增绘图，不改动任何统计结果、模型与既有图件。

双输出路径（Stage23 硬要求）
---------------------------
- **论文版**：覆盖同名主文件（``图S01_样本筛选与口径流转.png`` 等），图片内部**不含**
  「图 X-X / 附图 A-x」正式总图题（总图题交给 Word Caption），保留绘图区 / 坐标轴 / 图例与
  位于各子图下方的 ``(a)(b)(c)(d)`` 子图名；600 dpi PNG + 矢量 PDF；
- **独立展示版**：``<stem>_display.png|pdf``，底部带 Stage23 正式图题（PPT / 单独查看用）。

新增合并图
----------
``图S05S06_公司福利标签薪资关联与共现``：把原 S05（效应量与 BH-FDR 显著性）与
S06（高效应标签共现簇）的**同一份冻结数据**合并为双子图
``(a) 福利标签效应量与 BH-FDR 显著性`` / ``(b) 高效应福利标签共现结构``，
不重新计算任何统计量（只复用本轮 S05 / S06 的绘图与数据读取逻辑）。

约束
----
- 只读：``outputs/tables/*.xlsx``、``outputs/logs/metrics/*.json``、
  ``data/processed|interim|features/*.parquet``；
- 只写：``outputs/figures/supplementary/``（PNG 600dpi + PDF）
  与 ``outputs/figures/_supplementary_registry.json``；
- 绘图风格唯一来源 ``src/plot_style.py``（本脚本不修改该模块）；
- 所有标注数值均从冻结产物读取并经锚点校验（Stage23 锚点统一由
  ``src/figure_finalize.validate_anchors`` 在出图前断言），禁止硬编码结论。

运行::

    python scripts\\18b_supplementary_figures.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import figure_finalize  # noqa: E402
from src import plot_style  # noqa: E402
from src import project_paths  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

TABLES_DIR = project_paths.TABLES_DIR
METRICS_DIR = project_paths.METRICS_DIR
PROCESSED_DIR = project_paths.PROCESSED_DIR
SUPP_DIR = project_paths.FIGURES_DIR / 'supplementary'
REGISTRY_PATH = project_paths.FIGURES_DIR / '_supplementary_registry.json'

PALETTE = plot_style.PALETTE
FS = plot_style.FONT_SIZES
BLUE, ORANGE, GREEN, RED = PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]
PURPLE, BROWN, PINK, GRAY, OLIVE = PALETTE[4], PALETTE[5], PALETTE[6], PALETTE[7], PALETTE[8]
MUTED = plot_style.MUTED_COLOR

CLIFF_THRESHOLDS = (0.147, 0.33, 0.474)  # Romano et al. 2006，与统计阶段口径一致
SPLIT_LABELS = {'train': '训练集', 'validation': '验证集', 'test': '测试集'}
SPLIT_ORDER = ['train', 'validation', 'test']
CERT_ORDER = ['无认证', '行业认证', '最佳雇主', '两者均有']
TOP5_TAGS = ['免费健身设施', '就近租房补贴', '餐饮', '节日礼品', '弹性工作制']

ANCHOR_LOG: list = []


# ------------------------------------------------------------------ 基础工具
def read_sheet(filename: str, sheet: str) -> pd.DataFrame:
    return pd.read_excel(TABLES_DIR / filename, sheet_name=sheet)


def read_metrics(filename: str) -> dict:
    return json.loads((METRICS_DIR / filename).read_text(encoding='utf-8'))


def record_anchor(key: str, actual, expected, source: str) -> None:
    if isinstance(actual, (np.integer, np.floating)):
        actual = actual.item()
    if isinstance(expected, float):
        ok = abs(float(actual) - expected) < 1e-9
    else:
        ok = int(actual) == int(expected)
    ANCHOR_LOG.append({'锚点': key, '读取值': actual, '校验值': expected,
                       '一致': bool(ok), '来源': source})


def as_tag_set(value) -> set:
    if isinstance(value, (list, tuple, np.ndarray, set)):
        return {str(item) for item in value if item is not None and not pd.isna(item)}
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    return {value.strip()} if str(value).strip() else set()


def cert_class(value) -> str:
    atoms = as_tag_set(value) & {'最佳雇主', '行业认证'}
    if len(atoms) == 2:
        return '两者均有'
    if atoms == {'最佳雇主'}:
        return '最佳雇主'
    if atoms == {'行业认证'}:
        return '行业认证'
    return '无认证'


def legend_above(ax, ncol: int = 3, handles=None, labels=None, fix: int = 0):
    kwargs = {'loc': 'lower center', 'bbox_to_anchor': (0.5, 1.02 + 0.03 * fix),
              'ncol': ncol, 'frameon': False, 'fontsize': FS['legend'],
              'handlelength': 1.6, 'columnspacing': 1.3, 'borderaxespad': 0.0}
    if handles is not None:
        kwargs['handles'] = handles
        kwargs['labels'] = labels
    return ax.legend(**kwargs)


def note(ax, text: str, x: float = 0.975, y: float = 0.965, ha: str = 'right',
         va: str = 'top', box: bool = False):
    bbox = {'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.88} if box else None
    return ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
                   fontsize=FS['annotation'], linespacing=1.5, bbox=bbox)


def panel_fig(fix: int, panels: int, figsize, wspace: float = 0.34):
    if panels == 1:
        fig, ax = plt.subplots(figsize=(figsize[0], figsize[1] + 0.22 * fix))
        fig.subplots_adjust(bottom=0.27 + 0.035 * fix, top=0.87 - 0.01 * fix,
                            left=0.14, right=0.97)
        return fig, ax
    fig, axes = plt.subplots(1, panels, figsize=(figsize[0], figsize[1] + 0.22 * fix))
    fig.subplots_adjust(bottom=0.27 + 0.035 * fix, top=0.855 - 0.01 * fix,
                        left=0.085, right=0.975, wspace=wspace + 0.07 * fix)
    return fig, axes


def box_groups(ax, groups, labels, colors, ylabel, xlabel):
    artists = ax.boxplot(groups, widths=0.5, showfliers=False, patch_artist=True,
                         medianprops={'color': 'black', 'linewidth': 1.1},
                         whiskerprops={'linewidth': 0.8},
                         capprops={'linewidth': 0.8},
                         boxprops={'linewidth': 0.8})
    for patch, color in zip(artists['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    return artists


# ------------------------------------------------------------------ 图 S05 / S06 公共面板
def tag_effect_panel(ax, fix: int = 0, legend_ncol: int = 4):
    """S05 面板：公司福利标签 Cliff's δ 与 BH-FDR 校正后 q 的效应量—显著性总览。"""
    block = V['label_block']
    q = block['p_adjusted_bh'].astype(float).to_numpy()
    floor_q = 1e-12
    y = -np.log10(np.clip(q, floor_q, 1.0))
    x = block['cliff_delta'].astype(float).to_numpy()
    labels = block['effect_label'].to_numpy()
    capped = int((q < floor_q).sum())

    classes = ['negligible（可忽略）', 'small（小）', 'medium（中）', 'large（大）']
    class_colors = {'negligible（可忽略）': GRAY, 'small（小）': BLUE,
                    'medium（中）': ORANGE, 'large（大）': RED}

    ax.axhline(-np.log10(0.05), color='black', linestyle='--', linewidth=0.8)
    for threshold in (CLIFF_THRESHOLDS[0], -CLIFF_THRESHOLDS[0]):
        ax.axvline(threshold, color=MUTED, linestyle=':', linewidth=0.8)
    for name in classes:
        mask = labels == name
        ax.scatter(x[mask], y[mask], s=26, color=class_colors[name], alpha=0.85,
                   edgecolor='black', linewidth=0.35, label=name, zorder=3)
    ax.set_xlabel("Cliff's δ（有标签 − 无标签）")
    ax.set_ylabel('−log₁₀（BH-FDR 校正后 q 值）')
    ax.set_ylim(-0.8, -np.log10(floor_q) * 1.16)
    span = max(abs(x.min()), abs(x.max()))
    ax.set_xlim(-span * 1.16, span * 1.16)
    plot_style.apply_sci_axis(ax, grid_axis='y', grid=True)
    legend_above(ax, ncol=legend_ncol, fix=fix)
    note(ax, f'进入正式二元比较 {V["tag_tested"]}\n'
             f'校正后显著（q < 0.05）{V["tag_significant"]}\n'
             f'q ≤ 1e-12 的 {capped} 个标签绘制于上限\n'
             f'虚线：q = 0.05；点线：|δ| = {CLIFF_THRESHOLDS[0]:.3f}',
         x=0.02, y=0.97, ha='left', box=True)
    return ax


def tag_overlap_items() -> list:
    """S06 面板：5 个高效应标签命中岗位数 + 全部交集 / 全部并集（读数全部来自冻结口径）。"""
    counts = V['top5_counts']
    items = [(f'{tag}', int(counts[tag]), BLUE) for tag in TOP5_TAGS]
    items += [('5 个标签交集（共同命中）', V['top5_intersection'], RED),
              ('5 个标签并集（任一命中）', V['top5_union'], GREEN)]
    return items


def tag_overlap_bars(ax, show_universe_lines: bool = False):
    """S06 面板：标签与集合口径的岗位规模横向柱状图（可选交集 / 并集参考线）。"""
    items = tag_overlap_items()
    union = V['top5_union']
    positions = np.arange(len(items))[::-1]
    for position, (_, value, color) in zip(positions, items):
        ax.barh(position, value, height=0.62, color=color, edgecolor='black', linewidth=0.55)
        ax.text(value + union * 0.012, position, f'{value:,}', va='center', ha='left',
                fontsize=FS['annotation'])
    ax.set_yticks(positions)
    ax.set_yticklabels([item[0] for item in items])
    ax.set_xlim(0, union * 1.28)
    ax.set_ylim(-0.7, len(items) - 0.3)
    ax.set_xlabel('岗位数量（个）')
    ax.set_ylabel('标签与集合')
    if show_universe_lines:
        ax.axvline(V['top5_intersection'], color=RED, linestyle='--', linewidth=0.9)
        ax.axvline(union, color=GREEN, linestyle=':', linewidth=0.9)
    plot_style.apply_sci_axis(ax, grid_axis='x', grid=True)
    plot_style.format_integer_axis(ax, axis='x')
    return ax


def tag_jaccard_heatmap(ax, fix: int = 0):
    """S06 面板：5 个高效应标签两两 Jaccard 共现热力图。"""
    matrix = np.array(V['top5_jaccard'])
    image = ax.imshow(matrix, cmap='Blues', vmin=0.90, vmax=1.0, aspect='auto')
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(column, row, f'{value:.3f}', ha='center', va='center',
                    fontsize=FS['annotation'],
                    color='white' if value > 0.985 else 'black')
    ax.set_xticks(np.arange(len(TOP5_TAGS)))
    ax.set_xticklabels(TOP5_TAGS, rotation=30 + 8 * fix, ha='right')
    ax.set_yticks(np.arange(len(TOP5_TAGS)))
    ax.set_yticklabels(TOP5_TAGS)
    ax.set_xlabel('公司福利标签')
    ax.set_ylabel('公司福利标签')
    ax.set_xticks(np.arange(-0.5, len(TOP5_TAGS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(TOP5_TAGS), 1), minor=True)
    plot_style.apply_sci_axis(ax, grid=False)
    note(ax, f'最大两两 Jaccard = {V["top5_max_jaccard"]:.1f}',
         x=0.98, y=0.06, ha='right', va='bottom')
    return ax, image


# ------------------------------------------------------------------ 数据装载
def load_data() -> dict:
    data: dict = {}
    data['sample'] = read_sheet('29_eda_statistical_analysis.xlsx', '01_样本概况')
    data['salary_desc'] = read_sheet('29_eda_statistical_analysis.xlsx', '02_薪资描述统计')
    data['company_factor'] = read_sheet('29_eda_statistical_analysis.xlsx', '06_公司因素薪资')
    data['skill_salary'] = read_sheet('29_eda_statistical_analysis.xlsx', '08_技能薪资')
    data['skill_in_cat'] = read_sheet('29_eda_statistical_analysis.xlsx', '09_岗位内技能薪资')
    data['stats'] = read_sheet('29_eda_statistical_analysis.xlsx', '11_统计检验')
    data['skill_count'] = read_sheet('29_eda_statistical_analysis.xlsx', '13_技能数量薪资')
    data['model_sample'] = read_sheet('28_modeling_dataset_audit.xlsx', '08_模型样本筛选')
    data['model_overview'] = read_sheet('28_modeling_dataset_audit.xlsx', '01_总体结构')
    data['leak_blacklist'] = read_sheet('28_modeling_dataset_audit.xlsx', '04_目标泄漏黑名单')
    data['manifest'] = read_sheet('28_modeling_dataset_audit.xlsx', '03_Feature_Manifest')
    data['split'] = read_sheet('30_model_comparison.xlsx', '01_数据划分')
    data['feature_dim'] = read_sheet('30_model_comparison.xlsx', '02_特征维度')
    data['test_result'] = read_sheet('30_model_comparison.xlsx', '09_Test最终结果')
    data['skill_threshold'] = read_sheet('30_model_comparison.xlsx', '12_技能阈值选择')
    data['text_dim'] = read_sheet('30_model_comparison.xlsx', '13_文本维度选择')
    data['skill_inc'] = read_sheet('31_ablation_robustness_shap.xlsx', '02_技能增量')
    data['text_inc'] = read_sheet('31_ablation_robustness_shap.xlsx', '03_文本增量')
    data['split_compare'] = read_sheet('31_ablation_robustness_shap.xlsx', '04_Random_vs_GroupSplit')
    data['robust_extreme'] = read_sheet('31_ablation_robustness_shap.xlsx', '05_极端值敏感性')
    data['robust_target'] = read_sheet('31_ablation_robustness_shap.xlsx', '06_目标稳健性')
    data['shap_rank'] = read_sheet('31_ablation_robustness_shap.xlsx', '07_SHAP总排名')
    data['shap_stability'] = read_sheet('31_ablation_robustness_shap.xlsx', '11_SHAP稳定性')
    data['ablation'] = read_sheet('31_ablation_robustness_shap.xlsx', '01_消融结果')
    data['wide_stats'] = read_sheet('32_company_field_semantic_audit.xlsx', '02_宽表字段统计')
    data['issue_class'] = read_sheet('32_company_field_semantic_audit.xlsx', '07_问题分类')
    data['tag_per_job'] = read_sheet('32_company_field_semantic_audit.xlsx', '04c_每岗标签数分布')
    data['anomaly_fix'] = read_sheet('22_company_attribute_semantic_anomaly_audit.xlsx', '07_修复前后对照')
    data['anomaly_residual'] = read_sheet('22_company_attribute_semantic_anomaly_audit.xlsx', '08_残留异常')
    data['obs_audit'] = read_sheet('13_observation_snapshot_audit.xlsx', '01_总体统计')
    data['salary_parse'] = read_sheet('21_structured_field_salary_audit.xlsx', '02_薪资解析状态')
    data['skill_scope'] = read_sheet('27_skill_eda_scope_audit.xlsx', '01_样本口径')
    data['merge_audit'] = read_sheet('09_unique_job_merge_audit.xlsx', '合并审计汇总')
    data['eda_metrics'] = read_metrics('stage_13_eda.json')
    return data


def build_job_frame() -> pd.DataFrame:
    """岗位级只读宽表：薪资中点 + 数据子集 + 技能数量 + 公司标签 / 公司认证四类。"""
    model = pd.read_parquet(project_paths.JOB_SALARY_MODEL_DATASET_PARQUET,
                            columns=['实习岗位ID', '公司标签列表'])
    targets = pd.read_parquet(project_paths.SALARY_TARGETS_PARQUET,
                              columns=['实习岗位ID', '薪资中点'])
    splits = pd.read_parquet(project_paths.MODEL_SPLITS_PARQUET)
    skills = pd.read_parquet(project_paths.JOB_TEXT_FEATURES_PARQUET,
                             columns=['实习岗位ID', '技能数量'])
    certs = pd.read_parquet(project_paths.PROCESSED_UNIQUE_PARQUET,
                            columns=['实习岗位ID', '公司认证标签'])
    frame = (model.merge(targets, on='实习岗位ID', how='left')
                  .merge(splits[['实习岗位ID', 'split']], on='实习岗位ID', how='left')
                  .merge(skills, on='实习岗位ID', how='left')
                  .merge(certs, on='实习岗位ID', how='left'))
    frame['公司认证四类'] = frame['公司认证标签'].map(cert_class)
    return frame


def build_anchor_values(data: dict, jobs: pd.DataFrame) -> dict:
    sample = data['sample'].set_index('指标')['数值']
    salary = data['salary_desc'].set_index('指标')['数值']
    obs_row = data['obs_audit'].set_index('指标')['数值']
    parse = data['salary_parse'].set_index('薪资解析状态')['岗位数']
    scope = data['skill_scope'].set_index('口径')['岗位数']
    merge = data['merge_audit'].set_index('指标')['数值']
    kw = data['stats'][data['stats']['检验块'] == '多组比较（Kruskal–Wallis）']
    label_block = data['stats'][data['stats']['检验块'] == '单标签二元比较（present vs absent；Mann–Whitney + BH-FDR）']
    cert_kw = kw[kw['检验对象'] == '公司认证'].iloc[0]
    wide = data['wide_stats']
    cert_counts = jobs['公司认证四类'].value_counts()
    overlap = data['eda_metrics']['company_tag_analysis']['top_cliff_overlap']

    values = {
        'n_raw_obs': int(obs_row['原始记录数']),
        'n_unique_jobs': int(sample['全量岗位数（EDA 分析单元）']),
        'n_negotiable': int(sample['薪资面议岗位数']),
        'n_anomaly': int(sample['薪资逻辑异常岗位数']),
        'n_final': int(sample['正式薪资分析样本']),
        'n_parsed': int(parse['已解析']),
        'n_merged_dup': int(merge['合并重复记录数']),
        'median': float(salary['中位数']),
        'iqr': float(salary['IQR']),
        'p10': float(salary['P10']),
        'p90': float(salary['P90']),
        'cert_counts': {k: int(cert_counts[k]) for k in CERT_ORDER},
        'cert_epsilon2': float(cert_kw['效应量']),
        'cert_groups': int(cert_kw['参与检验组数']),
        'tag_atoms': int(wide[wide['数据表'].str.startswith('job_salary_model_dataset')]
                         ['explode后唯一原子标签数'].dropna().iloc[0]),
        'tag_atoms_stage13': int(data['eda_metrics']['company_tag_atoms']),
        'tag_tested': int(len(label_block)),
        'tag_significant': int((label_block['FDR显著'] == '是').sum()),
        'top5_tags': list(overlap['标签']),
        'top5_intersection': int(overlap['全部交集岗位数']),
        'top5_union': int(overlap['全部并集岗位数']),
        'top5_max_jaccard': float(overlap['两两最大Jaccard']),
        'skill_main': int(scope['REQUIREMENT_SECTION']),
        'skill_extended': int(scope['ALL_USABLE']),
        'skill_empty': int(scope['EMPTY_TEXT']),
        'skill_dims': int(data['feature_dim'][data['feature_dim']['特征块'] == '技能 multi-hot（D 组）']['列数'].iloc[0]),
        'feature_total': int(data['feature_dim'][data['feature_dim']['特征块'] == '合计']['列数'].iloc[0]),
        'text_dims': int(data['feature_dim'][data['feature_dim']['特征块'] == '文本语义 SVD（E 组）']['列数'].iloc[0]),
        'base_dims': int(data['ablation'].set_index('配置').loc['Base', '特征维度']),
        'split_counts': {k: int(data['split'][data['split']['split'] == k]['岗位数'].iloc[0])
                         for k in SPLIT_ORDER},
    }

    record_anchor('原始岗位观测', values['n_raw_obs'], 172063, '13_observation_snapshot_audit/01_总体统计')
    record_anchor('唯一岗位实体', values['n_unique_jobs'], 17144, '29_eda_statistical_analysis/01_样本概况')
    record_anchor('薪资面议岗位', values['n_negotiable'], 2245, '29_eda_statistical_analysis/01_样本概况')
    record_anchor('薪资逻辑异常', values['n_anomaly'], 16, '29_eda_statistical_analysis/01_样本概况')
    record_anchor('正式薪资样本', values['n_final'], 14883, '29_eda_statistical_analysis/01_样本概况')
    record_anchor('明确薪资可解析', values['n_parsed'], 14899, '21_structured_field_salary_audit/02_薪资解析状态')
    record_anchor('薪资中位数', values['median'], 175.0, '29_eda_statistical_analysis/02_薪资描述统计')
    record_anchor('薪资 IQR', values['iqr'], 90.0, '29_eda_statistical_analysis/02_薪资描述统计')
    record_anchor('薪资 P10', values['p10'], 110.0, '29_eda_statistical_analysis/02_薪资描述统计')
    record_anchor('薪资 P90', values['p90'], 300.0, '29_eda_statistical_analysis/02_薪资描述统计')
    for name, expect in [('无认证', 9104), ('行业认证', 460), ('最佳雇主', 4542), ('两者均有', 777)]:
        record_anchor(f'公司认证-{name}', values['cert_counts'][name], expect,
                      '29_eda_statistical_analysis/11_统计检验（认证块）')
    record_anchor('公司认证 ε²', values['cert_epsilon2'], 0.197545,
                  '29_eda_statistical_analysis/11_统计检验（认证块）')
    record_anchor('福利标签原子标签', values['tag_atoms'], 4328,
                  '32_company_field_semantic_audit/02_宽表字段统计')
    record_anchor('福利标签原子标签（指标日志）', values['tag_atoms_stage13'], 4328,
                  'outputs/logs/metrics/stage_13_eda.json:company_tag_atoms')
    record_anchor('福利标签进入比较', values['tag_tested'], 183,
                  '29_eda_statistical_analysis/11_统计检验（标签块）')
    record_anchor('福利标签校正后显著', values['tag_significant'], 114,
                  '29_eda_statistical_analysis/11_统计检验（标签块）')
    record_anchor('前5标签交集岗位数', values['top5_intersection'], 731,
                  'stage_13_eda.json:top_cliff_overlap.全部交集岗位数')
    record_anchor('前5标签并集岗位数', values['top5_union'], 784,
                  'stage_13_eda.json:top_cliff_overlap.全部并集岗位数')
    record_anchor('前5标签最大 Jaccard', values['top5_max_jaccard'], 1.0,
                  'stage_13_eda.json:top_cliff_overlap.两两最大Jaccard')
    record_anchor('技能主口径', values['skill_main'], 8822, '27_skill_eda_scope_audit/01_样本口径')
    record_anchor('技能扩展口径', values['skill_extended'], 16378, '27_skill_eda_scope_audit/01_样本口径')
    record_anchor('技能仅模板口径', values['skill_empty'], 766, '27_skill_eda_scope_audit/01_样本口径')
    record_anchor('技能列数', values['skill_dims'], 44, '30_model_comparison/02_特征维度')
    record_anchor('文本维度', values['text_dims'], 16, '30_model_comparison/02_特征维度')
    record_anchor('最终特征维度', values['feature_total'], 320, '30_model_comparison/02_特征维度')
    record_anchor('基础特征维度 A+B+C', values['base_dims'], 241, '31_ablation_robustness_shap/01_消融结果')
    record_anchor('训练集样本数', values['split_counts']['train'], 10418, '30_model_comparison/01_数据划分')
    record_anchor('验证集样本数', values['split_counts']['validation'], 2232, '30_model_comparison/01_数据划分')
    record_anchor('测试集样本数', values['split_counts']['test'], 2233, '30_model_comparison/01_数据划分')

    final_row = data['test_result'].set_index('模型').loc['FINAL（LightGBM）']
    record_anchor('主模型 test MAE', float(final_row['test_MAE']), 35.335506,
                  '30_model_comparison/09_Test最终结果')
    record_anchor('主模型 test RMSE', float(final_row['test_RMSE']), 64.623479,
                  '30_model_comparison/09_Test最终结果')
    record_anchor('主模型 test R²', float(final_row['test_R2']), 0.584606,
                  '30_model_comparison/09_Test最终结果')
    group_split = data['split_compare']
    group_mae = group_split[(group_split['划分方式'].str.startswith('Company Group Split')) &
                            (group_split['数据子集'] == 'test')]['MAE']
    record_anchor('按公司分组划分 test MAE', float(group_mae.iloc[0]), 52.049937,
                  '31_ablation_robustness_shap/04_Random_vs_GroupSplit')
    return values


# ------------------------------------------------------------------ 图 S01
def fig_s01(fix: int = 0):
    v = V
    labels = ['原始搜索观测', '唯一岗位实体', '明确薪资可解析', '正式薪资样本']
    values = [v['n_raw_obs'], v['n_unique_jobs'], v['n_parsed'], v['n_final']]
    colors = [BLUE, PALETTE[2], PALETTE[5], RED]
    fig, ax = panel_fig(fix, 1, (8.8, 4.9))
    top = max(values)
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        row = len(values) - 1 - index
        ax.barh(row, value, left=(top - value) / 2.0, height=0.62, color=color,
                edgecolor='black', linewidth=0.6)
        ax.text(top * 1.03, row, f'{value:,}（占原始观测 {value / top * 100:.2f}%）',
                ha='left', va='center', fontsize=FS['annotation'])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels[::-1])
    ax.set_xlim(-0.05 * top, 1.95 * top)
    ax.set_ylim(-0.75, len(labels) - 0.25)
    ax.set_xlabel('岗位数量（个）')
    ax.set_ylabel('筛选与口径层级')
    note(ax, f'折叠重复观测 {v["n_merged_dup"]:,}\n'
             f'面议 {v["n_negotiable"]:,}（占唯一岗位 {v["n_negotiable"] / v["n_unique_jobs"] * 100:.2f}%，不进正式样本）\n'
             f'薪资逻辑异常 {v["n_anomaly"]:,}（占可解析 {v["n_anomaly"] / v["n_parsed"] * 100:.3f}%，不进正式样本）',
         x=0.02, y=0.30, ha='left', va='center')
    plot_style.apply_sci_axis(ax, grid_axis='x', grid=True)
    plot_style.format_integer_axis(ax, axis='x')
    return fig, []


# ------------------------------------------------------------------ 图 S02
def fig_s02(fix: int = 0):
    anomaly = V['anomaly']
    fig, axes = panel_fig(fix, 2, (11.4, 4.9), wspace=0.42)
    ax_a, ax_b = axes

    type_counts = anomaly['异常类型'].value_counts()
    name_map = {'FULL_SHIFT': '性质与规模全量偏移', 'NATURE_ONLY': '仅性质栏位偏移',
                'SIZE_ONLY': '仅规模栏位偏移'}
    series = pd.Series({name_map.get(k, k): int(v) for k, v in type_counts.items()})
    plot_style.bar_ranked(ax_a, series, color=BLUE, xlabel='语义槽位异常类型',
                          ylabel='涉及记录数')
    plot_style.apply_sci_axis(ax_a, grid_axis='y', grid=True)
    plot_style.format_integer_axis(ax_a, axis='y')
    note(ax_a, f'合计 {int(type_counts.sum())} 条记录\n涉及 {int(anomaly["实习岗位ID"].nunique())} 个岗位')

    axes_names = ['公司规模槽位', '公司性质槽位', '公司所在地槽位']
    changed = [int((anomaly['公司规模_是否变化'] == '是').sum()),
               int((anomaly['公司性质_是否变化'] == '是').sum()),
               int((anomaly['公司所在地_是否变化'] == '是').sum())]
    residual_row = V['residual']
    residual = [int(residual_row.get('修复后规模像地点行数', 0)),
                int(residual_row.get('修复后性质像规模行数', 0)),
                int(residual_row.get('修复后残留异常行数', 0))]
    positions = np.arange(len(axes_names))
    width = 0.34 + 0.02 * fix
    ax_b.bar(positions - width / 2, changed, width=width, color=BLUE,
             edgecolor='black', linewidth=0.5, label='修复前异常记录数')
    ax_b.bar(positions + width / 2, residual, width=width, color=ORANGE,
             edgecolor='black', linewidth=0.5, label='修复后残留记录数')
    for position, value in zip(positions, changed):
        ax_b.text(position - width / 2, value + 12, f'{value:,}', ha='center',
                  va='bottom', fontsize=FS['annotation'])
    for position, value in zip(positions, residual):
        ax_b.text(position + width / 2, value + 12, f'{value:,}', ha='center',
                  va='bottom', fontsize=FS['annotation'])
    ax_b.set_xticks(positions)
    ax_b.set_xticklabels(axes_names)
    ax_b.set_xlabel('公司属性三字段槽位')
    ax_b.set_ylabel('槽位异常记录数')
    ax_b.set_ylim(0, max(changed) * 1.28)
    plot_style.apply_sci_axis(ax_b, grid_axis='y', grid=True)
    plot_style.format_integer_axis(ax_b, axis='y')
    legend_above(ax_b, ncol=2, fix=fix)

    plot_style.add_subfigure_caption(ax_a, 'a', '异常三类构成')
    plot_style.add_subfigure_caption(ax_b, 'b', '修复前后残留对照')
    return fig, [('a', '异常三类构成', ax_a), ('b', '修复前后残留对照', ax_b)]


# ------------------------------------------------------------------ 图 S03
def fig_s03(fix: int = 0):
    split = V['split_table']
    fig, axes = panel_fig(fix, 2, (11.4, 4.9), wspace=0.30)
    ax_a, ax_b = axes

    sizes = pd.Series({f'{SPLIT_LABELS[k]}\n（{k}）': int(split[k]) for k in ['train', 'validation', 'test']})
    plot_style.bar_ranked(ax_a, sizes, color=BLUE, xlabel='数据子集', ylabel='岗位数量')
    plot_style.apply_sci_axis(ax_a, grid_axis='y', grid=True)
    plot_style.format_integer_axis(ax_a, axis='y')
    total = int(sum(split.values()))
    shares = [split[k] / total * 100 for k in ['train', 'validation', 'test']]
    note(ax_a, f'合计 {total:,}\n占比 %.2f%% / %.2f%% / %.2f%%' % tuple(shares))

    groups = [JOB.loc[JOB['split'] == key, '薪资中点'].dropna().to_numpy() for key in SPLIT_ORDER]
    labels = [f'{SPLIT_LABELS[key]}\n(n = {len(group):,})' for key, group in zip(SPLIT_ORDER, groups)]
    box_groups(ax_b, groups, labels, [BLUE, ORANGE, GREEN], '薪资中点（元/天）', '数据子集')
    plot_style.apply_sci_axis(ax_b, grid_axis='y', grid=True)
    plot_style.format_integer_axis(ax_b, axis='y')
    note(ax_b, '中位数均为 %d 元/天' % int(V['median']))

    plot_style.add_subfigure_caption(ax_a, 'a', '子集规模')
    plot_style.add_subfigure_caption(ax_b, 'b', '子集薪资分布')
    return fig, [('a', '子集规模', ax_a), ('b', '子集薪资分布', ax_b)]


# ------------------------------------------------------------------ 图 S04
def fig_s04(fix: int = 0):
    stats = V['stats_table']
    pair = stats[(stats['检验块'] == '成对比较（Mann–Whitney + BH-FDR）') &
                 (stats['因素'] == '公司认证')].copy()
    pair['比较'] = pair['取值A'] + ' vs ' + pair['取值B']
    pair['abs_delta'] = pair['cliff_delta'].abs()
    pair = pair.sort_values('abs_delta', ascending=False)

    fig, axes = panel_fig(fix, 2, (11.6, 4.9), wspace=0.44)
    ax_a, ax_b = axes

    groups = [JOB.loc[JOB['公司认证四类'] == name, '薪资中点'].dropna().to_numpy()
              for name in CERT_ORDER]
    labels = [f'{name}\n(n = {len(group):,})' for name, group in zip(CERT_ORDER, groups)]
    box_groups(ax_a, groups, labels, [PALETTE[2], ORANGE, GREEN, RED],
               '薪资中点（元/天）', '公司认证状态')
    plot_style.apply_sci_axis(ax_a, grid_axis='y', grid=True)
    plot_style.format_integer_axis(ax_a, axis='y')
    note(ax_a, f'Kruskal–Wallis（{V["cert_groups"]} 组）\nε² = {V["cert_epsilon2"]:.6f}')

    positions = np.arange(len(pair))
    color_map = {True: RED, False: GRAY}
    for position, (_, row) in zip(positions, pair.iterrows()):
        significant = row['FDR显著'] == '是'
        ax_b.hlines(position, 0, row['abs_delta'], color=color_map[significant], linewidth=1.3)
        ax_b.plot(row['abs_delta'], position, 'o', markersize=5.2,
                  color=color_map[significant], markeredgecolor='black', markeredgewidth=0.5)
        ax_b.text(row['abs_delta'] + 0.022, position, f'{row["abs_delta"]:.6f}',
                  va='center', ha='left', fontsize=FS['annotation'])
    ax_b.set_yticks(positions)
    ax_b.set_yticklabels(pair['比较'])
    ax_b.set_xlim(0, pair['abs_delta'].max() * 1.34)
    ax_b.set_ylim(-0.7, len(pair) - 0.3)
    ax_b.set_xlabel("|Cliff's δ|（效应量）")
    ax_b.set_ylabel('组间两两比较')
    plot_style.apply_sci_axis(ax_b, grid_axis='x', grid=True)
    legend_above(ax_b, ncol=2, handles=[Line2D([], [], marker='o', linestyle='none',
                                               color=RED, markeredgecolor='black', markersize=5.2),
                                        Line2D([], [], marker='o', linestyle='none',
                                               color=GRAY, markeredgecolor='black', markersize=5.2)],
                 labels=['校正后显著（q < 0.05）', '校正后不显著'], fix=fix)

    plot_style.add_subfigure_caption(ax_a, 'a', '四类分布')
    plot_style.add_subfigure_caption(ax_b, 'b', '六组成对比较效应量')
    return fig, [('a', '四类分布', ax_a), ('b', '六组成对比较效应量', ax_b)]


# ------------------------------------------------------------------ 图 S05
def fig_s05(fix: int = 0):
    fig, ax = panel_fig(fix, 1, (8.8, 5.2))
    tag_effect_panel(ax, fix=fix, legend_ncol=4)
    return fig, []


# ------------------------------------------------------------------ 图 S06
def fig_s06(fix: int = 0):
    fig, axes = panel_fig(fix, 2, (11.8, 5.0), wspace=0.55)
    ax_a, ax_b = axes
    tag_overlap_bars(ax_a)
    tag_jaccard_heatmap(ax_b, fix=fix)

    plot_style.add_subfigure_caption(ax_a, 'a', '前 5 标签交集与并集规模')
    plot_style.add_subfigure_caption(ax_b, 'b', '前 5 标签两两 Jaccard 热力图')
    return fig, [('a', '前 5 标签交集与并集规模', ax_a),
                 ('b', '前 5 标签两两 Jaccard 热力图', ax_b)]


# ------------------------------------------------------------------ 图 S05S06（合并双子图）
def fig_s05s06(fix: int = 0):
    """图 5-3 公司福利标签的薪资关联及共现特征（S05 + S06 合并为 (a)(b) 双子图）。

    只复用 S05 / S06 的面板绘制与数据读取逻辑（同一份冻结数据），不重新计算任何统计量。
    """
    matrix = np.array(V['top5_jaccard'])
    off_diagonal = ~np.eye(len(TOP5_TAGS), dtype=bool)
    jaccard_min = float(matrix[off_diagonal].min())
    jaccard_max = float(matrix[off_diagonal].max())
    flat_index = int(np.where(off_diagonal, matrix, -np.inf).argmax())
    pair = np.unravel_index(flat_index, matrix.shape)
    inter = V['top5_intersection']
    union = V['top5_union']

    fig, axes = panel_fig(fix, 2, (13.2, 5.2), wspace=0.42)
    ax_a, ax_b = axes
    tag_effect_panel(ax_a, fix=fix, legend_ncol=2)
    tag_overlap_bars(ax_b, show_universe_lines=True)
    ax_b.text(0.5, 1.02,
              f'交集 {inter:,} / 并集 {union:,}（交集占并集 {inter / union:.2%}）\n'
              f'两两 Jaccard ∈ [{jaccard_min:.3f}, {jaccard_max:.3f}]，最大 {jaccard_max:.3f}\n'
              f'最大共现对：{TOP5_TAGS[pair[0]]} × {TOP5_TAGS[pair[1]]}',
              transform=ax_b.transAxes, ha='center', va='bottom',
              fontsize=FS['annotation'], linespacing=1.5)

    plot_style.add_subfigure_caption(ax_a, 'a', '福利标签效应量与 BH-FDR 显著性')
    plot_style.add_subfigure_caption(ax_b, 'b', '高效应福利标签共现结构')
    return fig, [('a', '福利标签效应量与 BH-FDR 显著性', ax_a),
                 ('b', '高效应福利标签共现结构', ax_b)]


# ------------------------------------------------------------------ 图 S07
def fig_s07(fix: int = 0):
    kw = V['kw_table'].copy()
    kw['对象'] = kw['检验对象'].astype(str)
    kw = kw.sort_values('效应量', ascending=True).reset_index(drop=True)
    fig, ax = panel_fig(fix, 1, (8.4, 5.6))
    positions = np.arange(len(kw))
    for position, (_, row) in zip(positions, kw.iterrows()):
        color = RED if row['对象'] == '公司认证' else BLUE
        ax.hlines(position, 0, row['效应量'], color=color, linewidth=1.4)
        ax.plot(row['效应量'], position, 'o', markersize=5.4, color=color,
                markeredgecolor='black', markeredgewidth=0.5)
        ax.text(row['效应量'] + 0.0045, position,
                f'{row["效应量"]:.6f}（取值数 {int(row["参与检验组数"])}）',
                va='center', ha='left', fontsize=FS['annotation'])
    ax.set_yticks(positions)
    ax.set_yticklabels(kw['对象'])
    ax.set_xlim(0, kw['效应量'].max() * 1.62)
    ax.set_ylim(-0.7, len(kw) - 0.3)
    ax.set_xlabel('ε²（Kruskal–Wallis 效应量）')
    ax.set_ylabel('结构化因素')
    plot_style.apply_sci_axis(ax, grid_axis='x', grid=True)
    note(ax, '仅纳入样本数 ≥ 30 的取值', x=0.985, y=0.03, ha='right', va='bottom')
    return fig, []


# ------------------------------------------------------------------ 图 S08
def fig_s08(fix: int = 0):
    table = V['skill_count_table']
    diffs = ['0', '1', '2', '3', '4', '5+']
    fig, axes = panel_fig(fix, 2, (11.6, 4.9), wspace=0.30)
    ax_a, ax_b = axes

    frame = JOB.copy()
    frame['档'] = frame['技能数量'].fillna(0).clip(upper=5).astype(int).astype(str)
    frame.loc[frame['技能数量'] >= 5, '档'] = '5+'
    groups = [frame.loc[frame['档'] == key, '薪资中点'].dropna().to_numpy() for key in diffs]
    labels = [f'{key}\n(n = {len(group):,})' for key, group in zip(diffs, groups)]
    box_groups(ax_a, groups, labels, [BLUE, BLUE, ORANGE, ORANGE, ORANGE, RED],
               '薪资中点（元/天）', '技能数量档')
    plot_style.apply_sci_axis(ax_a, grid_axis='y', grid=True)
    plot_style.format_integer_axis(ax_a, axis='y')
    note(ax_a, f'Kruskal–Wallis（{V["skill_count_groups"]} 组）\nε² = {V["skill_count_epsilon2"]:.6f}')

    sample = table.set_index('技能数量档')['样本数'].reindex(diffs).to_numpy(dtype=float)
    median = table.set_index('技能数量档')['中位数'].reindex(diffs).to_numpy(dtype=float)
    positions = np.arange(len(diffs))
    ax_b.bar(positions, sample, width=0.58, color=BLUE, edgecolor='black',
             linewidth=0.5, label='样本数（左轴）')
    ax_b.set_xlabel('技能数量档')
    ax_b.set_ylabel('岗位数量（个）')
    ax_b.set_ylim(0, sample.max() * 1.22)
    ax_b.set_xticks(positions)
    ax_b.set_xticklabels(diffs)
    plot_style.apply_sci_axis(ax_b, grid_axis='y', grid=True)
    plot_style.format_integer_axis(ax_b, axis='y')
    for position, value in zip(positions, sample):
        ax_b.text(position, value + sample.max() * 0.022, f'{int(value):,}', ha='center',
                  va='bottom', fontsize=FS['annotation'])

    twin = ax_b.twinx()
    twin.plot(positions, median, marker='o', markersize=5.2, linewidth=1.5, color=RED,
              markeredgecolor='black', markeredgewidth=0.5, label='薪资中点中位数（右轴）')
    twin.set_ylabel('薪资中点中位数（元/天）')
    twin.set_xlabel('技能数量档')
    twin.set_ylim(0, median.max() * 1.35)
    plot_style.apply_sci_axis(twin, grid_axis='y', grid=False)
    legend_above(ax_b, ncol=2,
                 handles=[Patch(facecolor=BLUE, edgecolor='black', label='样本数（左轴）'),
                          Line2D([], [], marker='o', color=RED, markeredgecolor='black',
                                 markersize=5.2, label='薪资中点中位数（右轴）')],
                 labels=['样本数（左轴）', '薪资中点中位数（右轴）'], fix=fix)

    plot_style.add_subfigure_caption(ax_a, 'a', '各档薪资分布')
    plot_style.add_subfigure_caption(ax_b, 'b', '样本量与中位数')
    return fig, [('a', '各档薪资分布', ax_a), ('b', '样本量与中位数', ax_b)]


# ------------------------------------------------------------------ 图 S09
def fig_s09(fix: int = 0):
    before_map = V['skill_salary_map']
    after_map = V['skill_in_cat_agg']
    skills = V['skill_in_cat_order']
    before = [before_map[skill] for skill in skills]
    after = [after_map[skill] for skill in skills]
    colors = [BLUE, ORANGE, GREEN, RED, PURPLE, BROWN]

    fig, ax = panel_fig(fix, 1, (8.6, 5.4))
    for index, skill in enumerate(skills):
        ax.plot([0, 1], [before[index], after[index]], marker='o', markersize=5.6,
                linewidth=1.6, color=colors[index % len(colors)],
                label=f'{skill}：{before[index]:+.0f} → {after[index]:+.1f}',
                markeredgecolor='black', markeredgewidth=0.45)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['整体差异\n（未控制岗位细分类）', '细分类内差异\n（控制岗位细分类）'])
    ax.set_xlim(-0.32, 1.32)
    values = before + after
    ax.set_ylim(min(values) - (max(values) - min(values)) * 0.18,
                max(values) + (max(values) - min(values)) * 0.30)
    ax.set_xlabel('对比口径（薪资中点中位数差）')
    ax.set_ylabel('有技能 − 无技能的薪资中点中位数差（元/天）')
    plot_style.apply_sci_axis(ax, grid_axis='y', grid=True)
    legend_above(ax, ncol=3, fix=fix)
    note(ax, '控制后为同岗位细分类内部对比\n（按细分类内有技能岗位数加权）；虚线 = 0',
         x=0.02, y=0.97, ha='left', box=True)
    return fig, []


# ------------------------------------------------------------------ 图 S10
def fig_s10(fix: int = 0):
    extreme = V['robust_extreme'].set_index('实验')
    target = V['robust_target'].set_index('目标口径')
    baseline_mae = float(extreme.loc['主模型（真实薪资 y）', 'MAE'])
    baseline_r2 = float(extreme.loc['主模型（真实薪资 y）', 'R²'])

    labels = ['主模型\n（真实薪资）', '训练目标 1%/99% 截断', '薪资中点\n（主任务）',
              '薪资下限', '薪资上限']
    values = [baseline_mae, float(extreme.loc['训练集 y winsorize 1%/99%', 'MAE']),
              float(target.loc['薪资中点（主任务）', 'MAE']),
              float(target.loc['薪资下限', 'MAE']), float(target.loc['薪资上限', 'MAE'])]
    values_r2 = [baseline_r2, float(extreme.loc['训练集 y winsorize 1%/99%', 'R²']),
                 float(target.loc['薪资中点（主任务）', 'R²']),
                 float(target.loc['薪资下限', 'R²']), float(target.loc['薪资上限', 'R²'])]
    colors = [BLUE, ORANGE, BLUE, ORANGE, ORANGE]

    fig, axes = panel_fig(fix, 2, (11.8, 4.9), wspace=0.26)
    ax_a, ax_b = axes
    positions = np.arange(len(labels))
    for ax, series, baseline, ylabel in ((ax_a, values, baseline_mae, 'MAE（元/天）'),
                                         (ax_b, values_r2, baseline_r2, 'R²')):
        ax.bar(positions, series, width=0.62, color=colors, edgecolor='black', linewidth=0.5)
        ax.axhline(baseline, color=RED, linestyle='--', linewidth=0.9)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_xlabel('稳健性检查实验')
        ax.set_ylabel(ylabel)
        ax.set_ylim(min(series) * 0.88, max(series) * 1.12)
        plot_style.apply_sci_axis(ax, grid_axis='y', grid=True)
        digits = 3 if ylabel == 'R²' else 6
        for position, value in zip(positions, series):
            ax.text(position, value + (max(series) - min(series)) * 0.03,
                    f'{value:.{digits}f}', ha='center', va='bottom', fontsize=FS['annotation'])
    note(ax_a, '虚线 = 主模型基准', x=0.985, y=0.03, ha='right', va='bottom')
    note(ax_b, '虚线 = 主模型基准', x=0.985, y=0.03, ha='right', va='bottom')
    legend_above(ax_b, ncol=2, handles=[Patch(facecolor=BLUE, edgecolor='black', label='主任务 / 主模型'),
                                        Patch(facecolor=ORANGE, edgecolor='black', label='稳健性替代口径')],
                 labels=['主任务 / 主模型', '稳健性替代口径'], fix=fix)

    plot_style.add_subfigure_caption(ax_a, 'a', 'MAE 对照')
    plot_style.add_subfigure_caption(ax_b, 'b', 'R² 对照')
    return fig, [('a', 'MAE 对照', ax_a), ('b', 'R² 对照', ax_b)]


# ------------------------------------------------------------------ 图 S11
def fig_s11(fix: int = 0):
    rows = V['increment_rows']
    fig, ax = panel_fig(fix, 1, (9.0, 5.2))
    positions = np.arange(len(rows))[::-1]
    color_map = {'validation': ORANGE, 'test': BLUE}
    for position, row in zip(positions, rows):
        color = color_map[row['子集']]
        delta = row['delta']
        ax.errorbar(delta, position, xerr=[[delta - row['lo']], [row['hi'] - delta]],
                    fmt='o', markersize=6.0, color=color, ecolor=color, elinewidth=1.4,
                    capsize=3.2, capthick=1.0, markeredgecolor='black', markeredgewidth=0.5)
        ax.text(row['hi'] + 0.06, position, f'{delta:+.3f} [{row["lo"]:+.3f}, {row["hi"]:+.3f}]',
                va='center', ha='left', fontsize=FS['annotation'])
    ax.axvline(0, color='black', linestyle='--', linewidth=0.9)
    ax.set_yticks(positions)
    ax.set_yticklabels([row['标签'] for row in rows])
    low = min(row['lo'] for row in rows)
    high = max(row['hi'] for row in rows)
    ax.set_xlim(low - 0.25, high + (high - low) * 0.95)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel('ΔMAE（元/天；正值表示加入该特征组后误差下降）')
    ax.set_ylabel('特征组增量比较')
    plot_style.apply_sci_axis(ax, grid_axis='x', grid=True)
    legend_above(ax, ncol=2,
                 handles=[Line2D([], [], marker='o', color=ORANGE, linestyle='none',
                                 markeredgecolor='black', markersize=5.6),
                          Line2D([], [], marker='o', color=BLUE, linestyle='none',
                                 markeredgecolor='black', markersize=5.6)],
                 labels=['验证集', '测试集'], fix=fix)
    note(ax, '配对 bootstrap（1000 轮，种子 42）95% 置信区间\n虚线 = 0（无增量）',
         x=0.99, y=0.03, ha='right', va='bottom')
    return fig, []


# ------------------------------------------------------------------ 图 S12
def fig_s12(fix: int = 0):
    rank = V['shap_rank'].head(10).copy()
    stability = V['shap_stability']
    fig, ax = panel_fig(fix, 1, (8.6, 5.4))
    positions = np.arange(len(rank))[::-1]
    color_map = {'正向（提高预测薪资）': RED, '负向（降低预测薪资）': BLUE}
    for position, (_, row) in zip(positions, rank.iterrows()):
        color = color_map.get(row['方向性'], GRAY)
        ax.hlines(position, 0, row['mean_abs_SHAP'], color=color, linewidth=1.4)
        ax.plot(row['mean_abs_SHAP'], position, 'o', markersize=5.4, color=color,
                markeredgecolor='black', markeredgewidth=0.5)
        ax.text(row['mean_abs_SHAP'] + 0.28, position, f'{row["mean_abs_SHAP"]:.3f}',
                va='center', ha='left', fontsize=FS['annotation'])
    ax.set_yticks(positions)
    ax.set_yticklabels([f'{int(row["排名"])}. {row["特征"]}' for _, row in rank.iterrows()])
    ax.set_xlim(0, rank['mean_abs_SHAP'].max() * 1.30)
    ax.set_ylim(-0.7, len(rank) - 0.3)
    ax.set_xlabel('平均 |SHAP|（元/天）')
    ax.set_ylabel('主模型 SHAP 排名前 10 特征')
    plot_style.apply_sci_axis(ax, grid_axis='x', grid=True)
    legend_above(ax, ncol=2,
                 handles=[Line2D([], [], marker='o', color=RED, linestyle='none',
                                 markeredgecolor='black', markersize=5.4),
                          Line2D([], [], marker='o', color=BLUE, linestyle='none',
                                 markeredgecolor='black', markersize=5.4)],
                 labels=['正向（提高预测薪资）', '负向（降低预测薪资）'], fix=fix)
    spearman = '；'.join(f'种子 {int(row["随机种子"])} = {row["与主模型 mean|SHAP| 排名 Spearman"]:.4f}'
                         for _, row in stability.iterrows())
    note(ax, f'与主模型平均 |SHAP| 排名的 Spearman：{spearman}\n'
             f'（{len(stability)} 个随机种子的前 10 特征集合完全一致）',
         x=0.99, y=0.03, ha='right', va='bottom')
    return fig, []


# ------------------------------------------------------------------ 图 S13
def fig_s13(fix: int = 0):
    threshold = V['skill_threshold'].copy().sort_values('阈值')
    text_dim = V['text_dim'].copy()
    fig, axes = panel_fig(fix, 2, (11.6, 4.9), wspace=0.30)
    ax_a, ax_b = axes

    best = threshold.loc[threshold['validation_MAE'].idxmin()]
    ax_a.plot(threshold['阈值'], threshold['validation_MAE'], marker='o', markersize=5.8,
              linewidth=1.6, color=BLUE, markeredgecolor='black', markeredgewidth=0.5)
    for _, row in threshold.iterrows():
        offset = 0.0016 if row['阈值'] != best['阈值'] else 0.0026
        ax_a.text(row['阈值'], row['validation_MAE'] + offset,
                  f'技能列 {int(row["技能列数"])}',
                  ha='center', va='bottom', fontsize=FS['annotation'])
    ax_a.plot(best['阈值'], best['validation_MAE'], marker='D', markersize=7.0, color=RED,
              markeredgecolor='black', markeredgewidth=0.6, linestyle='none')
    ax_a.set_xlabel('技能频率阈值（训练集统计）')
    ax_a.set_ylabel('验证集 MAE（元/天）')
    ax_a.set_xticks(sorted(threshold['阈值'].tolist()))
    plot_style.apply_sci_axis(ax_a, grid_axis='y', grid=True)
    note(ax_a, f'选定阈值 {int(best["阈值"])}（验证集 MAE 最低 = {best["validation_MAE"]:.6f}）')

    labels = [f'{int(v)}\n维' if str(v).strip().isdigit() else '不使用\n文本语义组'
              for v in text_dim['文本维度'].tolist()]
    values = text_dim['validation_MAE'].to_numpy(dtype=float)
    colors = [BLUE if str(v).isdigit() else GRAY for v in text_dim['文本维度'].tolist()]
    positions = np.arange(len(labels))
    ax_b.bar(positions, values, width=0.6, color=colors, edgecolor='black', linewidth=0.5)
    for position, value in zip(positions, values):
        ax_b.text(position, value + (values.max() - values.min()) * 0.06, f'{value:.6f}',
                  ha='center', va='bottom', fontsize=FS['annotation'])
    ax_b.set_xticks(positions)
    ax_b.set_xticklabels(labels)
    ax_b.set_xlabel('文本语义维度（SVD 分量数）')
    ax_b.set_ylabel('验证集 MAE（元/天）')
    ax_b.set_ylim(values.min() - (values.max() - values.min()) * 0.35,
                  values.max() + (values.max() - values.min()) * 0.20)
    plot_style.apply_sci_axis(ax_b, grid_axis='y', grid=True)
    legend_above(ax_b, ncol=2,
                 handles=[Patch(facecolor=BLUE, edgecolor='black', label='候选维度'),
                          Patch(facecolor=GRAY, edgecolor='black', label='不使用文本语义组（参考）')],
                 labels=['候选维度', '不使用文本语义组（参考）'], fix=fix)

    plot_style.add_subfigure_caption(ax_a, 'a', '技能阈值')
    plot_style.add_subfigure_caption(ax_b, 'b', '文本维度')
    return fig, [('a', '技能阈值', ax_a), ('b', '文本维度', ax_b)]


# ------------------------------------------------------------------ 图 S14
def fig_s14(fix: int = 0):
    table = V['tag_per_job']
    counts = table['每岗标签数'].to_numpy(dtype=int)
    jobs = table['岗位数'].to_numpy(dtype=float)
    cumulative = np.cumsum(jobs) / jobs.sum()
    positions = np.arange(len(counts))

    fig, ax = panel_fig(fix, 1, (9.6, 5.2))
    ax.bar(positions, jobs, width=0.66, color=BLUE, edgecolor='black', linewidth=0.5,
           label='岗位数量（左轴）')
    ax.set_xticks(positions)
    ax.set_xticklabels([str(int(v)) for v in counts], rotation=0)
    ax.set_xlabel('每个岗位命中的公司福利标签数量（个）')
    ax.set_ylabel('岗位数量（个）')
    ax.set_ylim(0, jobs.max() * 1.20)
    plot_style.apply_sci_axis(ax, grid_axis='y', grid=True)
    plot_style.format_integer_axis(ax, axis='y')
    for position, value in zip(positions, jobs):
        ax.text(position, value + jobs.max() * 0.02, f'{int(value):,}', ha='center',
                va='bottom', fontsize=FS['annotation'], rotation=90 if fix else 0)

    twin = ax.twinx()
    twin.plot(positions, cumulative, marker='o', markersize=4.6, linewidth=1.4, color=RED,
              markeredgecolor='black', markeredgewidth=0.4, label='累计岗位占比（右轴）')
    twin.set_ylim(0, 1.06)
    twin.set_ylabel('累计岗位占比')
    twin.set_xlabel('每个岗位命中的公司福利标签数量（个）')
    plot_style.apply_sci_axis(twin, grid_axis='y', grid=False)
    plot_style.format_percent_axis(twin, axis='y', decimals=0)
    legend_above(ax, ncol=2,
                 handles=[Patch(facecolor=BLUE, edgecolor='black', label='岗位数量（左轴）'),
                          Line2D([], [], marker='o', color=RED, markeredgecolor='black',
                                 markersize=4.6, label='累计岗位占比（右轴）')],
                 labels=['岗位数量（左轴）', '累计岗位占比（右轴）'], fix=fix)
    zero_share = jobs[counts == 0].sum() / jobs.sum() if (counts == 0).any() else 0.0
    note(ax, f'合计 {int(jobs.sum()):,} 个岗位\n无标签岗位 {int(jobs[counts == 0].sum()):,}'
             f'（{zero_share * 100:.1f}%）',
         x=0.99, y=0.97, ha='right', va='top')
    return fig, []


# ------------------------------------------------------------------ 图 S15
def fig_s15(fix: int = 0):
    blacklist = V['leak_blacklist']
    manifest = V['manifest']
    fig, axes = panel_fig(fix, 2, (11.4, 4.9), wspace=0.30)
    ax_a, ax_b = axes

    total_items = int(len(blacklist))
    from_layer = int((blacklist['是否存在于薪资源层'] == '是').sum())
    in_features = int((blacklist['是否出现在 X'] == '是').sum())
    labels = ['黑名单条目', '薪资源层中存在', '进入特征矩阵']
    values = [total_items, from_layer, in_features]
    colors = [BLUE, ORANGE, GREEN]
    positions = np.arange(len(labels))
    ax_a.bar(positions, values, width=0.6, color=colors, edgecolor='black', linewidth=0.5)
    for position, value in zip(positions, values):
        ax_a.text(position, value + total_items * 0.03, f'{value}', ha='center',
                  va='bottom', fontsize=FS['annotation'])
    ax_a.set_xticks(positions)
    ax_a.set_xticklabels(labels)
    ax_a.set_ylim(0, total_items * 1.24)
    ax_a.set_xlabel('薪资相关字段检查项')
    ax_a.set_ylabel('字段数量（个）')
    plot_style.apply_sci_axis(ax_a, grid_axis='y', grid=True)
    note(ax_a, '另设命名模式防护：列名含\n薪资 / 薪酬 / 工资 / 面议 / 待遇\n的字段一律禁止进入特征矩阵\n（实测命中 0）',
         x=0.98, y=0.97, ha='right', va='top')

    group_counts = (manifest[manifest['是否参与模型'] == 1]
                    .groupby('特征组')['字段名'].count())
    order = ['A_岗位基础', 'B_地域', 'C_公司', 'D_技能', 'E_文本语义']
    values_b = [int(group_counts.get(key, 0)) for key in order]
    labels_b = ['A\n岗位基础', 'B\n地域', 'C\n公司', 'D\n技能', 'E\n文本语义']
    positions_b = np.arange(len(order))
    colors_b = [BLUE, BLUE, BLUE, ORANGE, RED]
    ax_b.bar(positions_b, values_b, width=0.6, color=colors_b, edgecolor='black', linewidth=0.5)
    for position, value in zip(positions_b, values_b):
        ax_b.text(position, value + 0.5, f'{value}', ha='center', va='bottom',
                  fontsize=FS['annotation'])
    ax_b.set_xticks(positions_b)
    ax_b.set_xticklabels(labels_b)
    ax_b.set_ylim(0, max(values_b) * 1.35)
    ax_b.set_xlabel('特征组')
    ax_b.set_ylabel('参与建模字段数（个）')
    plot_style.apply_sci_axis(ax_b, grid_axis='y', grid=True)
    legend_above(ax_b, ncol=3,
                 handles=[Patch(facecolor=BLUE, edgecolor='black', label='基础特征组'),
                          Patch(facecolor=ORANGE, edgecolor='black', label='技能特征组'),
                          Patch(facecolor=RED, edgecolor='black', label='文本语义特征组')],
                 labels=['基础特征组', '技能特征组', '文本语义特征组'], fix=fix)
    note(ax_b, f'参与建模字段合计 {int(sum(values_b))}\n'
               f'编码后特征维度：基础 A+B+C = {V["base_dims"]}（消融基准），'
               f'技能 = {V["skill_dims"]}，文本 = {V["text_dims"]}，最终 = {V["feature_total"]}',
         x=0.5, y=0.97, ha='center', va='top', )

    plot_style.add_subfigure_caption(ax_a, 'a', '泄漏字段检查状态')
    plot_style.add_subfigure_caption(ax_b, 'b', '五组特征维度')
    return fig, [('a', '泄漏字段检查状态', ax_a), ('b', '五组特征维度', ax_b)]


# ------------------------------------------------------------------ 门禁与执行
GATE_KEYS = ('inward_ticks', 'light_grid', 'caption_structure_ok', 'caption_ink_visible',
             'caption_not_cropped', 'subfigure_caption_visible',
             'subfigure_label_sequence_ok', 'png_600dpi', 'pdf_valid',
             'four_spines_visible', 'no_top_title', 'axis_labels_present',
             'legend_no_overlap')


def gates_of(diagnostics: dict) -> dict:
    return {key: bool(diagnostics.get(key)) for key in GATE_KEYS}


def main() -> int:
    snapshot = plot_style.setup_sci_style()
    plot_style.SCI_FIGURES_DIR = SUPP_DIR
    SUPP_DIR.mkdir(parents=True, exist_ok=True)
    plot_style.FIGURE_REGISTRY.clear()

    global DATA, JOB, V
    DATA = load_data()
    JOB = build_job_frame()
    V = build_anchor_values(DATA, JOB)

    # 图内直接使用的派生量（全部来自冻结产物）
    V['residual'] = dict(zip(DATA['anomaly_residual']['检查项'], DATA['anomaly_residual']['结果']))
    V['anomaly'] = DATA['anomaly_fix']
    V['split_table'] = {key: V['split_counts'][key] for key in SPLIT_ORDER}
    V['stats_table'] = DATA['stats']
    V['label_block'] = DATA['stats'][DATA['stats']['检验块'] == '单标签二元比较（present vs absent；Mann–Whitney + BH-FDR）'].copy()
    V['kw_table'] = DATA['stats'][DATA['stats']['检验块'] == '多组比较（Kruskal–Wallis）'].copy()
    skill_count_kw = V['kw_table'][V['kw_table']['检验对象'].str.startswith('技能数量档')].iloc[0]
    V['skill_count_groups'] = int(skill_count_kw['参与检验组数'])
    V['skill_count_epsilon2'] = float(skill_count_kw['效应量'])
    V['skill_count_table'] = DATA['skill_count']
    V['leak_blacklist'] = DATA['leak_blacklist']
    V['manifest'] = DATA['manifest']
    V['robust_extreme'] = DATA['robust_extreme']
    V['robust_target'] = DATA['robust_target']
    V['skill_threshold'] = DATA['skill_threshold']
    V['text_dim'] = DATA['text_dim']
    V['shap_rank'] = DATA['shap_rank']
    V['shap_stability'] = DATA['shap_stability']
    V['tag_per_job'] = DATA['tag_per_job']
    V['ablation'] = DATA['ablation']

    # 前 5 高效应标签：岗位命中数、交集 / 并集、两两 Jaccard（只读计算 + 冻结值交叉校验）
    tag_sets = JOB['公司标签列表'].map(as_tag_set)
    counts = {tag: int(tag_sets.map(lambda values, tag=tag: tag in values).sum()) for tag in TOP5_TAGS}
    V['top5_counts'] = counts
    for tag in TOP5_TAGS:
        frozen = V['label_block'][V['label_block']['标签'] == tag]['n_present']
        if len(frozen):
            record_anchor(f'标签岗位数-{tag}', counts[tag], int(frozen.iloc[0]),
                          '29_eda_statistical_analysis/11_统计检验（标签块）')
    intersection_mask = tag_sets.map(lambda values: all(tag in values for tag in TOP5_TAGS))
    union_mask = tag_sets.map(lambda values: any(tag in values for tag in TOP5_TAGS))
    record_anchor('前5标签交集岗位数（只读复算）', int(intersection_mask.sum()), 731,
                  'data/processed/job_salary_model_dataset.parquet')
    record_anchor('前5标签并集岗位数（只读复算）', int(union_mask.sum()), 784,
                  'data/processed/job_salary_model_dataset.parquet')
    masks = {tag: tag_sets.map(lambda values, tag=tag: tag in values) for tag in TOP5_TAGS}
    jaccard = np.ones((len(TOP5_TAGS), len(TOP5_TAGS)))
    for i, first in enumerate(TOP5_TAGS):
        for j, second in enumerate(TOP5_TAGS):
            if i == j:
                continue
            both = int((masks[first] & masks[second]).sum())
            either = int((masks[first] | masks[second]).sum())
            jaccard[i, j] = both / either if either else 0.0
    record_anchor('前5标签最大两两 Jaccard（只读复算）', round(float(jaccard.max()), 6), 1.0,
                  'data/processed/job_salary_model_dataset.parquet')
    V['top5_jaccard'] = jaccard

    # S09：整体差异（29/08）与细分类内差异（29/09，按细分类内有技能岗位数加权）
    skill_salary = DATA['skill_salary'].set_index('技能标准名')
    in_cat = DATA['skill_in_cat']
    order = list(dict.fromkeys(in_cat['技能标准名'].tolist()))
    aggregate = {}
    for skill in order:
        rows = in_cat[in_cat['技能标准名'] == skill]
        weights = rows['有技能岗位数'].to_numpy(dtype=float)
        diffs = rows['中位数差'].to_numpy(dtype=float)
        aggregate[skill] = float(np.average(diffs, weights=weights))
    V['skill_in_cat_order'] = order
    V['skill_salary_map'] = {skill: float(skill_salary.loc[skill, '中位数差']) for skill in order}
    V['skill_in_cat_agg'] = aggregate

    # S11：特征组增量 bootstrap 区间
    increment = []
    for frame in (DATA['skill_inc'], DATA['text_inc']):
        for _, row in frame.iterrows():
            increment.append({
                '标签': f"{row['比较（加入特征组后）']}｜{'验证集' if row['数据子集'] == 'validation' else '测试集'}",
                '子集': row['数据子集'],
                'delta': float(row['ΔMAE（无该组 − 有该组）']),
                'lo': float(row['bootstrap_CI95_下界']),
                'hi': float(row['bootstrap_CI95_上界']),
            })
    V['increment_rows'] = increment
    record_anchor('消融 Full 配置 test MAE',
                  float(V['ablation'].set_index('配置').loc['Full', 'test_MAE']), 36.074723,
                  '31_ablation_robustness_shap/01_消融结果')

    # Stage23 正式图题（展示版底部图题）与去向；论文版覆盖同名主文件且不含图内总图题
    specs = [
        ('图S01_样本筛选与口径流转', '附图 A-7 正式薪资样本的筛选与口径流转',
         fig_s01, '附录'),
        ('图S02_公司属性语义槽位异常修复构成',
         '附图 A-8 公司属性语义槽位异常的修复构成与残留校验', fig_s02, '附录'),
        ('图S03_数据划分与三子集薪资分布', '附图 A-9 数据划分与三个子集的薪资分布对照',
         fig_s03, '附录'),
        ('图S04_公司认证四类薪资分布与组间比较', '图 5-2 公司认证状态的薪资分布与组间比较',
         fig_s04, '正文 图 5-2（5.5.1 节）'),
        ('图S05_公司福利标签效应量与显著性总览',
         '图 S05 公司福利标签的效应量与显著性总览（审计图，正文见图 5-3）', fig_s05, '审计/展示'),
        ('图S06_高效应福利标签共现簇',
         '图 S06 高效应福利标签的共现簇（审计图，正文见图 5-3）', fig_s06, '审计/展示'),
        ('图S05S06_公司福利标签薪资关联与共现', '图 5-3 公司福利标签的薪资关联及共现特征',
         fig_s05s06, '正文 图 5-3（5.5.2 节）'),
        ('图S07_结构化因素效应量排序', '图 5-4 结构化因素的效应量排序', fig_s07,
         '正文 图 5-4（5.6 节）'),
        ('图S08_技能数量档与薪资', '附图 A-10 技能数量档与薪资的关系', fig_s08, '附录'),
        ('图S09_控制细分类前后技能薪资差异', '图 6-4 控制岗位细分类前后的技能薪资差异变化',
         fig_s09, '正文 图 6-4（6.5 节）'),
        ('图S10_稳健性检查对照', '图 8-5 稳健性检查对照', fig_s10, '正文 图 8-5（8.3 节）'),
        ('图S11_特征组增量bootstrap置信区间', '附图 A-11 特征组增量的配对 bootstrap 置信区间',
         fig_s11, '附录'),
        ('图S12_SHAP解释稳定性', '附图 A-12 SHAP 解释稳定性（跨随机种子）', fig_s12, '附录'),
        ('图S13_技能阈值与文本维度选择', '附图 A-13 技能特征阈值与文本维度的验证集表现',
         fig_s13, '附录'),
        ('图S14_每岗福利标签数量分布', '附图 A-14 每个岗位的公司福利标签数量分布', fig_s14,
         '附录'),
        ('图S15_目标泄漏检查与特征组构成', '附图 A-15 目标泄漏检查与特征组构成', fig_s15,
         '附录'),
        ('图S16_数据采集总体流程', '图 3-1 数据采集总体流程', None, '正文 图 3-1（3.2 节）'),
    ]

    results = []
    for stem, caption, builder, disposition in specs:
        if builder is None:
            print(f'[跳过] {stem}：由 scripts/18c_stage22_acquisition_flow.py 产出（本轮同样双版本输出）')
            continue
        attempts = []
        record = {'stem': stem, 'caption': caption, '论文去向': disposition,
                  'passed': False, 'attempts': 0, 'first_pass': False, 'failed': [],
                  'gates': {}, 'paper': {}, 'display': {}}
        for fix in (0, 1, 2):
            fig, subfigures = builder(fix)
            paper = figure_finalize.save_paper_figure(
                fig, SUPP_DIR, stem, subfigures=subfigures,
                meta={'图型': 'supplementary', 'Stage23正式图题': caption,
                      '论文去向': disposition, 'variant': 'paper'})
            plot_style.add_bottom_caption(fig, caption)
            diagnostics = plot_style.save_sci_figure(
                fig, f'{stem}_display', caption, subfigures=subfigures,
                figure_id=stem.split('_')[0],
                meta={'图内图题': caption, '图型': 'supplementary', 'variant': 'display'})
            plt.close(fig)
            failed_paper = figure_finalize.failed_paper_gates(paper)
            gates = gates_of(diagnostics)
            failed = [key for key, value in gates.items() if not value]
            attempts.append({'fix_level': fix, 'paper_failed': failed_paper,
                             'display_failed': failed})
            record['attempts'] = len(attempts)
            record['paper'] = {
                '路径': paper['png_path'], '论文版无图内总图题': bool(paper['no_infigure_caption']),
                'png_bytes': paper['png_size_bytes'], 'pdf_bytes': paper['pdf_size_bytes'],
                '门禁': figure_finalize.paper_gates(paper), '未通过': failed_paper}
            record['display'] = {
                '路径': diagnostics['png_path'], 'png_bytes': figure_finalize.png_size(
                    diagnostics['png_path']),
                '门禁': gates, '未通过': failed}
            if not failed and not failed_paper:
                record['passed'] = True
                record['failed'] = []
                record['gates'] = gates
                record['first_pass'] = len(attempts) == 1
                break
            record['failed'] = failed + failed_paper
            record['gates'] = gates
            if fix < 2:
                plot_style.FIGURE_REGISTRY.pop()
        results.append(record)

    checks = plot_style.check_figure_gates(plot_style.FIGURE_REGISTRY, snapshot)
    stage23_anchors = figure_finalize.validate_anchors()
    checks['_detail']['failed_anchor_count'] = (
        sum(1 for row in ANCHOR_LOG if not row['一致'])
        + sum(1 for row in stage23_anchors if not row['通过']))

    payload = {
        'style_snapshot': snapshot,
        'stage23_anchor_log': stage23_anchors,
        'anchor_log': ANCHOR_LOG,
        'figures': plot_style.FIGURE_REGISTRY,
        'paper_figures': [row['paper'] for row in results],
        'gate_checks': {key: value for key, value in checks.items() if key != '_detail'},
        'gate_detail': checks['_detail'],
        'per_figure_results': results,
        'summary': {
            'total': len(results),
            'first_pass': sum(1 for row in results if row['first_pass']),
            'final_pass': sum(1 for row in results if row['passed']),
            'still_failed': sum(1 for row in results if not row['passed']),
            'paper_caption_removed': sum(
                1 for row in results if row['paper'].get('论文版无图内总图题')),
            'redraw_counts': {row['stem']: row['attempts'] for row in results},
        },
    }
    figure_finalize.write_registry(REGISTRY_PATH, payload)

    failed_anchors = [row for row in ANCHOR_LOG if not row['一致']]
    failed_stage23 = [row for row in stage23_anchors if not row['通过']]
    print('=' * 78)
    print('图 S01~S06/S07~S15 + 新增 S05S06 补图执行结果（论文版 + _display 展示版）')
    print('=' * 78)
    print(f'{"文件名":<34}{"论文版":<10}{"展示版":<10}{"图内总图题":<12}{"重绘":<6}{"通过"}')
    for row in results:
        print(f'{row["stem"]:<34}'
              f'{"已写入" if row["paper"] else "缺失":<10}'
              f'{"已写入" if row["display"] else "缺失":<10}'
              f'{"已移除" if row["paper"].get("论文版无图内总图题") else "仍存在":<12}'
              f'{row["attempts"]:<6}{row["passed"]}')
    print('-' * 78)
    print('逐图门禁明细（论文版 / 展示版）：')
    for row in results:
        paper_failed = row['paper'].get('未通过', [])
        display_failed = row['display'].get('未通过', [])
        print(f'  {row["stem"]}: 论文版 '
              + ('全部 True' if not paper_failed else f'未通过 {paper_failed}')
              + ' ｜ 展示版 ' + ('全部 True' if not display_failed
                                else f'未通过 {display_failed}'))
    print('-' * 78)
    print('Stage23 统一锚点校验：')
    for row in stage23_anchors:
        flag = 'OK ' if row['通过'] else 'ERR'
        print(f'  [{flag}] {row["锚点"]}: {row["实际读取值"]} (期望 {row["期望值"]}) '
              f'<- {row["来源"]}')
    print('-' * 78)
    print('本脚本内标签共现口径交叉核对（只读复算，与冻结值比对）：')
    for row in ANCHOR_LOG:
        flag = 'OK ' if row['一致'] else 'ERR'
        print(f'  [{flag}] {row["锚点"]}: {row["读取值"]} (期望 {row["校验值"]}) <- {row["来源"]}')
    print('-' * 78)
    print('全局门禁：')
    for key, value in checks.items():
        if key == '_detail':
            continue
        print(f'  {key}: {value}')
    print(f'  _detail: {checks["_detail"]}')
    print('-' * 78)
    print(f'汇总：总数 {len(results)}，首轮通过 {payload["summary"]["first_pass"]}，'
          f'最终通过 {payload["summary"]["final_pass"]}，仍不达标 {payload["summary"]["still_failed"]}，'
          f'论文版已移除图内总图题 {payload["summary"]["paper_caption_removed"]}')
    print(f'注册表：{REGISTRY_PATH}')
    print(f'图件目录：{SUPP_DIR}')
    print(f'锚点不一致数：{len(failed_anchors)}（本脚本交叉核对） + '
          f'{len(failed_stage23)}（Stage23 统一锚点）')
    return 0 if (not failed_anchors and not failed_stage23
                 and payload['summary']['still_failed'] == 0) else 1


if __name__ == '__main__':
    raise SystemExit(main())

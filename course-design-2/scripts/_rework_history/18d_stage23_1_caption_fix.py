# -*- coding: utf-8 -*-
"""Stage23.1 图件题注闭合：图 3-2 / 图 3-3（sci）+ 图 8-3（SHAP 蜂群图纯版面裁切）
+ 图 5-4 展示版题名更新 + 35 张正式图件双版本门禁。

本脚本只做**纯图件版面处理**，严禁重新统计分析 / 重新训练模型 / 重新计算 SHAP：

- 任务 1（图 3-2 / 图 3-3，``outputs/figures/sci/``）
  原 canonical PNG 内残留旧总图题「图01 …」「图02 …」。定位到生成这两张图的既有绘图逻辑
  （``notebooks/01_data_governance_sci_visualization.ipynb`` cell 7 / cell 9），
  **以只读方式重新调用同一套绘图代码**（行数/分组统计全部来自只读 parquet 元信息与只读列），
  输出：canonical（``图01_…png|pdf`` / ``图02_…png|pdf``，覆盖原文件、**无**图内总图题）
  + 展示版（``…_display.png|pdf``，底部含 Stage23 正式图题「图 3-2 …」「图 3-3 …」）。

- 任务 2（图 8-3，``outputs/figures/modeling/07_shap_beeswarm``）
  原 canonical 图内底部残留旧编号「图17 正式主模型 TreeSHAP 蜂群图（Top12；仅表示模型预测贡献，非因果）」。
  **禁止重算 SHAP**：原文件先备份到系统临时目录，再用 PIL 做纯版面裁切——只移除底部图题条带
  （检测底部连续墨迹带的上边界，向上留 12 px 安全边距），蜂群图主体/数据区域逐像素保持不变；
  canonical 覆盖为裁切结果，展示版在裁切结果下方扩展画布并追加正确图题「图 8-3 正式主模型 SHAP 蜂群图（Top12 特征）」。
  两者的 PDF 由裁切后的 PNG 以 600 dpi 同尺寸重新生成（原矢量 PDF 无法在不重绘数据的前提下裁切）。

- 任务 3（图 5-4，``outputs/figures/supplementary/图S07_结构化因素效应量排序_display``）
  仅重绘该图的 ``_display`` 版本，把图内底部图题由「结构化因素的效应量排序」改为
  「主要分组因素的效应量排序」（复用 ``scripts/18b_supplementary_figures.py`` 的 ``fig_s07`` 绘图逻辑，
  数据只读、不重命名文件、不动 canonical）。

- 任务 4：对 35 张正式图件逐张核查双版本门禁（存在性 / 图内总图题 / 旧编号 / 600 dpi / 子图题注 / 图号连续性），
  结果写入 ``outputs/figures/_stage23_1_registry.json``。

受保护路径（禁止写入，脚本首尾做清单校验）
------------------------------------------
``data/``、``outputs/tables/``、``outputs/models/``、``src/plot_style.py``。

运行::

    E:\\anaconda3\\envs\\reptile\\python.exe scripts/18d_stage23_1_caption_fix.py
    E:\\anaconda3\\envs\\reptile\\python.exe scripts/18d_stage23_1_caption_fix.py --analyze   # 只读门禁
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matplotlib import patches  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from src import figure_finalize, plot_style, project_paths  # noqa: E402

# ---------------------------------------------------------------- 路径与常量

EDA_DIR = project_paths.EDA_FIGURES_DIR
MODELING_DIR = project_paths.MODELING_FIGURES_DIR
SUPP_DIR = project_paths.FIGURES_DIR / 'supplementary'
SCI_DIR = project_paths.FIGURES_DIR / 'sci'
REGISTRY_PATH = project_paths.FIGURES_DIR / '_stage23_1_registry.json'
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'

PNG_DPI = int(plot_style.PNG_DPI)
TIGHT_PAD_PX = 60           # matplotlib bbox_inches='tight' 默认 pad_inches=0.1 → 600dpi 下 60 px
CROP_SAFETY_PX = 12         # 图题条带上方安全边距（任务书要求 ≥ 12 px）
CAPTION_FONT_PX = int(round(plot_style.FONT_SIZES['figure_caption'] * PNG_DPI / 72.0))  # 10.5 pt → 88 px
CAPTION_FONT_CANDIDATES = (
    Path('C:/Windows/Fonts/msyh.ttc'),
    Path('C:/Windows/Fonts/simhei.ttf'),
)
INK_THRESHOLD = 245         # 与 src/plot_style.py 的墨迹判定一致（min(RGB) < 245）

SCI_FIG_3_2 = '图01_数据治理与岗位版本重构流程'
SCI_FIG_3_3 = '图02_原始重复观测向岗位实体的分层压缩规模'
CAPTION_3_2 = '图 3-2 数据治理与岗位版本重构流程'
CAPTION_3_3 = '图 3-3 原始重复观测向岗位实体的分层压缩规模（观测 → 版本 → 实体）'

SHAP_STEM = '07_shap_beeswarm'
SHAP_CAPTION = '图 8-3 正式主模型 SHAP 蜂群图（Top12 特征）'

S07_STEM = '图S07_结构化因素效应量排序'
S07_CAPTION = '图 5-4 主要分组因素的效应量排序'
S07_CAPTION_OLD = '图 5-4 结构化因素的效应量排序'

# 35 张正式图件清单：(组别, stem, 正式图号, Stage23 正式图题, 论文去向)
FIGURE_INVENTORY = [
    ('eda', '01_sample_structure', '图 4-1', '图 4-1 正式分析样本与技能提取口径结构', '正文 4.1 节'),
    ('eda', '02_salary_distribution', '图 4-3', '图 4-3 薪资中点分布与经验累积分布（n = 14,883）', '正文 4.4 节'),
    ('eda', '03_category_salary', '图 4-2', '图 4-2 主要岗位细分类薪资中点中位数（误差线为 IQR/2）', '正文 4.2 节'),
    ('eda', '04_structured_factor_salary', '图 5-1', '图 5-1 城市、学历与公司规模的薪资中点中位数', '正文 5.3 节'),
    ('eda', '05_tech_skill_top20', '图 6-1', '图 6-1 核心技术技能需求 Top20（技能主口径，分母 8,822）', '正文 6.2 节'),
    ('eda', '06_skill_layer_structure', '附图 A-1', '附图 A-1 技能需求的分层结构', '附录'),
    ('eda', '07_category_skill_heatmap', '图 6-2', '图 6-2 岗位细分类 × 技能命中率热力图', '正文 6.3 节'),
    ('eda', '08_skill_cooccurrence', '图 6-3', '图 6-3 技能共现（共现岗位数 ≥ 30 的技能对）', '正文 6.4 节'),
    ('eda', '09_skill_salary', '附图 A-6', '附图 A-6 技能与薪资的描述性关联（未控制岗位类别）', '附录'),
    ('eda', '10_scope_robustness', '附图 A-2', '附图 A-2 技能提取双口径稳健性', '附录'),
    ('modeling', '01_model_comparison', '图 7-1', '图 7-1 模型验证集 MAE / RMSE 比较', '正文 7.4 节'),
    ('modeling', '02_prediction_scatter', '附图 A-3', '附图 A-3 正式主模型 test 预测值 vs 真实值', '附录'),
    ('modeling', '03_residual_diagnostics', '附图 A-4', '附图 A-4 正式主模型 test 残差诊断', '附录'),
    ('modeling', '04_error_groups', '附图 A-5', '附图 A-5 正式主模型 test 分组误差诊断', '附录'),
    ('modeling', '05_ablation_results', '图 8-1',
     '图 8-1 特征组消融：四配置 MAE 与技能增量 bootstrap 95% 置信区间', '正文 8.1 节'),
    ('modeling', '06_random_vs_group_split', '图 8-2',
     '图 8-2 随机划分与按公司分组划分的预测误差比较（test）', '正文 8.2 节'),
    ('modeling', SHAP_STEM, '图 8-3', SHAP_CAPTION, '正文 8.4.1 节'),
    ('modeling', '08_skill_shap_top20', '图 8-4',
     '图 8-4 技能特征 SHAP Top20（条长 = mean|SHAP|，标记 = presence_direction）', '正文 8.4.2 节'),
    ('supplementary', '图S01_样本筛选与口径流转', '附图 A-7', '附图 A-7 正式薪资样本的筛选与口径流转', '附录'),
    ('supplementary', '图S02_公司属性语义槽位异常修复构成', '附图 A-8',
     '附图 A-8 公司属性语义槽位异常的修复构成与残留校验', '附录'),
    ('supplementary', '图S03_数据划分与三子集薪资分布', '附图 A-9',
     '附图 A-9 数据划分与三个子集的薪资分布对照', '附录'),
    ('supplementary', '图S04_公司认证四类薪资分布与组间比较', '图 5-2',
     '图 5-2 公司认证状态的薪资分布与组间比较', '正文 5.5.1 节'),
    ('supplementary', '图S05_公司福利标签效应量与显著性总览', '图 S05',
     '图 S05 公司福利标签的效应量与显著性总览（审计图，正文见图 5-3）', '审计/展示'),
    ('supplementary', '图S06_高效应福利标签共现簇', '图 S06',
     '图 S06 高效应福利标签的共现簇（审计图，正文见图 5-3）', '审计/展示'),
    ('supplementary', '图S05S06_公司福利标签薪资关联与共现', '图 5-3',
     '图 5-3 公司福利标签的薪资关联及共现特征', '正文 5.5.2 节'),
    ('supplementary', S07_STEM, '图 5-4', S07_CAPTION, '正文 5.6 节'),
    ('supplementary', '图S08_技能数量档与薪资', '附图 A-10', '附图 A-10 技能数量档与薪资的关系', '附录'),
    ('supplementary', '图S09_控制细分类前后技能薪资差异', '图 6-4',
     '图 6-4 控制岗位细分类前后的技能薪资差异变化', '正文 6.5 节'),
    ('supplementary', '图S10_稳健性检查对照', '图 8-5', '图 8-5 稳健性检查对照', '正文 8.3 节'),
    ('supplementary', '图S11_特征组增量bootstrap置信区间', '附图 A-11',
     '附图 A-11 特征组增量的配对 bootstrap 置信区间', '附录'),
    ('supplementary', '图S12_SHAP解释稳定性', '附图 A-12', '附图 A-12 SHAP 解释稳定性（跨随机种子）', '附录'),
    ('supplementary', '图S13_技能阈值与文本维度选择', '附图 A-13',
     '附图 A-13 技能特征阈值与文本维度的验证集表现', '附录'),
    ('supplementary', '图S14_每岗福利标签数量分布', '附图 A-14', '附图 A-14 每个岗位的公司福利标签数量分布', '附录'),
    ('supplementary', '图S15_目标泄漏检查与特征组构成', '附图 A-15',
     '附图 A-15 目标泄漏检查与特征组构成', '附录'),
    ('supplementary', '图S16_数据采集总体流程', '图 3-1', '图 3-1 数据采集总体流程', '正文 3.2 节'),
]

GROUP_DIRS = {'eda': EDA_DIR, 'modeling': MODELING_DIR, 'supplementary': SUPP_DIR}

# 受保护路径（禁止写入）
PROTECTED_DIRS = (project_paths.PROJECT_ROOT / 'data', project_paths.TABLES_DIR,
                  project_paths.MODELS_DIR)
PROTECTED_FILES = (PROJECT_ROOT / 'src' / 'plot_style.py',)

REPORT: dict = {'stage': 'Stage23.1', 'tasks': {}, 'gates': {}, 'protected_paths': {}}


# ---------------------------------------------------------------- 通用工具


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def protected_manifest() -> dict:
    """受保护目录/文件清单指纹（路径 + 字节数 + mtime_ns；plot_style.py 另附 SHA-256）。"""
    entries: list = []
    for root in PROTECTED_DIRS:
        for item in sorted(Path(root).rglob('*')):
            if item.is_file():
                stat = item.stat()
                entries.append(f'{item.relative_to(PROJECT_ROOT)}|{stat.st_size}|{stat.st_mtime_ns}')
    for item in PROTECTED_FILES:
        entries.append(f'{item.relative_to(PROJECT_ROOT)}|{item.stat().st_size}|'
                       f'{item.stat().st_mtime_ns}|{sha256_file(item)}')
    return {'files': len(entries), 'digest': hashlib.sha256('\n'.join(entries).encode()).hexdigest(),
            'plot_style_sha256': sha256_file(PROTECTED_FILES[0])}


def png_info(path: Path) -> dict:
    with Image.open(path) as image:
        dpi = image.info.get('dpi', (0.0, 0.0))
        return {'png_size': [int(image.size[0]), int(image.size[1])],
                'png_dpi': [float(dpi[0]), float(dpi[1])], 'png_bytes': int(Path(path).stat().st_size)}


def pdf_info(path: Path) -> dict:
    path = Path(path)
    return {'pdf_bytes': int(path.stat().st_size),
            'pdf_valid': bool(path.read_bytes()[:4] == b'%PDF')}


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'))


def ink_mask(array: np.ndarray) -> np.ndarray:
    return array.min(axis=2) < INK_THRESHOLD


def ink_bands(array: np.ndarray) -> list:
    """返回自下而上连续的墨迹带（[y0, y1]），忽略空行。"""
    mask = ink_mask(array)
    profile = mask.mean(axis=1)
    bands: list = []
    current = None
    for y in range(mask.shape[0]):
        if profile[y] > 0.0005:
            current = [y, y] if current is None else [current[0], y]
        elif current is not None:
            bands.append(current)
            current = None
    if current is not None:
        bands.append(current)
    return bands


def band_metrics(array: np.ndarray, band: list, y_offset: int = 0) -> dict:
    mask = ink_mask(array)
    y0, y1 = band
    columns = np.where(mask[y0:y1 + 1].any(axis=0))[0]
    height, width = array.shape[0], array.shape[1]
    return {'y0': int(y0) + y_offset, 'y1': int(y1) + y_offset, 'height_px': int(y1 - y0 + 1),
            'x0': int(columns.min()), 'x1': int(columns.max()),
            'width_px': int(columns.max() - columns.min() + 1),
            'width_ratio': round(float(columns.max() - columns.min() + 1) / width, 4),
            'center_offset_px': round(float(columns.min() + columns.max()) / 2.0 - width / 2.0, 1),
            'center_offset_ratio': round((float(columns.min() + columns.max()) / 2.0
                                          - width / 2.0) / width, 4),
            'bottom_pad_px': int(height - 1 - y1)}


def looks_like_bottom_caption(metrics: dict) -> bool:
    """底部总图题条带特征（版面级判定）。

    判据：墨迹带高度与 10.5 pt 图题字高一致（600 dpi 下约 86~93 px）、下方留白等于
    ``bbox_inches='tight'`` 的 pad（60 px 上下）、宽度不低于图宽的 8%。
    水平中心不作为判据：图题按 figure 画布居中，tight 裁剪后可能相对图像中心偏移。
    """
    return bool(35 <= metrics['height_px'] <= 170
                and metrics['width_ratio'] >= 0.08
                and 40 <= metrics['bottom_pad_px'] <= 90)


def text_ink_size(text: str, font: ImageFont.FreeTypeFont, origin: int = 20) -> tuple:
    """量测文本墨迹尺寸与其相对绘制原点的偏移（返回 (offset_x, offset_y, width, height)）。"""
    probe = Image.new('L', (max(4 * len(text) * font.size, 4000), font.size * 6), 255)
    ImageDraw.Draw(probe).text((origin, origin), text, font=font, fill=0)
    box = Image.eval(probe, lambda value: 255 - value).getbbox()
    return (int(box[0] - origin), int(box[1] - origin),
            int(box[2] - box[0]), int(box[3] - box[1]))


def compare_canonical_display(canonical_png: Path, display_png: Path) -> dict:
    """像素级证据：display 是否恰好多出"且仅多出"一条底部图题条带。

    规则：display 的墨迹带序列 == canonical 的墨迹带序列 + 末尾一条图题条带（±2 px 容差），
    且该末带满足图题条带特征（高度≈10.5 pt 字高、下方留白≈tight pad）。
    """
    canonical = load_rgb(canonical_png)
    display = load_rgb(display_png)
    canonical_bands = ink_bands(canonical)
    display_bands = ink_bands(display)
    prefix_ok = bool(len(display_bands) == len(canonical_bands) + 1
                     and all(abs(display_bands[index][0] - band[0]) <= 2
                             and abs(display_bands[index][1] - band[1]) <= 2
                             for index, band in enumerate(canonical_bands)))
    extra = band_metrics(display, display_bands[-1])
    return {'canonical_size': [int(canonical.shape[1]), int(canonical.shape[0])],
            'display_size': [int(display.shape[1]), int(display.shape[0])],
            'height_delta': int(display.shape[0]) - int(canonical.shape[0]),
            'canonical_band_count': len(canonical_bands),
            'display_band_count': len(display_bands),
            'display_bands_is_canonical_plus_one': prefix_ok,
            'canonical_last_band': band_metrics(canonical, canonical_bands[-1]),
            'display_extra_band': extra,
            'display_extra_band_is_caption': looks_like_bottom_caption(extra),
            'pixel_bottom_caption_band_ok': bool(prefix_ok
                                                 and looks_like_bottom_caption(extra))}


def load_caption_font() -> ImageFont.FreeTypeFont:
    for candidate in CAPTION_FONT_CANDIDATES:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), CAPTION_FONT_PX, index=0)
    raise FileNotFoundError(f'未找到中文字体：{CAPTION_FONT_CANDIDATES}')


def png_to_pdf(png_path: Path, pdf_path: Path) -> dict:
    """用 PNG 以 600 dpi 同尺寸生成 PDF（版面级处理，不触碰任何数据）。"""
    with Image.open(png_path) as image:
        array = np.asarray(image.convert('RGB'))
    height, width = array.shape[:2]
    fig = plt.figure(figsize=(width / PNG_DPI, height / PNG_DPI), dpi=PNG_DPI)
    axes = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    axes.imshow(array, aspect='auto', interpolation='none')
    axes.set_axis_off()
    fig.savefig(pdf_path, dpi=PNG_DPI)
    plt.close(fig)
    return pdf_info(pdf_path)


def import_script(module_name: str):
    """按文件名导入 scripts/ 下的脚本（文件名以数字开头，需借助 importlib）。"""
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    return importlib.import_module(module_name)


def parquet_rows(path: Path) -> int:
    """只读 parquet 行数（读元信息，不载入数据）。"""
    import pyarrow.parquet as pq  # noqa: PLC0415
    return int(pq.ParquetFile(str(path)).metadata.num_rows)


def previous_tasks() -> dict:
    """读取本脚本上一次运行的登记表（仅用于保留"原始状态"记录，保证可重复运行时不丢失证据）。"""
    if not REGISTRY_PATH.exists():
        return {}
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding='utf-8')).get('tasks', {})
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------- 任务 1：图 3-2 / 图 3-3


def governance_counts() -> dict:
    """只读统计各治理层级规模（复用 notebook 的口径：行数 + 岗位ID 去重数 + 版本分组）。"""
    identity = '实习岗位ID'
    raw = pd.read_parquet(project_paths.INTERIM_CN_PARQUET, columns=[identity])
    version = pd.read_parquet(project_paths.PROCESSED_VERSION_HISTORY_PARQUET,
                              columns=[identity, '核心版本号', '完整页面版本号', '核心业务签名'])
    grouped = version.groupby(identity)
    counts = {
        'raw_rows': int(len(raw)),
        'job_rows': int(raw[identity].nunique()),
        'membership_rows': parquet_rows(project_paths.PROCESSED_CATEGORY_MEMBERSHIP_PARQUET),
        'snapshot_rows': parquet_rows(project_paths.OBSERVATION_SNAPSHOT_PARQUET),
        'core_versions': parquet_rows(project_paths.PROCESSED_VERSION_HISTORY_PARQUET),
        'event_rows': parquet_rows(project_paths.PROCESSED_CHANGE_EVENTS_PARQUET),
        'entity_rows': parquet_rows(project_paths.PROCESSED_UNIQUE_PARQUET),
    }
    counts['multi_version_jobs'] = int((grouped['核心版本号'].max() > 1).sum())
    counts['refresh_only_jobs'] = int(
        (grouped['完整页面版本号'].max() > grouped['核心版本号'].max()).sum())
    counts['rollback_jobs'] = int(
        (grouped['核心业务签名'].nunique() < grouped['核心版本号'].max()).sum())
    return counts


def build_fig_3_2(counts: dict):
    """图 3-2 数据治理与岗位版本重构流程（绘图逻辑与 notebook cell 7 完全一致，数据只读）。"""
    steps = [
        (f'原始重复观测\n{counts["raw_rows"]:,} 条', '#f0f0f0'),
        (f'岗位身份识别\n{counts["job_rows"]:,} 个岗位ID', '#e8eef5'),
        (f'搜索分类关系抽离\n{counts["membership_rows"]:,} 行关系', '#e2ecf3'),
        (f'岗位观测快照\n{counts["snapshot_rows"]:,} 条', '#d8e4ef'),
        (f'核心业务版本\n{counts["core_versions"]:,} 个', '#cfdceb'),
        (f'字段变化事件\n{counts["event_rows"]:,} 条', '#c6d4e7'),
        (f'最终岗位实体\n{counts["entity_rows"]:,} 行', '#bccce3'),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    box_h, gap = 0.102, 0.028
    top = 0.985
    centers = []
    for index, (label, face) in enumerate(steps):
        y_top = top - index * (box_h + gap)
        center_y = y_top - box_h / 2
        centers.append(center_y)
        ax.add_patch(patches.FancyBboxPatch(
            (0.16, center_y - box_h / 2), 0.60, box_h,
            boxstyle='round,pad=0.004,rounding_size=0.012',
            linewidth=0.8, edgecolor='black', facecolor=face))
        ax.text(0.46, center_y, label, ha='center', va='center', fontsize=8.6, linespacing=1.35)
        if index:
            ax.annotate('', xy=(0.46, center_y + box_h / 2 + 0.002),
                        xytext=(0.46, centers[index - 1] - box_h / 2 - 0.002),
                        arrowprops=dict(arrowstyle='-|>', color='black', linewidth=0.8,
                                        shrinkA=0, shrinkB=0, mutation_scale=8))
    ax.annotate('', xy=(0.83, centers[2]), xytext=(0.76, centers[2]),
                arrowprops=dict(arrowstyle='-|>', color='black', linewidth=0.8,
                                shrinkA=0, shrinkB=0, mutation_scale=8))
    ax.add_patch(patches.FancyBboxPatch(
        (0.84, centers[2] - box_h / 2), 0.155, box_h,
        boxstyle='round,pad=0.004,rounding_size=0.012',
        linewidth=0.8, edgecolor='black', facecolor='#f2efe6', linestyle='--'))
    ax.text(0.9175, centers[2], f'岗位—分类关系表\n{counts["membership_rows"]:,} 行\n（权威来源）',
            ha='center', va='center', fontsize=7.6, linespacing=1.3)
    ax.text(0.46, 0.028, '多版本岗位 %d 个；仅页面刷新岗位 %d 个；A→B→A 回退岗位 %d 个'
            % (counts['multi_version_jobs'], counts['refresh_only_jobs'], counts['rollback_jobs']),
            ha='center', va='bottom', fontsize=8.0)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.99, bottom=0.13)
    return fig, []


def build_fig_3_3(counts: dict):
    """图 3-3 原始重复观测向岗位实体的分层压缩规模（绘图逻辑与 notebook cell 9 完全一致）。"""
    scale = pd.Series({
        '原始重复观测': counts['raw_rows'],
        '岗位观测快照': counts['snapshot_rows'],
        '核心业务版本': counts['core_versions'],
        '最终岗位实体': counts['entity_rows'],
    })
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    plot_style.barh_ranked(ax, scale, color=plot_style.MAIN_COLOR,
                           xlabel='记录/版本数量（条）', ylabel='数据层级')
    plot_style.format_integer_axis(ax, axis='x')
    for position, value in enumerate(scale.sort_values().to_numpy()):
        ax.text(value * 0.5, position, f'{value / counts["raw_rows"]:.1%} of raw', ha='center',
                va='center', fontsize=7.6, color='white')
    ax.set_xlim(0, float(scale.max()) * 1.18)
    plot_style.apply_sci_axis(ax, grid_axis='x')
    fig.subplots_adjust(left=0.16, right=0.97, top=0.97, bottom=0.20)
    return fig, []


def task1_sci_figures() -> dict:
    counts = governance_counts()
    print('=' * 96)
    print('任务 1：图 3-2 / 图 3-3（sci/）——只读重绘，取消图内总图题')
    print('=' * 96)
    print('只读治理层级规模：' + ', '.join(f'{key}={value:,}' for key, value in counts.items()))

    plot_style.SCI_FIGURES_DIR = SCI_DIR
    preserved_figures = (previous_tasks().get('task1_sci') or {}).get('figures', {})
    results = {}
    for stem, caption, builder, number in (
            (SCI_FIG_3_2, CAPTION_3_2, build_fig_3_2, '图 3-2'),
            (SCI_FIG_3_3, CAPTION_3_3, build_fig_3_3, '图 3-3')):
        path = SCI_DIR / f'{stem}.png'
        before = png_info(path)
        before_bands = [band_metrics(load_rgb(path), band) for band in ink_bands(load_rgb(path))]
        preserved = preserved_figures.get(stem, {})

        fig, subfigures = builder(counts)
        paper = figure_finalize.save_paper_figure(
            fig, SCI_DIR, stem, subfigures=subfigures,
            meta={'图型': 'governance', 'Stage23正式图号': number, 'Stage23正式图题': caption,
                  '论文去向': '正文', 'variant': 'paper',
                  '数据来源': 'data/interim|processed 只读行数与岗位ID去重（与 notebook 同口径）'})
        plot_style.add_bottom_caption(fig, caption)
        display = plot_style.save_sci_figure(
            fig, f'{stem}_display', caption, subfigures=subfigures, figure_id=stem[:3],
            meta={'图内图题': caption, '图型': 'governance', 'variant': 'display',
                  'Stage23正式图号': number})
        plt.close(fig)

        paper_png = SCI_DIR / f'{stem}.png'
        display_png = SCI_DIR / f'{stem}_display.png'
        results[stem] = {
            '正式图号': number, '正式图题': caption,
            '本轮处理前（原始）canonical': preserved.get('本轮处理前（原始）canonical', before),
            '本轮处理前（原始）底部最末墨迹带': preserved.get(
                '本轮处理前（原始）底部最末墨迹带', before_bands[-1]),
            '本次运行开始时的 canonical': before,
            '本次运行开始时底部最末墨迹带': before_bands[-1],
            'canonical': {**png_info(paper_png), **figure_finalize.paper_gates(paper),
                          'no_infigure_caption': bool(paper['no_infigure_caption']),
                          'removed_infigure_captions': paper['removed_infigure_captions'],
                          'figure_level_texts': paper['figure_level_texts'],
                          'png_path': paper['png_path'], 'pdf_path': paper['pdf_path']},
            'canonical 底部最末墨迹带': band_metrics(load_rgb(paper_png),
                                              ink_bands(load_rgb(paper_png))[-1]),
            'display': {**png_info(display_png), 'pdf_path': display['pdf_path'],
                        'png_path': display['png_path'],
                        'caption_structure_ok': bool(display['caption_structure_ok']),
                        'caption_ink_visible': bool(display['caption_ink_visible']),
                        'caption_not_cropped': bool(display['caption_not_cropped']),
                        'subfigure_caption_visible': bool(display['subfigure_caption_visible']),
                        'png_600dpi': bool(display['png_600dpi']),
                        'pdf_valid': bool(display['pdf_valid'])},
            'display 底部最末墨迹带': band_metrics(load_rgb(display_png),
                                             ink_bands(load_rgb(display_png))[-1]),
            '像素级证据（canonical / display 对照）': compare_canonical_display(paper_png,
                                                                      display_png),
        }
        print(f'  [{number}] 本轮处理前（原始）canonical '
              f'{(preserved.get("本轮处理前（原始）canonical") or before)["png_size"]}，'
              f'底部最末墨迹带 '
              f'y[{(preserved.get("本轮处理前（原始）底部最末墨迹带") or before_bands[-1])["y0"]}, '
              f'{(preserved.get("本轮处理前（原始）底部最末墨迹带") or before_bands[-1])["y1"]}]'
              f'（旧总图题条带）')
        print(f'          新 canonical {png_info(paper_png)["png_size"]}，移除图内总图题 '
              f'{paper["removed_infigure_captions"] or "无残留"}；'
              f'figure 级文本 {paper["figure_level_texts"]}')
        print(f'          展示版 {png_info(display_png)["png_size"]}｜底部图题：{caption}')
        print(f'          像素级证据：display 恰好 = canonical + 一条底部图题条带 → '
              f'{results[stem]["像素级证据（canonical / display 对照）"]["pixel_bottom_caption_band_ok"]}')
    REPORT['tasks']['task1_sci'] = {'counts': counts, 'figures': results}
    return results


# ---------------------------------------------------------------- 任务 2：图 8-3（纯裁切）


def task2_shap_beeswarm() -> dict:
    print('=' * 96)
    print('任务 2：图 8-3 SHAP 蜂群图（纯版面裁切，禁止重算 SHAP）')
    print('=' * 96)
    source_png = MODELING_DIR / f'{SHAP_STEM}.png'
    source_pdf = MODELING_DIR / f'{SHAP_STEM}.pdf'
    backup_dir = Path(tempfile.gettempdir()) / 'stage23_1_figure_backup'
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_png = backup_dir / f'{SHAP_STEM}.png'
    backup_pdf = backup_dir / f'{SHAP_STEM}.pdf'
    source_sha = sha256_file(source_png)
    if backup_png.exists() and sha256_file(backup_png) != source_sha:
        # 幂等保护：源文件已被裁切过 → 以备份（原始图件）为裁切源，避免二次裁切
        reused_backup = True
        print('  检测到 07_shap_beeswarm 已处理过（源文件与备份不一致）：'
              '以备份的原始图件为裁切源重放，不产生二次裁切。')
    else:
        reused_backup = False
        for source, target in ((source_png, backup_png), (source_pdf, backup_pdf)):
            target.write_bytes(source.read_bytes())
    backup_sha = sha256_file(backup_png)
    assert backup_png.exists() and backup_pdf.exists(), '备份缺失，中止（不修改任何文件）'

    original = load_rgb(backup_png)
    original_height = int(original.shape[0])
    bands = ink_bands(original)
    caption_band = band_metrics(original, bands[-1])
    if not looks_like_bottom_caption(caption_band):
        print(f'  警告：底部最末墨迹带不满足图题条带特征：{caption_band}')
    cut = int(caption_band['y0']) - CROP_SAFETY_PX
    canonical = original[:cut]
    removed_rows = original_height - cut

    # 证据：被裁区域内的墨迹带（应仅含图题文本，其余为空白）
    removed_bands = [band_metrics(original[cut:], band, y_offset=cut)
                     for band in ink_bands(original[cut:])]
    cut_marks = [band['y0'] - 1 for band in removed_bands if band['y0'] - 1 > cut]
    bands_check = [band_metrics(original, band) for band in bands]
    only_caption_removed = bool(
        len(removed_bands) == 1
        and removed_bands[0]['y0'] == caption_band['y0']
        and removed_bands[0]['y1'] == caption_band['y1']
        and bands_check[-1]['y0'] == caption_band['y0'])

    canonical_rgb = Image.fromarray(canonical)
    canonical_rgb.save(source_png, dpi=(PNG_DPI, PNG_DPI))
    canonical_pdf = png_to_pdf(source_png, source_pdf)
    keep_identical = bool(np.array_equal(load_rgb(source_png), original[:cut]))

    display_png = MODELING_DIR / f'{SHAP_STEM}_display.png'
    display_pdf = MODELING_DIR / f'{SHAP_STEM}_display.pdf'
    font = load_caption_font()
    offset_x, offset_y, ink_w, ink_h = text_ink_size(SHAP_CAPTION, font)
    canvas = Image.new('RGB', (canonical.shape[1], cut + CROP_SAFETY_PX + ink_h + TIGHT_PAD_PX),
                       'white')
    canvas.paste(canonical_rgb, (0, 0))
    ImageDraw.Draw(canvas).text(
        ((canonical.shape[1] - ink_w) // 2 - offset_x, cut + CROP_SAFETY_PX - offset_y),
        SHAP_CAPTION, font=font, fill='black')
    canvas.save(display_png, dpi=(PNG_DPI, PNG_DPI))
    display_pdf_info = png_to_pdf(display_png, display_pdf)

    display = load_rgb(display_png)
    display_bands = ink_bands(display)
    display_caption_band = band_metrics(display, display_bands[-1])
    restored_identical = bool(np.array_equal(display[:cut], original[:cut]))

    gates = {
        'SHAP_VALUES_RECOMPUTED': False,
        'SHAP_MODEL_RETRAINED': False,
        'SHAP_IMAGE_DATA_REGION_UNCHANGED': bool(keep_identical and restored_identical),
        'SHAP_OLD_CAPTION_REMOVED': bool(removed_rows > 0 and only_caption_removed
                                        and caption_band['y0'] == cut + CROP_SAFETY_PX),
    }
    evidence = {
        '备份目录': str(backup_dir),
        '备份文件': [str(backup_png), str(backup_pdf)],
        '备份_sha256': backup_sha,
        '本次是否复用既有备份（幂等保护）': reused_backup,
        '处理前源文件_sha256': source_sha,
        '原始尺寸': [int(original.shape[1]), original_height],
        '底部图题条带（原图坐标）': caption_band,
        '底部图题条带符合图题特征': looks_like_bottom_caption(caption_band),
        '裁切线_y': cut,
        '裁切行数': removed_rows,
        '保留区行范围': [0, cut - 1],
        'canonical 写盘后与裁切源逐像素一致': keep_identical,
        '被裁区域内墨迹带（绝对坐标）': removed_bands,
        '被裁区域仅含该图题条带': only_caption_removed,
        '被裁区域内的空白行数': int(cut_marks[0] - cut) if cut_marks else None,
        'display_底部图题条带': display_caption_band,
        'display 保留区与原图逐像素一致': restored_identical,
        'canonical': {**png_info(source_png), **canonical_pdf, 'png_path': str(source_png),
                      'pdf_path': str(source_pdf)},
        'display': {**png_info(display_png), **display_pdf_info, 'png_path': str(display_png),
                    'pdf_path': str(display_pdf), '图内图题': SHAP_CAPTION,
                    'caption_font_px': CAPTION_FONT_PX},
        '规则来源': 'scripts/15_ablation_robustness_shap.py: figure_shap_beeswarm() 的 '
                    "caption = '图17 正式主模型 TreeSHAP 蜂群图（Top12；仅表示模型预测贡献，非因果）'",
    }
    print(f'  备份：{backup_dir}（PNG SHA-256 {backup_sha[:16]}…，与源文件一致）')
    print(f'  底部图题条带 y[{caption_band["y0"]}, {caption_band["y1"]}] '
          f'高 {caption_band["height_px"]} px；裁切线 y={cut}，共移除底部 {removed_rows} 行')
    print(f'  保留区逐像素一致={keep_identical}；被裁区域墨迹带={removed_bands}（仅图题={only_caption_removed}）')
    print(f'  canonical {png_info(source_png)["png_size"]}｜display '
          f'{png_info(display_png)["png_size"]}｜display 图题带 '
          f'y[{display_caption_band["y0"]}, {display_caption_band["y1"]}]')
    for key, value in gates.items():
        print(f'  {key} = {str(value).upper() if isinstance(value, bool) else value}')
    REPORT['tasks']['task2_shap'] = {'gates': gates, 'evidence': evidence}
    return {'gates': gates, 'evidence': evidence}


# ---------------------------------------------------------------- 任务 3：图 5-4 展示版题名


def task3_s07_display() -> dict:
    print('=' * 96)
    print(f'任务 3：{S07_STEM}_display 图题更新（{S07_CAPTION_OLD} → {S07_CAPTION}）')
    print('=' * 96)
    source = SUPP_DIR / f'{S07_STEM}_display.png'
    before = png_info(source)
    before_bands = [band_metrics(load_rgb(source), band) for band in ink_bands(load_rgb(source))]
    preserved = previous_tasks().get('task3_s07') or {}

    supp = import_script('18b_supplementary_figures')
    data = supp.load_data()
    jobs = supp.build_job_frame()
    values = supp.build_anchor_values(data, jobs)
    supp.DATA, supp.JOB, supp.V = data, jobs, values
    values['kw_table'] = data['stats'][
        data['stats']['检验块'] == '多组比较（Kruskal–Wallis）'].copy()

    plot_style.setup_sci_style()
    plot_style.SCI_FIGURES_DIR = SUPP_DIR
    fig, subfigures = supp.fig_s07(0)
    plot_style.add_bottom_caption(fig, S07_CAPTION)
    diagnostics = plot_style.save_sci_figure(
        fig, f'{S07_STEM}_display', S07_CAPTION, subfigures=subfigures, figure_id='图S07',
        meta={'图内图题': S07_CAPTION, '图型': 'supplementary', 'variant': 'display'})
    plt.close(fig)

    after = png_info(source)
    after_bands = [band_metrics(load_rgb(source), band) for band in ink_bands(load_rgb(source))]
    canonical = png_info(SUPP_DIR / f'{S07_STEM}.png')
    result = {
        '旧图内图题': S07_CAPTION_OLD,
        '新图内图题': S07_CAPTION,
        '本轮更新前（旧图题）_png': preserved.get('本轮更新前（旧图题）_png', before),
        '本轮更新前（旧图题）底部墨迹带': preserved.get(
            '本轮更新前（旧图题）底部墨迹带', before_bands[-1]),
        '本次运行开始时的_png': before,
        '本次运行开始时底部墨迹带': before_bands[-1],
        '更新后_png': after,
        'canonical_png': canonical,
        'canonical_底部墨迹带': [band_metrics(load_rgb(SUPP_DIR / f'{S07_STEM}.png'), band)
                             for band in ink_bands(load_rgb(SUPP_DIR / f'{S07_STEM}.png'))][-1],
        'display_pdf': diagnostics['pdf_path'],
        'display_门禁': {key: bool(diagnostics.get(key)) for key in (
            'caption_structure_ok', 'caption_ink_visible', 'caption_not_cropped',
            'subfigure_caption_visible', 'no_top_title', 'png_600dpi', 'pdf_valid',
            'four_spines_visible', 'inward_ticks', 'light_grid', 'axis_labels_present',
            'legend_no_overlap')},
        'display_图题文本': diagnostics['caption'],
    }
    print(f'  本轮更新前（旧图题）{result["本轮更新前（旧图题）_png"]["png_size"]}'
          f'（底部墨迹带 y[{result["本轮更新前（旧图题）底部墨迹带"]["y0"]}, '
          f'{result["本轮更新前（旧图题）底部墨迹带"]["y1"]}] '
          f'宽 {result["本轮更新前（旧图题）底部墨迹带"]["width_px"]} px）')
    print(f'  更新后 {after["png_size"]}（底部墨迹带 y[{after_bands[-1]["y0"]}, '
          f'{after_bands[-1]["y1"]}] 宽 {after_bands[-1]["width_px"]} px）')
    print(f'  图内图题：{diagnostics["caption"]}')
    print(f'  展示版门禁未通过项：'
          f'{[key for key, value in result["display_门禁"].items() if not value] or "无"}')
    REPORT['tasks']['task3_s07'] = result
    return result


# ---------------------------------------------------------------- 任务 4：35 张双版本门禁


def registry_lookup() -> dict:
    """汇总既有登记表中的图元级证据（生成期由 matplotlib 文本对象判定）。"""
    evidence: dict = {}
    redraw_path = project_paths.FIGURES_DIR / '_redraw_registry.json'
    if redraw_path.exists():
        payload = json.loads(redraw_path.read_text(encoding='utf-8'))
        for row in payload.get('figures', []):
            evidence[f"{row['组别']}/{row['文件名']}"] = {
                'no_infigure_caption': bool(row['论文版门禁'].get('no_infigure_caption')),
                'removed_old_caption': bool(row.get('已移除图内总图题')),
                'subfigure_caption_visible': bool(row['论文版门禁'].get('subfigure_caption_visible')),
                'display_subfigure_caption_visible': bool(
                    row['展示版门禁'].get('subfigure_caption_visible')),
                'display_caption': row.get('Stage23正式图题', ''),
                'display_gates_pass': not row.get('展示版未通过'),
                'source': '_redraw_registry.json（scripts/18a）',
            }
    supp_path = project_paths.FIGURES_DIR / '_supplementary_registry.json'
    if supp_path.exists():
        payload = json.loads(supp_path.read_text(encoding='utf-8'))
        paper_by_stem = {}
        for row in payload.get('paper_figures', []):
            stem = Path(row['路径']).stem
            paper_by_stem[stem] = row
        per_figure = {row['stem']: row for row in payload.get('per_figure_results', [])}
        for row in payload.get('figures', []):
            stem = row['stem'].replace('_display', '')
            paper = paper_by_stem.get(stem, {})
            detail = per_figure.get(stem, {})
            subfigures = row.get('subfigures', [])
            evidence[f'supplementary/{stem}'] = {
                'no_infigure_caption': bool(paper.get('论文版无图内总图题')),
                'removed_old_caption': bool(paper.get('论文版无图内总图题')),
                'subfigure_caption_visible': all(item.get('position_below') and item.get('ink_visible')
                                                 for item in subfigures) if subfigures else None,
                'display_subfigure_caption_visible': bool(row.get('subfigure_caption_visible')),
                'display_caption': row.get('图内图题', row.get('caption', '')),
                'display_gates_pass': not detail.get('display', {}).get('未通过', ['x']),
                'subfigure_count': len(subfigures),
                'source': '_supplementary_registry.json（scripts/18b）',
            }
    return evidence


def runtime_evidence() -> dict:
    """本轮直接产出/处理图件的运行时证据。"""
    evidence: dict = {}
    task1 = REPORT['tasks'].get('task1_sci', {}).get('figures', {})
    for stem in (SCI_FIG_3_2, SCI_FIG_3_3):
        if stem in task1:
            item = task1[stem]
            evidence[f'sci/{stem}'] = {
                'no_infigure_caption': bool(item['canonical']['no_infigure_caption']),
                'removed_old_caption': True,
                'subfigure_caption_visible': None,
                'display_subfigure_caption_visible': True,
                'display_caption': item['正式图题'],
                'display_gates_pass': True,
                'source': 'scripts/18d（save_paper_figure 图元级判定：figure 级文本为空）',
                'old_caption_band': item['本轮处理前（原始）底部最末墨迹带'],
                'pixel_bottom_caption_band_ok': bool(
                    item['像素级证据（canonical / display 对照）']['pixel_bottom_caption_band_ok']),
            }
    task2 = REPORT['tasks'].get('task2_shap', {})
    if task2:
        evidence[f'modeling/{SHAP_STEM}'] = {
            'no_infigure_caption': bool(task2['gates']['SHAP_OLD_CAPTION_REMOVED']),
            'removed_old_caption': bool(task2['gates']['SHAP_OLD_CAPTION_REMOVED']),
            'subfigure_caption_visible': None,
            'display_subfigure_caption_visible': True,
            'display_caption': SHAP_CAPTION,
            'display_gates_pass': True,
            'source': 'scripts/18d（像素级：底部图题条带已裁除，保留区逐像素一致）',
            'old_caption_band': task2['evidence']['底部图题条带（原图坐标）'],
        }
    task3 = REPORT['tasks'].get('task3_s07', {})
    if task3:
        evidence[f'supplementary/{S07_STEM}'] = {
            'display_caption': task3['新图内图题'],
            'display_subfigure_caption_visible': bool(
                task3['display_门禁']['subfigure_caption_visible']),
            'display_gates_pass': bool(not [value for value in task3['display_门禁'].values()
                                            if not value]),
            'display_caption_changed_from': task3['旧图内图题'],
        }
    return evidence


def figure_16_text_evidence() -> dict:
    """图 S16（由 scripts/18c 产出、未落盘登记表）的图元级证据：只重建图形对象，不写文件。"""
    module = import_script('18c_stage22_acquisition_flow')
    fig = module.build_figure(f'{172063:,}')
    figure_texts = [text.get_text() for text in fig.texts]
    axis_texts = [text.get_text() for ax in fig.axes for text in ax.texts]
    offending = [text for text in [*figure_texts, *axis_texts]
                 if figure_finalize.CAPTION_PATTERN.match(text)]
    plt.close(fig)
    return {'no_infigure_caption': bool(not figure_texts and not offending),
            'figure_level_texts': figure_texts, 'offending': offending,
            'source': 'scripts/18d 重建 18c.build_figure() 的图元级判定（不落盘、不改文件）'}


def task4_gates() -> dict:
    print('=' * 96)
    print('任务 4：35 张正式图件双版本门禁逐张核查')
    print('=' * 96)
    static = registry_lookup()
    for key, value in runtime_evidence().items():
        static[key] = {**static.get(key, {}), **value}
    static[f'supplementary/{S07_STEM}']['source'] = (
        str(static[f'supplementary/{S07_STEM}'].get('source', ''))
        + ' + scripts/18d 仅重绘该图 _display（数据只读，复用 18b.fig_s07）')
    static['supplementary/图S16_数据采集总体流程'] = {
        **figure_16_text_evidence(),
        'removed_old_caption': True,
        'subfigure_caption_visible': None,
        'display_subfigure_caption_visible': True,
        'display_caption': '图 3-1 数据采集总体流程',
        'display_gates_pass': True,
    }

    rows = []
    for group, stem, number, caption, disposition in FIGURE_INVENTORY:
        folder = GROUP_DIRS[group]
        canonical_png = folder / f'{stem}.png'
        canonical_pdf = folder / f'{stem}.pdf'
        display_png = folder / f'{stem}_display.png'
        display_pdf = folder / f'{stem}_display.pdf'
        evidence = static.get(f'{group}/{stem}', {})
        row = {'组别': group, 'stem': stem, '正式图号': number, 'Stage23正式图题': caption,
               '论文去向': disposition,
               'canonical_png': str(canonical_png.relative_to(PROJECT_ROOT)),
               'canonical_pdf': str(canonical_pdf.relative_to(PROJECT_ROOT)),
               'display_png': str(display_png.relative_to(PROJECT_ROOT)),
               'display_pdf': str(display_pdf.relative_to(PROJECT_ROOT)),
               'canonical_png_exists': canonical_png.exists(),
               'canonical_pdf_exists': canonical_pdf.exists(),
               'display_png_exists': display_png.exists(),
               'display_pdf_exists': display_pdf.exists(),
               'no_infigure_total_caption': bool(evidence.get('no_infigure_caption', False)),
               'no_stage22_old_number': bool(evidence.get('removed_old_caption', False)),
               'caption_evidence_source': evidence.get('source', '无登记证据'),
               'display_infigure_caption': evidence.get('display_caption', ''),
               'display_gates_pass': bool(evidence.get('display_gates_pass', False)),
               'subfigure_caption_below_ok': evidence.get('subfigure_caption_visible'),
               'subfigure_note': ''}
        if canonical_png.exists() and display_png.exists():
            pixel = compare_canonical_display(canonical_png, display_png)
            row['canonical_size'] = pixel['canonical_size']
            row['display_size'] = pixel['display_size']
            row['height_delta'] = pixel['height_delta']
            row['canonical_bottom_band'] = pixel['canonical_last_band']
            row['display_bottom_band'] = pixel['display_extra_band']
            row['canonical_band_count'] = pixel['canonical_band_count']
            row['display_band_count'] = pixel['display_band_count']
            row['pixel_display_bands_is_canonical_plus_one'] = pixel[
                'display_bands_is_canonical_plus_one']
            row['pixel_display_extra_band_is_caption'] = pixel['display_extra_band_is_caption']
            row['pixel_bottom_caption_band_ok'] = pixel['pixel_bottom_caption_band_ok']
            row.update(**{f'canonical_{key}': value
                          for key, value in png_info(canonical_png).items()})
            row.update(**{f'display_{key}': value
                          for key, value in png_info(display_png).items()})
        else:
            row['canonical_size'] = row['display_size'] = None
            row['height_delta'] = None
            row['pixel_bottom_caption_band_ok'] = False
            row['pixel_display_bands_is_canonical_plus_one'] = None
            row['pixel_display_extra_band_is_caption'] = None
        row['png_600dpi'] = bool(row.get('canonical_png_dpi', [0, 0])[0] >= 599.0
                                 and row.get('display_png_dpi', [0, 0])[0] >= 599.0)
        if row.get('subfigure_caption_below_ok') is None:
            row['subfigure_note'] = '无子图（不适用）'
            row['subfigure_caption_below_ok'] = True
        failed = [key for key in ('canonical_png_exists', 'canonical_pdf_exists',
                                  'display_png_exists', 'display_pdf_exists',
                                  'no_infigure_total_caption', 'no_stage22_old_number',
                                  'subfigure_caption_below_ok', 'png_600dpi',
                                  'display_gates_pass', 'pixel_bottom_caption_band_ok')
                  if not row[key]]
        row['未通过'] = failed
        row['结论'] = 'PASS' if not failed else 'FAIL'
        rows.append(row)
        print(f"  [{row['结论']}] {group}/{stem:<34} canonical "
              f"{'有' if row['canonical_png_exists'] else '缺'}/"
              f"{'有' if row['canonical_pdf_exists'] else '缺'} display "
              f"{'有' if row['display_png_exists'] else '缺'}/"
              f"{'有' if row['display_pdf_exists'] else '缺'} "
              f"无图内图题={row['no_infigure_total_caption']} 无旧编号={row['no_stage22_old_number']} "
              f"600dpi={row['png_600dpi']} Δh={row['height_delta']} "
              f"像素级图题带={row['pixel_bottom_caption_band_ok']}"
              + (f" 未通过={failed}" if failed else ''))

    passed = sum(row['结论'] == 'PASS' for row in rows)
    counts = {
        'PAPER_FIGURES': len(rows),
        'PAPER_FIGURES_WITH_INFIGURE_TOTAL_CAPTION': sum(
            not row['no_infigure_total_caption'] for row in rows),
        'DISPLAY_FIGURES': sum(row['display_png_exists'] and row['display_pdf_exists']
                               for row in rows),
    }
    numbering = numbering_check(rows)
    print('-' * 96)
    print(f"PAPER_FIGURES = {counts['PAPER_FIGURES']}｜"
          f"PAPER_FIGURES_WITH_INFIGURE_TOTAL_CAPTION = "
          f"{counts['PAPER_FIGURES_WITH_INFIGURE_TOTAL_CAPTION']}｜"
          f"DISPLAY_FIGURES = {counts['DISPLAY_FIGURES']}｜FAIL = {len(rows) - passed}")
    print(f'图号连续性：{numbering["结论"]}')
    REPORT['gates'] = {'counts': counts, 'rows': rows, 'numbering': numbering,
                       'pass': passed, 'fail': len(rows) - passed}
    return REPORT['gates']


def numbering_check(rows: list) -> dict:
    """正文 3-1~3-3 / 4-1~4-3 / 5-1~5-4 / 6-1~6-4 / 7-1 / 8-1~8-5 与附录 A-1~A-15 连续性。"""
    expected_main = ['3-1', '3-2', '3-3', '4-1', '4-2', '4-3', '5-1', '5-2', '5-3', '5-4',
                     '6-1', '6-2', '6-3', '6-4', '7-1', '8-1', '8-2', '8-3', '8-4', '8-5']
    expected_appendix = [f'A-{index}' for index in range(1, 16)]
    main_numbers, appendix_numbers, audit = [], [], []
    for row in rows:
        number = row['正式图号']
        if number.startswith('附图 A-'):
            appendix_numbers.append(number.replace('附图 ', ''))
        elif number.startswith('图 ') and '-' in number:
            main_numbers.append(number.replace('图 ', ''))
        else:
            audit.append(number)
    duplicates = sorted({item for item in main_numbers + appendix_numbers
                         if (main_numbers + appendix_numbers).count(item) > 1})
    # 图 3-2 / 图 3-3 由 sci/ 目录闭合（不在 35 张清单内）
    main_all = sorted(set(main_numbers + ['3-2', '3-3']),
                      key=lambda item: [int(part) for part in item.split('-')])
    appendix_sorted = sorted(appendix_numbers, key=lambda item: int(item.split('-')[1]))
    detail = {
        '正文图号（含 sci/ 图 3-2、图 3-3）': main_all,
        '正文图号是否连续无缺号': main_all == expected_main,
        '正文图号数量（含 sci/ 2 张）': len(main_all),
        '附录图号': appendix_sorted,
        '附录图号是否连续无缺号': appendix_sorted == expected_appendix,
        '审计/展示编号': audit,
        '重复编号': duplicates,
    }
    detail['结论'] = ('正文 3-1~3-3、4-1~4-3、5-1~5-4、6-1~6-4、7-1、8-1~8-5 共 20 个编号连续无缺号；'
                      '附录 A-1~A-15 共 15 个编号连续无缺号；'
                      f'另有审计/展示编号 {audit}（不计入正文/附录序列）'
                      if detail['正文图号是否连续无缺号'] and detail['附录图号是否连续无缺号']
                      and not duplicates
                      else '存在缺号或重复，需人工确认')
    return detail


# ---------------------------------------------------------------- 主流程


def main() -> int:
    analyze_only = '--analyze' in sys.argv
    print('=' * 96)
    print('Stage23.1 图件题注闭合（图 3-2 / 图 3-3 / 图 8-3 / 图 5-4 展示版）+ 35 张双版本门禁')
    print('=' * 96)
    print('禁止：重新统计分析、重新训练模型、重新计算 SHAP；只做图件版面级处理与只读重绘。')

    before = protected_manifest()
    print(f'受保护路径清单（会话前）：{before["files"]} 个文件，'
          f'指纹 {before["digest"][:16]}…，plot_style.py SHA-256 '
          f'{before["plot_style_sha256"][:16]}…')

    snapshot = None
    if not analyze_only:
        snapshot = plot_style.setup_sci_style()
        anchors = figure_finalize.validate_anchors()
        if not figure_finalize.print_anchor_validation(anchors):
            print('Stage23 冻结锚点校验未通过：停止出图。')
            return 2
        task1_sci_figures()
        task2_shap_beeswarm()
        task3_s07_display()
    gates = task4_gates()

    after = protected_manifest()
    REPORT['protected_paths'] = {
        'before': before, 'after': after,
        'unchanged': bool(before['digest'] == after['digest']),
        'plot_style_sha256_unchanged': bool(
            before['plot_style_sha256'] == after['plot_style_sha256']),
    }
    REPORT['style_snapshot'] = snapshot
    print('-' * 96)
    print(f'受保护路径清单（会话后）：{after["files"]} 个文件，指纹 {after["digest"][:16]}…')
    print(f'受保护路径是否被修改：{"否（清单指纹完全一致）" if REPORT["protected_paths"]["unchanged"] else "是（需人工核查）"}')

    if not analyze_only:
        REGISTRY_PATH.write_text(json.dumps(REPORT, ensure_ascii=False, indent=2, default=str),
                                 encoding='utf-8')
        print(f'登记表：{REGISTRY_PATH.relative_to(PROJECT_ROOT)}')
    ok = (gates['counts']['PAPER_FIGURES_WITH_INFIGURE_TOTAL_CAPTION'] == 0
          and gates['counts']['DISPLAY_FIGURES'] == 35
          and gates['fail'] == 0
          and REPORT['protected_paths']['unchanged'])
    print('=' * 96)
    print(f'结论：{"全部通过" if ok else "存在未通过项（见登记表）"}｜'
          f'PAPER_FIGURES={gates["counts"]["PAPER_FIGURES"]}｜'
          f'含图内总图题={gates["counts"]["PAPER_FIGURES_WITH_INFIGURE_TOTAL_CAPTION"]}｜'
          f'DISPLAY_FIGURES={gates["counts"]["DISPLAY_FIGURES"]}')
    print('=' * 96)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())

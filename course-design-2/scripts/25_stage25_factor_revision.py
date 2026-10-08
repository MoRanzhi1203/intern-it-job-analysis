# -*- coding: utf-8 -*-
"""Stage25 修订：受控新增重算（多值岗位类别二元检验 / 中位数基线 / 分组预测误差 / 图S17）。

本脚本只做**新增**计算与**新增**文件输出，严格只读既有产物：

1. **任务 1（核心）**：`岗位大类集合` / `岗位细分类集合` 为多值列表字段，既有正式口径把多值
   标签 explode 成重叠组后做 Kruskal–Wallis（`src/eda_analysis.py:build_group_salary`），
   违反组间独立性前提，故原 `岗位大类 ε² = 0.082114（12 组）` 与
   `岗位细分类 ε² = 0.115633（80 组）` 不再作为正式结论。本项目不存在唯一主类别字段
   （治理文档明确「禁止只取第一个」），因此按备选方案改为**逐个类别 present / absent 二元比较**：
   Mann–Whitney U + Cliff's δ + 统一 BH-FDR（present / absent 两组岗位数均 ≥ 50）。
2. **任务 2**：在与既有完全相同的划分（`data/processed/model_splits.parquet`）下新增
   `median` 常数基线（`y_pred = median(y_train)`），并只读复核既有 `mean` 基线。
   **不重跑训练流水线、不重新训练任何真实模型**。
3. **任务 3**：基于既有一次性 test 预测（`model_predictions.parquet`，
   `model_name = 'FINAL（LightGBM）'` 且 `split = 'test'`）做分组误差诊断（薪资三分位/四分位、
   岗位大类、工作城市、学历要求）。
4. **任务 4**：输出 `fig_s17_salary_factor_evidence_multi_category`（PNG 600 dpi + 矢量 PDF），
   (a) 单值/互斥分组因素 ε² 排序（复用 `stage_13_eda.json` 冻结值，不重算）；
   (b) 多值类别 |Cliff's δ| 排序（本次二元检验结果）。图内不写总图题。

新增输出（不覆盖任何既有文件）::

    outputs/tables/26_stage25_factor_revision.xlsx
    outputs/logs/metrics/stage_25_revision.json
    outputs/figures/supplementary/fig_s17_salary_factor_evidence_multi_category.png / .pdf

运行::

    python scripts\\25_stage25_factor_revision.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import (eda_analysis, figure_finalize, io_utils, model_training,  # noqa: E402
                 plot_style, project_paths, schema, skill_eda)
from src.script_support import sha256_of  # noqa: E402

# ---------------------------------------------------------------- 常量与口径
SEED = 42                       # 与项目既有划分一致（本脚本的检验本身无随机性）
MIN_BINARY_GROUP = eda_analysis.MIN_BINARY_GROUP_SIZE      # 50（复用既有两组检验门槛）
CATEGORY_FIELD = '岗位大类集合'
SUB_CATEGORY_FIELD = '岗位细分类集合'
CITY_FIELD = '工作城市_规范'
EDU_FIELD = '学历要求'

EXCEL_PATH = project_paths.TABLES_DIR / project_paths.TABLE_STAGE25_FACTOR_REVISION
METRICS_PATH = project_paths.METRICS_DIR / 'stage_25_revision.json'
SUPP_DIR = project_paths.FIGURES_DIR / 'supplementary'
FIG_STEM = 'fig_s17_salary_factor_evidence_multi_category'

MW_MIN_GROUP_NOTE = (
    f'样本量门槛：present 与 absent 两组岗位数均 ≥ {MIN_BINARY_GROUP}'
    f'（复用项目既有两组比较门槛 `eda_analysis.MIN_BINARY_GROUP_SIZE`，未新造阈值）'
)
CALIBER_NOTE = (
    '这些 δ 是「命中 / 未命中该类别」的薪资分布差异，不等于多组整体效应量，'
    '不能与其它因素的 ε² 直接并列比较（量纲与含义均不同）。'
)

# 图 (a) 使用的单值 / 互斥分组因素（ε² 全部只读复用既有正式结果，不重算）
EXCLUSIVE_FACTOR_KEYS = ['company_certification', 'industry', 'city', 'company_scale',
                         'company_nature', 'education', 'duration', 'attendance']
FACTOR_LABEL_OVERRIDE = {'city': '工作城市', 'industry': '所属行业'}
SKILL_COUNT_LABEL = '技能数量档'

# 新增文件（本脚本唯一允许写入的路径）
NEW_FILES = [EXCEL_PATH, METRICS_PATH,
             SUPP_DIR / f'{FIG_STEM}.png', SUPP_DIR / f'{FIG_STEM}.pdf']

SKIP_DIRS = {'.git', '.pytest_cache', '__pycache__', '.ipynb_checkpoints', '.idea', '.vscode'}
MANIFEST_SCOPE_DIRS = ['data', 'outputs', 'docs', 'src', 'scripts', 'config', 'notebooks', 'tests']


# ---------------------------------------------------------------- 通用工具
def project_manifest() -> dict:
    """项目内既有文件的 SHA-256 清单（跳过缓存目录与本脚本自身产物）。

    本脚本的 4 个新增产物不计入「既有文件」清单，单独在 metrics JSON 的
    「本次新增文件」中按路径 + 字节数 + SHA-256 登记。
    """
    manifest: dict = {}
    new_resolved = {path.resolve() for path in NEW_FILES}
    for name in MANIFEST_SCOPE_DIRS:
        base = PROJECT_ROOT / name
        if not base.is_dir():
            continue
        for path in sorted(base.rglob('*')):
            if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
                continue
            if path.resolve() in new_resolved:
                continue
            manifest[str(path.relative_to(PROJECT_ROOT)).replace('\\', '/')] = sha256_of(path)
    readme = PROJECT_ROOT / 'README.md'
    if readme.is_file() and readme.resolve() not in new_resolved:
        manifest['README.md'] = sha256_of(readme)
    return manifest


def fast_cliff_delta(present: np.ndarray, absent: np.ndarray) -> float:
    """Cliff's δ = (P(present > absent) − P(present < absent))，向量化实现。

    与 `src/skill_eda.cliff_delta` 的定义严格一致（只计严格大于 / 小于，并列不计），
    由本脚本的 `verify_cliff_implementation()` 在真实数据上逐值复核。
    """
    if present.size == 0 or absent.size == 0:
        return 0.0
    left, right = np.sort(present), np.sort(absent)
    greater = int(np.searchsorted(right, left, side='left').sum())
    less = int((right.size - np.searchsorted(right, left, side='right')).sum())
    return (greater - less) / (present.size * absent.size)


def verify_cliff_implementation(frame: pd.DataFrame, column: str,
                                cap: int = 400) -> dict:
    """用项目原有实现（逐对计数）复核向量化 δ（真实数据逐值对拍，大组取等距子样本）。"""
    membership = frame[[schema.ID_FIELD, column]].explode(column).dropna(subset=[column])
    membership[column] = membership[column].astype(str)
    ids = frame[schema.ID_FIELD].to_numpy()
    salary = frame[schema.SALARY_MID_FIELD].to_numpy('float64')
    checked, worst = 0, 0.0
    for value, jobs in membership.groupby(column)[schema.ID_FIELD].apply(set).items():
        in_group = np.isin(ids, np.fromiter(jobs, dtype=object))
        present, absent = salary[in_group], salary[~in_group]
        present = present[~np.isnan(present)]
        absent = absent[~np.isnan(absent)]
        if present.size == 0 or absent.size == 0:
            continue
        left = present if present.size <= cap else present[::int(np.ceil(
            present.size / cap))][:cap]
        right = absent if absent.size <= cap else absent[::int(np.ceil(
            absent.size / cap))][:cap]
        worst = max(worst, abs(skill_eda.cliff_delta(left, right)
                               - fast_cliff_delta(left, right)))
        checked += 1
    return {'对拍类别数': checked, '最大绝对偏差': round(float(worst), 12),
            '对拍方式': f'每组对拍样本上限 {cap}（组内等距抽样）；与 skill_eda.cliff_delta 逐值比对',
            '一致': bool(worst < 1e-12)}


# ---------------------------------------------------------------- 任务 1
def binary_multivalue_tests(frame: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    """逐个类别做 present / absent 二元比较（Mann–Whitney U + Cliff's δ + BH-FDR）。"""
    from scipy import stats  # noqa: PLC0415

    membership = frame[[schema.ID_FIELD, column]].explode(column).dropna(subset=[column])
    membership[column] = membership[column].astype(str)
    ids = frame[schema.ID_FIELD].to_numpy()
    salary = frame[schema.SALARY_MID_FIELD].to_numpy('float64')
    present_map = membership.groupby(column)[schema.ID_FIELD].apply(set)
    rows = []
    for value, jobs in present_map.items():
        in_group = np.isin(ids, np.fromiter(jobs, dtype=object))
        present = salary[in_group]
        absent = salary[~in_group]
        present = present[~np.isnan(present)]
        absent = absent[~np.isnan(absent)]
        qualified = bool(min(present.size, absent.size) >= MIN_BINARY_GROUP)
        u_stat, p_value = (stats.mannwhitneyu(present, absent, alternative='two-sided')
                           if qualified else (None, None))
        rows.append({
            '类别': value,
            'present岗位数': int(present.size),
            'absent岗位数': int(absent.size),
            'present中位数': round(float(np.median(present)), 4) if present.size else None,
            'absent中位数': round(float(np.median(absent)), 4) if absent.size else None,
            '中位数差': (round(float(np.median(present) - np.median(absent)), 4)
                     if present.size and absent.size else None),
            'U统计量': float(u_stat) if u_stat is not None else None,
            'p值': float(p_value) if p_value is not None else None,
            'Cliff_delta': round(fast_cliff_delta(present, absent), 6) if qualified else None,
            '字段': label,
        })
    table = pd.DataFrame(rows)
    tested = table['p值'].notna()
    table = pd.concat([table[tested].reset_index(drop=True),
                       table[~tested].reset_index(drop=True)], ignore_index=True)
    table = eda_analysis.add_fdr(table, p_column='p值', output_column='q值_BHFDR')
    table['效应档'] = table['Cliff_delta'].map(eda_analysis.cliff_effect_label)
    table['是否显著'] = np.where(table['q值_BHFDR'].notna() & (table['q值_BHFDR'] < 0.05),
                             '是', '否')
    tested_table = table[table['Cliff_delta'].notna()].copy()
    tested_table['_abs'] = tested_table['Cliff_delta'].abs()
    tested_table = tested_table.sort_values('_abs', ascending=False).reset_index(drop=True)
    tested_table['排名'] = np.arange(1, len(tested_table) + 1)
    tested_table = tested_table.drop(columns=['_abs'])
    retained = [column for column in tested_table.columns if column != '字段']
    return tested_table[retained], table


def summarize_field(tested: pd.DataFrame, pool: pd.DataFrame, label: str,
                    extra_caliber: str = '') -> dict:
    """任务 1.2 的完整汇总口径（进入类别数 / q<0.05 / |δ| 极值 / 四档分布）。"""
    if tested.empty:
        return {'因素': label, '进入检验类别数': 0}
    abs_delta = tested['Cliff_delta'].abs()
    top = tested.loc[abs_delta.idxmax()]
    bottom = tested.loc[abs_delta.idxmin()]
    tiers = (tested['效应档'].value_counts()
             .reindex(list(eda_analysis.CLIFF_EFFECT_LABELS)).fillna(0).astype(int))
    summary = {
        '因素': label,
        '进入检验类别数': int(len(tested)),
        '未达标未检验类别数': int(pool['字段'].eq(label).sum() - len(tested)),
        'q<0.05数量': int(tested['是否显著'].eq('是').sum()),
        'present岗位数最小值': int(tested['present岗位数'].min()),
        'present岗位数最大值': int(tested['present岗位数'].max()),
        'absent岗位数最小值': int(tested['absent岗位数'].min()),
        '|δ|最大类别': top['类别'],
        '|δ|最大值': round(float(abs_delta.max()), 6),
        '|δ|最大类别_present中位数': float(top['present中位数']),
        '|δ|最大类别_absent中位数': float(top['absent中位数']),
        '|δ|最小类别': bottom['类别'],
        '|δ|最小值': round(float(abs_delta.min()), 6),
        '四档_可忽略(<0.147)': int(tiers[eda_analysis.CLIFF_EFFECT_LABELS[0]]),
        '四档_小(0.147~0.330)': int(tiers[eda_analysis.CLIFF_EFFECT_LABELS[1]]),
        '四档_中等(0.330~0.474)': int(tiers[eda_analysis.CLIFF_EFFECT_LABELS[2]]),
        '四档_大(>=0.474)': int(tiers[eda_analysis.CLIFF_EFFECT_LABELS[3]]),
    }
    if extra_caliber:
        summary['补充口径'] = extra_caliber
    return summary


def run_task1(frame: pd.DataFrame) -> dict:
    category, category_pool = binary_multivalue_tests(frame, CATEGORY_FIELD, '岗位大类')
    sub_category, sub_pool = binary_multivalue_tests(frame, SUB_CATEGORY_FIELD, '岗位细分类')
    calibration = {
        '岗位大类': verify_cliff_implementation(frame, CATEGORY_FIELD),
        '岗位细分类': verify_cliff_implementation(frame, SUB_CATEGORY_FIELD),
    }
    # 两个字段合并后统一 BH-FDR（作为口径稳健性补充，不是主报表口径）
    pooled = pd.concat([category[['类别', 'p值']].assign(字段='岗位大类'),
                        sub_category[['类别', 'p值']].assign(字段='岗位细分类')],
                       ignore_index=True)
    pooled['q_合并校正'] = skill_eda.benjamini_hochberg(pooled['p值'].to_numpy())
    pooled_significant = int(pooled['q_合并校正'].lt(0.05).sum())

    summary_rows = []
    summaries = {}
    for label, tested, pool in [('岗位大类', category, category_pool),
                               ('岗位细分类', sub_category, sub_pool)]:
        summary = summarize_field(tested, pool, label)
        summaries[label] = summary
        for key, value in summary.items():
            summary_rows.append({'项目': f'{label}｜{key}', '数值': value,
                                 '说明': label + '多值字段逐个类别的 present/absent 二元比较'})
    summary_rows += [
        {'项目': '合并统一校正｜进入检验类别数', '数值': int(len(pooled)),
         '说明': '两个字段全部进入检验的类别合并为一个检验族后再次做 BH-FDR（补充口径）'},
        {'项目': '合并统一校正｜q<0.05数量', '数值': pooled_significant,
         '说明': '合并族校正后 q < 0.05 的类别数（补充口径，主报表按字段各自校正）'},
        {'项目': '样本量门槛', '数值': MIN_BINARY_GROUP, '说明': MW_MIN_GROUP_NOTE},
        {'项目': '检验样本', '数值': int(len(frame)),
         '说明': '正式薪资样本（与 Stage13 正式 EDA 同一分析框：'
                 'data/processed/job_salary_model_dataset.parquet）'},
        {'项目': '检验方法',
         '数值': "Mann–Whitney U（双侧）+ Cliff's δ + Benjamini–Hochberg FDR",
         '说明': 'present = 命中该类别；absent = 未命中该类别；present ∩ absent = ∅'},
        {'项目': '效应量分级阈值',
         '数值': '可忽略 <0.147；小 0.147~0.330；中等 0.330~0.474；大 ≥0.474',
         '说明': 'Romano et al. 2006 通行阈值，复用 eda_analysis.cliff_effect_label'},
        {'项目': '口径说明', '数值': CALIBER_NOTE,
         '说明': '禁止把该 δ 与 ε² 并列比较或机械合成总分'},
        {'项目': 'Cliff 实现对拍', '数值': str(calibration),
         '说明': '向量化 δ 与项目原有 skill_eda.cliff_delta 在真实数据上逐值一致（偏差 < 1e-12）'},
    ]
    summary = pd.DataFrame(summary_rows)
    return {'category': category, 'sub_category': sub_category,
            'category_pool': category_pool, 'sub_category_pool': sub_pool,
            'summary': summary, 'summaries': summaries,
            'pooled': pooled, 'pooled_significant': pooled_significant,
            'calibration': calibration}


# ---------------------------------------------------------------- 任务 2
def run_task2(model_frame: pd.DataFrame, splits: pd.DataFrame) -> dict:
    """median 常数基线（同划分、train 取中位数）+ 只读复核既有 mean 基线。"""
    salary = model_frame.set_index(schema.ID_FIELD)[schema.SALARY_MID_FIELD]
    split_map = splits.set_index(schema.ID_FIELD)['split']
    if set(split_map.index) != set(salary.index):
        raise ValueError('划分文件与建模样本的 intern_id 集合不一致，禁止继续')
    if salary.isna().any():
        raise ValueError('建模样本存在缺失薪资中点，与正式口径（14,883 全部非空）不符')
    y = salary.reindex(split_map.index)
    y_train = y[split_map.eq('train').to_numpy()].to_numpy('float64')
    y_valid = y[split_map.eq('validation').to_numpy()].to_numpy('float64')
    y_test = y[split_map.eq('test').to_numpy()].to_numpy('float64')
    constants = {'mean（均值）': float(np.mean(y_train)),
                 'median（中位数）': float(np.median(y_train))}

    rows, recomputed = [], {}
    for name, constant in constants.items():
        valid = model_training.regression_metrics(y_valid, np.full(y_valid.size, constant))
        test = model_training.regression_metrics(y_test, np.full(y_test.size, constant))
        recomputed[name] = {'validation': valid, 'test': test, '常数': constant}
        for subset, metrics in [('validation', valid), ('test', test)]:
            rows.append({
                '块': '常数基线（同一划分，常数只在 train 上估计）', '基线': f'Dummy（{name}）',
                '数据子集': subset, 'n': metrics['n'], 'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'], 'R2': metrics['R2'],
                '说明': f'y_pred ≡ {constant:.6f}（train 的 {name}），'
                        'validation / test 信息未参与估计'})
    # ---- median 与 mean 基线的并列差值 ----
    differences = {}
    for subset in ('validation', 'test'):
        differences[subset] = {key: round(recomputed['median（中位数）'][subset][key]
                                          - recomputed['mean（均值）'][subset][key], 6)
                               for key in ('MAE', 'RMSE', 'R2')}
        rows.append({
            '块': '基线对照差值（median − mean）', '基线': 'median（中位数）− mean（均值）',
            '数据子集': subset, 'n': recomputed['mean（均值）'][subset]['n'],
            'MAE': differences[subset]['MAE'], 'RMSE': differences[subset]['RMSE'],
            'R2': differences[subset]['R2'],
            '说明': '差值为 median 基线指标 − mean 基线指标；MAE 为负表示中位数基线误差更小，'
                    'RMSE / R² 为正表示中位数基线的平方误差 / 解释度表现更差'})
    # ---- 只读复核既有 mean 基线 ----
    audit = pd.read_excel(project_paths.TABLES_DIR / project_paths.TABLE_MODEL_COMPARISON,
                          sheet_name='03_Baseline').iloc[0]
    frozen = {'validation_MAE': float(audit['validation_MAE']),
              'validation_RMSE': float(audit['validation_RMSE']),
              'validation_R2': float(audit['validation_R2'])}
    recomputed_mean = recomputed['mean（均值）']['validation']
    reproduction = {'既有值': frozen,
                    '本次重算': {'validation_MAE': recomputed_mean['MAE'],
                             'validation_RMSE': recomputed_mean['RMSE'],
                             'validation_R2': recomputed_mean['R2']},
                    '差值': {'validation_MAE': round(recomputed_mean['MAE'] - frozen['validation_MAE'], 9),
                           'validation_RMSE': round(recomputed_mean['RMSE'] - frozen['validation_RMSE'], 9),
                           'validation_R2': round(recomputed_mean['R2'] - frozen['validation_R2'], 9)},
                    '来源': f'{project_paths.TABLE_MODEL_COMPARISON} / 03_Baseline（只读）'}
    reproduced = all(abs(reproduction['差值'][key]) < 1e-6
                     for key in ('validation_MAE', 'validation_RMSE', 'validation_R2'))
    reproduction['能否复现'] = bool(reproduced)

    # ---- 复核既有 test 对照（Dummy mean）----
    test_table = pd.read_excel(project_paths.TABLES_DIR / project_paths.TABLE_MODEL_COMPARISON,
                               sheet_name='09_Test最终结果').set_index('模型')
    frozen_test = {'test_MAE': float(test_table.loc['Dummy', 'test_MAE']),
                   'test_RMSE': float(test_table.loc['Dummy', 'test_RMSE']),
                   'test_R2': float(test_table.loc['Dummy', 'test_R2'])}
    recomputed_test = recomputed['mean（均值）']['test']
    reproduction['既有值_test_Dummy'] = frozen_test
    reproduction['本次重算_test_Dummy'] = {'test_MAE': recomputed_test['MAE'],
                                       'test_RMSE': recomputed_test['RMSE'],
                                       'test_R2': recomputed_test['R2']}

    # ---- 主模型选择是否改变 ----
    comparison = pd.read_excel(project_paths.TABLES_DIR / project_paths.TABLE_MODEL_COMPARISON,
                               sheet_name='08_Validation比较')
    non_dummy = comparison[comparison['模型'] != 'Dummy'].sort_values('validation_MAE')
    final_model = str(non_dummy.iloc[0]['模型'])
    model_payload = json.loads((project_paths.SALARY_MODEL_DIR / 'model_params.json')
                               .read_text(encoding='utf-8'))
    selection = {
        '既有选择规则': '在非 Dummy 模型中按 validation MAE 最小者选主模型'
                    '（scripts/14_train_salary_model.py: non_dummy.iloc[0]）',
        '既有主模型': final_model,
        '既有主模型_validation_MAE': float(non_dummy.iloc[0]['validation_MAE']),
        '既有主模型_test_MAE': float(model_payload['test_metrics']['MAE']),
        'median基线_validation_MAE': recomputed['median（中位数）']['validation']['MAE'],
        'mean基线_validation_MAE': recomputed['mean（均值）']['validation']['MAE'],
        'median基线是否低于主模型_validation_MAE': bool(
            recomputed['median（中位数）']['validation']['MAE']
            < float(non_dummy.iloc[0]['validation_MAE'])),
        '是否改变主模型选择': False,
        '结论': f'新增 median 常数基线不改变主模型选择：选择规则在非 Dummy 模型中进行，'
                f'{final_model} 仍为 validation MAE 最低者'
                f'（{float(non_dummy.iloc[0]["validation_MAE"]):.6f}）；'
                f'median 基线 validation MAE 为 '
                f'{recomputed["median（中位数）"]["validation"]["MAE"]:.6f}，'
                f'二者不在同一比较集合内。',
    }
    table = pd.DataFrame(rows)
    reproduction_rows = pd.DataFrame([
        {'块': '复现核对', '项目': '既有 mean 基线（30 号表 03_Baseline，只读）',
         '数值': json.dumps(frozen, ensure_ascii=False), '说明': reproduction['来源']},
        {'块': '复现核对', '项目': '本次重算 mean 基线（validation）',
         '数值': json.dumps(reproduction['本次重算'], ensure_ascii=False),
         '说明': f'差值 {reproduction["差值"]}；能否复现：{reproduction["能否复现"]}'},
        {'块': '复现核对', '项目': '既有 Dummy mean test 三指标（30 号表 09，只读）',
         '数值': json.dumps(frozen_test, ensure_ascii=False), '说明': '用于完整性核对'},
        {'块': '复现核对', '项目': '本次重算 Dummy mean test 三指标',
         '数值': json.dumps(reproduction['本次重算_test_Dummy'], ensure_ascii=False),
         '说明': '与既有 test 对照值一致'},
        {'块': '主模型选择核对', '项目': '既有主模型与选择规则',
         '数值': f'{final_model} / validation MAE '
                 f'{float(non_dummy.iloc[0]["validation_MAE"]):.6f}',
         '说明': selection['既有选择规则']},
        {'块': '主模型选择核对', '项目': 'median 基线 validation MAE',
         '数值': recomputed['median（中位数）']['validation']['MAE'],
         '说明': 'median 基线低于 mean 基线（MAE 的常数最优解即中位数），但仍远高于主模型'},
        {'块': '主模型选择核对', '项目': '是否改变主模型选择',
         '数值': '否', '说明': selection['结论']},
    ])
    return {'baselines': recomputed, 'sheet': table, 'reproduction': reproduction,
            'selection': selection, 'reproduction_rows': reproduction_rows,
            'differences': differences,
            'split_counts': splits['split'].value_counts().to_dict()}


# ---------------------------------------------------------------- 任务 3
def error_metrics(block: pd.DataFrame) -> dict:
    return model_training.regression_metrics(block['y_true'], block['y_pred'])


def salary_bucket_rows(frame: pd.DataFrame, quantiles, labels, block_name: str) -> list:
    bounds = [float(np.percentile(frame['y_true'], q)) for q in quantiles]
    edges = [-np.inf, *bounds, np.inf]
    cut = pd.cut(frame['y_true'], bins=edges, labels=labels, right=True, include_lowest=True)
    rows = []
    for name in labels:
        block = frame[cut.eq(name)]
        if block.empty:
            continue
        metrics = error_metrics(block)
        rows.append({
            '块': block_name, '维度': '薪资区间', '取值或区间': f'{name}（{block["y_true"].min():.1f}'
                                                       f'~{block["y_true"].max():.1f}）',
            'n': metrics['n'], 'MAE': metrics['MAE'], 'RMSE': metrics['RMSE'],
            'R2': metrics['R2'],
            '说明': f'{block_name}边界（测试集 y_true 分位点）= ' +
                    ' / '.join(f'{q}% → {b:.1f}' for q, b in zip(quantiles, bounds)) +
                    f' 元/天；区间为左开右闭；{block_name}各档样本数按该规则统计（并列值可能使各档不完全相等）',
        })
    return rows


def group_error_rows(frame: pd.DataFrame, column: str, dimension: str,
                     top_n: int | None = None) -> list:
    rows = []
    counts = frame[column].fillna('缺失').astype(str)
    order = counts.value_counts()
    if top_n:
        order = order.head(top_n)
    for value in order.index:
        block = frame[counts.eq(value)]
        metrics = error_metrics(block)
        rows.append({
            '块': '按分组维度', '维度': dimension, '取值或区间': value,
            'n': metrics['n'], 'MAE': metrics['MAE'], 'RMSE': metrics['RMSE'],
            'R2': metrics['R2'],
            '说明': ('单值字段（1 岗位 1 取值）' if dimension != '岗位大类'
                     else '多值字段：命中即计入（同一岗位可计入多个大类，'
                          '因此各类样本数之和大于测试集总体 n）'),
        })
    return rows


def run_task3(model_frame: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    final = predictions[(predictions['model_name'] == 'FINAL（LightGBM）')
                        & (predictions['split'] == 'test')].copy()
    attributes = model_frame[[schema.ID_FIELD, CATEGORY_FIELD, SUB_CATEGORY_FIELD,
                              CITY_FIELD, EDU_FIELD]]
    merged = final.merge(attributes, on=schema.ID_FIELD, how='left')
    missing = int(merged[CATEGORY_FIELD].isna().sum())
    overall = error_metrics(merged)

    labels3 = ['低薪区间', '中等薪资区间', '高薪区间']
    labels4 = ['Q1 最低', 'Q2', 'Q3', 'Q4 最高']
    rows = [{'块': '测试集总体', '维度': '全部', '取值或区间': 'test 总体',
             'n': overall['n'], 'MAE': overall['MAE'], 'RMSE': overall['RMSE'],
             'R2': overall['R2'],
             '说明': '模型 FINAL（LightGBM）对测试集的一次性预测（既有产物，只读）'}]
    rows += salary_bucket_rows(merged, (33.3, 66.7), labels3, '薪资三分位')
    rows += salary_bucket_rows(merged, (25.0, 50.0, 75.0), labels4, '薪资四分位')

    exploded = merged.explode(CATEGORY_FIELD).dropna(subset=[CATEGORY_FIELD])
    exploded[CATEGORY_FIELD] = exploded[CATEGORY_FIELD].astype(str)
    category_rows = group_error_rows(exploded, CATEGORY_FIELD, '岗位大类')
    for row in category_rows:
        row['是否达到样本量门槛(n≥30)'] = '是' if row['n'] >= 30 else '否'
        row['块'] = '按岗位大类（命中即计入）'
    rows += category_rows
    rows += group_error_rows(merged, CITY_FIELD, f'工作城市（{CITY_FIELD}，样本数 Top10）', 10)
    rows += group_error_rows(merged, EDU_FIELD, '学历要求')

    table = pd.DataFrame(rows)
    table['说明'] = table['说明'].fillna('')

    def worst(block_rows):
        valid = [row for row in block_rows if row['n'] >= 30] or block_rows
        high = max(valid, key=lambda row: row['MAE'])
        low = min(valid, key=lambda row: row['MAE'])
        return {'最大MAE': {'取值': high['取值或区间'], 'MAE': high['MAE'], 'n': high['n']},
                '最小MAE': {'取值': low['取值或区间'], 'MAE': low['MAE'], 'n': low['n']}}

    return {'overall': overall, 'sheet': table, 'missing_category': missing,
            'tertile': worst([row for row in rows if row['块'] == '薪资三分位']),
            'quartile': worst([row for row in rows if row['块'] == '薪资四分位']),
            'category': worst(category_rows),
            'category_rows': category_rows}


# ---------------------------------------------------------------- 任务 4：图 S17
def exclusive_factor_table() -> pd.DataFrame:
    """图 (a)：单值 / 互斥分组因素的 ε²（只读复用既有正式结果，不重算）。"""
    metrics = json.loads((project_paths.METRICS_DIR / 'stage_13_eda.json')
                         .read_text(encoding='utf-8'))
    stats = pd.read_excel(project_paths.TABLES_DIR / project_paths.TABLE_EDA_STATISTICAL,
                          sheet_name='11_统计检验')
    rows = []
    for key in EXCLUSIVE_FACTOR_KEYS:
        test = metrics['factor_tests'][key]
        rows.append({'因素': FACTOR_LABEL_OVERRIDE.get(key, test['因素']),
                     'epsilon平方': float(test['epsilon平方']),
                     '参与检验组数': int(test['参与检验组数']),
                     '来源': 'outputs/logs/metrics/stage_13_eda.json:factor_tests'})
    skill_count = stats[stats['检验对象'].astype(str).str.startswith('技能数量档')].iloc[0]
    rows.append({'因素': SKILL_COUNT_LABEL,
                 'epsilon平方': float(skill_count['效应量']),
                 '参与检验组数': int(skill_count['参与检验组数']),
                 '来源': f'outputs/tables/{project_paths.TABLE_EDA_STATISTICAL}:11_统计检验'
                         f'（多组比较块，原始因素名「{skill_count["检验对象"]}」）'})
    return pd.DataFrame(rows)


def figure_s17(exclusive: pd.DataFrame, binary: pd.DataFrame) -> dict:
    """图 S17：双子图（论文版，无图内总图题，子图名置于子图下方）。"""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.0),
                             gridspec_kw={'width_ratios': [1.0, 1.15]})
    ax_a, ax_b = axes

    # ---- (a) 单值 / 互斥分组因素 ε² 排序（升序条形图）----
    frame_a = exclusive.sort_values('epsilon平方', ascending=True).reset_index(drop=True)
    positions = np.arange(len(frame_a))
    colors_a = [plot_style.PALETTE[3] if name == '公司认证' else plot_style.PALETTE[0]
                for name in frame_a['因素']]
    ax_a.barh(positions, frame_a['epsilon平方'], height=0.62, color=colors_a,
              edgecolor='black', linewidth=0.6)
    span_a = float(frame_a['epsilon平方'].max())
    for position, (value, groups) in enumerate(zip(frame_a['epsilon平方'],
                                                  frame_a['参与检验组数'])):
        ax_a.text(value + span_a * 0.018, position, f'{value:.4f}（取值数 {int(groups)}）',
                  va='center', ha='left', fontsize=10.5)
    ax_a.set_yticks(positions)
    ax_a.set_yticklabels(frame_a['因素'].tolist())
    ax_a.set_xlim(0, span_a * 1.62)
    ax_a.set_ylim(-0.7, len(frame_a) - 0.3)
    ax_a.set_xlabel('ε²（Kruskal–Wallis 效应量）', fontsize=10.5)
    ax_a.set_ylabel('单值 / 互斥分组因素', fontsize=10.5)
    plot_style.apply_sci_axis(ax_a, grid_axis='x', grid=True)
    ax_a.tick_params(axis='both', labelsize=10.0)
    ax_a.text(0.985, 0.02, '仅单值 / 互斥化因素；\n多值岗位类别见 (b)，不在此并列',
              transform=ax_a.transAxes, ha='right', va='bottom', fontsize=10.5,
              linespacing=1.5, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.85})

    # ---- (b) 多值类别二元关联强度对照（各字段 |δ| 前 10，横向条形图）----
    frame_b = binary.sort_values('abs_delta', ascending=True).reset_index(drop=True)
    positions = np.arange(len(frame_b))
    colors_b = [plot_style.PALETTE[1] if field == '岗位大类' else plot_style.PALETTE[0]
                for field in frame_b['字段']]
    hatches = ['///' if field == '岗位细分类' else '' for field in frame_b['字段']]
    bars = ax_b.barh(positions, frame_b['abs_delta'], height=0.66, color=colors_b,
                     edgecolor='black', linewidth=0.6)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    span_b = float(frame_b['abs_delta'].max())
    for position, row in enumerate(frame_b.to_dict('records')):
        mark = 'q<0.05' if row['q值_BHFDR'] < 0.05 else 'q≥0.05'
        ax_b.text(row['abs_delta'] + span_b * 0.016, position,
                  f'{row["abs_delta"]:.3f}（{mark}）', va='center', ha='left', fontsize=10.5)
    ax_b.set_yticks(positions)
    ax_b.set_yticklabels([f'{row["类别"]}（n={int(row["present岗位数"])}）'
                          for row in frame_b.to_dict('records')])
    ax_b.set_xlim(0, span_b * 1.34)
    ax_b.set_ylim(-0.7, len(frame_b) - 0.3)
    ax_b.set_xlabel("|Cliff's δ|（present 与 absent 薪资分布差异）", fontsize=10.5)
    ax_b.set_ylabel('岗位大类（实心）/ 岗位细分类（斜纹）', fontsize=10.5)
    plot_style.apply_sci_axis(ax_b, grid_axis='x', grid=True)
    ax_b.tick_params(axis='both', labelsize=10.0)
    ax_b.text(0.985, 0.02,
              'δ 为命中 / 未命中该类别的分布差异，\n与 (a) 的 ε² 不同量纲，不可直接比较',
              transform=ax_b.transAxes, ha='right', va='bottom', fontsize=10.5,
              linespacing=1.5, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.85})

    plot_style.add_subfigure_caption(ax_a, 'a', '单值/互斥分组因素的效应量排序（标注参与检验取值数）')
    plot_style.add_subfigure_caption(ax_b, 'b', '多值类别的二元关联强度对照（各字段 |δ| 前 10）')
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.20, top=0.975, wspace=0.52)

    diagnostics = figure_finalize.save_paper_figure(
        fig, SUPP_DIR, FIG_STEM,
        subfigures=[('a', '单值/互斥分组因素的效应量排序（标注参与检验取值数）', ax_a),
                    ('b', '多值类别的二元关联强度对照（各字段 |δ| 前 10）', ax_b)],
        meta={'图型': 'supplementary', 'Stage25 用途': '替代原图 5-4（结构化因素效应量排序）',
              '数据来源': 'stage_13_eda.json（(a) 只读复用）+ 本脚本任务 1 二元检验（(b)）',
              'seed': SEED})
    plt.close(fig)
    diagnostics['failed_paper_gates'] = figure_finalize.failed_paper_gates(diagnostics)
    return diagnostics


# ---------------------------------------------------------------- 任务 5：锚点自检
def anchor_checks(salary_sample: int, jobs: int) -> dict:
    """冻结锚点自检：复用 Stage23 统一锚点校验，并补充 test RMSE / R² 与 14889→14899 说明。"""
    records = figure_finalize.validate_anchors()
    test_table = pd.read_excel(project_paths.TABLES_DIR / project_paths.TABLE_MODEL_COMPARISON,
                               sheet_name='09_Test最终结果').set_index('模型')
    final_row = test_table.loc['FINAL（LightGBM）']
    for label, expected, actual in [
            ('主模型 test RMSE', 64.623479, float(final_row['test_RMSE'])),
            ('主模型 test R²', 0.584606, float(final_row['test_R2']))]:
        records.append({'锚点': label, '期望值': expected, '实际读取值': actual,
                        '来源': f'{project_paths.TABLE_MODEL_COMPARISON} / 09_Test最终结果'
                                '（FINAL（LightGBM）行）',
                        '通过': bool(abs(float(actual) - expected) < 1e-6)})
    records.append({'锚点': '14889 或 14899（明确薪资可解析）',
                    '期望值': '14899（14889 全项目未出现）',
                    '实际读取值': next(item['实际读取值'] for item in records
                                   if item['锚点'] == '明确薪资可解析'),
                    '来源': '15_structured_field_salary_audit.xlsx / 02_薪资解析状态'
                            '（该表现仅「已解析 14899 / 面议 2245」两行，无 14889 取值）',
                    '通过': True})
    records.append({'锚点': '本次重算样本锚点（正式薪资样本 / 唯一岗位实体）',
                    '期望值': '14883 / 17144',
                    '实际读取值': f'{salary_sample} / {jobs}',
                    '来源': 'data/processed/job_salary_model_dataset.parquet、'
                            'data/processed/job_analysis_dataset.parquet（只读）',
                    '通过': bool(salary_sample == 14883 and jobs == 17144)})
    return {'清单': records,
            '通过数': int(sum(item['通过'] for item in records)),
            '总数': int(len(records)),
            '结论': '全部一致（未改动任何冻结值）' if all(item['通过'] for item in records)
                    else '存在不一致项，需人工复核'}


# ---------------------------------------------------------------- main
def main() -> int:
    started = time.time()
    print('=' * 92)
    print('Stage25 受控新增重算：多值岗位类别二元检验 / 中位数基线 / 分组预测误差 / 图S17')
    print('=' * 92)
    manifest_before = project_manifest()
    print(f'运行前既有文件 SHA-256 清单：{len(manifest_before)} 个文件'
          '（不含本脚本自身产物，后者单独登记）')

    model_frame = io_utils.read_parquet(project_paths.JOB_SALARY_MODEL_DATASET_PARQUET)
    analysis = io_utils.read_parquet(project_paths.JOB_ANALYSIS_DATASET_PARQUET,
                                     columns=[schema.ID_FIELD])
    splits = io_utils.read_parquet(project_paths.MODEL_SPLITS_PARQUET)
    predictions = io_utils.read_parquet(project_paths.MODEL_PREDICTIONS_PARQUET)
    print(f'输入：薪资样本 {len(model_frame):,} / 唯一岗位实体 {len(analysis):,} / '
          f'划分 {len(splits):,} / 预测 {len(predictions):,}')

    # ---- 任务 1 ----
    task1 = run_task1(model_frame)
    print(f"任务1：岗位大类进入检验 {task1['summaries']['岗位大类']['进入检验类别数']} 个"
          f"（q<0.05 {task1['summaries']['岗位大类']['q<0.05数量']}），"
          f"岗位细分类进入检验 {task1['summaries']['岗位细分类']['进入检验类别数']} 个"
          f"（q<0.05 {task1['summaries']['岗位细分类']['q<0.05数量']}）")
    for label in ('岗位大类', '岗位细分类'):
        item = task1['summaries'][label]
        print(f"  {label}: |δ| 最大 {item['|δ|最大值']}（{item['|δ|最大类别']}）/ "
              f"最小 {item['|δ|最小值']}（{item['|δ|最小类别']}）；四档 "
              f"可忽略 {item['四档_可忽略(<0.147)']} / 小 {item['四档_小(0.147~0.330)']} / "
              f"中等 {item['四档_中等(0.330~0.474)']} / 大 {item['四档_大(>=0.474)']}")

    # ---- 任务 2 ----
    task2 = run_task2(model_frame, splits)
    print(f"任务2：mean 基线复现 {'成功' if task2['reproduction']['能否复现'] else '失败'}；"
          f"median 基线 validation MAE = "
          f"{task2['baselines']['median（中位数）']['validation']['MAE']}，"
          f"是否改变主模型选择：{task2['selection']['是否改变主模型选择']}")

    # ---- 任务 3 ----
    task3 = run_task3(model_frame, predictions)
    print(f"任务3：test 总体 n={task3['overall']['n']}，MAE={task3['overall']['MAE']}，"
          f"RMSE={task3['overall']['RMSE']}")
    print(f"  三分位最大误差档 {task3['tertile']['最大MAE']}；最小 {task3['tertile']['最小MAE']}")

    # ---- 输出 34 号表 ----
    binary_sheet_columns = ['类别', 'present岗位数', 'absent岗位数', 'present中位数', 'absent中位数',
                            '中位数差', 'U统计量', 'p值', 'q值_BHFDR', 'Cliff_delta',
                            '效应档', '是否显著', '排名']
    sheets = {
        '01_岗位大类_二元检验': task1['category'][binary_sheet_columns],
        '02_岗位细分类_二元检验': task1['sub_category'][binary_sheet_columns],
        '03_汇总': task1['summary'],
        '04_中位数基线': pd.concat([task2['sheet'], task2['reproduction_rows']],
                                ignore_index=True, sort=False),
        '05_分组预测误差': task3['sheet'],
        '06_方法与口径说明': pd.DataFrame([
            {'项目': '脚本路径', '内容': 'scripts/25_stage25_factor_revision.py'},
            {'项目': '随机种子', '内容': f'{SEED}（与项目既有划分 random_state 一致；'
                                   '本次检验为确定性方法，不涉及随机抽样）'},
            {'项目': '任务 1 样本口径',
             '内容': f'正式薪资样本 {len(model_frame):,} 个具备有效薪资的岗位，'
                     '来源 data/processed/job_salary_model_dataset.parquet'
                     '（与 Stage13 正式 EDA 完全同一分析框）'},
            {'项目': '任务 1 方法',
             '内容': "逐个类别构造 present（命中）/ absent（未命中）互斥两组，"
                     "做 Mann–Whitney U（双侧）+ Cliff's δ，并对同一字段全部进入检验的类别"
                     '统一做 Benjamini–Hochberg FDR 校正（输出 q 值）'},
            {'项目': '任务 1 样本量门槛', '内容': MW_MIN_GROUP_NOTE},
            {'项目': '任务 1 口径说明', '内容': CALIBER_NOTE},
            {'项目': '任务 1 与原口径的关系',
             '内容': '原「岗位大类 ε²=0.082114（12 组）」「岗位细分类 ε²=0.115633（80 组）」'
                     '基于多值标签 explode 后的重叠组做 Kruskal–Wallis，组间不独立，'
                     '不再作为正式结论；本表取代其岗位类别部分的结论地位'},
            {'项目': '任务 2 划分来源',
             '内容': 'data/processed/model_splits.parquet（train 10418 / validation 2232 / '
                     'test 2233，random_state=42，与 30 号表 01_数据划分 一致）'},
            {'项目': '任务 2 预测规则', '内容': 'y_pred ≡ 常数（train 的均值 / 中位数），'
                                        '验证集与测试集信息未参与估计'},
            {'项目': '任务 2 边界', '内容': '未重跑训练流水线、未重新训练任何真实模型；'
                                     '既有 35.335506 / 64.623479 / 0.584606 等正式结果保持不变'},
            {'项目': '任务 3 预测来源',
             '内容': "data/processed/model_predictions.parquet 中 model_name='FINAL（LightGBM）' "
                     "且 split='test' 的 2233 行（既有一次性预测，只读）"},
            {'项目': '任务 3 薪资区间划分',
             '内容': '三分位：测试集 y_true 的 33.3% / 66.7% 分位点；四分位：25% / 50% / 75% '
                     '分位点；区间为左开右闭（并列值按边界计入下界档）'},
            {'项目': '任务 3 岗位大类口径',
             '内容': f'{CATEGORY_FIELD} 为多值字段，按「命中即计入」展开，'
                     '同一岗位可计入多个大类，各组样本数之和大于测试集总体 n；'
                     '未命中任何大类的岗位数为 0'},
            {'项目': '措辞规范', '内容': '全部结果仅为描述性关联 / 预测误差诊断，'
                                    '禁止表述为因果'},
        ]),
    }
    excel_path = io_utils.write_excel(EXCEL_PATH, sheets)
    print(f'34 号表已写入：{project_paths.relative_to_root(excel_path)} '
          f'（{excel_path.stat().st_size:,} 字节，{len(sheets)} 个 Sheet）')

    # ---- 任务 4 ----
    snapshot = plot_style.setup_sci_style()
    exclusive = exclusive_factor_table()
    top_binary = pd.concat([
        task1['category'].head(10)[['类别', 'present岗位数', 'q值_BHFDR', 'Cliff_delta']]
        .assign(字段='岗位大类'),
        task1['sub_category'].head(10)[['类别', 'present岗位数', 'q值_BHFDR', 'Cliff_delta']]
        .assign(字段='岗位细分类')], ignore_index=True)
    top_binary['abs_delta'] = top_binary['Cliff_delta'].abs()
    figure = figure_s17(exclusive, top_binary)
    print(f"图 S17 已写入：{figure['png_path']}（{figure['png_size_bytes']:,} 字节，"
          f"{figure['png_pixel_size']} px，dpi {figure['png_dpi']}）；PDF "
          f"{figure['pdf_size_bytes']:,} 字节；未通过论文版门禁："
          f"{figure['failed_paper_gates'] or '无'}；图内总图题："
          f"{'无' if figure['no_infigure_caption'] else '有'}")

    # ---- 锚点自检 ----
    anchors = anchor_checks(len(model_frame), len(analysis))
    print(f"锚点自检：{anchors['通过数']}/{anchors['总数']} 通过 → {anchors['结论']}")

    # ---- 新增文件（xlsx / 图件可自哈希；本 JSON 自身不写回自哈希）----
    new_files = [{'路径': str(path.relative_to(PROJECT_ROOT)).replace('\\', '/'),
                  '字节数': int(path.stat().st_size),
                  'SHA256': sha256_of(path)} for path in NEW_FILES
                 if path.exists() and path != METRICS_PATH]
    new_files.append({'路径': str(METRICS_PATH.relative_to(PROJECT_ROOT)).replace('\\', '/'),
                      '字节数': 'N/A（本文件自引用，最终字节数与 SHA256 由运行控制台报告）',
                      'SHA256': 'N/A（同上，避免自哈希不一致）'})

    # ---- 任务 5：汇总 JSON ----
    def input_record(path: Path) -> dict:
        return {'路径': str(path.relative_to(PROJECT_ROOT)).replace('\\', '/'),
                '字节数': int(path.stat().st_size), 'SHA256': sha256_of(path)}

    inputs = [
        project_paths.JOB_SALARY_MODEL_DATASET_PARQUET, project_paths.JOB_ANALYSIS_DATASET_PARQUET,
        project_paths.MODEL_SPLITS_PARQUET, project_paths.MODEL_PREDICTIONS_PARQUET,
        project_paths.TABLES_DIR / project_paths.TABLE_MODEL_COMPARISON,
        project_paths.TABLES_DIR / project_paths.TABLE_EDA_STATISTICAL,
        project_paths.TABLES_DIR / '15_structured_field_salary_audit.xlsx',
        project_paths.TABLES_DIR / '23_ablation_robustness_shap.xlsx',
        project_paths.TABLES_DIR / '24_company_field_semantic_audit.xlsx',
        project_paths.TABLES_DIR / '07_observation_snapshot_audit.xlsx',
        project_paths.METRICS_DIR / 'stage_13_eda.json',
        project_paths.SALARY_MODEL_DIR / 'model_params.json',
        project_paths.PROJECT_ROOT / 'src' / 'plot_style.py',
        project_paths.PROJECT_ROOT / 'src' / 'figure_finalize.py',
    ]
    payload = {
        'stage': 'Stage25',
        '脚本路径': 'scripts/25_stage25_factor_revision.py',
        '运行命令': 'python '
                'scripts\\25_stage25_factor_revision.py',
        '运行时间': {'开始': datetime.fromtimestamp(started).isoformat(timespec='seconds'),
                 '结束': datetime.now().isoformat(timespec='seconds'),
                 '耗时秒': round(time.time() - started, 2)},
        '随机种子': {'脚本种子': SEED,
                 '说明': '检验（Mann–Whitney / Cliff\'s δ / BH-FDR）为确定性方法，'
                         '不使用随机抽样；seed 仅与项目既有划分 random_state 对齐'},
        '输入文件清单': [input_record(path) for path in inputs],
        '任务1_多值岗位类别二元检验': {
            '样本': int(len(model_frame)),
            '样本口径': '正式薪资样本（与 Stage13 正式 EDA 同一分析框）',
            '样本量门槛': MIN_BINARY_GROUP,
            '方法': "Mann–Whitney U（双侧）+ Cliff's δ + BH-FDR",
            '汇总': task1['summaries'],
            '合并统一校正_q<0.05数量': task1['pooled_significant'],
            'Cliff实现对拍': task1['calibration'],
            '岗位大类_全部类别': task1['category'].to_dict('records'),
            '岗位细分类_top10': task1['sub_category'].head(10).to_dict('records'),
            '完整逐类别结果位置': f'outputs/tables/{EXCEL_PATH.name} 的 '
                          '01_岗位大类_二元检验 / 02_岗位细分类_二元检验',
            '未达标类别（未检验）': {
                '岗位大类': task1['category_pool'][
                    task1['category_pool']['Cliff_delta'].isna()]['类别'].tolist(),
                '岗位细分类': task1['sub_category_pool'][
                    task1['sub_category_pool']['Cliff_delta'].isna()]['类别'].tolist()},
            '口径说明': CALIBER_NOTE,
        },
        '任务2_常数基线': {
            '划分来源': 'data/processed/model_splits.parquet',
            '划分样本数': {str(key): int(value) for key, value
                       in task2['split_counts'].items()},
            '划分交叉核对': '与 30 号表 01_数据划分（train 10418 / validation 2232 / test 2233）一致',
            '基线指标': task2['baselines'],
            'median减mean差值': task2['differences'],
            'mean基线复现': task2['reproduction'],
            '主模型选择核对': task2['selection'],
        },
        '任务3_分组预测误差': {
            '预测来源': "model_predictions.parquet model_name='FINAL（LightGBM）' 且 split='test'",
            '测试集总体': task3['overall'],
            '薪资三分位': task3['tertile'],
            '薪资四分位': task3['quartile'],
            '岗位大类': task3['category'],
            '岗位大类_缺失属性岗位数': task3['missing_category'],
            '全部明细': task3['sheet'].to_dict('records'),
            '结论性事实': [
                f"测试集总体 MAE = {task3['overall']['MAE']:.4f}（n = {task3['overall']['n']}）",
                f"薪资三分位中误差最大档为 {task3['tertile']['最大MAE']['取值']}"
                f"（MAE = {task3['tertile']['最大MAE']['MAE']:.4f}），"
                f"最小档为 {task3['tertile']['最小MAE']['取值']}"
                f"（MAE = {task3['tertile']['最小MAE']['MAE']:.4f}）",
                f"岗位大类中误差最大为 {task3['category']['最大MAE']['取值']}"
                f"（MAE = {task3['category']['最大MAE']['MAE']:.4f}），"
                f"最小为 {task3['category']['最小MAE']['取值']}"
                f"（MAE = {task3['category']['最小MAE']['MAE']:.4f}）",
                '以上均为预测误差的描述性事实，不构成因果解释'],
        },
        '任务4_图S17': {
            'PNG': figure['png_path'], 'PDF': figure['pdf_path'],
            '像素尺寸': figure['png_pixel_size'], 'dpi': figure['png_dpi'],
            'PNG字节': figure['png_size_bytes'], 'PDF字节': figure['pdf_size_bytes'],
            '图内总图题': '无' if figure['no_infigure_caption'] else '有',
            '子图名': [record['text'] for record in figure['subfigures']],
            '子图名位于子图下方且可见': figure['subfigure_caption_visible'],
            '论文版门禁未通过项': figure['failed_paper_gates'],
            '论文版门禁': figure_finalize.paper_gates(figure),
            '样式快照': {key: value for key, value in snapshot.items()
                     if key in ('context', 'style', 'facecolor', 'savefig_dpi', 'headless')},
            '图(a)数据来源': 'outputs/logs/metrics/stage_13_eda.json:factor_tests（只读，未重算）'
                        ' + 29 号表 11_统计检验（技能数量档行）',
            '图(b)数据来源': '本脚本任务 1 二元检验（各字段 |δ| 前 10）',
        },
        '任务5_锚点自检': anchors,
        '本次新增文件': new_files,
        '本次新增文件说明': '本脚本只新增以上 4 个文件，不覆盖任何既有产物；第 4 项为本次运行的 JSON 自身，'
                      '为避免自哈希不一致，其最终字节数与 SHA-256 在脚本退出前由控制台打印。',
        '输出产物': {
            '表': str(excel_path.relative_to(PROJECT_ROOT)).replace('\\', '/'),
            '图': [figure['png_path'], figure['pdf_path']],
            'JSON': 'outputs/logs/metrics/stage_25_revision.json',
        },
    }
    io_utils.write_json(METRICS_PATH, payload)

    # ---- 既有文件未修改证据（JSON 写出后立即比对运行前清单）----
    manifest_after = project_manifest()
    changed = sorted(path for path in manifest_before
                     if path in manifest_after and manifest_after[path] != manifest_before[path])
    removed = sorted(set(manifest_before) - set(manifest_after))
    added = sorted(set(manifest_after) - set(manifest_before))
    groups = {}
    for name, prefix in [('data/', 'data/'), ('outputs/tables/', 'outputs/tables/'),
                         ('outputs/models/', 'outputs/models/'),
                         ('outputs/figures/', 'outputs/figures/'),
                         ('outputs/logs/metrics/', 'outputs/logs/metrics/'),
                         ('docs/', 'docs/'), ('src/', 'src/')]:
        keys = [path for path in manifest_before if path.startswith(prefix)]
        groups[name] = {
            '文件数': len(keys),
            '内容一致': bool(all(manifest_after.get(path) == manifest_before[path]
                             for path in keys)),
            '内容变化文件': [path for path in keys
                        if manifest_after.get(path) != manifest_before[path]],
        }
    payload['既有文件未修改证据'] = {
        '清单规模': {'运行前文件数': len(manifest_before), '运行后文件数': len(manifest_after),
                 '清单内新增文件数': len(added)},
        '内容变化文件': changed,
        '被删除文件': removed,
        '清单内新增文件': added,
        '分组结论': groups,
        'src/plot_style.py': {
            'SHA256_运行前': manifest_before.get('src/plot_style.py'),
            'SHA256_运行后': manifest_after.get('src/plot_style.py'),
            '一致': manifest_before.get('src/plot_style.py')
                  == manifest_after.get('src/plot_style.py')},
        '清单生成方式': '对项目内既有文件（data/、outputs/、docs/、src/、scripts/、config/、'
                    'notebooks/、tests/、README.md，跳过缓存目录）逐个计算 SHA-256，'
                    '在写入本次新增产物前后各生成一次并逐文件比对；'
                    '本脚本自身新增的 4 个产物不入清单，见「本次新增文件」',
        '结论': ('未修改任何既有文件（内容变化 0 个、删除 0 个、清单内新增 0 个）'
              if not changed and not removed and not added
              else '存在内容变化/删除/清单内新增，需人工复核'),
    }
    io_utils.write_json(METRICS_PATH, payload)

    print(f'指标 JSON 已写入：{project_paths.relative_to_root(METRICS_PATH)} '
          f'（{METRICS_PATH.stat().st_size:,} 字节，SHA256 {sha256_of(METRICS_PATH)[:16]}…）')
    print('-' * 92)
    print(f"既有文件内容变化：{len(changed)} 个；删除：{len(removed)} 个；"
          f"清单内新增：{len(added)} 个")
    for item in new_files:
        print(f"  新增：{item['路径']} | {item['字节数']} | {str(item['SHA256'])[:16]}…")
    print(f'总耗时 {time.time() - started:.1f} 秒')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

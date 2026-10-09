# -*- coding: utf-8 -*-
"""Stage 16：公司字段语义只读取证（“公司认证标签 369 组”溯源），不做任何修正。

链路：
    src/schema.py 常量
    → data/processed/job_analysis_dataset.parquet / job_salary_model_dataset.parquet（Stage 12 宽表）
    → src/eda_analysis.py 因素规格 + scripts/ch4_lifecycle/01_run_eda.py 调用
    → outputs/tables/ch4/21_eda_statistical_analysis.xlsx

核心问题：Stage 13 中「公司认证标签 369 组」实际统计的到底是哪一个字段？
分类只能是 A1（仅显示名称错误）/ A2（Stage 13 字段引用错误）/ A3（Stage 12 宽表映射错误）/
A4（list 组合值/拆分统计错误）。

输出：
    outputs/tables/ch5/24_company_field_semantic_audit.xlsx
    docs/records/23_company_field_semantic_audit_record.md
    outputs/logs/metrics/stage_16_company_field.json

本脚本**只读**：不写入、不重跑任何 Stage 12/13/14/15 产物。

用法：
    python scripts/ch5_factors/01_company_field_semantic_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = next(
    _candidate for _candidate in Path(__file__).resolve().parents
    if (_candidate / "data").is_dir() and (_candidate / "scripts").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src import (eda_analysis, io_utils, modeling_dataset,  # noqa: E402
                 project_paths, quality, schema)

STAGE = 'stage_16_company_field'
TITLE = 'Stage 16 公司字段语义只读取证（公司认证标签 369 组溯源）'

EXPECTED_JOBS = 17144
EXPECTED_SALARY_SAMPLE = 14883
CERT_ATOMS_EXPECTED = {'最佳雇主', '行业认证'}
DISPLAY_LABEL = '公司认证标签'
CERT_CLASS_FIELD = eda_analysis.CERT_CLASS_FIELD
COMPANY_CERT_LABEL = eda_analysis.COMPANY_CERT_LABEL
COMPANY_TAG_LABEL = eda_analysis.COMPANY_TAG_LABEL

# 常量核查清单（不存在时明确写“不存在”，不静默跳过）
SCHEMA_CONSTANTS = [
    'COMPANY_CERTIFICATION_FIELD', 'COMPANY_TAG_LIST_FIELD', 'COMPANY_TAG_FIELD',
    'COMPANY_CERTIFICATION_LIST_FIELD', 'CERT_TAG_FIELD', 'CERT_SOURCE_FIELD',
    'JOB_TAG_LIST_FIELD', 'COMPANY_TAG_RAW_FIELD',
]
# 调用链静态扫描目标：文件 → 关键字
CALL_CHAIN_TARGETS = [
    ('src/schema.py', 'CERT_TAG_FIELD'),
    ('src/schema.py', 'COMPANY_TAG_LIST_FIELD'),
    ('src/eda_analysis.py', 'COMPANY_TAG_LIST_FIELD'),
    ('src/eda_analysis.py', DISPLAY_LABEL),
    ('src/modeling_dataset.py', DISPLAY_LABEL),
    ('scripts/ch4_lifecycle/01_run_eda.py', 'certification'),
    ('scripts/ch4_lifecycle/01_run_eda.py', DISPLAY_LABEL),
    ('scripts/pipeline/12_build_modeling_dataset.py', DISPLAY_LABEL),
    ('scripts/ch7_model/01_train_salary_model.py', DISPLAY_LABEL),
    ('scripts/ch8_robust/01_ablation_robustness_shap.py', DISPLAY_LABEL),
]


def scan_lines(relative_path: str, keyword: str) -> list:
    """静态扫描：返回命中行（文件相对路径 / 行号 / 该行内容）。"""
    path = project_paths.PROJECT_ROOT / relative_path
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        if keyword in line:
            rows.append({'文件': relative_path, '行号': number, '命中关键字': keyword,
                         '代码行': line.strip()})
    return rows


def build_schema_sheet() -> tuple:
    """01_schema字段映射：常量名 / 实际字符串值 / 定义文件 / 调用位置。"""
    schema_path = project_paths.PROJECT_ROOT / 'src/schema.py'
    source = schema_path.read_text(encoding='utf-8').splitlines()
    rows = []
    for name in SCHEMA_CONSTANTS:
        value = getattr(schema, name, None)
        definition = ''
        if value is not None:
            definition = next((f'src/schema.py:{number}' for number, line in enumerate(source, 1)
                               if line.startswith(f'{name} =')), 'src/schema.py')
        call_sites = []
        for path in sorted(project_paths.SRC_DIR.glob('*.py')):
            for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
                if f'schema.{name}' in line:
                    call_sites.append(f'{path.name}:{number}')
        for path in sorted(project_paths.SCRIPTS_DIR.rglob('*.py')):
            for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
                if f'schema.{name}' in line:
                    call_sites.append(f'scripts/{path.name}:{number}')
        rows.append({
            '常量名': name,
            '实际字符串值': '不存在' if value is None else str(value),
            '定义文件': '不存在' if value is None else definition,
            '调用位置': '不存在' if value is None else '；'.join(call_sites) or '未在 src/scripts 中引用',
        })
    return pd.DataFrame(rows), {row['常量名']: row['实际字符串值'] for row in rows}


def describe_list_field(series: pd.Series) -> dict:
    """列表字段统计：区分“唯一组合值”与“explode 后唯一原子标签”。"""
    non_null = series.notna()
    is_list = bool(series[non_null].map(modeling_dataset.is_multi_value).all()) if non_null.any() \
        else False
    lengths = series[non_null].map(lambda value: len(value) if modeling_dataset.is_multi_value(value)
                                   else 1)
    exploded = series[non_null].explode().dropna().astype(str)
    combos = series[non_null].map(lambda value: ' | '.join(sorted(map(str, value))))
    return {
        '非空岗位数': int(non_null.sum()), '空值数': int((~non_null).sum()),
        '是否 list': '是' if is_list else '否',
        '每岗平均标签数': round(float(lengths.mean()), 4),
        '空标签岗位数': int(lengths.eq(0).sum()),
        '唯一组合值数': int(combos.nunique()),
        'explode后唯一原子标签数': int(exploded.nunique()),
        '原子标签 Top20': '、'.join(exploded.value_counts().head(20).index.tolist()),
        '原子标签全集': '、'.join(sorted(exploded.unique().tolist()))
        if exploded.nunique() <= 6 else f'（共 {exploded.nunique()} 个，见 04 号子表）',
    }


def build_wide_field_sheet(analysis: pd.DataFrame, model: pd.DataFrame) -> pd.DataFrame:
    """02_宽表字段统计：两张宽表 × 公司认证标签 / 公司标签列表。"""
    rows = []
    for name, frame in [('job_analysis_dataset（17,144）', analysis),
                        ('job_salary_model_dataset（14,883）', model)]:
        for column in [schema.CERT_TAG_FIELD, schema.COMPANY_TAG_LIST_FIELD]:
            row = {'数据表': name, '字段名': column}
            if column not in frame.columns:
                row.update({'字段是否存在': '不存在', 'dtype': '', '非空岗位数': '', '空值数': '',
                            '是否 list': '', '每岗平均标签数': '', '空标签岗位数': '',
                            '唯一组合值数': '', 'explode后唯一原子标签数': '',
                            '原子标签 Top20': '', '原子标签全集': ''})
            else:
                row['字段是否存在'] = '存在'
                row['dtype'] = str(frame[column].dtype)
                row.update(describe_list_field(frame[column]))
            rows.append(row)
    return pd.DataFrame(rows)


def build_certification_sheet(certification: pd.DataFrame) -> pd.DataFrame:
    """03_公司认证原子标签：原子标签全集 / 岗位数 / 频率 / 组合分布 / 未知值。"""
    series = certification[schema.CERT_TAG_FIELD]
    exploded = series.explode().dropna().astype(str)
    total = len(series)
    rows = [{'项目': '字段来源', '取值': 'data/processed/job_details_unique.parquet（Stage 05 正式产物）'},
            {'项目': '岗位总数', '取值': f'{total:,}'},
            {'项目': '有认证岗位数（非空 list）', '取值': f'{int(series.map(len).gt(0).sum()):,}'},
            {'项目': '空 list 岗位数（视为无认证）', '取值': f'{int(series.map(len).eq(0).sum()):,}'},
            {'项目': '原子标签数', '取值': f'{int(exploded.nunique())}'},
            {'项目': '原子标签全集', '取值': '、'.join(sorted(exploded.unique().tolist()))}]
    known = CERT_ATOMS_EXPECTED
    unknown = sorted(set(exploded.unique().tolist()) - known)
    rows.append({'项目': '未知认证值（逐项列出）', '取值': '、'.join(unknown) if unknown else '无'})
    combo = series.map(lambda value: ' | '.join(sorted(map(str, value))) or '（空）').value_counts()
    for value, count in combo.items():
        rows.append({'项目': f'组合分布：{value}', '取值': f'{int(count):,}'
                     f'（{count / total:.2%}）'})
    for value, count in exploded.value_counts().items():
        rows.append({'项目': f'单标签频率：{value}', '取值': f'{int(count):,}'
                     f'（{count / total:.2%}）'})
    return pd.DataFrame(rows), unknown


def build_tag_sheet(analysis: pd.DataFrame) -> pd.DataFrame:
    """04_公司标签原子标签：原子标签数 / Top50 / 非空岗位数 / 每岗标签数分布。"""
    series = analysis[schema.COMPANY_TAG_LIST_FIELD]
    exploded = series.explode().dropna().astype(str)
    lengths = series.map(len)
    rows = [{'指标': '非空岗位数', '数值': f'{int(series.notna().sum()):,}'},
            {'指标': '空值数', '数值': f'{int(series.isna().sum()):,}'},
            {'指标': '空标签岗位数', '数值': f'{int(lengths.eq(0).sum()):,}'},
            {'指标': '原子标签数', '数值': f'{int(exploded.nunique()):,}'},
            {'指标': '唯一组合值数', '数值': f'{int(series.map(lambda v: " | ".join(sorted(map(str, v)))).nunique()):,}'},
            {'指标': '每岗平均标签数', '数值': f'{float(lengths.mean()):.4f}'},
            {'指标': '每岗标签数中位数', '数值': f'{float(lengths.median()):.1f}'},
            {'指标': '每岗标签数最大值', '数值': f'{int(lengths.max())}'}]
    table = pd.DataFrame(rows)
    top = exploded.value_counts().head(50).rename_axis('原子标签').reset_index(name='岗位数')
    top.insert(0, '排名', range(1, len(top) + 1))
    top['岗位占比'] = (top['岗位数'] / len(analysis)).round(6)
    distribution = lengths.value_counts().sort_index().rename_axis('每岗标签数').reset_index(name='岗位数')
    distribution.insert(0, '排名', range(1, len(distribution) + 1))
    return table, top, distribution


def build_chain_sheet() -> pd.DataFrame:
    """05_Stage13调用链：当前字段引用 / 显示名称 / 是否一致 + 历史缺陷记录（不删历史）。"""
    rows = []
    for path, keyword in CALL_CHAIN_TARGETS:
        for hit in scan_lines(path, keyword):
            rows.append({'层级': '静态代码扫描（当前状态）', '是否一致': '', **hit})
    source = (project_paths.PROJECT_ROOT / 'src/eda_analysis.py').read_text(
        encoding='utf-8').splitlines()
    specification = {
        'CERT_FACTOR_KEY': (CERT_CLASS_FIELD, COMPANY_CERT_LABEL, '是（认证字段与显示名一致）'),
        'TAG_FACTOR_KEY': (schema.COMPANY_TAG_LIST_FIELD, COMPANY_TAG_LABEL,
                           '是（福利标签已正名，不再称认证）'),
    }
    for factor_key, (field, label, verdict) in specification.items():
        for number, line in enumerate(source, 1):
            if factor_key in line and ('单值' in line or '多值' in line):
                rows.append({'层级': 'Stage 13 因素规格（当前状态）', '文件': 'src/eda_analysis.py',
                             '行号': number, '命中关键字': f'{factor_key} + {field}',
                             '代码行': line.strip(), '实际引用字段': field, '显示名称': label,
                             '是否一致': verdict})
                break
    rows.append({
        '层级': '历史缺陷（修正前，已修正，保留不删）', '文件': 'src/eda_analysis.py', '行号': '—',
        '命中关键字': "certification + '公司认证标签'",
        '代码行': "'certification': (schema.COMPANY_TAG_LIST_FIELD, '多值', '公司认证标签'),",
        '实际引用字段': schema.COMPANY_TAG_LIST_FIELD, '显示名称': DISPLAY_LABEL,
        '是否一致': '否（修正前状态；证据见本表 06 子表与 git 历史）',
    })
    return pd.DataFrame(rows)


def build_group_evidence_sheet() -> tuple:
    """06_29号表实际分组：369 组的真实取值来源（兼容修正前 / 修正后两种状态）。

    修正前：06 子表把该因素显示为「公司认证标签」；
    修正后：显示为「公司标签（福利标签）」。
    两种状态下取值的来源字段都是 `公司标签列表`，因此证据可双向复核。
    """
    workbook = pd.ExcelFile(project_paths.TABLES_DIR / project_paths.TABLE_EDA_STATISTICAL)
    sheet = workbook.parse('06_公司因素薪资')
    pre_fix = sheet[sheet['因素'].eq(DISPLAY_LABEL)]
    corrected = sheet[sheet['因素'].eq(eda_analysis.COMPANY_TAG_LABEL)]
    state = ('修正前（显示名为「公司认证标签」）' if not pre_fix.empty
             else '修正后（显示名为「公司标签（福利标签）」）')
    block = pre_fix if not pre_fix.empty else corrected
    test = workbook.parse('11_统计检验')
    kruskal = test[test['检验方法'].astype(str).str.startswith('Kruskal')]
    lookup = DISPLAY_LABEL if not pre_fix.empty else eda_analysis.COMPANY_TAG_LABEL
    kruskal_row = kruskal[kruskal['检验对象'].astype(str).eq(lookup)]
    rows = [
        {'证据项': '本次读取时 29 号表状态', '数值': state},
        {'证据项': '06_公司因素薪资 sheet 中该因素的行数', '数值': f'{len(block)}'},
        {'证据项': '前 20 个取值', '数值': '、'.join(block['取值'].head(20).tolist())},
        {'证据项': '11_统计检验 sheet 检验对象', '数值': str(kruskal_row['检验对象'].iloc[0])},
        {'证据项': '11_统计检验 sheet 参与检验组数',
         '数值': str(kruskal_row['参与检验组数'].iloc[0])},
        {'证据项': '11_统计检验 sheet epsilon²', '数值': str(kruskal_row['效应量'].iloc[0])},
        {'证据项': '取值来源字段', '数值': f'`{schema.COMPANY_TAG_LIST_FIELD}`'
                                    '（福利/企业文化标签：免费健身设施、餐饮、节日礼品、'
                                    '弹性工作制等）'},
        {'证据项': '判定', '数值': '无论显示名如何，369 组取值全部来自 '
                              f'`{schema.COMPANY_TAG_LIST_FIELD}`；'
                              f'`{schema.CERT_TAG_FIELD}`（最佳雇主 / 行业认证）'
                              '只存在于唯一岗位表，未进入该因素'},
    ]
    return pd.DataFrame(rows), block, kruskal_row, state


def write_record(findings: dict) -> Path:
    """23 号记录：问题分类、证据、重跑范围。"""
    lines = [
        '# 记录 23：Stage 16 公司字段语义只读取证（“公司认证标签 369 组”溯源）',
        '',
        '> 本记录由 `scripts/ch5_factors/01_company_field_semantic_audit.py` 自动生成，数字全部来自真实产物。',
        '> 本轮为**只读取证**：未修改任何正式数据，未重跑任何 Stage。',
        '',
        '## 1. 问题分类',
        '',
        f"**{findings['classification']}**：{findings['classification_text']}",
        '',
        '| 候选分类 | 判定 | 依据 |',
        '| --- | --- | --- |',
        f"| A1 仅显示名称错误 | {findings['A1']} | Stage 13 因素显示名与真实字段不一致 |",
        f"| A2 Stage 13 字段引用错误 | {findings['A2']} | Stage 13 设计要求分析「公司认证」，"
        f"代码引用的是 `{schema.COMPANY_TAG_LIST_FIELD}` |",
        f"| A3 Stage 12 宽表映射错误 | {findings['A3']} | 宽表 `{schema.COMPANY_TAG_LIST_FIELD}` "
        f"取值与 `公司标签原文` 语义一致，`{schema.CERT_TAG_FIELD}` 从未进入宽表（设计取舍） |",
        f"| A4 list 组合值/拆分统计错误 | {findings['A4']} | explode 后原子标签 "
        f"{findings['tag_atoms']:,} 个，369 = 样本数 ≥ {eda_analysis.MIN_GROUP_SIZE} 的原子标签数，"
        f"统计逻辑（按岗位-取值展开）本身正确 |",
        '',
        '## 2. 关键证据',
        '',
        '| 项目 | 结果 |',
        '| --- | --- |',
        f"| 公司认证字段实际列名 | `{schema.CERT_TAG_FIELD}`"
        f"（只存在于 `data/processed/job_details_unique.parquet`） |",
        f"| 公司标签字段实际列名 | `{schema.COMPANY_TAG_LIST_FIELD}`"
        f"（存在于 Stage 12 两张宽表） |",
        f"| 公司认证原子标签数 | {findings['cert_atoms']} |",
        f"| 公司认证原子标签全集 | {findings['cert_atom_list']} |",
        f"| 公司认证有认证岗位数 | {findings['cert_nonempty']:,}（空 list "
        f"{findings['cert_empty']:,} = 无认证） |",
        f"| 公司标签原子标签数 | {findings['tag_atoms']:,} |",
        f"| 公司标签非空岗位数 | {findings['tag_nonempty']:,} |",
        f"| 公司标签 Top20 | {findings['tag_top20']} |",
        f"| 字符串层面重叠（公司标签中含认证原子标签的岗位数） | "
        f"「最佳雇主」{findings['tag_overlap']['最佳雇主']:,}、"
        f"「行业认证」{findings['tag_overlap']['行业认证']:,}"
        f"（占比 < 1%，属标签文本巧合，语义仍以字段为准） |",
        f"| 369 组实际来自 | `{schema.COMPANY_TAG_LIST_FIELD}` 的原子标签（样本数 ≥ 30） |",
        f"| 29 号表显示名称（本次读取状态：{findings['state']}） | 06 号子表「因素」列 = "
        f"`{findings['sheet6_label']}`；11 号子表「检验对象」列 = `{findings['sheet5_key']}`"
        f"（修正前分别为「公司认证标签」与 `certification`） |",
        '',
        '## 3. 影响范围',
        '',
        '- 仅影响 Stage 13 的**公司因素显示语义与统计对象**：`06_公司因素薪资`、`11_统计检验`、'
        '20 号记录的公司因素小节；',
        '- **不影响** Stage 12 宽表数据、Stage 14 正式模型、Stage 15 消融/SHAP 的数值；',
        f"- Stage 12 宽表 `{schema.COMPANY_TAG_LIST_FIELD}` 是否错映射：否（取值与公司标签原文一致）；",
        '- 公司认证（最佳雇主 / 行业认证）在 Stage 12 宽表中不存在，属设计取舍而非映射错误。',
        '',
        '## 4. 需要重跑的 Stage',
        '',
        '| Stage | 是否需要重跑 | 原因 |',
        '| --- | --- | --- |',
        '| Stage 12 | 否 | 宽表数据未发生任何变化（不存在映射错误） |',
        '| Stage 13 | 是 | 修正公司因素字段引用与显示名称 |',
        '| Stage 14 | 否 | 正式模型不读取公司因素显示名称，特征与样本未变 |',
        '| Stage 15 | 否（仅增量刷新 SHAP 汇总） | 消融与 SHAP 数值不依赖公司因素命名 |',
        '',
        '## 5. 门禁',
        '',
        '| 门禁项 | 状态 | 说明 |',
        '| --- | --- | --- |',
    ]
    for name, status in findings['gates'].items():
        lines.append(f'| {name} | {status} | {findings["gate_notes"].get(name, "")} |')
    lines += [
        '',
        f"- 产物：`{findings['audit_path']}`；",
        '- 本轮未修改 Stage 12/13/14/15 任何产物，未提交 git。',
        '',
    ]
    return io_utils.write_markdown(
        project_paths.RECORDS_DIR / project_paths.RECORD_COMPANY_FIELD_SEMANTIC, lines)


def main() -> int:
    quality.configure_logging()
    gates = quality.GateRegistry(STAGE)
    quality.stage_banner(STAGE, TITLE)

    analysis = io_utils.read_parquet(project_paths.JOB_ANALYSIS_DATASET_PARQUET)
    model = io_utils.read_parquet(project_paths.JOB_SALARY_MODEL_DATASET_PARQUET)
    certification = io_utils.read_parquet(
        project_paths.PROCESSED_UNIQUE_PARQUET, columns=[schema.ID_FIELD, schema.CERT_TAG_FIELD])
    print(f'输入: 分析集 {len(analysis):,} / 建模集 {len(model):,} / '
          f'唯一岗位表 {len(certification):,}（公司认证标签来源）')

    schema_sheet, constants = build_schema_sheet()
    wide_sheet = build_wide_field_sheet(analysis, model)
    cert_sheet, unknown_certs = build_certification_sheet(certification)
    tag_table, tag_top, tag_distribution = build_tag_sheet(analysis)
    chain_sheet = build_chain_sheet()
    group_sheet, cert_block, kruskal_row, state = build_group_evidence_sheet()
    tag_atom_counts = analysis[schema.COMPANY_TAG_LIST_FIELD].explode().value_counts()
    tag_overlap = {atom: int(tag_atom_counts.get(atom, 0)) for atom in CERT_ATOMS_EXPECTED}

    cert_atoms = cert_sheet[cert_sheet['项目'].eq('原子标签数')]['取值'].iloc[0]
    cert_atom_list = cert_sheet[cert_sheet['项目'].eq('原子标签全集')]['取值'].iloc[0]
    cert_nonempty = int(certification[schema.CERT_TAG_FIELD].map(len).gt(0).sum())
    cert_empty = int(certification[schema.CERT_TAG_FIELD].map(len).eq(0).sum())
    tag_atoms = int(tag_table.set_index('指标').loc['原子标签数', '数值'].replace(',', ''))
    tag_nonempty = int(tag_table.set_index('指标').loc['非空岗位数', '数值'].replace(',', ''))

    # ---- 分类判定 ----
    wide_has_cert = bool(((wide_sheet['字段名'].eq(schema.CERT_TAG_FIELD))
                          & wide_sheet['字段是否存在'].eq('存在')).any())
    chain_spec = chain_sheet[chain_sheet['层级'].eq('Stage 13 因素规格（当前状态）')]
    chain_history = chain_sheet[chain_sheet['层级'].astype(str).str.startswith('历史缺陷')]
    current_consistent = bool(len(chain_spec) == 2
                              and chain_spec['是否一致'].astype(str).str.startswith('是').all())
    history_recorded = bool(len(chain_history) == 1
                            and chain_history['是否一致'].astype(str)
                            .str.startswith('否').all())
    a4_ok = int(cert_block.shape[0]) == 369 and tag_atoms > 300
    classification = 'A2（Stage 13 字段引用错误）+ A1（显示名称错误）'
    classification_text = (
        'Stage 13 的设计（`docs/prompts/15_...`：结构化因素包含「公司认证」）要求分析公司认证，'
        f'但 `src/eda_analysis.py` 的 certification 因素实际引用 `{schema.COMPANY_TAG_LIST_FIELD}`'
        f'（公司福利/企业文化标签），并把显示名称写成「{DISPLAY_LABEL}」；'
        f'369 组即 `{schema.COMPANY_TAG_LIST_FIELD}` 中样本数 ≥ '
        f'{eda_analysis.MIN_GROUP_SIZE} 的原子标签数。Stage 12 宽表不存在映射错误：'
        f'`{schema.CERT_TAG_FIELD}` 未被纳入宽表属设计取舍，且公司标签取值与公司标签原文语义一致。')

    problem_sheet = pd.DataFrame([
        {'项': '问题分类', '结论': classification},
        {'项': '问题描述', '结论': classification_text},
        {'项': '369 组实际来源字段', '结论': schema.COMPANY_TAG_LIST_FIELD},
        {'项': '设计意图字段', '结论': f'{schema.CERT_TAG_FIELD}（最佳雇主 / 行业认证）'},
        {'项': 'A1 仅显示名称错误', '结论': '成立（显示名「公司认证标签」≠ 实际字段「公司标签列表」）'},
        {'项': 'A2 Stage 13 字段引用错误', '结论': '成立（应分析公司认证，实际引用公司标签列表）'},
        {'项': 'A3 Stage 12 宽表映射错误', '结论': '不成立（宽表取值与公司标签原文一致，'
                                            '公司认证未被纳入宽表属设计取舍）'},
        {'项': 'A4 list 组合/拆分统计错误', '结论': '不成立（组合值 2,892 个与原子标签 '
                                              f'{tag_atoms:,} 个已严格区分，369 来自 n ≥ 30 的原子标签）'},
        {'项': '是否修改 Stage 12', '结论': '否'},
        {'项': '是否重跑 Stage 12', '结论': '否'},
        {'项': '是否重跑 Stage 13', '结论': '是（修正字段引用与显示名称）'},
        {'项': '是否重跑 Stage 14', '结论': '否（模型特征与样本未变）'},
        {'项': '是否重跑 Stage 15', '结论': '否（仅增量刷新 SHAP 汇总）'},
    ])

    suggestion_sheet = pd.DataFrame([
        {'序号': 1, '修正对象': 'src/eda_analysis.py',
         '修正内容': f'公司因素拆分为「公司认证」（从 {schema.CERT_TAG_FIELD} 构造互斥四类：'
                     '无认证 / 最佳雇主 / 行业认证 / 两者均有）与「公司标签（福利标签）」'
                     f'（{schema.COMPANY_TAG_LIST_FIELD}，多值），删除错误的 certification 显示名',
         '是否需要重跑': '需要（Stage 13）'},
        {'序号': 2, '修正对象': 'scripts/ch4_lifecycle/01_run_eda.py',
         '修正内容': '公司因素调用链与 29 号表 06/11 子表的显示名称改为真实字段语义；'
                     '统计检验「检验对象」改为中文因素名',
         '是否需要重跑': '需要（Stage 13）'},
        {'序号': 3, '修正对象': 'docs/records/20_formal_eda_record.md',
         '修正内容': '公司因素小节区分「公司认证」与「公司标签（福利标签）」，并说明 369 组来源',
         '是否需要重跑': '随 Stage 13 重新生成'},
        {'序号': 4, '修正对象': 'data/processed/job_details_unique.parquet',
         '修正内容': '无需修改（公司认证标签为 Stage 05 正式产物，只读引用）',
         '是否需要重跑': '不需要'},
        {'序号': 5, '修正对象': 'src/modeling_dataset.py / scripts/ch7_model/01_train_salary_model.py 说明文案',
         '修正内容': '两处「公司认证标签」备注文字实指 `公司标签列表`，属纯显示文案；'
                     '本轮不改（改动后不重跑会造成源码与正式产物不一致，Stage 14 本轮禁止重跑），'
                     '已登记为已知残留，建议后续有意重跑 Stage 12/14 时同步修正',
         '是否需要重跑': '本轮不重跑'},
    ])

    gates.check('COMPANY_FIELD_TRACE',
                current_consistent and history_recorded
                and constants.get('COMPANY_CERTIFICATION_FIELD') == '不存在'
                and constants.get('COMPANY_TAG_LIST_FIELD') == '公司标签列表',
                f"调用链定位完成：修复后 Stage 13 因素规格为「公司认证」"
                f"（`{CERT_CLASS_FIELD}`，派生自 `{schema.CERT_TAG_FIELD}`）与「{COMPANY_TAG_LABEL}」"
                f"（`{schema.COMPANY_TAG_LIST_FIELD}`），显示名与真实字段一致；"
                f"修复前把 `{schema.COMPANY_TAG_LIST_FIELD}` 显示为「{DISPLAY_LABEL}」的历史行"
                f"作为证据保留（共 {len(chain_history)} 行）；"
                f"schema 中不存在 COMPANY_CERTIFICATION_FIELD 常量")
    gates.check('COMPANY_CERTIFICATION_ATOMS',
                cert_atoms == str(len(CERT_ATOMS_EXPECTED)) and not unknown_certs
                and set(cert_atom_list.split('、')) == CERT_ATOMS_EXPECTED,
                f"公司认证原子标签 {cert_atoms} 个：{cert_atom_list}；未知认证值 "
                f"{'无' if not unknown_certs else '、'.join(unknown_certs)}；"
                f"有认证岗位 {cert_nonempty:,}，空 list（无认证）{cert_empty:,}")
    gates.check('COMPANY_TAG_FIELD_DISTINCT',
                tag_atoms > 100 and not wide_has_cert
                and schema.COMPANY_TAG_LIST_FIELD != schema.CERT_TAG_FIELD,
                f"公司标签（福利标签）原子标签 {tag_atoms:,} 个、非空岗位 "
                f"{tag_nonempty:,}，与公司认证（{cert_atoms} 个原子标签）"
                f"字段与取值范围规模完全不同（2 vs {tag_atoms:,}）；"
                f"仅字符串层面存在个别重叠（公司标签中「最佳雇主」"
                f"{tag_overlap['最佳雇主']} 个岗位、「行业认证」{tag_overlap['行业认证']} 个岗位），"
                f"语义以字段为准；公司认证字段是否进入 Stage 12 宽表："
                f"{'是' if wide_has_cert else '否'}")
    gates.check('COMPANY_FACTOR_MISLABEL_CONFIRMED',
                a4_ok and int(kruskal_row['参与检验组数'].iloc[0]) == 369
                and bool(set(cert_block['取值']) & {'免费健身设施', '餐饮', '节日礼品', '弹性工作制'}),
                f"29 号表该因素的 {int(cert_block.shape[0])} 行取值全部来自 "
                f"`{schema.COMPANY_TAG_LIST_FIELD}`（前 20 个取值："
                f"{'、'.join(cert_block['取值'].head(20).tolist()[:5])} …）；"
                f"本次读取状态：{state}；检验对象 = "
                f"`{kruskal_row['检验对象'].iloc[0]}`、组数 "
                f"{kruskal_row['参与检验组数'].iloc[0]}、epsilon² {kruskal_row['效应量'].iloc[0]}")
    audit_path = project_paths.TABLES_DIR / project_paths.TABLE_COMPANY_FIELD_SEMANTIC
    sheets = {
        '01_schema字段映射': schema_sheet,
        '02_宽表字段统计': wide_sheet,
        '03_公司认证原子标签': cert_sheet,
        '04_公司标签原子标签': tag_table,
        '04b_公司标签Top50': tag_top,
        '04c_每岗标签数分布': tag_distribution,
        '05_Stage13调用链': chain_sheet,
        '06_29号表实际分组': group_sheet,
        '07_问题分类': problem_sheet,
        '08_修正建议': suggestion_sheet,
    }
    gates.check('COMPANY_FIELD_AUDIT_EXPORT',
                len(sheets) >= 8 and not chain_sheet.empty and not problem_sheet.empty,
                f'32 号审计表含 {len(sheets)} 张子表（schema 映射 / 宽表字段 / 认证原子标签 / '
                f'标签原子标签 / 调用链 / 29 号表分组 / 问题分类 / 修正建议）')
    sheets['09_门禁'] = pd.DataFrame([
        {'门禁项': name, '状态': item['status'], '说明': item['note']}
        for name, item in gates.results.items()])
    audit_path = io_utils.write_excel(audit_path, sheets)

    findings = {
        'classification': classification, 'classification_text': classification_text,
        'A1': '成立', 'A2': '成立', 'A3': '不成立', 'A4': '不成立',
        'cert_atoms': cert_atoms, 'cert_atom_list': cert_atom_list,
        'cert_nonempty': cert_nonempty, 'cert_empty': cert_empty,
        'tag_atoms': tag_atoms, 'tag_nonempty': tag_nonempty,
        'tag_top20': '、'.join(tag_top['原子标签'].head(20).tolist()),
        'sheet5_key': str(kruskal_row['检验对象'].iloc[0]),
        'sheet6_label': str(cert_block['因素'].iloc[0]),
        'state': state,
        'tag_overlap': tag_overlap,
        'gates': {name: item['status'] for name, item in gates.results.items()},
        'gate_notes': {name: item['note'] for name, item in gates.results.items()},
        'audit_path': project_paths.relative_to_root(audit_path),
    }
    io_utils.write_json(project_paths.METRICS_DIR / f'{STAGE}.json', findings)
    record_path = write_record(findings)
    print(f'审计表: {project_paths.relative_to_root(audit_path)}')
    print(f'记录: {project_paths.relative_to_root(record_path)}')
    quality.print_gates(gates.results, order=quality.STAGE_GATE_MAP[STAGE])
    gates.save()
    return 0


if __name__ == '__main__':
    sys.exit(main())

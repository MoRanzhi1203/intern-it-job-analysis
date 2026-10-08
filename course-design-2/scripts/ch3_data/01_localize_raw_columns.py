# -*- coding: utf-8 -*-
"""Stage 01：源记录标准化与公司属性语义槽位异常修复（唯一正式预处理入口）。

data/raw/shixiseng_job_details.parquet（英文列名；永久不可修改）
        ↓  A. 字段中文化（只改列名）
        ↓  B. 文本空白标准化（'' / 纯空格 → 缺失；派生 list 字段用 []）
        ↓  C. 公司属性语义槽位异常检测与分型修复（FULL_SHIFT / NATURE_ONLY / SIZE_ONLY）
        ↓  D. 数据类型统一
        ↓  E. 质量门禁 + 22 号合并审计表
data/interim/shixiseng_job_details_cn.parquet（正式工作副本，只保留业务字段）

边界（严格遵守）：
1. raw Parquet 永久不可修改，本阶段只读；
2. 本阶段是全流程中唯一允许做「源记录语义标准化与确定性纠正」的阶段，
   下游 Stage 只验证、不重复修复；
3. 公司属性槽位异常的修复规则为确定性规则，禁止推断、禁止回填未知公司性质；
4. 审计辅助列（异常类型 / 原值 / 修复值 / 检测证据 / 参考行 ID）只进入审计表，
   不进入 data/interim 正式宽表；
5. 公司认证标签映射（链接 → 最佳雇主 / 行业认证）仍由 Stage 03 在去重前完成。

用法：
    python scripts/ch3_data/01_localize_raw_columns.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import field_repair, io_utils, project_paths, quality, schema  # noqa: E402

STAGE = 'stage_01'
TITLE = 'Stage 01 源记录标准化与公司属性语义槽位异常修复'

REPAIRED_FIELDS = [field_repair.NATURE_FIELD, field_repair.SIZE_FIELD,
                   field_repair.LOCATION_FIELD]
# 非文本字段：不做空白标准化
NON_TEXT_FIELDS = {'id', 'search_page', 'created_at', 'updated_at'}
# 由一次性取证冻结、本阶段结转保留的 Sheet（与流水线重算的 Sheet 合并为同一张审计表）
CARRY_OVER_SHEETS = ('04_MySQL_vs_Parquet证据', '05_同公司正常参考')


def localize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """A. 仅重命名列，返回新的 DataFrame（不修改入参）。"""
    unmapped, unused = schema.validate_mapping_coverage(frame.columns)
    if unmapped:
        raise KeyError(f'存在未映射的原始字段: {unmapped}')
    if unused:
        raise KeyError(f'映射字典包含数据中不存在的字段: {unused}')
    return frame.rename(columns=schema.COLUMN_NAME_CN).copy()


def standardize_missing(frame: pd.DataFrame) -> tuple:
    """B. 文本空白标准化：'' / 纯空格 → 缺失；返回（新 DataFrame, 各列标准化行数）。"""
    standardized = frame.copy()
    counts: dict = {}
    for column in standardized.columns:
        if column in NON_TEXT_FIELDS or standardized[column].dtype != object:
            continue
        blank = standardized[column].fillna('').astype(str).str.strip().eq('')
        if not blank.any():
            continue
        standardized.loc[blank, column] = np.nan
        counts[column] = int(blank.sum())
    return standardized, counts


def unify_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """D. 数据类型统一：主键整型、页码浮点、时间字段 datetime64。"""
    unified = frame.copy()
    unified['记录ID'] = unified['记录ID'].astype('int64')
    unified['搜索页码'] = unified['搜索页码'].astype('float64')
    for column in ('数据创建时间', '数据更新时间'):
        unified[column] = pd.to_datetime(unified[column])
    return unified


def _normalized_view(frame: pd.DataFrame) -> pd.DataFrame:
    """逐格比较用视图：文本列统一 strip + 空值归一，非文本列原样。"""
    view = frame.copy()
    for column in view.columns:
        if column in NON_TEXT_FIELDS or view[column].dtype != object:
            continue
        view[column] = view[column].fillna('').astype(str).str.strip()
    return view


def verify_stage01(raw: pd.DataFrame, localized: pd.DataFrame,
                   repair_positions: set) -> dict:
    """B/D 步一致性校验：除已登记修复行外，取值与源数据逐格一致。"""
    problems = []
    if len(raw) != len(localized):
        problems.append(f'行数不一致: {len(raw)} != {len(localized)}')
    if raw.shape[1] != localized.shape[1]:
        problems.append(f'列数不一致: {raw.shape[1]} != {localized.shape[1]}')
    expected = schema.english_to_chinese(raw.columns)
    if list(localized.columns) != expected:
        problems.append('汉化后列名或列顺序与映射不一致')
    if len(set(localized.columns)) != len(localized.columns):
        problems.append('汉化后存在重复中文列名')
    if list(raw.columns) == list(localized.columns):
        problems.append('汉化未生效')
    if problems:
        return {'是否通过': False, '问题': '；'.join(problems)}

    raw_view = _normalized_view(raw.rename(columns=schema.COLUMN_NAME_CN))
    localized_view = _normalized_view(localized)
    changed_columns = {}
    for column in localized_view.columns:
        mismatch = raw_view[column].to_numpy() != localized_view[column].to_numpy()
        if mismatch.any():
            changed_columns[column] = set(np.flatnonzero(mismatch).tolist())
    unexpected = {column: len(rows) for column, rows in changed_columns.items()
                  if column not in REPAIRED_FIELDS}
    if unexpected:
        problems.append(f'出现未预期字段改动: {unexpected}')
    for column in REPAIRED_FIELDS:
        extra = changed_columns.get(column, set()) - repair_positions
        if extra:
            problems.append(f'{column} 出现未登记的取值改动 {len(extra)} 行')
    return {
        '是否通过': not problems,
        '问题': '；'.join(problems) if problems else '无',
        '发生改动的字段': '、'.join(sorted(changed_columns)) if changed_columns else '无',
    }


def build_conclusion_table(diagnostics: dict, checks: dict, blank_counts: dict,
                           raw_blank_total: int, cn_blank_total: int,
                           consistency: dict) -> pd.DataFrame:
    """01_最终结论。"""
    rows = [
        {'项目': '原始数据', '内容': f'{project_paths.relative_to_root(project_paths.RAW_PARQUET)}'
                                '（172063 行 × 30 列，永久不可修改）'},
        {'项目': '本阶段工作副本', '内容':
            project_paths.relative_to_root(project_paths.INTERIM_CN_PARQUET)},
        {'项目': '正式异常名称', '内容': '源记录公司属性语义槽位异常'
                                  '（禁止写作「字段错位 / 迁移错位」）'},
        {'项目': '异常来源结论', '内容': 'MySQL 与 Stage 00 raw Parquet 同 ID 30 列完全一致'
                                  '（467/467），异常来自源记录本身，不是导出/迁移造成'},
        {'项目': '性质栏位为人数区间行数', '内容': diagnostics['性质栏位为人数区间行数']},
        {'项目': '规模栏位为地点文本行数', '内容': diagnostics['规模栏位为地点文本行数']},
        {'项目': '两类症状同时出现行数', '内容': diagnostics['两类症状同时出现行数']},
        {'项目': '异常候选并集行数', '内容': diagnostics['异常候选并集行数']},
        {'项目': 'FULL_SHIFT 行数', '内容': diagnostics['FULL_SHIFT 行数']},
        {'项目': 'NATURE_ONLY 行数', '内容': diagnostics['NATURE_ONLY 行数']},
        {'项目': 'SIZE_ONLY 行数', '内容': diagnostics['SIZE_ONLY 行数']},
        {'项目': '修复后残留异常行数', '内容': checks['修复后残留异常行数']},
        {'项目': '性质被置为缺失行数', '内容': diagnostics['性质被置为缺失行数']},
        {'项目': '空白标准化字段数', '内容': f'{len(blank_counts)} 个字段'},
        {'项目': 'raw 中空字符串/纯空格总数', '内容': raw_blank_total},
        {'项目': '工作副本中残留空字符串/纯空格总数', '内容': cn_blank_total},
        {'项目': '一致性校验', '内容': f'通过={consistency["是否通过"]}；{consistency["问题"]}'},
    ]
    return pd.DataFrame(rows)


def build_type_table(diagnostics: dict) -> pd.DataFrame:
    """02_异常类型统计。"""
    rows = [
        {'指标': '性质栏位为人数区间行数', '数值': diagnostics['性质栏位为人数区间行数']},
        {'指标': '规模栏位为地点文本行数', '数值': diagnostics['规模栏位为地点文本行数']},
        {'指标': '两类症状同时出现行数', '数值': diagnostics['两类症状同时出现行数']},
        {'指标': '症状命中但所在地非空（不修复）行数',
         '数值': diagnostics['症状命中但所在地非空（不修复）行数']},
        {'指标': '异常候选并集行数', '数值': diagnostics['异常候选并集行数']},
        {'指标': 'FULL_SHIFT 行数', '数值': diagnostics['FULL_SHIFT 行数']},
        {'指标': 'NATURE_ONLY 行数', '数值': diagnostics['NATURE_ONLY 行数']},
        {'指标': 'SIZE_ONLY 行数', '数值': diagnostics['SIZE_ONLY 行数']},
        {'指标': '修复后规模可还原行数', '数值': diagnostics['修复后规模可还原行数']},
        {'指标': '修复后所在地可还原行数', '数值': diagnostics['修复后所在地可还原行数']},
        {'指标': '性质被置为缺失行数', '数值': diagnostics['性质被置为缺失行数']},
        {'指标': '涉及岗位数', '数值': diagnostics['涉及岗位数']},
        {'指标': '涉及公司数', '数值': diagnostics['涉及公司数']},
    ]
    return pd.DataFrame(rows)


def build_candidate_table(before: pd.DataFrame, audit_table: pd.DataFrame,
                          repaired: pd.DataFrame, hint: pd.DataFrame) -> pd.DataFrame:
    """03_全部候选行：异常类型 + 原值 + 修复值 + 参考提示。"""
    positions = audit_table['源记录序号'].to_numpy()
    table = audit_table.copy()
    table.insert(0, '记录ID', before['记录ID'].iloc[positions].to_numpy())
    for field in REPAIRED_FIELDS:
        table[f'修复后_{field}'] = repaired[field].iloc[positions].to_numpy()
    table = table.merge(hint, on=field_repair.COMPANY_NAME_FIELD, how='left')
    return table


def build_before_after_table(audit_table: pd.DataFrame, repaired: pd.DataFrame) -> pd.DataFrame:
    """07_修复前后对照。"""
    positions = audit_table['源记录序号'].to_numpy()
    rows = audit_table[['源记录序号', field_repair.JOB_ID_FIELD,
                        field_repair.COMPANY_NAME_FIELD, '异常类型']].copy()
    for field in REPAIRED_FIELDS:
        before_values = audit_table[f'{field}_原值'].to_numpy(dtype=object)
        after_values = repaired[field].iloc[positions].fillna('').astype(str).to_numpy()
        rows[f'{field}_修复前'] = before_values
        rows[f'{field}_修复后'] = after_values
        rows[f'{field}_是否变化'] = np.where(
            pd.Series(before_values).fillna('').astype(str).to_numpy() != after_values, '是', '否')
    return rows


def build_residual_table(checks: dict) -> pd.DataFrame:
    """08_残留异常。"""
    return pd.DataFrame([{'检查项': name, '结果': value} for name, value in checks.items()])


def write_record(raw: pd.DataFrame, localized: pd.DataFrame, diagnostics: dict, checks: dict,
                 blank_counts: dict, audit_path: Path, raw_blank_total: int,
                 cn_blank_total: int) -> Path:
    """写出 Stage 01 阶段记录（本阶段唯一权威记录）。"""
    lines = [
        '# 阶段 01 记录：源记录标准化与公司属性语义槽位异常修复',
        '',
        '> 本文件由 `scripts/ch3_data/01_localize_raw_columns.py` 自动生成，数字全部来自真实运行结果。',
        '',
        '## 1. 输入与输出',
        '',
        '| 项 | 内容 |',
        '| --- | --- |',
        f'| 输入（只读） | `{project_paths.relative_to_root(project_paths.RAW_PARQUET)}`'
        f'（{len(raw)} 行 × {raw.shape[1]} 列，英文列名，永久不可修改） |',
        f'| 输出 | `{project_paths.relative_to_root(project_paths.INTERIM_CN_PARQUET)}`'
        f'（{len(localized)} 行 × {localized.shape[1]} 列，中文列名） |',
        f'| 审计表 | `{project_paths.relative_to_root(audit_path)}` |',
        '',
        '## 2. 本阶段职责（A~E）',
        '',
        '| 步骤 | 内容 |',
        '| --- | --- |',
        '| A | 字段中文化（只改列名，不改取值） |',
        '| B | 文本空白标准化：`\'\'` / 纯空格 → 缺失（raw 中保留原样） |',
        '| C | 公司属性语义槽位异常检测与分型修复（FULL_SHIFT / NATURE_ONLY / SIZE_ONLY） |',
        '| D | 数据类型统一（主键整型 / 页码浮点 / 时间 datetime64） |',
        '| E | 质量门禁与 22 号合并审计表 |',
        '',
        '## 3. 文本空白标准化',
        '',
        f'- 涉及字段数：{len(blank_counts)}；',
        f'- raw 中文本 `\'\'` / 纯空格总数：{raw_blank_total}（保持原样，不修改）；',
        f'- 工作副本标准化后残留纯空白总数：{cn_blank_total}；',
        '- 标准化后工作副本中文本业务字段不再出现纯空白值；',
        '- 派生 list 字段（公司认证标签 / 岗位标签列表）使用 `[]` 而非缺失（由 Stage 03/06 生成）。',
        '',
        '## 4. 源记录公司属性语义槽位异常',
        '',
        '正式名称：**源记录公司属性语义槽位异常**。取证结论见'
        ' `docs/records/15_company_attribute_semantic_anomaly_record.md`：'
        'MySQL 与 Stage 00 raw Parquet 同 ID 30 列完全一致（467/467），'
        '即该异常来自源记录本身，不是导出/迁移过程造成。',
        '',
        '| 类型 | 行数 |',
        '| --- | --- |',
        f'| FULL_SHIFT | {diagnostics["FULL_SHIFT 行数"]} |',
        f'| NATURE_ONLY | {diagnostics["NATURE_ONLY 行数"]} |',
        f'| SIZE_ONLY | {diagnostics["SIZE_ONLY 行数"]} |',
        f'| 合计候选 | {diagnostics["异常候选并集行数"]} |',
        '',
        f'- 性质栏位为人数区间：{diagnostics["性质栏位为人数区间行数"]} 行；',
        f'- 规模栏位为地点文本：{diagnostics["规模栏位为地点文本行数"]} 行；',
        f'- 两类同时：{diagnostics["两类症状同时出现行数"]} 行；',
        f'- 症状命中但所在地非空（不修复）：'
        f'{diagnostics["症状命中但所在地非空（不修复）行数"]} 行；',
        f'- 涉及岗位 {diagnostics["涉及岗位数"]} 个、公司 {diagnostics["涉及公司数"]} 家。',
        '',
        '## 5. 确定性修复规则',
        '',
        '| 类型 | 触发条件 | 修复 |',
        '| --- | --- | --- |',
        f'| FULL_SHIFT | 性质像人数规模 且 规模像地点 且 所在地为空 '
        f'| 规模 ← 原性质；所在地 ← 原规模；性质 ← 缺失 |',
        f'| NATURE_ONLY | 性质像人数规模 且 规模非地点 且 所在地为空 '
        f'| 规模 ← 原性质；性质 ← 缺失；所在地保持原值 |',
        f'| SIZE_ONLY | 性质正常 且 规模像地点 且 所在地为空 '
        f'| 所在地 ← 原规模；规模 ← 缺失；性质保持原值 |',
        '',
        f'- 性质被置为缺失：{diagnostics["性质被置为缺失行数"]} 行；',
        '- 禁止众数填充 / 同公司回填 / 按行业或规模推断；未知性质统一为缺失，'
        '由建模阶段按「缺失 → 未知 + 缺失指示」处理；',
        '- 审计辅助列只进入 22 号审计表，不进入 `data/interim` 正式宽表。',
        '',
        '## 6. 修复后校验',
        '',
        '| 校验项 | 结果 |',
        '| --- | --- |',
    ]
    for name, value in checks.items():
        lines.append(f'| {name} | {value} |')
    lines += [
        '',
        '## 7. 下游边界',
        '',
        '- Stage 02~10 只做身份确认 / 关系拆分 / 版本时序 / 最终实体 / 文本与公司语义，'
        '不再修复公司属性；',
        '- Stage 11 只复核残留异常（`COMPANY_ATTRIBUTE_ANOMALY_RESIDUAL_CHECK`），'
        '残留即 FAIL，不再次修复；',
        '- 未修改 MySQL 原始数据、未修改 raw Parquet、未删除任何列、未去重。',
        '',
    ]
    return io_utils.write_markdown(
        project_paths.RECORDS_DIR / project_paths.RECORD_COLUMN_LOCALIZATION, lines)


def main() -> int:
    project_paths.ensure_directories()
    quality.configure_logging()
    gates = quality.GateRegistry(STAGE)

    quality.stage_banner(STAGE, TITLE)
    df_raw = io_utils.read_parquet(project_paths.RAW_PARQUET)
    print(f'输入（英文原始数据，只读）: {len(df_raw)} 行 × {df_raw.shape[1]} 列')

    # A. 中文化
    df_cn = localize_columns(df_raw)
    print(f'汉化完成（仅列名）: {len(df_cn)} 行 × {df_cn.shape[1]} 列')

    # B. 空白标准化
    df_standard, blank_counts = standardize_missing(df_cn)
    print(f'空白标准化: {len(blank_counts)} 个字段（raw 中保留原始空白表示，不修改）')

    # C. 公司属性语义槽位异常检测与分型修复
    config = field_repair.load_company_attribute_anomaly_config()
    city_vocabulary = field_repair.build_city_vocabulary(
        df_standard, config['city_normalization']['separators'])
    audit_table, diagnostics = field_repair.detect_company_attribute_slot_anomaly(
        df_standard, city_vocabulary, config)
    gates.check('COMPANY_ATTRIBUTE_ANOMALY_DETECT',
                diagnostics['异常候选并集行数'] > 0
                and diagnostics['性质栏位为人数区间行数'] > 0
                and diagnostics['规模栏位为地点文本行数'] > 0
                and diagnostics['症状命中但所在地非空（不修复）行数'] == 0
                and set(audit_table['异常类型'].unique()) <= set(field_repair.ANOMALY_TYPES),
                f'仅检测（未修复）：性质像规模 {diagnostics["性质栏位为人数区间行数"]} 行、'
                f'规模像地点 {diagnostics["规模栏位为地点文本行数"]} 行、两类同时 '
                f'{diagnostics["两类症状同时出现行数"]} 行 → 候选并集 '
                f'{diagnostics["异常候选并集行数"]} 行；'
                f'FULL_SHIFT {diagnostics["FULL_SHIFT 行数"]} / '
                f'NATURE_ONLY {diagnostics["NATURE_ONLY 行数"]} / '
                f'SIZE_ONLY {diagnostics["SIZE_ONLY 行数"]}')

    df_repaired = field_repair.apply_deterministic_company_attribute_repair(
        df_standard, audit_table)
    checks = field_repair.validate_company_attribute_repair(
        df_standard, df_repaired, audit_table, city_vocabulary, config)
    gates.check('COMPANY_ATTRIBUTE_ANOMALY_REPAIR',
                bool(checks['是否通过']) and checks['修复后残留异常行数'] == 0,
                f'确定性修复 {len(audit_table)} 行：规模还原 '
                f'{diagnostics["修复后规模可还原行数"]}、所在地还原 '
                f'{diagnostics["修复后所在地可还原行数"]}、性质置缺失 '
                f'{diagnostics["性质被置为缺失行数"]}；未修复行逐格未变，修复后残留 0')

    # D. 数据类型统一 + 一致性校验
    df_final = unify_dtypes(df_repaired)
    repair_positions = set(audit_table['源记录序号'].tolist()) if len(audit_table) else set()
    consistency = verify_stage01(df_raw, df_final, repair_positions)
    gates.check('COLUMN_LOCALIZATION',
                bool(consistency['是否通过'])
                and len(df_final) == len(df_raw)
                and list(df_final.columns) == schema.english_to_chinese(df_raw.columns),
                f'{len(df_final)} 行 × {df_final.shape[1]} 列；'
                f'除已登记的 {len(audit_table)} 行公司属性槽位修复外，取值与源数据逐格一致'
                f'（改动字段：{consistency.get("发生改动的字段", "无")}）')

    def _blank_only(frame: pd.DataFrame) -> int:
        """只统计“非空但纯空白”的取值（缺失值不计入）。"""
        total = 0
        for column in frame.columns:
            series = frame[column]
            if series.dtype != object:
                continue
            total += int((series.notna() & series.astype(str).str.strip().eq('')).sum())
        return total

    cn_blank_total = _blank_only(df_final)
    raw_blank_total = _blank_only(df_raw)
    gates.check('MISSING_STANDARDIZATION',
                cn_blank_total == 0 and raw_blank_total > 0 and len(blank_counts) > 0,
                f'raw 中文本空字符串/纯空格 {raw_blank_total} 个（保持原样），'
                f'工作副本标准化后残留 {cn_blank_total} 个；涉及字段 {len(blank_counts)} 个')

    io_utils.write_parquet(df_final, project_paths.INTERIM_CN_PARQUET, verify=True)

    # ---- 阶段记录与审计表 ----
    quality_table = quality.field_quality_table(
        df_final, name_map={v: v for v in schema.COLUMN_NAME_CN.values()},
        description_map={schema.COLUMN_NAME_CN[en]: desc
                         for en, desc in schema.COLUMN_DESCRIPTION.items()})
    quality_table.insert(1, '英文列名', list(df_raw.columns))
    io_utils.write_excel(project_paths.TABLES_DIR / project_paths.TABLE_COLUMN_DICTIONARY,
                         {'字段中英文对照表': quality_table})

    hint = field_repair.build_nature_lost_reference_hint(df_standard, audit_table)
    audit_sheets = {
        '01_最终结论': build_conclusion_table(diagnostics, checks, blank_counts,
                                          raw_blank_total, cn_blank_total, consistency),
        '02_异常类型统计': build_type_table(diagnostics),
        '03_全部候选行': build_candidate_table(df_standard, audit_table, df_final, hint),
        '06_确定性修复规则': field_repair.build_repair_rule_table(config),
        '07_修复前后对照': build_before_after_table(audit_table, df_final),
        '08_残留异常': build_residual_table(checks),
    }
    audit_path = io_utils.write_excel(
        project_paths.COMPANY_ATTRIBUTE_ANOMALY_TABLE_PATH, audit_sheets,
        carry_over_sheets=CARRY_OVER_SHEETS)

    reloaded = io_utils.read_parquet(project_paths.INTERIM_CN_PARQUET)
    sheets_written = pd.ExcelFile(audit_path).sheet_names
    gates.check('COMPANY_ATTRIBUTE_ANOMALY_AUDIT_EXPORT',
                audit_path.exists() and len(audit_sheets) == 6
                and bool(checks['是否通过'])
                and len(reloaded) == len(df_raw),
                f'22 号合并审计表 {len(sheets_written)} 张子表已写出'
                f'（本阶段重算 6 张 + 结转 2 张一次性取证结果：'
                f'{list(CARRY_OVER_SHEETS)}）；回读工作副本 {len(reloaded)} 行')

    record_path = write_record(df_raw, df_final, diagnostics, checks, blank_counts, audit_path,
                               raw_blank_total, cn_blank_total)
    print(f'阶段 record: {project_paths.relative_to_root(record_path)}')

    quality.print_gates(gates.results, order=quality.STAGE_GATE_MAP[STAGE])
    gates.save()
    print(f'Stage 01 完成: {len(df_final)} 行 × {df_final.shape[1]} 列')
    return 0


if __name__ == '__main__':
    sys.exit(main())

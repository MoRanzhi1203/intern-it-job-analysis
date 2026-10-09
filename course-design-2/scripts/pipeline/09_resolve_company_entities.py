# -*- coding: utf-8 -*-
"""Stage 09：公司实体高置信度识别（禁止按岗位ID直接做公司时序）。

处理路线：

data/interim/shixiseng_job_details_cn.parquet   原始汉化观测（公司名称 / 所在地 / 简介）
data/processed/job_version_history.parquet      岗位版本时序层
        ↓  名称规范化（仅低风险归一化，不做模糊合并）
        ↓  映射方式（Refinement R1 收紧）：
           EXACT_NORMALIZED_NAME / APPROVED_ALIAS → 允许进入正式公司简介时序
           MULTI_LOCATION_AMBIGUOUS（同名跨地域） / HIGH_CONFIDENCE_MATCH / UNRESOLVED
           → 禁止进入正式时序，标记人工复核
        ↓
data/processed/company_entity_map.parquet       公司实体映射层

关键变更：**公司所在地不再作为自动拆分正式公司实体的充分条件**。

用法：
    python scripts/pipeline/09_resolve_company_entities.py
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

from src import company_identity, io_utils, project_paths, quality, schema, text_utils  # noqa: E402

STAGE = 'stage_09'
TITLE = 'Stage 09 公司实体高置信度识别（跨地域不再自动拆分）'


def load_inputs() -> tuple:
    """读取原始汉化观测与岗位版本历史。"""
    columns = ['实习岗位ID', '公司名称', '公司所在地', '公司简介', '公司标签', '所属行业']
    raw = io_utils.read_parquet(project_paths.INTERIM_CN_PARQUET, columns=columns)
    versions = io_utils.read_parquet(project_paths.PROCESSED_VERSION_HISTORY_PARQUET,
                                     columns=['实习岗位ID', '公司名称', '公司所在地'])
    return raw, versions


def build_name_location_stats(raw: pd.DataFrame) -> pd.DataFrame:
    """按「公司名称 + 公司所在地」聚合岗位数与观测记录数。"""
    frame = raw.copy()
    frame['公司名称'] = frame['公司名称'].fillna('').astype(str)
    frame['公司所在地'] = frame['公司所在地'].fillna('').astype(str)
    stats = (frame.groupby(['公司名称', '公司所在地'], as_index=False)
             .agg(岗位数=('实习岗位ID', 'nunique'), 观测记录数=('实习岗位ID', 'size')))
    return stats


def build_profile_stats(raw: pd.DataFrame) -> pd.DataFrame:
    """公司简介指纹 → 公司名称集合（用于识别「同一简介被多个名称引用」）。"""
    frame = raw[['公司名称', '公司简介']].copy()
    frame['公司名称'] = frame['公司名称'].fillna('').astype(str)
    frame['公司简介指纹'] = frame['公司简介'].map(company_identity.company_fingerprint)
    frame = frame[frame['公司简介指纹'] != '']
    return (frame.drop_duplicates(['公司简介指纹', '公司名称'])[['公司简介指纹', '公司名称']]
            .reset_index(drop=True))


def build_audit_tables(entity_map: pd.DataFrame, diagnostics: dict) -> dict:
    """构建 18 号公司实体审计表。"""
    method_field = schema.COMPANY_MAPPING_METHOD_FIELD
    entity_field = schema.COMPANY_ENTITY_ID_FIELD
    formal = entity_map[entity_map[schema.COMPANY_FORMAL_FIELD] == 1]

    method_counts = (entity_map.groupby(method_field)
                     .agg(名称数=(schema.COMPANY_NAME_ORIGINAL_FIELD, 'nunique'),
                          实体数=(entity_field, 'nunique'),
                          岗位数=('岗位数', 'sum'), 观测记录数=('观测记录数', 'sum'))
                     .reset_index())
    method_counts['是否允许进入正式公司时序'] = method_counts[method_field].map(
        lambda method: int(method in company_identity.FORMAL_MAPPING_METHODS))

    overview = pd.DataFrame([
        {'指标': '原始公司名称数', '数值': int(entity_map[schema.COMPANY_NAME_ORIGINAL_FIELD]
                                          .replace('', pd.NA).nunique())},
        {'指标': '规范化公司名称数',
         '数值': int(entity_map[schema.COMPANY_NAME_NORMALIZED_FIELD].replace('', pd.NA).nunique())},
        {'指标': '公司实体总数（含未解析）',
         '数值': int(entity_map[entity_field].replace('', pd.NA).nunique())},
        {'指标': '进入正式公司时序的实体数', '数值': int(formal[entity_field].nunique())},
        {'指标': '进入正式公司时序的岗位数', '数值': int(formal['岗位数'].sum())},
        {'指标': '无需人工复核的映射行数',
         '数值': int((entity_map[schema.COMPANY_NEED_REVIEW_FIELD] == 0).sum())},
        {'指标': '需人工复核的映射行数',
         '数值': int((entity_map[schema.COMPANY_NEED_REVIEW_FIELD] == 1).sum())},
        {'指标': '同名跨地域（MULTI_LOCATION_AMBIGUOUS）行数',
         '数值': int((entity_map[method_field]
                      == company_identity.METHOD_MULTI_LOCATION_AMBIGUOUS).sum())},
        {'指标': '同名跨地域规范名称数',
         '数值': int(entity_map.loc[entity_map[schema.COMPANY_CROSS_LOCATION_FIELD] == 1,
                                    schema.COMPANY_NAME_NORMALIZED_FIELD].nunique())},
        {'指标': '无法解析公司行数',
         '数值': int((entity_map[method_field] == company_identity.METHOD_UNRESOLVED).sum())},
        {'指标': '已人工确认别名条目数',
         '数值': int((diagnostics['别名配置表']['状态'] == 'approved').sum())
         if not diagnostics['别名配置表'].empty else 0},
        {'指标': '待人工确认别名条目数',
         '数值': int((diagnostics['别名配置表']['状态'] == 'pending').sum())
         if not diagnostics['别名配置表'].empty else 0},
    ])

    normalized = (entity_map[entity_map[schema.COMPANY_NAME_NORMALIZED_FIELD] != '']
                  .groupby(schema.COMPANY_NAME_NORMALIZED_FIELD)
                  .agg(来源名称数=(schema.COMPANY_NAME_ORIGINAL_FIELD, 'nunique'),
                       实体数=(entity_field, 'nunique'),
                       岗位数=('岗位数', 'sum'),
                       所在地数量=(schema.COMPANY_LOCATION_COUNT_FIELD, 'max'),
                       映射方式=(method_field,
                             lambda values: '、'.join(sorted(set(values)))),
                       是否进入正式公司时序=(schema.COMPANY_FORMAL_FIELD, 'max'))
                  .reset_index().sort_values('岗位数', ascending=False))

    multi_location = entity_map[entity_map[schema.COMPANY_CROSS_LOCATION_FIELD] == 1]
    multi_location_names = (multi_location
                            .groupby(schema.COMPANY_NAME_NORMALIZED_FIELD, as_index=False)
                            .agg(所在地数量=(schema.COMPANY_LOCATION_COUNT_FIELD, 'max'),
                                 所在地集合=(schema.COMPANY_LOCATION_SET_FIELD,
                                        lambda values: '、'.join(sorted({
                                            str(item) for value in values
                                            for item in text_utils.as_list(value) if item}))),
                                 来源名称数=(schema.COMPANY_NAME_ORIGINAL_FIELD, 'nunique'),
                                 岗位数=('岗位数', 'sum'),
                                 映射方式=(method_field,
                                       lambda values: '、'.join(sorted(set(values)))),
                                 是否进入正式公司时序=(schema.COMPANY_FORMAL_FIELD, 'max'))
                            .sort_values('岗位数', ascending=False)
                            .rename(columns={schema.COMPANY_NAME_NORMALIZED_FIELD: '公司规范名称'}))

    formal_entity_table = (formal.groupby(entity_field, as_index=False)
                           .agg(公司规范名称=(entity_field, 'first'),
                                来源名称数=(schema.COMPANY_NAME_ORIGINAL_FIELD, 'nunique'),
                                岗位数=('岗位数', 'sum'),
                                观测记录数=('观测记录数', 'sum'),
                                所在地数量=(schema.COMPANY_LOCATION_COUNT_FIELD, 'max'),
                                映射方式=(method_field,
                                      lambda values: '、'.join(sorted(set(values)))),
                                公司简介指纹数量=(schema.COMPANY_PROFILE_FP_COUNT_FIELD, 'max'),
                                是否需人工复核=(schema.COMPANY_NEED_REVIEW_FIELD, 'max'))
                           .sort_values('岗位数', ascending=False))

    exclude_reason = (entity_map[entity_map[schema.COMPANY_FORMAL_FIELD] == 0]
                      .groupby([method_field, schema.COMPANY_FORMAL_EXCLUDE_REASON_FIELD],
                               as_index=False)
                      .agg(行数=(entity_field, 'size'),
                           名称数=(schema.COMPANY_NAME_ORIGINAL_FIELD, 'nunique'),
                           岗位数=('岗位数', 'sum'))
                      .sort_values('行数', ascending=False))

    high_confidence = entity_map[entity_map[method_field]
                                 == company_identity.METHOD_HIGH_CONFIDENCE_MATCH]
    review_queue = entity_map[entity_map[schema.COMPANY_NEED_REVIEW_FIELD] == 1]
    unresolved = entity_map[entity_map[method_field] == company_identity.METHOD_UNRESOLVED]

    return {
        '01_公司实体总体统计': overview,
        '02_规范化公司名称统计': normalized,
        '03_公司别名映射': diagnostics['别名配置表'],
        '04_高置信度候选': high_confidence,
        '05_人工复核候选': review_queue,
        '06_无法解析公司': unresolved,
        '07_一对多映射检查': diagnostics['一对多映射'],
        '08_多对一映射检查': diagnostics['多对一映射'],
        '09_同名跨地域公司': multi_location_names,
        '10_正式时序可用公司': formal_entity_table,
        '11_正式时序排除原因': exclude_reason,
        '12_旧EXACT_NAME_LOCATION影响': diagnostics['旧EXACT_NAME_LOCATION影响'],
        '13_简介指纹共享候选': diagnostics['简介指纹共享候选'],
        '14_映射方式统计': method_counts,
    }


def build_record_lines(metrics: dict, audit: dict) -> list:
    """生成 11 号公司实体识别阶段记录。"""
    overview = audit['01_公司实体总体统计'].set_index('指标')['数值']
    lines = [
        '# 阶段记录：Stage 09 公司实体高置信度识别（Refinement R1）',
        '',
        '> 本文件由 `scripts/pipeline/09_resolve_company_entities.py` 生成，全部数字来自真实运行结果。',
        '',
        '## 1. 为什么必须先做公司实体识别',
        '',
        '一个公司可能发布多个岗位，多个岗位会重复包含**同一份公司简介**。',
        '若按「实习岗位ID」直接统计公司简介变化，岗位数量多的公司会被重复加权。',
        '因此公司简介语义时序必须建立在**公司实体层**之上。',
        '',
        '## 2. 映射方式（保守分级，本轮收紧）',
        '',
        '| 映射方式 | 判定依据 | 是否进入正式公司时序 | 默认置信度 |',
        '| --- | --- | --- | --- |',
        '| EXACT_NORMALIZED_NAME | 规范化名称完全一致 | 是 | 1.00 |',
        '| APPROVED_ALIAS | 人工确认别名（`config/company_aliases.yml`） | 是 | 0.95 |',
        '| MULTI_LOCATION_AMBIGUOUS | 同名跨地域，所在地不足以自动拆分 | **否**（人工复核） | 0.70 |',
        '| HIGH_CONFIDENCE_MATCH | 名称不同但公司简介指纹完全一致 | 否（人工确认前不合并） | 0.90 |',
        '| UNRESOLVED | 公司名称为空 | 否 | 0.00 |',
        '',
        '> **所在地不再作为自动拆分正式公司实体的充分条件**：北京/上海/深圳可能只是同一家企业不同办公地点，',
        '旧统计范围 `EXACT_NAME_LOCATION`（名称@所在地）只保留为历史诊断字段，不再作为正式进入条件。',
        '',
        '## 3. 识别结果',
        '',
        f"- 原始公司名称数：{int(overview['原始公司名称数'])}；",
        f"- 公司实体总数（含未解析）：{int(overview['公司实体总数（含未解析）'])}；",
        f"- 进入正式公司时序的实体数：{int(overview['进入正式公司时序的实体数'])}"
        f"（覆盖岗位 {int(overview['进入正式公司时序的岗位数'])} 个）；",
        f"- 同名跨地域行数：{int(overview['同名跨地域（MULTI_LOCATION_AMBIGUOUS）行数'])}"
        f"（规范名称 {int(overview['同名跨地域规范名称数'])} 个）；",
        f"- 需人工复核映射行数：{int(overview['需人工复核的映射行数'])}；",
        f"- 无法解析公司行数：{int(overview['无法解析公司行数'])}。",
        '',
        '## 4. 映射方式分布',
        '',
        '| 映射方式 | 名称数 | 实体数 | 岗位数 | 是否进入正式公司时序 |',
        '| --- | --- | --- | --- | --- |',
    ]
    for row in audit['14_映射方式统计'].itertuples(index=False):
        lines.append(f'| {row.映射方式} | {int(row.名称数)} | {int(row.实体数)} | '
                     f'{int(row.岗位数)} | {"是" if row.是否允许进入正式公司时序 else "否"} |')
    lines += [
        '',
        '## 5. 正式时序排除原因',
        '',
        '| 映射方式 | 排除原因 | 行数 | 名称数 | 岗位数 |',
        '| --- | --- | --- | --- | --- |',
    ]
    for row in audit['11_正式时序排除原因'].itertuples(index=False):
        lines.append(f'| {row.映射方式} | {row.不可进入正式公司时序原因} | {int(row.行数)} | '
                     f'{int(row.名称数)} | {int(row.岗位数)} |')
    lines += [
        '',
        '## 6. 同名跨地域（前 15 条）',
        '',
        '| 公司规范名称 | 所在地数量 | 所在地集合 | 岗位数 | 映射方式 |',
        '| --- | --- | --- | --- | --- |',
    ]
    for row in audit['09_同名跨地域公司'].head(15).itertuples(index=False):
        lines.append(f'| {row.公司规范名称} | {int(row.所在地数量)} | {row.所在地集合} | '
                     f'{int(row.岗位数)} | {row.映射方式} |')
    lines += [
        '',
        '## 7. 别名与人工复核',
        '',
        '| 规范名称 | 别名列表 | 状态 | 确认依据 |',
        '| --- | --- | --- | --- |',
    ]
    for row in audit['03_公司别名映射'].itertuples(index=False):
        lines.append(f'| {row.规范名称} | {row.别名列表} | {row.状态} | {row.确认依据} |')
    lines += [
        '',
        f"- 一对多（同一实体由多个名称合并而来）记录数：{len(audit['07_一对多映射检查'])}；",
        f"- 多对一（同一名称拆分为多个实体）记录数：{len(audit['08_多对一映射检查'])}；",
        f"- 简介指纹共享候选名称数：{len(audit['13_简介指纹共享候选'])}。",
        '',
        '## 8. 运行门禁',
        '',
        '| 门禁项 | 状态 |',
        '| --- | --- |',
    ]
    for name, result in metrics['gates'].items():
        lines.append(f'| {name} | {result} |')
    lines += [
        '',
        '## 9. 产物清单',
        '',
        f"- 公司实体映射：`{metrics['map_path']}`（{metrics['map_rows']} 行）；",
        f"- 审计表：`{metrics['audit_path']}`；",
        f"- 指标 JSON：`{metrics['metrics_path']}`。",
        '',
    ]
    return lines


def main() -> int:
    quality.configure_logging()
    gates = quality.GateRegistry(STAGE)

    quality.stage_banner(STAGE, TITLE)
    raw, versions = load_inputs()
    print(f'输入: 原始汉化观测 {len(raw)} 行 / 版本历史 {len(versions)} 行')

    name_location_stats = build_name_location_stats(raw)
    profile_stats = build_profile_stats(raw)
    print(f'公司名称—所在地组合 {len(name_location_stats)} 组，简介指纹 {len(profile_stats)} 组')

    entity_map, diagnostics = company_identity.build_company_entity_map(
        name_location_stats, profile_stats)
    formal = entity_map[entity_map[schema.COMPANY_FORMAL_FIELD] == 1]
    method_field = schema.COMPANY_MAPPING_METHOD_FIELD
    entity_field = schema.COMPANY_ENTITY_ID_FIELD

    gates.check('COMPANY_IDENTITY_BUILD',
                len(entity_map) == len(name_location_stats)
                and entity_map[entity_field].replace('', pd.NA).nunique() > 0
                and all(method in company_identity.MAPPING_METHODS
                        for method in entity_map[method_field].unique()),
                f'{len(entity_map)} 行名称映射 → {formal[entity_field].nunique()} '
                f'个正式实体，映射方式取值合法')

    ambiguous = entity_map[entity_map[schema.COMPANY_NAME_NORMALIZED_FIELD] != '']
    one_name_multi_entity = int((ambiguous.groupby(schema.COMPANY_NAME_NORMALIZED_FIELD)
                                 [entity_field].nunique() > 1).sum())
    unresolved_rows = entity_map[entity_map[method_field] == company_identity.METHOD_UNRESOLVED]
    gates.check('COMPANY_IDENTITY_AMBIGUITY_CHECK',
                len(diagnostics['多对一映射']) == 0
                and bool((entity_map[schema.COMPANY_MAPPING_CONFIDENCE_FIELD].between(0, 1)).all()),
                f'同名多实体 {one_name_multi_entity} 个（本轮已不再按所在地自动拆分，'
                f'多对一映射 {len(diagnostics["多对一映射"])} 条），无法解析 {len(unresolved_rows)} 行，'
                '置信度均在 [0,1] 区间')

    multi_location = entity_map[entity_map[method_field]
                                == company_identity.METHOD_MULTI_LOCATION_AMBIGUOUS]
    cross_location_rows = entity_map[entity_map[schema.COMPANY_CROSS_LOCATION_FIELD] == 1]
    multi_location_names = entity_map.loc[cross_location_rows.index,
                                          schema.COMPANY_NAME_NORMALIZED_FIELD].nunique()
    gates.check('MULTI_LOCATION_COMPANY_CHECK',
                int(multi_location[schema.COMPANY_FORMAL_FIELD].sum()) == 0
                and len(multi_location) == len(cross_location_rows)
                and int(multi_location[schema.COMPANY_NEED_REVIEW_FIELD].sum()) == len(multi_location)
                and not entity_map[entity_field].astype(str).str.contains('@', regex=False).any(),
                f'同名跨地域 {len(multi_location)} 行（规范名称 {int(multi_location_names)} 个）'
                '全部标记 MULTI_LOCATION_AMBIGUOUS 与人工复核，且不再生成「名称@所在地」实体')

    formal_methods = set(formal[method_field].unique())
    gates.check('FORMAL_COMPANY_IDENTITY_FILTER',
                formal_methods <= set(company_identity.FORMAL_MAPPING_METHODS)
                and not formal[entity_field].astype(str).str.contains('@', regex=False).any(),
                f'正式公司时序仅使用 {sorted(formal_methods)}，'
                f'实体数 {formal[entity_field].nunique()}，'
                f'排除 MULTI_LOCATION_AMBIGUOUS / HIGH_CONFIDENCE_MATCH / UNRESOLVED')

    path = io_utils.write_parquet(entity_map[schema.COMPANY_ENTITY_MAP_COLUMNS],
                                  project_paths.COMPANY_ENTITY_MAP_PARQUET)
    audit_tables = build_audit_tables(entity_map, diagnostics)
    audit_path = io_utils.write_excel(project_paths.TABLES_DIR / project_paths.TABLE_COMPANY_IDENTITY,
                                      audit_tables)
    gates.check('COMPANY_IDENTITY_AUDIT_EXPORT',
                path.exists() and audit_path.exists()
                and {'09_同名跨地域公司', '10_正式时序可用公司', '11_正式时序排除原因',
                     '12_旧EXACT_NAME_LOCATION影响'} <= set(audit_tables),
                f'映射表 {len(entity_map)} 行、审计表 {len(audit_tables)} 张子表已写出')

    entity_total = int(entity_map[entity_field].replace('', pd.NA).nunique())
    formal_entities = int(formal[entity_field].nunique())
    identity_adjustment = [
        {'指标': '公司实体候选数（含未解析）', '数值': entity_total,
         '说明': '同一规范化名称不再按所在地拆分，跨地域名称合并为单一候选实体'},
        {'指标': '正式时序可用公司数', '数值': formal_entities,
         '说明': '仅 EXACT_NORMALIZED_NAME / APPROVED_ALIAS 且无需人工复核'},
        {'指标': '同名跨地域歧义公司数', '数值': int(multi_location_names),
         '说明': '标记 MULTI_LOCATION_AMBIGUOUS，禁止进入正式公司时序'},
        {'指标': '同名跨地域行数', '数值': int(len(multi_location)),
         '说明': '旧统计范围会按「名称@所在地」拆分出的行数'},
        {'指标': '旧统计范围 EXACT_NAME_LOCATION 实体数', '数值': int(len(multi_location)),
         '说明': '历史兼容字段，本轮不再作为正式进入条件'},
        {'指标': '人工复核候选行数',
         '数值': int((entity_map[schema.COMPANY_NEED_REVIEW_FIELD] == 1).sum()),
         '说明': 'MULTI_LOCATION_AMBIGUOUS / HIGH_CONFIDENCE_MATCH / UNRESOLVED'},
        {'指标': '高置信度匹配候选行数',
         '数值': int(len(entity_map[entity_map[method_field]
                               == company_identity.METHOD_HIGH_CONFIDENCE_MATCH])),
         '说明': '不同名称但公司简介指纹一致，未经人工确认不合并'},
    ]
    metrics = {
        'map_path': project_paths.relative_to_root(path),
        'map_rows': int(len(entity_map)),
        'audit_path': project_paths.relative_to_root(audit_path),
        'original_names': int(entity_map[schema.COMPANY_NAME_ORIGINAL_FIELD]
                              .replace('', pd.NA).nunique()),
        'entity_candidates': entity_total,
        'formal_entities': formal_entities,
        'formal_jobs': int(formal['岗位数'].sum()),
        'review_rows': int((entity_map[schema.COMPANY_NEED_REVIEW_FIELD] == 1).sum()),
        'unresolved_rows': int(len(unresolved_rows)),
        'multi_location_rows': int(len(multi_location)),
        'multi_location_names': int(multi_location_names),
        'multi_entity_names': one_name_multi_entity,
        'identity_adjustment': identity_adjustment,
        'metrics_path': project_paths.relative_to_root(
            project_paths.METRICS_DIR / f'{STAGE}.json'),
    }
    metrics['gates'] = {name: result['status'] for name, result in gates.results.items()}
    io_utils.write_json(project_paths.METRICS_DIR / f'{STAGE}.json', metrics)

    record_path = io_utils.write_markdown(
        project_paths.RECORDS_DIR / project_paths.RECORD_COMPANY_IDENTITY,
        build_record_lines(metrics, audit_tables))
    print(f'阶段记录: {project_paths.relative_to_root(record_path)}')

    quality.print_gates(gates.results, order=quality.STAGE_GATE_MAP[STAGE])
    gates.save()
    return 0


if __name__ == '__main__':
    sys.exit(main())

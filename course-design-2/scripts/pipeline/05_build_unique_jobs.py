# -*- coding: utf-8 -*-
"""Stage 05：最终岗位实体构建（最终核心版本优先 + 原子替换）。

处理路线：

data/processed/job_version_history.parquet     岗位版本时序层
data/processed/job_change_events.parquet       字段变化事件层
data/processed/job_category_membership.parquet 岗位分类关系层
        ↓  重算观测版本号并与版本历史逐项核对（版本还原）
        ↓  取每个岗位的最终核心版本，只在该版本内选代表观测
        ↓  业务字段完整度最高 → 数据更新时间最新 → 数据创建时间最新 → 稳定原始顺序第一条
        ↓
data/processed/job_details_unique_v2.parquet   临时文件（回读/唯一性/还原校验）
        ↓  全部 PASS 后原子替换
data/processed/job_details_unique.parquet      最终岗位实体层

实体选择规则已由「全历史完整度优先」改为「最终版本优先，完整度其次」。
用法：
    python scripts/pipeline/05_build_unique_jobs.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = next(
    _candidate for _candidate in Path(__file__).resolve().parents
    if (_candidate / "data").is_dir() and (_candidate / "scripts").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src import dedup, io_utils, project_paths, quality, schema, versioning  # noqa: E402

STAGE = 'stage_05'
TITLE = 'Stage 05 最终岗位实体构建'


def reconcile_versions(versioned: pd.DataFrame, versions: pd.DataFrame) -> dict:
    """版本还原校验：由观测快照重算的版本必须与版本历史完全一致。"""
    id_field = schema.ID_FIELD
    version_field = schema.CORE_VERSION_FIELD
    observed = versioned.groupby(id_field).agg(**{
        '重算核心版本数': (version_field, 'max'),
        '重算观测次数': (schema.OBSERVATION_TIME_FIELD, 'size'),
    })
    history = versions.groupby(id_field).agg(**{
        '历史核心版本数': (version_field, 'max'),
        '历史观测次数': (schema.VERSION_OBS_COUNT_FIELD, 'sum'),
    })
    final_rows = versioning.select_final_version(versions).set_index(id_field)
    joined = observed.join(history, how='outer')
    joined['最终版本首次观测时间'] = final_rows[schema.VERSION_FIRST_TIME_FIELD]
    joined['最终版本末次观测时间'] = final_rows[schema.VERSION_LAST_TIME_FIELD]
    recomputed_final = versioned.loc[
        versioned[version_field] == versioned.groupby(id_field)[version_field].transform('max')]
    recomputed_bounds = recomputed_final.groupby(id_field)[schema.OBSERVATION_TIME_FIELD].agg(
        ['min', 'max'])
    joined['重算首次'] = recomputed_bounds['min']
    joined['重算末次'] = recomputed_bounds['max']

    # 版本号 → 签名 映射必须逐一一致（A→B→A 场景下签名数可能少于版本数，故比较映射而非计数）
    recomputed_map = versioned.groupby([id_field, version_field])[
        schema.CORE_SIGNATURE_FIELD].first()
    history_map = versions.set_index([id_field, version_field])[schema.CORE_SIGNATURE_FIELD]

    checks = {
        'core_version_match': bool((joined['重算核心版本数'] == joined['历史核心版本数']).all()),
        'observation_count_match': bool((joined['重算观测次数'] == joined['历史观测次数']).all()),
        'signature_map_match': bool(
            (recomputed_map.reindex(history_map.index) == history_map).all()),
        'final_bound_match': bool((joined['重算首次'] == joined['最终版本首次观测时间']).all()
                                  and (joined['重算末次'] == joined['最终版本末次观测时间']).all()),
    }
    return checks


def build_entity_audit_tables(versions: pd.DataFrame, version_checks: dict,
                              category_checks: dict, metrics: dict) -> dict:
    """构造 15 号最终实体审计表集合。"""
    id_field = schema.ID_FIELD
    overall = pd.DataFrame([
        {'指标': '新最终唯一岗位数', '数值': metrics['unique_job_rows']},
        {'指标': '实习岗位ID唯一性', '数值': metrics['id_unique']},
        {'指标': '规范化URL唯一性', '数值': metrics['url_unique']},
        {'指标': '最终实体表列数', '数值': metrics['unique_job_columns']},
        {'指标': '核心版本总数', '数值': metrics['core_versions']},
        {'指标': '完整页面版本总数', '数值': metrics['full_versions']},
        {'指标': '多版本岗位数', '数值': metrics['multi_version_jobs']},
        {'指标': '版本还原检查', '数值': '通过' if all(version_checks.values()) else '未通过'},
        {'指标': '分类还原检查', '数值': '通过' if all(category_checks.values()) else '未通过'},
        {'指标': '临时表是否已原子替换', '数值': metrics['atomic_replaced']},
    ])

    core_counts = versions.groupby(id_field)[schema.CORE_VERSION_FIELD].max()
    final_version = core_counts.value_counts().sort_index().rename('岗位数量').reset_index()
    final_version.columns = ['核心版本数', '岗位数量']

    version_check_table = pd.DataFrame([
        {'检查项': '每个岗位的核心版本号与版本历史一致', '结果': version_checks['core_version_match']},
        {'检查项': '版本观测次数合计与观测快照数一致', '结果': version_checks['observation_count_match']},
        {'检查项': '版本号—核心签名映射与版本历史逐一一致',
         '结果': version_checks['signature_map_match']},
        {'检查项': '最终版本时间区间与版本历史一致', '结果': version_checks['final_bound_match']},
    ])
    category_check_table = pd.DataFrame([
        {'检查项': '分类集合可由关系表还原', '结果': category_checks['group_set_match']},
        {'检查项': '细分类集合可由关系表还原', '结果': category_checks['item_set_match']},
        {'检查项': '关系表岗位ID集合与实体岗位ID集合一致', '结果': category_checks['id_set_match']},
        {'检查项': '岗位大类计数与关系表一致', '结果': category_checks['group_count_match']},
        {'检查项': '岗位细分类计数与关系表一致', '结果': category_checks['item_count_match']},
        {'检查项': '原始记录数与关系表命中次数合计一致', '结果': category_checks['record_count_match']},
    ])

    field_rows = []
    for index, column in enumerate(schema.ENTITY_JOB_COLUMNS, start=1):
        description = (schema.DERIVED_FIELD_DESCRIPTION.get(column)
                       or schema.COLUMN_DESCRIPTION.get(column, ''))
        field_rows.append({'序号': index, '字段名': column, '字段说明': description})
    field_table = pd.DataFrame(field_rows)

    return {
        '01_总体统计': overall,
        '02_最终版本统计': final_version,
        '03_版本还原检查': version_check_table,
        '04_分类还原检查': category_check_table,
        '05_字段清单': field_table,
    }


def write_record(metrics: dict, overall: pd.DataFrame,
                 basis_table: pd.DataFrame, version_checks: pd.DataFrame,
                 gate_table: pd.DataFrame | None) -> list:
    """构造 Stage 05 阶段记录。"""
    lines = [
        '# 阶段 05 记录：最终岗位实体构建',
        '',
        '> 本文件由 `scripts/pipeline/05_build_unique_jobs.py` 自动生成，数字均来自真实运行结果。',
        '',
        '## 1. 输入与输出',
        '',
        '| 项 | 内容 |',
        '| --- | --- |',
        f'| 输入 | `{project_paths.relative_to_root(project_paths.PROCESSED_VERSION_HISTORY_PARQUET)}`'
        f'（{metrics["core_versions"]} 个核心版本） |',
        f'| 输入 | '
        f'`{project_paths.relative_to_root(project_paths.PROCESSED_CATEGORY_MEMBERSHIP_PARQUET)}`'
        f'（{metrics["membership_rows"]} 行分类关系） |',
        f'| 临时输出 | '
        f'`{project_paths.relative_to_root(project_paths.PROCESSED_UNIQUE_PARQUET_V2)}`'
        f'（{metrics["unique_job_rows"]} 行 × {metrics["unique_job_columns"]} 列） |',
        f'| 正式输出 | '
        f'`{project_paths.relative_to_root(project_paths.PROCESSED_UNIQUE_PARQUET)}`'
        f'（原子替换） |',
        '',
        '## 2. 实体选择规则',
        '',
        '第一步：确定每个岗位的最终核心版本；',
        '第二步：只在最终核心版本内部选代表观测；',
        '第三步：业务字段完整度最高 → 数据更新时间最新 → 数据创建时间最新 → 稳定原始顺序第一条。',
        '',
        '即 **版本优先，完整度其次**；不再使用「全历史记录中完整度最高」作为首要规则。',
        '',
    ]
    if len(basis_table):
        lines += [
            '| 代表观测选择依据 | 岗位数量 |',
            '| --- | --- |',
            *[f'| {row.代表记录选择依据} | {row.岗位数量} |'
              for row in basis_table.itertuples(index=False)],
            '',
        ]
    lines += [
        '## 3. 版本与分类还原',
        '',
        '| 检查项 | 结果 |',
        '| --- | --- |',
        *[f'| {row.检查项} | {row.结果} |' for row in version_checks.itertuples(index=False)],
        '',
        '## 4. 总体统计',
        '',
        '| 指标 | 数值 |',
        '| --- | --- |',
        *[f'| {row.指标} | {row.数值} |' for row in overall.itertuples(index=False)],
        '',
        '## 5. 输出文件',
        '',
        f'- `{project_paths.relative_to_root(project_paths.PROCESSED_UNIQUE_PARQUET)}`（最终岗位实体层）；',
        f'- `outputs/tables/{project_paths.TABLE_ENTITY_AUDIT}`；',
        '- 分类聚合字段由 `job_category_membership.parquet` 恢复，未回写原始观测明细。',
        '',
    ]
    if gate_table is not None:
        lines += [
            '## 6. 运行门禁',
            '',
            '| 门禁项 | 状态 | 说明 |',
            '| --- | --- | --- |',
            *[f'| {row.门禁项} | {row.状态} | {row.说明} |'
              for row in gate_table.itertuples(index=False)],
            '',
        ]
    lines += [
        '## 7. 本阶段边界',
        '',
        '- 未进入薪资解析、缺失值填补、EDA、统计检验或机器学习。',
        '',
    ]
    return lines


def main() -> int:
    project_paths.ensure_directories()
    quality.configure_logging()
    gates = quality.GateRegistry(STAGE)

    quality.stage_banner(STAGE, TITLE)
    id_field = schema.ID_FIELD
    version_field = schema.CORE_VERSION_FIELD

    snapshots = io_utils.read_parquet(project_paths.OBSERVATION_SNAPSHOT_PARQUET)
    versions = io_utils.read_parquet(project_paths.PROCESSED_VERSION_HISTORY_PARQUET)
    events = io_utils.read_parquet(project_paths.PROCESSED_CHANGE_EVENTS_PARQUET)
    membership = io_utils.read_parquet(project_paths.PROCESSED_CATEGORY_MEMBERSHIP_PARQUET)
    print(f'输入: 观测快照 {len(snapshots)} 行 / 版本 {len(versions)} 行 / '
          f'变化事件 {len(events)} 行 / 分类关系 {len(membership)} 行')

    # 1) 版本还原：重算版本号并与版本历史核对
    versioned = versioning.assign_snapshot_versions(snapshots)
    version_checks = reconcile_versions(versioned, versions)
    gates.check('VERSION_TO_ENTITY_RECONCILE',
                all(version_checks.values()),
                '由观测快照重算的版本号、观测次数、签名数与最终版本时间区间与版本历史完全一致')

    # 2) 最终核心版本内部选代表观测
    final_flag = (versioned[version_field]
                  == versioned.groupby(id_field)[version_field].transform('max'))
    final_observations = versioned.loc[final_flag].copy()
    # 观测快照中的「岗位详情链接」已是规范化链接，供实体表复用
    final_observations[schema.URL_NORM_FIELD] = final_observations[schema.URL_FIELD]
    representative, basis_table = dedup.select_representatives(
        final_observations, business_fields=schema.BUSINESS_FIELDS, time_fields=schema.TIME_FIELDS)
    unique_ids = int(snapshots[id_field].nunique())
    gates.check('FINAL_VERSION_SELECT',
                len(representative) == unique_ids
                and bool((representative[version_field]
                          == representative[id_field].map(
                              final_observations.groupby(id_field)[version_field].max())).all()),
                f'{len(representative)} 个岗位均在最终核心版本内完成代表观测选择')

    # 3) 分类聚合（权威来源：关系表）+ 冲突汇总 + 版本摘要
    category_aggregated = dedup.aggregate_categories_from_membership(membership)
    multi_value_ids = {field: list(versioning.field_multi_value_ids(snapshots, field))
                       for field in schema.BUSINESS_FIELDS}
    conflict_summary = dedup.summarize_conflicts(
        snapshots, multi_value_ids, business_fields=schema.BUSINESS_FIELDS,
        key_fields=schema.KEY_CONFLICT_FIELDS)
    version_summary = versioning.version_job_summary(versions)
    entity_summary = version_summary.join(
        versioning.select_final_version(versions).set_index(id_field)[
            [version_field, schema.VERSION_FIRST_TIME_FIELD, schema.VERSION_LAST_TIME_FIELD]]
        .rename(columns={version_field: schema.ENTITY_FINAL_VERSION_FIELD,
                         schema.VERSION_FIRST_TIME_FIELD: schema.ENTITY_FINAL_FIRST_TIME_FIELD,
                         schema.VERSION_LAST_TIME_FIELD: schema.ENTITY_FINAL_LAST_TIME_FIELD}))

    entity_v2 = dedup.build_unique_jobs(representative, category_aggregated,
                                        conflict_summary, entity_summary=entity_summary)
    if list(entity_v2.columns) != schema.ENTITY_JOB_COLUMNS:
        raise ValueError('最终实体表列与 schema.ENTITY_JOB_COLUMNS 不一致')
    for forbidden in [*schema.FORBIDDEN_UNIQUE_FIELDS, '来源搜索页码集合']:
        if forbidden in entity_v2.columns:
            raise ValueError(f'最终实体表不应包含字段: {forbidden}')
    gates.check('UNIQUE_ENTITY_BUILD',
                len(entity_v2) == unique_ids
                and list(entity_v2.columns) == schema.ENTITY_JOB_COLUMNS,
                f'{len(entity_v2)} 行 × {entity_v2.shape[1]} 列')

    # 4) 先写临时文件并回读校验
    io_utils.write_parquet(entity_v2, project_paths.PROCESSED_UNIQUE_PARQUET_V2, verify=True)
    reloaded_v2 = io_utils.read_parquet(project_paths.PROCESSED_UNIQUE_PARQUET_V2)
    gates.check('UNIQUE_ENTITY_ID_CHECK',
                bool(reloaded_v2[id_field].is_unique) and not reloaded_v2[id_field].isna().any(),
                f'实习岗位ID唯一 {bool(reloaded_v2[id_field].is_unique)}，无缺失')
    gates.check('UNIQUE_ENTITY_URL_CHECK',
                bool(reloaded_v2[schema.URL_FIELD].is_unique)
                and not reloaded_v2[schema.URL_FIELD].isna().any()
                and not reloaded_v2[schema.URL_FIELD].str.contains('pcm=', regex=False).any(),
                f'规范化岗位详情链接唯一 {bool(reloaded_v2[schema.URL_FIELD].is_unique)}，无跟踪参数残留')
    gates.check('COMPANY_CERT_FIELD_CHECK',
                schema.CERT_TAG_FIELD in reloaded_v2.columns
                and schema.CERT_SOURCE_FIELD not in reloaded_v2.columns,
                f'含「{schema.CERT_TAG_FIELD}」且不含「{schema.CERT_SOURCE_FIELD}」')

    # 5) 分类还原校验（关系表 ↔ 实体表）
    membership_reconcile = dedup.reconcile_category_membership(
        membership, reloaded_v2.set_index(id_field)[schema.CATEGORY_SET_FIELDS])
    membership_counts = membership.groupby(id_field)['原始命中次数'].sum()
    category_checks = {
        'group_set_match': membership_reconcile['group_set_match'],
        'item_set_match': membership_reconcile['item_set_match'],
        'id_set_match': membership_reconcile['id_set_match'],
        'group_count_match': bool(_counts_match(
            membership, schema.CATEGORY_GROUP_FIELD,
            reloaded_v2.set_index(id_field)[schema.CATEGORY_COUNT_FIELDS[0]])),
        'item_count_match': bool(_counts_match(
            membership, schema.CATEGORY_ITEM_FIELD,
            reloaded_v2.set_index(id_field)[schema.CATEGORY_COUNT_FIELDS[1]])),
        'record_count_match': bool(
            (membership_counts.reindex(reloaded_v2[id_field]).to_numpy()
             == reloaded_v2[schema.ORIGINAL_RECORD_FIELD].to_numpy()).all()),
    }
    gates.check('CATEGORY_TO_ENTITY_RECONCILE',
                all(category_checks.values()),
                '实体表分类集合、计数、原始记录数均可由关系表完整还原')

    # 6) 原子替换正式实体表
    os.replace(project_paths.PROCESSED_UNIQUE_PARQUET_V2,
               project_paths.PROCESSED_UNIQUE_PARQUET)
    final_reloaded = io_utils.read_parquet(project_paths.PROCESSED_UNIQUE_PARQUET)
    atomic_ok = (not project_paths.PROCESSED_UNIQUE_PARQUET_V2.exists()
                 and len(final_reloaded) == len(entity_v2)
                 and list(final_reloaded.columns) == schema.ENTITY_JOB_COLUMNS
                 and bool(final_reloaded[id_field].is_unique)
                 and bool(final_reloaded[schema.URL_FIELD].is_unique))
    if not atomic_ok:
        raise ValueError('原子替换后校验失败')
    print(f'原子替换完成: {project_paths.relative_to_root(project_paths.PROCESSED_UNIQUE_PARQUET)}'
          f'（{len(final_reloaded)} 行 × {final_reloaded.shape[1]} 列）')

    metrics = {
        'unique_job_rows': int(len(entity_v2)),
        'unique_job_columns': int(entity_v2.shape[1]),
        'raw_rows': int(entity_v2[schema.ORIGINAL_RECORD_FIELD].sum()),
        'merged_duplicate_rows': int(entity_v2[schema.ORIGINAL_RECORD_FIELD].sum()
                                     - len(entity_v2)),
        'core_versions': int(len(versions)),
        'full_versions': int(versions.groupby(id_field)[schema.FULL_VERSION_FIELD].max().sum()),
        'multi_version_jobs': int((versions.groupby(id_field)[version_field].max() > 1).sum()),
        'membership_rows': int(len(membership)),
        'id_unique': bool(final_reloaded[id_field].is_unique),
        'url_unique': bool(final_reloaded[schema.URL_FIELD].is_unique),
        'atomic_replaced': bool(atomic_ok),
    }
    io_utils.write_json(project_paths.METRICS_DIR / f'{STAGE}.json', metrics)

    audit_tables = build_entity_audit_tables(
        versions, version_checks, category_checks, metrics)
    io_utils.write_excel(project_paths.TABLES_DIR / project_paths.TABLE_ENTITY_AUDIT, audit_tables)
    gates.check('ENTITY_AUDIT_EXPORT',
                (project_paths.TABLES_DIR / project_paths.TABLE_ENTITY_AUDIT).exists(),
                f'{len(audit_tables)} 个 Sheet 已生成')

    record_path = project_paths.RECORDS_DIR / project_paths.RECORD_FINAL_ENTITY
    record_args = (metrics, audit_tables['01_总体统计'], basis_table,
                   audit_tables['03_版本还原检查'])
    io_utils.write_markdown(record_path, write_record(*record_args, gate_table=None))
    gate_table = quality.summarize_gates(gates.results, order=quality.STAGE_GATE_MAP[STAGE])
    io_utils.write_markdown(record_path, write_record(*record_args, gate_table=gate_table))
    print(f'阶段记录: {project_paths.relative_to_root(record_path)}')

    gates.save()
    print(f'Stage 05 完成: 最终岗位实体 {len(final_reloaded)} 行 × {final_reloaded.shape[1]} 列')
    return 0


def _counts_match(membership: pd.DataFrame, field: str, expected: pd.Series) -> bool:
    """校验实体表的类别计数 JSON 是否与关系表累计命中次数一致。"""
    id_field = schema.ID_FIELD
    aggregated = membership.groupby(id_field).apply(
        lambda part: dedup.category_counts_json(part, field), include_groups=False)
    return bool((aggregated.reindex(expected.index) == expected).all())


if __name__ == '__main__':
    sys.exit(main())

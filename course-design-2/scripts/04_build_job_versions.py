# -*- coding: utf-8 -*-
"""Stage 04：岗位版本时序构建（双签名 + 版本压缩 + 变化事件）。

处理路线：

data/interim/job_observation_snapshots.parquet
        ↓  按 观测时间 → 数据创建时间 → 稳定原始顺序 排序
        ↓  核心业务签名（不含发布时间/投递截止日期/岗位头图链接）
        ↓  完整页面签名（核心字段 + 发布时间/投递截止日期/岗位头图链接）
        ↓  连续相同签名压缩为一个版本（A→B→A 保留为三个版本，禁止重新合并）
        ↓
data/processed/job_version_history.parquet     岗位版本时序层
data/processed/job_change_events.parquet       字段变化事件层
outputs/tables/14_job_version_audit.xlsx

用法：
    python scripts/04_build_job_versions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src import io_utils, project_paths, quality, schema, versioning  # noqa: E402

STAGE = 'stage_04'
TITLE = 'Stage 04 岗位版本时序构建'


def build_audit_tables(versions: pd.DataFrame, events: pd.DataFrame, snapshots: pd.DataFrame,
                       field_table: pd.DataFrame, aba: pd.DataFrame,
                       time_issues: pd.DataFrame, metrics: dict) -> dict:
    """构造 14 号岗位版本审计表集合。"""
    id_field = schema.ID_FIELD
    version_field = schema.CORE_VERSION_FIELD
    full_field = schema.FULL_VERSION_FIELD

    per_job_core = versions.groupby(id_field)[version_field].max()
    per_job_full = versions.groupby(id_field)[full_field].max()

    overall = pd.DataFrame([
        {'指标': '观测快照总数', '数值': metrics['observation_snapshots']},
        {'指标': '核心版本总数', '数值': metrics['core_versions']},
        {'指标': '完整页面版本总数', '数值': metrics['full_versions']},
        {'指标': '平均核心版本数/岗位', '数值': metrics['avg_core_versions_per_job']},
        {'指标': '核心版本数 P50', '数值': metrics['core_versions_p50']},
        {'指标': '核心版本数 P95', '数值': metrics['core_versions_p95']},
        {'指标': '核心版本数 Max', '数值': metrics['core_versions_max']},
        {'指标': '单版本岗位数', '数值': metrics['single_version_jobs']},
        {'指标': '多版本岗位数', '数值': metrics['multi_version_jobs']},
        {'指标': '多版本岗位比例', '数值': metrics['multi_version_ratio']},
        {'指标': '仅页面刷新岗位数', '数值': metrics['page_refresh_only_jobs']},
        {'指标': 'A→B→A 岗位数', '数值': metrics['aba_jobs']},
        {'指标': '变化事件总数', '数值': metrics['change_events']},
        {'指标': '薪资变化岗位数', '数值': metrics['salary_changed_jobs']},
        {'指标': '城市变化岗位数', '数值': metrics['city_changed_jobs']},
        {'指标': '公司变化岗位数', '数值': metrics['company_changed_jobs']},
        {'指标': '标题变化岗位数', '数值': metrics['title_changed_jobs']},
        {'指标': '岗位描述变化岗位数', '数值': metrics['description_changed_jobs']},
        {'指标': '版本时间异常条数', '数值': metrics['version_time_issues']},
    ])

    core_distribution = (per_job_core.value_counts().sort_index()
                         .rename('岗位数量').reset_index())
    core_distribution.columns = ['核心版本数', '岗位数量']
    full_distribution = (per_job_full.value_counts().sort_index()
                         .rename('岗位数量').reset_index())
    full_distribution.columns = ['完整页面版本数', '岗位数量']

    compare = pd.DataFrame({
        id_field: per_job_core.index,
        '核心版本数': per_job_core.to_numpy(),
        '完整页面版本数': per_job_full.reindex(per_job_core.index).to_numpy(),
    })
    compare['完整与核心版本差额'] = compare['完整页面版本数'] - compare['核心版本数']
    compare['是否仅页面刷新'] = (compare['完整与核心版本差额'] > 0).map({True: '是', False: '否'})
    latest_title = versions.sort_values([id_field, version_field]).groupby(id_field)['岗位标题'].last()
    latest_company = versions.sort_values([id_field, version_field]).groupby(id_field)['公司名称'].last()
    compare['岗位标题'] = compare[id_field].map(latest_title)
    compare['公司名称'] = compare[id_field].map(latest_company)
    compare = compare.sort_values('完整与核心版本差额', ascending=False).reset_index(drop=True)

    def _changed_jobs(field: str) -> pd.DataFrame:
        subset = events[events['变化字段'] == field]
        return subset.reset_index(drop=True)

    aba_table = (aba if not aba.empty else pd.DataFrame(
        [{'说明': '未发现 A→B→A 回退岗位（同一核心业务状态未重复出现）'}]))
    time_table = (time_issues if not time_issues.empty else pd.DataFrame(
        [{'异常类型': '无', '说明': '版本时间顺序、版本号连续性、完整/核心版本关系全部正常'}]))

    return {
        '01_总体版本统计': overall,
        '02_核心版本数分布': core_distribution,
        '03_完整版本数分布': full_distribution,
        '04_核心vs完整版本': compare,
        '05_变化字段统计': field_table,
        '06_薪资变化岗位': _changed_jobs('薪资信息'),
        '07_城市变化岗位': _changed_jobs('工作城市'),
        '08_公司变化岗位': _changed_jobs('公司名称'),
        '09_标题变化岗位': _changed_jobs('岗位标题'),
        '10_A-B-A回退检查': aba_table,
        '11_版本时间异常检查': time_table,
    }


def write_record(metrics: dict, overall: pd.DataFrame, versions: pd.DataFrame,
                 events: pd.DataFrame, gate_table: pd.DataFrame | None) -> list:
    """构造 Stage 04 阶段记录。"""
    lines = [
        '# 阶段 04 记录：岗位版本时序构建',
        '',
        '> 本文件由 `scripts/04_build_job_versions.py` 自动生成，数字均来自真实运行结果。',
        '',
        '## 1. 输入与输出',
        '',
        '| 项 | 内容 |',
        '| --- | --- |',
        f'| 输入 | `{project_paths.relative_to_root(project_paths.OBSERVATION_SNAPSHOT_PARQUET)}`'
        f'（{metrics["observation_snapshots"]} 行观测快照） |',
        f'| 版本时序输出 | '
        f'`{project_paths.relative_to_root(project_paths.PROCESSED_VERSION_HISTORY_PARQUET)}`'
        f'（{len(versions)} 行 × {versions.shape[1]} 列） |',
        f'| 变化事件输出 | '
        f'`{project_paths.relative_to_root(project_paths.PROCESSED_CHANGE_EVENTS_PARQUET)}`'
        f'（{len(events)} 行） |',
        '',
        '## 2. 总体版本统计',
        '',
        '| 指标 | 数值 |',
        '| --- | --- |',
        *[f'| {row.指标} | {row.数值} |' for row in overall.itertuples(index=False)],
        '',
        '## 3. 两套签名',
        '',
        f'- 核心业务签名字段（{len(schema.CORE_SIGNATURE_FIELDS)} 个）：'
        f'{"、".join(schema.CORE_SIGNATURE_FIELDS)}；',
        f'- 完整页面签名额外字段：{"、".join(schema.FULL_SIGNATURE_EXTRA_FIELDS)}；',
        '- 签名实现：归一化（去空值/排序/去空白）→ 稳定 JSON → SHA256，禁止使用内存 hash；',
        '- 排除在核心签名之外的字段：'
        f'{"、".join(schema.CORE_SIGNATURE_EXCLUDED_FIELDS)}。',
        '',
        '## 4. 版本压缩规则',
        '',
        '- 排序字段：观测时间 → 数据创建时间 → 稳定原始顺序；',
        '- 连续相同核心业务签名压缩为一个核心版本；完整页面版本同步压缩；',
        '- `A → B → A` 保留为三个连续版本，禁止把版本 1 与版本 3 重新合并；',
        '- 版本时间定义：版本首次/末次观测时间取自该版本内观测快照的时间极值；'
        '最终版本的下一版本开始时间与到下一版本间隔均为空，不填任何人为时间。',
        '',
        '## 5. 变化事件',
        '',
        f'- 变化事件行数：{len(events)}；',
        f'- 薪资变化岗位数：{metrics["salary_changed_jobs"]}；',
        f'- 城市变化岗位数：{metrics["city_changed_jobs"]}；',
        f'- 公司变化岗位数：{metrics["company_changed_jobs"]}；',
        f'- 标题变化岗位数：{metrics["title_changed_jobs"]}；',
        f'- 岗位描述变化岗位数：{metrics["description_changed_jobs"]}；',
        '',
        '## 6. 用途边界',
        '',
        '- `job_version_history.parquet` 仅用于数据质量分析、岗位更新规律、版本演化与稳健性分析；',
        '- 不同岗位版本数不同，**不直接**作为薪资模型训练集，避免多版本岗位被重复加权；',
        '- 主模型训练使用 Stage 05 的一岗一行最终实体表。',
        '',
    ]
    if gate_table is not None:
        lines += [
            '## 7. 运行门禁',
            '',
            '| 门禁项 | 状态 | 说明 |',
            '| --- | --- | --- |',
            *[f'| {row.门禁项} | {row.状态} | {row.说明} |'
              for row in gate_table.itertuples(index=False)],
            '',
        ]
    lines += [
        '## 8. 本阶段边界',
        '',
        '- 未生成最终实体表（属 Stage 05）；',
        '- 未进入薪资清洗、缺失值填补、EDA、统计检验或机器学习。',
        '',
    ]
    return lines


def main() -> int:
    project_paths.ensure_directories()
    quality.configure_logging()
    gates = quality.GateRegistry(STAGE)

    quality.stage_banner(STAGE, TITLE)
    snapshots = io_utils.read_parquet(project_paths.OBSERVATION_SNAPSHOT_PARQUET)
    id_field = schema.ID_FIELD
    version_field = schema.CORE_VERSION_FIELD
    full_field = schema.FULL_VERSION_FIELD
    print(f'输入: {len(snapshots)} 行观测快照 × {snapshots.shape[1]} 列')

    versions, versioned_snapshots, events, summary = versioning.build_version_history(
        snapshots, core_fields=schema.CORE_SIGNATURE_FIELDS,
        full_fields=schema.FULL_SIGNATURE_FIELDS)
    per_job_core = versions.groupby(id_field)[version_field].max()
    per_job_full = versions.groupby(id_field)[full_field].max()
    print(f'核心版本 {len(versions)} 个，完整页面版本 {int(per_job_full.sum())} 个，'
          f'变化事件 {len(events)} 行')

    # 1) 签名与版本序列门禁
    signature_counts = versions.groupby([id_field, version_field])[
        schema.CORE_SIGNATURE_FIELD].nunique()
    ordered_versions = versions.sort_values([id_field, version_field], kind='stable')
    adjacent_same = (ordered_versions.groupby(id_field, sort=False)[schema.CORE_SIGNATURE_FIELD]
                     .transform(lambda series: series.eq(series.shift())).astype('int64').sum())
    gates.check('CORE_SIGNATURE_BUILD',
                int(signature_counts.max()) == 1 and int(adjacent_same) == 0,
                f'{len(versions)} 个核心版本各自对应唯一核心签名，'
                f'相邻核心版本签名必然不同（同一状态再次出现时保留为独立版本）')
    full_signature_counts = versions.groupby([id_field, version_field])[
        schema.FULL_SIGNATURE_FIELD].nunique()
    gates.check('FULL_SIGNATURE_BUILD',
                int(full_signature_counts.max()) == 1
                and bool((per_job_full >= per_job_core).all()),
                f'完整页面版本 {int(per_job_full.sum())} 个，且完整版本数 >= 核心版本数')

    ordered = versions.sort_values([id_field, version_field], kind='stable')
    version_bounds = ordered.groupby(id_field)[version_field].agg(['min', 'max', 'size'])
    contiguous = ((version_bounds['min'] == 1)
                  & (version_bounds['max'] == version_bounds['size']))
    full_non_decreasing = versioned_snapshots.groupby(id_field)[full_field].apply(
        lambda series: bool(series.is_monotonic_increasing))
    gates.check('VERSION_SEQUENCE_BUILD',
                bool(contiguous.all()) and bool(full_non_decreasing.all()),
                '同一岗位核心版本号严格为 1..K，完整页面版本号非递减')

    time_issues = versioning.check_version_time_order(versions, snapshots, versioned_snapshots)
    gates.check('VERSION_TIME_ORDER',
                len(time_issues) == 0,
                '版本时间顺序、版本号连续性、负间隔检查全部通过'
                if len(time_issues) == 0 else f'存在 {len(time_issues)} 条版本时间异常')

    boundary_ok = bool((versions[schema.VERSION_NEXT_TIME_FIELD].isna()
                        | (versions[schema.VERSION_NEXT_TIME_FIELD]
                           > versions[schema.VERSION_LAST_TIME_FIELD])).all())
    final_gap_empty = bool(versions.loc[versions[schema.VERSION_IS_FINAL_FIELD] == 1,
                                        [schema.VERSION_NEXT_TIME_FIELD,
                                         schema.VERSION_GAP_FIELD]].isna().all().all())
    gates.check('VERSION_CONTIGUITY',
                boundary_ok and final_gap_empty
                and int(versions[schema.VERSION_OBS_COUNT_FIELD].sum()) == len(snapshots),
                f'版本观测次数合计 {int(versions[schema.VERSION_OBS_COUNT_FIELD].sum())}'
                f' = 观测快照总数，版本时间区间不重叠且最终版本结束时间留空')

    expectations = versions[schema.VERSION_CHANGED_COUNT_FIELD].sum()
    gates.check('CHANGE_EVENT_BUILD',
                int(expectations) == len(events)
                and bool(events['变化字段'].isin(schema.CORE_SIGNATURE_FIELDS).all())
                and bool((events['旧值'] != events['新值']).all()),
                f'变化字段数合计 {int(expectations)} 与变化事件行数一致，且旧值 != 新值')

    # 2) 同岗位历史字段多值统计（与旧口径一致，用于回归核验）
    field_table = versioning.field_multi_value_stats(snapshots, schema.BUSINESS_FIELDS)
    conflict_fields = field_table[field_table['同岗位多值岗位数'] > 0]
    key_conflict_ids = set()
    for field in schema.KEY_CONFLICT_FIELDS:
        key_conflict_ids |= versioning.field_multi_value_ids(snapshots, field)
    top_field = field_table.iloc[0] if len(field_table) else None
    gates.check('FIELD_CONSISTENCY_AUDIT',
                len(field_table) == len(schema.BUSINESS_FIELDS),
                f'{len(field_table)} 个业务字段全部完成同岗位历史多值审计，'
                f'{len(conflict_fields)} 个字段存在多值')

    # 3) A→B→A 回退检查与变化岗位统计
    aba = versioning.find_aba_jobs(versions)
    changed_jobs = {entity_field: int(versions.groupby(id_field)[flag_field].max().sum())
                    for flag_field, entity_field in schema.VERSION_FLAG_TO_ENTITY_FIELD.items()}

    io_utils.write_parquet(versions, project_paths.PROCESSED_VERSION_HISTORY_PARQUET, verify=True)
    io_utils.write_parquet(events, project_paths.PROCESSED_CHANGE_EVENTS_PARQUET, verify=True)
    reloaded_versions = io_utils.read_parquet(project_paths.PROCESSED_VERSION_HISTORY_PARQUET)
    reloaded_events = io_utils.read_parquet(project_paths.PROCESSED_CHANGE_EVENTS_PARQUET)
    if list(reloaded_versions.columns) != schema.VERSION_HISTORY_COLUMNS:
        raise ValueError('版本历史表列与 schema.VERSION_HISTORY_COLUMNS 不一致')
    if list(reloaded_events.columns) != schema.CHANGE_EVENT_COLUMNS:
        raise ValueError('变化事件表列与 schema.CHANGE_EVENT_COLUMNS 不一致')
    print(f'回读校验通过: 版本 {len(reloaded_versions)} 行，变化事件 {len(reloaded_events)} 行')

    # 4) 指标与审计表
    metrics = {
        'observation_snapshots': int(len(snapshots)),
        'core_versions': int(len(versions)),
        'full_versions': int(per_job_full.sum()),
        'avg_core_versions_per_job': round(float(per_job_core.mean()), 4),
        'core_versions_p50': float(per_job_core.quantile(0.5)),
        'core_versions_p95': float(per_job_core.quantile(0.95)),
        'core_versions_max': int(per_job_core.max()),
        'single_version_jobs': int((per_job_core == 1).sum()),
        'multi_version_jobs': int((per_job_core > 1).sum()),
        'multi_version_ratio': round(float((per_job_core > 1).mean()), 6),
        'page_refresh_only_jobs': int((per_job_full > per_job_core).sum()),
        'aba_jobs': int(len(aba)),
        'change_events': int(len(events)),
        'version_time_issues': int(len(time_issues)),
        'salary_changed_jobs': changed_jobs['历史是否发生薪资变化'],
        'city_changed_jobs': changed_jobs['历史是否发生城市变化'],
        'company_changed_jobs': changed_jobs['历史是否发生公司变化'],
        'title_changed_jobs': changed_jobs['历史是否发生标题变化'],
        'description_changed_jobs': changed_jobs['历史是否发生岗位描述变化'],
        'multi_value_business_fields': int(len(conflict_fields)),
        'business_conflict_rows': int(field_table['同岗位多值岗位数'].sum()),
        'key_conflict_jobs': int(len(key_conflict_ids)),
        'top_conflict_field': (str(top_field['字段名称']) if top_field is not None else ''),
        'top_conflict_field_groups': (int(top_field['同岗位多值岗位数'])
                                      if top_field is not None else 0),
    }
    io_utils.write_json(project_paths.METRICS_DIR / f'{STAGE}.json', metrics)

    audit_tables = build_audit_tables(versions, events, snapshots, field_table, aba,
                                      time_issues, metrics)
    io_utils.write_excel(project_paths.TABLES_DIR / project_paths.TABLE_VERSION_AUDIT, audit_tables)
    gates.check('BUSINESS_CONFLICT_EXPORT',
                (project_paths.TABLES_DIR / project_paths.TABLE_VERSION_AUDIT).exists(),
                f'{len(audit_tables)} 个 Sheet 已生成（变化事件与冲突岗位全部留档）')
    gates.check('VERSION_AUDIT_EXPORT',
                (project_paths.TABLES_DIR / project_paths.TABLE_VERSION_AUDIT).exists()
                and len(audit_tables) >= 11,
                f'14 号版本审计表 {len(audit_tables)} 个 Sheet')

    record_path = project_paths.RECORDS_DIR / project_paths.RECORD_JOB_VERSION
    record_args = (metrics, audit_tables['01_总体版本统计'], versions, events)
    io_utils.write_markdown(record_path, write_record(*record_args, gate_table=None))
    gate_table = quality.summarize_gates(gates.results, order=quality.STAGE_GATE_MAP[STAGE])
    io_utils.write_markdown(record_path, write_record(*record_args, gate_table=gate_table))
    print(f'阶段记录: {project_paths.relative_to_root(record_path)}')

    gates.save()
    print(f'Stage 04 完成: 核心版本 {len(versions)} 个，完整页面版本 {int(per_job_full.sum())} 个，'
          f'多版本岗位 {metrics["multi_version_jobs"]} 个')
    return 0


if __name__ == '__main__':
    sys.exit(main())

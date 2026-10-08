# -*- coding: utf-8 -*-
"""Stage 03：岗位观测快照构建（搜索来源折叠 + 分类关系层）。

处理路线：

data/interim/shixiseng_job_details_cn.parquet
        ↓  删除岗位搜索链接（工作副本）
        ↓  公司介绍图片链接 → 公司认证标签（未知图标必须为 0）→ 删除原链接字段
        ↓  详情链接规范化 + ID—URL 一一对应校验
        ↓  同「实习岗位ID + 数据更新时间」下的搜索来源重复折叠（业务字段必须一致，否则留档）
        ↓
data/interim/job_observation_snapshots.parquet     岗位观测快照层
data/processed/job_category_membership.parquet     岗位分类关系层（权威来源）
outputs/tables/13_observation_snapshot_audit.xlsx

同岗位同观测时点的搜索分类重复不是时间观测，必须先折叠；
禁止直接把 172063 条原始记录全部当作独立时间观测。

用法：
    python scripts/03_build_job_observations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src import (company_cert, dedup, io_utils, project_paths,  # noqa: E402
                 quality, schema, url_utils, versioning)

STAGE = 'stage_03'
TITLE = 'Stage 03 岗位观测快照构建'


def build_audit_tables(snapshot: pd.DataFrame, conflicts: pd.DataFrame,
                       metrics: dict, category_summary: dict) -> dict:
    """构造 13 号观测快照审计表集合。"""
    id_field = schema.ID_FIELD
    record_count = schema.SNAPSHOT_RECORD_COUNT_FIELD

    overall = pd.DataFrame([
        {'指标': '原始记录数', '数值': metrics['raw_rows']},
        {'指标': '观测快照数', '数值': metrics['snapshot_rows']},
        {'指标': '搜索来源重复折叠记录数', '数值': metrics['collapsed_rows']},
        {'指标': '有观测快照的岗位数', '数值': metrics['observation_jobs']},
        {'指标': '平均快照数/岗位', '数值': metrics['avg_snapshots_per_job']},
        {'指标': '快照数 P50', '数值': metrics['snapshots_p50']},
        {'指标': '快照数 P95', '数值': metrics['snapshots_p95']},
        {'指标': '快照数 Max', '数值': metrics['snapshots_max']},
        {'指标': '同时间业务冲突组数', '数值': metrics['same_time_conflict_groups']},
        {'指标': '分类关系表行数', '数值': metrics['membership_rows']},
        {'指标': '分类关系表岗位数', '数值': category_summary['membership_ids']},
        {'指标': '实习岗位ID唯一值', '数值': metrics['unique_intern_id']},
        {'指标': '规范化详情链接唯一值', '数值': metrics['normalized_detail_url_unique']},
        {'指标': 'ID—规范化URL唯一组合', '数值': metrics['id_url_unique_pairs']},
        {'指标': 'ID 与规范化链接是否严格一一对应', '数值': metrics['id_url_one_to_one']},
        {'指标': '公司认证标签组合数量', '数值': metrics['cert_tag_combinations']},
    ])

    same_time = (snapshot[record_count].value_counts().sort_index()
                 .rename('观测时点数').reset_index())
    same_time.columns = ['同时间点原始记录数', '观测时点数']

    snapshot_records = (snapshot[record_count].value_counts().sort_index()
                        .rename('快照数').reset_index())
    snapshot_records.columns = ['快照包含的原始记录数', '快照数']

    observations_per_job = snapshot.groupby(id_field).size()
    job_distribution = observations_per_job.value_counts().sort_index().rename('岗位数量').reset_index()
    job_distribution.columns = ['岗位观测快照数', '岗位数量']

    fold_summary = snapshot.groupby(id_field, sort=True).agg(**{
        '原始记录数': (record_count, 'sum'),
        '观测快照数': (record_count, 'size'),
        '来源岗位大类数量': (schema.SNAPSHOT_GROUP_COUNT_FIELD, 'max'),
        '来源岗位细分类数量': (schema.SNAPSHOT_ITEM_COUNT_FIELD, 'max')}
    ).reset_index()
    fold_summary['折叠记录数'] = fold_summary['原始记录数'] - fold_summary['观测快照数']
    fold_summary = (fold_summary[fold_summary['折叠记录数'] > 0]
                    .sort_values('折叠记录数', ascending=False).reset_index(drop=True))

    return {
        '01_总体统计': overall,
        '02_同时间重复分布': same_time,
        '03_同时间业务字段冲突': conflicts,
        '04_快照观测数分布': snapshot_records,
        '05_岗位观测次数分布': job_distribution,
        '06_分类来源折叠统计': fold_summary,
    }


def write_record(metrics: dict, snapshot: pd.DataFrame, conflicts: pd.DataFrame,
                 overall: pd.DataFrame, gate_table: pd.DataFrame | None) -> list:
    """构造 Stage 03 阶段记录。"""
    lines = [
        '# 阶段 03 记录：岗位观测快照构建',
        '',
        '> 本文件由 `scripts/03_build_job_observations.py` 自动生成，数字均来自真实运行结果。',
        '',
        '## 1. 输入与输出',
        '',
        '| 项 | 内容 |',
        '| --- | --- |',
        f'| 输入 | `{project_paths.relative_to_root(project_paths.INTERIM_CN_PARQUET)}`'
        f'（{metrics["raw_rows"]} 行 × {metrics["raw_columns"]} 列） |',
        f'| 观测快照输出 | '
        f'`{project_paths.relative_to_root(project_paths.OBSERVATION_SNAPSHOT_PARQUET)}`'
        f'（{metrics["snapshot_rows"]} 行 × {snapshot.shape[1]} 列） |',
        f'| 分类关系输出 | '
        f'`{project_paths.relative_to_root(project_paths.PROCESSED_CATEGORY_MEMBERSHIP_PARQUET)}`'
        f'（{metrics["membership_rows"]} 行） |',
        f'| 主键 | `{schema.ID_FIELD}` |',
        f'| 观测时间 | `{schema.OBSERVATION_TIME_FIELD}`（= 数据更新时间） |',
        '',
        '## 2. 总体统计',
        '',
        '| 指标 | 数值 |',
        '| --- | --- |',
        *[f'| {row.指标} | {row.数值} |' for row in overall.itertuples(index=False)],
        '',
        '## 3. 搜索来源折叠',
        '',
        f'- 原始记录数：{metrics["raw_rows"]}；',
        f'- 观测快照数：{metrics["snapshot_rows"]}；',
        f'- 搜索来源重复折叠记录数：{metrics["collapsed_rows"]}；',
        f'- 有观测快照的岗位数：{metrics["observation_jobs"]}；',
        f'- 同「岗位ID + 数据更新时间」业务字段冲突组数：{metrics["same_time_conflict_groups"]}'
        f'（冲突组全部进入审计表，禁止随机取值或 keep="first"）。',
        '',
        '## 4. 公司认证标签与身份校验',
        '',
        f'- 公司认证标签：未知文件名 {metrics["cert_unknown_filename_count"]}，'
        f'非空 token {metrics["cert_token_count"]} 个、映射 {metrics["cert_mapped_token_count"]} 个；',
        f'- 已删除 `{schema.CERT_SOURCE_FIELD}`，新增 `{schema.CERT_TAG_FIELD}`；',
        f'- 实习岗位ID唯一值：{metrics["unique_intern_id"]}；',
        f'- 规范化详情链接唯一值：{metrics["normalized_detail_url_unique"]}；',
        f'- ID—规范化URL唯一组合：{metrics["id_url_unique_pairs"]}；',
        f'- 是否严格一一对应：{"是" if metrics["id_url_one_to_one"] else "否"}。',
        '',
        '## 5. 分类关系层',
        '',
        f'- 关系表行数：{metrics["membership_rows"]}（岗位 × 大类 × 细分类，含原始命中次数）；',
        f'- 平均岗位大类命中数：{metrics["avg_group_hits"]}；',
        f'- 平均岗位细分类命中数：{metrics["avg_category_hits"]}；',
        '- 岗位大类/岗位细分类属于搜索来源，**不进入**任何版本签名。',
        '',
        '## 6. 观测快照字段',
        '',
        f'- 快照字段（{len(schema.SNAPSHOT_COLUMNS)} 列）：'
        f'{"、".join(schema.SNAPSHOT_COLUMNS)}；',
        '- `观测时间` = `数据更新时间`；`原始观测记录数`/`来源岗位大类数量`/'
        '`来源岗位细分类数量` 记录折叠来源规模；',
        '- 同一观测时点的业务字段经归一化比较必须完全一致，才算同一状态。',
        '',
        '## 7. 输出文件',
        '',
        f'- `{project_paths.relative_to_root(project_paths.OBSERVATION_SNAPSHOT_PARQUET)}`；',
        f'- `{project_paths.relative_to_root(project_paths.PROCESSED_CATEGORY_MEMBERSHIP_PARQUET)}`；',
        f'- `outputs/tables/{project_paths.TABLE_OBSERVATION_AUDIT}`。',
        '',
    ]
    if gate_table is not None:
        lines += [
            '## 8. 运行门禁',
            '',
            '| 门禁项 | 状态 | 说明 |',
            '| --- | --- | --- |',
            *[f'| {row.门禁项} | {row.状态} | {row.说明} |'
              for row in gate_table.itertuples(index=False)],
            '',
        ]
    lines += [
        '## 9. 本阶段边界',
        '',
        '- 未执行岗位版本压缩（属 Stage 04）；',
        '- 未删除残差记录：所有原始观测都进入快照或被显式折叠审计；',
        '- 未进入薪资清洗、缺失值填补、EDA、统计检验或机器学习。',
        '',
    ]
    return lines


def main() -> int:
    project_paths.ensure_directories()
    quality.configure_logging()
    gates = quality.GateRegistry(STAGE)

    quality.stage_banner(STAGE, TITLE)
    df_raw = io_utils.read_parquet(project_paths.INTERIM_CN_PARQUET)
    raw_columns = int(df_raw.shape[1])
    print(f'输入: {len(df_raw)} 行 × {raw_columns} 列（中文列名）')
    gates.check('CN_DATA_LOAD',
                len(df_raw) > 0 and schema.CERT_SOURCE_FIELD in df_raw.columns
                and schema.SEARCH_URL_FIELD in df_raw.columns,
                f'{len(df_raw)} 行 × {raw_columns} 列，含「{schema.CERT_SOURCE_FIELD}」')

    # 1) 工作副本：删除岗位搜索链接
    work = df_raw.drop(columns=[schema.SEARCH_URL_FIELD])
    gates.check('ITEM_URL_DROP',
                schema.SEARCH_URL_FIELD not in work.columns and len(work) == len(df_raw),
                f'已删除「{schema.SEARCH_URL_FIELD}」，保留 {work.shape[1]} 列')

    # 2) 公司认证标签前置语义映射
    cert_series = df_raw[schema.CERT_SOURCE_FIELD]
    cert_audit = company_cert.audit_company_cert_values(cert_series)
    total_tokens, mapped_tokens = company_cert.mapped_token_count(cert_series)
    unknown_table = company_cert.find_unknown_icons(cert_series)
    work[schema.CERT_TAG_FIELD] = cert_series.map(company_cert.map_company_cert_tags)
    tag_combination_table = company_cert.build_tag_combination_table(work[schema.CERT_TAG_FIELD])
    gates.check('COMPANY_CERT_VALUE_AUDIT',
                cert_audit['总记录数'] == len(df_raw)
                and cert_audit['非空记录数'] + cert_audit['空值记录数'] == len(df_raw),
                f'非空 {cert_audit["非空记录数"]} / token {cert_audit["URL_token数量"]} / '
                f'文件名唯一值 {cert_audit["URL文件名唯一值数量"]}')
    gates.check('COMPANY_CERT_MAPPING',
                total_tokens == mapped_tokens and cert_audit['无法解析值数量'] == 0,
                f'非空 URL token {total_tokens} 个全部映射成功，映射率 100%')
    gates.check('COMPANY_CERT_UNKNOWN_CHECK',
                unknown_table.empty and cert_audit['未知文件名数量'] == 0,
                '未知文件名 0，未知 URL 0')

    work = work.drop(columns=[schema.CERT_SOURCE_FIELD])
    assert schema.CERT_TAG_FIELD in work.columns
    assert schema.CERT_SOURCE_FIELD not in work.columns
    gates.check('COMPANY_CERT_SOURCE_DROP',
                schema.CERT_TAG_FIELD in work.columns
                and schema.CERT_SOURCE_FIELD not in work.columns,
                f'新增「{schema.CERT_TAG_FIELD}」，删除「{schema.CERT_SOURCE_FIELD}」')

    # 3) 详情链接规范化 + ID—URL 一一对应
    work[schema.URL_NORM_FIELD] = url_utils.normalize_series(work[schema.URL_FIELD])
    raw_url_unique = int(work[schema.URL_FIELD].nunique())
    norm_url_unique = int(work[schema.URL_NORM_FIELD].nunique())
    unique_intern_id = int(work[schema.ID_FIELD].nunique())
    id_url_unique_pairs = int(work[[schema.ID_FIELD, schema.URL_NORM_FIELD]]
                              .drop_duplicates().shape[0])
    urls_per_id = work.groupby(schema.ID_FIELD)[schema.URL_NORM_FIELD].nunique()
    ids_per_url = work.groupby(schema.URL_NORM_FIELD)[schema.ID_FIELD].nunique()
    gates.check('DETAIL_URL_NORMALIZE',
                norm_url_unique == unique_intern_id,
                f'{raw_url_unique} → {norm_url_unique} 个唯一链接，ID 唯一值 {unique_intern_id}')
    gates.check('ID_URL_ONE_TO_ONE',
                int((urls_per_id > 1).sum()) == 0 and int((ids_per_url > 1).sum()) == 0
                and id_url_unique_pairs == unique_intern_id == norm_url_unique,
                f'唯一组合 {id_url_unique_pairs}，一对多 {int((urls_per_id > 1).sum())}')

    # 4) 观测时间 = 数据更新时间；折叠搜索来源重复
    work[schema.OBSERVATION_TIME_FIELD] = work['数据更新时间']
    snapshot, conflicts, collapse_summary = versioning.collapse_observation_snapshots(
        work, url_field=schema.URL_NORM_FIELD, business_fields=schema.BUSINESS_FIELDS)
    records_per_id = work.groupby(schema.ID_FIELD).size()
    snapshots_per_job = snapshot.groupby(schema.ID_FIELD).size()
    print(f'观测快照: {collapse_summary["snapshot_rows"]} 行'
          f'（原始 {collapse_summary["raw_rows"]} → 折叠 {collapse_summary["collapsed_rows"]}）')
    gates.check('OBSERVATION_BUILD',
                len(snapshot) == collapse_summary['snapshot_rows']
                and int(snapshot[schema.SNAPSHOT_RECORD_COUNT_FIELD].sum()) == len(work)
                and list(snapshot.columns) == schema.SNAPSHOT_COLUMNS,
                f'{len(snapshot)} 行快照，原始观测记录数合计 {int(snapshot[schema.SNAPSHOT_RECORD_COUNT_FIELD].sum())}')
    gates.check('CATEGORY_DUPLICATE_COLLAPSE',
                collapse_summary['observation_jobs'] == unique_intern_id
                and snapshot[[schema.ID_FIELD, schema.OBSERVATION_TIME_FIELD]].drop_duplicates().shape[0]
                == len(snapshot)
                and len(conflicts) == collapse_summary['same_time_conflict_groups'],
                f'岗位 {collapse_summary["observation_jobs"]} 个，'
                f'同时间业务冲突组 {len(conflicts)} 个（全部留档）')
    observation_times = snapshot.groupby(schema.ID_FIELD)[schema.OBSERVATION_TIME_FIELD]
    gates.check('OBSERVATION_TIME_CHECK',
                bool(observation_times.apply(lambda s: s.is_monotonic_increasing).all())
                and not snapshot[schema.OBSERVATION_TIME_FIELD].isna().any(),
                '每个岗位的观测时间序列单调不减，且无缺失')

    io_utils.write_parquet(snapshot, project_paths.OBSERVATION_SNAPSHOT_PARQUET, verify=True)

    # 5) 分类关系层（权威来源）
    membership = dedup.build_category_membership(work)
    membership_summary = dedup.reconcile_category_membership(
        membership, dedup.aggregate_source_categories(work, records_per_id))
    gates.check('CATEGORY_RELATION_BUILD',
                list(membership.columns) == schema.CATEGORY_MEMBERSHIP_COLUMNS
                and int(membership['原始命中次数'].sum()) == len(work),
                f'{len(membership)} 行，原始命中次数合计 {int(membership["原始命中次数"].sum())}')
    gates.check('CATEGORY_RELATION_RECONCILE',
                membership_summary['id_set_match'] and membership_summary['group_set_match']
                and membership_summary['item_set_match'],
                f'关系表可完整还原岗位ID集合与分类集合（{membership_summary["membership_ids"]} 个岗位）')
    gates.check('SOURCE_CATEGORY_AGG',
                set(membership[schema.ID_FIELD]) == set(snapshot[schema.ID_FIELD]),
                '分类关系表岗位ID集合与观测快照岗位ID集合一致')
    io_utils.write_parquet(membership, project_paths.PROCESSED_CATEGORY_MEMBERSHIP_PARQUET,
                           verify=True)

    # 6) 回读校验
    reloaded = io_utils.read_parquet(project_paths.OBSERVATION_SNAPSHOT_PARQUET)
    reloaded_membership = io_utils.read_parquet(project_paths.PROCESSED_CATEGORY_MEMBERSHIP_PARQUET)
    if len(reloaded) != len(snapshot) or list(reloaded.columns) != list(snapshot.columns):
        raise ValueError('观测快照回读不一致')
    print(f'回读校验通过: 快照 {len(reloaded)} 行，分类关系 {len(reloaded_membership)} 行')

    # 7) 审计表与指标
    metrics = {
        'raw_rows': int(len(df_raw)),
        'raw_columns': raw_columns,
        'work_columns': int(work.shape[1]),
        'snapshot_rows': int(len(snapshot)),
        'snapshot_columns': int(snapshot.shape[1]),
        'collapsed_rows': int(collapse_summary['collapsed_rows']),
        'observation_jobs': int(collapse_summary['observation_jobs']),
        'same_time_conflict_groups': int(collapse_summary['same_time_conflict_groups']),
        'avg_snapshots_per_job': round(float(snapshots_per_job.mean()), 4),
        'snapshots_p50': float(snapshots_per_job.quantile(0.5)),
        'snapshots_p95': float(snapshots_per_job.quantile(0.95)),
        'snapshots_max': int(snapshots_per_job.max()),
        'membership_rows': int(len(membership)),
        'unique_intern_id': unique_intern_id,
        'raw_detail_url_unique': raw_url_unique,
        'normalized_detail_url_unique': norm_url_unique,
        'id_url_unique_pairs': id_url_unique_pairs,
        'id_url_one_to_one': bool(int((urls_per_id > 1).sum()) == 0
                                  and int((ids_per_url > 1).sum()) == 0
                                  and id_url_unique_pairs == unique_intern_id == norm_url_unique),
        'cert_token_count': cert_audit['URL_token数量'],
        'cert_mapped_token_count': mapped_tokens,
        'cert_unknown_filename_count': cert_audit['未知文件名数量'],
        'cert_tag_combinations': int(tag_combination_table.shape[0]),
        'avg_records_per_job': round(float(records_per_id.mean()), 4),
        'duplicate_groups': int((records_per_id > 1).sum()),
        'single_record_groups': int((records_per_id == 1).sum()),
        'records_per_id_p50': float(records_per_id.quantile(0.5)),
        'records_per_id_p95': float(records_per_id.quantile(0.95)),
        'records_per_id_max': int(records_per_id.max()),
        'avg_group_hits': round(float(membership.groupby(schema.ID_FIELD)[
            schema.CATEGORY_GROUP_FIELD].nunique().mean()), 4),
        'avg_category_hits': round(float(membership.groupby(schema.ID_FIELD)[
            schema.CATEGORY_ITEM_FIELD].nunique().mean()), 4),
    }
    io_utils.write_json(project_paths.METRICS_DIR / f'{STAGE}.json', metrics)

    category_summary = {
        'membership_ids': int(membership[schema.ID_FIELD].nunique()),
        'membership_rows': int(len(membership)),
    }
    audit_tables = build_audit_tables(snapshot, conflicts, metrics, category_summary)
    io_utils.write_excel(project_paths.TABLES_DIR / project_paths.TABLE_OBSERVATION_AUDIT,
                         audit_tables)
    gates.check('OBSERVATION_AUDIT_EXPORT',
                (project_paths.TABLES_DIR / project_paths.TABLE_OBSERVATION_AUDIT).exists(),
                f'{len(audit_tables)} 个 Sheet 已生成')

    record_path = project_paths.RECORDS_DIR / project_paths.RECORD_OBSERVATION_SNAPSHOT
    record_args = (metrics, snapshot, conflicts, audit_tables['01_总体统计'])
    io_utils.write_markdown(record_path, write_record(*record_args, gate_table=None))
    gate_table = quality.summarize_gates(gates.results, order=quality.STAGE_GATE_MAP[STAGE])
    io_utils.write_markdown(record_path, write_record(*record_args, gate_table=gate_table))
    print(f'阶段记录: {project_paths.relative_to_root(record_path)}')

    gates.save()
    print(f'Stage 03 完成: 观测快照 {len(snapshot)} 行，折叠 {collapse_summary["collapsed_rows"]} 条搜索来源重复')
    return 0


if __name__ == '__main__':
    sys.exit(main())

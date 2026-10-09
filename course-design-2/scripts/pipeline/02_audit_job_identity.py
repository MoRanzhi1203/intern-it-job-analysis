# -*- coding: utf-8 -*-
"""Stage 02：岗位身份一致性审计（实习岗位ID / 岗位详情链接）。

data/interim/shixiseng_job_details_cn.parquet
        ↓
outputs/tables/ch3/03_identity_audit.xlsx、ch3/04_identity_conflicts.xlsx
docs/records/03_identity_audit_record.md

本阶段只在工作副本中删除「岗位搜索链接」，不覆盖 interim 文件。

用法：
    python scripts/pipeline/02_audit_job_identity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = next(
    _candidate for _candidate in Path(__file__).resolve().parents
    if (_candidate / "data").is_dir() and (_candidate / "scripts").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import io_utils, project_paths, quality, schema, url_utils  # noqa: E402

STAGE = 'stage_02'
TITLE = 'Stage 02 岗位身份一致性审计'


def audit_mapping(frame: pd.DataFrame, left: str, right: str, label: str) -> dict:
    """双向映射审计：统计一对一/一对多/多对一数量与冲突记录数。"""
    nunique_series = frame.groupby(left)[right].nunique(dropna=True)
    one_to_many_ids = nunique_series[nunique_series > 1].index
    many_side = frame.groupby(right)[left].nunique(dropna=True)
    many_to_one_ids = many_side[many_side > 1].index
    return {
        '方向': label,
        '左侧唯一值数': int(frame[left].nunique(dropna=True)),
        '右侧唯一值数': int(frame[right].nunique(dropna=True)),
        '一对一数量': int((nunique_series == 1).sum()),
        '一对多数量': int((nunique_series > 1).sum()),
        '一对多涉及记录数': int(frame[left].isin(one_to_many_ids).sum()),
        '多对一数量': int((many_side > 1).sum()),
        '多对一涉及记录数': int(frame[right].isin(many_to_one_ids).sum()),
        '最大右侧唯一值数': int(nunique_series.max()),
    }


def build_conflict_records(frame: pd.DataFrame, id_multi_ids, url_multi_ids,
                           normalized_urls_per_id: pd.Series | None = None) -> pd.DataFrame:
    """提取冲突原始记录，标注异常类型；冲突以原始链接为准，并附规范化后的链接数量。"""
    blank_id = frame[schema.ID_FIELD].isna() | frame[schema.ID_FIELD].astype(str).str.strip().eq('')
    blank_url = frame[schema.URL_FIELD].isna() | frame[schema.URL_FIELD].astype(str).str.strip().eq('')
    flag_id = frame[schema.ID_FIELD].isin(id_multi_ids)
    flag_url = frame[schema.URL_FIELD].isin(url_multi_ids)
    mask = flag_id | flag_url | blank_id | blank_url
    if not mask.any():
        return pd.DataFrame(columns=[schema.ID_FIELD, schema.URL_FIELD, '异常类型'])

    subset = frame.loc[mask].copy()
    conditions = [
        flag_id.loc[mask].to_numpy() & flag_url.loc[mask].to_numpy(),
        flag_id.loc[mask].to_numpy(),
        flag_url.loc[mask].to_numpy(),
        blank_id.loc[mask].to_numpy(),
        blank_url.loc[mask].to_numpy(),
    ]
    choices = ['ID对应多个URL', 'URL对应多个ID', 'ID和URL同时存在冲突',
               '实习岗位ID缺失', '岗位详情链接缺失']
    subset['异常类型'] = np.select(conditions, choices, default='其他')
    if normalized_urls_per_id is not None:
        subset['规范化后该ID的链接数'] = subset[schema.ID_FIELD].map(normalized_urls_per_id)
    ordered = [schema.ID_FIELD, schema.URL_FIELD, schema.URL_NORM_FIELD, '异常类型',
               '规范化后该ID的链接数', '岗位标题', '公司名称', '工作城市', '岗位大类',
               '岗位细分类', '发布时间', '薪资信息', schema.PAGE_FIELD]
    ordered = [c for c in ordered if c in subset.columns]
    return subset[ordered].reset_index(drop=True)


def write_record(metrics: dict, identity_stats: pd.DataFrame, normalize_table: pd.DataFrame,
                 structure_audit: pd.DataFrame, normalize_cause: pd.DataFrame) -> Path:
    """写出 Stage 02 阶段记录。"""
    lines = [
        '# 阶段 02 记录：岗位身份一致性审计',
        '',
        '> 本文件由 `scripts/pipeline/02_audit_job_identity.py` 自动生成，数字均来自真实运行结果。',
        '',
        '## 1. 输入与工作副本',
        '',
        '| 项 | 内容 |',
        '| --- | --- |',
        f'| 输入 | `{project_paths.relative_to_root(project_paths.INTERIM_CN_PARQUET)}`'
        f'（{metrics["raw_rows"]} 行 × {metrics["raw_columns"]} 列） |',
        f'| 工作副本 | {metrics["work_rows"]} 行 × {metrics["work_columns"]} 列（删除「{schema.SEARCH_URL_FIELD}」） |',
        f'| interim 文件是否被覆盖 | 否 |',
        '',
        '## 2. 身份字段统计',
        '',
        '| 字段 | 总记录数 | 非空数量 | 缺失数量 | 空字符串数量 | 唯一值数量 |',
        '| --- | --- | --- | --- | --- | --- |',
        *[f'| {row.字段} | {row.总记录数} | {row.非空数量} | {row.缺失数量} | '
          f'{row.空字符串数量} | {row.唯一值数量} |'
          for row in identity_stats.itertuples(index=False)],
        '',
        '## 3. 双向映射审计',
        '',
        '| 方向 | 左侧唯一值数 | 右侧唯一值数 | 一对一数量 | 一对多数量 | 一对多涉及记录数 | 多对一数量 |',
        '| --- | --- | --- | --- | --- | --- | --- |',
        *[f'| {row.方向} | {row.左侧唯一值数} | {row.右侧唯一值数} | {row.一对一数量} | '
          f'{row.一对多数量} | {row.一对多涉及记录数} | {row.多对一数量} |'
          for row in metrics['mapping_rows'].itertuples(index=False)],
        '',
        '## 4. URL 规范化',
        '',
        '| 指标 | 数值 |',
        '| --- | --- |',
        *[f'| {row.指标} | {row.数值} |' for row in normalize_table.itertuples(index=False)],
        '',
        'URL 结构审计：',
        '',
        '| 检查项 | 结果 |',
        '| --- | --- |',
        *[f'| {row.检查项} | {row.结果} |' for row in structure_audit.itertuples(index=False)],
        '',
        '差异原因判定：',
        '',
        '| 可能原因 | 是否成立 | 证据 |',
        '| --- | --- | --- |',
        *[f'| {row.可能原因} | {row.是否成立} | {row.证据} |'
          for row in normalize_cause.itertuples(index=False)],
        '',
        '## 5. 结论',
        '',
        f'- 实习岗位ID 唯一值：{metrics["unique_intern_id"]}；',
        f'- 规范化岗位详情链接唯一值：{metrics["normalized_detail_url_unique"]}；',
        f'- 唯一 ID—规范化URL 组合：{metrics["id_url_unique_pairs"]}；',
        f'- ID 与规范化链接是否严格一一对应：{"是" if metrics["id_url_one_to_one"] else "否"}；',
        f'- URL path 末段与实习岗位ID 一致：{metrics["url_path_matches_intern_id"]} / {metrics["raw_rows"]}；',
        '',
        '## 6. 本阶段边界',
        '',
        '- 未执行岗位去重；',
        '- 未修改 interim 数据；',
        '- 未进入薪资清洗、缺失值处理、EDA 或机器学习。',
        '',
    ]
    return io_utils.write_markdown(
        project_paths.RECORDS_DIR / project_paths.RECORD_IDENTITY_AUDIT, lines)


def main() -> int:
    project_paths.ensure_directories()
    quality.configure_logging()
    gates = quality.GateRegistry(STAGE)

    quality.stage_banner(STAGE, TITLE)
    df_raw = io_utils.read_parquet(project_paths.INTERIM_CN_PARQUET)
    raw_columns = df_raw.shape[1]
    print(f'输入: {len(df_raw)} 行 × {raw_columns} 列（中文列名）')

    # 工作副本：删除搜索分类链接
    if schema.SEARCH_URL_FIELD not in df_raw.columns:
        raise KeyError(f'缺少字段: {schema.SEARCH_URL_FIELD}')
    work = df_raw.drop(columns=[schema.SEARCH_URL_FIELD])
    print(f'工作副本: {len(work)} 行 × {work.shape[1]} 列（已删除「{schema.SEARCH_URL_FIELD}」）')
    if len(work) != len(df_raw) or work.shape[1] != raw_columns - 1:
        raise ValueError('删除「岗位搜索链接」后规模异常')

    identity_stats = quality.identity_field_stats(work, schema.IDENTITY_FIELDS)
    print('身份字段统计:')
    print(identity_stats.to_string(index=False))
    missing_identity = identity_stats.loc[identity_stats['缺失数量'] + identity_stats['空字符串数量'] > 0]
    if not missing_identity.empty:
        print('注意：存在缺失/空值的身份字段记录，本阶段不删除：')
        print(missing_identity.to_string(index=False))

    # 双向映射审计（原始链接）
    id_to_url = audit_mapping(work, schema.ID_FIELD, schema.URL_FIELD, '实习岗位ID → 岗位详情链接')
    url_to_id = audit_mapping(work, schema.URL_FIELD, schema.ID_FIELD, '岗位详情链接 → 实习岗位ID')
    mapping_rows = pd.DataFrame([id_to_url, url_to_id])
    print('双向映射审计:')
    print(mapping_rows.to_string(index=False))

    work[schema.URL_NORM_FIELD] = url_utils.normalize_series(work[schema.URL_FIELD])
    raw_url_unique = int(work[schema.URL_FIELD].nunique())
    norm_url_unique = int(work[schema.URL_NORM_FIELD].nunique())
    unique_intern_id = int(work[schema.ID_FIELD].nunique())
    id_url_unique_pairs = int(work[[schema.ID_FIELD, schema.URL_NORM_FIELD]].drop_duplicates().shape[0])
    urls_per_id_norm = work.groupby(schema.ID_FIELD)[schema.URL_NORM_FIELD].nunique()
    ids_per_url_norm = work.groupby(schema.URL_NORM_FIELD)[schema.ID_FIELD].nunique()

    path_tail = work[schema.URL_FIELD].map(url_utils.extract_intern_id_from_url)
    url_path_matches = int((path_tail == work[schema.ID_FIELD]).sum())
    structure = url_utils.audit_url_structure(work[schema.URL_FIELD])
    structure_audit = pd.DataFrame([
        {'检查项': '协议分布', '结果': str(structure['scheme_distribution'])},
        {'检查项': '域名分布', '结果': str(structure['netloc_distribution'])},
        {'检查项': '含 fragment 的记录数', '结果': structure['fragment_rows']},
        {'检查项': '含首尾空格的记录数', '结果': structure['leading_trailing_space_rows']},
        {'检查项': 'URL path 末段与实习岗位ID一致的记录数',
         '结果': f'{url_path_matches} / {len(work)}'},
    ])
    normalize_table = pd.DataFrame([
        {'指标': '规范化前详情链接唯一值数', '数值': raw_url_unique},
        {'指标': '规范化后详情链接唯一值数', '数值': norm_url_unique},
        {'指标': '实习岗位ID唯一值数', '数值': unique_intern_id},
        {'指标': '唯一 ID—规范化URL 组合数', '数值': id_url_unique_pairs},
        {'指标': '规范化后被消除的链接形式数', '数值': raw_url_unique - norm_url_unique},
        {'指标': '一个ID对应多个规范化链接的数量', '数值': int((urls_per_id_norm > 1).sum())},
        {'指标': '一个规范化链接对应多个ID的数量', '数值': int((ids_per_url_norm > 1).sum())},
    ])
    print('URL 规范化结果:')
    print(normalize_table.to_string(index=False))

    normalize_cause = pd.DataFrame([
        {'可能原因': 'URL 格式差异（协议/域名大小写/首尾空格）',
         '是否成立': '否' if (len(structure['scheme_distribution']) == 1
                              and len(structure['netloc_distribution']) == 1
                              and structure['leading_trailing_space_rows'] == 0) else '是',
         '证据': f'协议 {len(structure["scheme_distribution"])} 种 / '
                 f'域名 {len(structure["netloc_distribution"])} 种 / '
                 f'首尾空格 {structure["leading_trailing_space_rows"]} 条'},
        {'可能原因': 'query 跟踪参数差异（如 pcm）',
         '是否成立': '是' if (raw_url_unique != norm_url_unique and norm_url_unique == unique_intern_id) else '需复核',
         '证据': f'规范化前 {raw_url_unique} → 规范化后 {norm_url_unique}，ID 唯一值 {unique_intern_id}；'
                 f'跟踪参数 {sorted(url_utils.TRACKING_PARAMS)}'},
        {'可能原因': '真实一个 ID 对应多个 URL',
         '是否成立': '否' if int((urls_per_id_norm > 1).sum()) == 0 else '是',
         '证据': f'规范化后仍多链接的 ID 数量 {int((urls_per_id_norm > 1).sum())}'},
        {'可能原因': 'URL path 与岗位ID不一致的抓取异常',
         '是否成立': '否' if url_path_matches == len(work) else '是',
         '证据': f'path 末段与实习岗位ID一致 {url_path_matches} / {len(work)}'},
    ])
    print('差异原因判定:')
    print(normalize_cause.to_string(index=False))

    # 冲突记录（以原始链接为准，随后由规范化解释差异性质）
    raw_urls_per_id = work.groupby(schema.ID_FIELD)[schema.URL_FIELD].nunique()
    raw_ids_per_url = work.groupby(schema.URL_FIELD)[schema.ID_FIELD].nunique()
    conflict_records = build_conflict_records(
        work,
        raw_urls_per_id[raw_urls_per_id > 1].index,
        raw_ids_per_url[raw_ids_per_url > 1].index,
        normalized_urls_per_id=urls_per_id_norm)
    print(f'冲突原始记录数（以原始链接为准）: {len(conflict_records)}')
    if len(conflict_records):
        print(f'  其中规范化后已收敛为一对一的记录数: '
              f'{int((conflict_records["规范化后该ID的链接数"] == 1).sum())}')

    io_utils.write_excel(project_paths.TABLES_DIR / project_paths.TABLE_IDENTITY_AUDIT, {
        '身份字段统计': identity_stats,
        '双向映射审计': mapping_rows,
        'URL规范化汇总': normalize_table,
        'URL结构审计': structure_audit,
        'URL差异原因判定': normalize_cause,
        'URL query分布': url_utils.audit_url_queries(work[schema.URL_FIELD]),
    })
    io_utils.write_excel(project_paths.TABLES_DIR / project_paths.TABLE_IDENTITY_CONFLICTS, {
        '冲突概览': pd.DataFrame([
            {'指标': 'ID对应多个原始URL的ID数量',
             '数值': int((work.groupby(schema.ID_FIELD)[schema.URL_FIELD].nunique() > 1).sum())},
            {'指标': 'URL对应多个ID的URL数量', '数值': int((ids_per_url_norm > 1).sum())},
            {'指标': '冲突原始记录数', '数值': len(conflict_records)},
            {'指标': '规范化后 ID—URL 是否一一对应',
             '数值': bool(id_url_unique_pairs == unique_intern_id == norm_url_unique)},
        ]),
        '冲突原始记录': conflict_records,
    })

    metrics = {
        'raw_rows': int(len(df_raw)),
        'raw_columns': int(raw_columns),
        'work_rows': int(len(work)),
        'work_columns': int(work.shape[1]),
        'unique_intern_id': unique_intern_id,
        'raw_detail_url_unique': raw_url_unique,
        'normalized_detail_url_unique': norm_url_unique,
        'id_url_unique_pairs': id_url_unique_pairs,
        'id_url_one_to_one': bool(id_url_unique_pairs == unique_intern_id == norm_url_unique
                                  and int((urls_per_id_norm > 1).sum()) == 0
                                  and int((ids_per_url_norm > 1).sum()) == 0),
        'url_path_matches_intern_id': url_path_matches,
        'raw_url_multi_id_rows': int(len(conflict_records)),
        'mapping_rows': mapping_rows,
    }
    io_utils.write_json(project_paths.METRICS_DIR / f'{STAGE}.json',
                        {k: v for k, v in metrics.items() if k != 'mapping_rows'})

    record_path = write_record(metrics, identity_stats, normalize_table,
                               structure_audit, normalize_cause)
    print(f'阶段记录: {project_paths.relative_to_root(record_path)}')

    gates.assert_frame('IDENTITY_AUDIT', identity_stats,
                       expected_rows=2, required_columns=['字段', '非空数量', '唯一值数量'],
                       note=f'身份字段唯一值 {unique_intern_id} / {raw_url_unique}')
    gates.check('DETAIL_URL_NORMALIZE',
                norm_url_unique == unique_intern_id,
                f'{raw_url_unique} → {norm_url_unique} 个唯一链接，ID 唯一值 {unique_intern_id}')
    gates.check('ID_URL_ONE_TO_ONE',
                int((urls_per_id_norm > 1).sum()) == 0 and int((ids_per_url_norm > 1).sum()) == 0
                and id_url_unique_pairs == unique_intern_id,
                f'唯一组合 {id_url_unique_pairs}，一对多 {int((urls_per_id_norm > 1).sum())}')
    gates.save()
    print(f'Stage 02 完成: 唯一岗位ID {unique_intern_id}，规范化链接 {norm_url_unique}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

# -*- coding: utf-8 -*-
"""Stage 13：技能需求 EDA 统计范围审计（两种统计范围 + 分层榜单 + 稳健性）。

输入（全部来自已封版产物，不重跑 Stage 00~11）：
    data/processed/job_text_features.parquet      技能提取范围（match_scope 统计范围）
    data/features/job_skill_membership.parquet    岗位 × 规范技能 long-format
    data/processed/job_category_membership.parquet 岗位细分类
    data/processed/job_salary_targets.parquet     薪资目标（主目标 = 薪资中点）

输出：
    outputs/tables/ch6/19_skill_eda_scope_audit.xlsx  11 张子表（统计范围/榜单/结构/共现/薪资/稳健性）
    outputs/logs/metrics/stage_13_skill_eda.json

统计范围（封版，禁止混用）：
    主统计范围 = REQUIREMENT_SECTION（企业明确要求段落）
    扩展统计范围 = REQUIREMENT_SECTION + FULL_TEXT_FALLBACK（排除 EMPTY_TEXT）
    榜单层级 = 具体技术技能 / 技术领域 / 业务能力 / 办公工具（复用 feature_family / group）
    组级统计一律按 intern_id 去重，禁止跨层级或同层简单相加

用法：
    python scripts/ch6_skills/13b_skill_eda_scope_audit.py
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

from src import (io_utils, project_paths, quality, schema, skill_eda,  # noqa: E402
                 skill_extraction)

STAGE = 'stage_13_skill_eda'
TITLE = 'Stage 13 技能需求 EDA 统计范围审计（两种统计范围 + 分层榜单 + 稳健性）'

SALARY_FOCUS_SKILLS = ['Python', 'Java', 'SQL', 'MySQL', 'Excel', '大模型', 'Agent', '机器学习']
# 技能数量档位：0 / 1 / 2 / 3 / 4 / 5+（6 档对应 7 个分箱边界）
SKILL_COUNT_BINS = [0, 1, 2, 3, 4]
DEFAULT_NO_CAUSAL_NOTE = '本表为描述性关联，禁止解读为因果；技能要求更多 ≠ 因果导致薪资更高'


def build_scope_table(features: pd.DataFrame, membership: pd.DataFrame,
                      universe: dict) -> pd.DataFrame:
    """01_样本统计范围：三种统计范围岗位数与覆盖率。"""
    total = len(features)
    rows = []
    for name, scope, note in [
        (skill_eda.SCOPE_MAIN, skill_eda.SCOPE_MAIN, '论文正文主统计范围：企业明确提出的任职要求段落'),
        (skill_eda.SCOPE_FALLBACK, skill_eda.SCOPE_FALLBACK,
         '扩展统计范围：未识别要求段落时使用模型安全 JD 全文，不丢弃岗位'),
        (skill_eda.SCOPE_EMPTY, skill_eda.SCOPE_EMPTY,
         '独立缺失统计范围：无可用技能文本，禁止解释为「企业没有技能要求」'),
        ('ALL_USABLE', 'ALL_USABLE', '扩展分析分母 = REQUIREMENT_SECTION + FULL_TEXT_FALLBACK'),
    ]:
        ids = universe[scope]
        scopes = (skill_eda.ALL_USABLE_SCOPES if scope == 'ALL_USABLE' else scope)
        frame = skill_eda.filter_membership(membership, scopes)
        frame = frame[frame[schema.SKILL_MEMBERSHIP_ID_FIELD].isin(ids)]
        jobs_with_skill = frame[schema.SKILL_MEMBERSHIP_ID_FIELD].nunique()
        rows.append({
            '统计范围': name,
            '岗位数': len(ids),
            '占全部岗位比例': round(len(ids) / total, 6),
            '其中至少命中 1 项技能的岗位数': int(jobs_with_skill),
            '统计范围内技能覆盖率': round(jobs_with_skill / len(ids), 6) if ids else 0.0,
            '说明': note,
        })
    table = pd.DataFrame(rows)
    table.attrs['note'] = '技能数 = 0 只能表述为「未从可用文本中识别到规范技能」'
    return table


def build_layer_tables(main_rank: pd.DataFrame, layers: dict) -> dict:
    """04~07：四个层级的榜单（层级内统计，禁止跨层相加）。"""
    tables = {}
    mapping = [
        (skill_eda.LAYER_TECHNICAL, '04_具体技术技能', 20,
         '互联网 IT 实习岗位核心技术技能 Top20'),
        (skill_eda.LAYER_DOMAIN, '05_技术领域', 15,
         '互联网 IT 实习岗位主要技术方向 Top15'),
        (skill_eda.LAYER_BUSINESS, '06_业务能力', 15,
         '互联网 IT 实习岗位业务能力需求 Top15'),
        (skill_eda.LAYER_OFFICE, '07_办公工具', None,
         '互联网 IT 实习岗位办公工具需求'),
    ]
    for layer, sheet, top_n, title in mapping:
        table = skill_eda.layer_rank(main_rank, layer, top_n)
        table.insert(0, '榜单', title)
        table['层级技能总数'] = len(layers.get(layer, ()))
        table['统计范围'] = 'REQUIREMENT_SECTION（分母 = 主统计范围岗位数）'
        tables[sheet] = table
    return tables


def build_office_dedup_table(membership: pd.DataFrame, universe: dict, config) -> pd.DataFrame:
    """办公工具组的去重校验：组岗位数 ≠ 组内技能岗位数相加。"""
    unique_jobs, item_total, naive_sum = skill_eda.group_job_counts(
        membership, universe[skill_eda.SCOPE_MAIN], skill_eda.LAYER_OFFICE, config)
    rows = [{
        '校验项': '办公工具组岗位数（按 intern_id 去重）', '数值': unique_jobs,
        '说明': 'group = 办公工具，对 intern_id 去重后的岗位数（正式统计范围）',
    }, {
        '校验项': '组内技能项次（岗位 × 技能 行数）', '数值': item_total,
        '说明': '同一岗位可命中多个办公工具技能',
    }, {
        '校验项': '组内技能岗位数简单相加（错误统计范围）', '数值': naive_sum,
        '说明': '禁止把 Excel + PPT + Word + 办公软件 相加当作办公工具岗位数',
    }, {
        '校验项': '重复计数差额（相加 − 去重）', '数值': naive_sum - unique_jobs,
        '说明': '差额即为被重复计数的岗位次数，证明必须按 intern_id 去重',
    }]
    return pd.DataFrame(rows)


def build_skill_count_salary_table(membership: pd.DataFrame, salary: pd.DataFrame,
                                  universe: dict, scopes) -> pd.DataFrame:
    """技能数量（0/1/2/3/4/5+）与薪资中点的描述性关系。"""
    frame = skill_eda.filter_membership(membership, scopes)
    ids = universe['ALL_USABLE']
    frame = frame[frame[schema.SKILL_MEMBERSHIP_ID_FIELD].isin(ids)]
    counts = (frame.groupby(schema.SKILL_MEMBERSHIP_ID_FIELD)['canonical_skill']
              .nunique().reindex(list(ids), fill_value=0))
    valid = salary[skill_eda.valid_salary_mask(salary)]
    mid = valid.set_index(schema.ID_FIELD)[schema.SALARY_MID_FIELD]
    merged = pd.DataFrame({'技能数': counts})
    merged['薪资中点'] = mid.reindex(merged.index)
    merged = merged[merged['薪资中点'].notna()]
    bucket = pd.cut(merged['技能数'], bins=[-1, 0, 1, 2, 3, 4, 10_000],
                    labels=[str(item) for item in SKILL_COUNT_BINS] + ['5+'])
    rows = []
    for name, group in merged.groupby(bucket):
        rows.append({
            '技能数量档': str(name),
            '有效薪资岗位数': int(len(group)),
            '薪资中点中位数': round(float(group['薪资中点'].median()), 4),
            '薪资中点IQR': round(float(group['薪资中点'].quantile(0.75)
                                   - group['薪资中点'].quantile(0.25)), 4),
            '说明': DEFAULT_NO_CAUSAL_NOTE,
        })
    return pd.DataFrame(rows)


def build_salary_sheet(membership: pd.DataFrame, salary: pd.DataFrame, categories: pd.DataFrame,
                       universe: dict, main_rank: pd.DataFrame) -> pd.DataFrame:
    """10_技能薪资关联：主统计范围逐技能 + 扩展统计范围逐技能 + 控制细分类 + 技能数量档。"""
    tech_skills = (main_rank[main_rank['层级'] == skill_eda.LAYER_TECHNICAL]
                   .head(20)['技能标准名'].tolist())
    focus = sorted(set(tech_skills) | set(SALARY_FOCUS_SKILLS))
    main_table = skill_eda.salary_association(
        membership, salary, universe[skill_eda.SCOPE_MAIN], focus, scopes=skill_eda.SCOPE_MAIN)
    main_table.insert(0, '分析块', '主统计范围 REQUIREMENT_SECTION：逐技能有/无对比')
    ext_table = skill_eda.salary_association(
        membership, salary, universe['ALL_USABLE'], focus, scopes=skill_eda.ALL_USABLE_SCOPES)
    ext_table.insert(0, '分析块', '扩展统计范围 ALL_USABLE：逐技能有/无对比（稳健性）')
    within = skill_eda.salary_association_within_category(
        membership, salary, categories, universe[skill_eda.SCOPE_MAIN], SALARY_FOCUS_SKILLS,
        scopes=skill_eda.SCOPE_MAIN)
    if not within.empty:
        within.insert(0, '分析块', '主统计范围 + 控制岗位细分类：细分类内部中位数差 Top5')
        within['说明'] = '控制岗位类别结构差异后的描述性比较'
    count_table = build_skill_count_salary_table(membership, salary, universe,
                                                skill_eda.ALL_USABLE_SCOPES)
    count_table['分析块'] = '扩展统计范围：技能数量档 × 薪资中点'
    for table in (main_table, ext_table, within, count_table):
        if not table.empty:
            table['统计范围与措辞'] = DEFAULT_NO_CAUSAL_NOTE
    blocks = [table for table in (main_table, ext_table, within, count_table)
              if not table.empty]
    return pd.concat(blocks, ignore_index=True, sort=False) if blocks else pd.DataFrame()


def build_robustness_sheet(summary: dict, detail: pd.DataFrame) -> pd.DataFrame:
    """11_两种统计范围稳健性：Top10 / Top20 overlap + Spearman + 排名差明细。"""
    summary_rows = [{
        '分析块': '稳健性汇总',
        '技能标准名或指标': key,
        '数值': value,
        '说明': ('主统计范围 = REQUIREMENT_SECTION；扩展统计范围 = REQUIREMENT_SECTION + '
                'FULL_TEXT_FALLBACK；排名一致说明结论对文本结构识别方式稳健'),
    } for key, value in summary.items()]
    detail_rows = [{
        '分析块': 'Top50 技能排名对照',
        '技能标准名或指标': row['技能标准名'],
        '数值': None,
        '主统计范围排名': row['主统计范围排名'],
        '扩展统计范围排名': row['扩展统计范围排名'],
        '排名差': row['排名差'],
        '说明': '排名差 = 扩展统计范围排名 − 主统计范围排名（负值表示扩展统计范围中更靠前）',
    } for row in detail.to_dict('records')]
    columns = ['分析块', '技能标准名或指标', '数值', '主统计范围排名', '扩展统计范围排名', '排名差', '说明']
    return pd.DataFrame(summary_rows + detail_rows, columns=columns)


def main() -> int:
    quality.configure_logging()
    gates = quality.GateRegistry(STAGE)

    quality.stage_banner(STAGE, TITLE)
    config = skill_extraction.load_skill_config()
    features = io_utils.read_parquet(project_paths.JOB_TEXT_FEATURES_PARQUET,
                                     columns=[schema.ID_FIELD, schema.SKILL_SCOPE_FIELD,
                                              schema.SKILL_COUNT_FIELD, schema.SKILL_SET_FIELD])
    membership = io_utils.read_parquet(project_paths.JOB_SKILL_MEMBERSHIP_PARQUET)
    categories = io_utils.read_parquet(project_paths.PROCESSED_CATEGORY_MEMBERSHIP_PARQUET,
                                       columns=[schema.ID_FIELD, '岗位细分类'])
    salary = io_utils.read_parquet(project_paths.SALARY_TARGETS_PARQUET)
    print(f'输入: 岗位 {len(features)} 行 / long-format {len(membership)} 行 / '
          f'薪资 {len(salary)} 行')

    universe = skill_eda.load_scope_universe(features)
    scope_table = build_scope_table(features, membership, universe)
    scope_counts = {key: len(universe[key]) for key in
                    (skill_eda.SCOPE_MAIN, skill_eda.SCOPE_FALLBACK, skill_eda.SCOPE_EMPTY)}
    gates.check('SKILL_EDA_SCOPE_SAMPLE',
                sum(scope_counts.values()) == len(features)
                and scope_counts[skill_eda.SCOPE_MAIN] > 0
                and set(features[schema.SKILL_SCOPE_FIELD]) <= set(schema.SKILL_MATCH_SCOPE_VALUES),
                f'三种统计范围岗位数合计 {sum(scope_counts.values())} = 全部岗位 {len(features)}；'
                f'主统计范围 {scope_counts[skill_eda.SCOPE_MAIN]} / fallback '
                f'{scope_counts[skill_eda.SCOPE_FALLBACK]} / 空文本 {scope_counts[skill_eda.SCOPE_EMPTY]}；'
                f'扩展统计范围（可用文本）{len(universe["ALL_USABLE"])}')

    layers = skill_extraction.resolve_rank_layers(config)
    layer_map = skill_extraction.skill_layer_map(config)
    unassigned = sorted(set(config.skill_to_group) - set(layer_map))
    gates.check('SKILL_EDA_LAYER_SPLIT',
                not unassigned and len(layers) == 4
                and sum(len(item) for item in layers.values()) == config.skill_count,
                f'{len(layers)} 个层级覆盖全部 {config.skill_count} 个技能标准名'
                f'（{"、".join(f"{name} {len(items)}" for name, items in layers.items())}）；'
                '层级来自 skills.yml rank_layers，未新建技能分类字典')

    main_rank = skill_eda.rank_skills(membership, universe[skill_eda.SCOPE_MAIN], config,
                                      scopes=skill_eda.SCOPE_MAIN)
    ext_rank = skill_eda.rank_skills(membership, universe['ALL_USABLE'], config,
                                     scopes=skill_eda.ALL_USABLE_SCOPES)
    layer_tables = build_layer_tables(main_rank, layers)
    office_dedup = build_office_dedup_table(membership, universe, config)
    dedup_row = office_dedup.set_index('校验项')
    unique_jobs = int(dedup_row.loc['办公工具组岗位数（按 intern_id 去重）', '数值'])
    naive_sum = int(dedup_row.loc['组内技能岗位数简单相加（错误统计范围）', '数值'])
    gates.check('SKILL_EDA_GROUP_DEDUP',
                unique_jobs > 0 and naive_sum > unique_jobs,
                f'办公工具组岗位数（intern_id 去重）{unique_jobs} < 组内技能岗位数简单相加 '
                f'{naive_sum}，重复计数差额 {naive_sum - unique_jobs}；'
                '全部组级统计均按 intern_id 去重')

    matrix = skill_eda.category_skill_matrix(membership, categories,
                                             universe[skill_eda.SCOPE_MAIN], main_rank,
                                             scopes=skill_eda.SCOPE_MAIN)
    cooc = skill_eda.cooccurrence(membership, universe[skill_eda.SCOPE_MAIN], main_rank,
                                  layer=skill_eda.LAYER_TECHNICAL,
                                  scopes=skill_eda.SCOPE_MAIN)
    summary, detail = skill_eda.rank_robustness(main_rank, ext_rank)
    gates.check('SKILL_EDA_ROBUSTNESS',
                summary['Top10 overlap 比例'] >= 0.8
                and summary['Top20 overlap 比例'] >= 0.8
                and summary['Spearman 排名相关'] is not None
                and summary['Spearman 排名相关'] >= 0.8,
                f"Top10 overlap {summary['Top10 overlap']}/10、"
                f"Top20 overlap {summary['Top20 overlap']}/20、"
                f"Spearman {summary['Spearman 排名相关']}（共有技能 {summary['共有技能数']} 个）"
                '→ 技能需求结论对文本结构识别方式稳健')

    salary_sheet = build_salary_sheet(membership, salary, categories, universe, main_rank)
    parsed = int(salary[schema.SALARY_PARSE_STATUS_FIELD].eq('已解析').sum())
    anomaly = int(salary[schema.SALARY_ANOMALY_FIELD].fillna('').astype(str).str.strip().ne('').sum())
    valid_salary = salary[skill_eda.valid_salary_mask(salary)]
    testable = int(salary_sheet['检验是否达标'].eq('是').sum()) if '检验是否达标' in salary_sheet.columns else 0
    gates.check('SKILL_EDA_SALARY_SCOPE',
                parsed == 14899 and anomaly == 16 and len(valid_salary) == parsed - anomaly
                and not salary_sheet.empty and testable > 0,
                f'正式薪资样本 {len(valid_salary)} = 解析成功 {parsed} − 逻辑异常 {anomaly}'
                f'（非面议 + 解析成功 + 中点有效 + 无逻辑异常）；'
                f'主统计范围检验达标技能 {testable} 个，已输出 Mann–Whitney U / Cliff\'s delta / BH-FDR；'
                '全部为描述性关联，禁止因果解读')

    sheets = {
        '01_样本统计范围': scope_table,
        '02_主统计范围技能排名': main_rank,
        '03_扩展统计范围技能排名': ext_rank,
        **layer_tables,
        '08_岗位类别技能画像': matrix,
        '09_技能共现': cooc,
        '10_技能薪资关联': salary_sheet,
        '11_两种统计范围稳健性': build_robustness_sheet(summary, detail),
        '12_办公工具去重校验': office_dedup,
    }
    audit_path = io_utils.write_excel(
        project_paths.TABLES_DIR / project_paths.TABLE_SKILL_EDA_SCOPE, sheets)
    gates.check('SKILL_EDA_AUDIT_EXPORT',
                audit_path.exists() and len(sheets) >= 11
                and {'02_主统计范围技能排名', '03_扩展统计范围技能排名', '11_两种统计范围稳健性'} <= set(sheets),
                f'{len(sheets)} 张技能 EDA 子表已写出（两种统计范围 + 四层榜单 + 共现 + 薪资 + 稳健性）')

    metrics = {
        'scope_counts': scope_counts,
        'all_usable_jobs': len(universe['ALL_USABLE']),
        'skill_count': int(config.skill_count),
        'layer_sizes': {name: len(items) for name, items in layers.items()},
        'main_top20': main_rank.head(20)[['排名', '技能标准名', '层级', '岗位数', '岗位占比']]
        .to_dict('records'),
        'extended_top20': ext_rank.head(20)[['排名', '技能标准名', '岗位数', '岗位占比']]
        .to_dict('records'),
        'layer_top': {name: table.head(15)['技能标准名'].tolist()
                      for name, table in layer_tables.items()},
        'robustness': summary,
        'office_group': {'unique_jobs': unique_jobs, 'naive_sum': naive_sum},
        'cooccurrence_pairs': int(len(cooc)),
        'salary_valid_samples': int(len(valid_salary)),
        'salary_testable_skills': testable,
        'audit_path': project_paths.relative_to_root(audit_path),
    }
    metrics['gates'] = {name: result['status'] for name, result in gates.results.items()}
    io_utils.write_json(project_paths.METRICS_DIR / f'{STAGE}.json', metrics)

    quality.print_gates(gates.results, order=quality.STAGE_GATE_MAP[STAGE])
    gates.save()
    print(f'技能 EDA 审计表: {project_paths.relative_to_root(audit_path)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

# -*- coding: utf-8 -*-
"""Stage26.7：技术时间字段排除审计 + 业务时间维度（发布时间队列）专题分析。

本轮只做描述性业务时间分析与技术时间字段审计：

- 主时间轴 = 岗位发布时间（业务时间）；
- 不使用爬取/观测时间、数据创建时间、数据更新时间；
- 不重训模型、不改动任何已冻结正式结果；
- 输出：outputs/tables/59_business_time_dimension_analysis.xlsx、
        outputs/figures/time/01~03_*.png|pdf、
        outputs/logs/metrics/stage_26_7_time_and_audit.json。

用法：python scripts/33_stage26_7_time_and_audit.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / 'data' / 'processed'
FEATURES = ROOT / 'data' / 'features'
INTERIM = ROOT / 'data' / 'interim'
TABLES = ROOT / 'outputs' / 'tables'
FIGDIR = ROOT / 'outputs' / 'figures' / 'time'
METRICS = ROOT / 'outputs' / 'logs' / 'metrics'
MODELDIR = ROOT / 'outputs' / 'models' / 'salary_model'

ID = '实习岗位ID'
PUBLISH = '发布时间'
SALARY_MID = '薪资中点'
SCOPE = '技能提取范围'

sys.path.insert(0, str(ROOT))

from src import figure_finalize, project_paths  # noqa: E402

# ------------------------------------------------------------------ 技术时间关键词
TECH_TIME_KEYWORDS = [
    'crawl_time', 'scrape_time', 'crawl_date', 'scrape_date',
    '采集时间', '爬取时间', '观测时间', '数据创建时间', '数据更新时间',
    'created_at', 'updated_at', 'create_time', 'update_time',
]
# 业务时间字段（允许进入分析与条件入模）
# 原始/治理层允许保留的技术时间字段（仅追溯、版本排序、审计）
MIN_WINDOW_N = 30
SKILLS = ['Python', 'SQL', '人工智能', '大模型']
def scan_tech_time(columns) -> list:
    """返回列名中命中技术时间关键词的字段列表。"""
    hits = []
    for column in columns:
        low = str(column).lower()
        for keyword in TECH_TIME_KEYWORDS:
            if keyword.lower() in low:
                hits.append(str(column))
                break
    return sorted(set(hits))


def monthly_table(frame: pd.DataFrame, value_column: str, salary_column: str | None = None):
    """按自然月聚合：样本量 / 有效薪资样本量 / 薪资分位。"""
    grouped = frame.groupby('period')
    table = grouped[value_column].count().rename('样本岗位数').to_frame()
    if salary_column:
        grouped = frame.dropna(subset=[salary_column]).groupby('period')
        salary = grouped[salary_column]
        table['有效薪资岗位数 n'] = salary.count()
        table['薪资中位数 Median'] = salary.median()
        table['P25'] = salary.quantile(0.25)
        table['P75'] = salary.quantile(0.75)
        table['IQR'] = table['P75'] - table['P25']
        table['有效薪资岗位数 n'] = table['有效薪资岗位数 n'].fillna(0).astype(int)
    return table


def main() -> int:
    started = datetime.now()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)

    report: dict = {'执行时间': started.strftime('%Y-%m-%d %H:%M:%S')}

    # ============================================================ 1. 技术时间审计
    analysis = pd.read_parquet(PROCESSED / 'job_analysis_dataset.parquet')
    model = pd.read_parquet(PROCESSED / 'job_salary_model_dataset.parquet')
    entity = pd.read_parquet(PROCESSED / 'job_details_unique.parquet')
    obs = pd.read_parquet(INTERIM / 'job_observation_snapshots.parquet')
    raw = pd.read_parquet(ROOT / 'data' / 'raw' / 'shixiseng_job_details.parquet')

    analysis_hits = scan_tech_time(analysis.columns)
    model_hits = scan_tech_time(model.columns)
    entity_hits = scan_tech_time(entity.columns)
    obs_hits = scan_tech_time(obs.columns)
    raw_hits = scan_tech_time(raw.columns)

    manifest = json.loads((MODELDIR / 'feature_manifest.json').read_text(encoding='utf-8'))
    manifest_columns = (manifest['numeric_columns'] + manifest['categorical_columns']
                        + manifest['multi_value_columns'])
    manifest_hits = scan_tech_time(manifest_columns)

    tech_gate = {
        'CRAWL_TIME_IN_ANALYSIS_DATASET': int(len(analysis_hits)),
        'CREATE_TIME_IN_ANALYSIS_DATASET': int(sum('创建时间' in c or 'create_time' in c.lower()
                                                   for c in analysis_hits)),
        'UPDATE_TIME_IN_ANALYSIS_DATASET': int(sum('更新时间' in c or 'update_time' in c.lower()
                                                   for c in analysis_hits)),
        'CRAWL_TIME_IN_MODEL_DATASET': int(len(model_hits)),
        'CREATE_TIME_IN_MODEL_DATASET': int(sum('创建时间' in c or 'create_time' in c.lower()
                                                for c in model_hits)),
        'UPDATE_TIME_IN_MODEL_DATASET': int(sum('更新时间' in c or 'update_time' in c.lower()
                                                for c in model_hits)),
        'TECH_TIME_DERIVED_FEATURE_COUNT': int(len(manifest_hits)),
        '分析数据集技术时间命中字段': analysis_hits,
        '模型数据集技术时间命中字段': model_hits,
        '实体表技术时间命中字段（仅治理层，允许）': entity_hits,
        '观测快照技术时间命中字段（仅原始层，允许）': obs_hits,
        '原始表技术时间命中字段（仅原始层，允许）': raw_hits,
        '特征清单技术时间命中字段': manifest_hits,
    }

    audit_rows = [
        {'字段': '观测时间', '字段类型': '技术元数据（采集/爬取时间）', '是否业务时间': '否',
         '是否允许进入EDA': '否', '是否允许进入模型': '否',
         '用途': '仅原始观测层记录采集批次；不进入分析与模型'},
        {'字段': '数据创建时间', '字段类型': '技术元数据', '是否业务时间': '否',
         '是否允许进入EDA': '否', '是否允许进入模型': '否',
         '用途': '仅原始层数据追溯与写入审计'},
        {'字段': '数据更新时间', '字段类型': '技术元数据', '是否业务时间': '否',
         '是否允许进入EDA': '否', '是否允许进入模型': '否',
         '用途': '仅原始层数据追溯与写入审计'},
        {'字段': '最终版本首次观测时间 / 末次观测时间', '字段类型': '技术元数据（观测时间派生）',
         '是否业务时间': '否', '是否允许进入EDA': '否', '是否允许进入模型': '否',
         '用途': '仅实体治理阶段用于版本排序与代表观测选取'},
        {'字段': '发布时间', '字段类型': '业务时间', '是否业务时间': '是',
         '是否允许进入EDA': '是', '是否允许进入模型': '条件允许（仅发布时间位置扩展特征）',
         '用途': '业务时间专题分析主时间轴；发布月份与发布星期作扩展敏感性特征'},
        {'字段': '投递截止日期', '字段类型': '业务时间', '是否业务时间': '是',
         '是否允许进入EDA': '是', '是否允许进入模型': '否（只用于构造招聘周期）',
         '用途': '重构招聘周期与计划招聘覆盖'},
        {'字段': 'segment_start / segment_end', '字段类型': '业务时间派生（候选发布时间段）',
         '是否业务时间': '是', '是否允许进入EDA': '是', '是否允许进入模型': '否',
         '用途': '招聘生命周期分析与业务时间队列'},
        {'字段': 'episode_start / episode_end', '字段类型': '业务时间派生（重构招聘周期）',
         '是否业务时间': '是', '是否允许进入EDA': '是', '是否允许进入模型': '否',
         '用途': '计划持续时长与计划招聘覆盖'},
        {'字段': '发布月份 / 发布星期', '字段类型': '业务时间派生', '是否业务时间': '是',
         '是否允许进入EDA': '是', '是否允许进入模型': '是（仅发布时间位置扩展特征）',
         '用途': '发布时间位置特征（扩展敏感性对照）'},
    ]

    # ============================================================ 2. 业务时间聚合
    entity = entity[[ID, PUBLISH]].copy()
    entity[PUBLISH] = pd.to_datetime(entity[PUBLISH], errors='coerce')
    salary = model[[ID, SALARY_MID]].copy()
    scope = analysis[[ID, SCOPE]].copy()
    entity = entity.merge(salary, on=ID, how='left').merge(scope, on=ID, how='left')
    entity['period'] = entity[PUBLISH].dt.to_period('M').astype(str)
    entity['quarter'] = entity[PUBLISH].dt.to_period('Q').astype(str)
    entity['week'] = entity[PUBLISH].dt.to_period('W').astype(str)

    monthly = monthly_table(entity, ID, SALARY_MID)
    monthly.index.name = '月份'

    time_summary = {
        '发布时间范围': f'{entity[PUBLISH].min().date()} ~ {entity[PUBLISH].max().date()}',
        '唯一岗位实体数': int(len(entity)),
        '有效薪资样本数': int(entity[SALARY_MID].notna().sum()),
        '自然月窗口数': int(monthly.shape[0]),
        '薪资样本 n<30 的窗口数': int((monthly['有效薪资岗位数 n'] < MIN_WINDOW_N).sum()),
        '薪资样本 n<30 的窗口': monthly.index[monthly['有效薪资岗位数 n'] < MIN_WINDOW_N].tolist(),
        '发布年份分布': entity[PUBLISH].dt.year.value_counts().sort_index().to_dict(),
    }
    time_summary['发布年份分布'] = {int(k): int(v) for k, v in time_summary['发布年份分布'].items()}

    # 技能命中率（主口径：要求段落优先）
    membership = pd.read_parquet(FEATURES / 'job_skill_membership.parquet')
    main_scope = membership[membership['match_scope'] == 'REQUIREMENT_SECTION']
    scope_main = entity[entity[SCOPE] == 'REQUIREMENT_SECTION'][[ID, 'period']]
    denom = scope_main.groupby('period')[ID].count().rename('主口径岗位数')
    skill_rows = {}
    for skill in SKILLS:
        ids = set(main_scope.loc[main_scope['canonical_skill'] == skill, 'intern_id'])
        hit = scope_main[scope_main[ID].isin(ids)].groupby('period')[ID].count()
        skill_rows[skill] = hit.reindex(denom.index).fillna(0).astype(int)
    skill_table = pd.DataFrame(skill_rows)
    skill_rate = skill_table.div(denom, axis=0)
    structure = pd.concat([denom, skill_table,
                           skill_rate.add_suffix('命中率')], axis=1)

    quarter = monthly_table(entity.assign(period=entity['quarter']), ID, SALARY_MID)
    quarter.index.name = '季度'
    week_2026 = entity[entity[PUBLISH].dt.year == 2026].copy()
    week_2026['period'] = week_2026['week']
    weekly = monthly_table(week_2026, ID, SALARY_MID)
    weekly.index.name = 'ISO周'

    sparse = monthly.copy()
    sparse['是否小样本(n<30)'] = np.where(sparse['有效薪资岗位数 n'] < MIN_WINDOW_N, '是', '否')
    sparse['处理'] = np.where(sparse['是否小样本(n<30)'] == '是',
                             '仅展示，不作薪资趋势解释', '可参与趋势描述')

    # ============================================================ 3. 图件
    plt.rcParams.update({
        'font.family': ['Times New Roman', 'Microsoft YaHei'],
        'font.size': 8.6,
        'axes.unicode_minus': False,
        'axes.edgecolor': 'black',
        'axes.linewidth': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.dpi': 600,
    })
    grid = {'linestyle': '--', 'linewidth': 0.45, 'alpha': 0.18}
    months = list(monthly.index)
    positions = np.arange(len(months))
    tick_pos = positions[::4]
    tick_lab = [months[i] for i in tick_pos]
    figures = []

    # 图T1：发布时间队列的样本岗位数量（论文版式：15.5 cm 版心宽、无图内总图题）
    with plt.rc_context({'font.size': 11.0, 'axes.labelsize': 12.0,
                         'xtick.labelsize': 11.0, 'ytick.labelsize': 11.0,
                         'legend.fontsize': 11.0}):
        fig, ax = plt.subplots(figsize=(15.5 / 2.54 * 11 / 12, 3.4))
        ax.bar(positions, monthly['样本岗位数'].to_numpy(), width=0.68,
               color='#4C72B0', edgecolor='black', linewidth=0.4,
               label='样本岗位数（含面议）')
        ax.plot(positions, monthly['有效薪资岗位数 n'].to_numpy(), color='#DD8452',
                linewidth=1.2, marker='o', markersize=2.4, label='有效薪资岗位数')
        ax.set_xticks(tick_pos)
        # rotation_mode='anchor' + ha='right'：把标签末端锚定在其刻度上（实测扎入坐标区 0 px、
        # 末端越过刻度 57 px@600dpi）；默认模式按旋转包围盒对齐会让文字整体左偏约 135 px，
        # 而 ha='center' + anchor 会让文字绕锚点转上去 79 px、扎进坐标区，故不采用
        ax.set_xticklabels(tick_lab, rotation=45, ha='right', rotation_mode='anchor')
        ax.set_xlabel('发布时间队列（自然月）')
        ax.set_ylabel('岗位数（个）')
        ax.grid(True, axis='y', **grid)
        ax.set_axisbelow(True)
        # 图例放在坐标区左上角、单列竖排
        ax.legend(loc='upper left', ncol=1, frameon=False)
        fig.subplots_adjust(left=0.105, right=0.985, bottom=0.24, top=0.965)
        png1 = FIGDIR / 'fig_01_publish_cohort_count.png'
        diagnostics = figure_finalize.save_paper_figure(
            fig, FIGDIR, 'fig_01_publish_cohort_count', subfigures=[],
            meta={'数据来源': '唯一岗位表按岗位发布时间聚合到自然月',
                  '口径': '柱＝样本岗位数（含薪资面议），线＝有效薪资岗位数；'
                          '逐月对照只描述本样本，不代表市场岗位存量',
                  '用途': '第4章 图 4-9（E8 新图编号顺延后）重绘'})
        plt.close(fig)
        print('图T1 门禁未通过项:', figure_finalize.failed_paper_gates(diagnostics) or '无')
    figures.append(str(png1.relative_to(ROOT)).replace('\\', '/'))

    # 图T2、图T3：论文版式。绘制与门禁逻辑统一在 49 号重绘脚本的 save_salary_figure /
    # save_skill_figure 中：无图内总图题、600 dpi PNG + 矢量 PDF、内向刻度、浅虚线网格，
    # 薪资图对 n < 30 的窗口加浅色底纹，技能图横轴沿用「只排给出命中率的窗口」的原有口径，
    # 避免与全文其他重绘图件样式漂移
    figure_spec = importlib.util.spec_from_file_location(
        '_fig49_time_cohort', ROOT / 'scripts' / '49_redraw_time_cohort_figures.py')
    figure_module = importlib.util.module_from_spec(figure_spec)
    sys.modules['_fig49_time_cohort'] = figure_module
    figure_spec.loader.exec_module(figure_module)
    full_median = float(pd.read_excel(TABLES / '21_eda_statistical_analysis.xlsx',
                                      sheet_name='02_薪资描述统计')
                        .set_index('指标')['数值']['中位数'])
    diagnostics2 = figure_module.save_salary_figure(monthly, full_median, FIGDIR)
    plt.close('all')
    print('图T2 门禁未通过项:', figure_finalize.failed_paper_gates(diagnostics2) or '无')
    figures.append(str(Path(diagnostics2['png_path']).relative_to(ROOT)).replace('\\', '/'))

    diagnostics3 = figure_module.save_skill_figure(structure, FIGDIR)
    plt.close('all')
    print('图T3 门禁未通过项:', figure_finalize.failed_paper_gates(diagnostics3) or '无')
    figures.append(str(Path(diagnostics3['png_path']).relative_to(ROOT)).replace('\\', '/'))

    # ============================================================ 4. 结果表
    audit = pd.DataFrame(audit_rows)
    tech_sheet = pd.DataFrame([
        {'门禁项': key, '数值': value} for key, value in tech_gate.items()
        if not isinstance(value, list)
    ])
    out = TABLES / project_paths.TABLE_BUSINESS_TIME_DIMENSION
    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        audit.to_excel(writer, sheet_name='01_时间字段审计', index=False)
        monthly.reset_index().to_excel(writer, sheet_name='02_发布时间月度样本数', index=False)
        monthly.reset_index().to_excel(writer, sheet_name='03_发布时间月度薪资分布', index=False)
        structure.reset_index().to_excel(writer, sheet_name='04_发布时间技能或岗位结构', index=False)
        sparse.reset_index().to_excel(writer, sheet_name='05_稀疏时间窗口标记', index=False)
        tech_sheet.to_excel(writer, sheet_name='06_技术时间字段排除审计', index=False)
        quarter.reset_index().to_excel(writer, sheet_name='07_季度敏感性', index=False)
        weekly.reset_index().to_excel(writer, sheet_name='08_2026周度敏感性', index=False)

    summary = {
        '技术时间审计': tech_gate,
        '业务时间专题': {
            **time_summary,
            '主时间轴': '岗位发布时间',
            '主时间粒度': '自然月（敏感性：自然季、2026 自然周）',
            '时间窗口数（月）': int(monthly.shape[0]),
            'n<30 窗口数': int((monthly['有效薪资岗位数 n'] < MIN_WINDOW_N).sum()),
            '图件': figures,
            '结果表': str(out.relative_to(ROOT)).replace('\\', '/'),
        },
        '未重训模型': True,
    }
    (METRICS / 'stage_26_7_time_and_audit.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print('=' * 78)
    print('Stage26.7 技术时间审计：', json.dumps(
        {k: v for k, v in tech_gate.items() if not isinstance(v, list)}, ensure_ascii=False))
    print('发布时间范围：', time_summary['发布时间范围'])
    print('月度窗口：%d；n<30 窗口：%d' % (monthly.shape[0], (monthly['有效薪资岗位数 n'] < 30).sum()))
    print('结果表：', out)
    for path in figures:
        print('图件：', path)
    print('=' * 78)
    return 0


if __name__ == '__main__':
    sys.exit(main())

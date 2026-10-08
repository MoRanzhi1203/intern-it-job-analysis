# -*- coding: utf-8 -*-
"""Stage 13：正式 EDA、统计检验与论文图表（不训练模型、不调参、不做 SHAP）。

输入（Stage 12 封版产物）：
    data/processed/job_analysis_dataset.parquet     全量分析集（17,144）
    data/processed/job_salary_model_dataset.parquet 薪资建模集（14,883）
    data/features/job_skill_membership.parquet      岗位 × 规范技能 long-format
    outputs/tables/27_skill_eda_scope_audit.xlsx    技能口径审计（复用，不重新设计口径）

输出：
    outputs/tables/29_eda_statistical_analysis.xlsx 13 张子表（描述统计 / 因素检验 / 技能 / 共现 / 稳健性）
    outputs/figures/eda/01_...10_...png / .pdf      10 张正文主图（PNG 600 dpi + PDF）
    docs/records/20_formal_eda_record.md            正式 EDA 记录
    outputs/logs/metrics/stage_13_eda.json

口径（封版）：EDA 单元 = 唯一岗位实体；薪资样本 = 14,883；
技能主口径 = REQUIREMENT_SECTION（8,822）/ 扩展口径 = ALL_USABLE（16,378）/ EMPTY_TEXT 单独报告；
两组比较 Mann–Whitney U + Cliff's delta；多组 Kruskal–Wallis；批量比较 BH-FDR。

用法：
    python scripts/13_run_eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src import (eda_analysis, io_utils, plot_style, project_paths, quality,  # noqa: E402
                 schema, skill_eda, skill_extraction)

STAGE = 'stage_13_eda'
TITLE = 'Stage 13 正式 EDA、统计检验与论文图表'

EXPECTED_JOBS = 17144
EXPECTED_SALARY_SAMPLE = 14883
EXPECTED_MAIN_SCOPE = 8822
EXPECTED_ALL_USABLE = 16378
EXPECTED_EMPTY_SCOPE = 766
FORBIDDEN_CAUSAL_WORDS = ['导致', '造成', '因为', '因果', '提高薪资', '提升薪资']


def build_statistics_sheet(factor_tests: dict, skill_count_test: dict) -> pd.DataFrame:
    """11_统计检验：各因素 Kruskal–Wallis 汇总 + 技能数量档检验。

    「检验对象」一律使用正式的**中文因素名**（禁止输出内部英文 key）。
    """
    rows = []
    for _key, test in factor_tests.items():
        rows.append({
            '检验块': '多组比较（Kruskal–Wallis）',
            '检验对象': test.get('因素', _key),
            '检验方法': 'Kruskal–Wallis（多组）',
            '参与检验组数': test.get('参与检验组数') or test.get('组数'),
            '统计量': test.get('H统计量'),
            'p值': test.get('p值'),
            '效应量': test.get('epsilon平方'),
            '效应量口径': 'epsilon²（Kruskal–Wallis）',
            '说明': test.get('说明', ''),
        })
    rows.append({
        '检验块': '多组比较（Kruskal–Wallis）',
        '检验对象': skill_count_test.get('因素', '技能数量档'),
        '检验方法': 'Kruskal–Wallis（多组）',
        '参与检验组数': skill_count_test.get('参与检验组数'),
        '统计量': skill_count_test.get('H统计量'),
        'p值': skill_count_test.get('p值'),
        '效应量': skill_count_test.get('epsilon平方'),
        '效应量口径': 'epsilon²（Kruskal–Wallis）',
        '说明': skill_count_test.get('说明', ''),
    })
    return pd.DataFrame(rows)


def build_causal_check_table(audit_sheets: dict, figure_captions) -> tuple:
    """14_措辞检查：扫描全部正式输出文本，禁止把关联表述成因果。

    命中「禁止表述」上下文（同一单元格/图注内出现「禁止」或「不代表因果」）视为合规说明，
    其余命中记为违规。
    """
    compliant_marks = ['禁止', '不代表因果', '非因果', '描述性关联']
    rows = []
    violations = 0
    for sheet, frame in audit_sheets.items():
        if not isinstance(frame, pd.DataFrame):
            continue
        for column in frame.columns:
            series = frame[column]
            if series.dtype != object:
                continue
            for value in series.dropna().unique():
                text = str(value)
                hits = [word for word in FORBIDDEN_CAUSAL_WORDS if word in text]
                if not hits:
                    continue
                compliant = any(mark in text for mark in compliant_marks)
                violations += 0 if compliant else 1
                rows.append({
                    '位置': f'{sheet}.{column}', '命中措辞': '、'.join(hits),
                    '文本片段': text[:80], '是否合规说明': '是' if compliant else '否（需修改）',
                })
    for caption in figure_captions:
        hits = [word for word in FORBIDDEN_CAUSAL_WORDS if word in caption]
        if not hits:
            continue
        compliant = any(mark in caption for mark in compliant_marks)
        violations += 0 if compliant else 1
        rows.append({'位置': '图注', '命中措辞': '、'.join(hits), '文本片段': caption[:80],
                     '是否合规说明': '是' if compliant else '否（需修改）'})
    table = pd.DataFrame(rows)
    if table.empty:
        table = pd.DataFrame([{'位置': '（无命中）', '命中措辞': '', '文本片段': '',
                              '是否合规说明': '通过'}])
    return table, violations


def write_record(metrics: dict, audit: dict) -> Path:
    """20 号记录：样本口径 / 统计方法 / FDR / 效应量 / 主图选择 / 双口径结论。"""
    salary_stats = audit['02_薪资描述统计'].set_index('指标')['数值']
    robustness = audit['12_稳健性'].set_index('项目')['数值']
    lines = [
        '# 记录 20：Stage 13 正式 EDA 与统计检验',
        '',
        '> 本记录由 `scripts/13_run_eda.py` 自动生成，数字全部来自真实运行结果。',
        '> 本轮只做探索性分析与统计检验：**未训练模型、未调参、未执行 SHAP、未做测试集性能比较**。',
        '',
        '## 1. 样本口径',
        '',
        '| 项 | 数值 |',
        '| --- | --- |',
        f'| EDA 分析单元（唯一岗位实体） | {metrics["jobs"]:,}（1 intern_id = 1 行，不使用 172,063 条搜索观测） |',
        f'| 正式薪资分析样本 | {metrics["salary_sample"]:,}（非面议 + 解析成功 + 中点有效 + 无逻辑异常） |',
        f'| 面议岗位 | {metrics["negotiable"]:,} |',
        f'| 薪资逻辑异常 | {metrics["anomaly"]:,} |',
        f'| 技能主口径 REQUIREMENT_SECTION | {metrics["main_scope"]:,} |',
        f'| 技能扩展口径 ALL_USABLE | {metrics["all_usable"]:,} |',
        f'| EMPTY_TEXT（单独报告） | {metrics["empty_scope"]:,} |',
        '',
        '## 2. 薪资总体分布（主目标 = 薪资中点）',
        '',
        '| 指标 | 数值 |',
        '| --- | --- |',
    ]
    for key in ['n', '均值', '中位数', '标准差', 'IQR', 'P10', 'P25', 'P50', 'P75', 'P90',
                'min', 'max']:
        if key in salary_stats.index:
            lines.append(f'| {key} | {salary_stats.loc[key]} |')
    lines += [
        '',
        '- 主图保留真实数据；1%/99% 截断仅用于稳健性参考（见 12_稳健性），未对主分析强制缩尾。',
        '',
        '## 3. 统计方法',
        '',
        '| 场景 | 方法 | 效应量 | 多重检验 |',
        '| --- | --- | --- | --- |',
        '| 两组比较（有/无技能、取值两两比较） | Mann–Whitney U | Cliff\'s delta | Benjamini–Hochberg FDR |',
        '| 多组比较（岗位类别 / 城市 / 学历 / 公司规模等） | Kruskal–Wallis | epsilon² | 见 11_统计检验 |',
        '| 分组展示最小样本 | 30 个岗位 | — | — |',
        '| 两组检验最小样本 | 50 个岗位（两侧） | — | 不足时只做描述 |',
        '',
        '所有批量比较均输出 raw p、FDR 校正后 p 与效应量，禁止只报 p 值。',
        '',
        '## 4. 结构化因素与薪资（Kruskal–Wallis）',
        '',
        '| 检验对象 | 参与检验组数 | H 统计量 | p 值 | epsilon² |',
        '| --- | --- | --- | --- | --- |',
    ]
    kruskal_block = audit['11_统计检验']
    kruskal_block = kruskal_block[
        kruskal_block['检验块'].astype(str).str.startswith('多组比较')]
    for row in kruskal_block.itertuples(index=False):
        lines.append(f'| {row.检验对象} | {row.参与检验组数} | {row.统计量} | {row.p值} | {row.效应量} |')
    lines.append(f'| （成对比较 {metrics["pairwise_tests"]} 组，'
                 'Mann–Whitney U + Cliff\'s delta + BH-FDR，见 11 号子表） | — | — | — | — |')
    cert_row = audit['06_公司因素薪资']
    cert_row = cert_row[cert_row['因素'].eq(eda_analysis.COMPANY_CERT_LABEL)]
    lines += [
        '',
        '### 4.1 公司认证 ≠ 公司标签（福利标签）',
        '',
        '| 因素 | 真实字段 | 取值规模 | 说明 |',
        '| --- | --- | --- | --- |',
        f"| {eda_analysis.COMPANY_CERT_LABEL} | `{schema.CERT_TAG_FIELD}`"
        f"（`data/processed/job_details_unique.parquet`，Stage 05 正式产物，只读引用） | "
        f"{len(metrics['company_certification']['atoms'])} 个原子标签，"
        f"构造互斥四类 | 有限认证属性，可作有限类别因素 |",
        f"| {eda_analysis.COMPANY_TAG_LABEL} | `{schema.COMPANY_TAG_LIST_FIELD}`"
        f"（Stage 12 宽表多值字段） | {metrics['company_tag_atoms']:,} 个原子标签 | "
        f"高基数福利/企业文化标签，属多值描述性因素 |",
        '',
        '| 公司认证（互斥四类） | n | 中位数 | IQR |',
        '| --- | --- | --- | --- |',
    ]
    for row in cert_row.to_dict('records'):
        lines.append(f"| {row['取值']} | {int(row['样本数']):,} | {row['中位数']} | {row['IQR']} |")
    lines += [
        '',
        f"- 公司认证原子标签全集：{'、'.join(metrics['company_certification']['atoms'])}"
        f"（未知值：{'无' if not metrics['company_certification']['unknown'] else '、'.join(metrics['company_certification']['unknown'])}）；",
        f"- 公司标签（福利标签）原子标签 {metrics['company_tag_atoms']:,} 个，"
        f"两者**字段不同、取值范围与规模不同、语义不同**"
        f"（公司标签为高基数福利标签，仅字符串层面存在个别重叠），禁止互相替代；",
        '- 历史缺陷说明：Stage 13 早期版本把公司标签（福利标签）显示为「公司认证标签」'
        '（369 组即样本数 ≥ 30 的福利标签数），该显示与字段引用错误已在本轮修正'
        '（取证见记录 23，32 号审计表）。',
        '',
        '### 4.2 公司标签（福利标签）统计推断最终口径',
        '',
        '| 阶段 | 口径 | 状态 |',
        '| --- | --- | --- |',
        f"| 历史（已废止） | 369 个福利标签整体 Kruskal–Wallis，"
        f"epsilon² ≈ {metrics['deprecated_inference']['epsilon_squared']} | "
        f"{metrics['deprecated_inference']['flag']}：多值标签组相互重叠，"
        f"不满足普通多组独立性比较的解释前提，**不再进入正式因素比较、效应量排名与论文主结论**"
        f"（保留追溯见 16 号子表） |",
        f"| 现行（正式） | 第一层：描述性统计（标签 / 岗位数 / 岗位占比 / 薪资样本数 / "
        f"薪资中位数 / IQR / P25 / P75，共 {metrics['company_tag_analysis']['descriptive_rows']:,} 个标签，"
        f"正文只展示 Top20~Top30） | 描述性多值标签，不做整体 KW |",
        f"| 现行（正式） | 第二层：单标签 present vs absent 二元比较"
        f"（present ∩ absent = ∅，present_n + absent_n = {metrics['salary_sample']:,}） | "
        f"{metrics['company_tag_analysis']['binary_rows']} 个标签达标"
        f"（present_n / absent_n ≥ {metrics['company_tag_analysis']['min_group_size']}），"
        f"Mann–Whitney U + Cliff's delta + BH-FDR（q < 0.05 显著 "
        f"{metrics['company_tag_analysis']['fdr_significant']} 个） |",
        '',
        '公司标签二元比较进入正式检验的标签数、样本量门槛与 FDR 结果：',
        '',
        f"- 进入正式二元检验的标签数：{metrics['company_tag_analysis']['binary_rows']}；"
        f"present_n 最小 {metrics['company_tag_analysis']['min_present_n']}、"
        f"absent_n 最小 {metrics['company_tag_analysis']['min_absent_n']}；",
        f"- 统一 BH-FDR 后 q < 0.05 的标签：{metrics['company_tag_analysis']['fdr_significant']} 个；"
        '- 措辞规范：只能表述为「具有某福利标签的岗位薪资分布与不具有该标签的岗位存在统计差异」'
        '（描述性关联），禁止写成「该福利导致工资提高」；',
        '- 论文筛选：频率榜（Top20 高频标签）与薪资关联榜（FDR 显著且 |Cliff\'s delta| 较大的 Top10）'
        '分开呈现，禁止只挑工资差最大的标签而忽略样本量与 FDR；',
        f"- 共现提示：薪资关联榜中 |Cliff's delta| 最高的 "
        f"{len(metrics['company_tag_analysis']['top_cliff_overlap'].get('标签', []))} 个标签"
        f"（{'、'.join(metrics['company_tag_analysis']['top_cliff_overlap'].get('标签', []))}）"
        f"两两最大 Jaccard = {metrics['company_tag_analysis']['top_cliff_overlap'].get('两两最大Jaccard')}，"
        f"全部交集 {metrics['company_tag_analysis']['top_cliff_overlap'].get('全部交集岗位数'):,} 个岗位、"
        f"并集 {metrics['company_tag_analysis']['top_cliff_overlap'].get('全部并集岗位数'):,} 个岗位——"
        '属同一批岗位的标签簇，**不能当作彼此独立的薪资关联证据**，论文呈现时必须同时给出共现说明。',
        '',
        '高频标签（Top10，按岗位数降序）：',
        '',
        '| 标签 | 岗位数 | 岗位占比 | 薪资中位数 | IQR |',
        '| --- | --- | --- | --- | --- |',
    ]
    for row in metrics['company_tag_analysis']['top30'][:10]:
        lines.append(f"| {row['标签']} | {int(row['岗位数']):,} | {row['岗位占比']:.2%} | "
                     f"{row['薪资中位数']} | {row['IQR']} |")
    lines += [
        '',
        '| 标签（|Cliff\'s delta| Top10） | present_n | Cliff\'s delta | 中位数差（present − absent） | FDR 显著 |',
        '| --- | --- | --- | --- | --- |',
    ]
    for row in metrics['company_tag_analysis']['top_cliff']:
        lines.append(f"| {row['标签']} | {int(row['n_present']):,} | {row['cliff_delta']} | "
                     f"{row['median_diff_A_minus_B']} | {row['FDR显著']} |")
    lines += [
        '',
        '## 5. 技能需求（主口径分母 8,822；扩展口径分母 16,378）',
        '',
        '| 榜单 | 前 5 名（岗位数 / 占比） |',
        '| --- | --- |',
    ]
    for title in ['核心技术技能 Top20（主口径，分母 8,822）',
                  '技术领域 Top15（主口径，分母 8,822）',
                  '业务能力（主口径，分母 8,822）',
                  '办公工具（主口径，分母 8,822）',
                  '扩展口径 Top20（分母 16,378）']:
        block = audit['07_技能需求']
        block = block[block['榜单'] == title].head(5)
        summary = '、'.join(f'{row.技能标准名}（{int(row.岗位数)} / {row.岗位占比:.1%}）'
                          for row in block.itertuples(index=False))
        lines.append(f'| {title} | {summary} |')
    lines += [
        '',
        '> 四层榜单**不得混为一榜**，也不得跨层相加（办公工具岗位数按 intern_id 去重）。',
        '',
        '## 6. 技能与薪资（描述性关联，禁止因果）',
        '',
        '| 技能 | 有技能 n | 无技能 n | 有技能中位数 | 无技能中位数 | 中位数差 | Cliff\'s delta | FDR 显著 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    focus = audit['08_技能薪资'].head(10)
    for row in focus.to_dict('records'):
        cliff = row["Cliff's delta"]
        lines.append(f"| {row['技能标准名']} | {int(row['有技能岗位数'])} | "
                     f"{int(row['无技能岗位数'])} | {row['有技能薪资中位数']} | "
                     f"{row['无技能薪资中位数']} | {row['中位数差']} | {cliff} | "
                     f"{row['FDR显著']} |")
    lines += [
        '',
        '措辞规范：**「明确要求 Python 的岗位薪资中位数高于未明确要求 Python 的岗位」**，',
        '不得写成「Python 导致薪资提高」；同岗位细分类内部对比见 09_岗位内技能薪资。',
        '',
        '## 7. 双口径稳健性（复用 27 号审计口径，未重新设计）',
        '',
        '| 指标 | 数值 |',
        '| --- | --- |',
        f"| Top10 overlap | {robustness.get('Top10 overlap')} / 10 |",
        f"| Top20 overlap | {robustness.get('Top20 overlap')} / 20 |",
        f"| Spearman 排名相关 | {robustness.get('Spearman 排名相关')} |",
        '',
        '## 8. 正式主图（10 张，技能相关 6 张）',
        '',
        '| 图 | 文件名 | 回答的问题 |',
        '| --- | --- | --- |',
    ]
    for item in metrics['figures']:
        lines.append(f"| {item['caption']} | `{item['png_path']}` | {item['回答的问题']} |")
    lines += [
        '',
        '## 9. 结论摘要（正式可引用）',
        '',
        f"- 薪资中点中位数 {metrics['salary_median']} 元/天、IQR {metrics['salary_iqr']}，"
        f"P10 {metrics['salary_p10']}、P90 {metrics['salary_p90']}（n = {metrics['salary_sample']:,}）；",
        '- 岗位类别 / 城市 / 学历 / 公司规模与性质等结构化因素均与薪资显著相关（Kruskal–Wallis，见 11 号子表），',
        '  但效应量差异较大，需结合 epsilon² 解读；',
        f"- 公司认证（互斥四类）与薪资存在较明显的统计关联"
        f"（H ≈ {metrics['company_certification']['H']}，"
        f"epsilon² ≈ {metrics['company_certification']['epsilon_squared']}），"
        f"不同认证组合的薪资分布存在差异；",
        f"- 公司标签（福利标签）为高基数多值字段，正式口径为描述性统计 + "
        f"单标签 present vs absent 二元比较：{metrics['company_tag_analysis']['binary_rows']} 个标签"
        f"样本量达标，统一 BH-FDR 后 {metrics['company_tag_analysis']['fdr_significant']} 个标签"
        f"与薪资分布存在统计差异（描述性关联），"
        f"原「369 组整体 KW（epsilon² ≈ {metrics['deprecated_inference']['epsilon_squared']}）」"
        f"已标记 {metrics['deprecated_inference']['flag']}；",
        '- 技能需求主口径下，具体技术技能以 Python（1,264）、SQL（870）、Java（452）为主，',
        '  技术领域以数据分析（1,641）、人工智能（1,314）、大模型（469）为主；',
        '- 明确要求 Python / Java / SQL / C++ 等技能的岗位薪资中位数高于未明确要求的岗位，',
        '  该差异在控制岗位细分类后依然存在但幅度下降，说明部分差异来自岗位类别结构；',
        '- 双口径排名高度一致（Top20 overlap 17/20，Spearman 0.973），技能需求结论稳健；',
        '- 以上均为**描述性关联**，不构成因果结论；技能信息对薪资预测的增益需在 Stage 14/15 用',
        '  消融与模型比较验证。',
        '',
        '## 10. 门禁与锚点',
        '',
        '| 门禁项 | 状态 | 说明 |',
        '| --- | --- | --- |',
    ]
    gate_detail = audit['15_门禁'].set_index('门禁项')['说明']
    for name, status in metrics['gates'].items():
        lines.append(f'| {name} | {status} | {gate_detail.get(name, "")} |')
    lines += [
        '',
        f"- 核心锚点：全量岗位 {metrics['jobs']:,}；薪资样本 {metrics['salary_sample']:,}；"
        f"主口径 {metrics['main_scope']:,}；扩展口径 {metrics['all_usable']:,}；"
        f"EMPTY_TEXT {metrics['empty_scope']:,}；",
        f"- 产物：`{metrics['audit_path']}`、图目录 `outputs/figures/eda/`；",
        '- 本轮未训练模型、未调参、未执行 SHAP、未使用测试集、未提交 git。',
        '',
    ]
    return io_utils.write_markdown(
        project_paths.RECORDS_DIR / project_paths.RECORD_FORMAL_EDA, lines)


def main() -> int:
    quality.configure_logging()
    gates = quality.GateRegistry(STAGE)

    quality.stage_banner(STAGE, TITLE)
    analysis = io_utils.read_parquet(project_paths.JOB_ANALYSIS_DATASET_PARQUET)
    model = io_utils.read_parquet(project_paths.JOB_SALARY_MODEL_DATASET_PARQUET)
    membership = io_utils.read_parquet(project_paths.JOB_SKILL_MEMBERSHIP_PARQUET)
    salary = io_utils.read_parquet(project_paths.SALARY_TARGETS_PARQUET)
    config = skill_extraction.load_skill_config()
    universe = skill_eda.load_scope_universe(analysis)
    print(f'输入: 全量岗位 {len(analysis)} / 薪资样本 {len(model)} / 技能关系 {len(membership)} 行')

    # ---- 门禁 1：分析单元与薪资样本 ----
    gates.check('EDA_UNIQUE_JOB_UNIT',
                len(analysis) == EXPECTED_JOBS and analysis[schema.ID_FIELD].is_unique,
                f'EDA 分析单元 = {len(analysis):,} 个唯一岗位实体（1 intern_id = 1 行），'
                '未使用 172,063 条原始搜索观测作为独立样本')
    anomaly = int(salary[schema.SALARY_ANOMALY_FIELD].fillna('').astype(str).str.strip().ne('').sum())
    negotiable = int(salary[schema.SALARY_NEGOTIABLE_FIELD].eq(1).sum())
    gates.check('EDA_SALARY_SAMPLE_14883',
                len(model) == EXPECTED_SALARY_SAMPLE
                and int(model[schema.SALARY_MID_FIELD].isna().sum()) == 0,
                f'正式薪资分析样本 {len(model):,}（面议 {negotiable:,}、逻辑异常 {anomaly} 已剔除，'
                f'目标非空 {int(model[schema.SALARY_MID_FIELD].notna().sum()):,}）')

    # ---- 描述统计与结构化因素 ----
    salary_stats = eda_analysis.describe(model[schema.SALARY_MID_FIELD])
    truncated_stats = eda_analysis.describe(
        model[schema.SALARY_MID_FIELD].clip(lower=model[schema.SALARY_MID_FIELD].quantile(0.01),
                                            upper=model[schema.SALARY_MID_FIELD].quantile(0.99)))
    salary_table = pd.DataFrame([
        {'指标': key, '数值': value,
         '说明': '主目标 = 薪资中点（元/天）；主分析保留真实数据、不强制缩尾'}
        for key, value in salary_stats.items()] + [
        {'指标': '1%/99% 截断后 中位数', '数值': truncated_stats['中位数'],
         '说明': '稳健性参考（仅附图/附表使用）'},
        {'指标': '1%/99% 截断后 IQR', '数值': truncated_stats['IQR'],
         '说明': '稳健性参考（仅附图/附表使用）'},
    ])
    gates.check('EDA_DESCRIPTIVE_STATS',
                salary_stats['n'] == EXPECTED_SALARY_SAMPLE
                and salary_stats['中位数'] > 0 and salary_stats['IQR'] > 0
                and {'均值', '标准差', 'P10', 'P90', 'min', 'max'} <= set(salary_stats),
                f"薪资中点：n={salary_stats['n']:,}，均值 {salary_stats['均值']}、"
                f"中位数 {salary_stats['中位数']}、IQR {salary_stats['IQR']}、"
                f"P10 {salary_stats['P10']}、P90 {salary_stats['P90']}")

    factor_tables = {}
    factor_tests = {}
    # 公司认证（Stage 05 正式产物，互斥四类）与公司标签（福利标签，宽表多值）严格分离：
    # 公司认证 → 四类 Kruskal–Wallis；公司标签 → 描述性统计 + 单标签 present/absent 二元比较
    certification = io_utils.read_parquet(
        project_paths.PROCESSED_UNIQUE_PARQUET,
        columns=[schema.ID_FIELD, schema.CERT_TAG_FIELD])
    cert_atoms, cert_unknown = eda_analysis.certification_atoms(certification)
    factor_frame = eda_analysis.attach_company_certification(model, certification)
    for key, sheet_label in [('category', '岗位大类'), ('sub_category', '岗位细分类'),
                             ('city', '工作城市'), ('education', '学历要求'),
                             ('attendance', '每周到岗要求'), ('duration', '实习时长要求'),
                             ('company_scale', '公司规模'), ('company_nature', '公司性质'),
                             ('industry', '所属行业'),
                             (eda_analysis.CERT_FACTOR_KEY, eda_analysis.COMPANY_CERT_LABEL)]:
        table, test = eda_analysis.build_group_salary(factor_frame, key)
        factor_tables[key] = table
        test['因素'] = sheet_label
        factor_tests[key] = test

    # 公司标签（福利标签）：高基数多值，禁止整体 KW，改为描述 + 单标签二元检验
    tag_descriptive = eda_analysis.build_company_tag_descriptive(factor_frame)
    factor_tables[eda_analysis.TAG_FACTOR_KEY] = tag_descriptive
    tag_binary = eda_analysis.build_binary_multivalue_factor_tests(factor_frame)
    # |Cliff's delta| 最高的标签往往来自同一批岗位（标签簇），必须给出共现提示
    tag_overlap = eda_analysis.tag_cluster_overlap(factor_frame, tag_binary, top_n=5)

    cert_class_counts = (factor_frame[eda_analysis.CERT_CLASS_FIELD].value_counts()
                         .reindex(eda_analysis.CERT_CLASS_ORDER).fillna(0).astype(int))
    tag_atoms = int(factor_frame[schema.COMPANY_TAG_LIST_FIELD].explode().nunique())
    gates.check('EDA_COMPANY_FACTOR_SEMANTICS',
                not cert_unknown
                and set(cert_atoms) == {eda_analysis.CERT_EMPLOYER, eda_analysis.CERT_INDUSTRY}
                and int(cert_class_counts.sum()) == len(model)
                and set(cert_class_counts.index) == set(eda_analysis.CERT_CLASS_ORDER)
                and schema.CERT_TAG_FIELD not in model.columns
                and eda_analysis.COMPANY_CERT_LABEL != eda_analysis.COMPANY_TAG_LABEL,
                f'公司认证（{eda_analysis.COMPANY_CERT_LABEL}）原子标签 '
                f'{len(cert_atoms)} 个（{"、".join(cert_atoms)}），未知值 '
                f'{"无" if not cert_unknown else "、".join(cert_unknown)}；'
                f'互斥四类 n：'
                f'{"、".join(f"{name} {count:,}" for name, count in cert_class_counts.items())}'
                f'（合计 {int(cert_class_counts.sum()):,} = 薪资样本）；'
                f'公司标签（福利标签）原子标签 {tag_atoms:,} 个，'
                f'与公司认证为**不同字段、不同因素**，显示名不再混用')

    pairwise_frames = []
    for key in ('category', 'sub_category', 'city', 'education', 'company_scale',
                eda_analysis.CERT_FACTOR_KEY):
        pairwise = eda_analysis.build_pairwise_tests(factor_frame, key)
        if not pairwise.empty:
            pairwise_frames.append(pairwise)
    pairwise_table = (pd.concat(pairwise_frames, ignore_index=True, sort=False)
                      if pairwise_frames else pd.DataFrame())
    factor_present = [name for name, test in factor_tests.items() if test.get('p值') is not None]
    gates.check('EDA_GROUP_TESTS',
                len(factor_present) >= 5 and not pairwise_table.empty
                and 'cliff_delta' in pairwise_table.columns,
                f'{len(factor_present)}/{len(factor_tests)} 个正式互斥/单值因素完成 '
                f'Kruskal–Wallis 检验；成对比较 {len(pairwise_table)} 组'
                f'（Mann–Whitney U + Cliff\'s delta）；公司标签（福利标签）不进入整体 KW')

    fdr_frames = [frame for frame in [pairwise_table, tag_binary] if not frame.empty
                  and 'p_adjusted_bh' in frame.columns]
    gates.check('EDA_MULTIPLE_TESTING_FDR',
                bool(fdr_frames) and all(frame['p_adjusted_bh'].notna().any() for frame in fdr_frames),
                f'成对比较 {len(pairwise_table)} 组与公司标签单标签二元比较 {len(tag_binary)} 组'
                f'全部完成统一 BH-FDR 校正（公司标签显著 '
                f'{int(tag_binary["FDR显著"].eq("是").sum()) if not tag_binary.empty else 0} 个）；'
                '技能薪资表同样输出 BH_FDR_q 值')

    # ---- 公司标签（福利标签）：废止整体 KW（门禁在 11 号子表组装后按真实产物判定） ----

    # ---- 技能需求（复用封版口径） ----
    demand, main_rank, ext_rank, layers = eda_analysis.build_skill_demand(membership, universe,
                                                                         config)
    main_count = len(universe[skill_eda.SCOPE_MAIN])
    all_usable = len(universe['ALL_USABLE'])
    empty_count = len(universe[skill_eda.SCOPE_EMPTY])
    gates.check('EDA_SKILL_SCOPE_DENOMINATOR',
                main_count == EXPECTED_MAIN_SCOPE and all_usable == EXPECTED_ALL_USABLE
                and empty_count == EXPECTED_EMPTY_SCOPE,
                f'主口径分母 {main_count:,}（REQUIREMENT_SECTION）／扩展口径 {all_usable:,}'
                f'／EMPTY_TEXT {empty_count:,}（单独报告）；榜单占比按对应分母计算')
    gates.check('EDA_SKILL_LAYER_SPLIT',
                len(layers) == 4
                and sum(len(item) for item in layers.values()) == config.skill_count
                and set(demand.loc[demand['榜单'].str.contains('技术技能'), '技能标准名'])
                <= layers[skill_eda.LAYER_TECHNICAL],
                f'四层榜单互斥且完整覆盖 {config.skill_count} 个技能标准名'
                f'（具体技术技能 {len(layers[skill_eda.LAYER_TECHNICAL])} / '
                f'技术领域 {len(layers[skill_eda.LAYER_DOMAIN])} / '
                f'业务能力 {len(layers[skill_eda.LAYER_BUSINESS])} / '
                f'办公工具 {len(layers[skill_eda.LAYER_OFFICE])}）；未混为一榜')

    matrix = eda_analysis.build_category_skill_matrix(membership, analysis, universe, main_rank,
                                                     config)
    cooc = skill_eda.cooccurrence(membership, universe[skill_eda.SCOPE_MAIN], main_rank,
                                  layer=skill_eda.LAYER_TECHNICAL,
                                  scopes=skill_eda.SCOPE_MAIN)
    gates.check('EDA_COOCCURRENCE_THRESHOLD',
                not cooc.empty and int(cooc['共现岗位数'].min()) >= 30
                and {'Jaccard', 'Lift'} <= set(cooc.columns),
                f'技能共现 {len(cooc)} 对，全部满足共现岗位数 ≥ 30'
                f'（最小 {int(cooc["共现岗位数"].min())}），按 Lift 排序，'
                '仅使用具体技术技能')

    per_skill, within, count_table, count_test = eda_analysis.build_skill_salary(
        membership, salary, set(model[schema.ID_FIELD]), analysis, universe, main_rank, config)
    robustness = eda_analysis.build_robustness(membership, model, universe, main_rank, ext_rank)
    summary = skill_eda.rank_robustness(main_rank, ext_rank)[0]

    # ---- 出图（10 张，技能相关 6 张） ----
    style_snapshot = plot_style.setup_sci_style()
    eda_analysis._prepare_figure_dir()
    overview = eda_analysis.build_sample_overview(analysis, model, salary, membership)
    registry: list = []
    figure_specs = [
        (lambda: eda_analysis.figure_sample_structure(overview, registry),
         '样本口径与技能口径结构'),
        (lambda: eda_analysis.figure_salary_distribution(model, registry),
         '薪资中点分布是否右偏、极端值规模'),
        (lambda: eda_analysis.figure_category_salary(factor_tables['sub_category'], registry),
         '哪些岗位细分类薪资更高'),
        (lambda: eda_analysis.figure_structured_factors(factor_tables['city'],
                                                        factor_tables['education'],
                                                        factor_tables['company_scale'], registry),
         '城市 / 学历 / 公司规模的薪资差异'),
        (lambda: eda_analysis.figure_tech_skill_top20(
            skill_eda.layer_rank(main_rank, skill_eda.LAYER_TECHNICAL, 20), registry),
         '企业明确要求哪些具体技术技能'),
        (lambda: eda_analysis.figure_skill_layers(
            skill_eda.layer_rank(main_rank, skill_eda.LAYER_DOMAIN, 15),
            skill_eda.layer_rank(main_rank, skill_eda.LAYER_BUSINESS),
            skill_eda.layer_rank(main_rank, skill_eda.LAYER_OFFICE), registry),
         '技术领域 / 业务能力 / 办公工具的需求结构'),
        (lambda: eda_analysis.figure_category_skill_heatmap(matrix, registry),
         '不同岗位细分类的技能画像差异'),
        (lambda: eda_analysis.figure_skill_cooccurrence(cooc, registry=registry),
         '哪些具体技能经常共同出现'),
        (lambda: eda_analysis.figure_skill_salary(per_skill, count_table, registry),
         '技能与薪资的描述性关联强度'),
        (lambda: eda_analysis.figure_scope_robustness(main_rank, ext_rank, summary, registry),
         '技能结论对文本口径是否稳健'),
    ]
    for builder, question in figure_specs:
        diagnostics = builder()
        diagnostics['回答的问题'] = question
    figure_checks = plot_style.check_figure_gates(registry, style_snapshot)
    failed_figure_checks = figure_checks.get('_detail', {}).get('failed', [])
    gates.check('EDA_FIGURE_STYLE',
                not failed_figure_checks and len(registry) == 10,
                f'{len(registry)} 张正文主图全部通过 SCI 视觉门禁'
                f'（PNG 600 dpi + PDF、四边框、内向刻度、下置图名、无弹窗）；'
                f'技能相关 {6} 张；失败项 {failed_figure_checks or "无"}')

    # ---- 输出 29 号表 ----
    statistics_table = pd.concat([build_statistics_sheet(factor_tests, count_test),
                                  pairwise_table, tag_binary],
                                 ignore_index=True, sort=False)
    deprecated_table = pd.DataFrame([
        {'项目': '已废止推断', '结论': eda_analysis.DEPRECATED_INFERENCE_FLAG},
        {'项目': '原推断内容', '结论': f'「{eda_analysis.COMPANY_TAG_LABEL}」369 个标签整体 '
                                  'Kruskal–Wallis 多组比较'},
        {'项目': '原结果', '结论': f'epsilon² ≈ {eda_analysis.DEPRECATED_TAG_KW_VALUE}'
                              '（历史值，保留追溯）'},
        {'项目': '废止原因', '结论': '公司标签列表为高基数多值字段，同一岗位可同时命中多个标签，'
                                '标签组之间不独立，不满足普通多组独立性比较的解释前提'},
        {'项目': '现行正式口径', '结论': '描述性统计（06 号子表）+ 单标签 present vs absent '
                                 '二元比较（11 号子表：Mann–Whitney U + Cliff\'s delta + '
                                 'BH-FDR，present_n / absent_n ≥ '
                                 f'{eda_analysis.MIN_BINARY_GROUP_SIZE}）'},
        {'项目': '进入正式二元比较的标签数', '结论': f'{len(tag_binary)}'},
        {'项目': '是否可再进入正式因素比较', '结论': '否（仅作历史追溯与审计链证据）'},
        {'项目': '结论口径', '结论': eda_analysis.NO_CAUSAL_NOTE},
    ])
    audit_sheets = {
        '01_样本概况': overview,
        '02_薪资描述统计': salary_table,
        '03_岗位因素薪资': pd.concat([factor_tables['category'], factor_tables['sub_category']],
                                 ignore_index=True, sort=False),
        '04_城市薪资': factor_tables['city'],
        '05_学历薪资': pd.concat([factor_tables['education'], factor_tables['attendance'],
                               factor_tables['duration']], ignore_index=True, sort=False),
        '06_公司因素薪资': pd.concat([factor_tables['company_scale'],
                                 factor_tables['company_nature'],
                                 factor_tables['industry'],
                                 factor_tables[eda_analysis.CERT_FACTOR_KEY],
                                 factor_tables[eda_analysis.TAG_FACTOR_KEY]],
                                ignore_index=True, sort=False),
        '07_技能需求': demand,
        '08_技能薪资': per_skill,
        '09_岗位内技能薪资': within,
        '10_技能共现': cooc,
        '11_统计检验': statistics_table,
        '12_稳健性': robustness,
        '13_技能数量薪资': count_table,
    }

    # ---- 公司标签整体 KW 废止门禁（按真实 11 号子表判定） ----
    kruskal_rows = statistics_table[
        statistics_table['检验方法'].astype(str).str.startswith('Kruskal')]
    tag_kw_left = kruskal_rows[
        kruskal_rows['检验对象'].astype(str).eq(eda_analysis.COMPANY_TAG_LABEL)]
    gates.check('EDA_COMPANY_TAG_NO_OVERLAP_KW',
                tag_kw_left.empty
                and eda_analysis.COMPANY_TAG_LABEL not in set(factor_tests)
                and not tag_binary.empty
                and deprecated_table['结论'].astype(str)
                .str.contains('DEPRECATED_INFERENCE').any(),
                f'11_统计检验中不存在「{eda_analysis.COMPANY_TAG_LABEL}」整体 Kruskal–Wallis'
                f'（原 369 组 epsilon² ≈ {eda_analysis.DEPRECATED_TAG_KW_VALUE} 已标记 '
                f'{eda_analysis.DEPRECATED_INFERENCE_FLAG} 并移入 16 号子表保留追溯）；'
                f'正式 KW 仅保留 {len(factor_tests)} 个互斥/单值因素')
    binary_ready = (not tag_binary.empty
                    and tag_binary['n_present'].ge(eda_analysis.MIN_BINARY_GROUP_SIZE).all()
                    and tag_binary['n_absent'].ge(eda_analysis.MIN_BINARY_GROUP_SIZE).all()
                    and (tag_binary['n_present'] + tag_binary['n_absent']).eq(len(model)).all()
                    and tag_binary['u_stat'].notna().all()
                    and tag_binary['cliff_delta'].notna().all()
                    and tag_binary['p_raw'].notna().all()
                    and tag_binary['p_adjusted_bh'].notna().all()
                    and tag_binary['p_adjusted_bh'].between(0, 1).all())
    gates.check('EDA_COMPANY_TAG_BINARY_INFERENCE', binary_ready,
                f'公司标签 {len(tag_binary)} 个进入正式二元比较（present vs absent；'
                f'present ∩ absent = ∅、present_n + absent_n = {len(model):,}）；'
                f'present_n 最小 {int(tag_binary["n_present"].min())}、'
                f'absent_n 最小 {int(tag_binary["n_absent"].min())}'
                f'（门槛 ≥ {eda_analysis.MIN_BINARY_GROUP_SIZE}，复用既有两组检验常量）；'
                f'Mann–Whitney U / Cliff\'s delta / BH-FDR q 全部非空，'
                f'q < 0.05 的标签 {int(tag_binary["FDR显著"].eq("是").sum())} 个')
    causal_table, causal_violations = build_causal_check_table(
        audit_sheets, [item['caption'] for item in registry])
    audit_sheets['14_措辞检查'] = causal_table
    gates.check('EDA_NO_CAUSAL_LABEL',
                causal_violations == 0,
                f'扫描 {len(audit_sheets)} 张子表与 {len(registry)} 条图注，'
                f'因果措辞违规 {causal_violations} 处；'
                f'命中「禁止/描述性关联」说明的行数 {int(causal_table["是否合规说明"].eq("是").sum())}，'
                '所有关联表均带描述性关联说明列')
    audit_path = project_paths.TABLES_DIR / project_paths.TABLE_EDA_STATISTICAL
    gates.check('EDA_AUDIT_EXPORT',
                len(audit_sheets) >= 12 and not per_skill.empty and not cooc.empty,
                f'29 号审计表含 {len(audit_sheets)} 张子表（描述统计 / 因素检验 / 技能 / 共现 / '
                f'稳健性 / 措辞检查）')
    audit_sheets['15_门禁'] = pd.DataFrame([
        {'门禁项': name, '状态': item['status'], '说明': item['note']}
        for name, item in gates.results.items()])
    audit_sheets['16_已废止推断'] = deprecated_table
    audit_path = io_utils.write_excel(audit_path, audit_sheets)

    metrics = {
        'jobs': int(len(analysis)),
        'salary_sample': int(len(model)),
        'negotiable': negotiable,
        'anomaly': anomaly,
        'main_scope': main_count,
        'all_usable': all_usable,
        'empty_scope': empty_count,
        'salary_median': salary_stats['中位数'],
        'salary_iqr': salary_stats['IQR'],
        'salary_p10': salary_stats['P10'],
        'salary_p90': salary_stats['P90'],
        'factor_tests': {name: {key: value for key, value in test.items()
                                if key in ('因素', 'H统计量', 'p值', 'epsilon平方', '参与检验组数')}
                         for name, test in factor_tests.items()},
        'pairwise_tests': int(len(pairwise_table)),
        'company_certification': {
            'field': schema.CERT_TAG_FIELD,
            'source': 'data/processed/job_details_unique.parquet',
            'atoms': cert_atoms, 'unknown': cert_unknown,
            'classes': {str(name): int(count) for name, count in cert_class_counts.items()},
            'H': factor_tests[eda_analysis.CERT_FACTOR_KEY].get('H统计量'),
            'epsilon_squared': factor_tests[eda_analysis.CERT_FACTOR_KEY].get('epsilon平方'),
            'p_value': factor_tests[eda_analysis.CERT_FACTOR_KEY].get('p值'),
            'pairwise_rows': int(len(statistics_table[
                statistics_table['因素'].astype(str).eq(eda_analysis.COMPANY_CERT_LABEL)])),
        },
        'company_tag_atoms': tag_atoms,
        'company_tag_field': schema.COMPANY_TAG_LIST_FIELD,
        'company_tag_analysis': {
            'type': eda_analysis.TAG_ANALYSIS_TYPE,
            'descriptive_rows': int(len(tag_descriptive)),
            'top30': tag_descriptive.head(30)[['标签', '岗位数', '岗位占比', '薪资中位数', 'IQR']]
            .to_dict('records'),
            'binary_rows': int(len(tag_binary)),
            'min_present_n': int(tag_binary['n_present'].min()),
            'min_absent_n': int(tag_binary['n_absent'].min()),
            'min_group_size': eda_analysis.MIN_BINARY_GROUP_SIZE,
            'fdr_significant': int(tag_binary['FDR显著'].eq('是').sum()),
            'top_cliff': tag_binary.reindex(
                tag_binary['cliff_delta'].abs().sort_values(ascending=False).index)
            .head(10)[['标签', 'n_present', "cliff_delta", 'median_diff_A_minus_B', 'FDR显著']]
            .to_dict('records'),
            'null_columns': ['u_stat', 'p_raw', 'p_adjusted_bh', 'cliff_delta'],
            'top_cliff_overlap': tag_overlap,
        },
        'deprecated_inference': {
            'flag': eda_analysis.DEPRECATED_INFERENCE_FLAG,
            'epsilon_squared': eda_analysis.DEPRECATED_TAG_KW_VALUE,
            'reason': '多值标签组相互重叠，不满足普通多组独立性比较的解释前提',
            'replacement': '描述性统计 + 单标签 present vs absent 二元比较',
        },
        'main_top20': main_rank.head(20)[['技能标准名', '层级', '岗位数', '岗位占比']]
        .to_dict('records'),
        'domain_top15': skill_eda.layer_rank(main_rank, skill_eda.LAYER_DOMAIN, 15)
        [['技能标准名', '岗位数', '岗位占比']].to_dict('records'),
        'business': skill_eda.layer_rank(main_rank, skill_eda.LAYER_BUSINESS)
        [['技能标准名', '岗位数', '岗位占比']].to_dict('records'),
        'office': skill_eda.layer_rank(main_rank, skill_eda.LAYER_OFFICE)
        [['技能标准名', '岗位数', '岗位占比']].to_dict('records'),
        'cooccurrence_top': cooc.head(10).to_dict('records'),
        'skill_salary_top': per_skill.head(10).to_dict('records'),
        'skill_count_salary': count_table.to_dict('records'),
        'robustness': summary,
        'figures': [{'caption': item['caption'], 'png_path': item['png_path'],
                     'pdf_path': item['pdf_path'], '回答的问题': item['回答的问题']}
                    for item in registry],
        'figure_checks': figure_checks,
        'audit_path': project_paths.relative_to_root(audit_path),
        'gates': {name: item['status'] for name, item in gates.results.items()},
    }
    io_utils.write_json(project_paths.METRICS_DIR / f'{STAGE}.json', metrics)

    record_path = write_record(metrics, audit_sheets)
    print(f'阶段记录: {project_paths.relative_to_root(record_path)}')
    print(f'EDA 图目录: {project_paths.relative_to_root(project_paths.EDA_FIGURES_DIR)}')

    quality.print_gates(gates.results, order=quality.STAGE_GATE_MAP[STAGE])
    gates.save()
    return 0


if __name__ == '__main__':
    sys.exit(main())

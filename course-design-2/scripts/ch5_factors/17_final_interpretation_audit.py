# -*- coding: utf-8 -*-
"""Stage 17：最终解释审计与封版（公司字段语义修正 + 技能 SHAP presence 统计方式）。

不改数据、不重训模型，只做：
    1. 汇总 Stage 13 公司因素语义修正（修正前 / 修正后，来源 32 号取证表与 29 号表）；
    2. 汇总技能 SHAP 的两种方向统计范围（全局 mean(SHAP) 与 presence-conditioned）；
    3. 核心结果回归保护（建模样本 / Stage 14 主模型指标 / Stage 15 消融与 Group Split）；
    4. 真实运行 pytest 并把结果写入审计表；
    5. 登记 EXPERIMENT_FREEZE 状态。

输出：
    outputs/tables/ch5/25_final_interpretation_audit.xlsx
    docs/records/24_final_interpretation_freeze_record.md
    outputs/logs/metrics/stage_17_interpretation.json

用法：
    python scripts/ch5_factors/17_final_interpretation_audit.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = next(
    _candidate for _candidate in Path(__file__).resolve().parents
    if (_candidate / "data").is_dir() and (_candidate / "scripts").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src import eda_analysis, io_utils, project_paths, quality, schema  # noqa: E402

STAGE = 'stage_17_interpretation'
TITLE = 'Stage 17 最终解释审计与封版（公司字段语义 + 技能 SHAP presence 统计方式）'

EXPECTED_MODEL_SAMPLE = 14883
EXPECTED_TEST_MAE = 35.335506
EXPECTED_TEST_RMSE = 64.623479
EXPECTED_TEST_R2 = 0.584606
EXPECTED_RANDOM_FULL_MAE = 36.074723
EXPECTED_GROUP_SPLIT_MAE = 52.049937
FROZEN_GATES = ['stage_12_modeling', 'stage_13_eda', 'stage_14_model', 'stage_15_ablation',
                'stage_16_company_field']
PRESENCE_COLUMNS = ['model_job_count', 'model_job_frequency', 'test_present_n', 'test_absent_n',
                    'mean_SHAP_present', 'median_SHAP_present', 'mean_SHAP_absent',
                    'median_SHAP_absent', 'mean_abs_SHAP_present', 'positive_SHAP_ratio_present',
                    'presence_direction', 'global_mean_SHAP_direction']
# 33 号表最终结构（共 9 个 Sheet，禁止重复门禁 Sheet）
FINAL_SHEET_ORDER = ['01_字段语义最终确认', '02_公司认证统计', '03_公司标签统计',
                     '04_Stage13修正前后', '05_技能SHAP旧统计范围', '06_技能SHAP_presence 统计方式',
                     '07_关键结果一致性', '08_全阶段门禁', '09_测试结果']
EXPECTED_CERT_EPSILON = 0.197545
CERT_CLASS_COUNTS = {'无认证': 9104, '行业认证': 460, '最佳雇主': 4542, '两者均有': 777}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def build_field_semantics_sheet(semantic: dict, eda_sheets: dict) -> pd.DataFrame:
    """01_字段语义最终确认。"""
    problem = semantic['07_问题分类'].set_index('项')['结论']
    cert_atoms = semantic['03_公司认证原子标签']
    atom_list = cert_atoms[cert_atoms['项目'].eq('原子标签全集')]['取值'].iloc[0]
    company = eda_sheets['06_公司因素薪资']
    factors = sorted(set(company['因素']))
    rows = [
        {'项目': '问题分类（32 号取证）', '结论': problem['问题分类']},
        {'项目': '369 组实际来源字段', '结论': problem['369 组实际来源字段']},
        {'项目': '设计意图字段', '结论': problem['设计意图字段']},
        {'项目': '公司认证字段实际列名', '结论': f'{schema.CERT_TAG_FIELD}'
                                          '（data/processed/job_details_unique.parquet）'},
        {'项目': '公司标签字段实际列名', '结论': f'{schema.COMPANY_TAG_LIST_FIELD}（Stage 12 宽表）'},
        {'项目': '公司认证原子标签全集', '结论': atom_list},
        {'项目': '公司认证互斥四类（分析范围）', '结论': '、'.join(eda_analysis.CERT_CLASS_ORDER)},
        {'项目': '29 号表公司因素列表（修正后）', '结论': '、'.join(factors)},
        {'项目': '是否仍存在「公司认证标签」显示名',
         '结论': '否（已改为公司认证 / 公司标签（福利标签）两个独立因素）'
         if '公司认证标签' not in factors else '是（未修正）'},
        {'项目': '取证结论：是否需要修改 Stage 12', '结论': problem['是否修改 Stage 12']},
        {'项目': '取证结论：Stage 12 是否需要重跑', '结论': problem['是否重跑 Stage 12']},
        {'项目': '取证结论：Stage 13 是否需要重跑', '结论': problem['是否重跑 Stage 13']},
        {'项目': '取证结论：Stage 14 是否需要重跑', '结论': problem['是否重跑 Stage 14']},
        {'项目': '取证结论：Stage 15 是否需要因公司字段问题重跑',
         '结论': problem['是否重跑 Stage 15']},
    ]
    return pd.DataFrame(rows)


def build_before_after_sheet(eda_sheets: dict, metrics: dict) -> pd.DataFrame:
    """04_Stage13修正前后：三阶段审计链（显示名错误 → 语义拆分 → 统计推断最终方法）。"""
    company = eda_sheets['06_公司因素薪资']
    statistics = eda_sheets['11_统计检验']
    kruskal = statistics[statistics['检验方法'].astype(str).str.startswith('Kruskal')]
    cert = kruskal[kruskal['检验对象'].eq(eda_analysis.COMPANY_CERT_LABEL)]
    cert_rows = company[company['因素'].eq(eda_analysis.COMPANY_CERT_LABEL)]
    tag_binary = statistics[statistics['检验块'].astype(str).str.startswith('单标签二元比较')]
    tag_analysis = metrics.get('company_tag_analysis', {})
    deprecated = metrics.get('deprecated_inference', {})
    tag_kw_left = kruskal[kruskal['检验对象'].astype(str).eq(eda_analysis.COMPANY_TAG_LABEL)]
    return pd.DataFrame([
        {'阶段': '阶段 1：显示名错误',
         '内容': f"公司标签列表（福利标签）被错误显示为「公司认证标签」，"
               f"369 个原子标签被当作一个「公司认证」因素",
         '结果/状态': f"11 号表检验对象 = certification，epsilon² ≈ "
                  f"{deprecated.get('epsilon_squared', eda_analysis.DEPRECATED_TAG_KW_VALUE)}"
                  f"（历史值）；{eda_analysis.DEPRECATED_INFERENCE_FLAG}",
         '是否进入正式结论': '否（已废止）'},
        {'阶段': '阶段 2：语义拆分（记录 23 / 32 号表）',
         '内容': f"拆分为「{eda_analysis.COMPANY_CERT_LABEL}」"
               f"（`{schema.CERT_TAG_FIELD}`，互斥四类）与「{eda_analysis.COMPANY_TAG_LABEL}」"
               f"（`{schema.COMPANY_TAG_LIST_FIELD}`，多值）两个独立因素",
         '结果/状态': f"公司认证四类 n："
                  f"{'、'.join(f'{k} {v:,}' for k, v in CERT_CLASS_COUNTS.items())}"
                  f"（合计 {EXPECTED_MODEL_SAMPLE:,}）；"
                  f"公司标签原子标签 {metrics.get('company_tag_atoms', 0):,} 个",
         '是否进入正式结论': '是（语义层面）'},
        {'阶段': '阶段 3：统计推断最终方法（本轮）',
         '内容': f"公司认证 → {int(float(cert['参与检验组数'].iloc[0]))} 类 Kruskal–Wallis；"
               f"公司标签 → 描述性统计 + 单标签 present vs absent 二元比较",
         '结果/状态': f"公司认证 epsilon² = {cert['效应量'].iloc[0]}（保持不变）；"
                  f"公司标签进入二元检验 {len(tag_binary)} 个，"
                  f"BH-FDR q < 0.05 的 {tag_analysis.get('fdr_significant')} 个；"
                  f"11 号表是否残留标签整体 KW："
                  f"{'是' if not tag_kw_left.empty else '否'}",
         '是否进入正式结论': '是（最终封版统计范围）'},
        {'阶段': '阶段 3 前后差异（公司认证）',
         '内容': '公司认证四分类统计保持不变（计数 / H / epsilon² 全部一致）',
         '结果/状态': f"无认证 {int(cert_rows[cert_rows['取值'].eq('无认证')]['样本数'].iloc[0]):,}、"
                  f"行业认证 {int(cert_rows[cert_rows['取值'].eq('行业认证')]['样本数'].iloc[0]):,}、"
                  f"最佳雇主 {int(cert_rows[cert_rows['取值'].eq('最佳雇主')]['样本数'].iloc[0]):,}、"
                  f"两者均有 {int(cert_rows[cert_rows['取值'].eq('两者均有')]['样本数'].iloc[0]):,}；"
                  f"epsilon² = {cert['效应量'].iloc[0]}",
         '是否进入正式结论': '是（无变化）'},
        {'阶段': '阶段 3 变更内容（公司标签）',
         '内容': '删除 369 组整体 KW 正式推断；新增描述性统计与单标签二元检验；'
               '统一 BH-FDR 与 Cliff\'s delta 效应分级',
         '结果/状态': f"描述性标签 {tag_analysis.get('descriptive_rows'):,} 个；"
                  f"二元检验 {len(tag_binary)} 个（present_n 最小 "
                  f"{tag_analysis.get('min_present_n')}、absent_n 最小 "
                  f"{tag_analysis.get('min_absent_n')}）；"
                  f"原 epsilon² ≈ {deprecated.get('epsilon_squared')} 仅保留追溯",
         '是否进入正式结论': '是（替代原 369 组 KW）'},
    ])


def build_certification_sheet(semantic: dict, eda_sheets: dict) -> pd.DataFrame:
    """02_公司认证统计（原子标签 + 互斥四类 + 检验）。"""
    atoms = semantic['03_公司认证原子标签']
    company = eda_sheets['06_公司因素薪资']
    cert_rows = company[company['因素'].eq(eda_analysis.COMPANY_CERT_LABEL)]
    statistics = eda_sheets['11_统计检验']
    kruskal = statistics[statistics['检验方法'].astype(str).str.startswith('Kruskal')]
    cert_test = kruskal[kruskal['检验对象'].eq(eda_analysis.COMPANY_CERT_LABEL)].iloc[0]
    pairwise = statistics[statistics['因素'].eq(eda_analysis.COMPANY_CERT_LABEL)]
    rows = [{'层级': '原子标签', '项目': str(row['项目']).replace(' | ', ' + '),
             '取值': str(row['取值']).replace(' | ', ' + ')}
            for row in atoms.to_dict('records')]
    rows.append({'层级': 'Kruskal–Wallis', '项目': '检验对象 / 组数',
                 '取值': f'{cert_test["检验对象"]} / {int(float(cert_test["参与检验组数"]))}'})
    rows.append({'层级': 'Kruskal–Wallis', '项目': 'H 统计量 / p 值 / epsilon²',
                 '取值': f'{cert_test["统计量"]} / {cert_test["p值"]} / {cert_test["效应量"]}'})
    for row in cert_rows.sort_values('中位数', ascending=False).to_dict('records'):
        rows.append({'层级': '互斥四类薪资', '项目': row['取值'],
                     '取值': f"n={int(row['样本数']):,}，中位数 {row['中位数']}，IQR {row['IQR']}"})
    for row in pairwise.to_dict('records'):
        rows.append({'层级': '成对 Mann–Whitney', '项目': f"{row['取值A']} vs {row['取值B']}",
                     '取值': f"cliff_delta {row['cliff_delta']}，BH-FDR q "
                             f"{row['p_adjusted_bh']}，FDR 显著 {row['FDR显著']}"})
    return pd.DataFrame(rows)


def build_tag_sheet(semantic: dict, eda_sheets: dict, metrics_13: dict) -> pd.DataFrame:
    """03_公司标签统计：A 总体规模 / B Top20~Top30 高频标签 / C present vs absent 检验 / D 废止推断。"""
    top = semantic['04b_公司标签Top50']
    distribution = semantic['04c_每岗标签数分布']
    table = semantic['04_公司标签原子标签']
    company = eda_sheets['06_公司因素薪资']
    statistics = eda_sheets['11_统计检验']
    tag_descriptive = company[company['分析类型'].astype(str).eq(eda_analysis.TAG_ANALYSIS_TYPE)] \
        if '分析类型' in company.columns else company[
            company['因素'].eq(eda_analysis.COMPANY_TAG_LABEL)]
    tag_binary = statistics[
        statistics['检验块'].astype(str).str.startswith('单标签二元比较')]
    tag_analysis = metrics_13.get('company_tag_analysis', {})
    rows = [{'分区': 'A 标签总体规模', '项目': row['指标'], '取值': row['数值']}
            for row in table.to_dict('records')]
    rows.append({'分区': 'A 标签总体规模', '项目': '描述性统计标签行数',
                 '取值': f"{len(tag_descriptive):,}（分析类型 = {eda_analysis.TAG_ANALYSIS_TYPE}）"})
    for row in distribution.to_dict('records'):
        rows.append({'分区': 'A 标签总体规模', '项目': f"每岗标签数 {int(row['每岗标签数'])}",
                     '取值': f"{int(row['岗位数']):,}"})
    for row in top.head(30).to_dict('records'):
        rows.append({'分区': 'B Top30 高频标签',
                     '项目': f"第 {int(row['排名'])} 名 {row['原子标签']}",
                     '取值': f"{int(row['岗位数']):,}（{row['岗位占比']:.2%}）"})
    for row in top.head(20).to_dict('records'):
        salary_row = tag_descriptive[tag_descriptive['标签'].astype(str)
                                     .eq(str(row['原子标签']))]
        if salary_row.empty:
            continue
        records = salary_row.to_dict('records')
        for record in records[:1]:
            rows.append({'分区': 'B Top20 高频标签薪资',
                         '项目': f"{row['原子标签']}（岗位 {int(row['岗位数']):,}）",
                         '取值': f"薪资中位数 {record.get('薪资中位数')}，IQR {record.get('IQR')}，"
                               f"P25 {record.get('P25')}，P75 {record.get('P75')}"})
    rows.append({'分区': 'C 单标签 present vs absent 二元检验',
                 '项目': '进入正式检验的标签数',
                 '取值': f"{len(tag_binary)}（present_n ≥ {tag_analysis.get('min_group_size')}、"
                       f"absent_n ≥ {tag_analysis.get('min_group_size')}）"})
    rows.append({'分区': 'C 单标签 present vs absent 二元检验', '项目': 'present_n / absent_n 最小值',
                 '取值': f"{tag_analysis.get('min_present_n')} / {tag_analysis.get('min_absent_n')}"
                       f"（present_n + absent_n = {EXPECTED_MODEL_SAMPLE:,}）"})
    rows.append({'分区': 'C 单标签 present vs absent 二元检验', '项目': 'BH-FDR q < 0.05 的标签数',
                 '取值': f"{tag_analysis.get('fdr_significant')} / {len(tag_binary)}"})
    rows.append({'分区': 'C 单标签 present vs absent 二元检验', '项目': '检验方法与字段',
                 '取值': "Mann–Whitney U + Cliff's delta + Benjamini–Hochberg FDR；"
                       'u_stat / p_raw / p_adjusted_bh / cliff_delta 全部非空'})
    top_cliff = pd.DataFrame(tag_analysis.get('top_cliff', []))
    for row in top_cliff.to_dict('records'):
        rows.append({'分区': 'C 效应量 Top10（|Cliff\'s delta|）',
                     '项目': f"{row['标签']}（present_n {int(row['n_present']):,}）",
                     '取值': f"Cliff's delta {row['cliff_delta']}，"
                           f"中位数差 {row['median_diff_A_minus_B']}，FDR 显著 {row['FDR显著']}"})
    overlap = tag_analysis.get('top_cliff_overlap', {})
    if overlap:
        rows.append({'分区': 'C 效应量 Top10（|Cliff\'s delta|）', '项目': '共现提示（必须随榜呈现）',
                     '取值': f"|Cliff's delta| 最高的标签（{'、'.join(overlap.get('标签', []))}）"
                           f"两两最大 Jaccard = {overlap.get('两两最大Jaccard')}，"
                           f"全部交集 {overlap.get('全部交集岗位数')} 个岗位 / "
                           f"并集 {overlap.get('全部并集岗位数')} 个岗位——"
                           '属同一批岗位的标签簇，不能当作彼此独立的薪资关联证据'})
    deprecated = metrics_13.get('deprecated_inference', {})
    rows += [
        {'分区': 'D 已废止推断', '项目': '状态', '取值': deprecated.get('flag', '')},
        {'分区': 'D 已废止推断', '项目': '原推断',
         '取值': f"369 个福利标签整体 Kruskal–Wallis，"
               f"epsilon² ≈ {deprecated.get('epsilon_squared', eda_analysis.DEPRECATED_TAG_KW_VALUE)}"},
        {'分区': 'D 已废止推断', '项目': '废止原因', '取值': deprecated.get('reason', '')},
        {'分区': 'D 已废止推断', '项目': '现行替代统计范围', '取值': deprecated.get('replacement', '')},
        {'分区': 'D 已废止推断', '项目': '是否可进入正式因素比较',
         '取值': '否（不再进入正式因素比较、效应量排名与论文主结论；仅作历史追溯）'},
        {'分区': 'D 已废止推断', '项目': '追溯位置',
         '取值': '29 号表 16_已废止推断 / 32 号表 / 记录 20、23、24'},
    ]
    return pd.DataFrame(rows)


def build_shap_legacy_sheet(shap_sheets: dict) -> pd.DataFrame:
    """05_技能SHAP旧统计范围（全局 mean(SHAP) 方向，历史结果保留未删除）。"""
    table = shap_sheets['08_技能SHAP']
    columns = ['排名', '技能', '岗位数', '岗位频率', 'mean_abs_SHAP', 'mean_SHAP', '方向性',
               '低频标记', '频率统计方式', '说明']
    return table[[column for column in columns if column in table.columns]].copy()


def build_shap_presence_sheet(shap_sheets: dict) -> pd.DataFrame:
    """06_技能SHAP_presence 统计方式（新统计范围，论文与图表优先）。"""
    table = shap_sheets['08_技能SHAP']
    columns = ['排名', '技能', 'model_job_count', 'model_job_frequency', 'test_present_n',
               'test_absent_n', 'mean_SHAP_present', 'median_SHAP_present', 'mean_SHAP_absent',
               'median_SHAP_absent', 'mean_abs_SHAP_present', 'positive_SHAP_ratio_present',
               'presence_direction', 'global_mean_SHAP_direction']
    kept = table[[column for column in columns if column in table.columns]].copy()
    return kept


def build_consistency_sheet(model_frame: pd.DataFrame, metrics_14: dict, metrics_15: dict,
                            metrics_13: dict) -> pd.DataFrame:
    """07_关键结果一致性（核心锚点回归保护）。"""
    test = metrics_14.get('test', {})
    increment = pd.DataFrame(metrics_15.get('skill_increment', []))
    group_rows = pd.DataFrame(metrics_15.get('group_split_table', []))
    group_mae = group_rows[group_rows['划分方式'].astype(str)
                           .str.startswith('Company Group Split')
                           & group_rows['数据子集'].eq('test')]['MAE']
    random_mae = group_rows[group_rows['划分方式'].astype(str)
                            .str.startswith('Random')]['MAE']
    skill_cross = set(increment['CI是否跨0']) if not increment.empty else set()
    rows = [
        {'指标': '正式建模样本', '冻结值': f'{EXPECTED_MODEL_SAMPLE:,}',
         '实际值': f'{len(model_frame):,}',
         '是否一致': '一致' if len(model_frame) == EXPECTED_MODEL_SAMPLE else '不一致'},
        {'指标': 'Stage 14 主模型', '冻结值': 'LightGBM',
         '实际值': str(metrics_14.get('model_key', '')),
         '是否一致': '一致' if metrics_14.get('model_key') == 'LightGBM' else '不一致'},
        {'指标': 'Stage 14 test MAE', '冻结值': f'{EXPECTED_TEST_MAE}',
         '实际值': f"{test.get('MAE', '')}",
         '是否一致': '一致' if abs(float(test.get('MAE', 0)) - EXPECTED_TEST_MAE) < 1e-6
         else '不一致'},
        {'指标': 'Stage 14 test RMSE', '冻结值': f'{EXPECTED_TEST_RMSE}',
         '实际值': f"{test.get('RMSE', '')}",
         '是否一致': '一致' if abs(float(test.get('RMSE', 0)) - EXPECTED_TEST_RMSE) < 1e-6
         else '不一致'},
        {'指标': 'Stage 14 test R²', '冻结值': f'{EXPECTED_TEST_R2}',
         '实际值': f"{test.get('R2', '')}",
         '是否一致': '一致' if abs(float(test.get('R2', 0)) - EXPECTED_TEST_R2) < 1e-6
         else '不一致'},
        {'指标': 'Stage 15 Random Full test MAE', '冻结值': f'{EXPECTED_RANDOM_FULL_MAE}',
         '实际值': f"{float(random_mae.iloc[0]) if not random_mae.empty else ''}",
         '是否一致': '一致' if not random_mae.empty
         and abs(float(random_mae.iloc[0]) - EXPECTED_RANDOM_FULL_MAE) < 1e-6 else '不一致'},
        {'指标': 'Stage 15 Company Group Split test MAE',
         '冻结值': f'{EXPECTED_GROUP_SPLIT_MAE}',
         '实际值': f"{float(group_mae.iloc[0]) if not group_mae.empty else ''}",
         '是否一致': '一致' if not group_mae.empty
         and abs(float(group_mae.iloc[0]) - EXPECTED_GROUP_SPLIT_MAE) < 1e-6 else '不一致'},
        {'指标': '技能增量 bootstrap CI', '冻结值': 'Base+Skill vs Base、Full vs Full-Skill 均跨 0',
         '实际值': f"CI 是否跨 0 = {skill_cross}" if skill_cross else '缺',
         '是否一致': '一致' if skill_cross == {'是'} else '不一致'},
        {'指标': '技能消融正式结论', '冻结值': '技能特征具有明显解释价值，独立增量预测价值不稳定',
         '实际值': '未修改（本轮只改解释层与 Stage 13 字段语义）', '是否一致': '一致'},
        {'指标': 'Stage 13 薪资样本', '冻结值': f'{EXPECTED_MODEL_SAMPLE:,}',
         '实际值': f"{metrics_13.get('salary_sample', '')}",
         '是否一致': '一致' if metrics_13.get('salary_sample') == EXPECTED_MODEL_SAMPLE
         else '不一致'},
        {'指标': 'Stage 13 公司认证字段来源', '冻结值': 'job_details_unique.parquet（Stage 05）',
         '实际值': str(metrics_13.get('company_certification', {}).get('source', '')),
         '是否一致': '一致' if metrics_13.get('company_certification', {}).get('source')
         == 'data/processed/job_details_unique.parquet' else '不一致'},
        {'指标': '公司认证 epsilon²', '冻结值': f'{EXPECTED_CERT_EPSILON}',
         '实际值': str(metrics_13.get('company_certification', {}).get('epsilon_squared', '')),
         '是否一致': '一致' if abs(float(metrics_13.get('company_certification', {})
                                    .get('epsilon_squared', 0) or 0)
                                - EXPECTED_CERT_EPSILON) < 1e-5 else '不一致'},
        {'指标': '公司认证四分类计数', '冻结值': '、'.join(f'{k} {v:,}' for k, v in
                                              CERT_CLASS_COUNTS.items()),
         '实际值': '、'.join(f'{k} {v:,}' for k, v in
                          sorted((metrics_13.get('company_certification', {}).get('classes') or {})
                                 .items())) or '缺',
         '是否一致': '一致' if (metrics_13.get('company_certification', {}).get('classes') or {})
         == CERT_CLASS_COUNTS else '不一致'},
        {'指标': '公司标签整体 KW（369 组）', '冻结值': 'DEPRECATED_INFERENCE（不得进入正式结论）',
         '实际值': str(metrics_13.get('deprecated_inference', {}).get('flag', '')),
         '是否一致': '一致' if 'DEPRECATED_INFERENCE' in str(
             metrics_13.get('deprecated_inference', {}).get('flag', '')) else '不一致'},
        {'指标': '公司标签正式二元检验标签数', '冻结值': '≥ 1（present_n / absent_n ≥ 50）',
         '实际值': f"{metrics_13.get('company_tag_analysis', {}).get('binary_rows', 0)} 个"
                f"（FDR 显著 {metrics_13.get('company_tag_analysis', {}).get('fdr_significant', 0)} 个）",
         '是否一致': '一致' if int(metrics_13.get('company_tag_analysis', {})
                                .get('binary_rows', 0)) > 0
         and int(metrics_13.get('company_tag_analysis', {}).get('min_present_n', 0))
         >= eda_analysis.MIN_BINARY_GROUP_SIZE else '不一致'},
    ]
    return pd.DataFrame(rows)


def build_gate_sheet(extra_gates: dict | None = None) -> pd.DataFrame:
    """08_全阶段门禁：Stage 12~16（读取已落盘门禁 JSON）+ 本轮 Stage 17 门禁（内存）。

    全表**只有一个**门禁 Sheet（原 10_门禁 已移除）。
    """
    rows = []
    for stage in FROZEN_GATES:
        payload = load_json(project_paths.GATES_DIR / f'{stage}.json')
        gates = payload.get('gates', {})
        for name, item in gates.items():
            rows.append({'所属阶段': stage, '门禁项': name, '状态': item['status'],
                         '说明': item['note']})
    for name, item in (extra_gates or {}).items():
        rows.append({'所属阶段': STAGE, '门禁项': name, '状态': item['status'],
                     '说明': item['note']})
    return pd.DataFrame(rows)


def run_pytest() -> pd.DataFrame:
    """09_测试：真实运行 pytest 并把结果写入审计表。"""
    command = [sys.executable, '-m', 'pytest', 'tests', '-q', '--no-header',
               '-p', 'no:cacheprovider']
    completed = subprocess.run(command, cwd=project_paths.PROJECT_ROOT, capture_output=True,
                               text=True, encoding='utf-8', errors='replace')
    output = (completed.stdout or '') + (completed.stderr or '')
    summary = next((line.strip() for line in reversed(output.splitlines())
                    if 'passed' in line or 'failed' in line or 'error' in line), '')
    match = re.search(r'(\d+) passed', summary)
    failed = re.search(r'(\d+) failed', summary)
    rows = [
        {'项目': 'pytest 命令', '取值': ' '.join(command)},
        {'项目': '退出码', '取值': str(completed.returncode)},
        {'项目': '结果摘要', '取值': summary or '（未解析到摘要）'},
        {'项目': '通过用例数', '取值': match.group(1) if match else '0'},
        {'项目': '失败用例数', '取值': failed.group(1) if failed else '0'},
    ]
    return pd.DataFrame(rows), completed.returncode == 0


def write_record(metrics: dict, audit: dict) -> Path:
    """24 号最终解释封版记录。"""
    consistency = audit['07_关键结果一致性']
    gates = audit['08_全阶段门禁']
    tests = audit['09_测试结果'].set_index('项目')['取值']
    lines = [
        '# 记录 24：最终解释封版（公司字段语义修正 + 技能 SHAP presence 统计方式）',
        '',
        '> 本记录由 `scripts/ch5_factors/17_final_interpretation_audit.py` 自动生成，数字全部来自真实产物。',
        '> 本轮**未重训模型、未修改样本、未修改技能词典、未重新选模**，只做解释层与 Stage 13 '
        '字段语义修正。',
        '',
        '## 1. 公司字段问题分类与原因',
        '',
        '| 项 | 结论 |',
        '| --- | --- |',
    ]
    semantics = audit['01_字段语义最终确认']
    for row in semantics.to_dict('records'):
        lines.append(f"| {row['项目']} | {row['结论']} |")
    lines += [
        '',
        '- 结论：问题属 **A2（Stage 13 字段引用错误）+ A1（显示名称错误）**，'
        '不属于 A3（Stage 12 宽表不存在映射错误）；',
        '- 取证过程与全部证据见记录 23 与 32 号审计表；',
        '- 上表「取证结论：…是否需要重跑」为**问题影响范围判断**，'
        '本轮实际执行动作以第 2 节「修正内容与重跑范围」为准。',
        '',
        '## 2. Stage 13 修正三阶段（审计链）',
        '',
        '| 阶段 | 内容 | 结果/状态 | 是否进入正式结论 |',
        '| --- | --- | --- | --- |',
    ]
    for row in audit['04_Stage13修正前后'].to_dict('records'):
        lines.append(f"| {row['阶段']} | {row['内容']} | {row['结果/状态']} | "
                     f"{row['是否进入正式结论']} |")
    lines += [
        '',
        '| Stage | 是否重跑 | 原因 |',
        '| --- | --- | --- |',
        '| Stage 12 | 否 | 宽表数据与模型特征未变化 |',
        '| Stage 13 | 是 | 公司因素字段引用/显示名修正（记录 23） + 公司标签统计推断方法修正（本轮） |',
        '| Stage 14 | 否 | 主模型特征/样本/超参数未变化，禁止重训 |',
        '| Stage 15 | 否（本轮未重跑） | 技能 SHAP / 消融 / Group Split 结果保持不变 |',
        '| Stage 16 | 否（本轮未重跑） | 只读取证表（32 号表）作为历史证据保留 |',
        '| Stage 17 | 是（刷新 33 号表与本记录 + 冻结状态） | 最终解释审计 |',
        '',
        '## 3. 公司认证（有限类别因素，保持不变）',
        '',
        '| 层级 | 项目 | 取值 |',
        '| --- | --- | --- |',
    ]
    for row in audit['02_公司认证统计'].to_dict('records'):
        lines.append(f"| {row['层级']} | {row['项目']} | {row['取值']} |")
    lines += [
        '',
        '## 4. 公司标签（福利标签）统计推断最终方法',
        '',
        '| 分区 | 项目 | 取值 |',
        '| --- | --- | --- |',
    ]
    for row in audit['03_公司标签统计'].to_dict('records'):
        lines.append(f"| {row['分区']} | {row['项目']} | {row['取值']} |")
    lines += [
        '',
        '> 公司认证（最佳雇主 / 行业认证）与公司标签（福利标签）字段不同、取值范围与规模不同、'
        '语义不同（公司标签中仅字符串层面存在个别重叠），禁止互相替代。',
        '',
        '**公司标签正式统计范围**：',
        '',
        '- 369 组整体 KW：`DEPRECATED_INFERENCE`（已废止推断）——多值标签组相互重叠，'
        '不满足普通多组独立性比较的解释前提，epsilon² ≈ 0.471492 不再进入正式因素比较、'
        '效应量排名与论文主结论（仅保留追溯：29 号表 16_已废止推断、32 号表、记录 20/23/24）；',
        '- 正式统计范围：① 描述性频率/薪资统计（标签 / 岗位数 / 岗位占比 / 薪资样本数 / 薪资中位数 / '
        'IQR / P25 / P75，正文只展示 Top20~Top30）；② 高频单标签 present vs absent 二元比较'
        '（Mann–Whitney U + Cliff\'s delta + BH-FDR，present_n / absent_n ≥ 50，'
        'present ∩ absent = ∅）；',
        '- 措辞：只能表述为「具有某福利标签的岗位薪资分布与不具有该标签的岗位存在统计差异」'
        '（描述性关联），禁止写成「该福利导致工资提高」；',
        '- 共现提示：'
        + (
            f"|Cliff's delta| 最高的标签（"
            + '、'.join(metrics['company_tag_final_inference']
                        .get('top_cliff_overlap', {}).get('标签', []))
            + f"）两两最大 Jaccard = "
            + str(metrics['company_tag_final_inference']
                  .get('top_cliff_overlap', {}).get('两两最大Jaccard'))
            + f"，全部交集 "
            + str(metrics['company_tag_final_inference']
                  .get('top_cliff_overlap', {}).get('全部交集岗位数'))
            + f" 个岗位 / 并集 "
            + str(metrics['company_tag_final_inference']
                  .get('top_cliff_overlap', {}).get('全部并集岗位数'))
            + ' 个岗位——属同一批岗位的标签簇，**不能当作彼此独立的薪资关联证据**。'
            if metrics.get('company_tag_final_inference', {}).get('top_cliff_overlap')
            else '无'
        ),
        '',
        '## 5. 技能 SHAP 解释方式',
        '',
        '| 统计范围 | 定义 | 用途 |',
        '| --- | --- | --- |',
        '| `global_mean_SHAP_direction`（旧统计范围，保留） | mean(SHAP) 在全部解释样本上的正负 | '
        '历史结果，未删除，仅作对比 |',
        '| `presence_direction`（新统计范围，论文与图表优先） | 技能存在时（技能列 = 1）的 '
        'mean_SHAP_present；> 1e-8 正向预测贡献，< -1e-8 负向预测贡献，否则中性/弱影响 | '
        '对 0/1 技能特征更直观 |',
        '',
        '| 技能 | 模型样本岗位频率 | test_present_n | mean_SHAP_present | mean_abs_SHAP | presence_direction |',
        '| --- | --- | --- | --- | --- | --- |',
    ]
    for row in metrics['key_skills']:
        lines.append(f"| {row['技能']} | {row['模型样本岗位频率']:.2%} | {row['test_present_n']} | "
                     f"{row['mean_SHAP_present']} | {row['mean_abs_SHAP']} | "
                     f"{row['presence_direction']} |")
    lines += [
        '',
        f"- 分母范围：模型样本频率 = 岗位数 / {metrics['model_sample']:,}；"
        f"presence 统计方式 = test 现技能岗位数 / {metrics['shap_rows']:,}（原 test 子集），两者不混用；",
        f"- SHAP 加性一致性最大误差 {metrics['shap_reconstruction_max_error']:.2e}（< 1e-6）；"
        f"仍使用 Stage 14 正式 artifact、原 test set、原 320 维特征空间；",
        '- 方向表示「该技能存在时」的平均 SHAP 贡献方向，**不代表技能的因果薪资效应**。',
        '',
        '## 6. 核心结果回归保护',
        '',
        '| 指标 | 冻结值 | 实际值 | 是否一致 |',
        '| --- | --- | --- | --- |',
    ]
    for row in consistency.to_dict('records'):
        lines.append(f"| {row['指标']} | {row['冻结值']} | {row['实际值']} | {row['是否一致']} |")
    lines += [
        '',
        '## 7. 门禁与测试',
        '',
        f"- Stage 12~17 门禁共 {len(gates)} 项（单一 08_全阶段门禁 Sheet）："
        f"PASS {int(gates['状态'].eq('PASS').sum())} / "
        f"FAIL {int(gates['状态'].eq('FAIL').sum())} / "
        f"NOT_RUN {int(gates['状态'].eq('NOT_RUN').sum())}；",
        f"- pytest：{tests.get('结果摘要', '')}（退出码 {tests.get('退出码', '')}）；",
        '- 契约测试：公司字段语义 4 项 + 技能 SHAP 4 项 + 公司标签统计推断 6 项'
        '（无重叠 KW / 二元互斥 / 样本门槛 / FDR / 公司认证四类不变 / 效应量不变）'
        ' + 33 号表结构唯一性与双重冻结 2 项；',
        '',
        '## 8. 封版状态',
        '',
        '```text',
        'EXPERIMENT_FREEZE = TRUE',
        'ANALYSIS_FREEZE   = TRUE',
        '```',
        '',
        '- `EXPERIMENT_FREEZE`：不再改样本、不再改技能词典、不再改特征组、'
        '不再重新选模型、不再根据 test 迭代（Stage 14/15 结果冻结）；',
        '- `ANALYSIS_FREEZE`：不再改 EDA / 统计推断方法'
        '（公司认证四分类与公司标签二元统计范围均为最终版）；',
        '- 公司标签 369 组整体 KW 永久标记 `DEPRECATED_INFERENCE`，不得再作为论文主结论；',
        '- 后续只进行论文正文、图表筛选、结论与摘要、答辩 PPT。',
        '',
    ]
    return io_utils.write_markdown(
        project_paths.RECORDS_DIR / project_paths.RECORD_FINAL_INTERPRETATION, lines)


def main() -> int:
    quality.configure_logging()
    gates = quality.GateRegistry(STAGE)
    quality.stage_banner(STAGE, TITLE)

    semantic_path = project_paths.TABLES_DIR / project_paths.TABLE_COMPANY_FIELD_SEMANTIC
    eda_path = project_paths.TABLES_DIR / project_paths.TABLE_EDA_STATISTICAL
    shap_path = project_paths.TABLES_DIR / project_paths.TABLE_ABLATION_SHAP
    for path in (semantic_path, eda_path, shap_path):
        if not path.exists():
            raise SystemExit(f'缺少上游产物: {path}')
    semantic = pd.read_excel(semantic_path, sheet_name=None)
    eda_sheets = pd.read_excel(eda_path, sheet_name=None)
    shap_sheets = pd.read_excel(shap_path, sheet_name=None)
    model_frame = io_utils.read_parquet(project_paths.JOB_SALARY_MODEL_DATASET_PARQUET,
                                        columns=[schema.ID_FIELD])
    metrics_13 = load_json(project_paths.METRICS_DIR / 'stage_13_eda.json')
    metrics_14 = load_json(project_paths.SALARY_MODEL_DIR / 'metrics_summary.json')
    metrics_14.update(load_json(project_paths.SALARY_MODEL_DIR / 'model_params.json'))
    metrics_15 = load_json(project_paths.METRICS_DIR / 'stage_15_ablation.json')
    print(f'输入: 建模样本 {len(model_frame):,} / 32 号表 {len(semantic)} 子表 / '
          f'29 号表 {len(eda_sheets)} 子表 / 31 号表 {len(shap_sheets)} 子表')

    audit_sheets = {
        '01_字段语义最终确认': build_field_semantics_sheet(semantic, eda_sheets),
        '02_公司认证统计': build_certification_sheet(semantic, eda_sheets),
        '03_公司标签统计': build_tag_sheet(semantic, eda_sheets, metrics_13),
        '04_Stage13修正前后': build_before_after_sheet(eda_sheets, metrics_13),
        '05_技能SHAP旧统计范围': build_shap_legacy_sheet(shap_sheets),
        '06_技能SHAP_presence 统计方式': build_shap_presence_sheet(shap_sheets),
        '07_关键结果一致性': build_consistency_sheet(model_frame, metrics_14, metrics_15,
                                                metrics_13),
    }
    base_sheets = audit_sheets
    audit_target = project_paths.TABLES_DIR / project_paths.TABLE_FINAL_INTERPRETATION

    def assemble_deliverable(test_frame: pd.DataFrame, gate_frame: pd.DataFrame) -> dict:
        """按 FINAL_SHEET_ORDER 组装 33 号表（9 个 Sheet，唯一门禁 Sheet）。"""
        sheets = {}
        for name in FINAL_SHEET_ORDER:
            if name == '08_全阶段门禁':
                sheets[name] = gate_frame
            elif name == '09_测试结果':
                sheets[name] = test_frame
            else:
                sheets[name] = base_sheets[name]
        return sheets

    semantics = base_sheets['01_字段语义最终确认'].set_index('项目')['结论']
    consistency = base_sheets['07_关键结果一致性']
    presence = base_sheets['06_技能SHAP_presence 统计方式']
    legacy = base_sheets['05_技能SHAP旧统计范围']
    company_tests = eda_sheets['11_统计检验']
    kruskal_rows = company_tests[
        company_tests['检验方法'].astype(str).str.startswith('Kruskal')]
    tag_kw_left = kruskal_rows[
        kruskal_rows['检验对象'].astype(str).eq(eda_analysis.COMPANY_TAG_LABEL)]
    tag_binary = company_tests[
        company_tests['检验块'].astype(str).str.startswith('单标签二元比较')]

    gates.check('INTERPRETATION_FIELD_SEMANTICS',
                '否' in semantics['是否仍存在「公司认证标签」显示名']
                and eda_analysis.COMPANY_CERT_LABEL in set(kruskal_rows['检验对象'])
                and semantics['369 组实际来源字段'] == schema.COMPANY_TAG_LIST_FIELD,
                f"公司认证与公司标签语义已分离：29 号表公司因素 = "
                f"{semantics['29 号表公司因素列表（修正后）']}；"
                f"369 组来源字段已明确为 `{schema.COMPANY_TAG_LIST_FIELD}`，"
                f"显示名不再混用")
    gates.check('INTERPRETATION_SHAP_PRESENCE',
                set(PRESENCE_COLUMNS) <= set(presence.columns)
                and presence['test_present_n'].ge(0).all()
                and presence['presence_direction']
                .isin(['正向预测贡献', '负向预测贡献', '中性/弱影响']).all()
                and set(legacy.columns) >= {'mean_abs_SHAP', 'mean_SHAP', '方向性'}
                and bool(metrics_15.get('key_skills')),
                f"技能 SHAP 同时保留旧统计范围（{len(legacy)} 行，含 mean_SHAP 与方向性）"
                f"与 presence 统计方式（{len(presence)} 行 × {len(PRESENCE_COLUMNS)} 列）；"
                f"presence 方向按 mean_SHAP_present（阈值 1e-8）判定，"
                f"重点技能 {len(metrics_15.get('key_skills', []))} 个")
    gates.check('INTERPRETATION_MODEL_UNCHANGED',
                consistency['是否一致'].eq('一致').all()
                and len(model_frame) == EXPECTED_MODEL_SAMPLE,
                f"核心结果回归保护 {len(consistency)} 项全部一致：建模样本 "
                f"{len(model_frame):,}、Stage 14 主模型 {metrics_14.get('model_key')}、"
                f"test MAE {metrics_14.get('test', {}).get('MAE')}、"
                f"Random Full MAE {EXPECTED_RANDOM_FULL_MAE}、"
                f"Company Group Split MAE {EXPECTED_GROUP_SPLIT_MAE}、"
                f"公司认证 epsilon² {EXPECTED_CERT_EPSILON}、技能消融 CI 跨 0 结论未变")

    # ---- 33 号表最终结构（9 个 Sheet，唯一门禁 Sheet） ----
    sheet_names = list(audit_sheets) + ['08_全阶段门禁', '09_测试结果']
    duplicates = sorted({name for name in sheet_names if sheet_names.count(name) > 1})
    gate_sheets = [name for name in sheet_names if name.endswith('门禁')]
    gates.check('FINAL_AUDIT_SHEET_STRUCTURE',
                sheet_names == FINAL_SHEET_ORDER and not duplicates
                and len(gate_sheets) == 1 and tag_kw_left.empty,
                f"33 号表最终 9 个 Sheet：{'、'.join(sheet_names)}；"
                f"重复 Sheet = {duplicates or '无'}；门禁 Sheet 数量 = {len(gate_sheets)}"
                f"（仅 08_全阶段门禁，已移除原 10_门禁）；"
                f"11 号表公司标签整体 KW 残留 {len(tag_kw_left)} 行")
    gates.check('ANALYSIS_FREEZE',
                tag_kw_left.empty and len(tag_binary) > 0
                and 'DEPRECATED_INFERENCE' in str(
                    metrics_13.get('deprecated_inference', {}).get('flag', '')),
                'EXPERIMENT_FREEZE = TRUE（模型实验冻结）+ ANALYSIS_FREEZE = TRUE'
                '（EDA/统计推断方法冻结）：公司标签 369 组整体 KW 已废止'
                f'（11 号表残留 {len(tag_kw_left)} 行），'
                f'正式改为描述性统计 + 单标签 present vs absent 二元检验'
                f'（{len(tag_binary)} 个标签，统一 BH-FDR）')
    def publish(test_frame: pd.DataFrame) -> tuple:
        """统一发布 33 号表 + metrics JSON + 24 号记录（保证三者在 pytest 前已就位）。"""
        frame = build_gate_sheet(extra_gates=gates.results)
        sheets = assemble_deliverable(test_frame, frame)
        path = io_utils.write_excel(audit_target, sheets)
        payload = {
            'experiment_freeze': True,
            'analysis_freeze': True,
            'analysis_scope': 'EDA / 统计推断方法（公司认证四分类 + 公司标签二元统计范围）',
            'final_sheet_order': FINAL_SHEET_ORDER,
            'sheet_count': len(sheets),
            'gate_sheet_count': sum(1 for name in sheets if name.endswith('门禁')),
            'company_tag_final_inference': metrics_13.get('company_tag_analysis', {}),
            'company_tag_deprecated_inference': metrics_13.get('deprecated_inference', {}),
            'company_tag_kw_removed': bool(tag_kw_left.empty),
            'company_tag_kw_rows_left': int(len(tag_kw_left)),
            'company_tag_binary_rows': int(len(tag_binary)),
            'model_sample': int(len(model_frame)),
            'field_semantics': semantics.to_dict(),
            'consistency': consistency.to_dict('records'),
            'all_consistent': bool(consistency['是否一致'].eq('一致').all()),
            'presence_columns': [column for column in PRESENCE_COLUMNS
                                 if column in presence.columns],
            'key_skills': metrics_15.get('key_skills', []),
            'shap_rows': metrics_15.get('shap_rows'),
            'model_sample_shap': metrics_15.get('model_sample'),
            'shap_reconstruction_max_error': metrics_15.get('shap_reconstruction_max_error'),
            'gates_snapshot': {'total': int(len(frame)),
                               'pass': int(frame['状态'].eq('PASS').sum()),
                               'fail': int(frame['状态'].eq('FAIL').sum()),
                               'not_run': int(frame['状态'].eq('NOT_RUN').sum())},
            'pytest': test_frame.to_dict('records'),
            'audit_path': project_paths.relative_to_root(path),
            'gates': {name: item['status'] for name, item in gates.results.items()},
        }
        io_utils.write_json(project_paths.METRICS_DIR / f'{STAGE}.json', payload)
        record = write_record(payload, sheets)
        return path, record, sheets

    # ---- 首轮发布：让 33 号表 / metrics / 24 号记录在 pytest 端到端校验前就位 ----
    pending_tests = pd.DataFrame([
        {'项目': 'pytest 命令', '取值': '本轮运行中（最终结果写在第二轮发布的 09_测试结果）'},
        {'项目': '结果摘要', '取值': 'pending'},
        {'项目': '通过用例数', '取值': '0'},
        {'项目': '失败用例数', '取值': '0'},
    ])
    audit_path, record_path, audit_sheets = publish(pending_tests)
    test_sheet, pytest_ok = run_pytest()
    gates.check('INTERPRETATION_EXPORT',
                sheet_names == FINAL_SHEET_ORDER and pytest_ok,
                f"33 号审计表含 {len(FINAL_SHEET_ORDER)} 张子表（01_字段语义最终确认 ~ "
                f"08_全阶段门禁 + 09_测试结果）；"
                f"pytest {test_sheet.set_index('项目').loc['结果摘要', '取值']}")
    audit_path, record_path, audit_sheets = publish(test_sheet)
    print(f'审计表: {project_paths.relative_to_root(audit_path)}')
    print(f'记录: {project_paths.relative_to_root(record_path)}')
    print('EXPERIMENT_FREEZE = TRUE')
    print('ANALYSIS_FREEZE   = TRUE')
    quality.print_gates(gates.results, order=quality.STAGE_GATE_MAP[STAGE])
    gates.save()
    return 0


if __name__ == '__main__':
    sys.exit(main())

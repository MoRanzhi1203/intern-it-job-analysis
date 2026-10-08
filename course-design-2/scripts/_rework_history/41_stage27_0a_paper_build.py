# -*- coding: utf-8 -*-
"""Stage27.0A 装配：截图证据原生化后的分章源 → Markdown → Word。

与 Stage27.0 的差别：
1. 每张工程证据图独立成图、独立图题；正文不再出现“子图 (a)/(b)”“代码部分(a)(b)(c)”；
2. 原有科研统计图与流程图全部保留，仅按新的插入顺序顺延图号；
3. 不新增 / 不删除表、公式、算法；不改动任何正文正式数值。

用法：
    E:\\anaconda3\\envs\\reptile\\python.exe scripts\\41_stage27_0a_paper_build.py
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC23 = ROOT / 'docs' / 'paper' / 'stage23'
SRC27A = ROOT / 'docs' / 'paper' / 'stage27_0a'
MD_OUT = ROOT / 'docs' / 'paper' / '课程设计论文_截图证据原生化修订稿_Stage27.0A.md'
DOCX_OUT = ROOT / 'outputs' / 'paper' / '课程设计论文_截图证据原生化修订版_Stage27.0A.docx'
FIGROOT = ROOT / 'outputs' / 'figures'
REFS = ROOT / 'docs' / 'paper' / '参考文献_Stage26.2.md'
REGISTRY35 = ROOT / 'outputs' / 'tables' / '35_visual_evidence_native_layout_registry.xlsx'


# 最终图号顺序（按实际插入顺序计算，不预设硬编码图题）
ORDER = [
    ('3-1', '实习僧平台“互联网IT”岗位菜单页面', 'evidence_native/01_web/E01A_shixiseng_it_menu.png'),
    ('3-2', '实习僧平台岗位详情页面', 'evidence_native/01_web/E01B_job_detail_page.png'),
    ('3-3', '数据采集总体流程', 'supplementary/图S16_数据采集总体流程.png'),
    ('3-4', '岗位详情链接规范化核心代码',
     'evidence_native/02_code/E02A_normalize_detail_url.png'),
    ('3-5', '规范化链接与岗位标识一致性校验核心代码',
     'evidence_native/02_code/E02B_url_identity_validation.png'),
    ('3-6', '岗位采集运行界面（双标签页协同）',
     'evidence_native/01_web/E03_dual_tab_runtime.png'),
    ('3-7', '原始岗位观测数据的岗位与薪资字段片段',
     'evidence_native/03_data/E04A_raw_job_salary_fields.png'),
    ('3-8', '原始岗位观测数据的业务时间与公司字段片段',
     'evidence_native/03_data/E04B_raw_business_time_company_fields.png'),
    ('3-9', '公司属性语义槽位异常识别核心代码',
     'evidence_native/02_code/E07A_slot_anomaly_detect_code.png'),
    ('3-10', '公司属性语义槽位异常修复前后数据片段',
     'evidence_native/03_data/E07B_slot_repair_before_after.png'),
    ('3-11', '薪资字段解析核心代码', 'evidence_native/02_code/E08A_parse_salary_code.png'),
    ('3-12', '薪资字段解析前后数据片段',
     'evidence_native/03_data/E08B_salary_parse_before_after.png'),
    ('3-13', '岗位描述薪资信息清理核心代码',
     'evidence_native/02_code/E09A_salary_cleaning_code.png'),
    ('3-14', '岗位描述要求段提取示例',
     'evidence_native/03_data/E09B_requirement_section_example.png'),
    ('3-15', '要求段技能规范化结果片段',
     'evidence_native/03_data/E09C_canonical_skill_result.png'),
    ('3-16', '最终建模数据集岗位与地域字段片段',
     'evidence_native/03_data/E10A_model_dataset_job_region_fields.png'),
    ('3-17', '最终建模数据集公司属性字段片段',
     'evidence_native/03_data/E10B1_model_dataset_company_fields.png'),
    ('3-18', '最终建模数据集技能与文本字段片段',
     'evidence_native/03_data/E10B2_model_dataset_skill_text_fields.png'),
    ('4-1', '正式薪资分析样本与岗位描述处理结果',
     'supplementary/图S69_正式薪资分析样本与岗位描述处理结果.png'),
    ('4-2', '主要岗位细分类薪资中位数及四分位区间',
     'supplementary/图S70_主要岗位细分类薪资中位数及四分位区间.png'),
    ('4-3', '互联网 IT 实习岗位样本的省域分布',
     'supplementary/图S66_互联网IT实习岗位样本的省域分布.png'),
    ('4-4', '薪资中点分布与经验累积分布（n = 14,883）',
     'supplementary/图S55_薪资中点分布与经验累积分布.png'),
    ('4-5', '招聘周期持续时长分布与累积分布（严格口径）',
     'supplementary/图S50_招聘周期持续时长分布与累积分布.png'),
    ('4-6', '活跃计划周期薪资中位数与四分位区间',
     'supplementary/图S71_活跃计划周期薪资中位数与四分位区间.png'),
    ('4-7', '发布时间队列的样本岗位数量变化', 'time/01_publish_cohort_count.png'),
    ('4-8', '发布时间队列的薪资中位数及四分位区间', 'time/02_salary_by_publish_time.png'),
    ('4-9', '核心技能需求的发布时间队列变化',
     'time/03_skill_or_category_time_structure.png'),
    ('5-1', 'Kruskal–Wallis 与 ε² 计算核心代码',
     'evidence_native/02_code/E11A_kruskal_wallis_code.png'),
    ('5-2', "Mann–Whitney U 与 Cliff's δ 计算核心代码",
     'evidence_native/02_code/E11B_mannwhitney_cliff_code.png'),
    ('5-3', 'BH-FDR 多重检验校正核心代码', 'evidence_native/02_code/E11C_bhfdr_code.png'),
    ('5-4', '城市、学历与公司规模的薪资中点中位数',
     'supplementary/图S59_城市学历与公司规模薪资中点中位数.png'),
    ('5-5', '公司认证状态的薪资分布与组间比较',
     'supplementary/图S60_公司认证状态的薪资分布与组间比较.png'),
    ('5-6', '高效应福利标签的共现结构', 'supplementary/图S62_高效应福利标签共现结构.png'),
    ('6-1', '核心技术技能需求 Top20（技能主口径，分母 8,822）', 'eda/05_tech_skill_top20.png'),
    ('6-2', '岗位细分类 × 技能命中率热力图',
     'supplementary/图S63_岗位细分类技能命中率热力图.png'),
    ('6-3', '控制岗位细分类前后的技能薪资差异变化',
     'supplementary/图S09_控制细分类前后技能薪资差异.png'),
    ('7-1', '薪资预测模型构建与评估流程',
     'supplementary/图S72_薪资预测模型构建与评估流程（节点简化）.png'),
    ('7-2', '训练集、验证集与测试集划分核心代码',
     'evidence_native/02_code/E12A_build_splits_code.png'),
    ('7-3', '训练集拟合与验证/测试集变换核心代码',
     'evidence_native/02_code/E12B_fit_transform_code.png'),
    ('7-4', '高频技能列与文本语义特征构建核心代码',
     'evidence_native/02_code/E13A_skill_text_feature_code.png'),
    ('7-5', 'A/B/C/D/E 特征矩阵拼接核心代码',
     'evidence_native/02_code/E13B_feature_matrix_stack_code.png'),
    ('7-6', '候选回归模型与超参数配置核心代码',
     'evidence_native/02_code/E14A_candidate_models_code.png'),
    ('8-1', '特征组消融实验的验证集与测试集 MAE',
     'supplementary/图S73_特征组消融实验的验证集与测试集MAE（横轴标签简化）.png'),
    ('8-2', '特征组消融配置核心代码', 'evidence_native/02_code/E15A_ablation_config_code.png'),
    ('8-3', 'Company Group Split 核心代码',
     'evidence_native/02_code/E15B_company_group_split_code.png'),
    ('8-4', '回顾性发布时间排序划分核心代码',
     'evidence_native/02_code/E15C_temporal_split_code.png'),
    ('8-5', 'TreeSHAP 贡献值提取核心代码',
     'evidence_native/02_code/E16A_treeshap_values_code.png'),
    ('8-6', '技能特征 SHAP 方向统计核心代码',
     'evidence_native/02_code/E16B_shap_skill_direction_code.png'),
    ('8-7', '正式主模型 SHAP 蜂群图（Top12 特征）',
     'supplementary/图S64_主模型SHAP蜂群图.png'),
    ('8-8', '技能特征 SHAP Top20（条长 = 平均绝对 SHAP 值，标记 = 技能存在时平均贡献方向）',
     'supplementary/图S68_技能特征SHAP贡献.png'),
]

# 默认打印宽度（cm），工程证据图按 Registry 的 print_width_cm 覆盖
BASE_WIDTH = {
    '4-1': 15.5, '4-3': 15.5, '4-5': 15.5, '4-6': 13.0, '4-7': 15.5, '4-8': 15.5, '4-9': 15.5,
    '5-5': 15.5, '5-6': 15.5, '6-2': 15.5, '8-1': 15.5, '8-7': 12.5, '8-8': 11.5,
}


def fig(no, title, src):
    return ['> 【插图：图 %s %s】' % (no, title), '',
            '<!-- source: outputs/figures/%s -->' % src, '']


def para(text):
    return [text, '']


def build_figure_manifest():
    fig_main = [(no, title, path) for no, title, path in ORDER]
    widths = dict(BASE_WIDTH)
    if REGISTRY35.exists():
        reg = pd.read_excel(REGISTRY35, sheet_name='01_native_layout_registry')
        cm = dict(zip(reg['figure_file'], reg['print_width_cm']))
        for no, title, path in ORDER:
            name = Path(path).name
            if name in cm:
                widths[no] = float(cm[name])
    return fig_main, widths


# --------------------------------------------------------------------------- #
# 插入内容：每张独立图 = 图前引导 + 图题 + 图后解释
# --------------------------------------------------------------------------- #
INS_3_1 = para('本文的岗位搜索项来自平台的互联网 IT 一级菜单。') + \
    fig('3-1', '实习僧平台“互联网IT”岗位菜单页面',
        'evidence_native/01_web/E01A_shixiseng_it_menu.png') + \
    para('实习僧平台的“互联网 IT”岗位菜单如图 3-1 所示：菜单展开后按岗位分组给出岗位名称，'
         '本文的岗位搜索项即由这些分组标题与岗位名称构成，采集范围与平台自身的分类口径一致，'
         '可枚举、可复现。') + \
    para('岗位各字段的来源可在岗位详情页上核对。') + \
    fig('3-2', '实习僧平台岗位详情页面',
        'evidence_native/01_web/E01B_job_detail_page.png') + \
    para('岗位详情页的字段布局如图 3-2 所示：薪资标注、工作城市、学历要求、发布时间、'
         '投递截止日期与职位描述均直接呈现在页面上，本文采集的字段与页面元素一一对应。')

INS_3_2 = para('上述可靠性机制中，详情链接的规范化是岗位唯一键得以成立的前提。') + \
    fig('3-4', '岗位详情链接规范化核心代码',
        'evidence_native/02_code/E02A_normalize_detail_url.png') + \
    para('岗位详情链接规范化的实现如图 3-4 所示：统一协议与域名的大小写、完整保留承载岗位'
         '身份的路径、删除页面片段与无业务含义的跟踪参数，因此同一详情页在多次抓取中只对应'
         '一个链接。') + \
    para('在此基础上还需校验规范化链接与岗位标识的一致性。') + \
    fig('3-5', '规范化链接与岗位标识一致性校验核心代码',
        'evidence_native/02_code/E02B_url_identity_validation.png') + \
    para('规范化链接与岗位标识一致性校验的实现如图 3-5 所示：以规范化链接与原岗位标识逐一'
         '配对计数，17,144 个实体与 17,144 个规范化链接严格一一对应，这是后文以岗位为分析'
         '单位的事实前提。') + \
    para('采集过程以单机多进程方式运行，每个进程持有独立的浏览器实例与任务队列。') + \
    fig('3-6', '岗位采集运行界面（双标签页协同）',
        'evidence_native/01_web/E03_dual_tab_runtime.png') + \
    para('岗位采集运行界面如图 3-6 所示，两个标签页同时处于岗位详情页抓取状态，说明多进程'
         '之间的任务分配与进度记录相互独立；页面的翻页与任务续跑由任务进度表与详情进度表'
         '共同支撑，因此重复抓取不会追加重复记录，采集中断后也能从未完成的页码继续执行。')

INS_3_3 = para('为说明原始观测的字段形态，下面两图给出原始岗位观测表中的真实记录片段。') + \
    fig('3-7', '原始岗位观测数据的岗位与薪资字段片段',
        'evidence_native/03_data/E04A_raw_job_salary_fields.png') + \
    para('原始岗位观测数据的岗位与薪资字段如图 3-7 所示：片段中的每条记录对应一次页面观测，'
         '同一岗位标识可重复出现；岗位名称、搜索分类、工作城市与薪资原文均为平台原始文本。') + \
    para('同一批记录的业务时间与公司字段如下。') + \
    fig('3-8', '原始岗位观测数据的业务时间与公司字段片段',
        'evidence_native/03_data/E04B_raw_business_time_company_fields.png') + \
    para('原始岗位观测数据的业务时间与公司字段如图 3-8 所示，与图 3-7 的记录顺序一致；'
         '两图均不含采集时间类字段，该部分字段只在追溯、审计与版本排序中使用，不进入正式'
         '分析数据集。')

INS_3_5 = para('该类异常的检测规则如下。') + \
    fig('3-9', '公司属性语义槽位异常识别核心代码',
        'evidence_native/02_code/E07A_slot_anomaly_detect_code.png') + \
    para('公司属性语义槽位异常识别的实现如图 3-9 所示：判定只依据公司性质栏位是否呈人数'
         '区间形态、公司规模栏位是否呈地点文本形态以及公司所在地栏位是否为空；按此规则筛出'
         '的候选异常共 467 条，其中规模与性质整体偏移 340 条、仅性质字段异常 111 条、'
         '仅规模字段异常 16 条，症状命中但公司所在地非空因而可修复的补充情形为 0 条。') + \
    para('修复采用确定性回填，真实记录的修复前后对照如下。') + \
    fig('3-10', '公司属性语义槽位异常修复前后数据片段',
        'evidence_native/03_data/E07B_slot_repair_before_after.png') + \
    para('公司属性语义槽位异常修复前后的数据片段如图 3-10 所示：可还原的槽位按异常类型回填，'
         '无法从本行还原的公司性质一律置为缺失，不做众数填充，也不按行业或规模推断；修复后'
         '重新扫描的残留异常为 0。')

INS_3_6 = para('薪资文本的解析规则如下。') + \
    fig('3-11', '薪资字段解析核心代码',
        'evidence_native/02_code/E08A_parse_salary_code.png') + \
    para('薪资字段解析的实现如图 3-11 所示：先判定薪资面议，再在删除单位后提取数值，只有'
         '一个数值时上下限取同一数值，随后执行上下界逻辑检查并计算中点。') + \
    para('四类真实取值的解析前后对照如下。') + \
    fig('3-12', '薪资字段解析前后数据片段',
        'evidence_native/03_data/E08B_salary_parse_before_after.png') + \
    para('薪资字段解析前后的数据片段如图 3-12 所示：区间形式得到下限、上限与中点，单值形式'
         '三列同值，面议形式三个数值列保持缺失，逻辑异常形式被单独标记；由于全部薪资文本均'
         '以天为单位，解析结果统一为元/天，不做任何单位折算，各处理环节的样本规模见表 3-2。')

INS_3_7 = para('岗位描述在用于建模前需要清除其中与目标直接相关的薪资信息。') + \
    fig('3-13', '岗位描述薪资信息清理核心代码',
        'evidence_native/02_code/E09A_salary_cleaning_code.png') + \
    para('岗位描述薪资信息清理的实现如图 3-13 所示：清理只删除薪资表达本身，技术版本号、'
         '实习月数与每周到岗天数等业务数字均保留。') + \
    para('要求段落的提取结果如下。') + \
    fig('3-14', '岗位描述要求段提取示例',
        'evidence_native/03_data/E09B_requirement_section_example.png') + \
    para('岗位描述要求段提取示例如图 3-14 所示：提取按固定段落标题定位任职要求、岗位要求与'
         '任职资格三类段落，命中时标记为 REQUIREMENT_SECTION，未命中时回退全文并标记为'
         ' FULL_TEXT_FALLBACK。') + \
    para('要求段口径下抽取的规范技能如下。') + \
    fig('3-15', '要求段技能规范化结果片段',
        'evidence_native/03_data/E09C_canonical_skill_result.png') + \
    para('要求段技能规范化结果如图 3-15 所示：技能结果按标准技能名、技能族与技能组三级给出，'
         '并记录该次命中所属的匹配口径，说明技能需求统计建立在要求段落优先的口径之上，'
         '而非对全文技能词等同处理。')

INS_3_8 = para('最终建模数据集的字段片段分三图给出，先给出岗位与地域字段。') + \
    fig('3-16', '最终建模数据集岗位与地域字段片段',
        'evidence_native/03_data/E10A_model_dataset_job_region_fields.png') + \
    para('最终建模数据集的岗位与地域字段如图 3-16 所示：目标变量薪资中点、岗位大类、工作城市、'
         '学历要求、实习时长与每周到岗要求均以岗位为一行。') + \
    para('公司侧字段如下。') + \
    fig('3-17', '最终建模数据集公司属性字段片段',
        'evidence_native/03_data/E10B1_model_dataset_company_fields.png') + \
    para('最终建模数据集的公司属性字段如图 3-17 所示，与图 3-16 的记录顺序一致：公司名称、'
         '公司规模、公司性质、所属行业与公司认证标签均以岗位为一行挂接。') + \
    para('技能指示列与文本长度如下。') + \
    fig('3-18', '最终建模数据集技能与文本字段片段',
        'evidence_native/03_data/E10B2_model_dataset_skill_text_fields.png') + \
    para('最终建模数据集的技能与文本字段如图 3-18 所示，仍与图 3-16 的记录顺序一致；'
         '三图只展示字段级别的建模宽表形态，编码后的高维矩阵不在此展开。正式建模样本为'
         ' 14,883 个岗位，经训练集内编码、技能列筛选及文本降维后形成正式模型输入。')

INS_5_1 = para('上述检验与校正由以下三段实现完成。') + \
    fig('5-1', 'Kruskal–Wallis 与 ε² 计算核心代码',
        'evidence_native/02_code/E11A_kruskal_wallis_code.png') + \
    para('Kruskal–Wallis 与 ε² 的实现如图 5-1 所示，用于工作城市、学历要求、实习时长要求、'
         '每周到岗要求、公司规模、公司性质、所属行业与公司认证等单值互斥因素。') + \
    fig('5-2', "Mann–Whitney U 与 Cliff's δ 计算核心代码",
        'evidence_native/02_code/E11B_mannwhitney_cliff_code.png') + \
    para('Mann–Whitney U 与 Cliff\'s δ 的实现如图 5-2 所示，用于岗位大类、岗位细分类、'
         '公司福利标签及技能等二元命中因素，每个取值构造命中该取值与未命中该取值两组后比较。') + \
    fig('5-3', 'BH-FDR 多重检验校正核心代码',
        'evidence_native/02_code/E11C_bhfdr_code.png') + \
    para('BH-FDR 多重检验校正的实现如图 5-3 所示，同一因素内的多组成对检验统一经该实现校正；'
         '本章与第 6 章涉及的城市、学历、行业、认证、技能等因素均通过这一套实现完成检验，'
         '检验族与样本量门槛保持一致。')

INS_7_2 = para('数据划分的实现如下。') + \
    fig('7-2', '训练集、验证集与测试集划分核心代码',
        'evidence_native/02_code/E12A_build_splits_code.png') + \
    para('训练集、验证集与测试集划分的实现如图 7-2 所示：划分按薪资十分位分箱分层、随机种子'
         '固定为 42，三个子集比例为 70%、15% 与 15%，对应 10,418、2,232 与 2,233 个岗位。') + \
    para('预处理的拟合与变换范围如下。') + \
    fig('7-3', '训练集拟合与验证/测试集变换核心代码',
        'evidence_native/02_code/E12B_fit_transform_code.png') + \
    para('训练集拟合与验证/测试集变换的实现如图 7-3 所示：预处理器只在训练集上拟合，随后'
         '分别变换训练集、验证集与测试集；测试集因此不参与任何编码、降维与技能列筛选，'
         '其取值分布不会影响正式模型输入。')

INS_7_3 = para('技能列筛选与文本语义降维的实现如下。') + \
    fig('7-4', '高频技能列与文本语义特征构建核心代码',
        'evidence_native/02_code/E13A_skill_text_feature_code.png') + \
    para('高频技能列与文本语义特征构建的实现如图 7-4 所示：技能列按训练集频次不低于 100'
         '筛选，文本语义向量在训练集上拟合截断奇异值分解并保留 16 维。') + \
    para('五组主体特征的拼接顺序如下。') + \
    fig('7-5', 'A/B/C/D/E 特征矩阵拼接核心代码',
        'evidence_native/02_code/E13B_feature_matrix_stack_code.png') + \
    para('A/B/C/D/E 特征矩阵拼接的实现如图 7-5 所示：特征矩阵由数值块、类别指示块、多值'
         '指示块、技能指示块与文本语义块按固定顺序横向拼接，列名顺序与拼接顺序严格一致；'
         'A+B+C 编码后 216 维、D 组 56 维、E 组 16 维，合计 288 维，加入发布时间位置扩展'
         '特征后为 290 维。')

INS_7_4 = para('候选模型与超参数的构建方式如下。') + \
    fig('7-6', '候选回归模型与超参数配置核心代码',
        'evidence_native/02_code/E14A_candidate_models_code.png') + \
    para('候选回归模型与超参数配置如图 7-6 所示：四类真实模型与常数基线使用完全相同的输入'
         '矩阵，超参数在预先设定的小网格上遍历。') + \
    para('主模型的锁定过程如下。') + \
    ['> 【插图：验证集模型选择与主模型锁定核心代码（取消正式图号，仅作实现说明）】', '',
     '<!-- source: outputs/figures/evidence_native/02_code/E14B_model_selection_code.png -->', ''] + \
    para('各候选配置在训练集上拟合，并依据验证集 MAE 确定主模型及超参数；'
         '配置锁定后，在独立测试集上完成正式评估。')

INS_8_2 = para('消融配置的实现如下。') + \
    fig('8-2', '特征组消融配置核心代码',
        'evidence_native/02_code/E15A_ablation_config_code.png') + \
    para('特征组消融配置如图 8-2 所示：消融只改变参与建模的特征组，基础为 A+B+C，'
         '基础+技能为 A+B+C+D，基础+文本为 A+B+C+E，完整模型为 A+B+C+D+E。') + \
    para('按公司分组的划分实现如下。') + \
    fig('8-3', 'Company Group Split 核心代码',
        'evidence_native/02_code/E15B_company_group_split_code.png') + \
    para('Company Group Split 的实现如图 8-3 所示：划分以公司实体为单位构造互斥分组，'
         '保证同一公司的岗位只出现在一个子集中；该划分仅使用公司实体标识构造分组，'
         '不利用测试集结果进行模型选择、超参数调整或正式特征筛选。') + \
    para('回顾性时间划分的实现如下。') + \
    fig('8-4', '回顾性发布时间排序划分核心代码',
        'evidence_native/02_code/E15C_temporal_split_code.png') + \
    para('回顾性发布时间排序划分的实现如图 8-4 所示：划分按业务发布日期排序后按累计占比'
         '切割，同一天的岗位不拆分；三种设置共用同一模型族、同一超参数与同一预处理纪律，'
         '只改变特征组或划分方式。')

INS_8_4 = para('TreeSHAP 贡献值的提取方式如下。') + \
    fig('8-5', 'TreeSHAP 贡献值提取核心代码',
        'evidence_native/02_code/E16A_treeshap_values_code.png') + \
    para('TreeSHAP 贡献值提取的实现如图 8-5 所示：解释值直接取自模型的贡献输出，稀疏返回'
         '先做稠密化处理，基准值列被单独剥离。') + \
    para('技能特征的方向统计口径如下。') + \
    fig('8-6', '技能特征 SHAP 方向统计核心代码',
        'evidence_native/02_code/E16B_shap_skill_direction_code.png') + \
    para('技能特征 SHAP 方向统计的实现如图 8-6 所示：技能特征按技能列取值是否为 1 划分'
         '存在与不存在两组，分别计算平均绝对 SHAP 值与技能存在时的平均贡献，用于判定方向；'
         '该实现只用于解释已经锁定的模型结果，不重新训练模型，也不据此调整特征或超参数。')


REPLACEMENTS = {
    '03_数据获取与预处理.md': [
        ('数据采集总体流程见图 3-1。', '数据采集总体流程见图 3-3。'),
        ('> 【插图：图 3-1 数据采集总体流程】', '> 【插图：图 3-3 数据采集总体流程】'),
    ],
    '05_实习岗位薪资影响因素分析.md': [
        ('城市、学历与公司规模的薪资中点中位数见图 5-1：',
         '城市、学历与公司规模的薪资中点中位数见图 5-4：'),
        ('> 【插图：图 5-1 城市、学历与公司规模的薪资中点中位数】',
         '> 【插图：图 5-4 城市、学历与公司规模的薪资中点中位数】'),
        ('公司认证状态的薪资分布与组间比较见图 5-2。',
         '公司认证状态的薪资分布与组间比较见图 5-5。'),
        ('> 【插图：图 5-2 公司认证状态的薪资分布与组间比较】',
         '> 【插图：图 5-5 公司认证状态的薪资分布与组间比较】'),
        ('图 5-3 给出高效应标签的共现结构：', '图 5-6 给出高效应标签的共现结构：'),
        ('> 【插图：图 5-3 高效应福利标签的共现结构】',
         '> 【插图：图 5-6 高效应福利标签的共现结构】'),
    ],
    '08_模型稳健性与解释.md': [
        ('整体解释结果见图 8-2。', '整体解释结果见图 8-7。'),
        ('> 【插图：图 8-2 正式主模型 SHAP 蜂群图（Top12 特征）】',
         '> 【插图：图 8-7 正式主模型 SHAP 蜂群图（Top12 特征）】'),
        ('技能 SHAP Top20 见图 8-3。', '技能 SHAP Top20 见图 8-8。'),
        ('> 【插图：图 8-3 技能特征 SHAP Top20（条长 = 平均绝对 SHAP 值，'
         '标记 = 技能存在时平均贡献方向）】',
         '> 【插图：图 8-8 技能特征 SHAP Top20（条长 = 平均绝对 SHAP 值，'
         '标记 = 技能存在时平均贡献方向）】'),
    ],
}

INSERTIONS = {
    '03_数据获取与预处理.md': [
        ('不构成对全部实习岗位市场的推断。', INS_3_1),
        ('解析失败的页面单独记录、不写入正式字段并留档处理。', INS_3_2),
        ('但不能理解为岗位实际的招聘完成时间。', INS_3_3),
        ('说明异常来自源记录的语义槽位本身，与数据迁移过程无关。', INS_3_5),
        ('含上下限倒置与非正数两类，数量很小，直接剔除并留档。', INS_3_6),
        ('发布时点不可得的版本类特征、与保留特征高度冗余的连续数值特征、'
         '近常量特征与数据治理元数据不进入正式预测模型，其诊断与移出依据见 7.1 节。', INS_3_8),
    ],
    '05_实习岗位薪资影响因素分析.md': [
        ('组间差异不显著时，只说明在本文样本下未观察到稳定差异。', INS_5_1),
    ],
    '07_薪资预测模型构建与结果分析.md': [
        ('三种划分的口径与对照结果见 8.2 节。', INS_7_2),
        ('主模型选择与多种子结果一致。', INS_7_4),
    ],
    '08_模型稳健性与解释.md': [
        ('这一差异来自训练集构成变化，而非使用测试集信息调整特征。'
         '三类泛化场景的对照见表 8-3。', INS_8_2),
        ('本文的基准值为 192.87 元/天。', INS_8_4),
    ],
}

ALGO_ANCHORS = {
    '03_数据获取与预处理.md': ['10: 将别名映射为规范技能、去除重复技能，返回 K 与 source',
                           INS_3_7],
    '07_薪资预测模型构建与结果分析.md': ['9: 输出 MAE、RMSE、R²', INS_7_3],
}

CHAPTER_FILES = ['00_摘要与Abstract.md', '01_绪论.md', '02_相关理论与分析方法.md',
                 '03_数据获取与预处理.md', '04_互联网IT实习岗位特征分析.md',
                 '05_实习岗位薪资影响因素分析.md', '06_实习岗位技能需求分析.md',
                 '07_薪资预测模型构建与结果分析.md', '08_模型稳健性与解释.md',
                 '09_总结与展望.md']


def apply_file(name: str) -> list:
    text = (SRC23 / name).read_text(encoding='utf-8')
    for old, new in REPLACEMENTS.get(name, []):
        if text.count(old) != 1:
            raise AssertionError('%s 替换锚点命中 %d 次：%s' % (name, text.count(old), old[:40]))
        text = text.replace(old, new)
    for anchor, payload in INSERTIONS.get(name, []):
        if text.count(anchor) != 1:
            raise AssertionError('%s 插入锚点命中 %d 次：%s' % (name, text.count(anchor), anchor[:40]))
        text = text.replace(anchor, anchor + '\n\n' + '\n'.join(payload))
    algo = ALGO_ANCHORS.get(name)
    if algo:
        anchor, payload = algo
        if text.count(anchor) != 1:
            raise AssertionError('%s 算法锚点命中异常：%s' % (name, anchor[:40]))
        idx = text.index(anchor)
        close = text.index('```', idx)
        end = text.index('\n', close) + 1
        text = text[:end] + '\n' + '\n'.join(payload) + text[end:]
    return text.splitlines()


def heading_shift(line: str) -> str:
    if re.match(r'^#{2,4}\s+', line):
        return line[1:]
    return line


def build_sources() -> dict:
    SRC27A.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in CHAPTER_FILES:
        lines = apply_file(name)
        out[name] = lines
        (SRC27A / name).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return out


def build_markdown(sources: dict) -> Path:
    parts = [
        '<!-- 截图证据原生化修订稿（Stage27.0A）分章来源：docs/paper/stage27_0a/*.md -->',
        '<!-- 图片落位清单：outputs/tables/35_visual_evidence_native_layout_registry.xlsx -->',
        '<!-- 本轮只拆分工程证据截图并重排图号，未改动任何正式结果数值与科研统计图 -->',
        '', '',
        '# 基于实习僧平台的互联网 IT 实习岗位薪资影响因素分析与预测', '', '',
    ]
    for name in CHAPTER_FILES:
        for line in sources[name]:
            parts.append(heading_shift(line))
        parts.append('')
    parts.append('# 参考文献')
    parts.append('')
    refs = REFS.read_text(encoding='utf-8')
    block = refs.split('## 一、论文用文献表')[1].split('## 二、')[0]
    for line in block.splitlines():
        if re.match(r'^\[\d+\]', line.strip()):
            parts.append(line.strip())
            parts.append('')
    MD_OUT.write_text('\n'.join(parts) + '\n', encoding='utf-8')
    return MD_OUT


def _load_builder(name='stage24_builder_0a'):
    spec = importlib.util.spec_from_file_location(
        name, str(ROOT / 'scripts' / '24a_build_stage24_docx.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fig_main, widths = build_figure_manifest()
    mod.SRC23 = SRC27A
    mod.OUTDOCX = DOCX_OUT
    mod.FIGROOT = FIGROOT
    mod.FIG_MAIN = fig_main
    mod.FIG_WIDTH_CM = widths
    mod.REFS = REFS
    return mod


def main() -> int:
    sources = build_sources()
    print('分章源写出:', SRC27A)
    md = build_markdown(sources)
    print('Markdown 写出:', md)
    mod = _load_builder()
    builder = mod.build(DOCX_OUT)
    print('[构建计数] 正文图=%d 正文表=%d 公式=%d 算法=%d'
          % (builder.stats['fig_main'], builder.stats['tbl_main'],
             builder.stats['equations'], builder.stats['algorithms']))
    print('[缺失图片]', builder.missing_figs or '无')
    print('Word 写出:', DOCX_OUT, DOCX_OUT.stat().st_size, 'bytes')
    info = mod.self_check(DOCX_OUT)
    print('-' * 78)
    print('图题数:', len(info['fig_caps']), '| 表题数:', len(info['tbl_caps']))
    print('SEQ eq 域:', info['seq_fields'], '| 静态编号残留:', len(info['static_eq']))
    print('原生 OMML:', info['omml'], '| 算法用表:', info['algo_tbl'])
    print('禁用工程串:', {k: v for k, v in info['scan'].items() if v})
    print('旧值扫描:', {k: v for k, v in info['scan_old'].items() if v})
    for cap in info['fig_caps']:
        print('   ', cap)
    return 0


if __name__ == '__main__':
    sys.exit(main())

# -*- coding: utf-8 -*-
"""Stage27.0 工程证据增强稿装配：分章源 → 合并 Markdown → Word。

原则：
1. 以 Stage26.9A 基线分章源（docs/paper/stage23/*.md）为唯一文本来源，
   除“图号顺延”“插入工程证据图及其图前引导 / 图后解释”外不改动任何正文文字；
2. 不新增表格、公式、算法，不删除任何原有图、表、公式与算法；
3. 复用 scripts/24a_build_stage24_docx.py 的排版机制，保证公式 / 表格 / 算法与
   Stage26.9A 完全一致。

用法：
    E:\\anaconda3\\envs\\reptile\\python.exe scripts\\37_stage27_0_paper_build.py
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC23 = ROOT / 'docs' / 'paper' / 'stage23'
SRC27 = ROOT / 'docs' / 'paper' / 'stage27_0'
MD_OUT = ROOT / 'docs' / 'paper' / '课程设计论文_工程证据增强稿_Stage27.0.md'
DOCX_OUT = ROOT / 'outputs' / 'paper' / '课程设计论文_工程证据增强版_Stage27.0.docx'
FIGROOT = ROOT / 'outputs' / 'figures'
REFS = ROOT / 'docs' / 'paper' / '参考文献_Stage26.2.md'

BUILD_STAMP = 'Stage27.0 工程证据增强稿'


# --------------------------------------------------------------------------- #
# 证据图登记（顺序即最终排版顺序）
# --------------------------------------------------------------------------- #
EVIDENCE_FIGS = [
    ('3-1', '实习僧平台数据来源与岗位详情页面示例',
     'evidence/01_source/E01_source_pages.png', 15.5),
    ('3-3', '岗位详情链接规范化与重复抓取控制核心代码',
     'evidence/01_source/E02_url_dedup_code.png', 15.5),
    ('3-4', '岗位采集运行界面（双标签页协同）',
     'evidence/01_source/E03_crawl_runtime.png', 13.0),
    ('3-5', '原始岗位观测数据片段',
     'evidence/02_preprocess/E04_raw_observations.png', 15.5),
    ('3-7', '岗位观测压缩与唯一实体生成核心代码',
     'evidence/02_preprocess/E05_entity_code.png', 15.5),
    ('3-8', '同一岗位从多次观测到唯一实体的处理示例',
     'evidence/02_preprocess/E06_entity_example.png', 15.5),
    ('3-9', '公司属性语义槽位异常识别与修复示例',
     'evidence/02_preprocess/E07_company_slot_repair.png', 15.5),
    ('3-10', '薪资字段解析与正式目标构造示例',
     'evidence/02_preprocess/E08_salary_parse.png', 15.5),
    ('3-11', '岗位文本安全处理与技能抽取示例',
     'evidence/02_preprocess/E09_text_skill.png', 15.5),
    ('3-12', '最终建模数据集代表性字段片段',
     'evidence/02_preprocess/E10_model_dataset.png', 15.5),
    ('5-1', '分组薪资差异检验与多重校正核心代码',
     'evidence/03_analysis/E11_stats_test_code.png', 15.5),
    ('7-2', '训练验证测试划分与训练集拟合核心代码',
     'evidence/04_model/E12_split_code.png', 15.5),
    ('7-3', 'A/B/C/D/E 异构特征构建与拼接核心代码',
     'evidence/04_model/E13_feature_pipeline_code.png', 15.5),
    ('7-4', '候选回归模型训练与验证集选择核心代码',
     'evidence/04_model/E14_model_selection_code.png', 15.5),
    ('8-2', '特征组消融与替代泛化划分核心代码',
     'evidence/04_model/E15_ablation_generalization_code.png', 15.5),
    ('8-3', 'TreeSHAP 模型解释与技能方向统计核心代码',
     'evidence/05_explain/E16_shap_code.png', 15.5),
]

# 原有正式图（图号顺延后的新编号 → 标题 → 源文件），全部保留，不删不换
BASE_FIGS = [
    ('3-2', '数据采集总体流程', 'supplementary/图S16_数据采集总体流程.png', None),
    ('3-6', '数据治理与岗位版本重构流程', 'sci/图01_数据治理与岗位版本重构流程.png', None),
    ('4-1', '正式薪资分析样本与岗位描述处理结果',
     'supplementary/图S69_正式薪资分析样本与岗位描述处理结果.png', 15.5),
    ('4-2', '主要岗位细分类薪资中位数及四分位区间',
     'supplementary/图S70_主要岗位细分类薪资中位数及四分位区间.png', None),
    ('4-3', '互联网 IT 实习岗位样本的省域分布',
     'supplementary/图S66_互联网IT实习岗位样本的省域分布.png', 15.5),
    ('4-4', '薪资中点分布与经验累积分布（n = 14,883）',
     'supplementary/图S55_薪资中点分布与经验累积分布.png', None),
    ('4-5', '招聘周期持续时长分布与累积分布（严格口径）',
     'supplementary/图S50_招聘周期持续时长分布与累积分布.png', 15.5),
    ('4-6', '活跃计划周期薪资中位数与四分位区间',
     'supplementary/图S71_活跃计划周期薪资中位数与四分位区间.png', 13.0),
    ('4-7', '发布时间队列的样本岗位数量变化', 'time/01_publish_cohort_count.png', 15.5),
    ('4-8', '发布时间队列的薪资中位数及四分位区间',
     'time/02_salary_by_publish_time.png', 15.5),
    ('4-9', '核心技能需求的发布时间队列变化',
     'time/03_skill_or_category_time_structure.png', 15.5),
    ('5-2', '城市、学历与公司规模的薪资中点中位数',
     'supplementary/图S59_城市学历与公司规模薪资中点中位数.png', None),
    ('5-3', '公司认证状态的薪资分布与组间比较',
     'supplementary/图S60_公司认证状态的薪资分布与组间比较.png', 15.5),
    ('5-4', '高效应福利标签的共现结构',
     'supplementary/图S62_高效应福利标签共现结构.png', 15.5),
    ('6-1', '核心技术技能需求 Top20（技能主口径，分母 8,822）',
     'eda/05_tech_skill_top20.png', None),
    ('6-2', '岗位细分类 × 技能命中率热力图',
     'supplementary/图S63_岗位细分类技能命中率热力图.png', 15.5),
    ('6-3', '控制岗位细分类前后的技能薪资差异变化',
     'supplementary/图S09_控制细分类前后技能薪资差异.png', None),
    ('7-1', '薪资预测模型构建与评估流程',
     'supplementary/图S72_薪资预测模型构建与评估流程（节点简化）.png', None),
    ('8-1', '特征组消融实验的验证集与测试集 MAE',
     'supplementary/图S73_特征组消融实验的验证集与测试集MAE（横轴标签简化）.png', 15.5),
    ('8-4', '正式主模型 SHAP 蜂群图（Top12 特征）',
     'supplementary/图S64_主模型SHAP蜂群图.png', 12.5),
    ('8-5', '技能特征 SHAP Top20（条长 = 平均绝对 SHAP 值，标记 = 技能存在时平均贡献方向）',
     'supplementary/图S68_技能特征SHAP贡献.png', 11.5),
]


def fig_block(no, title, note):
    """生成插图标记 + 证据来源注释。"""
    return ['> 【插图：图 %s %s】' % (no, title), '', '<!-- source: %s -->' % note, '']


def para(text):
    return [text, '']


# --------------------------------------------------------------------------- #
# 插入内容（图前引导 1~2 句 + 图题 + 图后解释 1 段）
# --------------------------------------------------------------------------- #
INSERT_E01 = para('本文的岗位搜索项与岗位详情页均来自该菜单入口，平台菜单展开后的岗位分组'
                  '与岗位详情页的字段布局如下所示。') + \
    fig_block('3-1', '实习僧平台数据来源与岗位详情页面示例',
              '输出：outputs/figures/evidence/01_source/E01_source_pages.png'
              '（附件2 原图2.13 菜单展开页 + 原图2.34 职位详情页，已裁切去浏览器外框）') + \
    para('图中子图 (a) 为“互联网 IT”一级菜单展开后的岗位分组与岗位名称入口，子图 (b) 为'
         '岗位详情页面的字段布局：薪资标注、工作城市、学历要求、发布时间与职位描述均直接'
         '呈现在页面上，本文采集的字段与页面元素一一对应。该图说明本文的分析对象来自平台'
         '自身的分类口径，采集范围可枚举、可复现。')

INSERT_E02_E03 = para('上述可靠性机制中，详情链接规范化与去重控制是岗位唯一键得以成立的'
                      '前提，其当前实现如下。') + \
    fig_block('3-3', '岗位详情链接规范化与重复抓取控制核心代码',
              '输出：outputs/figures/evidence/01_source/E02_url_dedup_code.png'
              '（src/url_utils.py 19-41；scripts/03_build_job_observations.py 236-244）') + \
    para('代码显示，链接规范化统一协议与域名的大小写、完整保留承载岗位身份的路径、删除'
         '页面片段与无业务含义的跟踪参数，因此同一详情页在多次抓取中只对应一个链接；在此'
         '基础上以规范化链接与原岗位标识做一一对应校验，17,144 个实体与 17,144 个规范化'
         '链接严格配对，这是后文以岗位为分析单位的事实前提。') + \
    para('采集过程以单机多进程方式运行，每个进程持有独立的浏览器实例与任务队列。') + \
    fig_block('3-4', '岗位采集运行界面（双标签页协同）',
              '输出：outputs/figures/evidence/01_source/E03_crawl_runtime.png'
              '（附件2 原图2.25 双标签页运行界面图，浏览器地址栏色带已遮盖）') + \
    para('图中两个标签页同时处于岗位详情页抓取状态，说明多进程之间的任务分配与进度记录'
         '相互独立；页面的翻页与任务续跑由任务进度表与详情进度表共同支撑，因此重复抓取'
         '不会追加重复记录，采集中断后也能从未完成的页码继续执行。')

INSERT_E04 = para('为说明原始观测的字段形态，图 3-5 给出原始岗位观测表中的真实记录片段。') + \
    fig_block('3-5', '原始岗位观测数据片段',
              '输出：outputs/figures/evidence/02_preprocess/E04_raw_observations.png'
              '（data/raw/shixiseng_job_details.parquet 等距抽样 6 行，展示级匿名化）') + \
    para('片段中的每条记录对应一次页面观测，同一岗位标识可重复出现；岗位名称、搜索分类、'
         '工作城市、薪资原文、学历要求、实习时长、发布时间、投递截止日期与公司名称均为'
         '平台原始文本。表中不含采集时间类字段，该部分字段只在追溯、审计与版本排序中使用，'
         '不进入正式分析数据集。')

INSERT_E05 = para('实体层的压缩逻辑与一个真实岗位的处理过程如下。') + \
    fig_block('3-7', '岗位观测压缩与唯一实体生成核心代码',
              '输出：outputs/figures/evidence/02_preprocess/E05_entity_code.png'
              '（src/versioning.py 241-260；src/dedup.py 450-467）') + \
    para('代码部分给出核心业务签名与完整页面签名的构造方式、按观测顺序对连续相同状态签名'
         '做压缩的版本生成方式，以及唯一实体代表观测的选择顺序：版本优先，完整度其次，'
         '同级再按数据更新时间、数据创建时间与稳定原始顺序确定。')

INSERT_E06 = para('同一岗位从多次观测到唯一实体的完整处理示例如下。') + \
    fig_block('3-8', '同一岗位从多次观测到唯一实体的处理示例',
              '输出：outputs/figures/evidence/02_preprocess/E06_entity_example.png'
              '（真实冻结记录，岗位标识与公司名称仅在展示副本中匿名化）') + \
    para('示例为一个真实岗位从 7 次观测压缩为 2 个核心业务版本、再生成 1 个唯一实体的过程：'
         '该岗位的城市与公司全程未变，薪资由 100-200 元/天变为 200-300 元/天，平台发布时间'
         '随之更新，版本层保留这一变化，实体层取最终核心版本作为代表状态。同一岗位在整个'
         '数据链路中因此只计一次，而其在平台上的业务字段变化仍可追溯。')

INSERT_E07 = para('该类异常的检测规则、真实记录的修复前后对照与修复校验结果如下。') + \
    fig_block('3-9', '公司属性语义槽位异常识别与修复示例',
              '输出：outputs/figures/evidence/02_preprocess/E07_company_slot_repair.png'
              '（src/field_repair.py 126-140；'
              'outputs/tables/22_company_attribute_semantic_anomaly_audit.xlsx）') + \
    para('三段证据共同说明该处理是确定性的：判定规则只依据公司性质栏位是否呈人数区间形态、'
         '公司规模栏位是否呈地点文本形态以及公司所在地栏位是否为空；修复按异常类型回填可'
         '还原的槽位，无法从本行还原的公司性质一律置为缺失，不做众数填充，也不按行业或'
         '规模推断；修复后重新扫描的残留异常为 0。')

INSERT_E08 = para('薪资文本的解析规则与四类真实取值的处理结果如下。') + \
    fig_block('3-10', '薪资字段解析与正式目标构造示例',
              '输出：outputs/figures/evidence/02_preprocess/E08_salary_parse.png'
              '（scripts/11_clean_structured_fields.py 80-107；'
              'data/processed/job_salary_targets.parquet 真实记录）') + \
    para('代码部分给出区间与单值的统一处理逻辑：先判定薪资面议，再在删除单位后提取数值，'
         '只有一个数值时上下限取同一数值，随后执行上下界逻辑检查并计算中点。数据部分给出'
         '四类真实取值：区间形式得到下限、上限与中点，单值形式三列同值，面议形式三个数值列'
         '保持缺失，逻辑异常形式被单独标记。由于全部薪资文本均以天为单位，解析结果统一为'
         '元/天，不做任何单位折算。')

INSERT_E09 = para('薪资泄漏清理规则、要求段落提取与规范技能结果如下。') + \
    fig_block('3-11', '岗位文本安全处理与技能抽取示例',
              '输出：outputs/figures/evidence/02_preprocess/E09_text_skill.png'
              '（src/text_utils.py 344-360；'
              'data/interim/job_text_version_corpus.parquet；'
              'data/features/job_skill_membership.parquet）') + \
    para('清理环节只删除薪资表达本身，技术版本号、实习月数与每周到岗天数等业务数字均保留；'
         '要求段落提取按固定段落标题定位任职要求、岗位要求与任职资格三类段落，命中时标记为'
         ' REQUIREMENT_SECTION，未命中时回退全文并标记为 FULL_TEXT_FALLBACK；技能结果按'
         '标准技能名、技能族与技能组三级给出，并记录该次命中所属的匹配口径。三者共同说明'
         '技能需求统计建立在要求段落优先的口径之上，而非对全文技能词等同处理。')

INSERT_E10 = para('最终建模数据集的代表性字段片段如下。') + \
    fig_block('3-12', '最终建模数据集代表性字段片段',
              '输出：outputs/figures/evidence/02_preprocess/E10_model_dataset.png'
              '（data/processed/job_salary_model_dataset.parquet 等冻结产物）') + \
    para('图中只展示字段级别的建模宽表形态，目标变量薪资中点、岗位与地域字段、公司侧字段、'
         '技能指示列与文本长度均以岗位为一行，编码后的高维输入矩阵不在此展开。正式建模样本'
         '为 14,883 个岗位，经训练集内编码、技能列筛选及文本降维后形成正式模型输入。')

INSERT_E11 = para('上述口径由一套统一的检验函数实现，各因素共用同一实现，不针对单一因素'
                  '另写代码。') + \
    fig_block('5-1', '分组薪资差异检验与多重校正核心代码',
              '输出：outputs/figures/evidence/03_analysis/E11_stats_test_code.png'
              '（src/eda_analysis.py 149-164 / 183-194 / 197-212）') + \
    para('代码部分给出三个环节：单值互斥因素用 Kruskal–Wallis 检验并同时计算 ε²；多值重叠'
         '因素的每个取值构造命中与未命中两组，用 Mann–Whitney U 检验并计算 Cliff\'s δ；'
         '同一因素内的多组成对检验统一做 BH-FDR 校正。本章与后文涉及的城市、学历、行业、'
         '认证、技能等因素均通过这一套实现完成检验，检验族与样本量门槛保持一致。')

INSERT_E12 = para('数据划分与预处理拟合范围由同一段实现控制，其关键部分如下。') + \
    fig_block('7-2', '训练验证测试划分与训练集拟合核心代码',
              '输出：outputs/figures/evidence/04_model/E12_split_code.png'
              '（src/model_training.py 46-66；scripts/14_train_salary_model.py 507-514）') + \
    para('代码部分表明三件事：划分按薪资十分位分箱分层、随机种子固定为 42；三个子集的'
         '比例为 70%、15% 与 15%，对应 10,418、2,232 与 2,233 个岗位；预处理器只在训练集上'
         '拟合，随后分别变换训练集、验证集与测试集。测试集因此不参与任何编码、降维与技能列'
         '筛选，其取值分布不会影响正式模型输入。')

INSERT_E13 = para('五组主体特征的编码方式与拼接顺序如下。') + \
    fig_block('7-3', 'A/B/C/D/E 异构特征构建与拼接核心代码',
              '输出：outputs/figures/evidence/04_model/E13_feature_pipeline_code.png'
              '（src/model_training.py 178-191 / 155-176）') + \
    para('代码部分显示特征矩阵由数值块、类别指示块、多值指示块、技能指示块与文本语义块按'
         '固定顺序横向拼接，列名顺序与拼接顺序严格一致，模型解释与审计据此对齐。技能列按'
         '训练集频次不低于 100 筛选，文本语义向量在训练集上拟合截断奇异值分解并保留 16 维；'
         'A+B+C 编码后 216 维、D 组 56 维、E 组 16 维，合计 288 维，加入发布时间位置扩展'
         '特征后为 290 维。')

INSERT_E14 = para('候选模型的构建方式与主模型的锁定过程如下。') + \
    fig_block('7-4', '候选回归模型训练与验证集选择核心代码',
              '输出：outputs/figures/evidence/04_model/E14_model_selection_code.png'
              '（src/model_training.py 324-348；scripts/14_train_salary_model.py 595-602 / 628-632）') + \
    para('代码部分给出四类真实模型与常数基线的构建方式、超参数小网格的遍历方式以及主模型的'
         '锁定条件：每个候选配置在训练集上拟合、在验证集上计算平均绝对误差，取非基线模型中'
         '验证集平均绝对误差最低者为主模型。测试集不参与模型与超参数的选择，其表现只在配置'
         '锁定后报告一次。')

INSERT_E15 = para('消融配置与两种替代划分的实现方式如下。') + \
    fig_block('8-2', '特征组消融与替代泛化划分核心代码',
              '输出：outputs/figures/evidence/04_model/E15_ablation_generalization_code.png'
              '（src/ablation_shap.py 32-40；src/model_training.py 72-87；'
              'scripts/26b_stage26_1_temporal_tightening.py 945-960）') + \
    para('代码部分给出三件事：消融只改变参与建模的特征组，基础对应 A+B+C，基础+技能对应'
         ' A+B+C+D，基础+文本对应 A+B+C+E，完整模型对应 A+B+C+D+E；按公司分组划分以公司'
         '实体为单位构造互斥分组，保证同一公司的岗位只出现在一个子集中；回顾性时间划分按'
         '业务发布日期排序后按累计占比切割，同一天的岗位不拆分。三种设置共用同一模型族、'
         '同一超参数与同一预处理纪律，只改变特征组或划分方式。')

INSERT_E16 = para('TreeSHAP 的计算方式与技能方向统计口径如下。') + \
    fig_block('8-3', 'TreeSHAP 模型解释与技能方向统计核心代码',
              '输出：outputs/figures/evidence/05_explain/E16_shap_code.png'
              '（src/ablation_shap.py 124-136 / 206-231）') + \
    para('代码部分显示解释值直接取自模型的贡献输出，并对稀疏返回做稠密化处理，基准值列被'
         '单独剥离；技能特征的统计按技能列取值是否为 1 划分存在与不存在两组，分别计算'
         '平均绝对 SHAP 值与技能存在时的平均贡献，用于判定方向。该实现只用于解释已经锁定'
         '的模型结果，不重新训练模型，也不据此调整特征或超参数。')


# --------------------------------------------------------------------------- #
# 分章源改写规则：(文件, [(锚点, 插入内容)], [(旧串, 新串)])
# --------------------------------------------------------------------------- #
REPLACEMENTS = {
    '03_数据获取与预处理.md': [
        ('数据采集总体流程见图 3-1。', '数据采集总体流程见图 3-2。'),
        ('> 【插图：图 3-1 数据采集总体流程】', '> 【插图：图 3-2 数据采集总体流程】'),
        ('治理环节与观测向实体的分层压缩规模见图 3-2。',
         '治理环节与观测向实体的分层压缩规模见图 3-6。'),
        ('> 【插图：图 3-2 数据治理与岗位版本重构流程】',
         '> 【插图：图 3-6 数据治理与岗位版本重构流程】'),
        ('岗位版本重构与唯一实体生成的完整流程见图 3-2。',
         '岗位版本重构与唯一实体生成的完整流程见图 3-6。'),
    ],
    '05_实习岗位薪资影响因素分析.md': [
        ('城市、学历与公司规模的薪资中点中位数见图 5-1：',
         '城市、学历与公司规模的薪资中点中位数见图 5-2：'),
        ('> 【插图：图 5-1 城市、学历与公司规模的薪资中点中位数】',
         '> 【插图：图 5-2 城市、学历与公司规模的薪资中点中位数】'),
        ('公司认证状态的薪资分布与组间比较见图 5-2。',
         '公司认证状态的薪资分布与组间比较见图 5-3。'),
        ('> 【插图：图 5-2 公司认证状态的薪资分布与组间比较】',
         '> 【插图：图 5-3 公司认证状态的薪资分布与组间比较】'),
        ('图 5-3 给出高效应标签的共现结构：', '图 5-4 给出高效应标签的共现结构：'),
        ('> 【插图：图 5-3 高效应福利标签的共现结构】',
         '> 【插图：图 5-4 高效应福利标签的共现结构】'),
    ],
    '08_模型稳健性与解释.md': [
        ('整体解释结果见图 8-2。', '整体解释结果见图 8-4。'),
        ('> 【插图：图 8-2 正式主模型 SHAP 蜂群图（Top12 特征）】',
         '> 【插图：图 8-4 正式主模型 SHAP 蜂群图（Top12 特征）】'),
        ('技能 SHAP Top20 见图 8-3。', '技能 SHAP Top20 见图 8-5。'),
        ('> 【插图：图 8-3 技能特征 SHAP Top20（条长 = 平均绝对 SHAP 值，'
         '标记 = 技能存在时平均贡献方向）】',
         '> 【插图：图 8-5 技能特征 SHAP Top20（条长 = 平均绝对 SHAP 值，'
         '标记 = 技能存在时平均贡献方向）】'),
    ],
}

INSERTIONS = {
    '03_数据获取与预处理.md': [
        ('不构成对全部实习岗位市场的推断。', INSERT_E01),
        ('解析失败的页面单独记录、不写入正式字段并留档处理。', INSERT_E02_E03),
        ('但不能理解为岗位实际的招聘完成时间。', INSERT_E04),
        ('岗位版本重构与唯一实体生成的完整流程见图 3-6。', INSERT_E05 + INSERT_E06),
        ('说明异常来自源记录的语义槽位本身，与数据迁移过程无关。', INSERT_E07),
        ('含上下限倒置与非正数两类，数量很小，直接剔除并留档。', INSERT_E08),
        ('发布时点不可得的版本类特征、与保留特征高度冗余的连续数值特征、'
         '近常量特征与数据治理元数据不进入正式预测模型，其诊断与移出依据见 7.1 节。',
         INSERT_E10),
    ],
    '05_实习岗位薪资影响因素分析.md': [
        ('组间差异不显著时，只说明在本文样本下未观察到稳定差异。', INSERT_E11),
    ],
    '07_薪资预测模型构建与结果分析.md': [
        ('三种划分的口径与对照结果见 8.2 节。', INSERT_E12),
        ('主模型选择与多种子结果一致。', INSERT_E14),
    ],
    '08_模型稳健性与解释.md': [
        ('这一差异来自训练集构成变化，而非使用测试集信息调整特征。'
         '三类泛化场景的对照见表 8-3。', INSERT_E15),
        ('本文的基准值为 192.87 元/天。', INSERT_E16),
    ],
}

# 算法块后的插入：定位算法代码围栏结束行（``` 行）之后
ALGO_ANCHORS = {
    '03_数据获取与预处理.md': ['10: 将别名映射为规范技能、去除重复技能，返回 K 与 source',
                           INSERT_E09],
    '07_薪资预测模型构建与结果分析.md': ['9: 输出 MAE、RMSE、R²', INSERT_E13],
}


def apply_file(name: str) -> list:
    text = (SRC23 / name).read_text(encoding='utf-8')
    lines = text.splitlines()

    for old, new in REPLACEMENTS.get(name, []):
        count = text.count(old)
        if count != 1:
            raise AssertionError('%s 替换锚点命中 %d 次：%s' % (name, count, old[:40]))
        text = text.replace(old, new)

    for anchor, payload in INSERTIONS.get(name, []):
        if payload is None:
            continue
        count = text.count(anchor)
        if count != 1:
            raise AssertionError('%s 插入锚点命中 %d 次：%s' % (name, count, anchor[:40]))
        text = text.replace(anchor, anchor + '\n\n' + '\n'.join(payload))

    # 算法块：在围栏 ``` 结束行之后插入
    algo = ALGO_ANCHORS.get(name)
    if algo:
        anchor, payload = algo
        if text.count(anchor) != 1:
            raise AssertionError('%s 算法锚点命中异常：%s' % (name, anchor[:40]))
        idx = text.index(anchor)
        close = text.index('```', idx)          # 锚点位于围栏内，其后第一个围栏即结束行
        end = text.index('\n', close) + 1
        text = text[:end] + '\n' + '\n'.join(payload) + text[end:]

    return text.splitlines()


def heading_shift(line: str) -> str:
    """分章源标题层级上提一级：`## 3 x` → `# 3 x`，`### 3.1 x` → `## 3.1 x`。"""
    m = re.match(r'^(#{2,4})\s+(.*)$', line)
    if m:
        return line[1:]
    return line


def build_sources() -> dict:
    SRC27.mkdir(parents=True, exist_ok=True)
    files = ['00_摘要与Abstract.md', '01_绪论.md', '02_相关理论与分析方法.md',
             '03_数据获取与预处理.md', '04_互联网IT实习岗位特征分析.md',
             '05_实习岗位薪资影响因素分析.md', '06_实习岗位技能需求分析.md',
             '07_薪资预测模型构建与结果分析.md', '08_模型稳健性与解释.md',
             '09_总结与展望.md']
    out = {}
    for name in files:
        lines = apply_file(name)
        out[name] = lines
        (SRC27 / name).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return out


def build_markdown(sources: dict) -> Path:
    parts = [
        '<!-- 工程证据增强稿（Stage27.0）分章来源：docs/paper/stage27_0/*.md -->',
        '<!-- 图片落位清单：outputs/tables/34_visual_evidence_registry.xlsx -->',
        '<!-- 本轮仅新增工程视觉证据（图 3-1/3-3/3-4/3-5/3-7~3-12、图 5-1、'
        '图 7-2~7-4、图 8-2/8-3），未改动任何正式结果数值 -->',
        '',
        '',
        '# 基于实习僧平台的互联网 IT 实习岗位薪资影响因素分析与预测',
        '',
        '',
    ]
    for name in ['00_摘要与Abstract.md', '01_绪论.md', '02_相关理论与分析方法.md',
                 '03_数据获取与预处理.md', '04_互联网IT实习岗位特征分析.md',
                 '05_实习岗位薪资影响因素分析.md', '06_实习岗位技能需求分析.md',
                 '07_薪资预测模型构建与结果分析.md', '08_模型稳健性与解释.md',
                 '09_总结与展望.md']:
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


def _figure_manifest():
    merged = {}
    for no, title, path, width in EVIDENCE_FIGS + BASE_FIGS:
        merged[no] = (title, path, width)
    order = ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9', '3-10', '3-11',
             '3-12', '4-1', '4-2', '4-3', '4-4', '4-5', '4-6', '4-7', '4-8', '4-9',
             '5-1', '5-2', '5-3', '5-4', '6-1', '6-2', '6-3', '7-1', '7-2', '7-3', '7-4',
             '8-1', '8-2', '8-3', '8-4', '8-5']
    assert sorted(order) == sorted(merged), '图号清单不一致'
    fig_main = [(no, merged[no][0], merged[no][1]) for no in order]
    widths = {no: merged[no][2] for no in order if merged[no][2]}
    return fig_main, widths


def _load_builder_module(name='stage24_builder'):
    spec = importlib.util.spec_from_file_location(
        name, str(ROOT / 'scripts' / '24a_build_stage24_docx.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fig_main, widths = _figure_manifest()
    mod.SRC23 = SRC27
    mod.OUTDOCX = DOCX_OUT
    mod.FIGROOT = FIGROOT
    mod.FIG_MAIN = fig_main
    mod.FIG_WIDTH_CM = widths
    mod.REFS = REFS
    return mod


def build_docx() -> Path:
    mod = _load_builder_module()
    builder = mod.build(DOCX_OUT)
    print('[构建计数] 正文图=%d 正文表=%d 公式=%d 算法=%d'
          % (builder.stats['fig_main'], builder.stats['tbl_main'],
             builder.stats['equations'], builder.stats['algorithms']))
    print('[缺失图片]', builder.missing_figs or '无')
    return DOCX_OUT


def main() -> int:
    sources = build_sources()
    print('分章源写出:', SRC27)
    md = build_markdown(sources)
    print('Markdown 写出:', md)
    docx = build_docx()
    print('Word 写出:', docx, docx.stat().st_size, 'bytes')

    mod = _load_builder_module('stage24_builder_chk')
    info = mod.self_check(docx)
    print('-' * 78)
    print('图题数:', len(info['fig_caps']))
    print('表题数:', len(info['tbl_caps']))
    print('SEQ eq 域:', info['seq_fields'], '静态编号残留:', len(info['static_eq']))
    print('原生 OMML:', info['omml'])
    print('算法用表:', info['algo_tbl'])
    print('禁用工程串:', {k: v for k, v in info['scan'].items() if v})
    print('旧值扫描:', {k: v for k, v in info['scan_old'].items() if v})
    print('正式锚点:', info['anchors'])
    for cap in info['fig_caps']:
        print('   ', cap)
    return 0


if __name__ == '__main__':
    sys.exit(main())

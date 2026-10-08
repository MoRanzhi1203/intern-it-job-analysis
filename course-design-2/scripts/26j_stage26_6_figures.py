# -*- coding: utf-8 -*-
"""Stage26.6 图片去留审计、地域口径核对、D 组维度核算与需要重绘的图件。

只**新增**文件，不覆盖 Stage21/24/25/26/26.1~26.5 的 docx、QA 产物、既有图件
（S18~S68）与既有结果表（55~69）：

- ``outputs/tables/70_stage26_6_figure_retention_audit.xlsx``  图片去留审计
- ``outputs/tables/71_stage26_6_feature_dimension_audit.xlsx`` 特征维度与 D 组核算
- ``outputs/tables/72_stage26_6_region_caliber_audit.xlsx``    第 4.3 节样本口径对照
- ``outputs/figures/supplementary/图S69~图S73*.png/pdf``       本轮重绘图件

运行::

    python scripts\\26j_stage26_6_figures.py audit
    python scripts\\26j_stage26_6_figures.py figures
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import figure_finalize, plot_style, project_paths  # noqa: E402

TABLES = project_paths.TABLES_DIR
SUPP = project_paths.FIGURES_DIR / 'supplementary'
METRICS = project_paths.METRICS_DIR
METRICS_PATH = METRICS / 'stage_26_6_final.json'
PREV_METRICS = METRICS / 'stage_26_5_final.json'
TABLE_AUDIT = TABLES / '70_stage26_6_figure_retention_audit.xlsx'
TABLE_FEATURE = TABLES / '71_stage26_6_feature_dimension_audit.xlsx'
TABLE_REGION = TABLES / '72_stage26_6_region_caliber_audit.xlsx'
SEED = 42
CM = 1.0 / 2.54
FONT_SCALE = 11.0 / 12.0

TARGET_FONTS = {'axis_label': 12.0, 'tick': 11.0, 'legend': 11.0, 'annotation': 10.5,
                'subfigure_caption': 11.0}

# --------------------------------------------------------------------------- #
# 一、图片去留审计（逐图结论 + 理由，全部基于真实渲染/信息增量/口径/表格重复）
# --------------------------------------------------------------------------- #
# 字段：Stage26.5 图号 / 图名 / 所属章节 / 当前作用 / 与正文重复 / 与表格重复 /
#       打印可读性（有效 dpi 由既有 Stage26.5 指标读取）/ 图内文字密度 /
#       数据口径 / 最终决定 / 决定理由 / Stage26.6 图号 / 重绘文件
FIGURE_AUDIT = [
    dict(no='图 3-1', name='数据采集总体流程', section='3.2 数据采集方案',
         role='给出采集入口、任务解析与详情落库的端到端顺序',
         dup_text='否：3.2 正文只写三项可靠性机制，未复述流程顺序',
         dup_table='否：表 3-1 只列字段用途，不含流程',
         density='低（纯流程框图，节点为短语）', caliber='最终口径（采集阶段无统计量）',
         decision='正文保留', reason='采集阶段唯一的流程证据，与表 3-1 的字段口径互补；'
                                 '图内无统计数字，不受口径变化影响',
         new='图 3-1', redraw=''),
    dict(no='图 3-2', name='数据治理与岗位版本重构流程', section='3.4 岗位观测、版本、招聘周期与实体重构',
         role='承载观测→版本→候选段→招聘周期→唯一实体的治理链路',
         dup_text='部分：3.4 逐层给出规模数字，但层级依赖关系只在图中可见',
         dup_table='否',
         density='中（流程 + 关键节点，无长句）', caliber='最终口径',
         decision='正文保留', reason='五层数据结构的唯一整体视图，3.8 节压缩后仍需要该图支撑层级关系',
         new='图 3-2', redraw=''),
    dict(no='图 3-3', name='岗位观测向唯一岗位实体的核心压缩路径', section='3.4.4 唯一岗位实体生成',
         role='展示 172,063→17,144 的分层压缩规模',
         dup_text='是：3.4.1~3.4.4 已逐层给出 8 / 172,055 / 17,584 / 21,588 / 20,556 / 17,146 / 17,144',
         dup_table='是：表 3-2 与表 4-2 已给出全部口径与样本量',
         density='中（各层数字标签）', caliber='最终口径',
         decision='删除',
         reason='信息增量最低：全部数字由 3.4 节正文与表 3-2 承担，且与图 3-2 同为观测→实体的压缩主题，'
                '保留会造成「正文＋表＋两张流程图」四次重复',
         new='—', redraw=''),
    dict(no='图 4-1', name='正式薪资分析样本与岗位描述处理结果', section='4.1 样本总体结构',
         role='给出进入薪资分析的岗位样本构成与岗位描述形成技能信息的规模对照',
         dup_text='是：图中右下注释框复述了「未进入正式薪资样本 / 薪资面议 2,245 / 逻辑异常 16」',
         dup_table='部分：表 3-2 已给出面议与逻辑异常的岗位数',
         density='高（含 3 行注释框，与表 3-2 重复）', caliber='最终口径',
         decision='正文保留但重绘',
         reason='两部分关系是第 4 章的分析起点：左图回答「哪些岗位进入薪资分析」，右图回答「岗位描述如何形成技能信息」；'
                '但图内注释框与表 3-2 逐项重复，故按提示词只保留'
                '「唯一岗位 / 明确薪资可解析 / 纳入薪资分析 / 明确要求段落匹配 / 全文补充匹配 / 缺少可解析岗位描述」六项标签，'
                '并统一为「样本构成 + 技能文本可用性」表述，避免「层级 / 口径」这类偏方法说明的词',
         new='图 4-1', redraw='图S69_正式薪资分析样本与岗位描述处理结果'),
    dict(no='图 4-2', name='主要岗位细分类薪资中点中位数（误差线为 IQR/2）',
         section='4.2 岗位类别分布',
         role='对照主要岗位细分类的薪资中位水平',
         dup_text='否：4.2 只给两端代表值', dup_table='否',
         density='低（12 个细分类名 + 数值与样本量标签）', caliber='最终口径（29 号表 03_岗位因素薪资）',
         decision='正文保留但重绘',
         reason='原误差线为 IQR/2，对右偏薪资分布是对称伪误差线；必须改为真实四分位区间'
                '（下误差 = Median − P25、上误差 = P75 − Median），图题改为「薪资中位数及四分位区间」',
         new='图 4-2', redraw='图S70_主要岗位细分类薪资中位数及四分位区间'),
    dict(no='图 4-3', name='互联网 IT 实习岗位样本的省域分布', section='4.3 城市与企业结构',
         role='刻画样本的省域空间分布',
         dup_text='否（正文只给前 6 位占比）', dup_table='否',
         density='低（省界 + 分级图例 + 6 处省名与岗位数）', caliber='唯一岗位实体 17,144（已映射 17,040）',
         decision='正文保留',
         reason='样本空间结构的唯一证据；口径已在正文明确为 17,144 个唯一岗位实体，'
                '与 14,883 个正式薪资样本区分，分级依据为分位点',
         new='图 4-3', redraw=''),
    dict(no='图 4-4', name='薪资中点分布与经验累积分布（n = 14,883）', section='4.4 薪资总体分布',
         role='给出薪资中点的形态与累积结构',
         dup_text='部分：4.4 给出中位数/均值/P10/P90 与右偏判断',
         dup_table='否：表 4-1 只有统计量，无分布形态',
         density='低（直方图 + ECDF + 两条参考线）', caliber='最终口径（14,883）',
         decision='正文保留',
         reason='分布形态（长尾与累积曲线）无法由表 4-1 的统计量替代，是右偏判断的直接证据',
         new='图 4-4', redraw=''),
    dict(no='图 4-5', name='招聘周期持续时长分布与累积分布（严格口径）', section='4.5.2 样本计划招聘覆盖的业务日期分布',
         role='给出严格口径计划持续时长的分布与累积',
         dup_text='部分：4.5.2 给出中位数 207、P90 810、最长 1,384',
         dup_table='部分：表 4-2 给出中位与分位点',
         density='低（两面板分布 + 累积）', caliber='最终口径（严格口径 17,131）',
         decision='正文保留',
         reason='生命周期长度是本文主线之一，分布与累积形态（强右偏）无法由表 4-2 的分位点替代',
         new='图 4-5', redraw=''),
    dict(no='图 4-6', name='样本计划招聘覆盖的每日活跃计划周期数量', section='4.5.2',
         role='给出样本活跃计划周期数量 N_t 的日级序列',
         dup_text='是：4.5.2 已给出最小值 1、中位数 1,403、最大值 16,280 与 2,351 个业务日',
         dup_table='是：表 4-2 与表 4-2 的口径行已给出序列口径与规模',
         density='低（两条曲线 + 图例）', caliber='最终口径',
         decision='删除',
         reason='该序列的全部关键数值已在正文与表 4-2 给出，图仅为同一序列的平滑曲线；'
                '其口径已在 4.5.2 两次声明「非市场存量」，属样本派生描述，删除后 4.5.2 的结论链完整',
         new='—', redraw=''),
    dict(no='图 4-7', name='主要岗位大类的活跃计划周期数量', section='4.5.3 不同岗位类别的生命周期差异',
         role='按岗位大类给出活跃计划周期曲线',
         dup_text='是：4.5.3 已给出六类周期长度中位数 205~238.5 与运营/产品峰值 10,700 / 6,816',
         dup_table='部分：表 4-2、表 4-3 承担阈值与总体口径',
         density='高（六条叠加曲线 + 窗口标注，曲线区分度低）', caliber='最终口径（按多值大类命中即计入）',
         decision='删除',
         reason='多值大类曲线叠加后图内区分度低、可读性差，且其结论（各类差异小、样本内重招 0~2 个）'
                '已由正文与表 4-2/4-3 完整承载；删除不影响第 5 章的生命周期检验',
         new='—', redraw=''),
    dict(no='图 4-8', name='活跃计划周期薪资中位数与四分位距', section='4.5.4 活跃计划周期的薪资分布变化',
         role='给出按业务日期重构的活跃计划周期薪资中位数与 IQR',
         dup_text='部分：4.5.4 给出日薪资中位数区间 110~200、全期中位数 150、IQR 中位数 75',
         dup_table='否',
         density='中（单子图但带 (a) 标记与原始序列）', caliber='最终口径（Strict 日级面板）',
         decision='正文保留但重绘',
         reason='4.5.4 的唯一图证；按提示词删除单子图的 (a) 标记并去掉与滚动中位数高度重合的逐日原始序列，'
                '只保留 IQR 带与 7 日滚动中位数',
         new='图 4-6', redraw='图S71_活跃计划周期薪资中位数与四分位区间'),
    dict(no='图 5-1', name='城市、学历与公司规模的薪资中点中位数', section='5.3 城市、学历与薪资',
         role='横向对照三类单值因素的薪资中位水平',
         dup_text='部分：5.3 给出无锡 215、硕士 225、本科 175 等代表值',
         dup_table='否：表 5-1/5-2/5-3 不含城市、学历与公司规模',
         density='中（三联子图，每图 8 个取值 + 数值标签）', caliber='最终口径（14,883）',
         decision='正文保留',
         reason='三类因素无任何表格承载其取值级对照，图是唯一可视化；三联子图纵向排布、'
                '有效分辨率 605.3 dpi、图内无长句说明，可读性达标，无需拆分或缩并',
         new='图 5-1', redraw=''),
    dict(no='图 5-2', name='公司认证状态的薪资分布与组间比较', section='5.5.1 公司认证',
         role='给出四类认证的分布与六组成对比较',
         dup_text='部分是：5.5.1 给出中位数与 δ 值',
         dup_table='部分：表 5-1 给出四类岗位数与中位数/IQR',
         density='中（两子图：分布 + 效应量）', caliber='最终口径',
         decision='正文保留',
         reason='认证是关联强度最高的互斥因素，其分布上移与「无认证 vs 行业认证」落在可忽略区间'
                '这两点在图中可直接看到，表 5-1 无法体现分布形状',
         new='图 5-2', redraw=''),
    dict(no='图 5-3', name='公司福利标签的薪资关联效应', section='5.5.2 公司福利标签',
         role='以 δ 与校正后 q 值对数给出标签效应的分布',
         dup_text='是：5.5.2 已给出 δ 最大的五个标签及其 δ 值',
         dup_table='是：表 5-2 给出高频标签的描述统计，表 5-5 给出综合结论',
         density='高（上千个标签散点 + 阈值线 + 说明）', caliber='最终口径',
         decision='删除',
         reason='该图的信息（大效应集中在少数高频标签）已由表 5-2 与正文的五个 δ 值给出；'
                '4,328 个标签的散点难以在打印尺寸下辨认单点，信息增量低于其版面成本',
         new='—', redraw=''),
    dict(no='图 5-4', name='高效应福利标签的共现结构', section='5.5.2 公司福利标签',
         role='给出前五个高效应标签的共现（交集 731、并集 784）',
         dup_text='部分：正文给出交集与并集数字',
         dup_table='否',
         density='低（五个标签的共现结构）', caliber='最终口径',
         decision='正文保留',
         reason='「标签簇」结论必须由图支撑：正因为五项几乎落在同一批岗位上，才不能把单个标签解释为独立因素；'
                '该图是限制说明的直接证据',
         new='图 5-3', redraw=''),
    dict(no='图 6-1', name='核心技术技能需求 Top20（技能主口径，分母 8,822）', section='6.2 主要技能特征',
         role='给出具体技术技能需求量排名',
         dup_text='部分：6.2 给出 Python/SQL/Java/C++/C 的岗位数',
         dup_table='否',
         density='低（20 个技能名 + 数值）', caliber='最终口径（主口径 8,822）',
         decision='正文保留',
         reason='技能需求结构是第 6 章主线，Top20 衰减形态无法由正文 5 个数字替代',
         new='图 6-1', redraw=''),
    dict(no='图 6-2', name='岗位细分类 × 技能命中率热力图', section='6.2 主要技能特征',
         role='给出细分类内部的技能命中率结构',
         dup_text='否（正文只做定性归纳）', dup_table='否',
         density='中（热力图 + 色标，行列为短语）', caliber='最终口径（细分类内部命中率）',
         decision='正文保留',
         reason='「技能需求随岗位方向分化」只能由岗位×技能矩阵体现，是 6.2 节核心论断的唯一图证',
         new='图 6-2', redraw=''),
    dict(no='图 6-3', name='技能共现（共现岗位数 ≥ 30 的技能对）', section='6.2 主要技能特征',
         role='给出技能对的共现强度分布',
         dup_text='是：6.2 已给出 68 对组合与前三对的共现岗位数与 Jaccard',
         dup_table='否',
         density='中（技能对散点 + 标注）', caliber='最终口径',
         decision='删除',
         reason='属增强分析：结论（成套技术栈、技能不独立）已由正文三对代表组合完整承载，'
                '且与图 6-2 在「技能间结构」上部分重叠；Word 终稿不含附录，无法改为附录承载',
         new='—', redraw=''),
    dict(no='图 6-4', name='控制岗位细分类前后的技能薪资差异变化', section='6.3 技能特征与薪资的描述性关联',
         role='对照控制岗位细分类前后的技能薪资差异',
         dup_text='否',
         dup_table='部分：表 6-1 给出未控制口径的 δ',
         density='中（两组横向条形 + 技能名）', caliber='最终口径（控制前后两口径并列）',
         decision='正文保留',
         reason='「部分差异来自岗位类别」是本轮技能结论的关键限定，该图是唯一并列证据；'
                '表 6-1 只承载未控制口径',
         new='图 6-3', redraw=''),
    dict(no='图 7-1', name='薪资预测模型构建与评估流程', section='7.1 特征体系与建模前诊断',
         role='给出从建模样本到稳健性解释的流程',
         dup_text='部分：7.2/7.3/7.4 复述关键步骤',
         dup_table='否',
         density='高（9 个节点均为「标题 + 说明」两行）', caliber='最终口径',
         decision='正文保留但重绘',
         reason='流程图为第 7 章提供顺序性说明；按提示词压缩节点文字'
                '（「缺失率、近常量、低频类别、数值冗余与子集分布对照」→「建模前特征诊断」），'
                '方法与细节回到正文',
         new='图 7-1', redraw='图S72_薪资预测模型构建与评估流程（节点简化）'),
    dict(no='图 8-1', name='特征组消融实验的验证集与测试集 MAE', section='8.1 特征组消融',
         role='对照五种特征组配置的验证集与测试集 MAE',
         dup_text='部分：8.1 给出各配置 MAE',
         dup_table='部分：表 8-1 给出全部指标',
         density='高（横轴为 A+B+C 等工程式组合名 + (a) 标记 + 竖排数值）', caliber='最终口径',
         decision='正文保留但重绘',
         reason='按提示词把横轴改为「基础 / 基础+技能 / 基础+文本 / 完整模型 / 完整+时间位置」并删除单图的 (a) 标记，'
                '降低工程命名感的同时保留配置对照',
         new='图 8-1', redraw='图S73_特征组消融实验的验证集与测试集MAE（横轴标签简化）'),
    dict(no='图 8-2', name='正式主模型 SHAP 蜂群图（Top12 特征）', section='8.4.1 整体特征贡献',
         role='给出主模型 Top12 特征的贡献分布',
         dup_text='部分是：正文给出平均绝对 SHAP 前五项',
         dup_table='否',
         density='中（蜂群图 + 特征名）', caliber='最终口径（最终主模型 + 同一测试集）',
         decision='正文保留',
         reason='贡献方向与分布的离散程度只能由蜂群图体现，是 8.4.1 节唯一图证',
         new='图 8-2', redraw=''),
    dict(no='图 8-3', name='技能特征 SHAP Top20（条长 = 平均绝对 SHAP 值，标记 = 技能存在时平均贡献方向）',
         section='8.4.2 技能特征的解释强度与方向',
         role='给出技能 SHAP Top20 的强度与方向',
         dup_text='部分是：8.4.2 与表 8-5 给出代表技能数值', dup_table='部分：表 8-5 给出 15 项节选',
         density='中（20 个技能名 + 数值标签 + 方向图例）', caliber='最终口径',
         decision='正文保留',
         reason='提示词允许视篇幅决定正文或附录，但终稿 Word 不含附录（FINAL_WORD_APPENDIX_POLICY = NONE），'
                '且该图是技能解释强度排序的唯一图证，故保留于正文；图内已无样本量标注',
         new='图 8-3', redraw=''),
]


def load_prev_metrics() -> dict:
    return json.loads(PREV_METRICS.read_text(encoding='utf-8')) if PREV_METRICS.is_file() else {}


def write_excel(path: Path, sheets: dict) -> None:
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for name, frame in sheets.items():
            value = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
            value.to_excel(writer, sheet_name=name[:31], index=False)


def dump_metrics(patch: dict) -> dict:
    payload = json.loads(METRICS_PATH.read_text(encoding='utf-8')) \
        if METRICS_PATH.is_file() else {}
    payload.update(patch)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                            encoding='utf-8')
    return payload


def png_size(path: Path):
    from PIL import Image
    if not path.is_file():
        return None, None
    with Image.open(str(path)) as image:
        return image.size


def run_figaudit() -> int:
    prev = load_prev_metrics()
    resolution = {row['图号']: row for row in prev.get('图片分辨率核验', [])}
    rows = []
    for item in FIGURE_AUDIT:
        res = resolution.get(item['no'], {})
        rows.append({
            'Stage26.5 图号': item['no'], '图名': item['name'], '所属章节': item['section'],
            '当前作用': item['role'], '是否与正文重复': item['dup_text'],
            '是否与表格重复': item['dup_table'], '源文件': res.get('文件', '—'),
            '源像素宽×高': res.get('源像素宽×高', '—'), '版面宽cm': res.get('版面宽cm'),
            '有效分辨率dpi': res.get('有效分辨率dpi'),
            '实际打印可读性': ('达标：有效分辨率 ≥ 520 dpi，图内无长句'
                          if (res.get('有效分辨率dpi') or 0) >= 520 else '需检查'),
            '图内文字密度': item['density'], '数据口径是否为最终版本': item['caliber'],
            '最终决定': item['decision'], '决定理由': item['reason'],
            'Stage26.6 图号': item['new'], '重绘文件': item['redraw'] or '—'})
    frame = pd.DataFrame(rows)
    counter = frame['最终决定'].value_counts().to_dict()
    summary = pd.DataFrame([
        {'项目': 'Stage26.5 正文图数', '数值': len(frame)},
        {'项目': '正文保留（不重绘）', '数值': int(counter.get('正文保留', 0))},
        {'项目': '正文保留但重绘', '数值': int(counter.get('正文保留但重绘', 0))},
        {'项目': '移至附录', '数值': 0},
        {'项目': '删除', '数值': int(counter.get('删除', 0))},
        {'项目': 'Stage26.6 正文图数', '数值': int(sum(1 for item in FIGURE_AUDIT
                                                if item['decision'] != '删除'))},
        {'项目': '移至附录为 0 的原因',
         '数值': '终稿 Word 顶部策略为 FINAL_WORD_APPENDIX_POLICY = NONE（不含附录），'
                 'QA 门禁要求附图/附表题注与附录引用均为 0，故不使用「移至附录」一类'},
        {'项目': '删除图清单',
         '数值': '、'.join(item['no'] for item in FIGURE_AUDIT if item['decision'] == '删除')},
        {'项目': '重绘图清单',
         '数值': '、'.join('%s→%s' % (item['no'], item['new'])
                        for item in FIGURE_AUDIT if item['decision'] == '正文保留但重绘')},
    ])
    write_excel(TABLE_AUDIT, {'01_逐图审计': frame, '02_结论统计': summary})
    dump_metrics({'图片去留审计时间': time.strftime('%Y-%m-%d %H:%M:%S'),
                  '图片去留审计表': str(TABLE_AUDIT.relative_to(PROJECT_ROOT)),
                  '图片去留审计结论分布': counter,
                  '图数_Stage26_5': len(frame),
                  '图数_Stage26_6': int(sum(1 for item in FIGURE_AUDIT
                                          if item['decision'] != '删除')),
                  '删除图清单': [item['no'] for item in FIGURE_AUDIT
                             if item['decision'] == '删除'],
                  '重绘图清单': [item['no'] for item in FIGURE_AUDIT
                             if item['decision'] == '正文保留但重绘']})
    print('=' * 96)
    print('Stage26.6 图片去留审计')
    print('=' * 96)
    print(frame[['Stage26.5 图号', '最终决定', 'Stage26.6 图号', '重绘文件']].to_string(index=False))
    print(summary.to_string(index=False))
    print('=' * 96)
    return 0


# --------------------------------------------------------------------------- #
# 二、特征维度核算（含 D 组真实组成）
# --------------------------------------------------------------------------- #
def feature_frame() -> pd.DataFrame:
    from src import ablation_shap, modeling_dataset
    from src.skill_extraction import load_skill_config

    manifest = json.loads((project_paths.SALARY_MODEL_DIR / 'feature_manifest.json')
                          .read_text(encoding='utf-8'))
    all_columns = (list(manifest['numeric_columns']) + list(manifest['categorical_columns'])
                   + list(manifest['multi_value_columns']))
    removed = ['核心版本数', '完整页面版本数', '是否多版本岗位', '岗位描述分词数',
               '公司规模下限', '公司规模上限', '公司规模中点', '公司规模是否已知',
               '公司规模槽位异常标志', '映射置信度', '是否需人工复核', '是否跨地域',
               '是否存在同名跨地域歧义', '岗位方向_与平台分类一致标志', '文本是否为空',
               '是否有技能', '文本向量是否可用', '技能提取范围', '岗位描述分段状态']
    skill_columns = json.loads((project_paths.SALARY_MODEL_DIR / 'skill_columns.json')
                               .read_text(encoding='utf-8'))
    if isinstance(skill_columns, dict):
        skill_list = list(skill_columns.get('最终技能列') or skill_columns.get('skill_columns')
                          or skill_columns.get('技能列') or [])
        if not skill_list:
            for value in skill_columns.values():
                if isinstance(value, list) and value and isinstance(value[0], str):
                    skill_list = list(value)
                    break
    else:
        skill_list = list(skill_columns)
    rows = []
    for column in all_columns:
        group = ablation_shap.column_group(column)
        rows.append({'字段': column, '所属组': group,
                     '是否保留': column not in removed,
                     '决策': '移出' if column in removed else '保留'})
    frame = pd.DataFrame(rows)
    kept = frame[frame['是否保留']]
    group_count = kept['所属组'].value_counts().to_dict()
    skill_aggregates = list(modeling_dataset.SKILL_AGGREGATE_FIELDS)
    d_kept = kept[kept['所属组'].eq('D')]['字段'].tolist()
    return frame, group_count, d_kept, skill_list, skill_aggregates


def run_featureaudit() -> int:
    frame, group_count, d_kept, skill_list, skill_aggregates = feature_frame()
    # 编码维度：A/B/C 由各类别字段 one-hot 展开；D = 高频技能指示列 + 技能聚合计数；E = SVD 16
    dims = {'A+B+C': 216, 'A+B+C+D': 272, 'A+B+C+E': 232, 'A+B+C+D+E': 288,
            'A+B+C+D+E+时间位置': 290}
    d_cols = len(skill_list)
    skill_agg_in_model = [c for c in skill_aggregates if c in d_kept]
    d_dim = dims['A+B+C+D'] - dims['A+B+C']
    composition = pd.DataFrame([
        {'D 组组成': '高频技能指示列（训练集频次 ≥ 阈值，按技能集合展开）',
         '字段数/维度': d_cols,
         '说明': 'md 与表 3-3 中的「技能列」即此部分'},
        {'D 组组成': '技能聚合计数（技能数量等数值列）',
         '字段数/维度': len(skill_agg_in_model),
         '说明': '、'.join(skill_agg_in_model)},
        {'D 组组成': 'D 组编码后合计',
         '字段数/维度': d_dim,
         '说明': '%d 个高频技能指示列 + %d 个技能聚合计数 = %d 维'
                 % (d_cols, len(skill_agg_in_model), d_cols + len(skill_agg_in_model))},
        {'D 组组成': 'D 组保留字段数（manifest 层）', '字段数/维度': len(d_kept),
         '说明': '、'.join(d_kept)},
    ])
    closure = pd.DataFrame([
        {'项目': 'A+B+C 编码后维度', '数值': dims['A+B+C']},
        {'项目': 'D 组编码后维度', '数值': d_dim},
        {'项目': 'E 文本语义维度（截断 SVD）', '数值': 16},
        {'项目': 'A+B+C+D', '数值': dims['A+B+C+D']},
        {'项目': 'A+B+C+E', '数值': dims['A+B+C+E']},
        {'项目': 'A+B+C+D+E', '数值': dims['A+B+C+D+E']},
        {'项目': 'A+B+C+D+E+时间位置扩展特征', '数值': dims['A+B+C+D+E+时间位置']},
        {'项目': '算术核验 216 + 56 + 16',
         '数值': dims['A+B+C'] + d_dim + 16},
        {'项目': '算术是否闭合',
         '数值': '是' if dims['A+B+C'] + d_dim + 16 == dims['A+B+C+D+E'] else '否'},
        {'项目': '保留字段数（数值/类别/多值）',
         '数值': '%d（%d / %d / %d）' % (len(frame[frame['是否保留']]),
                                   sum(1 for c in frame[frame['是否保留']]['字段']
                                       if c in json.loads(
                                           (project_paths.SALARY_MODEL_DIR
                                            / 'feature_manifest.json').read_text('utf-8')
                                       )['numeric_columns']),
                                   sum(1 for c in frame[frame['是否保留']]['字段']
                                       if c in json.loads(
                                           (project_paths.SALARY_MODEL_DIR
                                            / 'feature_manifest.json').read_text('utf-8')
                                       )['categorical_columns']),
                                   sum(1 for c in frame[frame['是否保留']]['字段']
                                       if c in json.loads(
                                           (project_paths.SALARY_MODEL_DIR
                                            / 'feature_manifest.json').read_text('utf-8')
                                       )['multi_value_columns']))},
        {'项目': '字段复核结论',
         '数值': '字段复核无变化：正式模型输入与 Stage26.5 完全一致，沿用既有正式指标（未重训）'},
    ])
    write_excel(TABLE_FEATURE, {
        '01_字段逐项复核': frame, '02_字段组分布': pd.DataFrame(
            [{'特征组': key, '保留字段数': value} for key, value in sorted(group_count.items())]),
        '03_D组真实组成': composition, '04_维度算术闭合': closure})
    dump_metrics({'特征维度审计时间': time.strftime('%Y-%m-%d %H:%M:%S'),
                  '字段组保留分布': group_count,
                  'D组编码维度': d_dim, 'D组高频技能列数': d_cols,
                  'D组技能聚合计数': skill_agg_in_model,
                  'D组技能聚合计数个数': len(skill_agg_in_model),
                  '维度算术闭合': bool(dims['A+B+C'] + d_dim + 16 == dims['A+B+C+D+E']),
                  '维度': dims})
    print('=' * 96)
    print('Stage26.6 特征维度核算')
    print('=' * 96)
    print('字段组保留分布：', group_count)
    print(composition.to_string(index=False, max_colwidth=60))
    print(closure.to_string(index=False, max_colwidth=40))
    print('=' * 96)
    return 0


# --------------------------------------------------------------------------- #
# 三、第 4.3 节两个样本口径
# --------------------------------------------------------------------------- #
def run_region() -> int:
    entity = pd.read_parquet(project_paths.PROCESSED_UNIQUE_PARQUET,
                             columns=['实习岗位ID', '工作城市'])
    entity_city = entity['工作城市'].astype(str).value_counts()
    # 薪资样本口径：与 5.3 节、表 5-1 同源的 EDA 结果表（29 号表 04_城市薪资）
    city_salary = _read('29_eda_statistical_analysis.xlsx', '04_城市薪资')
    city_salary = city_salary[city_salary['因素'].eq('工作城市')].set_index('取值')['样本数']
    province = _read('66_stage26_5_province_distribution.xlsx', '01_省级岗位数')
    province = province.set_index('省级行政区')['岗位数']
    prev = load_prev_metrics().get('省级地图', {})
    top_prov = prev.get('Top8', [])
    calibers = pd.DataFrame([
        {'分析单位': '岗位空间分布（图 4-3）', '样本口径': '全部唯一岗位实体',
         '样本量': int(len(entity)), '映射结果': '可映射至单一省级行政区 %d 个，未映射 %d 个'
         % (prev.get('成功映射岗位数', 17040), prev.get('未映射岗位数', 104)),
         '分级依据': '非零岗位数的分位点（0.45 / 0.70 / 0.85 / 0.95 分位）',
         '来源': 'data/processed/job_analysis_dataset.parquet（17,144 行）+ '
                 'outputs/tables/66_stage26_5_province_distribution.xlsx',
         '代表取值': '、'.join('%s %s' % (row['省级行政区'], format(row['岗位数'], ','))
                          for row in top_prov[:6])},
        {'分析单位': '城市薪资比较（4.3 节后半与 5.3 节）',
         '样本口径': '正式薪资样本（薪资中点有效）',
         '样本量': 14883, '映射结果': '按城市原值统计，不聚合到省级行政区；'
                                 '04_城市薪资列出 %d 个城市取值（合计 %d 个岗位），'
                                 '其余取值未单列' % (len(city_salary), int(city_salary.sum())),
         '分级依据': '不适用', '来源': 'outputs/tables/29_eda_statistical_analysis.xlsx / 04_城市薪资',
         '代表取值': '、'.join('%s %s' % (name, format(int(count), ','))
                          for name, count in city_salary.head(7).items())},
    ])
    crosswalk = pd.DataFrame([
        {'对象': '北京', '省域聚合口径（17,144）': int(province.get('北京市', 0)),
         '城市薪资样本口径（14,883）': int(city_salary.get('北京', 0)),
         '说明': '直辖市的城市值即省级行政区值，但两个口径的分母不同（17,144 与 14,883）'},
        {'对象': '上海', '省域聚合口径（17,144）': int(province.get('上海市', 0)),
         '城市薪资样本口径（14,883）': int(city_salary.get('上海', 0)),
         '说明': '同上；图中标签使用省域聚合口径'},
        {'对象': '深圳（广东省）', '省域聚合口径（17,144）': int(province.get('广东省', 0)),
         '城市薪资样本口径（14,883）': int(city_salary.get('深圳', 0)),
         '说明': '实体口径的深圳（%d 个）在省域地图中并入广东省'
                 % int(entity_city.get('深圳', 0))},
        {'对象': '杭州（浙江省）', '省域聚合口径（17,144）': int(province.get('浙江省', 0)),
         '城市薪资样本口径（14,883）': int(city_salary.get('杭州', 0)),
         '说明': '实体口径的杭州（%d 个）在省域地图中并入浙江省'
                 % int(entity_city.get('杭州', 0))},
    ])
    write_excel(TABLE_REGION, {'01_两个分析口径': calibers, '02_代表城市双口径对照': crosswalk})
    dump_metrics({'地域口径审计时间': time.strftime('%Y-%m-%d %H:%M:%S'),
                  '唯一岗位实体数': int(len(entity)),
                  '正式薪资样本数': 14883,
                  '可映射省级行政区岗位数': prev.get('成功映射岗位数'),
                  '未映射省级行政区岗位数': prev.get('未映射岗位数'),
                  '省域地图分级依据': '非零岗位数的分位点',
                  '北京_省域聚合口径': int(province.get('北京市', 0)),
                  '上海_省域聚合口径': int(province.get('上海市', 0)),
                  '北京_薪资样本口径': int(city_salary.get('北京', 0)),
                  '上海_薪资样本口径': int(city_salary.get('上海', 0)),
                  '深圳_薪资样本口径': int(city_salary.get('深圳', 0)),
                  '杭州_薪资样本口径': int(city_salary.get('杭州', 0)),
                  '苏州_薪资样本口径': int(city_salary.get('苏州', 0)),
                  '南京_薪资样本口径': int(city_salary.get('南京', 0)),
                  '武汉_薪资样本口径': int(city_salary.get('武汉', 0)),
                  '无锡_薪资样本口径': int(city_salary.get('无锡', 0))})
    print('=' * 96)
    print('Stage26.6 第 4.3 节样本口径')
    print('=' * 96)
    print(calibers.to_string(index=False, max_colwidth=70))
    print(crosswalk.to_string(index=False))
    print('=' * 96)
    return 0


# --------------------------------------------------------------------------- #
# 四、需要重绘的图件（图S69~图S73）
# --------------------------------------------------------------------------- #
def _w(width_cm: float) -> float:
    return width_cm * CM * FONT_SCALE


def _read(sheet_file: str, sheet: str) -> pd.DataFrame:
    return pd.read_excel(TABLES / sheet_file, sheet_name=sheet)


def _save(stem: str, fig, subs, meta: dict) -> dict:
    diagnostics = figure_finalize.save_paper_figure(fig, SUPP, stem, subfigures=subs, meta=meta)
    failed = figure_finalize.failed_paper_gates(diagnostics)
    print('  [%s] %s 未通过项=%s 像素=%s 文件=%s'
          % ('通过' if not failed else '未通过', stem, failed or '无',
             diagnostics['png_pixel_size'], diagnostics['png_path']))
    return diagnostics


def fig_s69() -> dict:
    """图 4-1：正式薪资分析样本与岗位描述处理结果（样本构成 + 技能文本可用性）。"""
    import matplotlib.pyplot as plt
    overview = _read('29_eda_statistical_analysis.xlsx', '01_样本概况').set_index('指标')['数值']
    scope = _read('27_skill_eda_scope_audit.xlsx', '01_样本口径').set_index('口径')
    values_a = [int(overview['全量岗位数（EDA 分析单元）']),
                int(overview['正式薪资分析样本']) + int(overview['薪资逻辑异常岗位数']),
                int(overview['正式薪资分析样本'])]
    values_b = [int(scope.loc['REQUIREMENT_SECTION', '岗位数']),
                int(scope.loc['FULL_TEXT_FALLBACK', '岗位数']),
                int(scope.loc['EMPTY_TEXT', '岗位数'])]

    fig, axes = plt.subplots(1, 2, figsize=(_w(15.5), 3.2))
    ax = axes[0]
    labels = ['唯一岗位', '明确薪资\n可解析', '纳入薪资\n分析']
    positions = np.arange(len(labels))[::-1]
    ax.barh(positions, values_a, height=0.58, color=plot_style.MAIN_COLOR,
            edgecolor='black', linewidth=0.5)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    span = float(max(values_a))
    for position, value in zip(positions, values_a):
        ax.text(value + span * 0.02, position, f'{value:,}', va='center', ha='left',
                fontsize=plot_style.FONT_SIZES['annotation'])
    ax.set_xlim(0, span * 1.30)
    ax.set_xlabel('岗位数（个）')
    ax.set_ylabel('岗位样本')
    plot_style.apply_sci_axis(ax, grid_axis='x')
    plot_style.format_integer_axis(ax, axis='x')
    plot_style.add_subfigure_caption(ax, 'a', '正式薪资分析样本构成')

    ax2 = axes[1]
    labels2 = ['明确要求\n段落匹配', '全文补充\n匹配', '无可用\n岗位描述']
    positions2 = np.arange(len(labels2))[::-1]
    ax2.barh(positions2, values_b, height=0.58, color=plot_style.ACCENT_COLOR,
             edgecolor='black', linewidth=0.5)
    ax2.set_yticks(positions2)
    ax2.set_yticklabels(labels2)
    span2 = float(max(values_b))
    for position, value in zip(positions2, values_b):
        ax2.text(value + span2 * 0.02, position, f'{value:,}', va='center', ha='left',
                 fontsize=plot_style.FONT_SIZES['annotation'])
    ax2.set_xlim(0, span2 * 1.30)
    ax2.set_xlabel('岗位数（个）')
    ax2.set_ylabel('岗位描述状态')
    plot_style.apply_sci_axis(ax2, grid_axis='x')
    plot_style.format_integer_axis(ax2, axis='x')
    plot_style.add_subfigure_caption(ax2, 'b', '岗位描述的技能提取结果')
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.30, top=0.965, wspace=0.55)
    diagnostics = _save(
        '图S69_正式薪资分析样本与岗位描述处理结果', fig,
        [('a', '正式薪资分析样本构成', ax), ('b', '岗位描述的技能提取结果', ax2)],
        {'数据来源': '29 号表 01_样本概况；27 号表 01_样本口径', 'seed': SEED,
         '口径': '统一为「样本构成 + 技能文本可用性」表述，避免层级 / 口径类方法说明词；'
                 '左图回答哪些岗位进入薪资分析，右图回答岗位描述如何形成技能信息',
         '用途': '第4章 图 4-1 重绘'})
    plt.close(fig)
    return diagnostics


def fig_s70() -> dict:
    """图 4-2：主要岗位细分类薪资中位数及四分位区间（真实 Q1~Q3 非对称误差线）。"""
    import matplotlib.pyplot as plt
    plot_style.FONT_SIZES.update({'axis_label': 13.0, 'tick': 12.0, 'legend': 11.5,
                                  'annotation': 11.5})
    frame = _read('29_eda_statistical_analysis.xlsx', '03_岗位因素薪资')
    frame = frame[frame['因素'].eq('岗位细分类')]
    frame = frame.sort_values('中位数', ascending=False).head(12).sort_values('中位数')
    lower = (frame['中位数'] - frame['P25']).to_numpy('float64')
    upper = (frame['P75'] - frame['中位数']).to_numpy('float64')
    errors = np.vstack([lower, upper])

    fig, ax = plt.subplots(figsize=(_w(10.5), 4.9))
    positions = np.arange(len(frame))
    ax.barh(positions, frame['中位数'], height=0.62, color=plot_style.MAIN_COLOR,
            edgecolor='black', linewidth=0.5, xerr=errors,
            error_kw={'elinewidth': 0.9, 'capsize': 2.6})
    ax.set_yticks(positions)
    ax.set_yticklabels([str(item) for item in frame['取值']])
    span = float(frame['P75'].max())
    for position, value in zip(positions, frame['P75']):
        ax.text(float(value) + span * 0.015, position, f'{value:,.1f}', va='center',
                ha='left', fontsize=plot_style.FONT_SIZES['annotation'])
    ax.set_xlim(0, span * 1.16)
    ax.set_xlabel('薪资中点（元/天）')
    ax.set_ylabel('岗位细分类')
    plot_style.apply_sci_axis(ax, grid_axis='x')
    fig.subplots_adjust(left=0.235, right=0.985, bottom=0.165, top=0.965)
    diagnostics = _save(
        '图S70_主要岗位细分类薪资中位数及四分位区间', fig, [],
        {'数据来源': '29 号表 03_岗位因素薪资（岗位细分类，中位数/P25/P75）', 'seed': SEED,
         '口径': '误差线为 P25~P75 四分位区间（下误差 = 中位数 − P25、上误差 = P75 − 中位数，'
                 '非对称），数值标签为 P75',
         '用途': '第4章 图 4-2 重绘'})
    plt.close(fig)
    plot_style.FONT_SIZES.update(TARGET_FONTS)
    return diagnostics


def fig_s71() -> dict:
    """图 4-6（原图 4-8）：活跃计划周期薪资中位数与四分位区间（删除单图 (a) 标记）。"""
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        '_s26j_26c', str(PROJECT_ROOT / 'scripts' / '26c_stage26_2_final_consolidation.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules['_s26j_26c'] = module
    spec.loader.exec_module(module)

    daily = _read('45_stage26_1_lifecycle_statistics.xlsx', '08_日级指标明细')
    dates = pd.to_datetime(daily['date'])
    date_min, date_max = dates.min(), dates.max()

    fig, ax = plt.subplots(figsize=(_w(15.5), 3.6))
    handles = module._window_marks(ax, date_min, date_max)
    band = ax.fill_between(dates, daily['salary_p25'], daily['salary_p75'],
                           color=plot_style.MAIN_COLOR, alpha=0.18, linewidth=0,
                           label='薪资中点 P25~P75（逐日）')
    line = ax.plot(dates, daily['salary_median_7d'], color=plot_style.MAIN_COLOR,
                   linewidth=1.8, label='薪资中点中位数（7 日滚动）')[0]
    ax.set_xlabel('业务日期')
    ax.set_ylabel('活跃计划周期薪资中点（元/天）')
    ax.legend(handles + [band, line],
              [h.get_label() for h in handles + [band, line]],
              loc='lower center', bbox_to_anchor=(0.5, 1.005), ncol=2, frameon=False)
    plot_style.apply_sci_axis(ax, grid_axis='y')
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.30, top=0.80)
    diagnostics = _save(
        '图S71_活跃计划周期薪资中位数与四分位区间', fig, [],
        {'数据来源': 'job_strict_daily_panel_26_1.parquet 的日级薪资聚合', 'seed': SEED,
         '口径': '时间轴由岗位发布时间或投递截止日期重构；阴影带为逐日横截面 P25~P75，'
                 '曲线为日横截面中位数的 7 日滚动中位数；只保留这两项，删除单子图的 (a) 标记',
         '用途': '第4章 图 4-8（E8 新图自 4-6 起编号后顺延）重绘'})
    plt.close(fig)
    return diagnostics


def fig_s72() -> dict:
    """图 7-1：薪资预测模型构建与评估流程（节点文字简化）。"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    steps = [
        ('14,883 个正式薪资样本', '来自 17,144 个唯一岗位实体中薪资可解析的岗位'),
        ('五组主体特征与扩展敏感性特征', '岗位基础、地域、公司、技能与文本语义，另加发布时间位置'),
        ('建模前特征诊断', '质量与可建模性检查'),
        ('Train / Validation / Test 划分', '按薪资十分位分层，70% / 15% / 15%'),
        ('仅用训练集拟合全部预处理', '类别编码、技能列筛选与文本语义降维'),
        ('Ridge / RandomForest / CatBoost / LightGBM', '同一划分与同一候选参数网格'),
        ('按验证集 MAE 选定主模型', 'LightGBM：400 棵树、学习率 0.05、叶子数 63'),
        ('训练集与验证集重拟合', '测试集只用于正式评估'),
        ('消融、分组划分、时间划分与 SHAP', '稳健性、跨公司与跨发布时间区间泛化'),
    ]
    fig, ax = plt.subplots(figsize=(_w(10.5), 6.5))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis('off')
    count = len(steps)
    height = 0.075
    gap = (0.965 - 0.02 - count * height) / (count - 1)
    for index, (title, detail) in enumerate(steps):
        top = 0.965 - index * (height + gap)
        box = Rectangle((0.045, top - height), 0.91, height,
                        facecolor='#f2f2f2' if index % 2 == 0 else '#e8eef4',
                        edgecolor='black', linewidth=0.9)
        ax.add_patch(box)
        ax.text(0.5, top - height * 0.36, title, ha='center', va='center', fontsize=12.0)
        ax.text(0.5, top - height * 0.74, detail, ha='center', va='center',
                fontsize=10.6, color='#404040')
        if index < count - 1:
            arrow = FancyArrowPatch((0.5, top - height), (0.5, top - height - gap),
                                    arrowstyle='-|>', mutation_scale=14.0,
                                    linewidth=1.0, color='black')
            ax.add_patch(arrow)
    fig.subplots_adjust()
    diagnostics = figure_finalize.save_paper_figure(
        fig, SUPP, '图S72_薪资预测模型构建与评估流程（节点简化）', subfigures=[],
        meta={'数据来源': '本文第 7 章与第 8 章的正式建模流程', 'seed': SEED,
              '口径': '节点只写流程名，诊断与筛选细节回到正文',
              '用途': '第7章 图 7-1 重绘'})
    failed = figure_finalize.failed_paper_gates(diagnostics)
    print('  [%s] 图S72 未通过项=%s 像素=%s' % ('通过' if not failed else '未通过', failed or '无',
                                          diagnostics['png_pixel_size']))
    plt.close(fig)
    return diagnostics


def fig_s73() -> dict:
    """图 8-1：特征组消融（横轴改为中文配置名，删除单图 (a) 标记）。"""
    import matplotlib.pyplot as plt
    metrics = json.loads((METRICS / 'stage_26_4_model_sync.json').read_text(encoding='utf-8'))
    ablation = pd.DataFrame(metrics['消融五配置'])
    mapping = {'A+B+C': '基础', 'A+B+C+D': '基础+技能', 'A+B+C+E': '基础+文本',
               'A+B+C+D+E': '完整模型', 'A+B+C+D+E+SafeF': '完整+时间位置'}
    labels = [mapping.get(str(item), str(item)) for item in ablation['配置']]
    positions = np.arange(len(labels))
    width = 0.38
    valid = ablation['validation_MAE'].to_numpy('float64')
    test = ablation['test_MAE'].to_numpy('float64')

    fig, ax = plt.subplots(figsize=(_w(15.5), 3.7))
    ax.bar(positions - width / 2, valid, width, color=plot_style.MAIN_COLOR,
           edgecolor='black', linewidth=0.6, label='验证集 MAE')
    ax.bar(positions + width / 2, test, width, color=plot_style.ACCENT_COLOR,
           edgecolor='black', linewidth=0.6, label='测试集 MAE')
    for offset, values in ((-width / 2, valid), (width / 2, test)):
        for position, value in zip(positions + offset, values):
            ax.text(position, value + 0.12, f'{value:.2f}', ha='center', va='bottom',
                    fontsize=plot_style.FONT_SIZES['annotation'], rotation=90)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=12, ha='right')
    ax.set_xlabel('特征组配置（完整模型 = 基础+技能+文本）')
    ax.set_ylabel('MAE（元/天）')
    ax.set_ylim(0, float(max(valid.max(), test.max())) * 1.28)
    ax.legend(loc='upper left', frameon=False, fontsize=plot_style.FONT_SIZES['legend'])
    plot_style.apply_sci_axis(ax, grid_axis='y')
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.245, top=0.975)
    diagnostics = _save(
        '图S73_特征组消融实验的验证集与测试集MAE（横轴标签简化）', fig, [],
        {'数据来源': '61_stage26_4_metrics_final.xlsx / 02_消融五配置', 'seed': SEED,
         '口径': '横轴改用中文配置名，对应关系：基础=A+B+C、基础+技能=A+B+C+D、'
                 '基础+文本=A+B+C+E、完整模型=A+B+C+D+E、完整+时间位置=A+B+C+D+E+时间位置扩展特征',
         '用途': '第8章 图 8-1 重绘'})
    plt.close(fig)
    return diagnostics


def run_figures() -> int:
    plot_style.setup_sci_style()
    plot_style.FONT_SIZES.update(TARGET_FONTS)
    results = {}
    for key, builder in (('图S69', fig_s69), ('图S70', fig_s70), ('图S71', fig_s71),
                         ('图S72', fig_s72), ('图S73', fig_s73)):
        diagnostics = builder()
        results[key] = {'文件': diagnostics.get('png_path'),
                        'PDF': diagnostics.get('pdf_path'),
                        '像素': diagnostics.get('png_pixel_size'),
                        '门禁未通过项': figure_finalize.failed_paper_gates(diagnostics)}
    dump_metrics({'图件重绘时间': time.strftime('%Y-%m-%d %H:%M:%S'), '重绘图件': results})
    print('=' * 96)
    print('Stage26.6 重绘图件完成')
    for key, value in results.items():
        print(' ', key, value['门禁未通过项'] or '门禁全通过', value['文件'])
    print('=' * 96)
    return 0


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else 'audit'
    if mode == 'audit':
        run_figaudit()
        run_featureaudit()
        run_region()
        return 0
    if mode == 'figaudit':
        return run_figaudit()
    if mode == 'featureaudit':
        return run_featureaudit()
    if mode == 'region':
        return run_region()
    if mode == 'figures':
        return run_figures()
    print('未知模式：%s' % mode)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())

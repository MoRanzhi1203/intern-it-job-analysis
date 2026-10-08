# -*- coding: utf-8 -*-
"""Stage24 正文排版：生成 `outputs/paper/课程设计论文_终稿_Stage24.docx`。

复现方式：
    E:\\anaconda3\\python.exe scripts\\24a_build_stage24_docx.py            # 构建 + 自检
    E:\\anaconda3\\python.exe scripts\\24a_build_stage24_docx.py --no-com   # 不做 Word 域更新

设计要点：
1. 打开 Stage21 docx 的副本，保留其前 11 个段落（封面）与全部既有样式，
   删除第 11 段起的所有正文段落（保留 body 末尾的 sectPr）。
2. 分节：封面（无页码）→ 前置（罗马数字）→ 正文（阿拉伯数字从 1 开始）。
   最终论文**不含任何附录**（附录 A / 附录 B 一律不装配，见下方开关）。
3. 正文 18 图 / 23 表；21 个 Word 原生 OMML 独立公式（段落级公式块 + SEQ 域自动编号，
   不使用任何表格承载公式或编号）；正文段落中的数学量统一渲染为 Word 段内公式
   （m:oMath，非 m:oMathPara，Stage26.3 §11）；2 个伪代码块。
4. 只读 docs/paper/stage23/*.md、docs/paper/参考文献_Stage26.2.md、outputs/figures/**，
   不重算任何统计量。附录 A/B 的 helper 代码保留（供历史审计），
   但 INCLUDE_APPENDICES=False 时**入口不调用**，`24c_appendix_data` 亦不导入。
5. Stage26.3 结构精简与图文规范终修：正文 26 图全部按论文实际打印宽度重制
   （横排 1×2 / 1×3 子图改纵排、删除重复面板、热力图放大、图 7-1 换为建模流程图），
   表号按正文出现顺序严格递增重排并删除与表 4-2 重复的表 3-4，
   正文段落中的数学量统一转为 Word 段内公式（m:oMath），
   移除 9 个后验 / 冗余 / 近常量特征后重跑全部正式建模结果。
6. Stage26.4 终稿精简与学术化统一：图 4-3 / 4-5 / 4-6 / 4-7 / 5-1 / 5-2 重排为纵排并删除
   图内说明框，图 5-3 拆分为 5-3 与 5-4（原 5-4 顺延为 5-5），图 6-2 放大，
   图 8-5 / 8-6 按新模型（正式特征体系 A+B+C+D+E）SHAP 重绘；删除图 7-2；
   正文绝对值统一使用 Word 数学 delimiter（m:d + 竖线分隔符），不再使用普通竖线拼接。
"""
from __future__ import annotations

import importlib.util
import math
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Pt, RGBColor, Twips
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

# --------------------------------------------------------------------------- #
# 附录开关（Stage24 最终论文取消全部附录）
# --------------------------------------------------------------------------- #
INCLUDE_APPENDIX_A = False      # 附录 A：15 张附图
INCLUDE_APPENDIX_B = False      # 附录 B：12 张附表（需导入 24c_appendix_data）
INCLUDE_APPENDICES = False      # 总开关：为 False 时不写入任何附录标题/分节/内容

ROOT = Path(__file__).resolve().parents[1]
STAGE21 = ROOT / 'outputs' / 'paper' / '课程设计论文_终稿_Stage21.docx'
OUTDOCX = ROOT / 'outputs' / 'paper' / '课程设计论文_终稿_Stage24.docx'
SRC23 = ROOT / 'docs' / 'paper' / 'stage23'
REFS = ROOT / 'docs' / 'paper' / '参考文献_Stage26.2.md'
FIGROOT = ROOT / 'outputs' / 'figures'

TITLE = '基于实习僧平台的互联网 IT 实习岗位薪资影响因素分析与预测'
PORTRAIT_AVAIL = 15.5      # 21.0 - 3.0 - 2.5
LANDSCAPE_AVAIL = 24.2     # 29.7 - 3.0 - 2.5
PORTRAIT_USABLE = 24.7     # 29.7 - 2.5 - 2.5（纵向可用高度）
LANDSCAPE_USABLE = 16.0    # 21.0 - 2.5 - 2.5（横向可用高度）
CELL_MARGIN_CM = 0.100     # 单元格左右内边距合计（w:tblCellMar 28+28 dxa ≈ 0.050×2 cm）
                           # Stage26.1 版式参数：由 57+57 dxa 收紧为 28+28 dxa，
                           # 在表宽不变的前提下把列内文本区扩宽，减少长文本列折行

CHAPTERS = ['01_绪论.md', '02_相关理论与分析方法.md', '03_数据获取与预处理.md',
            '04_互联网IT实习岗位特征分析.md', '05_实习岗位薪资影响因素分析.md',
            '06_实习岗位技能需求分析.md', '07_薪资预测模型构建与结果分析.md',
            '08_模型稳健性与解释.md', '09_总结与展望.md']

# --------------------------------------------------------------------------- #
# 图 / 表 正式映射（唯一权威，取自 Stage24 任务书）
# --------------------------------------------------------------------------- #
FIG_MAIN = [
    ('3-1', '数据采集总体流程', 'supplementary/图S16_数据采集总体流程.png'),
    ('3-2', '数据治理与岗位版本重构流程', 'sci/图01_数据治理与岗位版本重构流程.png'),
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
    ('4-7', '发布时间队列的样本岗位数量变化',
     'time/01_publish_cohort_count.png'),
    ('4-8', '发布时间队列的薪资中位数及四分位区间',
     'time/02_salary_by_publish_time.png'),
    ('4-9', '核心技能需求的发布时间队列变化',
     'time/03_skill_or_category_time_structure.png'),
    ('5-1', '城市、学历与公司规模的薪资中点中位数',
     'supplementary/图S59_城市学历与公司规模薪资中点中位数.png'),
    ('5-2', '公司认证状态的薪资分布与组间比较',
     'supplementary/图S60_公司认证状态的薪资分布与组间比较.png'),
    ('5-3', '高效应福利标签的共现结构',
     'supplementary/图S62_高效应福利标签共现结构.png'),
    ('6-1', '核心技术技能需求 Top20（技能主口径，分母 8,822）', 'eda/05_tech_skill_top20.png'),
    ('6-2', '岗位细分类 × 技能命中率热力图',
     'supplementary/图S63_岗位细分类技能命中率热力图.png'),
    ('6-3', '控制岗位细分类前后的技能薪资差异变化',
     'supplementary/图S09_控制细分类前后技能薪资差异.png'),
    ('7-1', '薪资预测模型构建与评估流程',
     'supplementary/图S72_薪资预测模型构建与评估流程（节点简化）.png'),
    ('8-1', '特征组消融实验的验证集与测试集 MAE',
     'supplementary/图S73_特征组消融实验的验证集与测试集MAE（横轴标签简化）.png'),
    ('8-2', '正式主模型 SHAP 蜂群图（Top12 特征）',
     'supplementary/图S64_主模型SHAP蜂群图.png'),
    ('8-3', '技能特征 SHAP Top20（条长 = 平均绝对 SHAP 值，标记 = 技能存在时平均贡献方向）',
     'supplementary/图S68_技能特征SHAP贡献.png'),
]

# Stage26.4 版式参数：图件按论文实际打印宽度生成（见 scripts/26g_stage26_4_figure_rebuild.py），
# 因此这里给出与其生成时一致的打印宽度；未列出的图仍按 10.5 cm 排版。
# Stage26.5：新增图 4-3（省域分布地图，15.5 cm）；图 4-6 换为重绘的覆盖活跃图（15.5 cm）；
# 图 8-3 换为去样本量标注的技能 SHAP 图（11.5 cm）。
# Stage26.6：图 4-1 / 4-2 / 4-6（原图 4-8）/ 7-1 / 8-1 换为本轮重绘件，
# 图件由 23 张收敛为 18 张（见 scripts/26j_stage26_6_figures.py 与图去留审计表 70）。
FIG_WIDTH_CM = {
    '4-1': 15.5, '4-3': 15.5, '4-5': 15.5, '4-6': 13.0,
    '4-7': 15.5, '4-8': 15.5, '4-9': 15.5,
    '5-2': 15.5, '5-3': 15.5,
    '6-2': 15.5, '8-1': 15.5, '8-2': 12.5, '8-3': 11.5,
}

FIG_APP = [
    ('A-1', '技能需求的分层结构', 'eda/06_skill_layer_structure.png'),
    ('A-2', '技能提取双口径稳健性', 'eda/10_scope_robustness.png'),
    ('A-3', '正式主模型 test 预测值 vs 真实值', 'modeling/02_prediction_scatter.png'),
    ('A-4', '正式主模型 test 残差诊断', 'modeling/03_residual_diagnostics.png'),
    ('A-5', '正式主模型 test 分组误差诊断', 'modeling/04_error_groups.png'),
    ('A-6', '技能与薪资的描述性关联（未控制岗位类别）', 'eda/09_skill_salary.png'),
    ('A-7', '正式薪资样本的筛选与口径流转', 'supplementary/图S01_样本筛选与口径流转.png'),
    ('A-8', '公司属性语义槽位异常的修复构成与残留校验',
     'supplementary/图S02_公司属性语义槽位异常修复构成.png'),
    ('A-9', '数据划分与三个子集的薪资分布对照',
     'supplementary/图S03_数据划分与三子集薪资分布.png'),
    ('A-10', '技能数量档与薪资的关系', 'supplementary/图S08_技能数量档与薪资.png'),
    ('A-11', '特征组增量的配对 bootstrap 置信区间',
     'supplementary/图S11_特征组增量bootstrap置信区间.png'),
    ('A-12', 'SHAP 解释稳定性（跨随机种子）', 'supplementary/图S12_SHAP解释稳定性.png'),
    ('A-13', '技能特征阈值与文本维度的验证集表现',
     'supplementary/图S13_技能阈值与文本维度选择.png'),
    ('A-14', '每个岗位的公司福利标签数量分布',
     'supplementary/图S14_每岗福利标签数量分布.png'),
    ('A-15', '目标泄漏检查与特征组构成',
     'supplementary/图S15_目标泄漏检查与特征组构成.png'),
]

TBL_MAIN_TITLE = {
    '2-1': '四个模型的超参数候选范围',
    '3-1': '数据采集主要字段与用途',
    '3-2': '数据口径与样本构成',
    '3-3': '建模特征体系（A/B/C/D/E 五组主体特征与发布时间位置扩展特征）',
    '4-1': '薪资中点描述统计',
    '4-2': '候选发布时间段与 Strict / Relaxed 招聘周期总体特征',
    '4-3': '重招识别阈值的敏感性',
    '5-1': '公司认证状态的薪资分布与组间比较',
    '5-2': '高频公司福利标签描述统计与代表性二元比较',
    '5-3': '招聘生命周期特征与薪资的统计关联',
    '5-4': '多变量中位数回归的主要变量系数与 bootstrap 区间',
    '5-5': '薪资影响因素的多证据综合对照',
    '6-1': '技能与薪资描述性关联（摘要）',
    '7-1': '建模前特征质量诊断',
    '7-2': 'Train / Validation / Test 基本分布对照',
    '7-3': '不同模型在验证集上的预测性能',
    '7-4': '最终模型在测试集上的预测结果',
    '7-5': '分组预测误差（薪资四分位区间与岗位大类）',
    '8-1': '特征组消融实验结果与增量检验',
    '8-2': '随机划分与按公司分组划分对比',
    '8-3': '三类泛化场景对照',
    '8-4': '稳健性检查（极端值处理与目标口径）',
    '8-5': '技能特征 SHAP（技能出现口径）节选',
}

# Stage26.5：以下表格强制整表不跨页（对除末行外的所有单元格段落写入 keepNext），
# 用于修复表题与表体被拆到两页、首行孤立的版式问题。
KEEP_TOGETHER_TABLES = {'5-3', '5-5', '7-1', '7-2', '8-3', '8-4'}

FORMULAS = [
    ('2-1', r'H=\frac{12}{N(N+1)}\sum_{j=1}^{k}n_j\bar{R}_j^{2}-3(N+1)'),
    ('2-2', r'\varepsilon^{2}=\frac{H-k+1}{N-k}'),
    ('2-3', r'U_1=n_1n_2+\frac{n_1(n_1+1)}{2}-R_1'),
    ('2-4', r'\delta=P(X>Y)-P(X<Y)'),
    ('2-5', r'\delta=\frac{\#(x_i>y_j)-\#(x_i<y_j)}{n_Xn_Y}'),
    ('2-6', r'p_{(i)}\le\frac{i}{m}\alpha'),
    ('2-7', r'\mathbf{X}\approx\mathbf{U}_k\boldsymbol{\Sigma}_k\mathbf{V}_k^{\top}'),
    ('2-8', r'\min_{\boldsymbol{\beta}}\sum_{i=1}^{n}(y_i-\mathbf{x}_i^{\top}\boldsymbol{\beta})^{2}'
            r'+\lambda\|\boldsymbol{\beta}\|_2^{2}'),
    ('2-9', r'\hat{y}=\frac{1}{B}\sum_{b=1}^{B}T_b(\mathbf{x})'),
    ('2-10', r'\hat{y}_i=\sum_{m=1}^{M}f_m(\mathbf{x}_i),\ f_m\in\mathcal{F}'),
    ('2-11', r'\mathrm{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|'),
    ('2-12', r'\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^{2}}'),
    ('2-13', r'R^{2}=1-\frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^{2}}{\sum_{i=1}^{n}(y_i-\bar{y})^{2}}'),
    ('2-14', r'f(\mathbf{x})=\phi_0+\sum_{j=1}^{p}\phi_j'),
]

# =========================================================================== #
# 一、LaTeX 子集 → OMML（Word 原生公式）
# =========================================================================== #
CMD_SYM = {
    'varepsilon': 'ε', 'phi': 'φ', 'varphi': 'φ', 'lambda': 'λ', 'alpha': 'α',
    'Delta': 'Δ',
    'delta': 'δ', 'beta': 'β', 'Sigma': 'Σ', 'top': '⊤', 'le': '≤', 'ge': '≥',
    'cdots': '⋯', 'approx': '≈', 'in': '∈', 'cdot': '·', 'times': '×',
    'theta': 'θ', 'mu': 'μ', 'sigma': 'σ', 'eta': 'η', 'rho': 'ρ',
    '#': '#', '|': '‖', 'mid': '∣', '%': '%', '{': '{', '}': '}', '&': '&',
    ',': ' ', ';': ' ', ' ': ' ',
}

# --------------------------------------------------------------------------- #
# 段内公式规则（Stage26.3 §11）：正文段落中的数学量统一渲染为 Word 段内公式
#   （m:oMath，非 m:oMathPara）。规则与 `outputs/tables/55_stage26_3_inline_math_inventory.xlsx`
#   的扫描口径一一对应，规则按“最长匹配优先”应用，避免下标/上标被拆散。
# --------------------------------------------------------------------------- #
GREEK = r'ε²|ε\^2|R²|R\^2|ΔMAE|φ_\{?[0-9a-zA-Z]\}?|\|φ_\{?[0-9a-zA-Z]\}?\|' \
        r'|ρ|α|β|λ|δ|Δ'
SUBSCRIPT_LETTERS = 'A-Za-zRrNnOoStUuqpGgDdCcHhkKmXxYy'
SINGLE_LETTERS = 'NkHRXYMBFnmpqxiC'

INLINE_MATH_RULES = [
    # 阈值比较：整套写成公式
    (re.compile(r'\|δ\|\s*小于\s*0\.147'), lambda m: r'\abs{\delta}<0.147'),
    (re.compile(r'\|ρ\|\s*[<>=]\s*0\.85'), lambda m: r'\abs{\rho}>0.85'),
    (re.compile(r'(q|p)\s*值?\s*([<>=])\s*0\.05'), lambda m: '%s%s0.05' % (m.group(1), m.group(2))),
    # 绝对值包裹的效应量：统一走 \abs{}（= Word 数学 delimiter m:d，非普通竖线）
    (re.compile(r'\|Cliff[\'’]s\s*δ\|'), lambda m: r"\abs{\mathrm{Cliff's}\ \delta}"),
    (re.compile(r'\|Spearman\s*ρ\|'), lambda m: r'\abs{\mathrm{Spearman}\ \rho}'),
    (re.compile(r'\|φ_\{?([0-9a-zA-Z])\}?\|'), lambda m: r'\abs{\varphi_{%s}}' % m.group(1)),
    (re.compile(r'\|δ\|'), lambda m: r'\abs{\delta}'),
    (re.compile(r'\|ρ\|'), lambda m: r'\abs{\rho}'),
    (re.compile(r'\|ε²\|'), lambda m: r'\abs{\varepsilon^{2}}'),
    # 希腊字母与统计量
    (re.compile(r'ε²|ε\^2'), lambda m: r'\varepsilon^{2}'),
    (re.compile(r'R²|R\^2'), lambda m: r'R^{2}'),
    (re.compile(r'ΔMAE'), lambda m: r'\Delta\mathrm{MAE}'),
    (re.compile(r'φ_\{?([0-9a-zA-Z])\}?'), lambda m: r'\varphi_{%s}' % m.group(1)),
    (re.compile(r'ρ'), lambda m: r'\rho'),
    (re.compile(r'α'), lambda m: r'\alpha'),
    (re.compile(r'β'), lambda m: r'\beta'),
    (re.compile(r'λ'), lambda m: r'\lambda'),
    (re.compile(r'δ'), lambda m: r'\delta'),
    (re.compile(r'Δ'), lambda m: r'\Delta'),
    # 帽符号与均值符号
    (re.compile(r'ŷ_\{?([0-9a-zA-Z])\}?'), lambda m: r'\hat{y}_{%s}' % m.group(1)),
    (re.compile(r'ŷ'), lambda m: r'\hat{y}'),
    (re.compile(r'ȳ'), lambda m: r'\bar{y}'),
    (re.compile(r'R̄_\{?([0-9a-zA-Z])\}?'), lambda m: r'\bar{R}_{%s}' % m.group(1)),
    (re.compile(r'R̄'), lambda m: r'\bar{R}'),
    # 长单词下标（如 ReopenRate_t）
    (re.compile(r'ReopenRate_t'), lambda m: r'\mathrm{ReopenRate}_{t}'),
    # 带上下标的数学量
    (re.compile(r'([%s])_\{?([A-Za-z0-9,]+)\}?\^\(([0-9a-zA-Z])\)' % SUBSCRIPT_LETTERS),
     lambda m: '%s_{%s}^{(%s)}' % (m.group(1), m.group(2), m.group(3))),
    (re.compile(r'(?<![A-Za-z0-9_])([%s])_\{?([A-Za-z0-9,−+\-]+)\}?(?![A-Za-z0-9])'
                % SUBSCRIPT_LETTERS),
     lambda m: '%s_{%s}' % (m.group(1), m.group(2).replace('−', '-'))),
    (re.compile(r'(?<![A-Za-z0-9_])([%s])_\(([0-9a-zA-Z]{1,3})\)' % SUBSCRIPT_LETTERS),
     lambda m: '%s_{(%s)}' % (m.group(1), m.group(2))),
    # 单字母数学变量（后接 为/表示/与/和/的/值/取/大于/小于/比较符）
    (re.compile(r'(?<![A-Za-z0-9_])([%s])(?=\s*(?:为|表示|与|和|的|值|取|大于|小于|[=<>]))'
                % SINGLE_LETTERS), lambda m: m.group(1)),
]

# §11.2：数据字段名与程序参数名保留普通文本，不参与段内公式转换
PROTECTED_NAMES = [
    'n_estimators', 'learning_rate', 'num_leaves', 'max_features', 'max_depth',
    'min_samples_leaf', 'random_state', 'intern_id', 'publish_month',
    'publish_weekday', 'initial_observed_deadline', 'final_observed_deadline',
    'CORE_SIGNATURE', 'FULL_SIGNATURE', 'FULL_TEXT_FALLBACK',
    'REQUIREMENT_SECTION', 'EMPTY_TEXT', 'company_entity_id',
    # Stage26.8 §19：特征组组合名整体保留普通正文，避免行尾 C 被段内公式规则
    # 误判为单字母数学变量（原 Stage26.6 Word 出现“A+B+ 数学斜体C”共 3 处）。
    'A+B+C+D+E', 'A+B+C+D', 'A+B+C+E', 'A+B+C',
]
PROTECTED_RE = re.compile('|'.join(re.escape(name) for name in PROTECTED_NAMES))


def inline_math_spans(text: str):
    """返回正文中需要转为段内公式的片段 (start, end, latex)，按最长匹配优先。"""
    candidates = []
    for pattern, build in INLINE_MATH_RULES:
        for match in pattern.finditer(text):
            latex = build(match)
            candidates.append((-(match.end() - match.start()), match.start(), match.end(), latex))
    candidates.sort()
    protected = [(m.start(), m.end()) for m in PROTECTED_RE.finditer(text)]
    chosen, taken = [], []
    for _, start, end, latex in candidates:
        if any(not (end <= p0 or start >= p1) for p0, p1 in protected):
            continue
        if any(not (end <= s0 or start >= s1) for s0, s1 in taken):
            continue
        taken.append((start, end))
        chosen.append((start, end, latex))
    chosen.sort()
    return chosen


def _xml_esc(t: str) -> str:
    return (t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


def _tok(s: str):
    toks, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c == '\\':
            j = i + 1
            if j < n and s[j].isalpha():
                while j < n and s[j].isalpha():
                    j += 1
                toks.append(('cmd', s[i + 1:j]))
                i = j
            elif j < n:
                toks.append(('cmd', s[j]))
                i = j + 1
            else:
                i += 1
        elif c in '{}^_':
            toks.append((c, c))
            i += 1
        elif c in '()':
            toks.append(('ch', c))
            i += 1
        elif c == ' ':
            toks.append(('sp', ' '))
            i += 1
        else:
            toks.append(('ch', c))
            i += 1
    return toks


class _P:
    def __init__(self, toks):
        self.t = toks
        self.i = 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def next(self):
        t = self.peek()
        self.i += 1
        return t

    # ---- 序列 ---------------------------------------------------------- #
    def seq(self, stop_paren=False):
        items = []
        while True:
            t = self.peek()
            if t is None or t[0] == '}':
                break
            if stop_paren and t[0] == 'ch' and t[1] == ')':
                break
            if t[0] in '^_':
                self.next()
                base = items.pop() if items else ('sym', '')
                arg = self.atom()
                items.append(self.bind(base, t[0], arg))
                continue
            items.append(self.atom())
        return items

    @staticmethod
    def bind(base, kind, arg):
        if kind == '^':
            if base[0] == 'sub':
                return ('subsup', base[1], base[2], arg)
            return ('sup', base, arg)
        if base[0] == 'sup':
            return ('subsup', base[1], arg, base[2])
        return ('sub', base, arg)

    def group(self):
        self.next()                       # '{'
        inner = self.seq()
        if self.peek() and self.peek()[0] == '}':
            self.next()
        return ('grp', inner)

    # ---- 原子 ---------------------------------------------------------- #
    def atom(self):
        t = self.next()
        if t is None:
            return ('sym', '')
        if t[0] == '{':
            self.i -= 1
            return self.group()
        if t[0] == 'sp':
            return ('sym', ' ')
        if t[0] == 'ch':
            if t[1] == '(':
                inner = self.seq(stop_paren=True)
                if self.peek() and self.peek()[0] == 'ch' and self.peek()[1] == ')':
                    self.next()
                return ('delim', inner)
            if t[1] == ')':
                return ('sym', ')')
            return ('sym', t[1])
        if t[0] == 'cmd':
            return self.cmd(t[1])
        return ('sym', '')

    def _sub_sup(self):
        sub = sup = None
        while self.peek() and self.peek()[0] in '^_':
            kind = self.next()[0]
            val = self.atom()
            if kind == '_':
                sub = val
            else:
                sup = val
        return sub, sup

    def cmd(self, name):
        if name == 'frac':
            num = self.atom()
            den = self.atom()
            return ('frac', num, den)
        if name in ('sum', 'prod', 'int'):
            chr_ = {'sum': '∑', 'prod': '∏', 'int': '∫'}[name]
            sub, sup = self._sub_sup()
            body = self.atom()
            return ('nary', chr_, sub, sup, body)
        if name == 'sqrt':
            return ('rad', self.atom())
        if name in ('bar', 'hat', 'vec', 'tilde'):
            chr_ = {'bar': '̄', 'hat': '̂', 'vec': '⃗', 'tilde': '̃'}[name]
            return ('acc', chr_, self.atom())
        if name == 'mathbf':
            return ('sty', 'b', self.atom())
        if name == 'boldsymbol':
            return ('sty', 'bi', self.atom())
        if name == 'mathrm':
            return ('sty', 'p', self.atom())
        if name == 'abs':
            # Stage26.4 §11.1：绝对值必须使用真正的 Word 数学 delimiter（m:d + 竖线分隔符），
            # 不能把普通竖线字符放进数学 run。
            return ('absdelim', self.atom())
        if name == 'norm':
            # Stage26.5 §6：向量范数同样使用 m:d 包围（begChr/endChr = ‖），
            # 避免上下标绑定到右竖线上导致 ‖β‖₂² 被错排。
            return ('normdelim', self.atom())
        if name == 'mathcal':
            return self.atom()
        if name in ('min', 'max', 'arg', 'log', 'exp', 'ln', 'sup', 'inf'):
            sub, _ = self._sub_sup()
            if sub is None:
                return ('sym', name)
            return ('limlow', name, sub)
        if name in CMD_SYM:
            return ('sym', CMD_SYM[name])
        return ('sym', name)


MATH_SIZE = {'half_points': None, 'bold': False}


def _run(text, sty=None):
    if text == '':
        return ''
    rpr = ('<m:rPr><m:sty m:val="%s"/></m:rPr>' % sty) if sty else ''
    half = MATH_SIZE['half_points']
    size_xml = ('<w:sz w:val="%d"/><w:szCs w:val="%d"/>' % (half, half)) if half else ''
    if MATH_SIZE['bold']:
        size_xml += '<w:b/><w:bCs/>'
    return ('<m:r>%s<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/>'
            '%s</w:rPr><m:t xml:space="preserve">%s</m:t></m:r>'
            % (rpr, size_xml, _xml_esc(text)))


def _render(node, sty=None):
    kind = node[0]
    if kind == 'sym':
        return _run(node[1], sty)
    if kind == 'grp':
        return ''.join(_render(c, sty) for c in node[1])
    if kind == 'sty':
        return _render(node[2], node[1])
    if kind == 'frac':
        return ('<m:f><m:fPr/><m:num>%s</m:num><m:den>%s</m:den></m:f>'
                % (_render(node[1], sty), _render(node[2], sty)))
    if kind == 'sup':
        return ('<m:sSup><m:sSupPr/><m:e>%s</m:e><m:sup>%s</m:sup></m:sSup>'
                % (_render(node[1], sty), _render(node[2], sty)))
    if kind == 'sub':
        return ('<m:sSub><m:sSubPr/><m:e>%s</m:e><m:sub>%s</m:sub></m:sSub>'
                % (_render(node[1], sty), _render(node[2], sty)))
    if kind == 'subsup':
        return ('<m:sSubSup><m:sSubSupPr/><m:e>%s</m:e><m:sub>%s</m:sub>'
                '<m:sup>%s</m:sup></m:sSubSup>'
                % (_render(node[1], sty), _render(node[2], sty), _render(node[3], sty)))
    if kind == 'nary':
        _, chr_, sub, sup, body = node
        return ('<m:nary><m:naryPr><m:chr m:val="%s"/><m:limLoc m:val="undOvr"/>'
                '<m:grow m:val="1"/></m:naryPr><m:sub>%s</m:sub><m:sup>%s</m:sup>'
                '<m:e>%s</m:e></m:nary>'
                % (_xml_esc(chr_),
                   _render(sub, sty) if sub else '',
                   _render(sup, sty) if sup else '',
                   _render(body, sty)))
    if kind == 'rad':
        return ('<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg/>'
                '<m:e>%s</m:e></m:rad>' % _render(node[1], sty))
    if kind == 'acc':
        return ('<m:acc><m:accPr><m:chr m:val="%s"/></m:accPr><m:e>%s</m:e></m:acc>'
                % (_xml_esc(node[1]), _render(node[2], sty)))
    if kind == 'delim':
        return ('<m:d><m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr>'
                '<m:e>%s</m:e></m:d>' % ''.join(_render(c, sty) for c in node[1]))
    if kind == 'absdelim':
        return ('<m:d><m:dPr><m:begChr m:val="|"/><m:endChr m:val="|"/>'
                '<m:grow m:val="1"/></m:dPr><m:e>%s</m:e></m:d>'
                % _render(node[1], sty))
    if kind == 'normdelim':
        return ('<m:d><m:dPr><m:begChr m:val="‖"/><m:endChr m:val="‖"/>'
                '<m:grow m:val="1"/></m:dPr><m:e>%s</m:e></m:d>'
                % _render(node[1], sty))
    if kind == 'limlow':
        return ('<m:limLow><m:limLowPr/><m:e>%s</m:e><m:lim>%s</m:lim></m:limLow>'
                % (_render(('sym', node[1]), sty), _render(node[2], sty)))
    raise ValueError('unknown node %r' % (node,))


def latex_to_omath(latex: str, half_points=None, bold=False):
    """把受限 LaTeX 子集转换为 OMML 元素（m:oMath，段内 / 段落级通用）。"""
    MATH_SIZE['half_points'] = half_points
    MATH_SIZE['bold'] = bold
    try:
        body = _render(('grp', _P(_tok(latex)).seq()))
    finally:
        MATH_SIZE['half_points'] = None
        MATH_SIZE['bold'] = False
    xml = '<m:oMath %s>%s</m:oMath>' % (nsdecls('m', 'w'), body)
    return parse_xml(xml)


# =========================================================================== #
# 二、通用格式辅助
# =========================================================================== #
SECT_ORDER = ['headerReference', 'footerReference', 'footnotePr', 'endnotePr', 'type',
              'pgSz', 'pgMar', 'paperSrc', 'pgBorders', 'lnNumType', 'pgNumType', 'cols',
              'formProt', 'vAlign', 'noEndnote', 'titlePg', 'textDirection', 'bidi',
              'rtlGutter', 'docGrid', 'printerSettings', 'sectPrChange']

PPR_ORDER = ['pStyle', 'keepNext', 'keepLines', 'pageBreakBefore', 'framePr', 'widowControl',
             'numPr', 'suppressLineNumbers', 'pBdr', 'shd', 'tabs', 'suppressAutoHyphens',
             'kinsoku', 'wordWrap', 'overflowPunct', 'topLinePunct', 'autoSpaceDE',
             'autoSpaceDN', 'bidi', 'adjustRightInd', 'snapToGrid', 'spacing', 'ind',
             'contextualSpacing', 'mirrorIndents', 'suppressOverlap', 'jc', 'textDirection',
             'textAlignment', 'textboxTightWrap', 'outlineLvl', 'divId', 'cnfStyle', 'rPr',
             'sectPr', 'pPrChange']

TBL_ORDER = ['tblStyle', 'tblpPr', 'tblOverlap', 'bidiVisual', 'tblStyleRowBandSize',
             'tblStyleColBandSize', 'tblW', 'jc', 'tblCellSpacing', 'tblInd', 'tblBorders',
             'shd', 'tblLayout', 'tblCellMar', 'tblLook', 'tblCaption', 'tblDescription']


def insert_sect_el(sect_pr, el):
    tag = el.tag.split('}')[-1]
    idx = SECT_ORDER.index(tag)
    for child in sect_pr:
        ct = child.tag.split('}')[-1]
        if ct in SECT_ORDER and SECT_ORDER.index(ct) > idx:
            child.addprevious(el)
            return
    sect_pr.append(el)


def order_tbl_pr(pr):
    def key(el):
        tag = el.tag.split('}')[-1]
        return TBL_ORDER.index(tag) if tag in TBL_ORDER else len(TBL_ORDER)
    for el in sorted(list(pr), key=key):
        pr.append(el)


def style_font(style, ascii_font, ea_font, size, bold=False):
    style.font.name = ascii_font
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor(0, 0, 0)
    rpr = style.element.get_or_add_rPr()
    rf = rpr.find(qn('w:rFonts'))
    if rf is None:
        rf = OxmlElement('w:rFonts')
        rpr.append(rf)
    rf.set(qn('w:ascii'), ascii_font)
    rf.set(qn('w:hAnsi'), ascii_font)
    rf.set(qn('w:eastAsia'), ea_font)


def set_latin_word_wrap(style):
    """写 w:wordWrap = 1：禁止西文/数字在“词内”断行。

    Stage24 逐页视觉 QA 发现：设为 0（允许词内断行）时，长数字会被拆到两行
    （如 52.049937 → 52.049 / 937）。置 1 后长数字整体移到下一行。
    """
    ppr = style.element.get_or_add_pPr()
    for el in ppr.findall(qn('w:wordWrap')):
        ppr.remove(el)
    el = OxmlElement('w:wordWrap')
    el.set(qn('w:val'), '1')
    tag = 'wordWrap'
    idx = PPR_ORDER.index(tag)
    for child in ppr:
        ct = child.tag.split('}')[-1]
        if ct in PPR_ORDER and PPR_ORDER.index(ct) > idx:
            child.addprevious(el)
            return
    ppr.append(el)


def add_field(paragraph, instr, placeholder=''):
    run = paragraph.add_run()
    b = OxmlElement('w:fldChar'); b.set(qn('w:fldCharType'), 'begin')
    it = OxmlElement('w:instrText'); it.set(qn('xml:space'), 'preserve'); it.text = instr
    s = OxmlElement('w:fldChar'); s.set(qn('w:fldCharType'), 'separate')
    t = OxmlElement('w:t'); t.text = placeholder
    e = OxmlElement('w:fldChar'); e.set(qn('w:fldCharType'), 'end')
    for el in (b, it, s, t, e):
        run._r.append(el)
    return run


def text_width_twips(doc):
    """正文版心宽度（twips）= 页宽 - 左页边距 - 右页边距（纵向 15.5 cm → 8788）。"""
    sec = doc.sections[-1]
    emu = int(sec.page_width) - int(sec.left_margin) - int(sec.right_margin)
    return int(round(emu / 635.0))


def add_seq_field(paragraph, identifier='eq', size=12, placeholder='1'):
    """追加 `SEQ <identifier> \\* ARABIC` 域：公式序号由 Word 自动生成。

    先建 run 并写入 w:rPr，再追加域的 5 个 run 级元素，保证 w:rPr 位于
    w:r 首位（合法 OOXML 顺序）。
    """
    run = paragraph.add_run()
    run.font.name = 'Times New Roman'
    run.font.size = Pt(size)
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    b = OxmlElement('w:fldChar'); b.set(qn('w:fldCharType'), 'begin')
    it = OxmlElement('w:instrText'); it.set(qn('xml:space'), 'preserve')
    it.text = 'SEQ %s \\* ARABIC' % identifier
    s = OxmlElement('w:fldChar'); s.set(qn('w:fldCharType'), 'separate')
    t = OxmlElement('w:t'); t.text = placeholder
    e = OxmlElement('w:fldChar'); e.set(qn('w:fldCharType'), 'end')
    for el in (b, it, s, t, e):
        run._r.append(el)
    return run


def set_page(sec, landscape=False):
    if landscape:
        sec.orientation = WD_ORIENT.LANDSCAPE
        sec.page_width, sec.page_height = Cm(29.7), Cm(21.0)
    else:
        sec.orientation = WD_ORIENT.PORTRAIT
        sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.top_margin = sec.bottom_margin = Cm(2.5)
    sec.left_margin, sec.right_margin = Cm(3.0), Cm(2.5)
    sec.header_distance = sec.footer_distance = Cm(1.5)


def set_pgnum(sec, fmt, start=None):
    sect_pr = sec._sectPr
    for el in sect_pr.findall(qn('w:pgNumType')):
        sect_pr.remove(el)
    if fmt is None:
        return
    pg = OxmlElement('w:pgNumType')
    pg.set(qn('w:fmt'), fmt)
    if start is not None:
        pg.set(qn('w:start'), str(start))
    insert_sect_el(sect_pr, pg)


def own_footer(sec, roman=False):
    sec.footer.is_linked_to_previous = True
    sec.footer.is_linked_to_previous = False
    p = sec.footer.paragraphs[0]
    p.text = ''
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    if roman is None:
        return p
    add_field(p, 'PAGE \\* ROMAN' if roman else 'PAGE', 'I' if roman else '1')
    for r in p.runs:
        r.font.size = Pt(10.5)
        r.font.name = 'Times New Roman'
        r._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    return p


def blank_footer(sec):
    sec.footer.is_linked_to_previous = True
    sec.footer.is_linked_to_previous = False
    for p in sec.footer.paragraphs:
        p.text = ''
        p.paragraph_format.first_line_indent = Pt(0)


def disp_len(s):
    return sum(2 if ord(c) > 0x2E80 else 1 for c in str(s))


# =========================================================================== #
# 三、构建器
# =========================================================================== #
class Builder:
    def __init__(self):
        self.doc = Document(str(STAGE21))
        self.stats = {
            'fig_main': 0, 'fig_app': 0, 'tbl_main': 0, 'tbl_app': 0,
            'tbl_algo': 0, 'tbl_formula': 0, 'equations': 0, 'algorithms': 0,
            'headings': {1: 0, 2: 0, 3: 0}, 'equations_native': 0,
            'inline_math': 0,
        }
        self.chapter_heads = []
        self.appendix_rows = []
        self.missing_figs = []
        normal_size = self.doc.styles['Normal'].font.size
        self.math_half_points = int(round(normal_size.pt * 2)) if normal_size else 24
        self._prepare_document()

    # ---------------------------------------------------------------- 文档
    def _prepare_document(self):
        doc = self.doc
        body = doc.element.body
        children = list(body)
        sentinel = None
        for el in reversed(children):
            if el.tag == qn('w:sectPr'):
                sentinel = el
                break
        assert sentinel is not None, 'Stage21 docx 缺少 body 级 sectPr'
        kept = 0
        for el in children:
            if el is sentinel:
                break
            if el.tag == qn('w:p'):
                kept += 1
                if kept > 11:
                    body.remove(el)
            else:
                body.remove(el)
        assert kept >= 11, 'Stage21 docx 封面段落不足 11 个（实际 %d）' % kept
        assert len(doc.sections) == 2, '分节数异常：%d' % len(doc.sections)
        # 样式
        st = doc.styles
        for name in ('Normal', 'Heading 1', 'Heading 2', 'Heading 3', 'Title',
                     'Figure Caption', 'Table Caption', 'Table Text', 'Reference',
                     'Body List', 'toc 1', 'toc 2', 'toc 3'):
            set_latin_word_wrap(st[name])
        # Stage26.1 版式参数：目录条目（toc 1/2/3，共 100+ 条）由继承 Normal 的
        # 1.5 倍行距改为单倍行距并清零段间距，使目录由 3 页收为 2 页；条目层级、
        # 文字与域内容均不变。
        for name in ('toc 1', 'toc 2', 'toc 3'):
            tpf = st[name].paragraph_format
            tpf.line_spacing = 1.0
            tpf.space_before = Pt(0)
            tpf.space_after = Pt(0)
        for name in ('Heading 1', 'Heading 2', 'Heading 3'):
            st[name].paragraph_format.keep_with_next = True
            st[name].paragraph_format.widow_control = True
        st['Heading 1'].paragraph_format.page_break_before = True
        st['Normal'].paragraph_format.widow_control = True
        if 'Appendix Text' not in [s.name for s in st]:
            aps = st.add_style('Appendix Text', 1)
            aps.base_style = st['Normal']
            style_font(aps, 'Times New Roman', '宋体', 9)
            pf = aps.paragraph_format
            pf.first_line_indent = Pt(0)
            pf.line_spacing = 1.0
            pf.space_before = Pt(0)
            pf.space_after = Pt(0)
            set_latin_word_wrap(aps)
        # 封面页脚空白
        blank_footer(doc.sections[0])

    # ------------------------------------------------------------ 段落工具
    def para(self, text='', style='Normal', indent=True, align=None, size=None,
             keep_next=False, keep_together=False, bold_all=False, first_indent=None):
        p = self.doc.add_paragraph(style=style)
        pf = p.paragraph_format
        if first_indent is not None:
            pf.first_line_indent = first_indent
        elif not indent:
            pf.first_line_indent = Pt(0)
        if align is not None:
            p.alignment = align
        if keep_next:
            pf.keep_with_next = True
        if keep_together:
            pf.keep_together = True
        if text:
            self._runs(p, text, size=size, bold_all=bold_all, ea='宋体', math=True)
        return p

    def _plain_runs(self, p, text, size=None, bold=False, ea='宋体',
                    ascii_font='Times New Roman'):
        """写普通文本 run（不解析段内公式）。"""
        if not text:
            return
        r = p.add_run(text)
        r.font.name = ascii_font
        r._element.rPr.rFonts.set(qn('w:eastAsia'), ea)
        if size:
            r.font.size = Pt(size)
        if bold:
            r.bold = True

    def _runs(self, p, text, size=None, bold_all=False, ea='宋体',
              ascii_font='Times New Roman', math=False):
        """写一段文本：`**加粗**` 与（math=True 时）段内数学片段。

        Stage26.3 §11：段落中的数学量渲染为 Word 段内公式对象（m:oMath）；
        数学片段之外的部分仍为普通文本 run，段内公式不使用任何表格承载。
        """
        for seg in re.split(r'(\*\*.+?\*\*)', text):
            if not seg:
                continue
            bold = seg.startswith('**') and seg.endswith('**') and len(seg) > 4
            body = seg[2:-2] if bold else seg
            if not body:
                continue
            if not math:
                self._plain_runs(p, body, size=size, bold=bold or bold_all, ea=ea,
                                 ascii_font=ascii_font)
                continue
            half = int(round(size * 2)) if size else self.math_half_points
            bold_math = bool(bold or bold_all)
            cursor = 0
            for start, end, latex in inline_math_spans(body):
                self._plain_runs(p, body[cursor:start], size=size,
                                 bold=bold_math, ea=ea, ascii_font=ascii_font)
                p._p.append(latex_to_omath(latex, half, bold_math))
                self.stats['inline_math'] += 1
                cursor = end
            self._plain_runs(p, body[cursor:], size=size, bold=bold_math, ea=ea,
                             ascii_font=ascii_font)
        return p

    def heading(self, level, text):
        p = self.doc.add_paragraph(style='Heading %d' % level)
        pf = p.paragraph_format
        pf.first_line_indent = Pt(0)
        pf.keep_with_next = True
        # Stage26.1 版式参数：标题段前后间距由模板默认值（H1 18/12 pt、H2/H3 12/6 pt）
        # 收紧为 H1 0/6 pt、H2/H3 6/3 pt：一级标题因强制另起页，页顶的 18 pt 间距
        # 完全是空白浪费；二级/三级标题共 91 处，收紧后不影响层级可读性。
        sp_before, sp_after = {1: (0, 6), 2: (6, 3)}.get(level, (6, 3))
        pf.space_before = Pt(sp_before)
        pf.space_after = Pt(sp_after)
        style_size = self.doc.styles['Heading %d' % level].font.size
        if style_size:
            self._runs(p, text, size=style_size.pt, ea='黑体', ascii_font='Times New Roman',
                       math=True)
        else:
            self._runs(p, text, ea='黑体', ascii_font='Times New Roman', math=True)
        self.stats['headings'][level] += 1
        if level == 1:
            self.chapter_heads.append(text)
        return p

    # -------------------------------------------------------------- 图片
    def figure(self, path, caption, size_pt=10.5, width_cm=None):
        if not path.exists():
            self.missing_figs.append(str(path))
        with Image.open(str(path)) as im:
            w_px, h_px = im.size
        # Stage25 逐页视觉 QA：图片宽度由 14.8 cm 收紧到 13.5 cm（仍占版心 15.5 cm 的
        # 87%），使第 6 章中"图 + 图题"整块能落到前一页页底，消除页底 9~12 cm 留白与
        # 章末 3 行溢页（物理页 p59 近空白页）。
        # Stage26.1 版式参数：本轮正文由 98 页增至 103 页（新增 表 3-4 并扩充
        # 表 4-2 / 7-4 / 8-1 / 8-5），为在不改全局字号与页边距、不删正文的前提下
        # 通过"总页数 < 100"门禁，图片宽度再由 13.5 cm 收紧为 10.5 cm
        # （占版心 68%），同时缩小图块上下间距；图形仍为 600 dpi 原图缩放，不失真。
        width_cm = width_cm or 10.5
        if width_cm * h_px / float(w_px) > 18.0:
            width_cm = 18.0 * w_px / float(h_px)
        p = self.doc.add_paragraph()
        pf = p.paragraph_format
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf.first_line_indent = Pt(0)
        # 单倍行距：Normal 为 1.5 倍行距，含内嵌图片的行会被放大到 1.5×图高，
        # 导致图片块整体下移、页底出现大片留白（Stage24 逐页视觉 QA 修复项）。
        pf.line_spacing = 1.0
        pf.space_before = Pt(1)
        pf.space_after = Pt(1)
        pf.keep_with_next = True
        pf.keep_together = True
        p.add_run().add_picture(str(path), width=Cm(width_cm))
        for dp in p._p.findall('.//' + qn('wp:docPr')):
            dp.set('descr', caption)
            dp.set('name', caption)
        cap = self.doc.add_paragraph(style='Figure Caption')
        cap.paragraph_format.first_line_indent = Pt(0)
        cap.paragraph_format.keep_together = True
        # Stage26.1 版式参数：图题上下间距由模板默认 6/12 pt 收紧为 2/4 pt
        cap.paragraph_format.space_before = Pt(2)
        cap.paragraph_format.space_after = Pt(4)
        self._runs(cap, caption, size=size_pt, math=True)
        return p

    # -------------------------------------------------------------- 表格
    def _cell_write(self, cell, text, size_pt, align, bold=False):
        para = cell.paragraphs[0]
        para.style = self.doc.styles['Appendix Text' if size_pt <= 9 else 'Table Text']
        pf = para.paragraph_format
        pf.first_line_indent = Pt(0)
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)
        if size_pt <= 9:
            # 附录长表与正文紧凑表用固定行距，保证密度与可读性
            # Stage26.1 版式参数：正文表内文字固定行距由 10.5 pt 收紧为 9.2 pt
            pf.line_spacing = Pt(9.2)
        para.alignment = align
        if text == '':
            return
        self._runs(para, text, size=size_pt, bold_all=bold, math=True)
        if bold:
            for r in para.runs:
                r.bold = True

    def data_table(self, grid, caption=None, size_pt=10.5, avail=PORTRAIT_AVAIL,
                   tag='main', header_repeat=True, cant_split=True, cap_size=None,
                   keep_together=False):
        """写一张三线表。返回 (table, n_rows)。"""
        if caption:
            cp = self.doc.add_paragraph(style='Table Caption')
            cp.paragraph_format.first_line_indent = Pt(0)
            cp.paragraph_format.keep_with_next = True
            cp.paragraph_format.keep_together = True
            # Stage26.1 版式参数：表题上下间距由模板默认 12/6 pt 收紧为 4/2 pt
            cp.paragraph_format.space_before = Pt(4)
            cp.paragraph_format.space_after = Pt(2)
            self._runs(cp, caption, size=cap_size or size_pt, math=True)
        ncol = len(grid[0])
        table = self.doc.add_table(rows=0, cols=ncol)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        cm_w, numeric = alloc_widths(grid, avail, size_pt)
        for j in range(ncol):
            table.columns[j].width = Cm(cm_w[j])      # 关键：写入 w:tblGrid，固定版式按此渲染
        for i, grid_row in enumerate(grid):
            trow = table.add_row()
            cells = trow.cells
            for j in range(ncol):
                val = str(grid_row[j]) if j < len(grid_row) else ''
                cell = cells[j]
                cell.width = Cm(cm_w[j])
                if i == 0:
                    align = WD_ALIGN_PARAGRAPH.CENTER
                elif numeric[j]:
                    align = WD_ALIGN_PARAGRAPH.RIGHT
                else:
                    align = WD_ALIGN_PARAGRAPH.LEFT
                self._cell_write(cell, val, size_pt, align, bold=(i == 0))
                va = OxmlElement('w:vAlign'); va.set(qn('w:val'), 'center')
                cell._tc.get_or_add_tcPr().append(va)
            if cant_split:
                trow._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
        if keep_together and len(table.rows) > 1:
            # Stage26.5 §33：对除末行外的每一行写入 keepNext，使整张表与表题同页。
            for trow in table.rows[:-1]:
                for cell in trow.cells:
                    for para in cell.paragraphs:
                        para.paragraph_format.keep_with_next = True
        if header_repeat and len(table.rows) > 1:
            table.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
        pin_table(table, sum(cm_w), mode='three_line',
                  header_sz=6 if size_pt > 9 else 4,
                  top_sz=12 if size_pt > 9 else 8, bottom_sz=12 if size_pt > 9 else 8)
        return table

    # -------------------------------------------------------------- 公式
    def equation(self, no, latex):
        """Word 原生公式块 + SEQ 域自动编号（不使用任何表格）。

        段落内元素顺序（Word“公式 + 编号”的制表位原生机制）：
            <w:p><w:pPr><w:tabs>
                    <w:tab w:val="center" w:pos="4394"/>   # 版心中点
                    <w:tab w:val="right"  w:pos="8788"/>   # 版心右边界
                  </w:tabs></w:pPr>
              <w:r><w:tab/></w:r>                         # 跳到居中制表位 → 公式居中
              <m:oMath>…</m:oMath>                        # 与改造前同源的 OMML（可编辑）
              <w:r><w:tab/></w:r>                         # 跳到右制表位 → 编号右对齐
              <w:r><w:t>(</w:t></w:r>                     # 编号圆括号左
              SEQ eq \\* ARABIC 域                        # 序号由 SEQ 域自动生成（全书连续）
              <w:r><w:t>)</w:t></w:r></w:p>

        Stage26.1 起编号显示形式为全书连续的 (1)、(2)、(3)……：段落内不再写入任何
        固定章节前缀（如 "2-"），编号完全由 SEQ eq \\* ARABIC 域按文档顺序生成。
        """
        p = self.doc.add_paragraph()          # 正文级段落：不在任何表格内
        pf = p.paragraph_format
        pf.first_line_indent = Pt(0)
        pf.space_before = Pt(2)               # Stage26.1 版式参数：由 4 pt 收紧为 2 pt
        pf.space_after = Pt(2)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        tw = text_width_twips(self.doc)
        pf.tab_stops.add_tab_stop(Twips(tw // 2), WD_TAB_ALIGNMENT.CENTER)
        pf.tab_stops.add_tab_stop(Twips(tw), WD_TAB_ALIGNMENT.RIGHT)
        r = p.add_run()
        r._r.append(OxmlElement('w:tab'))     # 居中制表位
        p._p.append(latex_to_omath(latex))    # 原生公式块（OMML）
        r = p.add_run()
        r._r.append(OxmlElement('w:tab'))     # 右制表位
        self._runs(p, '(', size=12)
        add_seq_field(p, 'eq', size=12)       # 序号：SEQ eq \* ARABIC
        self._runs(p, ')', size=12)
        self.stats['equations_native'] += 1
        return p

    # ------------------------------------------------------------ 伪代码
    def algorithm(self, title, lines):
        p = self.doc.add_paragraph()
        pf = p.paragraph_format
        pf.first_line_indent = Pt(0)
        pf.space_before = Pt(8)
        pf.space_after = Pt(2)
        pf.keep_with_next = True
        self._runs(p, '**%s**' % title, size=10.5)
        table = self.doc.add_table(rows=1, cols=1)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        cell = table.rows[0].cells[0]
        cell.width = Cm(avail_cm())
        table.columns[0].width = Cm(avail_cm())
        first = True
        for ln in lines:
            para = cell.paragraphs[0] if first else cell.add_paragraph()
            first = False
            para.style = self.doc.styles['Appendix Text']
            f = para.paragraph_format
            f.first_line_indent = Pt(0)
            f.left_indent = Pt(0)
            f.space_before = Pt(0)
            f.space_after = Pt(0)
            f.line_spacing = 1.0
            self._runs(para, ln.rstrip(), size=9, math=True)
        pin_table(table, avail_cm(), mode='box', top_sz=4, bottom_sz=4, header_border=False)
        self.stats['tbl_algo'] += 1
        self.stats['algorithms'] += 1
        return table


def avail_cm():
    return PORTRAIT_AVAIL


# --------------------------------------------------------------------------- #
# 表格版式
# --------------------------------------------------------------------------- #
def col_metrics(grid, size_pt):
    """每列最大显示长度（含表头）、数值列判定、单位宽度。"""
    n = len(grid[0])
    unit = 0.37 * size_pt / 10.5 / 2.0
    lens, numeric = [], []
    for j in range(n):
        vals = [str(r[j]) if j < len(r) else '' for r in grid]
        lens.append(max(1, max(disp_len(v) for v in vals)))
        body = [v.strip() for v in vals[1:] if v.strip() not in ('', '—', '-')]
        numeric.append(bool(body) and all(
            re.fullmatch(r'[+\-−]?[\d.,%×eE\s]*', v) and any(ch.isdigit() for ch in v)
            for v in body))
    return lens, numeric, unit


def min_lines(lens, unit, avail):
    """在给定版心下，各行至少需要的文本行数（按最长列反算）。"""
    return max(1, int(math.ceil(sum(lens) * unit / avail - 1e-9)))


def pick_layout(grid, size_pt=9):
    """按“每页可容纳行数”选择纵向 / 横向，尽量少翻页。"""
    lens, _, unit = col_metrics(grid, size_pt)
    best = None
    for land, avail, usable in ((False, PORTRAIT_AVAIL, PORTRAIT_USABLE),
                                (True, LANDSCAPE_AVAIL, LANDSCAPE_USABLE)):
        k = min_lines(lens, unit, avail)
        cost = k / usable
        if best is None or cost < best[0] - 1e-9:
            best = (cost, land)
    return best[1]


def alloc_widths(grid, avail, size_pt):
    """列宽分配：先按“文本区预算”分配，再加回单元格左右内边距。

    修复（Stage24 逐页视觉 QA）：原先按 `avail` 直接分配，未扣除 w:tblCellMar
    的左右内边距（57+57 dxa ≈ 0.20 cm/列），窄数字列（如表 7-2 的 n 列 "2,233"）
    文本区因此偏窄而被拆行。改为按 `avail - n*0.20` 分配后再补回内边距，
    表宽仍等于 `avail` 不变。
    """
    lens, numeric, unit = col_metrics(grid, size_pt)
    n = len(grid[0])
    k = min_lines(lens, unit, avail)
    w = [max(lens[j] / float(k) * unit, 0.42) for j in range(n)]
    body = max(1.0, avail - n * CELL_MARGIN_CM)
    s = float(sum(w))
    if s > body:
        w = [x * body / s for x in w]
    elif s < body - 1e-6:
        extra = body - s
        tot = float(sum(lens))
        w = [w[i] + extra * lens[i] / tot for i in range(n)]
    return [x + CELL_MARGIN_CM for x in w], numeric


def needs_landscape(grid, size_pt):
    return pick_layout(grid, size_pt)


def pin_table(table, total_cm, mode='three_line', top_sz=12, bottom_sz=12, header_sz=6,
              header_border=True):
    pr = table._tbl.tblPr
    for tag in ('w:tblW', 'w:tblLayout', 'w:tblBorders', 'w:tblCellMar'):
        for el in pr.findall(qn(tag)):
            pr.remove(el)
    w = OxmlElement('w:tblW')
    w.set(qn('w:w'), str(int(round(total_cm * 567))))
    w.set(qn('w:type'), 'dxa')
    pr.append(w)
    tb = OxmlElement('w:tblBorders')
    if mode == 'none':
        edges = [('top', 0), ('left', 0), ('bottom', 0), ('right', 0),
                 ('insideH', 0), ('insideV', 0)]
    elif mode == 'box':
        edges = [('top', top_sz), ('left', top_sz), ('bottom', bottom_sz),
                 ('right', bottom_sz), ('insideH', 0), ('insideV', 0)]
    else:
        edges = [('top', top_sz), ('left', 0), ('bottom', bottom_sz), ('right', 0),
                 ('insideH', 0), ('insideV', 0)]
    for edge, sz in edges:
        el = OxmlElement('w:' + edge)
        if sz:
            el.set(qn('w:val'), 'single'); el.set(qn('w:sz'), str(sz))
            el.set(qn('w:space'), '0'); el.set(qn('w:color'), '000000')
        else:
            el.set(qn('w:val'), 'none'); el.set(qn('w:sz'), '0')
            el.set(qn('w:space'), '0'); el.set(qn('w:color'), 'auto')
        tb.append(el)
    pr.append(tb)
    lay = OxmlElement('w:tblLayout'); lay.set(qn('w:type'), 'fixed')
    pr.append(lay)
    mar = OxmlElement('w:tblCellMar')
    for side, val in (('left', 28), ('right', 28), ('top', 0), ('bottom', 0)):
        el = OxmlElement('w:' + side)
        el.set(qn('w:w'), str(val)); el.set(qn('w:type'), 'dxa')
        mar.append(el)
    pr.append(mar)
    order_tbl_pr(pr)
    if header_border and mode != 'none' and table.rows:
        for cell in table.rows[0].cells:
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_b = OxmlElement('w:tcBorders')
            bot = OxmlElement('w:bottom')
            bot.set(qn('w:val'), 'single'); bot.set(qn('w:sz'), str(header_sz))
            bot.set(qn('w:space'), '0'); bot.set(qn('w:color'), '000000')
            tc_b.append(bot)
            tc_pr.append(tc_b)


# =========================================================================== #
# 四、Markdown 解析
# =========================================================================== #
def split_md_row(line):
    line = line.strip()
    if line.startswith('|'):
        line = line[1:]
    if line.endswith('|'):
        line = line[:-1]
    out = []
    for c in re.split(r'(?<!\\)\|', line):
        c = c.strip().replace('\\|', '|')
        out.append(c)
    return out


def is_sep(line):
    if not line.strip().startswith('|'):
        return False
    cells = split_md_row(line)
    return bool(cells) and all(re.fullmatch(r':?-{2,}:?', c or '-') for c in cells)


def scan_file(path):
    lines = path.read_text(encoding='utf-8').splitlines()
    tables = []
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith('|') and i + 1 < len(lines) and is_sep(lines[i + 1]):
            start = i
            grid = [split_md_row(lines[i])]
            i += 2
            while i < len(lines) and lines[i].strip().startswith('|'):
                grid.append(split_md_row(lines[i]))
                i += 1
            n = len(grid[0])
            grid = [(r + [''] * (n - len(r)))[:n] for r in grid]
            tables.append((start, i - 1, grid))
        else:
            i += 1
    fences = []
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith('```'):
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith('```'):
                j += 1
            fences.append((i, j, [lines[k] for k in range(i + 1, j)]))
            i = j + 1
        else:
            i += 1
    return lines, tables, fences


def parse_chapter(path):
    """把一章 Markdown 解析为有序块列表。"""
    lines, tables, fences = scan_file(path)
    markers = []
    for idx, ln in enumerate(lines):
        s = ln.strip()
        m = re.match(r'^>\s*【插图：图\s*([\w\-]+)\s*(.*?)】\s*$', s)
        if m:
            markers.append((idx, 'fig', m.group(1)))
            continue
        m = re.match(r'^>\s*【插表：表\s*([\w\-]+)\s*(.*?)】\s*$', s)
        if m:
            markers.append((idx, 'tbl', m.group(1)))
    # 表块 → 最近的未占用表格
    tbl_assign = {}
    used = set()
    for idx, kind, key in markers:
        if kind != 'tbl':
            continue
        best, bestd = None, None
        for ti, (s0, s1, grid) in enumerate(tables):
            if ti in used:
                continue
            d = min(abs(idx - s0), abs(idx - s1))
            if bestd is None or d < bestd:
                best, bestd = ti, d
        if best is not None:
            used.add(best)
            tbl_assign[idx] = tables[best][2]
    table_spans = set()
    for ti in used:
        s0, s1, _ = tables[ti]
        table_spans.update(range(s0, s1 + 1))
    fence_spans = set()
    for f0, f1, _ in fences:
        fence_spans.update(range(f0, f1 + 1))

    blocks = []
    i = 0
    while i < len(lines):
        if i in fence_spans or i in table_spans:
            i += 1
            continue
        s = lines[i].strip()
        if not s or s.startswith('<!--'):
            i += 1
            continue
        m = re.match(r'^(#{1,6})\s+(.*)$', s)
        if m:
            blocks.append(('h', len(m.group(1)), m.group(2).strip()))
            i += 1
            continue
        m = re.match(r'^>\s*【插图：图\s*([\w\-]+)\s*(.*?)】\s*$', s)
        if m:
            blocks.append(('fig', m.group(1)))
            i += 1
            continue
        m = re.match(r'^>\s*【插表：表\s*([\w\-]+)\s*(.*?)】\s*$', s)
        if m:
            grid = tbl_assign.get(i)
            blocks.append(('tbl', m.group(1), grid))
            i += 1
            continue
        if s.startswith('>'):
            i += 1
            continue
        m = re.match(r'^\$\$(.*?)\$\$\s*$', s)
        if m:
            body = m.group(1).strip()
            tm = re.search(r'\\tag\{([^}]+)\}', body)
            no = tm.group(1) if tm else ''
            body = re.sub(r'\\tag\{[^}]*\}', '', body).strip()
            blocks.append(('eq', no, body))
            i += 1
            continue
        m = re.match(r'^\*\*(算法\s*[\w\-]+\s+.*?)\*\*\s*$', s)
        if m:
            title = m.group(1).strip()
            body = []
            for f0, f1, fl in fences:
                if f0 == i + 1:
                    body = fl
                    break
            if not body:
                for f0, f1, fl in fences:
                    if f0 > i and f0 - i <= 3:
                        body = fl
                        break
            blocks.append(('algo', title, body))
            i += 1
            continue
        if re.match(r'^[-*]\s+', s):
            blocks.append(('li', re.sub(r'^[-*]\s+', '', s)))
            i += 1
            continue
        blocks.append(('p', s))
        i += 1
    return blocks


def read_references(path):
    text = path.read_text(encoding='utf-8')
    part = text.split('## 一、论文用文献表')[1].split('## 二、')[0]
    return [ln.strip() for ln in part.splitlines()
            if re.match(r'^\[\d+\]', ln.strip())]


# =========================================================================== #
# 五、正文装配
# =========================================================================== #
def load_appendix_data():
    """惰性导入 24c_appendix_data（仅在 INCLUDE_APPENDIX_B 为真时调用）。"""
    spec = importlib.util.spec_from_file_location(
        'stage24_appendix_data', str(Path(__file__).resolve().parent / '24c_appendix_data.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(out_path=OUTDOCX):
    b = Builder()
    doc = b.doc

    # ---------------- 前置部分（罗马数字） ---------------- #
    front = doc.sections[1]
    set_page(front)
    set_pgnum(front, 'upperRoman', start=1)
    own_footer(front, roman=True)

    abs_lines = (SRC23 / '00_摘要与Abstract.md').read_text(encoding='utf-8').splitlines()

    def block_range(start_kw, end_kw):
        s = next(i for i, l in enumerate(abs_lines) if l.strip() == start_kw)
        e = next(i for i in range(s + 1, len(abs_lines)) if abs_lines[i].strip() == end_kw)
        return s, e

    b.heading(1, '摘要')
    s, e = block_range('## 摘要', '## 关键词')
    for ln in abs_lines[s + 1:e]:
        t = ln.strip()
        if not t:
            continue
        if t.startswith('**') and t.endswith('**'):
            b.para(t[2:-2], indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, bold_all=True)
        else:
            b.para(t)
    s, e = block_range('## 关键词', '## Abstract')
    kw = next(abs_lines[i].strip() for i in range(s + 1, e) if abs_lines[i].strip())
    p = b.para('', indent=False, keep_next=False)
    p.paragraph_format.space_before = Pt(12)
    b._runs(p, '**关键词：**' + kw)

    p = b.heading(1, 'Abstract')
    p.paragraph_format.page_break_before = True
    s, e = block_range('## Abstract', '## Key words')
    for ln in abs_lines[s + 1:e]:
        t = ln.strip()
        if t:
            # Stage26.6 §9.2：英文摘要统一行距 1.4，配合正文压缩使整段控制在单页
            _ap = b.para(t)
            _ap.paragraph_format.line_spacing = 1.4
    # Key words 为文件最后一段
    ks = next(i for i, l in enumerate(abs_lines) if l.strip() == '## Key words')
    kw_en = next(abs_lines[i].strip() for i in range(ks + 1, len(abs_lines))
                 if abs_lines[i].strip())
    p = b.para('', indent=False)
    p.paragraph_format.space_before = Pt(6)
    b._runs(p, '**Key words: **' + kw_en)

    p = b.para('', indent=False, align=WD_ALIGN_PARAGRAPH.CENTER)
    p.paragraph_format.page_break_before = True
    p.paragraph_format.space_after = Pt(12)
    r = p.add_run('目　录')
    r.font.size = Pt(16); r.bold = True
    r.font.name = '黑体'; r._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
    p = b.para('', indent=False)
    add_field(p, 'TOC \\o "1-3" \\h \\z \\u', '（打开文档后按 F9 更新目录）')

    # ---------------- 正文（阿拉伯数字，从 1 开始） ---------------- #
    body = doc.add_section(WD_SECTION.NEW_PAGE)
    set_page(body)
    set_pgnum(body, 'decimal', start=1)
    own_footer(body, roman=False)

    for fname in CHAPTERS:
        blocks = parse_chapter(SRC23 / fname)
        for blk in blocks:
            kind = blk[0]
            if kind == 'h':
                lvl = {1: None, 2: 1, 3: 2}.get(blk[1], 3)
                if lvl:
                    b.heading(lvl, blk[2])
            elif kind == 'p':
                b.para(blk[1])
            elif kind == 'li':
                b.para(blk[1], style='Body List')
            elif kind == 'fig':
                key = blk[1]
                cap = next((('图 %s %s' % (k, t)) for k, t, _ in FIG_MAIN if k == key), None)
                src = next((p for k, _, p in FIG_MAIN if k == key), None)
                if cap is None or src is None:
                    raise KeyError('未登记的正文图：%s' % key)
                b.figure(FIGROOT / src, cap, width_cm=FIG_WIDTH_CM.get(key))
                b.stats['fig_main'] += 1
            elif kind == 'tbl':
                key = blk[1]
                grid = blk[2]
                if grid is None:
                    raise KeyError('未匹配到表格数据：表 %s' % key)
                cap = '表 %s %s' % (key, TBL_MAIN_TITLE[key])
                # Stage26.1 版式参数：本轮新增 表 3-4 并扩充 表 4-2 / 7-4 / 8-1 / 8-5，
                # 首轮构建总页数达 106 页（门禁 < 100）。在不改全局字号 / 页边距、
                # 不删正文的前提下，把正文表内文字由 10.5 pt 收紧为 9 pt（表题仍为
                # 10.5 pt，与图题一致），使表格行高按固定 10.5 pt 行距排布。
                b.data_table(grid, cap, size_pt=9, avail=PORTRAIT_AVAIL, tag='main',
                             cap_size=10.5, keep_together=(key in KEEP_TOGETHER_TABLES))
                b.stats['tbl_main'] += 1
            elif kind == 'eq':
                b.equation(blk[1], blk[2])
                b.stats['equations'] += 1
            elif kind == 'algo':
                b.algorithm(blk[1], blk[2])

    # 参考文献
    b.heading(1, '参考文献')
    for ref in read_references(REFS):
        b.para(ref, style='Reference', indent=False)

    # ---------------- 附录（默认关闭：最终论文不含附录） ---------------- #
    # INCLUDE_APPENDICES=False 时不写入任何附录标题 / 分节 / 图表，也不导入 24c。
    if INCLUDE_APPENDICES and (INCLUDE_APPENDIX_A or INCLUDE_APPENDIX_B):
        app = doc.add_section(WD_SECTION.NEW_PAGE)
        set_page(app)
        set_pgnum(app, 'decimal', start=None)

        if INCLUDE_APPENDIX_A:
            b.heading(1, '附录 A 附录图')
            for no, title, src in FIG_APP:
                b.figure(FIGROOT / src, '附图 %s %s' % (no, title))
                b.stats['fig_app'] += 1

        if INCLUDE_APPENDIX_B:
            b.heading(1, '附录 B 附表')
            appendix_data = load_appendix_data()
            specs = appendix_data.build_appendix_b()
            orient = {'land': False}

            def ensure_orientation(land):
                """仅在需要时切换横向/纵向，避免产生空 section 与空白页。"""
                if orient['land'] == land:
                    return
                sec = doc.add_section(WD_SECTION.NEW_PAGE)
                set_page(sec, landscape=land)
                set_pgnum(sec, 'decimal', start=None)
                orient['land'] = land

            for spec in specs:
                n_rows = 0
                for pi, part in enumerate(spec['parts']):
                    grid = part['grid']
                    land = needs_landscape(grid, 9)
                    ensure_orientation(land)
                    if pi == 0:
                        cp = b.doc.add_paragraph(style='Table Caption')
                        cp.paragraph_format.first_line_indent = Pt(0)
                        cp.paragraph_format.keep_with_next = True
                        b._runs(cp, spec['caption'], size=9)
                    else:
                        sp = b.doc.add_paragraph()
                        f = sp.paragraph_format
                        f.first_line_indent = Pt(0)
                        f.space_before = Pt(8)
                        f.space_after = Pt(2)
                        f.keep_with_next = True
                        b._runs(sp, '**%s**' % part['sheet'], size=9)
                    b.data_table(grid, None, size_pt=9,
                                 avail=LANDSCAPE_AVAIL if land else PORTRAIT_AVAIL,
                                 tag='appendix', cant_split=False)
                    n_rows += len(grid) - 1
                b.stats['tbl_app'] += 1
                b.appendix_rows.append((spec['no'], len(spec['parts']), n_rows))

    doc.save(str(out_path))
    return b


# =========================================================================== #
# 六、自检
# =========================================================================== #
FORBIDDEN = ['Stage23', 'Stage24', 'Gate', 'PASS', 'CONTENT_FREEZE', 'outputs/',
             'data/', '.py', 'source:', 'Git']
FORBIDDEN_OLD = ['172050', '0.471492', '369组', '369 组', '图17',
                 '模型记忆公司', '对公司的熟悉', '可迁移能力的下限估计',
                 '共用同一测试集', '10 个特征', '6 个特征', '17,678',
                 '327 维', '532 个', '2,878', '1 元/天以内',
                 '前期固定图件', '请配合表格阅读', '44 号表', '45 号表', '47 号表',
                 'Stage25', 'Stage26', '本轮', '旧 F 组', '旧口径', '旧实现',
                 # Stage26.4：旧维度与旧指标不得残留
                 '308 维', '310 维', '35.281040', '64.783455', '0.582546',
                 '36.451863', '52.093534', '32.132572', '61.5%',
                 'presence 口径', 'mean|SHAP|', '模型产物原始字段名',
                 # Stage26.6：工程命名、伪对称误差线、一次评估表述与已删图号不得残留
                 'Safe-F', 'SafeF', 'IQR/2', 'IQR / 2',
                 '测试集只使用一次', '一次性评估', '完全不再使用测试集',
                 # Stage26.8：图 4-7 / 4-8 已由业务时间专题重新占用为正式图号，
                 # 仅保留仍未使用的已删图号（图 3-3 / 5-4 / 6-4）。
                 '图 3-3', '图 5-4', '图 6-4',
                 '中文预训练文本嵌入模型']
ANCHORS = ['172,063', '172063', '17,144', '17144', '20,556', '20556',
           '17,146', '17146', '17,650', '17650', '14,883', '14883',
           '2,233', '2,245', '2,232', '2,398', '2,674', '2,402',
           '10,418', '10,521', '10,083', '1,688',
           # Stage26.4 正式模型与泛化关键值
           '288', '290', '216', '272', '232',
           '36.00', '51.98', '31.72', '35.48', '64.81', '0.582',
    '66.30', '103.57', '47.62', '0.563', '0.136', '0.656',
    '1.683309', '1000 次', '17,131', '207', '314', '763',
    '14.88', '7.42', '5.96', '1.726', '0.609',
    # Stage26.5 新增与更新
    '61.96', '0.6389', '6.81', '10.13', '5,293', '5,265',
    # Stage26.6 新增（地域两口径、D 组维度、嵌入模型名）
    '17,040', '4,253', '4,795', '947', '611', '56 维', '44 个高频技能指示列',
    'bge-small-zh-v1.5', '发布时间位置扩展特征',
]


def all_text(doc):
    chunks = []
    for p in doc.paragraphs:
        chunks.append(p.text)
    for t in doc.tables:
        for row in t.rows:
            for c in row.cells:
                chunks.append(c.text)
    return chunks


def inventory_check(full_text):
    """按复核后的内联数学清单逐条核验：命中文本在最终 docx 可见文本中应为 0 条。

    Stage26.3：清单优先取 59 号（十章 md 复核后的最终清单），缺失时回退 55 号。
    单字母数学变量（p、N、k 等）按“后接 为/表示/与/和/的/值/取/大于/小于/比较符”的
    同一口径用正则核验，避免把普通英文字母误判为未转换。
    """
    path, sheet = None, None
    for name, sheet_name in (('64_stage26_4_inline_math_recheck.xlsx', '01_段内公式实例清单'),
                             ('59_stage26_3_inline_math_final.xlsx', '01_内联数学符号最终清单'),
                             ('55_stage26_3_inline_math_inventory.xlsx', '01_内联数学符号清单')):
        candidate = ROOT / 'outputs' / 'tables' / name
        if candidate.exists():
            path, sheet = candidate, sheet_name
            break
    if path is None:
        return {'清单来源': '—', '清单条目数': 0, '未转换条目数': 0, '未转换示例': []}
    import pandas as pd                                       # noqa: PLC0415
    rows = pd.read_excel(path, sheet_name=sheet)
    single = re.compile(r'(?<![A-Za-z0-9_])([%s])'
                        r'(?=\s*(?:为|表示|与|和|的|值|取|大于|小于|[=<>]))' % SINGLE_LETTERS)
    remain = []
    for _, row in rows.iterrows():
        token = str(row['命中文本'])
        if row['类型'] == '单字母数学变量':
            if single.search(full_text):
                remain.append(f"{row['文件']}#{row['行号']} {token}（单字母口径仍有命中）")
            continue
        if re.search(r'(?<![A-Za-z0-9_])%s(?![A-Za-z0-9])' % re.escape(token), full_text):
            remain.append(f"{row['文件']}#{row['行号']} {token}")
    return {'清单来源': path.name, '清单条目数': int(len(rows)),
            '未转换条目数': len(remain), '未转换示例': remain[:10]}


def self_check(path):
    doc = Document(str(path))
    body = doc.element.body
    paras = doc.paragraphs
    styles = {}
    for p in paras:
        styles[p.style.name] = styles.get(p.style.name, 0) + 1
    tabs = doc.tables
    omml = len(body.findall('.//' + qn('m:oMath')))
    drawings = len(body.findall('.//' + qn('w:drawing')))
    algo_tbl = 0
    formula_tbl = 0
    omath_in_tbl = 0
    omath_para_in_tbl = 0
    for t in tabs:
        # “公式用表”= 用表格承载带编号的独立公式（表内出现 SEQ eq 域）；
        # 段内公式（m:oMath）出现在表格单元中是允许的（伪代码块与紧凑表内的数学量）
        if any(re.search(r'SEQ\s+eq\b', (el.text or ''))
               for el in t._tbl.findall('.//' + qn('w:instrText'))):
            formula_tbl += 1
        omath_para_in_tbl += len(t._tbl.findall('.//' + qn('m:oMathPara')))
        n_math = len(t._tbl.findall('.//' + qn('m:oMath')))
        if n_math:
            omath_in_tbl += n_math
        txt = t.rows[0].cells[0].text if t.rows and t.rows[0].cells else ''
        if '输入：' in txt:
            algo_tbl += 1
    # 公式：正文级段落内的 m:oMath + SEQ 域自动编号 + 段落制表位
    def _has_seq(paragraph):
        for el in paragraph._p.findall('.//' + qn('w:instrText')):
            if re.search(r'SEQ\s+eq\b', el.text or ''):
                return True
        for el in paragraph._p.findall('.//' + qn('w:fldSimple')):
            if re.search(r'SEQ\s+eq\b', el.get(qn('w:instr')) or ''):
                return True
        return False

    eq_paras = [p for p in paras if _has_seq(p)]
    inline_math = 0
    for p in paras:
        if _has_seq(p):
            continue
        inline_math += len(p._p.findall('./' + qn('m:oMath')))
    # 段内公式残留检查：正文可见纯文本中不应再有可转换的数学写法
    residual_para = [(p.style.name, p.text) for p in paras if inline_math_spans(p.text)]
    residual_cell = [(cell.text,) for t in tabs for row in t.rows for cell in row.cells
                     if inline_math_spans(cell.text)]
    residual_para = list(dict.fromkeys(residual_para))
    residual_cell = list(dict.fromkeys(residual_cell))
    # Stage26.4 §11.1：绝对值必须落在真正的 Word 数学 delimiter（m:d + 竖线分隔符）中。
    # 逐处抽取 OMML 核验：分别统计独立公式 / 段内公式 / 表格单元内的绝对值 delimiter，
    # 并检查数学 run 内是否残留普通竖线。
    delim_kinds = {}

    def _delims(element):
        found = []
        for node in element.findall('.//' + qn('m:d')):
            pr = node.find(qn('m:dPr'))
            begin = end = None
            if pr is not None:
                beg = pr.find(qn('m:begChr'))
                fin = pr.find(qn('m:endChr'))
                begin = beg.get(qn('m:val')) if beg is not None else '('
                end = fin.get(qn('m:val')) if fin is not None else ')'
            key = '%s|%s' % (begin, end)
            delim_kinds[key] = delim_kinds.get(key, 0) + 1
            if begin == '|' and end == '|':
                inner = ''.join(piece.text or ''
                                for piece in node.findall('.//' + qn('m:t')))
                found.append(inner)
        return found

    abs_equation, abs_paragraph, abs_table = [], [], []
    for paragraph in paras:
        target = abs_equation if _has_seq(paragraph) else abs_paragraph
        target.extend(_delims(paragraph._p))
    for table in tabs:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    abs_table.extend(_delims(paragraph._p))
    abs_delim = len(abs_equation) + len(abs_paragraph) + len(abs_table)
    abs_delim_texts = abs_equation + abs_paragraph + abs_table
    pipe_math_equation, pipe_math_inline = [], []
    for paragraph in paras:
        target = pipe_math_equation if _has_seq(paragraph) else pipe_math_inline
        for run in paragraph._p.findall('.//' + qn('m:r')):
            for piece in run.findall(qn('m:t')):
                if '|' in (piece.text or ''):
                    target.append(piece.text)
    abs_pattern = re.compile(r'\|[^|\n]{0,12}\|')
    abs_text_residual = list(dict.fromkeys(
        text.strip()[:80] for text in all_text(doc) if abs_pattern.search(text)))
    seq_fields = 0
    seq_arabic = 0
    for el in list(body.findall('.//' + qn('w:instrText'))) + \
            list(body.findall('.//' + qn('w:fldSimple'))):
        instr = (el.text or '') if el.tag == qn('w:instrText') else (el.get(qn('w:instr')) or '')
        if re.search(r'SEQ\s+eq\b', instr):
            seq_fields += 1
            if 'ARABIC' in instr:
                seq_arabic += 1
    static_eq = [p.text for p in eq_paras
                 if re.search(r'\(\d+\)', p.text)
                 and not p._p.findall('.//' + qn('w:instrText'))
                 and not p._p.findall('.//' + qn('w:fldSimple'))]
    eq_tabs = sorted({tuple((int(ts.position.twips), ts.alignment.name)
                            for ts in p.paragraph_format.tab_stops)
                      for p in eq_paras})
    fig_caps = [p.text for p in paras if p.style.name == 'Figure Caption']
    tbl_caps = [p.text for p in paras if p.style.name == 'Table Caption']
    landscape = sum(1 for s in doc.sections if s.page_width > s.page_height)
    full = '\n'.join(all_text(doc))
    scan = {k: full.count(k) for k in FORBIDDEN}
    scan_old = {k: full.count(k) for k in FORBIDDEN_OLD}
    anchors = {k: full.count(k) for k in ANCHORS}
    heads = [(p.style.name, p.text) for p in paras
             if p.style.name in ('Heading 1', 'Heading 2', 'Heading 3')]
    # 与源图/表逐一比对
    missing = [n for _, _, src in FIG_MAIN + FIG_APP if not (FIGROOT / src).exists()]
    return {
        'path': str(path), 'size': path.stat().st_size,
        'paragraphs': len(paras), 'styles': styles, 'tables': len(tabs),
        'omml': omml, 'drawings': drawings, 'algo_tbl': algo_tbl,
        'formula_tbl': formula_tbl, 'omath_in_tbl': omath_in_tbl,
        'omath_para_in_tbl': omath_para_in_tbl,
        'inline_math_paragraphs': inline_math, 'residual_para': residual_para,
        'residual_cell': residual_cell, 'inventory': inventory_check(full),
        'abs_delim': abs_delim, 'abs_delim_texts': abs_delim_texts,
        'abs_equation': abs_equation, 'abs_paragraph': abs_paragraph,
        'abs_table': abs_table, 'delim_kinds': delim_kinds,
        'pipe_math_equation': pipe_math_equation, 'pipe_math_inline': pipe_math_inline,
        'abs_text_residual': abs_text_residual,
        'eq_para_texts': [p.text for p in eq_paras], 'eq_tabs': eq_tabs,
        'seq_fields': seq_fields, 'seq_arabic': seq_arabic, 'static_eq': static_eq,
        'fig_caps': fig_caps, 'tbl_caps': tbl_caps, 'landscape_sections': landscape,
        'scan': scan, 'scan_old': scan_old, 'anchors': anchors,
        'headings': heads, 'sections': len(doc.sections), 'missing_fig_files': missing,
    }


def try_word_com(path):
    """用 Word COM 更新域（TOC / 页码）并保存，失败则跳过。"""
    try:
        import win32com.client as win32
    except Exception as e:                                  # pragma: no cover
        return 'win32com 不可用：%s' % e
    word = None
    notes = []
    try:
        word = win32.DispatchEx('Word.Application')
        word.Visible = False
        word.DisplayAlerts = 0
        wdoc = word.Documents.Open(str(path), ReadOnly=False, AddToRecentFiles=False)
        try:
            wdoc.Fields.Update()
        except Exception as e:
            notes.append('Fields.Update 失败：%s' % e)
        try:
            if wdoc.TablesOfContents.Count:
                wdoc.TablesOfContents(1).Update()
        except Exception as e:
            notes.append('TOC 更新失败：%s' % e)
        try:
            wdoc.Repaginate()
        except Exception as e:
            notes.append('Repaginate 失败：%s' % e)
        pages = None
        try:
            pages = wdoc.ComputeStatistics(2)      # wdStatisticPages
        except Exception as e:
            notes.append('页数统计失败：%s' % e)
        saved = False
        try:
            wdoc.Save()
            saved = True
        except Exception as e:
            notes.append('Save 失败（改用 SaveAs2）：%s' % e)
            try:
                wdoc.SaveAs2(str(path), FileFormat=16)   # wdFormatDocumentDefault
                saved = True
            except Exception as e2:
                notes.append('SaveAs2 亦失败：%s' % e2)
        try:
            wdoc.Saved = True
            wdoc.Close(0)
        except Exception:
            pass
        return ('Word 域更新：保存=%s；统计页数=%s；%s'
                % ('成功' if saved else '失败', pages, '；'.join(notes) or '无异常'))
    except Exception as e:                                  # pragma: no cover
        return 'Word 域更新失败（不影响 docx 生成）：%s' % e
    finally:
        if word is not None:
            try:
                word.Quit()
            except Exception:
                pass


def main():
    out = OUTDOCX
    for a in sys.argv[1:]:
        if a.startswith('--out='):
            out = Path(a.split('=', 1)[1]).resolve()
    if '--com-only' in sys.argv:
        print(try_word_com(out))
        return
    do_com = '--no-com' not in sys.argv
    do_build = '--check-only' not in sys.argv
    if do_build:
        b = build(out)
        print('=' * 78)
        print('DOCX 生成完成：', out)
        print('文件大小：', out.stat().st_size, 'bytes')
        print('-' * 78)
        print('[构建计数] 正文图=%d 附录图=%d 正文表=%d 附表=%d 公式=%d 算法=%d'
              % (b.stats['fig_main'], b.stats['fig_app'], b.stats['tbl_main'],
                 b.stats['tbl_app'], b.stats['equations'], b.stats['algorithms']))
        print('[附录开关] INCLUDE_APPENDICES=%s INCLUDE_APPENDIX_A=%s INCLUDE_APPENDIX_B=%s'
              % (INCLUDE_APPENDICES, INCLUDE_APPENDIX_A, INCLUDE_APPENDIX_B))
        print('[构建计数] 公式用表=%d 原生公式=%d 算法用表=%d'
              % (b.stats['tbl_formula'], b.stats['equations_native'], b.stats['tbl_algo']))
        print('[缺失图片] ', b.missing_figs if b.missing_figs else '无')
        print('-' * 78)
        if INCLUDE_APPENDICES and INCLUDE_APPENDIX_B:
            print('[附录 B 行数]')
            for no, nparts, nrows in b.appendix_rows:
                print('   %s parts=%d rows=%d' % (no, nparts, nrows))
            print('[附录 B 合计行数] %d' % sum(r for _, _, r in b.appendix_rows))
            appendix_data = load_appendix_data()
            rep = appendix_data.sanitize_report()
            print('[附录数据脱敏] 命中单元格数 = %d，替换次数 = %d'
                  % (rep['cell_rewrites'], rep['total_replacements']))
            for r in rep['rules']:
                print('   hits=%-4d rule=%s' % (r['hits'], r['rule']))
                print('        before=%r' % r['before'])
                print('        after =%r' % r['after'])
        else:
            print('[附录] 已关闭：未插入任何附录标题 / 附图 / 附表，未导入 24c_appendix_data')

    if do_com and do_build:
        print('-' * 78)
        print(try_word_com(out))

    print('=' * 78)
    info = self_check(out)
    print('[自检] 文件 =', info['path'])
    print('[自检] 文件大小 = %d bytes (%.2f MB)' % (info['size'], info['size'] / 1048576.0))
    print('[自检] 分节数 = %d（其中横向 section = %d）'
          % (info['sections'], info['landscape_sections']))
    print('[自检] 段落数 =', info['paragraphs'])
    print('[自检] Heading1/2/3 = %d / %d / %d'
          % (info['styles'].get('Heading 1', 0), info['styles'].get('Heading 2', 0),
             info['styles'].get('Heading 3', 0)))
    print('[自检] Figure Caption =', len(info['fig_caps']))
    print('[自检] Table Caption =', len(info['tbl_caps']))
    print('[自检] Reference =', info['styles'].get('Reference', 0))
    print('[自检] toc 1/2/3 =', sum(info['styles'].get('toc %d' % i, 0) for i in (1, 2, 3)))
    print('[自检] 正文/附录图片(inline drawing) =', info['drawings'])
    print('[自检] 原生表总数 = %d（含公式用表 %d、算法用表 %d）'
          % (info['tables'], info['formula_tbl'], info['algo_tbl']))
    print('[自检] 实例化 OMML 公式 =', info['omml'])
    print('[自检] 段内公式 m:oMath = %d（正文段落内，非 m:oMathPara；表内 = %d）'
          % (info['inline_math_paragraphs'], info['omath_in_tbl']))
    print('[自检] 段内公式残留（正文段落）= %d 处 %s'
          % (len(info['residual_para']), info['residual_para'][:6] or ''))
    print('[自检] 段内公式残留（表格单元）= %d 处 %s'
          % (len(info['residual_cell']), info['residual_cell'][:6] or ''))
    print('[自检] 内联公式清单核验 = %s' % (info['inventory'],))
    print('[自检] 绝对值 delimiter（m:d 竖线分隔符）实测 = %d 处：独立公式 %d、段内公式 %d、表格单元 %d'
          % (info['abs_delim'], len(info['abs_equation']), len(info['abs_paragraph']),
             len(info['abs_table'])))
    print('[自检] 数学 run 内普通竖线残留 = %d 处（独立公式 %d、段内/表格 %d）'
          % (len(info['pipe_math_equation']) + len(info['pipe_math_inline']),
             len(info['pipe_math_equation']), len(info['pipe_math_inline'])))
    print('[自检] m:d 分隔符写法分布 = %s' % (info['delim_kinds'],))
    print('[自检] 绝对值 delimiter 内容 = 段内 %s；独立公式 %s'
          % (info['abs_paragraph'] + info['abs_table'], info['abs_equation']))
    print('[自检] 正文可见文本中的未转换绝对值写法 = %d 处 %s'
          % (len(info['abs_text_residual']), info['abs_text_residual'][:6]))
    print('[自检] 公式用表数 = %d / 原生公式数 = %d（段落级；表格单元内段内公式 = %d，'
          '其中“单元内仅有公式”被 Word 规范化为 m:oMathPara 的段落 = %d）'
          % (info['formula_tbl'], len(info['eq_para_texts']), info['omath_in_tbl'],
             info['omath_para_in_tbl']))
    print('[自检] SEQ eq 域 = %d（含 ARABIC = %d）；静态编号残留 = %d'
          % (info['seq_fields'], info['seq_arabic'], len(info['static_eq'])))
    print('[自检] 公式段落制表位(twips, 对齐) =', info['eq_tabs'])
    print('[自检] 公式编号显示值 =', [t for t in info['eq_para_texts']])
    print('[自检] 缺失图片文件 =', info['missing_fig_files'] or '无')
    print('[自检] 禁用工程串扫描 =', info['scan'])
    print('[自检] 旧值/过强表述扫描 =', info['scan_old'])
    print('[自检] 正式锚点出现次数 =', info['anchors'])
    print('[自检] 图题清单：')
    for c in info['fig_caps']:
        print('     ', c)
    print('[自检] 表题清单：')
    for c in info['tbl_caps']:
        print('     ', c)
    print('[自检] 一级标题清单：')
    for name, text in info['headings']:
        if name == 'Heading 1':
            print('     ', text)
    print('=' * 78)


if __name__ == '__main__':
    main()

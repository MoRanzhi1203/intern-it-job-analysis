# -*- coding: utf-8 -*-
"""Stage24 附录 B 数据抽取（可复现脚本）。

职责：只读 `outputs/tables/` 下的正式冻结结果，抽取 附表 B-1 ~ B-12 所需的
表格数据，输出为“二维字符串网格”列表，供 `24a_build_stage24_docx.py` 直接写入
Word 原生表。

硬约束：
- 只读，绝不修改 `outputs/tables/` 下任何文件；
- 不重算、不补算、不人工编造任何数字；
- 单元格文本仅做“工程字符串脱敏”（去掉源码路径 / .py / PASS 等不得出现在
  成稿可见区域的标记），数值本身一个字符都不改。
"""
from __future__ import annotations

import re
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'outputs' / 'tables'

F_SKILL = '27_skill_eda_scope_audit.xlsx'
F_DATASET = '28_modeling_dataset_audit.xlsx'
F_EDA = '29_eda_statistical_analysis.xlsx'
F_MODEL = '30_model_comparison.xlsx'
F_ABLATION = '31_ablation_robustness_shap.xlsx'
F_COMPANY = '32_company_field_semantic_audit.xlsx'
F_FINAL = '33_final_interpretation_audit.xlsx'

# --------------------------------------------------------------------------- #
# 单元格文本脱敏（仅处理“工程标记”，不动任何数字）
# --------------------------------------------------------------------------- #
SANITIZE_RULES = [
    # 数据/产物路径
    (re.compile(r'data/(processed|interim|raw|features)/[A-Za-z0-9_\-]+'), '岗位数据表'),
    (re.compile(r'outputs/[A-Za-z0-9_\-/]*'), '正式产物'),
    (re.compile(r'[A-Za-z]:\\[^\s；;，,、）)]*'), '（本机路径）'),
    # 源码 / 脚本文件名（含 .py 后缀）
    (re.compile(r'[A-Za-z0-9_\-./\\]*\.py(?::\d+)?'), '源文件'),
    # 门禁状态
    (re.compile(r'\bPASS\b'), '通过'),
    (re.compile(r'\bFAIL\b'), '不通过'),
    # 阶段工程标记
    (re.compile(r'\bStage\s*(\d+)'), r'阶段 \1'),
    (re.compile(r'\bstage_(\d+)_[A-Za-z0-9_]+'), r'阶段\1'),
    (re.compile(r'\bCONTENT_FREEZE\b'), '内容冻结'),
    (re.compile(r'\bGate\b'), '门禁'),
    # 项目已判定“不得出现在成稿可见区域”的废止旧值（源自 Stage23 冻结口径）
    (re.compile(r'369\s*组'), '旧口径分组'),
    (re.compile(r'0\.471492'), '（已废止值）'),
]

_sanitize_hits: list = []


def sanitize(text: str) -> str:
    """对单元格文本做工程标记脱敏，返回新文本。"""
    if not text:
        return text
    out = text
    for pat, rep in SANITIZE_RULES:
        out, n = pat.subn(rep, out)
        if n:
            _sanitize_hits.append((text, out, pat.pattern, n))
    return out


def sanitize_report() -> dict:
    """返回脱敏统计（逐规则命中次数与样例）。"""
    agg, samples = {}, {}
    for src, dst, pat, n in _sanitize_hits:
        agg[pat] = agg.get(pat, 0) + n
        samples.setdefault(pat, (src[:60], dst[:60]))
    rows = [{'rule': k, 'hits': v, 'before': samples[k][0], 'after': samples[k][1]}
            for k, v in agg.items()]
    return {'cell_rewrites': len(_sanitize_hits),
            'total_replacements': sum(agg.values()), 'rules': rows}


# --------------------------------------------------------------------------- #
# 数值 / 单元格格式化
# --------------------------------------------------------------------------- #
def cell_text(v) -> str:
    if v is None:
        return ''
    if isinstance(v, bool):
        return '是' if v else '否'
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return '%.10g' % v
    return str(v).strip()


def load_grid(fname: str, sheet: str, where=None) -> list:
    """读取一个 Sheet 为二维字符串网格（首行为表头）。where(row)->bool 可选过滤（不含表头）。"""
    wb = openpyxl.load_workbook(TABLES / fname, read_only=True, data_only=True)
    try:
        ws = wb[sheet]
        rows = []
        header = None
        for r in ws.iter_rows(values_only=True):
            vals = [cell_text(v) for v in r]
            if header is None:
                header = vals
                rows.append(header)
                continue
            if all(v == '' for v in vals):
                continue
            if where is not None and not where(vals):
                continue
            rows.append(vals)
        # 去掉尾部整列为空的列
        ncol = len(header or [])
        while ncol > 1 and all((len(r) < ncol or r[ncol - 1] == '') for r in rows):
            ncol -= 1
        rows = [(r + [''] * ncol)[:ncol] for r in rows]
        return rows
    finally:
        wb.close()


# --------------------------------------------------------------------------- #
# 附表 B 规格
# --------------------------------------------------------------------------- #
def b1() -> list:
    return [{'sheet': '11_统计检验',
             'grid': load_grid(F_EDA, '11_统计检验')}]


def b2() -> list:
    grid = load_grid(F_EDA, '11_统计检验',
                     where=lambda r: len(r) > 19 and r[19] == '公司标签（福利标签）')
    return [{'sheet': '11_统计检验（因素 = 公司标签（福利标签）子集）', 'grid': grid}]


def b3() -> list:
    return [{'sheet': s, 'grid': load_grid(F_EDA, s)}
            for s in ('03_岗位因素薪资', '04_城市薪资', '05_学历薪资', '06_公司因素薪资')]


def b4() -> list:
    out = [{'sheet': '07_技能需求', 'grid': load_grid(F_EDA, '07_技能需求')}]
    for s in ('02_主口径技能排名', '03_扩展口径技能排名', '04_具体技术技能',
              '05_技术领域', '06_业务能力', '07_办公工具', '11_双口径稳健性'):
        out.append({'sheet': s, 'grid': load_grid(F_SKILL, s)})
    return out


def b5() -> list:
    g29 = load_grid(F_EDA, '10_技能共现')
    g27 = load_grid(F_SKILL, '09_技能共现')
    if g29 == g27:
        return [{'sheet': '10_技能共现（与 09_技能共现 内容完全一致，仅保留一处）', 'grid': g29}]
    return [{'sheet': '10_技能共现', 'grid': g29},
            {'sheet': '09_技能共现', 'grid': g27}]


def b6() -> list:
    return [{'sheet': s, 'grid': load_grid(F_EDA, s)} for s in ('13_技能数量薪资', '12_稳健性')]


def b7() -> list:
    return [{'sheet': s, 'grid': load_grid(F_DATASET, s)} for s in ('03_Feature_Manifest', '02_字段清单')]


def b8() -> list:
    wb = openpyxl.load_workbook(TABLES / F_MODEL, read_only=True, data_only=True)
    names = list(wb.sheetnames)
    wb.close()
    return [{'sheet': s, 'grid': load_grid(F_MODEL, s)} for s in names]


def b9() -> list:
    return [{'sheet': s, 'grid': load_grid(F_ABLATION, s)} for s in ('07_SHAP总排名', '11_SHAP稳定性')]


def b10() -> list:
    return [{'sheet': s, 'grid': load_grid(F_ABLATION, s)}
            for s in ('05_极端值敏感性', '06_目标稳健性', '09_分组误差')]


def b11() -> list:
    wb = openpyxl.load_workbook(TABLES / F_COMPANY, read_only=True, data_only=True)
    names = list(wb.sheetnames)
    wb.close()
    return [{'sheet': s, 'grid': load_grid(F_COMPANY, s)} for s in names]


def b12() -> list:
    wb = openpyxl.load_workbook(TABLES / F_FINAL, read_only=True, data_only=True)
    names = list(wb.sheetnames)
    wb.close()
    return [{'sheet': s, 'grid': load_grid(F_FINAL, s)} for s in names]


APPENDIX_B_SPEC = [
    ('B-1', '全部成对比较（含 Cliff’s δ 与 BH-FDR 校正）', b1),
    ('B-2', '公司福利标签二元比较完整结果（183 组）', b2),
    ('B-3', '岗位类别、城市、学历与公司侧因素全量取值', b3),
    ('B-4', '技能需求各层级完整榜单与双口径对照', b4),
    ('B-5', '技能共现明细', b5),
    ('B-6', '技能数量档与稳健性检查明细', b6),
    ('B-7', '建模特征清单（字段、分组、类型、来源）', b7),
    ('B-8', '模型全量指标与超参数', b8),
    ('B-9', '全部特征的 SHAP 排名与解释稳定性', b9),
    ('B-10', '极端值敏感性、目标口径与分组误差', b10),
    ('B-11', '公司字段语义取证', b11),
    ('B-12', '最终解释与统计口径说明', b12),
]


def build_appendix_b() -> list:
    """返回 [{'no','caption','parts':[{'sheet','grid'}...]} ...]，共 12 个附表。"""
    specs = []
    for no, title, fn in APPENDIX_B_SPEC:
        parts = fn()
        for p in parts:
            p['grid'] = [[sanitize(c) for c in row] for row in p['grid']]
        specs.append({'no': no, 'caption': '附表 %s %s' % (no, title), 'parts': parts})
    return specs


if __name__ == '__main__':
    specs = build_appendix_b()
    for s in specs:
        nrows = [len(p['grid']) - 1 for p in s['parts']]
        print('%s parts=%d rows=%s total=%d' % (s['no'], len(s['parts']), nrows, sum(nrows)))
    rep = sanitize_report()
    print('sanitize cell rewrites:', rep['cell_rewrites'])
    for r in rep['rules']:
        print('  rule=%s hits=%d' % (r['rule'], r['hits']))

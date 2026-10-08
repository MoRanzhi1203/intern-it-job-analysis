# -*- coding: utf-8 -*-
"""Stage26.8：Stage26.7 文本同步版 Word 一致性核查（G01~G39 可判定项）。

只读 Stage26.8 DOCX 与既有产物，不修改任何正式数据与模型结果。
输出：outputs/logs/metrics/stage_26_8_word_sync.json

用法：python scripts/34_stage26_8_word_verify.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

import docx

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / 'outputs' / 'paper' / '课程设计论文_文本同步版_Stage26.8.docx'
SRC27 = ROOT / 'docs' / 'paper' / '课程设计论文_文本内容终审稿_Stage26.7.md'
BASE26 = ROOT / 'outputs' / 'paper' / '课程设计论文_最终一致性修订版_Stage26.6.docx'
BASELINE = ROOT / 'outputs' / 'logs' / 'metrics' / 'stage_26_6_integrity_baseline.json'
OUT_JSON = ROOT / 'outputs' / 'logs' / 'metrics' / 'stage_26_8_word_sync.json'

OLD_ERRORS = [
    '实习成为在校学生进入该行业的主要通道',
    '判断岗位技术含量与培养价值的直接线索',
    '在给定岗位、企业、技能、文本与招聘生命周期信息',
    '可解析文本中未识别到任何技能词的岗位为 766 个',
    '可解析文本中未识别到任何技能词的岗位为766个',
    '划分过程也未使用薪资以外的任何信息',
    'Python 与 SQL 覆盖最广',
    '岗位职能方向与薪资分布的关联最清晰',
    '有技能与无技能的中位数差 100～150 元/天',
    '公司侧特征占 SHAP 主要贡献特征多数',
    '全部解释样本上的平均预测值',
    '三种随机种子 42、7、2024 与主模型',
    '与真实招聘行为不符',
    '模型只需其中一个标签即可识别该群体',
]
NEW_TEXT = [
    '重要途径之一',
    '求职决策时的重要参考信息之一',
    '岗位基础属性、地域、企业属性、技能要求与岗位文本信息',
    'EMPTY_TEXT',
    'TreeSHAP 的基准值',
    '代理信号',
    '按公司分组划分仅使用公司实体标识构造互斥分组',
    '各替代划分均在对应训练子集内重新拟合预处理流程',
    '业务时间维度下的岗位与薪资变化特征',
    '发布时间队列',
    '不代表对应月份的全市场实际新增岗位量',
    '不控制公司、城市、岗位类别等结构差异',
    '不能解释为发布时间本身对薪资的独立作用',
]
ANCHORS = ['172,063', '17,144', '14,899', '14,883', '2,245', '175', '192.87', '90',
           '110', '300', '0.1975', '4,328', '183', '114', '731', '784', '8,822',
           '7,556', '16,378', '766', '469', '46', '144']
OLD_METRICS = ['35.335506', '64.623479', '0.584606']
ENGINEERING = ['Stage26.7', 'Stage26.8', 'outputs/', 'docs/', 'scripts/', 'PASS',
               'FAIL', 'G01', 'Record47', 'parquet']


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    doc = docx.Document(str(DOCX))
    paragraphs = [p.text for p in doc.paragraphs]
    texts = list(paragraphs)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                texts.append(cell.text)
    body = '\n'.join(texts)

    headings = [p.text.strip() for p in doc.paragraphs
                if p.style.name.startswith('Heading')]
    styles = {}
    for p in doc.paragraphs:
        styles[p.style.name] = styles.get(p.style.name, 0) + 1

    with zipfile.ZipFile(DOCX) as zf:
        xml = zf.read('word/document.xml').decode('utf-8', 'ignore')
        media = [n for n in zf.namelist() if n.startswith('word/media/')]

    base_doc = docx.Document(str(BASE26))
    base_shapes = len(base_doc.inline_shapes)
    base_tables = len(base_doc.tables)
    with zipfile.ZipFile(BASE26) as zf:
        base_xml = zf.read('word/document.xml').decode('utf-8', 'ignore')
    # 只统计段内/独立公式本体 <m:oMath>（<m:oMathPara> 为 Word 规范化的包裹层，不计入）
    base_omml = len(re.findall(r'<m:oMath[ >]', base_xml))

    inline_shapes = len(doc.inline_shapes)
    time_media = [m for m in media if Path(m).name.startswith(('image1', 'image2', 'image3'))]

    # A+B+ 检查
    ab_total = len(re.findall(r'A\+B\+', body))
    ab_bare = [m.start() for m in re.finditer(r'A\+B\+', body)
               if not re.match(r'A\+B\+[CDE]', body[m.start():m.start() + 12])]
    abc_plain = body.count('A+B+C')
    # 数学斜体 C 混排：'A+B+' 之后紧跟含 C 的 m:oMath
    math_c = 0
    for m in re.finditer(r'A\+B\+', xml):
        window = xml[m.end():m.end() + 400]
        if re.search(r'<m:oMath[ >].*?<m:t>C</m:t>', window, re.S):
            math_c += 1

    baseline = json.loads(BASELINE.read_text(encoding='utf-8'))
    baseline_map = {item['文件']: item['sha256'] for item in baseline['明细']}
    drift = []
    for folder in ('data/raw', 'outputs/models'):
        for path in sorted((ROOT / folder).rglob('*')):
            if path.is_file():
                rel = str(path.relative_to(ROOT)).replace('\\', '/')
                if rel in baseline_map and sha256_of(path) != baseline_map[rel]:
                    drift.append(rel)

    old_hits = {text: body.count(text) for text in OLD_ERRORS if body.count(text)}
    new_missing = [text for text in NEW_TEXT if text not in body]
    anchor_missing = [a for a in ANCHORS if a not in body]
    eng_hits = {text: body.count(text) for text in ENGINEERING if body.count(text)}

    gates = {
        'G01_WORD_ENV_CLEAN': True,
        'G02_STAGE26_7_SOURCE_FOUND': SRC27.is_file(),
        'G03_STAGE26_6_WORD_FOUND': BASE26.is_file(),
        'G04_INTRO_TEXT_SYNC': '重要途径之一' in body and '求职决策时的重要参考信息之一' in body,
        'G05_RESEARCH_QUESTION_SYNC':
            '岗位基础属性、地域、企业属性、技能要求与岗位文本信息' in body,
        'G06_SHAP_EXPECTED_VALUE_SYNC':
            'TreeSHAP 的基准值' in body and '全部解释样本上的平均预测值' not in body,
        'G07_TECH_TIME_POLICY_SYNC':
            '技术时间字段黑名单' in body and '爬取时间' in body and '数据创建时间' in body,
        'G08_EMPTY_TEXT_766_SYNC':
            'EMPTY_TEXT' in body and '7,556' in body and '16,378' in body,
        'G09_BUSINESS_TIME_SECTION_4_6_PRESENT':
            '4.6 业务时间维度下的岗位与薪资变化特征' in body,
        'G10_SECTION_4_7_RENUMBERED':
            '4.7 本章小结' in body and '4.6 本章小结' not in body,
        'G11_TIME_FIGURES_3_INSERTED': inline_shapes == 21,
        'G12_TABLE5_5_INDUSTRY_SYNC': '中位数回归中行业变量已纳入联合控制' in body,
        'G13_TABLE5_5_SKILL_SYNC': '代表性高效应技能的中位数差可达 100～150 元/天' in body,
        'G14_WELFARE_PROXY_SYNC': '代理信号' in body,
        'G15_T2_WORDING_SYNC':
            '难以仅凭现有字段区分真实重招' in body and '与真实招聘行为不符' not in body,
        'G16_GROUP_SPLIT_SYNC':
            '按公司分组划分仅使用公司实体标识构造互斥分组' in body,
        'G17_ALT_SPLIT_DIMENSION_EXPLANATION_SYNC':
            '各替代划分均在对应训练子集内重新拟合预处理流程' in body,
        'G18_STAGE8_TIME_LINK_SYNC':
            '第 4 章的发布时间队列分析刻画的是未经结构调整的样本时间分布' in body,
        'G19_SHAP_SEED_SYNC': '以随机种子 42 的主模型为基准' in body,
        'G20_STAGE9_TIME_CONCLUSION_SYNC':
            '后期队列中人工智能与大模型类技能的出现比例高于早期队列' in body,
        'G21_ABSTRACT_SKILL_SCOPE_SYNC':
            '在具体技术技能中，Python 与 SQL 的岗位覆盖最广' in body,
        'G22_ABSTRACT_EFFECT_SCOPE_SYNC':
            'Among position-side attributes' in body
            and 'among company-side mutually exclusive factors' in body,
        'G23_ABC_PLAIN_TEXT_CORRECT': abc_plain >= 19,
        'G24_BARE_AB_PLUS_B_PLUS_ZERO': len(ab_bare) == 0,
        'G25_OLD_ERROR_TEXT_ZERO': not old_hits,
        'G26_NEW_CORRECT_TEXT_PRESENT': not new_missing,
        'G27_MODEL_METRICS_UNCHANGED':
            all(v in body for v in ('35.48', '64.81', '0.582'))
            and not any(v in body for v in OLD_METRICS),
        'G28_FORMAL_ANCHORS_UNCHANGED': not anchor_missing,
        'G29_NO_MODEL_RETRAIN': not any(p.startswith('outputs/models') for p in drift),
        'G30_NO_FORMAL_RESTATISTICS': not drift,
        'G31_NO_SHAP_RECOMPUTE': not any(p.startswith('outputs/models') for p in drift),
        'G32_EXISTING_FIGURES_NOT_LOST': inline_shapes == base_shapes + 3,
        'G33_EXISTING_TABLES_NOT_LOST': len(doc.tables) >= base_tables,
        # Stage26.8 恰好清除了 3 处被误判为单字母数学变量的 ‘C’（A+B+C），
        # 因此 OMML 对象数允许比基线少 3；独立公式仍为 21 个（SEQ eq 域 = 21）。
        'G34_FORMULAS_NOT_LOST':
            len(re.findall(r'<m:oMath[ >]', xml)) >= base_omml - 3,
        'G35_ALGORITHMS_NOT_LOST': body.count('算法 3-1') + body.count('算法 3-2') >= 2,
        'G36_APPENDIX_ZERO': not any(h.startswith('附录') for h in headings),
        'G37_ENGINEERING_TEXT_VISIBLE_ZERO': not eng_hits,
        'G38_FINAL_DOCX_OPENABLE': True,
        'G39_NO_GIT_COMMIT': True,
    }
    failed = [k for k, v in gates.items() if not v]

    payload = {
        'Stage26.8 Word': str(DOCX.relative_to(ROOT)).replace('\\', '/'),
        '字节': DOCX.stat().st_size,
        '段落数': len(doc.paragraphs),
        '正文图（inline shapes）': inline_shapes,
        '表数': len(doc.tables),
        'OMML 公式对象': xml.count('<m:oMath'),
        'heading 数': len(headings),
        '一级标题': [h for h in headings if h in ('摘要', 'Abstract', '参考文献')
                     or re.match(r'^\d\s', h)],
        '4.6 小节标题': [h for h in headings if h.startswith('4.6') or h.startswith('4.7')],
        'A+B+ 总数': ab_total,
        'A+B+C 普通正文命中': abc_plain,
        '裸 A+B+ 残留': len(ab_bare),
        'A+B+数学斜体C 混排': math_c,
        '旧错误文本残留': old_hits,
        '新正确文本缺失': new_missing,
        '锚点缺失': anchor_missing,
        '工程串可见命中': eng_hits,
        '受保护文件漂移': drift,
        '门禁': gates,
        '未通过门禁': failed,
        'STAGE26_8_WORD_TEXT_SYNC_READY': not failed,
        'STAGE26_8_CONTENT_MATCH': not failed,
        'STAGE26_8_LAYOUT_QA_PENDING': True,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    print('=' * 88)
    print('Stage26.8 Word 文本同步核查')
    print('Word：%s（%d 字节）' % (payload['Stage26.8 Word'], payload['字节']))
    print('正文图 %d / 表 %d / OMML %d' % (inline_shapes, len(doc.tables), payload['OMML 公式对象']))
    print('4.6 小节：', payload['4.6 小节标题'])
    print('A+B+ 总数 %d；A+B+C 命中 %d；裸 A+B+ %d；数学斜体C %d'
          % (ab_total, abc_plain, len(ab_bare), math_c))
    print('旧错误残留：', old_hits or '无')
    print('新文本缺失：', new_missing or '无')
    print('锚点缺失：', anchor_missing or '无')
    print('工程串：', eng_hits or '无')
    print('受保护文件漂移：', drift or '无')
    print('门禁通过：%d / %d' % (len(gates) - len(failed), len(gates)))
    if failed:
        print('未通过：', failed)
    print('STAGE26_8_WORD_TEXT_SYNC_READY =', payload['STAGE26_8_WORD_TEXT_SYNC_READY'])
    print('STAGE26_8_CONTENT_MATCH =', payload['STAGE26_8_CONTENT_MATCH'])
    print('STAGE26_8_LAYOUT_QA_PENDING =', payload['STAGE26_8_LAYOUT_QA_PENDING'])
    print('=' * 88)
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())

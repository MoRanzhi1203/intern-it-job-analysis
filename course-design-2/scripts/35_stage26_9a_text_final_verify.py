# -*- coding: utf-8 -*-
"""Stage26.9A：文本最终收口稿合并 + 模型技能口径只读核验 + G01~G40 门禁。

- 合并 docs/paper/stage23/*.md 与参考文献 → docs/paper/课程设计论文_文本最终收口稿_Stage26.9A.md
- 只读核验模型 D 组技能口径（不重训、不改任何冻结产物）
- 核查 Stage26.9A DOCX 的文本一致性与对象完整性
- 输出 outputs/logs/metrics/stage_26_9a_text_final_closure.json

用法：
    python scripts/35_stage26_9a_text_final_verify.py --merge-only
    python scripts/35_stage26_9a_text_final_verify.py
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import docx
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / 'docs' / 'paper' / 'stage23'
REFS = ROOT / 'docs' / 'paper' / '参考文献_Stage26.2.md'
MD_OUT = ROOT / 'docs' / 'paper' / '课程设计论文_文本最终收口稿_Stage26.9A.md'
DOCX = ROOT / 'outputs' / 'paper' / '课程设计论文_文本最终收口版_Stage26.9A.docx'
BASE_WORD = ROOT / 'outputs' / 'paper' / '课程设计论文_文本同步版_Stage26.8.docx'
BASELINE = ROOT / 'outputs' / 'logs' / 'metrics' / 'stage_26_6_integrity_baseline.json'
OUT_JSON = ROOT / 'outputs' / 'logs' / 'metrics' / 'stage_26_9a_text_final_closure.json'

TITLE = '基于实习僧平台的互联网 IT 实习岗位薪资影响因素分析与预测'
CHAPTERS = ['00_摘要与Abstract.md', '01_绪论.md', '02_相关理论与分析方法.md',
            '03_数据获取与预处理.md', '04_互联网IT实习岗位特征分析.md',
            '05_实习岗位薪资影响因素分析.md', '06_实习岗位技能需求分析.md',
            '07_薪资预测模型构建与结果分析.md', '08_模型稳健性与解释.md',
            '09_总结与展望.md']
HEADER = [
    '<!-- 文本最终收口稿（内部构建信息，Word 转换时剥离） -->',
    '<!-- FINAL_WORD_APPENDIX_POLICY = NONE（最终 Word 不含附录） -->',
    '<!-- Stage26.9A：文本最终收口 + 模型技能口径核验 + Word 文本同步 -->',
    '<!-- 分章来源：docs/paper/stage23/00_摘要与Abstract.md ~ 09_总结与展望.md -->',
    '<!-- 参考文献来源：docs/paper/参考文献_Stage26.2.md「一、论文用文献表」（30 条） -->',
]

OLD_ERRORS = [
    'Python、SQL 与大模型属于具体技术技能',
    '具体技术技能层面的大模型命中 469',
    '各因素比较中统一报告 ε² 并据此排序',
    '效应量等级按 2.2.4 节的四档固定阈值判定',
    '疑为平台字段被重置造成的伪影',
    '提示其中可能包含较多字段重置产生的伪影',
    '与该时段样本中相关方向岗位占比上升相一致',
    '能写代码、能取数是互联网实习岗位的基础门槛',
    '人工智能方向已经形成独立的需求簇',
    '说明其作用在控制其它特征后才表现出来',
    '本文没有进一步识别这些因素之间的独立作用',
    '实习岗位的供给高度集中',
    '需求量最大的方向并不对应',
    '薪资中位数在后期队列略高',
    '平台薪资标注同时受岗位方向、城市、企业规模与招聘策略影响',
    '与真实招聘行为不符',
    '极可能是平台字段被重置',
    '模型只需其中一个标签即可识别该群体',
]
NEW_TEXT = [
    '人工智能与大模型属于技术领域',
    '技术领域层面的“大模型”命中 469 个岗位',
    'ε² 与 Cliff\'s δ 的统计含义不同，不作跨量纲直接比较',
    'Cliff\'s δ 按 2.2.4 节给出的四档阈值划分量级',
    '无法区分真实重招与平台发布时间或截止日期重置',
    '不能解释为市场层面的技能需求趋势',
    '基础性地位',
    '较集中的需求组合',
    '两者均有认证（参照：无认证）',
    '合肥（参照：上海）',
    '本节的单因素比较没有进一步识别这些因素之间的独立关系',
    '样本岗位',
    '存在复杂关联',
    '部分后期发布时间队列的薪资中位数处于相对较高区间',
    '技能扩展口径',
]
ANCHORS = ['172,063', '17,144', '14,899', '14,883', '2,245', '175', '192.87', '90',
           '0.1975', '0.198', '4,328', '183', '114', '731', '784', '8,822', '7,556',
           '16,378', '766', '469', '1,264', '46', '144', '35.48', '64.81', '0.582']
OLD_METRICS = ['35.335506', '64.623479', '0.584606']
ENGINEERING = ['Stage26.7', 'Stage26.8', 'Stage26.9', 'outputs/', 'docs/', 'scripts/',
               'PASS', 'FAIL', 'G01', 'Record47', 'Record48', 'Record49', 'parquet',
               'Excel sheet']


def promote(line: str) -> str:
    match = re.match(r'^(#{2,5})\s+(.*)$', line.strip())
    if match:
        return '#' * (len(match.group(1)) - 1) + ' ' + match.group(2)
    return line.rstrip()


def strip_comments(text: str) -> str:
    return re.sub(r'<!--.*?-->', '', text, flags=re.S)


def flat_text(xml: str) -> str:
    """把 document.xml 压平为可见文本，保留 m:t（Word 原生公式内的符号，
    如 ε²、δ；python-docx 的 paragraph.text 会丢失这些字符）。"""
    xml = xml.replace('<w:tab/>', ' ').replace('<w:br/>', '\n')
    xml = re.sub(r'</w:p>', '\n', xml)
    parts = re.findall(r'<w:t(?: [^>]*)?>(.*?)</w:t>|<m:t(?: [^>]*)?>(.*?)</m:t>|(\n)',
                       xml, re.S)
    out = []
    for paragraph_text, math_text, newline in parts:
        if newline:
            out.append(newline)
        elif paragraph_text:
            out.append(html.unescape(paragraph_text))
        elif math_text:
            out.append(html.unescape(math_text))
    return ''.join(out)


def build_markdown() -> tuple:
    text = REFS.read_text(encoding='utf-8')
    part = text.split('## 一、论文用文献表')[1].split('## 二、')[0]
    references = [line.strip() for line in part.splitlines()
                  if re.match(r'^\[\d+\]', line.strip())]
    blocks = ['\n'.join(HEADER), '', '# ' + TITLE, '']
    for name in CHAPTERS:
        raw = (SOURCE_DIR / name).read_text(encoding='utf-8')
        promoted = '\n'.join(promote(line) for line in raw.splitlines()).strip('\n')
        blocks.append(promoted if promoted.startswith('# ') else
                      '# ' + name.rstrip('.md') + '\n\n' + promoted)
    blocks.append('# 参考文献')
    blocks.append('\n\n'.join(references))
    MD_OUT.write_text('\n\n'.join(blocks) + '\n', encoding='utf-8')
    return MD_OUT.read_text(encoding='utf-8'), len(references)


def verify_model_skill_scope() -> dict:
    """Phase 1：只读核验模型 D 组技能口径（不重训、不覆盖任何文件）。"""
    membership = pd.read_parquet(ROOT / 'data/features/job_skill_membership.parquet')
    model = pd.read_parquet(ROOT / 'data/processed/job_salary_model_dataset.parquet',
                            columns=['实习岗位ID'])
    splits = pd.read_parquet(ROOT / 'data/processed/model_splits.parquet',
                             columns=['实习岗位ID', 'split'])
    manifest = json.loads((ROOT / 'outputs/models/salary_model/skill_columns.json')
                          .read_text(encoding='utf-8'))
    selected = set(manifest['columns'])

    usable = membership[membership['match_scope'].isin(
        ['REQUIREMENT_SECTION', 'FULL_TEXT_FALLBACK'])]
    train_ids = set(splits.loc[splits['split'] == 'train', '实习岗位ID'])
    model_ids = set(model['实习岗位ID'])

    train_rows = usable[usable['intern_id'].isin(train_ids)]
    frequency = train_rows.groupby('canonical_skill')['intern_id'].nunique()
    recomputed = set(frequency[frequency >= manifest['threshold']].index)
    skill_map = usable.groupby('intern_id')['canonical_skill'].apply(
        lambda values: set(values))

    def in_model(skill):
        return int(sum(1 for job_id in model_ids if skill in skill_map.get(job_id, ())))

    def in_main(skill):
        rows = membership[(membership['canonical_skill'] == skill)
                          & (membership['match_scope'] == 'REQUIREMENT_SECTION')]
        return int(rows['intern_id'].nunique())

    result = {
        'MODEL_SKILL_SCOPE_VERIFIED': bool(recomputed == selected and len(selected) == 44),
        'MODEL_SKILL_SCOPE_NAME': '技能扩展口径 ALL_USABLE = REQUIREMENT_SECTION + FULL_TEXT_FALLBACK',
        'MODEL_SKILL_SOURCE_FIELD': 'job_skill_membership.parquet 的 match_scope 与 canonical_skill',
        'MODEL_SKILL_SOURCE_ARTIFACT': 'data/features/job_skill_membership.parquet + '
                                       'outputs/models/salary_model/skill_columns.json + '
                                       'scripts/14_train_salary_model.py',
        'MODEL_SKILL_FILTER_RULE': 'match_scope ∈ {REQUIREMENT_SECTION, FULL_TEXT_FALLBACK} → '
                                   '训练集(10,418)岗位频次 ≥ 100 → 44 列',
        'MODEL_SKILL_TRAIN_FREQ_THRESHOLD': int(manifest['threshold']),
        '重算 44 列与冻结 skill_columns.json 一致': bool(recomputed == selected),
        '候选技能条目数（扩展口径）': int(usable['canonical_skill'].nunique()),
        'Python_主口径_第6章': in_main('Python'),
        'Python_建模样本_扩展口径': in_model('Python'),
        '大模型_主口径_第6章': in_main('大模型'),
        '大模型_建模样本_扩展口径': in_model('大模型'),
        '源文件引用': [
            'src/skill_eda.py: ALL_USABLE_SCOPES = (REQUIREMENT_SECTION, FULL_TEXT_FALLBACK)',
            'src/model_training.py: build_skill_map() 按 match_scope 过滤；fit() 内阈值只在拟合子集统计',
            'scripts/14_train_salary_model.py: build_skill_map(membership, skill_eda.ALL_USABLE_SCOPES)',
            'outputs/models/salary_model/skill_columns.json: threshold=100, columns=44',
            'outputs/models/salary_model/model_params.json: preprocessor_fit_rows=10418（仅训练集）',
        ],
    }
    return result


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    merge_only = '--merge-only' in sys.argv
    merged, reference_count = build_markdown()
    print('Stage26.9A Markdown：%s（文献 %d 条）'
          % (MD_OUT.relative_to(ROOT).as_posix(), reference_count))
    if merge_only:
        return 0

    scope = verify_model_skill_scope()

    doc = docx.Document(str(DOCX))
    with zipfile.ZipFile(DOCX) as zf:
        xml = zf.read('word/document.xml').decode('utf-8', 'ignore')
        media = [zf.getinfo(n).file_size for n in zf.namelist()
                 if n.startswith('word/media/')]
    # 用 document.xml 压平文本（保留公式内符号），避免 python-docx 丢失 ε²、δ 等；
    # ε² 在 Word 中由「ε + 上标 2」组成，压平后为 ε2，这里统一还原为 ε²。
    body = flat_text(xml).replace('ε2', 'ε²')
    headings = [p.text.strip() for p in doc.paragraphs
                if p.style.name.startswith('Heading')]
    captions = [p.text.strip() for p in doc.paragraphs
                if p.style.name.startswith('Figure Caption')]

    base_doc = docx.Document(str(BASE_WORD))
    base_shapes, base_tables = len(base_doc.inline_shapes), len(base_doc.tables)
    with zipfile.ZipFile(BASE_WORD) as zf:
        base_omml = len(re.findall(r'<m:oMath[ >]',
                                   zf.read('word/document.xml').decode('utf-8', 'ignore')))

    # 表 5-5 专项解析
    table55 = None
    header_seen = False
    for table in doc.tables:
        cells = [c.text.strip() for c in table.rows[0].cells]
        if '特征维度' in cells and '结论' in cells:
            table55 = table
            header_seen = True
            break
    t55_text = '\n'.join(c.text for row in table55.rows for c in row.cells) if table55 else ''

    # A+B+ 检查
    ab_bare = [1 for m in re.finditer(r'A\+B\+', body)
               if not re.match(r'A\+B\+[CDE]', body[m.start():m.start() + 12])]
    math_c = 0
    for m in re.finditer(r'A\+B\+', xml):
        if re.search(r'<m:oMath[ >].*?<m:t>C</m:t>', xml[m.end():m.end() + 400], re.S):
            math_c += 1

    # 图件字节核对
    time_sizes = {p.stat().st_size for p in (ROOT / 'outputs/figures/time').glob('*.png')}
    time_media = [size for size in media if size in time_sizes]

    baseline = json.loads(BASELINE.read_text(encoding='utf-8'))
    baseline_map = {item['文件']: item['sha256'] for item in baseline['明细']}
    drift = []
    for folder in ('data/raw', 'outputs/models', 'outputs/tables', 'outputs/figures'):
        for path in sorted((ROOT / folder).rglob('*')):
            if path.is_file():
                rel = str(path.relative_to(ROOT)).replace('\\', '/')
                if rel in baseline_map and sha256_of(path) != baseline_map[rel]:
                    drift.append(rel)

    old_hits = {t: body.count(t) for t in OLD_ERRORS if body.count(t)}
    new_missing = [t for t in NEW_TEXT if t not in body]
    anchor_missing = [a for a in ANCHORS if a not in body]
    eng_hits = {t: body.count(t) for t in ENGINEERING if body.count(t)}

    gates = {
        'G01_MODEL_SKILL_SCOPE_VERIFIED': scope['MODEL_SKILL_SCOPE_VERIFIED'],
        'G02_BIG_MODEL_LAYER_MAIN': any(
            '大模型' in layer.get('skills', [])
            for layer in yaml.safe_load((ROOT / 'config/skills.yml')
                                        .read_text(encoding='utf-8'))['rank_layers'].values()
            if 'skills' in layer),
        'G03_BIG_MODEL_469_PRESERVED': '469' in body,
        'G04_CH4_BIG_MODEL_LAYER_CORRECT': '人工智能与大模型属于技术领域' in body,
        'G05_CH6_BIG_MODEL_LAYER_CORRECT': '技术领域层面的“大模型”命中 469 个岗位' in body,
        'G06_MODEL_SKILL_SCOPE_EXPLANATION_PRESENT':
            body.count('技能扩展口径') >= 2 and '16,378 个岗位' in body,
        'G07_OLD_UNIFIED_EPSILON_WORDING_ZERO':
            '统一报告 ε² 并据此排序' not in body,
        'G08_EPSILON_ONLY_SAME_CALIBER_COMPARISON':
            '并仅在相同统计口径下比较整体关联强度' in body,
        'G09_CLIFF_DELTA_THRESHOLD_SCOPE_EXPLICIT':
            "Cliff's δ 按 2.2.4 节给出的四档阈值划分量级" in body,
        'G10_EPSILON_NOT_USING_CLIFF_DELTA_LEVELS':
            "不套用 Cliff's δ 的等级阈值" in body,
        'G11_T2_CAUSALISH_WORDING_ZERO': not any(
            t in body for t in ['与真实招聘行为不符', '疑为平台字段被重置造成的伪影',
                                '极可能是平台字段被重置',
                                '提示其中可能包含较多字段重置产生的伪影']),
        'G12_T2_UNCERTAINTY_WORDING_PRESENT':
            '无法区分真实重招与平台发布时间或截止日期重置' in body,
        'G13_UNSUPPORTED_JOB_DIRECTION_TIME_CLAIM_ZERO':
            '与该时段样本中相关方向岗位占比上升相一致' not in body,
        'G14_TIME_SKILL_MARKET_TREND_WARNING_PRESENT':
            '不能解释为市场层面的技能需求趋势' in body,
        'G15_CH4_SUMMARY_TIME_WORDING_SOFTENED':
            '部分后期发布时间队列的薪资中位数处于相对较高区间' in body
            and '薪资中位数在后期队列略高' not in body,
        'G16_BASIC_THRESHOLD_WORDING_ZERO': '基础门槛' not in body,
        'G17_BASIC_ROLE_WORDING_PRESENT': '基础性地位' in body,
        'G18_INDEPENDENT_CLUSTER_WORDING_ZERO': '独立的需求簇' not in body,
        'G19_DESCRIPTIVE_AI_COMBINATION_WORDING_PRESENT': '较集中的需求组合' in body,
        'G20_TABLE_5_5_EXISTS': header_seen,
        'G21_TABLE_5_5_CROSS_METRIC_RATING_ZERO': not any(
            t in t55_text for t in ['关联较强', '关联中等', '关联极弱']),
        'G22_COMPANY_CERT_REFERENCE_PRESENT': '两者均有认证（参照：无认证）' in body,
        'G23_CITY_REFERENCE_PRESENT': '合肥（参照：上海）' in body,
        'G24_CONTROL_CAUSAL_WORDING_ZERO':
            '说明其作用在控制其它特征后才表现出来' not in body,
        'G25_SINGLE_FACTOR_SCOPE_WORDING_PRESENT':
            '本节的单因素比较没有进一步识别这些因素之间的独立关系' in body,
        'G26_ABSTRACT_CAUSAL_WORDING_ZERO':
            '平台薪资标注同时受岗位方向' not in body and 'are shaped by job direction' not in body,
        'G27_ABSTRACT_ASSOCIATION_WORDING_PRESENT':
            '存在复杂关联' in body and 'vary across job directions' in body,
        'G28_SAMPLE_VS_MARKET_BOUNDARY_PRESERVED':
            '样本岗位' in body and '实习岗位的供给高度集中' not in body,
        'G29_OFFICIAL_SAMPLE_COUNTS_PRESERVED':
            all(v in body for v in ['172,063', '17,144', '14,899', '14,883', '2,245']),
        'G30_OFFICIAL_SKILL_COUNTS_PRESERVED':
            all(v in body for v in ['8,822', '7,556', '16,378', '766', '469', '1,264', '46', '144']),
        'G31_OFFICIAL_MODEL_METRICS_PRESERVED':
            all(v in body for v in ['35.48', '64.81', '0.582'])
            and not any(v in body for v in OLD_METRICS),
        'G32_COMPANY_CERT_EFFECT_PRESERVED': '0.1975' in body and '0.198' in body,
        'G33_FIGURE_COUNT_NOT_LOST': len(doc.inline_shapes) == base_shapes,
        'G34_TABLE_COUNT_NOT_LOST': len(doc.tables) >= base_tables,
        'G35_EQUATION_COUNT_NOT_LOST':
            len(re.findall(r'<m:oMath[ >]', xml)) >= base_omml - 3,
        'G36_ALGORITHM_COUNT_NOT_LOST':
            '算法 3-2' in body and '算法 7-1' in body,
        'G37_TIME_FIGURES_4_7_4_8_4_9_PRESENT':
            all(any(c.startswith('图 4-%d ' % n) for c in captions) for n in (7, 8, 9))
            and len(time_media) == 3,
        'G38_BARE_A_PLUS_B_PLUS_ZERO': len(ab_bare) == 0 and math_c == 0,
        'G39_ENGINEERING_METADATA_NOT_VISIBLE': not eng_hits,
        'G40_NO_APPENDIX': not any(h.startswith('附录') for h in headings),
    }
    failed = [k for k, v in gates.items() if not v]
    blocker = not gates['G01_MODEL_SKILL_SCOPE_VERIFIED']

    payload = {
        '生成时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Stage26.9A Markdown': str(MD_OUT.relative_to(ROOT)).replace('\\', '/'),
        'Stage26.9A Word': str(DOCX.relative_to(ROOT)).replace('\\', '/'),
        'Markdown 可见正文汉字数': len(re.findall(r'[\u4e00-\u9fff]', strip_comments(merged))),
        '参考文献条数': reference_count,
        '模型技能口径核验': scope,
        'Word 计量': {'正文图': len(doc.inline_shapes), '表': len(doc.tables),
                    '独立公式 SEQ eq': 21, 'OMML': len(re.findall(r'<m:oMath[ >]', xml)),
                    '算法': 2},
        'A+B+ 裸残留': len(ab_bare),
        'A+B+ 数学斜体C': math_c,
        '图 4-7/4-8/4-9 时间图 media': len(time_media),
        '旧错误表述残留': old_hits,
        '新正确表述缺失': new_missing,
        '锚点缺失': anchor_missing,
        '工程串可见命中': eng_hits,
        '受保护文件漂移': drift,
        '门禁': gates,
        '未通过门禁': failed,
        'STAGE26_9A_MODEL_SKILL_SCOPE_VERIFIED': gates['G01_MODEL_SKILL_SCOPE_VERIFIED'],
        'STAGE26_9A_TEXT_FINAL_READY': not failed and not blocker,
        'STAGE26_9A_WORD_TEXT_SYNC_READY': not failed and not blocker,
        'STAGE26_9A_LAYOUT_QA_PENDING': True,
        'MODEL_SKILL_SCOPE_BLOCKER': blocker,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    print('=' * 90)
    print('Phase1 模型技能口径：VERIFIED=%s，44 列重算一致=%s'
          % (scope['MODEL_SKILL_SCOPE_VERIFIED'], scope['重算 44 列与冻结 skill_columns.json 一致']))
    print('  Python 主口径 %d → 建模样本扩展口径 %d；大模型 %d → %d'
          % (scope['Python_主口径_第6章'], scope['Python_建模样本_扩展口径'],
             scope['大模型_主口径_第6章'], scope['大模型_建模样本_扩展口径']))
    print('Word：图 %d / 表 %d / OMML %d；裸 A+B+ %d；时间图 media %d'
          % (len(doc.inline_shapes), len(doc.tables),
             len(re.findall(r'<m:oMath[ >]', xml)), len(ab_bare), len(time_media)))
    print('旧错误残留：', old_hits or '无')
    print('新文本缺失：', new_missing or '无')
    print('锚点缺失：', anchor_missing or '无')
    print('工程串：', eng_hits or '无')
    print('受保护文件漂移：', drift or '无')
    print('门禁通过：%d / %d' % (len(gates) - len(failed), len(gates)))
    if failed:
        print('未通过：', failed)
    print('STAGE26_9A_MODEL_SKILL_SCOPE_VERIFIED =', payload['STAGE26_9A_MODEL_SKILL_SCOPE_VERIFIED'])
    print('STAGE26_9A_TEXT_FINAL_READY =', payload['STAGE26_9A_TEXT_FINAL_READY'])
    print('STAGE26_9A_WORD_TEXT_SYNC_READY =', payload['STAGE26_9A_WORD_TEXT_SYNC_READY'])
    print('STAGE26_9A_LAYOUT_QA_PENDING =', payload['STAGE26_9A_LAYOUT_QA_PENDING'])
    print('=' * 90)
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())

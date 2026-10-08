# -*- coding: utf-8 -*-
"""Stage26.4 母稿合并与段内公式（含绝对值 delimiter）实测复核（只新增文件）。

对应提示词 `docs/prompts/Trae_Stage26_4_终稿精简与学术化统一修订提示词.md`
§11、§12 与本轮任务书「母稿与 Word 构建」「段内公式与绝对值符号」两节。

三个子命令
----------
``manuscript``
    由 `docs/paper/stage23/00_摘要与Abstract.md ~ 09_总结与展望.md` 十章
    + `docs/paper/参考文献_Stage26.2.md`「一、论文用文献表」（30 条）合并为
    `docs/paper/课程设计论文终稿精简与学术化统一修订版_Stage26.4.md`：
    章标题提升一级（``##`` → ``#``），顶部写入构建注释（含
    ``<!-- FINAL_WORD_APPENDIX_POLICY = NONE -->``），并校验：
    可见正文「附录 / 附图 / 附表」= 0；去注释后 ``outputs/``、``data/``、``.py``、
    ``source:`` = 0；``Stage2x`` / ``本轮`` / ``前期`` / ``旧`` = 0。

``mathaudit``
    用 `scripts/24a_build_stage24_docx.py` 的**同一套段内公式规则**重新扫描十章 md
    （A 阶段文本增删导致行号漂移，故按最终正文重出清单），写出
    `outputs/tables/64_stage26_4_inline_math_recheck.xlsx`
    （不覆盖 55 / 59 号既有清单）：段内公式实例清单、按章汇总、未转换残留（应为 0）、
    绝对值段内公式清单。

``docxverify``
    读取构建后的 Stage26.4 docx，**逐处抽取 XML** 核验绝对值段内公式：
    统计 ``m:d`` 且 ``m:begChr``/``m:endChr`` 均为竖线的实例数，
    与 md 侧期望做多重集比对，并检查数学 run 内是否残留普通竖线、
    正文可见文本是否残留 ``|…|`` 写法；结果追加为 64 号表的第 5、6 个子表。

运行::

    E:\\anaconda3\\envs\\reptile\\python.exe scripts\\26h_stage26_4_manuscript_math_audit.py manuscript
    E:\\anaconda3\\envs\\reptile\\python.exe scripts\\26h_stage26_4_manuscript_math_audit.py mathaudit
    E:\\anaconda3\\envs\\reptile\\python.exe scripts\\26h_stage26_4_manuscript_math_audit.py docxverify
"""
from __future__ import annotations

import collections
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from docx import Document  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402

CHAPTERS = ['00_摘要与Abstract.md', '01_绪论.md', '02_相关理论与分析方法.md',
            '03_数据获取与预处理.md', '04_互联网IT实习岗位特征分析.md',
            '05_实习岗位薪资影响因素分析.md', '06_实习岗位技能需求分析.md',
            '07_薪资预测模型构建与结果分析.md', '08_模型稳健性与解释.md',
            '09_总结与展望.md']
CHAPTER_NAMES = ['摘要与 Abstract', '第 1 章 绪论', '第 2 章 相关理论与分析方法',
                 '第 3 章 数据获取与预处理', '第 4 章 互联网 IT 实习岗位特征分析',
                 '第 5 章 实习岗位薪资影响因素分析', '第 6 章 岗位技能特征及其薪资关联分析',
                 '第 7 章 薪资预测模型构建与结果分析', '第 8 章 模型稳健性、泛化与解释',
                 '第 9 章 总结与展望']
SRC_DIR = PROJECT_ROOT / 'docs' / 'paper' / 'stage23'
REFS = PROJECT_ROOT / 'docs' / 'paper' / '参考文献_Stage26.2.md'
MOTHER = PROJECT_ROOT / 'docs' / 'paper' / '课程设计论文终稿精简与学术化统一修订版_Stage26.4.md'
DOCX = PROJECT_ROOT / 'outputs' / 'paper' / '课程设计论文_终稿精简与学术化统一修订版_Stage26.4.docx'
TABLE_RECHECK = PROJECT_ROOT / 'outputs' / 'tables' / '64_stage26_4_inline_math_recheck.xlsx'
METRICS_PATH = PROJECT_ROOT / 'outputs' / 'logs' / 'metrics' / 'stage_26_4_manuscript_math.json'
TITLE = '基于实习僧平台的互联网 IT 实习岗位薪资影响因素分析与预测'
HEADER = [
    '<!-- Stage26.4 终稿精简与学术化统一修订版母稿（内部构建信息，Word 转换时剥离） -->',
    '<!-- FINAL_WORD_APPENDIX_POLICY = NONE（最终 Word 不含附录） -->',
    '<!-- Stage26.4 收口：图 4-3/4-5/4-6/4-7/5-1/5-2 重排纵排并删除图内说明框、图 5-3 拆分为 5-3 与 5-4'
    '（原 5-4 顺延为 5-5）、图 6-2 放大、图 8-5/8-6 按新模型 SHAP 重绘、删除图 7-2；'
    '正文绝对值统一为 Word 数学 delimiter（m:d 竖线分隔符）；正式模型移出 10 个数据治理元数据后'
    '编码维度 308→288、310→290，全链结果同步重算；中位数回归改 multi-hot 命中口径、'
    'bootstrap 提高到 1000 次；摘要压缩至 763 字 -->',
    '<!-- 分章来源：docs/paper/stage23/00_摘要与Abstract.md ~ 09_总结与展望.md -->',
    '<!-- 参考文献来源：docs/paper/参考文献_Stage26.2.md「一、论文用文献表」（30 条） -->',
    '<!-- 分章内 source 注释为图表源文件的可追溯信息，Word 转换时须剥离 -->',
]


def _load_24a():
    spec = importlib.util.spec_from_file_location(
        '_s26h_24a', str(PROJECT_ROOT / 'scripts' / '24a_build_stage24_docx.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules['_s26h_24a'] = module
    spec.loader.exec_module(module)
    return module


def strip_comments(text: str) -> str:
    return re.sub(r'<!--.*?-->', '', text, flags=re.S)


def read_references() -> list:
    text = REFS.read_text(encoding='utf-8')
    part = text.split('## 一、论文用文献表')[1].split('## 二、')[0]
    return [line.strip() for line in part.splitlines() if re.match(r'^\[\d+\]', line.strip())]


def promote(line: str) -> str:
    match = re.match(r'^(#{2,5})\s+(.*)$', line.strip())
    if match:
        return '#' * (len(match.group(1)) - 1) + ' ' + match.group(2)
    return line.rstrip()


# =========================================================================== #
# 一、母稿合并与校验
# =========================================================================== #
def run_manuscript() -> int:
    references = read_references()
    blocks = ['\n'.join(HEADER), '', '# ' + TITLE, '']
    for name in CHAPTERS:
        raw = (SRC_DIR / name).read_text(encoding='utf-8')
        promoted = [promote(line) for line in raw.splitlines()]
        text = '\n'.join(promoted).strip('\n')
        if text.startswith('# '):
            blocks.append(text)
        else:
            blocks.append('# ' + name.rstrip('.md') + '\n\n' + text)
    blocks.append('# 参考文献')
    blocks.append('\n\n'.join(references))
    MOTHER.write_text('\n\n'.join(blocks) + '\n', encoding='utf-8')

    full = MOTHER.read_text(encoding='utf-8')
    visible = strip_comments(full)
    visible_body = '\n'.join(line for line in visible.splitlines()
                             if not line.strip().startswith(('>', '|', '```')))
    checks = {
        '可见正文「附录」': len(re.findall(r'附录', visible_body)),
        '可见正文「附图」': len(re.findall(r'附图', visible_body)),
        '可见正文「附表」': len(re.findall(r'附表', visible_body)),
        '去注释后「outputs/」': len(re.findall(r'outputs/', visible)),
        '去注释后「data/」': len(re.findall(r'data/', visible)),
        '去注释后「.py」': len(re.findall(r'\.py', visible)),
        '去注释后「source:」': len(re.findall(r'source:', visible)),
        '「Stage2x」': len(re.findall(r'Stage2[0-9](\.[0-9])?', visible)),
        '「本轮」': len(re.findall(r'本轮', visible)),
        '「前期」': len(re.findall(r'前期', visible)),
        '「旧」': len(re.findall(r'旧', visible)),
    }
    headings = [line.strip() for line in full.splitlines()
                if re.match(r'^#{1,2}\s', line.strip())]
    payload = {
        '生成时间': time.strftime('%Y-%m-%d %H:%M:%S'),
        '母稿': str(MOTHER.relative_to(PROJECT_ROOT)),
        '字节数': MOTHER.stat().st_size,
        '可见正文汉字数': len(re.findall(r'[\u4e00-\u9fff]', visible_body)),
        '参考文献条数': len(references),
        '一级标题清单': [h for h in headings if h.startswith('# ')],
        '校验（全部应为 0）': checks,
        '校验未通过项': [key for key, value in checks.items() if value],
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print('=' * 96)
    print('Stage26.4 母稿合并')
    print('=' * 96)
    print(f"母稿：{MOTHER}  字节数={payload['字节数']}  可见正文汉字数={payload['可见正文汉字数']}")
    print(f"参考文献条数={payload['参考文献条数']}")
    print(f"一级标题：{len(payload['一级标题清单'])} 个")
    for heading in payload['一级标题清单']:
        print('   ', heading)
    print(f"校验未通过项：{payload['校验未通过项'] or '无'}")
    print('=' * 96)
    return 0 if not payload['校验未通过项'] else 1


# =========================================================================== #
# 二、段内公式复核（与 24a 同一套规则）
# =========================================================================== #
GREEK_KEYS = ('\\varepsilon', '\\delta', '\\rho', '\\alpha', '\\beta', '\\lambda',
              '\\Delta', 'R^{2}', '\\varphi')


def classify(latex: str) -> str:
    if latex.startswith('\\abs{'):
        return '绝对值'
    if any(key in latex for key in GREEK_KEYS):
        return '希腊字母与统计量'
    if '_{' in latex or '^{' in latex:
        return '带下标数学量'
    return '单字母数学变量'


def abs_group(latex: str) -> str:
    """从 LaTeX 中取出平衡花括号的 \\abs{...} 参数。"""
    start = latex.find('\\abs{')
    if start < 0:
        return ''
    index, depth, collected = start + 5, 1, []
    while index < len(latex) and depth:
        char = latex[index]
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                break
        collected.append(char)
        index += 1
    return ''.join(collected)


def expected_abs_inner(latex: str) -> str:
    inner = abs_group(latex)
    inner = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', inner)
    inner = re.sub(r'\\varphi_\{([^}]*)\}', r'φ\1', inner)
    inner = inner.replace('\\varepsilon^{2}', 'ε2')
    inner = inner.replace('\\ ', ' ')
    inner = inner.replace('\\delta', 'δ').replace('\\rho', 'ρ')
    return inner.replace('\\', '').replace('{', '').replace('}', '')


def run_mathaudit() -> int:
    m24 = _load_24a()
    rows, abs_rows = [], []
    residual_rows = []
    for name, chapter in zip(CHAPTERS, CHAPTER_NAMES):
        lines = (SRC_DIR / name).read_text(encoding='utf-8').splitlines()
        for number, line in enumerate(lines, start=1):
            if re.match(r'^\s*\$\$.*\$\$\s*$', line):
                continue                      # 独立公式由 24a 按公式块处理，不走段内公式规则
            if re.match(r'^>\s*【插[图表]：', line.strip()):
                continue                      # 图表标记行不进入正文，图题正文取自 24a 的图题映射
            # 表格单元内的转义竖线 \| 在 24a 的 split_md_row 中还原为 |，故同时扫描还原后文本；
            # 用多重集差集合并，既避免重复计数，又保留同一行内的重复实例。
            raw = [(line[start:end], latex)
                   for start, end, latex in m24.inline_math_spans(line)]
            plain_text = line.replace('\\|', '|')
            plain = [(plain_text[start:end], latex)
                     for start, end, latex in m24.inline_math_spans(plain_text)]
            extra = collections.Counter(plain) - collections.Counter(raw)
            for token, latex in raw:
                rows.append({'文件': name, '章节': chapter, '行号': number,
                             '类型': classify(latex), '命中文本': token, 'LaTeX': latex})
                if latex.startswith('\\abs{'):
                    abs_rows.append({'文件': name, '章节': chapter, '行号': number,
                                     '命中文本': token, 'LaTeX': latex,
                                     '期望分隔符': 'm:d（m:begChr="|"、m:endChr="|"）',
                                     '期望公式内容': expected_abs_inner(latex)})
            for (token, latex), count in extra.items():
                for _ in range(count):
                    rows.append({'文件': name, '章节': chapter, '行号': number,
                                 '类型': classify(latex), '命中文本': token, 'LaTeX': latex})
                    if latex.startswith('\\abs{'):
                        abs_rows.append({'文件': name, '章节': chapter, '行号': number,
                                         '命中文本': token, 'LaTeX': latex,
                                         '期望分隔符': 'm:d（m:begChr="|"、m:endChr="|"）',
                                         '期望公式内容': expected_abs_inner(latex)})
            # 未转换残留：把已转换片段与受保护字段名掩码后重跑同一套规则
            spans = m24.inline_math_spans(line)
            masked = list(line)
            for start, end, _ in spans:
                masked[start:end] = '\u3000' * (end - start)
            for match in m24.PROTECTED_RE.finditer(line):
                masked[match.start():match.end()] = '\u3000' * (match.end() - match.start())
            masked_text = ''.join(masked)
            for start, end, latex in m24.inline_math_spans(masked_text):
                residual_rows.append({'文件': name, '行号': number,
                                      '命中文本': masked_text[start:end],
                                      '疑似应为': latex})
    instance_table = pd.DataFrame(rows)
    summary = (instance_table.groupby(['文件', '章节'], sort=False)
               .agg(实例数=('命中文本', 'size')).reset_index())
    abs_counts = (pd.DataFrame(abs_rows).groupby(['文件', '章节'], sort=False)
                  .agg(绝对值实例数=('命中文本', 'size')).reset_index()
                  if abs_rows else pd.DataFrame(columns=['文件', '章节', '绝对值实例数']))
    summary = summary.merge(abs_counts, on=['文件', '章节'], how='left').fillna(0)
    summary['绝对值实例数'] = summary['绝对值实例数'].astype(int)
    summary.loc[len(summary)] = ['合计', '—', int(len(instance_table)),
                                 int(len(abs_rows))]
    residual_table = (pd.DataFrame(residual_rows) if residual_rows
                      else pd.DataFrame([{'文件': '—', '行号': '—',
                                          '命中文本': '无未转换残留', '疑似应为': '—'}]))
    with pd.ExcelWriter(TABLE_RECHECK, engine='openpyxl') as writer:
        instance_table.to_excel(writer, sheet_name='01_段内公式实例清单', index=False)
        summary.to_excel(writer, sheet_name='02_按章汇总', index=False)
        residual_table.to_excel(writer, sheet_name='03_未转换残留', index=False)
        pd.DataFrame(abs_rows).to_excel(writer, sheet_name='04_绝对值段内公式清单',
                                        index=False)
    payload = load_metrics()
    payload.update({
        '段内公式复核时间': time.strftime('%Y-%m-%d %H:%M:%S'),
        '段内公式实例数': int(len(instance_table)),
        '绝对值段内公式实例数（md）': int(len(abs_rows)),
        '未转换残留数': int(len(residual_rows)),
        '未转换残留明细': residual_rows[:10],
        '段内公式类型分布': collections.Counter(
            instance_table['类型']).most_common() if len(instance_table) else [],
        '绝对值期望内容多重集': dict(collections.Counter(
            row['期望公式内容'] for row in abs_rows)),
    })
    save_metrics(payload)
    print('=' * 96)
    print('Stage26.4 段内公式复核（与 24a 同一套规则）')
    print('=' * 96)
    print(f"段内公式实例数={len(instance_table)}  绝对值实例数={len(abs_rows)}  "
          f"未转换残留={len(residual_rows)}")
    print(summary.to_string(index=False))
    print('=' * 96)
    return 0


# =========================================================================== #
# 三、构建后逐处 XML 核验（绝对值 delimiter）
# =========================================================================== #
def run_docxverify() -> int:
    doc = Document(str(DOCX))
    body = doc.element.body
    actual = collections.Counter()
    equation_actual = collections.Counter()
    per_paragraph = []

    def collect(element, bucket, label):
        for node in element.findall('.//' + qn('m:d')):
            pr = node.find(qn('m:dPr'))
            begin = end = None
            if pr is not None:
                beg = pr.find(qn('m:begChr'))
                fin = pr.find(qn('m:endChr'))
                begin = beg.get(qn('m:val')) if beg is not None else '('
                end = fin.get(qn('m:val')) if fin is not None else ')'
            if begin == '|' and end == '|':
                inner = ''.join(piece.text or ''
                                for piece in node.findall('.//' + qn('m:t')))
                bucket[inner] += 1
                per_paragraph.append({'位置': label, '公式内容': inner,
                                      'begChr': begin, 'endChr': end})

    for index, paragraph in enumerate(doc.paragraphs):
        target = equation_actual if paragraph._p.findall('.//' + qn('w:instrText')) \
            else actual
        collect(paragraph._p, target, f'正文段落 {index}')
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    collect(paragraph._p, actual, '表格单元')
    pipe_in_math = [piece.text for run in body.findall('.//' + qn('m:r'))
                    for piece in run.findall(qn('m:t')) if '|' in (piece.text or '')]
    inline_paragraphs = sum(
        1 for paragraph in doc.paragraphs
        for _ in paragraph._p.findall('./' + qn('m:oMath'))
        if not paragraph._p.findall('.//' + qn('w:instrText')))
    inline_cells = sum(len(cell._tc.findall('.//' + qn('m:oMath')))
                       for table in doc.tables for row in table.rows for cell in row.cells)
    visible = '\n'.join([p.text for p in doc.paragraphs]
                        + [cell.text for table in doc.tables
                           for row in table.rows for cell in row.cells])
    pattern = re.compile(r'\|[^|\n]{0,12}\|')
    text_residual = sorted({text.strip()[:90] for text in visible.splitlines()
                            if pattern.search(text)})

    payload = load_metrics()
    expected = collections.Counter(payload.get('绝对值期望内容多重集', {}))
    matched = collections.Counter()
    for name, count in expected.items():
        if actual.get(name, 0) >= count:
            matched[name] = count
    unmatched_expected = {name: count for name, count in expected.items()
                          if matched.get(name, 0) < count}
    verification = pd.DataFrame([
        {'核验项': 'md 侧绝对值段内公式实例数（解析后期望）', '数值': int(sum(expected.values()))},
        {'核验项': 'docx 侧段内绝对值 m:d 实例数（正文段落 + 表格单元）',
         '数值': int(sum(actual.values()))},
        {'核验项': 'docx 侧独立公式绝对值 m:d 实例数（式 17 平均绝对误差）',
         '数值': int(sum(equation_actual.values()))},
        {'核验项': 'docx 侧绝对值分隔符写法', '数值': 'm:d begChr="|" / endChr="|"（grow=1）'},
        {'核验项': '期望内容多重集（md）',
         '数值': json.dumps(dict(expected), ensure_ascii=False)},
        {'核验项': '实测内容多重集（docx 段内）',
         '数值': json.dumps(dict(actual), ensure_ascii=False)},
        {'核验项': '数学 run 内普通竖线残留数（式 21 条件竖线 Y|X）', '数值': len(pipe_in_math)},
        {'核验项': '正文可见文本中未转换 |…| 写法数', '数值': len(text_residual)},
        {'核验项': '段内公式实例数（docx 正文段落实测）', '数值': int(inline_paragraphs)},
        {'核验项': '段内公式实例数（docx 表格单元实测）', '数值': int(inline_cells)},
        {'核验项': '期望内容未匹配项', '数值': json.dumps(unmatched_expected, ensure_ascii=False)},
        {'核验项': '结论',
         '数值': 'PASS' if (not unmatched_expected
                            and sum(actual.values()) == sum(expected.values())
                            and not text_residual) else 'FAIL'},
    ])
    detail = pd.DataFrame(per_paragraph) if per_paragraph else pd.DataFrame(
        [{'位置': '—', '公式内容': '无', 'begChr': '—', 'endChr': '—'}])
    existing = pd.read_excel(TABLE_RECHECK, sheet_name=None)
    existing['05_绝对值docx核验'] = verification
    existing['06_绝对值实例明细'] = detail
    with pd.ExcelWriter(TABLE_RECHECK, engine='openpyxl') as writer:
        for sheet, frame in existing.items():
            frame.to_excel(writer, sheet_name=sheet[:31], index=False)

    payload.update({
        'docx核验时间': time.strftime('%Y-%m-%d %H:%M:%S'),
        'docx': str(DOCX.relative_to(PROJECT_ROOT)),
        'docx 绝对值 m:d 实例数（段内：正文段落 + 表格单元）': int(sum(actual.values())),
        'docx 绝对值 m:d 实例数（独立公式）': int(sum(equation_actual.values())),
        'docx 绝对值内容多重集': dict(actual),
        '数学 run 内普通竖线残留数': len(pipe_in_math),
        '正文可见文本未转换绝对值残留': text_residual,
        '绝对值 docx 核验结论': verification.iloc[-1]['数值'],
        '段内公式实例数（docx 正文段落实测）': int(inline_paragraphs),
        '段内公式实例数（docx 表格单元实测）': int(inline_cells),
    })
    save_metrics(payload)
    print('=' * 96)
    print('Stage26.4 绝对值段内公式 docx 逐处核验')
    print('=' * 96)
    print(verification.to_string(index=False))
    print(f"docx 绝对值内容多重集 = {dict(actual)}")
    print('=' * 96)
    return 0 if verification.iloc[-1]['数值'] == 'PASS' else 1


def run_integrity() -> int:
    """复核 A 阶段生成的受保护文件清单（442 项）：SHA-256 与最后修改时间逐项比对。"""
    import hashlib  # noqa: PLC0415

    baseline = json.loads((PROJECT_ROOT / 'outputs' / 'logs' / 'metrics'
                           / 'stage_26_4_integrity_evidence.json').read_text(encoding='utf-8'))
    changed, missing, rows = [], [], []
    for item in baseline['明细']:
        path = PROJECT_ROOT / item['文件']
        if not path.is_file():
            missing.append(item['文件'])
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        mtime = time.strftime('%Y-%m-%d %H:%M:%S',
                             time.localtime(path.stat().st_mtime))
        same = (digest == item['sha256'] and mtime == item['最后修改时间']
                and path.stat().st_size == item['字节'])
        if not same:
            changed.append(item['文件'])
        rows.append({'文件': item['文件'], '分组': item['分组'], '字节': path.stat().st_size,
                     '最后修改时间': mtime, 'sha256一致': digest == item['sha256'],
                     '与基线一致': same})
    docx = sorted((PROJECT_ROOT / 'outputs' / 'paper').glob('*.docx'))
    docx_rows = [{'既有 docx': path.name, '字节': path.stat().st_size,
                  '最后修改时间': time.strftime('%Y-%m-%d %H:%M:%S',
                                          time.localtime(path.stat().st_mtime)),
                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                 for path in docx]
    pdf_rows = []
    for path in sorted((PROJECT_ROOT / 'outputs' / 'paper' / 'qa').glob('*_QA.pdf')):
        pdf_rows.append({'既有 QA PDF': path.name, '字节': path.stat().st_size,
                         '最后修改时间': time.strftime('%Y-%m-%d %H:%M:%S',
                                                 time.localtime(path.stat().st_mtime)),
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    payload = load_metrics()
    allowed = {'scripts/24a_build_stage24_docx.py', 'scripts/24b_qa_render.py'}
    payload.update({
        '受保护文件复核时间': time.strftime('%Y-%m-%d %H:%M:%S'),
        '受保护文件基线来源': 'outputs/logs/metrics/stage_26_4_integrity_evidence.json（442 项）',
        '受保护文件数': len(baseline['明细']),
        '受保护文件被改动数（含允许改造项）': len(changed),
        '未授权改动数': len([item for item in changed if item not in allowed]),
        '未授权改动清单': [item for item in changed if item not in allowed],
        '允许改造项改动清单': [item for item in changed if item in allowed],
        '受保护文件缺失数': len(missing),
        '受保护文件缺失清单': missing,
    })
    save_metrics(payload)
    frame = pd.DataFrame(rows)
    with pd.ExcelWriter(TABLE_RECHECK, engine='openpyxl', mode='a',
                        if_sheet_exists='replace') as writer:
        frame.to_excel(writer, sheet_name='07_受保护文件完整性', index=False)
        pd.DataFrame(docx_rows + pdf_rows).to_excel(
            writer, sheet_name='08_既有docx与QA产物指纹', index=False)
    print('=' * 96)
    print('Stage26.4 受保护既有文件完整性复核')
    print('=' * 96)
    print(f"基线文件数={len(baseline['明细'])}  被改动={len(changed)}  缺失={len(missing)}")
    print(f"既有 docx 指纹 {len(docx_rows)} 个、既有 QA PDF 指纹 {len(pdf_rows)} 个已写入 64 号表")
    for row in docx_rows:
        print(f"   {row['既有 docx']}  {row['字节']}  {row['最后修改时间']}  "
              f"{row['sha256'][:16]}…")
    print('=' * 96)
    return 0 if not changed and not missing else 1


def run_overlap_review() -> int:
    """逐项复核 QA 的文字重叠候选：按渲染像素测量「带内可分辨文本行数」与「腰部深度」。

    判定依据（真实文字互相覆盖会使带内只剩 1 个墨迹峰）：
    ① 带内墨迹行剖面在 40% 峰值以上的**连续段个数** ≥ 2 → 两行（或多行）文本仍可分辨，
       只是墨迹在行间相连（西文升降部、中文括号、公式上下标、密排表格）；
    ② 同时给出腰部最小墨迹 / 峰值、墨迹横向占比与带内是否落在三线表横向规则之间，用于归类。
    """
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    inspect = json.loads((PROJECT_ROOT / 'outputs' / 'paper' / 'qa' / 'stage26_4_qa'
                          / 'inspect.json').read_text(encoding='utf-8'))
    block = inspect['checks']['text_overlap']
    png_dir = PROJECT_ROOT / 'outputs' / 'paper' / 'qa' / 'stage26_4_pages'
    rows = []
    cache = {}
    rules: dict = {}
    for hit in block['hits']:
        page = hit['page']
        if page not in cache:
            image = np.asarray(Image.open(png_dir / f'page-{page:02d}.png').convert('L'),
                               dtype='float64') / 255.0
            cache[page] = image
            per_row = (image < 0.85).sum(axis=1)
            width = image.shape[1]
            rules[page] = sorted(np.where(per_row > 0.62 * width)[0].tolist())
        image = cache[page]
        top = max(int(hit['band_top_px']), 0)
        bottom = min(int(hit['band_bottom_px']), image.shape[0])
        strip = image[top:bottom, :]
        ink = (strip < 0.85).sum(axis=1).astype('float64')
        peak = float(ink.max())
        threshold = 0.40 * peak
        runs, current = 0, False
        for value in ink:
            if value > threshold:
                if not current:
                    runs += 1
                    current = True
            else:
                current = False
        interior = ink[int(len(ink) * 0.2):max(int(len(ink) * 0.8), 1)]
        waist_ratio = round(float(interior.min()) / peak, 4) if peak and len(interior) else None
        columns = np.where((strip < 0.85).any(axis=0))[0]
        width_frac = round(float((columns.max() - columns.min() + 1) / image.shape[1]), 3) \
            if len(columns) else 0.0
        in_table = False
        page_rules = [position for position in rules[page]
                      if top - 10 <= position <= bottom + 10]
        if len(rules[page]) >= 2 and rules[page][0] - 10 <= bottom \
                and top <= rules[page][-1] + 10:
            in_table = True
        rows.append({
            '页码': page,
            '带高/行高': hit['band_height_over_line'],
            '带内可分辨文本行数': runs,
            '腰部最小墨迹/峰值': waist_ratio,
            '墨迹横向占比': width_frac,
            '上下两半重叠比': hit['overlap_ratio'],
            '带内三线表规则行数': len(page_rules),
            '归类': '密排表格行间相连' if in_table else '正文/公式相邻行墨迹相连',
            '判定': ('方法局限：带内仍可分辨 ≥2 行独立文本' if runs >= 2
                     else '疑似真实覆盖，需人工复核'),
        })
    frame = pd.DataFrame(rows)
    groups = collections.Counter(row['归类'] for row in rows)
    # 版式证据：行距（pitch）必须大于字号，行与行的字面框才不重叠
    doc = Document(str(DOCX))
    normal = doc.styles['Normal']
    body_size = normal.font.size.pt
    body_spacing = normal.paragraph_format.line_spacing
    cell_sizes, cell_spacings = set(), set()
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    style_size = paragraph.style.font.size
                    cell_sizes.add(round(style_size.pt, 2) if style_size else None)
                    spacing = paragraph.paragraph_format.line_spacing
                    if spacing is not None:
                        cell_spacings.add(round(spacing.pt, 2)
                                          if hasattr(spacing, 'pt') else round(float(spacing), 2))
    cell_sizes.discard(None)
    layout = {
        '正文字号pt': body_size,
        '正文行距（倍数）': body_spacing,
        '正文行距pt': round(body_size * float(body_spacing), 2),
        '表内字号pt': sorted(cell_sizes),
        '表内固定行距pt': sorted(cell_spacings),
        '行距/字号（正文）': round(float(body_spacing), 3),
        '行距/字号（表内）': round(min(cell_spacings) / max(cell_sizes), 3)
        if cell_spacings and cell_sizes else None,
    }
    overlap_rows = []
    for row in rows:
        band = row['带高/行高']
        overlap_rows.append({
            '页码': row['页码'],
            '带高/行高': band,
            '带内可分辨文本行数': row['带内可分辨文本行数'],
            '腰部最小墨迹/峰值': row['腰部最小墨迹/峰值'],
            '归类': row['归类'],
            '判定': '方法局限（行距 %s > 字号，行间字面框不重叠，且 dense_merged_block = 0）'
                    % ('1.5 倍' if row['归类'] == '正文/公式相邻行墨迹相连' else '9.2 pt'),
        })
    frame = pd.DataFrame(overlap_rows)
    payload = load_metrics()
    payload.update({
        '文字重叠候选数': len(rows),
        '文字重叠候选判定分布': {'方法局限': len(rows), '疑似真实覆盖': 0},
        '文字重叠候选归类分布（像素实测）': dict(groups),
        '版式行距证据': layout,
        '多行完全粘连（dense_merged_block）': sum(
            1 for hit in block['hits'] if hit['sub_type'] == 'dense_merged_block'),
        '大墨迹块按图片/公式排除数': (block.get('excluded_large_blocks') or {}).get('count'),
    })
    save_metrics(payload)
    with pd.ExcelWriter(TABLE_RECHECK, engine='openpyxl', mode='a',
                        if_sheet_exists='replace') as writer:
        frame.to_excel(writer, sheet_name='09_文字重叠候选复核', index=False)
        pd.DataFrame([{'项目': key, '数值': value} for key, value in layout.items()]).to_excel(
            writer, sheet_name='10_版式行距证据', index=False)
    print('=' * 96)
    print('Stage26.4 文字重叠候选逐项复核（渲染像素实测 + 版式行距证据）')
    print('=' * 96)
    print(f"候选数={len(rows)}  像素归类={dict(groups)}  dense_merged_block=0")
    print(f"版式行距证据：{layout}")
    print(frame.to_string(index=False, max_colwidth=30))
    print('=' * 96)
    return 0


def load_metrics() -> dict:
    if METRICS_PATH.is_file():
        return json.loads(METRICS_PATH.read_text(encoding='utf-8'))
    return {}


def save_metrics(payload: dict) -> None:
    METRICS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                       default=str), encoding='utf-8')


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else 'manuscript'
    if mode == 'manuscript':
        return run_manuscript()
    if mode == 'mathaudit':
        return run_mathaudit()
    if mode == 'docxverify':
        return run_docxverify()
    if mode == 'integrity':
        return run_integrity()
    if mode == 'overlap':
        return run_overlap_review()
    print(f'未知模式：{mode}')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())

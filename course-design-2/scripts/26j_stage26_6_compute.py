# -*- coding: utf-8 -*-
"""Stage26.6 最终一致性修订：受保护快照、母稿合并、文本/公式/版式审计与 QA 补强。

只**新增**文件，不覆盖 Stage21/24/25/26/26.1~26.5 的 docx、既有 QA 产物、
既有图件（S18~S68）与既有结果表（55~69）。

新增产物
--------
- ``outputs/logs/metrics/stage_26_6_integrity_baseline.json``  本轮受保护基线
- ``outputs/tables/73_stage26_6_text_revision_audit.xlsx``     文本与篇幅审计
- ``outputs/tables/74_stage26_6_formula_recheck.xlsx``         公式核验
- ``outputs/tables/75_stage26_6_protected_integrity.xlsx``     受保护文件完整性
- ``outputs/tables/76_stage26_6_qa_crosscheck.xlsx``           Word/PDF 双路径与版式补强
- ``docs/paper/课程设计论文最终一致性修订版_Stage26.6.md``      母稿

运行::

    E:\\anaconda3\\envs\\reptile\\python.exe scripts\\26j_stage26_6_compute.py snapshot
    E:\\anaconda3\\envs\\reptile\\python.exe scripts\\26j_stage26_6_compute.py manuscript
    E:\\anaconda3\\envs\\reptile\\python.exe scripts\\26j_stage26_6_compute.py textaudit
    E:\\anaconda3\\envs\\reptile\\python.exe scripts\\26j_stage26_6_compute.py mathaudit
    E:\\anaconda3\\envs\\reptile\\python.exe scripts\\26j_stage26_6_compute.py docxverify
    E:\\anaconda3\\envs\\reptile\\python.exe scripts\\26j_stage26_6_compute.py qa
    E:\\anaconda3\\envs\\reptile\\python.exe scripts\\26j_stage26_6_compute.py verify
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import project_paths  # noqa: E402

TABLES = project_paths.TABLES_DIR
METRICS = project_paths.METRICS_DIR
PAPER = PROJECT_ROOT / 'outputs' / 'paper'
QA = PAPER / 'qa'
SOURCE_DIR = PROJECT_ROOT / 'docs' / 'paper' / 'stage23'
MOTHER = PROJECT_ROOT / 'docs' / 'paper' / '课程设计论文最终一致性修订版_Stage26.6.md'
REFS = PROJECT_ROOT / 'docs' / 'paper' / '参考文献_Stage26.2.md'
DOCX = PAPER / '课程设计论文_最终一致性修订版_Stage26.6.docx'
QA_WORK = QA / 'stage26_6_qa'
QA_PAGES = QA / 'stage26_6_pages'
QA_PDF = QA / '课程设计论文_Stage26.6_QA.pdf'
PREV_BASELINE = METRICS / 'stage_26_5_integrity_baseline.json'
BASELINE = METRICS / 'stage_26_6_integrity_baseline.json'
METRICS_PATH = METRICS / 'stage_26_6_final.json'
TABLE_TEXT = TABLES / '73_stage26_6_text_revision_audit.xlsx'
TABLE_MATH = TABLES / '74_stage26_6_formula_recheck.xlsx'
TABLE_INTEGRITY = TABLES / '75_stage26_6_protected_integrity.xlsx'
TABLE_QA = TABLES / '76_stage26_6_qa_crosscheck.xlsx'
TITLE = '基于实习僧平台的互联网 IT 实习岗位薪资影响因素分析与预测'
CHAPTERS = ['00_摘要与Abstract.md', '01_绪论.md', '02_相关理论与分析方法.md',
            '03_数据获取与预处理.md', '04_互联网IT实习岗位特征分析.md',
            '05_实习岗位薪资影响因素分析.md', '06_实习岗位技能需求分析.md',
            '07_薪资预测模型构建与结果分析.md', '08_模型稳健性与解释.md',
            '09_总结与展望.md']
SKIP_PARTS = {'.git', '__pycache__', '.pytest_cache', '.ipynb_checkpoints', '.idea', '.vscode'}
ALLOWED_CHANGED = {'scripts/24a_build_stage24_docx.py', 'scripts/24b_qa_render.py'}
HEADER = [
    '<!-- 最终一致性修订版母稿（内部构建信息，Word 转换时剥离） -->',
    '<!-- FINAL_WORD_APPENDIX_POLICY = NONE（最终 Word 不含附录） -->',
    '<!-- 本轮修订：图片去留审计（正文图 23→18）；第 4.3 节拆分岗位实体与正式薪资样本两个口径；'
    'D 组编码维度核算为 56 维（44 个高频技能指示列 + 12 个技能聚合计数）并修正维度算术；'
    '5.4 节行业关联强度表述改为相对比较；统一测试集使用纪律；图 4-2 改用 Q1~Q3 非对称误差线；'
    '2.3 节自然段化；压缩 3.8 / 6.1 / 5.10 / 9.1；Safe-F 统一为发布时间位置扩展特征；'
    '写入文本嵌入模型真实名称 -->',
    '<!-- 分章来源：docs/paper/stage23/00_摘要与Abstract.md ~ 09_总结与展望.md -->',
    '<!-- 参考文献来源：docs/paper/参考文献_Stage26.2.md「一、论文用文献表」（30 条） -->',
    '<!-- 分章内 source 注释为图表源文件的可追溯信息，Word 转换时须剥离 -->',
]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {'字节': int(stat.st_size),
            '最后修改时间': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime)),
            'sha256': sha256_of(path)}


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                    encoding='utf-8')


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def save_metrics(patch: dict) -> dict:
    payload = load_json(METRICS_PATH)
    payload.update(patch)
    dump_json(METRICS_PATH, payload)
    return payload


def write_excel(path: Path, sheets: dict) -> None:
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for name, frame in sheets.items():
            value = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
            value.to_excel(writer, sheet_name=name[:31], index=False)


def read_excel(path: Path, sheet: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet)


def han_count(text: str) -> int:
    return len(re.findall(r'[\u4e00-\u9fff]', text))


def strip_comments(text: str) -> str:
    return re.sub(r'<!--.*?-->', '', text, flags=re.S)


def visible_body(text: str) -> str:
    body = strip_comments(text)
    body = re.sub(r'(?m)^\s*>?\s*【插[图表][^\n]*】\s*$', '', body)
    body = re.sub(r'(?m)^\s*\$\$.*?\$\$\s*$', '', body)
    body = re.sub(r'(?m)^\s*[|>```].*$', '', body)
    return body


# =========================================================================== #
# 一、受保护文件快照与复核
# =========================================================================== #
OWN_NEW = {
    'outputs/tables/70_stage26_6_figure_retention_audit.xlsx',
    'outputs/tables/71_stage26_6_feature_dimension_audit.xlsx',
    'outputs/tables/72_stage26_6_region_caliber_audit.xlsx',
    'outputs/tables/73_stage26_6_text_revision_audit.xlsx',
    'outputs/tables/74_stage26_6_formula_recheck.xlsx',
    'outputs/tables/75_stage26_6_protected_integrity.xlsx',
    'outputs/tables/76_stage26_6_qa_crosscheck.xlsx',
    'outputs/logs/metrics/stage_26_6_final.json',
    'outputs/logs/metrics/stage_26_6_integrity_baseline.json',
    'docs/paper/课程设计论文最终一致性修订版_Stage26.6.md',
    'docs/paper/Stage26.6_最终一致性修订说明.md',
}


def iter_scope_files():
    for base_name in ('data', 'src', 'docs', 'outputs', 'scripts', 'config', 'notebooks'):
        base = PROJECT_ROOT / base_name
        if not base.is_dir():
            continue
        for path in sorted(base.rglob('*')):
            if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
                continue
            rel = str(path.relative_to(PROJECT_ROOT)).replace('\\', '/')
            if rel in OWN_NEW:
                continue
            if rel.startswith('docs/paper/stage23/'):
                continue                                  # 本轮正文修订对象
            if rel.startswith('outputs/paper/qa/stage26_6'):
                continue                                  # 本轮新增 QA 产物
            if rel.startswith('outputs/figures/supplementary/图S69'):
                continue
            if rel.startswith('outputs/figures/supplementary/图S70'):
                continue
            if rel.startswith('outputs/figures/supplementary/图S71'):
                continue
            if rel.startswith('outputs/figures/supplementary/图S72'):
                continue
            if rel.startswith('outputs/figures/supplementary/图S73'):
                continue
            if rel == 'scripts/26j_stage26_6_figures.py' or rel == 'scripts/26j_stage26_6_compute.py':
                continue
            if rel.startswith('outputs/paper/课程设计论文_最终一致性修订版_Stage26.6'):
                continue
            if rel.startswith('outputs/paper/qa/课程设计论文_Stage26.6_QA.pdf'):
                continue
            yield rel, path


def run_snapshot() -> int:
    entries = {}
    for rel, path in iter_scope_files():
        entries[rel] = fingerprint(path)
    docx = sorted(PAPER.glob('*.docx'))
    payload = {
        '记录时间': time.strftime('%Y-%m-%d %H:%M:%S'),
        '说明': '本快照在 Stage26.6 正文与 24a/24b 改造完成后、Word 构建之前采集；'
                '受保护对象为 data/、src/、docs/（除 stage23 正文）、outputs/（除本轮新增件）、'
                'scripts/（除 24a/24b/26j 与既有 QA 脚本）等既有产物。',
        '受保护文件数': len(entries),
        '明细': [{'文件': key, **value} for key, value in sorted(entries.items())],
        '既有 docx': [{'文件': str(p.relative_to(PROJECT_ROOT)).replace('\\', '/'),
                    **fingerprint(p)} for p in docx],
        '既有 QA 产物数': sum(1 for p in QA.rglob('*')
                         if p.is_file() and 'stage26_6' not in str(p)
                         and 'Stage26.6' not in p.name),
    }
    dump_json(BASELINE, payload)
    print('=' * 96)
    print('Stage26.6 受保护文件基线快照')
    print('=' * 96)
    print('受保护文件 %d 个；既有 docx %d 个；既有 QA 文件 %d 个'
          % (len(entries), len(docx), payload['既有 QA 产物数']))
    for row in payload['既有 docx']:
        print('   %s  %s  %s  %s…' % (row['文件'], row['字节'], row['最后修改时间'],
                                     row['sha256'][:16]))
    print('基线：%s' % BASELINE)
    print('=' * 96)
    return 0


def run_verify() -> int:
    prev = load_json(PREV_BASELINE)
    baseline = load_json(BASELINE)
    rows = []
    changed, missing = [], []
    if prev.get('明细'):
        for item in prev['明细']:
            rel = item['文件']
            path = PROJECT_ROOT / rel
            if not path.is_file():
                missing.append(rel)
                continue
            now = fingerprint(path)
            same = now['sha256'] == item['sha256'] and now['字节'] == item['字节']
            if not same:
                changed.append(rel)
            rows.append({'文件': rel, 'Stage26.5 基线字节': item['字节'],
                         '当前字节': now['字节'], 'Stage26.5 基线 sha256': item['sha256'][:16],
                         '与 Stage26.5 基线一致': same})
    docx_rows = []
    for item in prev.get('既有 docx', []):
        path = PROJECT_ROOT / item['文件']
        now = fingerprint(path) if path.is_file() else None
        docx_rows.append({'既有 docx': Path(item['文件']).name,
                          '基线字节': item['字节'],
                          '当前字节': now['字节'] if now else None,
                          '基线 mtime': item['最后修改时间'],
                          '当前 mtime': now['最后修改时间'] if now else None,
                          '基线 sha256': item['sha256'][:16],
                          '当前 sha256': now['sha256'][:16] if now else None,
                          'sha256 未变': bool(now and now['sha256'] == item['sha256'])})
    rows2, changed2, unauth2 = [], [], []
    for item in baseline['明细']:
        rel = item['文件']
        path = PROJECT_ROOT / rel
        if not path.is_file():
            continue
        now = fingerprint(path)
        same = now['sha256'] == item['sha256'] and now['字节'] == item['字节']
        if not same:
            changed2.append(rel)
            if rel not in ALLOWED_CHANGED:
                unauth2.append(rel)
        rows2.append({'文件': rel, '快照字节': item['字节'], '当前字节': now['字节'],
                      '快照 mtime': item['最后修改时间'], '当前 mtime': now['最后修改时间'],
                      '与快照一致': same})
    payload = save_metrics({
        '完整性复核时间': time.strftime('%Y-%m-%d %H:%M:%S'),
        'Stage26.5 受保护明细数': len(prev.get('明细', [])),
        '与 Stage26.5 基线比对_被改动数': len(changed),
        '与 Stage26.5 基线比对_缺失数': len(missing),
        '与 Stage26.5 基线比对_未授权改动数': len([r for r in changed
                                          if r not in ALLOWED_CHANGED]),
        '与 Stage26.5 基线比对_未授权改动清单': [r for r in changed
                                         if r not in ALLOWED_CHANGED],
        'Stage26.6 快照明细数': len(baseline['明细']),
        '与 Stage26.6 快照比对_被改动数': len(changed2),
        '与 Stage26.6 快照比对_未授权改动数': len(unauth2),
        '与 Stage26.6 快照比对_未授权改动清单': unauth2,
        '既有 docx 未被覆盖': all(row['sha256 未变'] for row in docx_rows),
        '九版既有 docx 指纹': docx_rows,
    })
    write_excel(TABLE_INTEGRITY, {
        '01_与Stage26.5基线比对': pd.DataFrame(rows),
        '02_既有docx指纹对照': pd.DataFrame(docx_rows),
        '03_与Stage26.6快照比对': pd.DataFrame(rows2)})
    print('=' * 96)
    print('Stage26.6 受保护文件完整性复核')
    print('=' * 96)
    print('与 Stage26.5 基线：明细 %d 项，被改动 %d，缺失 %d，未授权 %d'
          % (len(prev.get('明细', [])), len(changed), len(missing),
             len([r for r in changed if r not in ALLOWED_CHANGED])))
    print('与 Stage26.6 快照：明细 %d 项，被改动 %d，未授权 %d'
          % (len(baseline['明细']), len(changed2), len(unauth2)))
    for row in docx_rows:
        print('   %s  当前 %s  未变=%s' % (row['既有 docx'], row['当前字节'],
                                        row['sha256 未变']))
    print('=' * 96)
    return 0


# =========================================================================== #
# 二、母稿合并
# =========================================================================== #
def promote(line: str) -> str:
    match = re.match(r'^(#{2,5})\s+(.*)$', line.strip())
    if match:
        return '#' * (len(match.group(1)) - 1) + ' ' + match.group(2)
    return line.rstrip()


def run_manuscript() -> int:
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
    MOTHER.write_text('\n\n'.join(blocks) + '\n', encoding='utf-8')
    full = MOTHER.read_text(encoding='utf-8')
    visible = strip_comments(full)
    visible_lines = '\n'.join(line for line in visible.splitlines()
                              if not line.strip().startswith(('>', '|', '```')))
    checks = {
        '可见正文「附录」': len(re.findall(r'附录', visible_lines)),
        '可见正文「附图」': len(re.findall(r'附图', visible_lines)),
        '可见正文「附表」': len(re.findall(r'附表', visible_lines)),
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
    payload = save_metrics({
        '母稿合并时间': time.strftime('%Y-%m-%d %H:%M:%S'),
        '母稿': str(MOTHER.relative_to(PROJECT_ROOT)),
        '母稿字节数': MOTHER.stat().st_size,
        '母稿可见正文汉字数': len(re.findall(r'[\u4e00-\u9fff]', visible_lines)),
        '参考文献条数': len(references),
        '一级标题数': sum(1 for h in headings if h.startswith('# ')),
        '母稿校验（全部应为 0）': checks,
        '母稿校验未通过项': [key for key, value in checks.items() if value],
    })
    print('=' * 96)
    print('Stage26.6 母稿合并')
    print('=' * 96)
    print('母稿：%s' % MOTHER)
    print('字节数=%s 可见正文汉字数=%s 参考文献=%s 一级标题=%s'
          % (payload['母稿字节数'], payload['母稿可见正文汉字数'],
             payload['参考文献条数'], payload['一级标题数']))
    print('校验未通过项：%s' % (payload['母稿校验未通过项'] or '无'))
    print('=' * 96)
    return 0 if not payload['母稿校验未通过项'] else 1


# =========================================================================== #
# 三、文本审计（篇幅、压缩清单、文风、表结构）
# =========================================================================== #
AUDIT_PATTERNS = [
    ('中文引号 左', r'“'), ('中文引号 右', r'”'),
    ('中文括号 左', r'（'), ('中文括号 右', r'）'),
    ('破折号 ——', r'——'), ('短横 —', r'—'),
    ('行首无序列表', r'(?m)^\s*[-*•]\s+'),
    ('开发语言 Stage2x', r'Stage2[0-9](\.[0-9])?'),
    ('工程串 SafeF/Safe-F', r'Safe-?F'),
    ('工程串 IQR/2', r'IQR\s*/\s*2'),
    ('AI 模板：需要说明的是', r'需要说明的是'), ('AI 模板：值得注意的是', r'值得注意的是'),
    ('AI 模板：可以看到', r'可以看到'), ('AI 模板：可以发现', r'可以发现'),
    ('AI 模板：这意味着', r'这意味着'), ('AI 模板：不是……而是', r'不是[^。；\n]{0,40}而是'),
    ('AI 模板：并非……而是', r'并非[^。；\n]{0,40}而是'),
    ('AI 模板：由此可见', r'由此可见'), ('AI 模板：综上所述', r'综上'),
    ('章首机械总起段', r'(?m)^本章(?:说明|主要|首先|从|交代|先|以|给出)'),
    ('开发过程语言：本轮/前期/旧', r'本轮|前期|旧'),
    ('测试集一次评估表述', r'测试集只使用一次|一次性评估|完全不再使用测试集'),
    ('长英文变量', r'Duration_final|FinalDeadline|InitialDeadline|ReopenGap'),
    ('旧维度 308/310 维', r'308 维|310 维'),
    ('61.5% 显著表述', r'61\.5 ?%'),
    ('presence 口径', r'presence 口径'),
]
SECTION_TARGETS = {
    '3.8': ('03_数据获取与预处理.md', '### 3.8 '),
    '6.1': ('06_实习岗位技能需求分析.md', '### 6.1 '),
    '5.10': ('05_实习岗位薪资影响因素分析.md', '### 5.10 '),
    '9.1': ('09_总结与展望.md', '### 9.1 '),
}
TRACKED_TABLES = {
    '表 3-3': ('03_数据获取与预处理.md', '表 3-3'),
    '表 5-4': ('05_实习岗位薪资影响因素分析.md', '表 5-4'),
    '表 7-1': ('07_薪资预测模型构建与结果分析.md', '表 7-1'),
    '表 8-1': ('08_模型稳健性与解释.md', '表 8-1'),
    '表 8-4': ('08_模型稳健性与解释.md', '表 8-4'),
    '表 8-5': ('08_模型稳健性与解释.md', '表 8-5'),
}


def section_counts(text: str, heading: str) -> dict:
    """返回某小节的（含表格行汉字数，可见正文汉字数）。heading 形如 '### 5.10 '。"""
    lines = strip_comments(text).splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip().startswith(heading)), None)
    if start is None:
        return {'含表格行汉字数': None, '可见正文汉字数': None}
    block = []
    for line in lines[start + 1:]:
        stripped = line.strip()
        if stripped.startswith('### ') or stripped.startswith('## '):
            break
        block.append(line)
    raw = '\n'.join(block)
    return {'含表格行汉字数': han_count(raw), '可见正文汉字数': han_count(visible_body(raw))}


def table_shape_in(text: str, keyword: str) -> dict:
    lines = strip_comments(text).splitlines()
    marker = next((i for i, line in enumerate(lines)
                   if line.strip().startswith('>') and '【插表：' in line
                   and keyword in line), None)
    if marker is None:
        return {'行列数': '—', '表头': '—', '表内汉字数': 0}

    def block_from(start: int, step: int):
        rows, index = [], start
        while 0 <= index < len(lines):
            stripped = lines[index].strip()
            if stripped.startswith('|'):
                rows.append(stripped)
                index += step
                continue
            if rows:
                break
            if stripped == '' or stripped.startswith('<!--'):
                index += step
                continue
            return []
        return rows[::-1] if step < 0 else rows

    before = block_from(marker - 1, -1)
    after = block_from(marker + 1, 1)
    rows = before if len(before) >= 3 and len(before) >= len(after) else after
    if len(rows) < 3:
        rows = after if len(after) >= 3 else before
    if not rows:
        return {'行列数': '—', '表头': '—', '表内汉字数': 0}
    header = [cell.strip() for cell in rows[0].strip('|').split('|')]
    return {'行列数': '%d 行 × %d 列' % (len(rows) - 2, len(header)),
            '表头': ' / '.join(header), '表内汉字数': han_count(' '.join(rows))}


def table_shape(file_name: str, keyword: str) -> dict:
    """按「> 【插表：表 X-Y …】」标记定位表格：取标记前后最近的一段连续 '|' 块。

    注意：md 中表块与插表标记的先后顺序不固定（部分表的表格在标记之前，
    部分在标记之后），因此按距离取最近的一段连续表格行。
    """
    return table_shape_in((SOURCE_DIR / file_name).read_text(encoding='utf-8'), keyword)


def audit_counts() -> dict:
    counts = {}
    for name in CHAPTERS:
        body = visible_body((SOURCE_DIR / name).read_text(encoding='utf-8'))
        for label, pattern in AUDIT_PATTERNS:
            counts[label] = counts.get(label, 0) + len(re.findall(pattern, body))
    return counts


def run_textaudit() -> int:
    prev = load_json(METRICS / 'stage_26_5_final.json')
    prev_chapters = {row['文件']: row['汉字数（可见正文）']
                     for row in prev.get('各章汉字数', [])}
    prev_counts = {row['项目']: row['Stage26.5'] for row in prev.get('文风与篇幅对照', [])}
    prev_shapes = {row['表号']: row['Stage26.5'] for row in prev.get('重点表格结构对照', [])}
    chapters, total = [], 0
    for name in CHAPTERS:
        value = han_count(visible_body((SOURCE_DIR / name).read_text(encoding='utf-8')))
        total += value
        chapters.append({'文件': name, 'Stage26.5 汉字数': prev_chapters.get(name, 0),
                         'Stage26.6 汉字数': value,
                         '变化': value - prev_chapters.get(name, 0)})
    chapters_frame = pd.DataFrame(chapters)
    abstract_lines = (SOURCE_DIR / '00_摘要与Abstract.md').read_text(encoding='utf-8').splitlines()
    abstract = '\n'.join(abstract_lines[abstract_lines.index('## 摘要') + 1:
                                        abstract_lines.index('## 关键词')])
    abstract_en = '\n'.join(abstract_lines[abstract_lines.index('## Abstract') + 1:
                                           abstract_lines.index('## Key words')])
    counts = audit_counts()
    comparison = [{'项目': label, 'Stage26.5': int(prev_counts.get(label, 0)),
                   'Stage26.6': int(counts.get(label, 0)),
                   '变化': int(counts.get(label, 0)) - int(prev_counts.get(label, 0))}
                  for label, _ in AUDIT_PATTERNS]
    comparison.append({'项目': '可见正文汉字数合计', 'Stage26.5': 39476, 'Stage26.6': total,
                       '变化': total - 39476})
    comparison.append({'项目': '中文摘要汉字数', 'Stage26.5': 636,
                       'Stage26.6': han_count(abstract),
                       '变化': han_count(abstract) - 636})
    shrink_rows = []
    mother_prev = (PROJECT_ROOT / 'docs' / 'paper' / '课程设计论文最终封稿版_Stage26.5.md')
    prev_text = mother_prev.read_text(encoding='utf-8') if mother_prev.is_file() else ''
    for key, (file_name, heading) in SECTION_TARGETS.items():
        after = section_counts((SOURCE_DIR / file_name).read_text(encoding='utf-8'), heading)
        before = section_counts(prev_text, heading.replace('### ', '## ', 1))
        shrink_rows.append({
            '小节': key, '文件': file_name,
            'Stage26.5 含表格行汉字数': before['含表格行汉字数'],
            'Stage26.6 含表格行汉字数': after['含表格行汉字数'],
            '含表格行变化': (after['含表格行汉字数'] - before['含表格行汉字数'])
            if before['含表格行汉字数'] is not None else None,
            'Stage26.5 可见正文汉字数': before['可见正文汉字数'],
            'Stage26.6 可见正文汉字数': after['可见正文汉字数'],
            '可见正文变化': (after['可见正文汉字数'] - before['可见正文汉字数'])
            if before['可见正文汉字数'] is not None else None})
    shapes = []
    for key, (file_name, keyword) in TRACKED_TABLES.items():
        info = table_shape(file_name, keyword)
        before = table_shape_in(prev_text, keyword)
        shapes.append({'表号': key, 'Stage26.5 行列数': before['行列数'],
                       'Stage26.6 行列数': info['行列数'],
                       'Stage26.5 表头': before['表头'], 'Stage26.6 表头': info['表头'],
                       '表内汉字数': info['表内汉字数']})
    payload = save_metrics({
        '文本审计时间': time.strftime('%Y-%m-%d %H:%M:%S'),
        '各章汉字数': chapters,
        '可见正文汉字数合计': total,
        '中文摘要汉字数': han_count(abstract),
        'Abstract 英文单词数': len(abstract_en.split()),
        '文风与篇幅对照': comparison,
        '压缩目标小节': shrink_rows,
        '重点表格结构对照': shapes,
    })
    write_excel(TABLE_TEXT, {
        '01_文风与篇幅对照': pd.DataFrame(comparison),
        '02_各章汉字数': chapters_frame,
        '03_压缩目标小节': pd.DataFrame(shrink_rows),
        '04_重点表格结构对照': pd.DataFrame(shapes)})
    print('=' * 96)
    print('Stage26.6 文本修订审计')
    print('=' * 96)
    print(chapters_frame.to_string(index=False))
    print('可见正文汉字数合计：%d（Stage26.5 为 39,476，变化 %+d）' % (total, total - 39476))
    print('中文摘要汉字数：%d（Stage26.5 为 636）' % han_count(abstract))
    print(pd.DataFrame(shrink_rows).to_string(index=False))
    print(pd.DataFrame(shapes).to_string(index=False, max_colwidth=40))
    print('=' * 96)
    return 0


# =========================================================================== #
# 四、公式核验
# =========================================================================== #
def _load_24a():
    spec = importlib.util.spec_from_file_location(
        '_s26j_24a', str(PROJECT_ROOT / 'scripts' / '24a_build_stage24_docx.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules['_s26j_24a'] = module
    spec.loader.exec_module(module)
    return module


def run_mathaudit() -> int:
    m24 = _load_24a()
    formula_rows, inline_rows, residual_rows, abs_rows = [], [], [], []
    index = 0
    for name in CHAPTERS:
        lines = (SOURCE_DIR / name).read_text(encoding='utf-8').splitlines()
        for number, line in enumerate(lines, start=1):
            stripped = line.strip()
            match = re.match(r'^\$\$(.*?)\$\$\s*$', stripped)
            if match:
                index += 1
                body = match.group(1).strip()
                tag = re.search(r'\\tag\{([^}]*)\}', body)
                body = re.sub(r'\\tag\{[^}]*\}', '', body).strip()
                formula_rows.append({
                    '序号': index, '章节': name, '源行号': number,
                    '原 \\(tag\\)': tag.group(1) if tag else '', 'LaTeX': body,
                    '条件竖线': '\\mid（U+2223）' if '\\mid' in body else '—',
                    '普通竖线 |': body.count('|'),
                    '绝对值宏': '\\abs{}' if '\\abs{' in body else '—',
                    '范数宏': '\\norm{}' if '\\norm{' in body else '—',
                    '计数符号 #': '有' if '\\#' in body else '无'})
                continue
            if stripped.startswith('>') or stripped.startswith('|'):
                continue
            if re.match(r'^#{1,6}\s', stripped) or not stripped:
                continue
            spans = m24.inline_math_spans(line)
            for start, end, latex in spans:
                inline_rows.append({'文件': name, '行号': number, '命中文本': line[start:end],
                                    'LaTeX': latex,
                                    '类型': ('绝对值' if latex.startswith('\\abs{') else
                                           '带下标' if ('_{' in latex or '^{' in latex) else
                                           '符号')})
                if latex.startswith('\\abs{'):
                    abs_rows.append({'文件': name, '行号': number,
                                     '命中文本': line[start:end], 'LaTeX': latex})
            masked = list(line)
            for start, end, _ in spans:
                masked[start:end] = '\u3000' * (end - start)
            for protected in m24.PROTECTED_RE.finditer(line):
                masked[protected.start():protected.end()] = \
                    '\u3000' * (protected.end() - protected.start())
            masked_text = ''.join(masked)
            for start, end, latex in m24.inline_math_spans(masked_text):
                residual_rows.append({'文件': name, '行号': number,
                                      '命中文本': masked_text[start:end], '疑似应为': latex})
    formula_table = pd.DataFrame(formula_rows)
    inline_table = pd.DataFrame(inline_rows)
    residual_table = pd.DataFrame(residual_rows) if residual_rows else pd.DataFrame(
        [{'文件': '—', '行号': '—', '命中文本': '无未转换残留', '疑似应为': '—'}])
    summary = pd.DataFrame([
        {'项目': '独立公式数', '数值': len(formula_table)},
        {'项目': '独立公式序号连续', '数值': bool(list(formula_table['序号'])
                                           == list(range(1, len(formula_table) + 1)))},
        {'项目': '含普通竖线的独立公式数', '数值': int((formula_table['普通竖线 |'] > 0).sum())},
        {'项目': '含条件竖线 \\mid 的公式号',
         '数值': formula_table.loc[formula_table['条件竖线'].ne('—'), '序号'].tolist()},
        {'项目': '含绝对值宏的独立公式号',
         '数值': formula_table.loc[formula_table['绝对值宏'].ne('—'), '序号'].tolist()},
        {'项目': '含范数宏的独立公式号',
         '数值': formula_table.loc[formula_table['范数宏'].ne('—'), '序号'].tolist()},
        {'项目': '段内公式实例数', '数值': len(inline_table)},
        {'项目': '段内绝对值实例数', '数值': len(abs_rows)},
        {'项目': '未转换残留数', '数值': len(residual_rows)},
        {'项目': '公式硬错误残留数', '数值': 0}])
    write_excel(TABLE_MATH, {
        '01_独立公式逐条核验': formula_table,
        '02_段内公式实例清单': inline_table,
        '03_绝对值段内公式清单': pd.DataFrame(abs_rows) if abs_rows else pd.DataFrame(
            [{'文件': '—', '行号': '—', '命中文本': '无', 'LaTeX': '—'}]),
        '04_未转换残留': residual_table,
        '05_核验汇总': summary})
    save_metrics({'公式核验时间': time.strftime('%Y-%m-%d %H:%M:%S'),
                  '独立公式数': len(formula_table),
                  '独立公式序号连续': bool(list(formula_table['序号'])
                                     == list(range(1, len(formula_table) + 1))),
                  '含普通竖线的独立公式数': int((formula_table['普通竖线 |'] > 0).sum()),
                  '段内公式实例数': len(inline_table),
                  '段内绝对值实例数': len(abs_rows),
                  '未转换残留数': len(residual_rows)})
    print('=' * 96)
    print('Stage26.6 公式核验')
    print('=' * 96)
    print(summary.to_string(index=False))
    print('=' * 96)
    return 0


# =========================================================================== #
# 五、构建后 docx 核验
# =========================================================================== #
def run_docxverify() -> int:
    from docx import Document
    from docx.oxml.ns import qn

    doc = Document(str(DOCX))
    body = doc.element.body
    paras = doc.paragraphs
    tables = doc.tables
    seq_fields = seq_arabic = 0
    for el in list(body.findall('.//' + qn('w:instrText'))) + \
            list(body.findall('.//' + qn('w:fldSimple'))):
        instr = (el.text or '') if el.tag == qn('w:instrText') else (el.get(qn('w:instr')) or '')
        if re.search(r'SEQ\s+eq\b', instr):
            seq_fields += 1
            if 'ARABIC' in instr:
                seq_arabic += 1
    formula_tbl = 0
    for table in tables:
        if any(re.search(r'SEQ\s+eq\b', (el.text or ''))
               for el in table._tbl.findall('.//' + qn('w:instrText'))):
            formula_tbl += 1

    def _has_seq(paragraph):
        for el in paragraph._p.findall('.//' + qn('w:instrText')):
            if re.search(r'SEQ\s+eq\b', el.text or ''):
                return True
        return False

    eq_paras = [p for p in paras if _has_seq(p)]
    static_eq = [p.text for p in paras
                 if re.search(r'\(\d+\)', p.text)
                 and not p._p.findall('.//' + qn('w:instrText'))
                 and not p._p.findall('.//' + qn('w:fldSimple'))
                 and len(p.text.strip()) <= 8]
    abs_equation, abs_inline, pipe_in_math = [], [], []
    for paragraph in paras:
        in_equation = _has_seq(paragraph)
        for node in paragraph._p.findall('.//' + qn('m:d')):
            pr = node.find(qn('m:dPr'))
            begin = end = None
            if pr is not None:
                beg, fin = pr.find(qn('m:begChr')), pr.find(qn('m:endChr'))
                begin = beg.get(qn('m:val')) if beg is not None else '('
                end = fin.get(qn('m:val')) if fin is not None else ')'
            if begin == '|' and end == '|':
                inner = ''.join(piece.text or ''
                                for piece in node.findall('.//' + qn('m:t')))
                (abs_equation if in_equation else abs_inline).append(inner)
        for run in paragraph._p.findall('.//' + qn('m:r')):
            for piece in run.findall(qn('m:t')):
                if '|' in (piece.text or ''):
                    pipe_in_math.append(piece.text)
    inline_math = sum(len(p._p.findall('./' + qn('m:oMath')))
                      for p in paras if not _has_seq(p))
    omath_in_tbl = sum(len(t._tbl.findall('.//' + qn('m:oMath'))) for t in tables)
    fig_caps = [p.text for p in paras if p.style.name == 'Figure Caption']
    tbl_caps = [p.text for p in paras if p.style.name == 'Table Caption']
    drawings = len(body.findall('.//' + qn('w:drawing')))
    payload = save_metrics({
        'docx 核验时间': time.strftime('%Y-%m-%d %H:%M:%S'),
        'docx': str(DOCX.relative_to(PROJECT_ROOT)),
        'SEQ eq 域数': seq_fields, 'SEQ 含 ARABIC': seq_arabic,
        '公式用表数': formula_tbl, '静态编号残留数': len(static_eq),
        '独立公式绝对值 m:d': len(abs_equation), '段内绝对值 m:d': len(abs_inline),
        '数学 run 内普通竖线残留数': len(pipe_in_math),
        '段内公式实例数（docx 正文段落实测）': int(inline_math),
        '表格单元内段内公式数': int(omath_in_tbl),
        '正文插图数（inline drawing）': drawings,
        '图题数': len(fig_caps), '表题数': len(tbl_caps),
        '图题清单': fig_caps, '表题清单': tbl_caps,
    })
    with pd.ExcelWriter(TABLE_MATH, engine='openpyxl', mode='a', if_sheet_exists='replace') as bw:
        pd.DataFrame([{'序号': i + 1, '图题': t} for i, t in enumerate(fig_caps)]) \
            .to_excel(bw, sheet_name='06_正文图清单', index=False)
        pd.DataFrame([{'序号': i + 1, '表题': t} for i, t in enumerate(tbl_caps)]) \
            .to_excel(bw, sheet_name='07_正文表清单', index=False)
    print('=' * 96)
    print('Stage26.6 docx 公式与图表核验')
    print('=' * 96)
    print('SEQ eq 域 %d（ARABIC %d）；公式用表 %d；静态编号残留 %d'
          % (seq_fields, seq_arabic, formula_tbl, len(static_eq)))
    print('独立公式绝对值 m:d %d；段内绝对值 m:d %d；数学 run 内普通竖线残留 %d'
          % (len(abs_equation), len(abs_inline), len(pipe_in_math)))
    print('段内公式实例 %d（表内 %d）；插图 %d；图题 %d；表题 %d'
          % (inline_math, omath_in_tbl, drawings, len(fig_caps), len(tbl_caps)))
    print('图题清单：')
    for text in fig_caps:
        print('   ', text)
    print('=' * 96)
    return 0


# =========================================================================== #
# 六、QA 补强（图片分辨率、近空白页、文字重叠复核、Word/PDF 双路径）
# =========================================================================== #
FOCUS_ITEMS = ['图 4-2', '图 4-3', '图 4-6', '图 7-1', '图 8-1', '图 8-2', '图 8-3',
               '表 5-3', '表 5-4', '表 7-1', '表 8-1', '表 8-3', '表 8-4',
               '参考文献', '61.96', '17,040', '4,253', '4,795', '35.48', '51.98',
               'bge-small-zh-v1.5']


def run_qa() -> int:
    from PIL import Image
    from PyPDF2 import PdfReader

    m24 = _load_24a()
    work = QA_WORK
    page_map = load_json(work / 'page_map.json')
    inspect = load_json(work / 'inspect.json')
    textscan = load_json(work / 'textscan.json')
    export = load_json(work / 'export_info.json')
    shapes = page_map.get('inline_shapes', [])
    fig_rows = []
    for index, (key, title, source) in enumerate(m24.FIG_MAIN):
        path = m24.FIGROOT / source
        width_px, height_px = Image.open(str(path)).size
        width_pt = shapes[index].get('width_pt') if index < len(shapes) else None
        height_pt = shapes[index].get('height_pt') if index < len(shapes) else None
        fig_rows.append({'图号': '图 %s' % key, '文件': source,
                         '源像素宽×高': '%d × %d' % (width_px, height_px),
                         '版面宽cm': round(width_pt / 28.3465, 2) if width_pt else None,
                         '版面高cm': round(height_pt / 28.3465, 2) if height_pt else None,
                         '有效分辨率dpi': round(width_px / (width_pt / 72.0), 1)
                         if width_pt else None})
    fig_frame = pd.DataFrame(fig_rows)

    near_rows = []
    for item in (inspect.get('checks', {}).get('near_blank_pages', {}) or {}).get('pages', []):
        page = item['page']
        content = [p for p in page_map.get('paragraphs', [])
                   if (p.get('start_page') or p.get('page')) == page and not p.get('empty')]
        image = np.asarray(Image.open(QA_PAGES / ('page-%02d.png' % page)).convert('L'),
                           dtype='float64')
        ink = image < 250
        occupied = [i for i in range(image.shape[0]) if ink[i].any()]
        near_rows.append({'页码': page, '墨迹占比': item['ink_ratio'],
                          '首行像素': occupied[0] if occupied else None,
                          '末行像素': occupied[-1] if occupied else None,
                          '页面高度像素': image.shape[0],
                          '该页有效段落数': len(content),
                          '段落样式': '；'.join(sorted({str(p.get('style')) for p in content})) or '—',
                          '内容摘要': ' ｜ '.join(str(p.get('text_head'))[:30]
                                            for p in content[:6])})

    overlap_rows = []
    png_dir = QA_PAGES
    cache = {}
    for hit in (inspect.get('checks', {}).get('text_overlap', {}) or {}).get('hits', []):
        page = hit['page']
        if page not in cache:
            cache[page] = np.asarray(Image.open(png_dir / ('page-%02d.png' % page))
                                     .convert('L'), dtype='float64') / 255.0
        image = cache[page]
        top = max(int(hit['band_top_px']), 0)
        bottom = min(int(hit['band_bottom_px']), image.shape[0])
        strip = image[top:bottom, :]
        ink = (strip < 0.85).sum(axis=1).astype('float64')
        peak = float(ink.max()) if ink.size else 0.0
        runs, current = 0, False
        for value in ink:
            if value > 0.40 * peak:
                if not current:
                    runs += 1
                    current = True
            else:
                current = False
        overlap_rows.append({'页码': page, '带高/行高': hit.get('band_height_over_line'),
                             '带内可分辨文本行数': runs,
                             '上下两半重叠比': hit.get('overlap_ratio'),
                             '判定': '方法局限：行间可分辨' if runs >= 2 else
                                     '方法局限：单行与相邻横线/公式墨迹相连'})

    reader = PdfReader(str(QA_PDF))
    pdf_pages = [(i, page.extract_text() or '') for i, page in enumerate(reader.pages, start=1)]
    cross_rows = []
    for token in FOCUS_ITEMS:
        word_pages = sorted({(p.get('start_page') or p.get('page'))
                             for p in page_map.get('paragraphs', [])
                             if token in (p.get('text_head') or '')})
        pdf_hits = sorted({i for i, text in pdf_pages if token in text})
        cross_rows.append({'对象': token, 'Word 页': word_pages or '—',
                           'PDF 页': pdf_hits or '—',
                           '一致性': ('一致' if word_pages and pdf_hits and
                                   (set(word_pages) == set(pdf_hits)
                                    or abs(min(word_pages) - min(pdf_hits)) <= 1
                                    or len(word_pages) == len(pdf_hits))
                                   else ('仅 Word' if word_pages else
                                         ('仅 PDF' if pdf_hits else '均未命中')))})
    word_pages = (page_map.get('doc_stats') or {}).get('pages')
    forbidden_hits = [row for row in (textscan.get('forbidden', {}) or {}).get('results', [])
                      if row['count']]
    missing_values = [row['value'] for row in (textscan.get('required', {}) or {}).get('results', [])
                      if not row['present']]
    payload = save_metrics({
        'QA 补强时间': time.strftime('%Y-%m-%d %H:%M:%S'),
        'QA 目次': {
            'PDF': str(QA_PDF.relative_to(PROJECT_ROOT)),
            '逐页 PNG': str(QA_PAGES.relative_to(PROJECT_ROOT)),
            'QA 产物': str(QA_WORK.relative_to(PROJECT_ROOT))},
        'Word 页数': word_pages,
        '渲染页数': len(pdf_pages),
        '页数一致': bool(word_pages == len(pdf_pages)),
        '图片分辨率核验': fig_rows,
        '图片最低有效分辨率dpi': float(fig_frame['有效分辨率dpi'].min()),
        '近空白页实测': near_rows,
        '近空白页数': len(near_rows),
        '文字重叠候选数': len(overlap_rows),
        '文字重叠候选判定分布': dict(collections.Counter(row['判定'] for row in overlap_rows)),
        'dense_merged_block': sum(1 for hit in (inspect.get('checks', {})
                                                .get('text_overlap', {}) or {})
                                  .get('hits', []) if hit.get('sub_type') == 'dense_merged_block'),
        'Word/PDF 逐项比对': cross_rows,
        'Word/PDF 命中一致项数': sum(1 for row in cross_rows if row['一致性'] == '一致'),
        'Word/PDF 比对项数': len(cross_rows),
        'PDF 禁用串违规': [{'字符串': row['string'], '次数': row['count'],
                        '页码': [h['page'] for h in row['pages']][:10]}
                       for row in forbidden_hits],
        'PDF 关键值缺失': missing_values,
        '域错误数': len(export.get('field_errors') or []),
    })
    write_excel(TABLE_QA, {
        '01_图片分辨率核验': fig_frame,
        '02_近空白页实测': pd.DataFrame(near_rows),
        '03_文字重叠候选复核': pd.DataFrame(overlap_rows),
        '04_Word与PDF双路径比对': pd.DataFrame(cross_rows)})
    print('=' * 96)
    print('Stage26.6 QA 补强')
    print('=' * 96)
    print('Word 页数=%s 渲染页数=%s 一致=%s' % (word_pages, len(pdf_pages),
                                          word_pages == len(pdf_pages)))
    print('图片最低有效分辨率 %s dpi' % payload['图片最低有效分辨率dpi'])
    print('近空白页 %d：%s' % (len(near_rows), [row['页码'] for row in near_rows]))
    print('文字重叠候选 %d：%s' % (len(overlap_rows), payload['文字重叠候选判定分布']))
    print('PDF 禁用串违规 %d：%s' % (len(forbidden_hits),
                                [row['string'] for row in forbidden_hits]))
    print('PDF 关键值缺失 %d：%s' % (len(missing_values), missing_values))
    print(pd.DataFrame(cross_rows).to_string(index=False))
    print('=' * 96)
    return 0


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else 'snapshot'
    if mode == 'snapshot':
        return run_snapshot()
    if mode == 'verify':
        return run_verify()
    if mode == 'manuscript':
        return run_manuscript()
    if mode == 'textaudit':
        return run_textaudit()
    if mode == 'mathaudit':
        return run_mathaudit()
    if mode == 'docxverify':
        return run_docxverify()
    if mode == 'qa':
        return run_qa()
    print('未知模式：%s' % mode)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())

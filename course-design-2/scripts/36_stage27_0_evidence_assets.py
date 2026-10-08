# -*- coding: utf-8 -*-
"""Stage27.0 工程证据资产生成：核心代码片段图 + 冻结数据片段图。

设计约束（见 docs/prompts/32_Stage27.0...提示词）：
1. 代码片段逐字符取自当前 `src/`、`scripts/` 真实文件，只做整行截取；
2. 数据片段只读取自冻结产物，不做任何人工编造；
3. 匿名化只作用于展示副本，不修改任何冻结数据；
4. 代码图不出现 IDE 侧边栏、文件树、绝对路径、账号 / 密码 / token / cookie。

用法：
    python scripts\\36_stage27_0_evidence_assets.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / 'outputs' / 'figures' / 'evidence'
RAW_SRC = EV / '01_source' / '_raw'

# --------------------------------------------------------------------------- #
# 字体
# --------------------------------------------------------------------------- #
FONT_MONO = r'C:\Windows\Fonts\simsun.ttc'      # index 1 = NSimSun（等宽，含 CJK）
FONT_SANS = r'C:\Windows\Fonts\msyh.ttc'        # Microsoft YaHei
MONO_INDEX = 1
CODE_SIZE = 26
CODE_LEAD = 40
UI_TEXT = (0x1F, 0x1F, 0x1F)
UI_MUTED = (0x66, 0x66, 0x66)
UI_BORDER = (0xC8, 0xCD, 0xD4)
UI_PANEL_BG = (0xFB, 0xFC, 0xFD)
UI_HEAD_BG = (0xEE, 0xF2, 0xF7)
UI_LINE = (0xD8, 0xDE, 0xE6)
UI_LABEL = (0x1A, 0x3D, 0x6B)
UI_NOTE = (0x8A, 0x4B, 0x00)

TOKEN_COLORS = {
    'comment': (0x1F, 0x7A, 0x33),
    'string': (0xA3, 0x15, 0x15),
    'keyword': (0x00, 0x00, 0xC0),
    'builtin': (0x26, 0x7F, 0x99),
    'number': (0x09, 0x86, 0x58),
    'name': UI_TEXT,
    'op': (0x40, 0x40, 0x40),
}

KEYWORDS = {
    'def', 'class', 'return', 'if', 'elif', 'else', 'for', 'while', 'in', 'not', 'and',
    'or', 'import', 'from', 'as', 'with', 'try', 'except', 'finally', 'raise', 'lambda',
    'yield', 'pass', 'break', 'continue', 'None', 'True', 'False', 'is', 'global', 'assert',
    'del', 'async', 'await',
}
BUILTINS = {
    'len', 'range', 'int', 'float', 'str', 'bool', 'list', 'dict', 'tuple', 'set', 'sum',
    'min', 'max', 'abs', 'round', 'sorted', 'enumerate', 'zip', 'print', 'map', 'any', 'all',
    'isinstance', 'np', 'pd', 're', 'self',
}


def mono(size: int = CODE_SIZE) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_MONO, size, index=MONO_INDEX)


def sans(size: int = 24, bold: bool = False) -> ImageFont.FreeTypeFont:
    # msyh.ttc 0=Light 1=Regular 2=Bold 取决于版本，这里用 0 与 2 兜底
    try:
        return ImageFont.truetype(FONT_SANS, size, index=2 if bold else 0)
    except Exception:                                        # pragma: no cover
        return ImageFont.truetype(FONT_SANS, size, index=0)


def adv(font: ImageFont.FreeTypeFont, text: str) -> float:
    return float(font.getlength(text))


# --------------------------------------------------------------------------- #
# 源码读取与校验
# --------------------------------------------------------------------------- #
def read_source(rel_path: str) -> list:
    path = ROOT / rel_path
    if not path.exists():
        raise FileNotFoundError(f'缺少源码文件：{rel_path}')
    return path.read_text(encoding='utf-8').splitlines()


def excerpt(rel_path: str, start: int, end: int) -> dict:
    """取 [start, end] 闭区间行（1-based），返回代码文本与来源定位。"""
    lines = read_source(rel_path)
    if start < 1 or end > len(lines) or end < start:
        raise ValueError(f'行号越界：{rel_path} {start}-{end}（文件共 {len(lines)} 行）')
    body = lines[start - 1:end]
    payload = '\n'.join(body)
    return {
        'file': rel_path,
        'line_start': start,
        'line_end': end,
        'lines': body,
        'sha256': hashlib.sha256(payload.encode('utf-8')).hexdigest(),
        'n_lines': len(body),
    }


def ranges_excerpt(rel_path: str, ranges) -> dict:
    """多个连续行区间拼成一个面板（仍逐字符来自源文件）。"""
    lines = read_source(rel_path)
    body = []
    for start, end in ranges:
        body.extend(lines[start - 1:end])
    payload = '\n'.join(body)
    return {
        'file': rel_path,
        'line_start': ranges[0][0],
        'line_end': ranges[-1][1],
        'lines': body,
        'sha256': hashlib.sha256(payload.encode('utf-8')).hexdigest(),
        'n_lines': len(body),
    }


# --------------------------------------------------------------------------- #
# 语法高亮分词
# --------------------------------------------------------------------------- #
IDENT = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
NUMBER = re.compile(r'\d+(?:\.\d+)?')


def tokenize(line: str):
    out, i, n = [], 0, len(line)
    while i < n:
        ch = line[i]
        if ch == '#':
            out.append(('comment', line[i:]))
            break
        if ch in '"\'':
            j = i + 1
            while j < n:
                if line[j] == '\\':
                    j += 2
                    continue
                if line[j] == ch:
                    j += 1
                    break
                j += 1
            out.append(('string', line[i:j]))
            i = j
            continue
        m = IDENT.match(line, i)
        if m:
            word = m.group(0)
            kind = 'keyword' if word in KEYWORDS else ('builtin' if word in BUILTINS else 'name')
            out.append((kind, word))
            i = m.end()
            continue
        m = NUMBER.match(line, i)
        if m:
            out.append(('number', m.group(0)))
            i = m.end()
            continue
        out.append(('op', ch))
        i += 1
    return out


# --------------------------------------------------------------------------- #
# 面板渲染
# --------------------------------------------------------------------------- #
def panel_header(draw, width, label, sub=None, height=52):
    draw.rectangle([0, 0, width, height], fill=UI_HEAD_BG, outline=UI_BORDER)
    x = 18
    f_bold = sans(25, bold=True)
    draw.text((x, 12), label, font=f_bold, fill=UI_LABEL)
    x += adv(f_bold, label) + 22
    if sub:
        f_sub = sans(22)
        draw.text((x, 15), sub, font=f_sub, fill=UI_MUTED)
    return height


def code_panel(spec: dict, label: str, sub: str = None, show_line_no: bool = True,
               highlights=()):
    """把一个源码区间渲染为干净代码片段图（等宽、适度高亮、可选行号）。"""
    lines = spec['lines']
    RENDERED_TEXT.extend(lines)
    RENDERED_TEXT.append(label)
    font = mono(CODE_SIZE)
    unit = adv(font, 'M')
    line_no_w = 0
    if show_line_no:
        line_no_w = int(adv(font, str(spec['line_end'])) + 26)

    text_w = 0
    for line in lines:
        text_w = max(text_w, adv(font, line))
    width = int(66 + line_no_w + text_w)
    head_h = 52
    height = int(head_h + 16 + len(lines) * CODE_LEAD + 16)

    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width - 1, height - 1], outline=UI_BORDER)
    panel_header(draw, width - 1, label, sub, height=head_h)
    draw.rectangle([0, head_h, width - 1, head_h + 14], fill=UI_PANEL_BG)

    hl = set(highlights)
    y = head_h + 10
    for offset, line in enumerate(lines):
        baseline = y + 4
        if show_line_no:
            no_text = str(spec['line_start'] + offset)
            f_no = mono(CODE_SIZE - 4)
            draw.text((20, baseline + 3), no_text, font=f_no, fill=(0x99, 0xA3, 0xAE))
        x = 52 + line_no_w
        for kind, piece in tokenize(line):
            color = TOKEN_COLORS[kind]
            if (spec['line_start'] + offset) in hl:
                color = TOKEN_COLORS[kind]
            draw.text((x, baseline), piece, font=font, fill=color)
            x += adv(font, piece)
        y += CODE_LEAD
    return img


def _fit_text(text, font, max_w):
    """按宽度截断（超出加省略号），用于窄列。"""
    if adv(font, text) <= max_w:
        return text
    out = ''
    for ch in text:
        if adv(font, out + ch + '…') > max_w:
            break
        out += ch
    return out + '…'


def table_panel(title, columns, rows, col_max_px=None, font_size=23, align=None,
                note=None, highlight_rows=(), row_heights=None, header_sub=None):
    """把二维数据渲染为干净的三线表图。"""
    RENDERED_TEXT.append(title)
    RENDERED_TEXT.extend(str(c) for c in columns)
    for row in rows:
        RENDERED_TEXT.extend(str(v) for v in row)
    if note:
        RENDERED_TEXT.extend(str(n) for n in note)
    f_head = sans(font_size, bold=True)
    f_body = sans(font_size)
    pad = 16
    gap = 26
    if col_max_px is None:
        col_max_px = [None] * len(columns)

    widths = []
    for j, col in enumerate(columns):
        w = adv(f_head, str(col))
        for row in rows:
            w = max(w, adv(f_body, str(row[j]) if j < len(row) else ''))
        w += pad * 2
        if col_max_px[j]:
            w = min(w, col_max_px[j])
        widths.append(w)

    head_h = 52
    row_lead = int(font_size * 1.75)
    if row_heights is None:
        row_heights = [row_lead] * len(rows)
    body_h = sum(row_heights)
    note_h = 0 if not note else int(font_size * 1.9) * len(note)
    width = int(sum(widths) + gap * 2 + 16)
    height = int(head_h + 18 + row_lead + 12 + body_h + note_h + 18)

    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width - 1, height - 1], outline=UI_BORDER)
    panel_header(draw, width - 1, title, header_sub, height=head_h)

    x0 = gap + 8
    y = head_h + 26
    # 表头
    x = x0
    for j, col in enumerate(columns):
        draw.text((x + pad, y), _fit_text(str(col), f_head, widths[j] - 2 * pad),
                  font=f_head, fill=UI_LABEL)
        x += widths[j]
    draw.line([x0, y + row_lead - 6, x0 + sum(widths), y + row_lead - 6], fill=(0, 0, 0), width=2)

    y += row_lead
    for i, row in enumerate(rows):
        h = row_heights[i]
        if i in highlight_rows:
            draw.rectangle([x0 - 6, y - 4, x0 + sum(widths) + 6, y + h - 4], fill=UI_HEAD_BG)
        x = x0
        for j, col in enumerate(columns):
            value = str(row[j]) if j < len(row) else ''
            value = _fit_text(value, f_body, widths[j] - 2 * pad)
            if align and align[j] == 'c':
                tx = x + (widths[j] - adv(f_body, value)) / 2
            elif align and align[j] == 'r':
                tx = x + widths[j] - pad - adv(f_body, value)
            else:
                tx = x + pad
            draw.text((tx, y), value, font=f_body, fill=UI_TEXT)
            x += widths[j]
        y += h
        draw.line([x0, y - 6, x0 + sum(widths), y - 6], fill=UI_LINE, width=1)
    draw.line([x0, y - 6, x0 + sum(widths), y - 6], fill=(0, 0, 0), width=2)

    if note:
        yy = y + 4
        f_note = sans(max(font_size - 3, 16))
        for line in note:
            draw.text((x0, yy), line, font=f_note, fill=UI_NOTE)
            yy += int(font_size * 1.9)
    return img


def stack(panels, out_path: Path, gap=18, margin=16, bg=(255, 255, 255)):
    """纵向拼接若干面板为一张证据图。"""
    panels = [p for p in panels if p is not None]
    width = max(p.width for p in panels) + margin * 2
    height = sum(p.height for p in panels) + gap * (len(panels) - 1) + margin * 2
    canvas = Image.new('RGB', (width, height), bg)
    y = margin
    for p in panels:
        canvas.paste(p, ((width - p.width) // 2, y))
        y += p.height + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, dpi=(300, 300))
    return canvas, out_path


def side_by_side(left: Image.Image, right: Image.Image, out_path: Path,
                 gap=16, margin=16, target_h=None):
    """横向拼接两张等比例缩放的截图。"""
    if target_h:
        left = left.resize((int(left.width * target_h / left.height), target_h), Image.LANCZOS)
        right = right.resize((int(right.width * target_h / right.height), target_h), Image.LANCZOS)
    else:
        h = min(left.height, right.height)
        if left.height != h:
            left = left.resize((int(left.width * h / left.height), h), Image.LANCZOS)
        if right.height != h:
            right = right.resize((int(right.width * h / right.height), h), Image.LANCZOS)
    width = left.width + right.width + gap + margin * 2
    height = max(left.height, right.height) + margin * 2
    canvas = Image.new('RGB', (width, height), (255, 255, 255))
    canvas.paste(left, (margin, margin))
    canvas.paste(right, (margin + left.width + gap, margin))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, dpi=(300, 300))
    return canvas, out_path


def labelled(panel: Image.Image, tag: str):
    """在面板左上角外侧加 (a)/(b) 标记条。"""
    bar = 44
    img = Image.new('RGB', (panel.width, panel.height + bar), (255, 255, 255))
    img.paste(panel, (0, bar))
    draw = ImageDraw.Draw(img)
    draw.text((8, 6), tag, font=sans(26, bold=True), fill=UI_LABEL)
    return img


# --------------------------------------------------------------------------- #
# 匿名化（只作用于展示副本）
# --------------------------------------------------------------------------- #
class Anonymizer:
    def __init__(self):
        self.company = {}
        self.job = {}

    def comp(self, name: str) -> str:
        name = str(name)
        if name not in self.company:
            self.company[name] = '公司%s' % chr(ord('A') + len(self.company))
        return self.company[name]

    def ident(self, intern_id: str) -> str:
        text = str(intern_id)
        return text if len(text) <= 8 else '%s…%s' % (text[:4], text[-4:])


def wrap_text(text, font, max_w):
    """按像素宽度折行（中文逐字，西文按词优先）。"""
    lines, current = [], ''
    for token in re.findall(r'[A-Za-z0-9_\.\-/]+|\s+|[^A-Za-z0-9_\s]', str(text)):
        if token.isspace():
            current += ' '
            continue
        if adv(font, current + token) > max_w and current.strip():
            lines.append(current.rstrip())
            current = token
        else:
            current += token
    if current.strip():
        lines.append(current.rstrip())
    return lines


def text_panel(title, blocks, width_px=1500, font_size=23, note=None, header_sub=None):
    """渲染纯文本对照面板（例：原始岗位描述 → 模型安全版）。"""
    RENDERED_TEXT.append(title)
    for label, text, _ in blocks:
        RENDERED_TEXT.append(str(label))
        RENDERED_TEXT.append(str(text))
    if note:
        RENDERED_TEXT.extend(str(n) for n in note)
    f_body = sans(font_size)
    f_label = sans(font_size, bold=True)
    pad = 18
    head_h = 52
    col_w = width_px - pad * 2
    prepared = []
    for label, text, color in blocks:
        lines = wrap_text(text, f_body, col_w - 14)
        prepared.append((label, lines, color))
    body_h = sum(int(font_size * 1.62) * (len(lines) + 1) + 16 for _, lines, _ in prepared)
    note_h = 0 if not note else int(font_size * 1.85) * len(note) + 8
    height = int(head_h + 18 + body_h + note_h + 12)
    img = Image.new('RGB', (width_px, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width_px - 1, height - 1], outline=UI_BORDER)
    panel_header(draw, width_px - 1, title, header_sub, height=head_h)
    y = head_h + 14
    for label, lines, color in prepared:
        draw.text((pad, y), label, font=f_label, fill=color)
        y += int(font_size * 1.62)
        for line in lines:
            draw.text((pad + 14, y), line, font=f_body, fill=UI_TEXT)
            y += int(font_size * 1.62)
        y += 8
        draw.line([pad, y - 4, width_px - pad, y - 4], fill=UI_LINE, width=1)
        y += 8
    if note:
        f_note = sans(max(font_size - 3, 16))
        for line in note:
            draw.text((pad, y), line, font=f_note, fill=UI_NOTE)
            y += int(font_size * 1.85)
    return img


def crop_image(path: Path, top: int = 0, bottom=None, left: int = 0, right=None):
    with Image.open(path) as im:
        img = im.convert('RGB')
    w, h = img.size
    return img.crop((left, top, right or w, bottom or h))


def resize_w(img: Image.Image, width: int):
    return img.resize((width, int(img.height * width / img.width)), Image.LANCZOS)


# --------------------------------------------------------------------------- #
# 冻结数据读取（只读）
# --------------------------------------------------------------------------- #
def load_frozen():
    return {
        'raw': pd.read_parquet(ROOT / 'data/raw/shixiseng_job_details.parquet'),
        'version': pd.read_parquet(ROOT / 'data/processed/job_version_history.parquet'),
        'salary': pd.read_parquet(ROOT / 'data/processed/job_salary_targets.parquet'),
        'model': pd.read_parquet(ROOT / 'data/processed/job_salary_model_dataset.parquet'),
        'unique': pd.read_parquet(ROOT / 'data/processed/job_details_unique.parquet'),
        'corpus': pd.read_parquet(ROOT / 'data/interim/job_text_version_corpus.parquet'),
        'membership': pd.read_parquet(ROOT / 'data/features/job_skill_membership.parquet'),
        'anomaly': pd.ExcelFile(ROOT / 'outputs/tables/'
                                '22_company_attribute_semantic_anomaly_audit.xlsx'),
    }


def hhmm(value: str) -> str:
    text = str(value)
    return text[:16] if len(text) >= 16 else text


# =========================================================================== #
# 证据资产生成
# =========================================================================== #
REGISTRY = []
RENDERED_TEXT = []
# 一档：真正的凭证形态（键值对或长随机串），出现即判不通过
CREDENTIAL_PATTERNS = [
    r'password\s*[:=]\s*\S', r'passwd\s*[:=]\s*\S', r'pwd\s*[:=]\s*\S',
    r'token\s*[:=]\s*[A-Za-z0-9_\-\.]{8,}', r'api[_-]?key\s*[:=]\s*\S',
    r'secret\s*[:=]\s*\S', r'authorization\s*[:=]\s*\S',
    r'cookie\s*[:=]\s*\S', r'root@', r'127\.0\.0\.1', r'mysql\+pymysql',
    r'[A-Za-z]:\\Users\\', r'\bD:\\', r'\bE:\\', r'\bC:\\',
]
# 二档：敏感词根（变量名 / 跟踪参数名等同形词，人工复核）
KEYWORD_PATTERNS = [
    'password', 'passwd', 'token', 'cookie', 'authorization', 'api_key', 'secret',
    'proxy', '代理',
]
PRIVACY_PATTERNS = CREDENTIAL_PATTERNS + KEYWORD_PATTERNS


def record(evidence_id, chapter, section, evidence_type, source_type, source_file,
           source_function, source_line_range, data_columns, formal_title, caption_note,
           reuse, privacy, consistent, insert, priority, status, out_file, panel_source=None):
    REGISTRY.append({
        'evidence_id': evidence_id,
        'chapter': chapter,
        'section': section,
        'evidence_type': evidence_type,
        'source_type': source_type,
        'source_file': source_file,
        'source_function_or_table': source_function,
        'source_line_range': source_line_range,
        'data_columns': data_columns,
        'formal_title': formal_title,
        'caption_note': caption_note,
        'reuse_or_regenerate': reuse,
        'privacy_checked': privacy,
        'current_pipeline_consistent': consistent,
        'insert_to_word': insert,
        'priority': priority,
        'status': status,
        'figure_file': out_file,
        'panel_source': panel_source or '',
    })


def privacy_scan(text: str):
    """返回（凭证级命中, 同形词命中）。凭证级命中必须为 0。"""
    credentials, keywords = [], []
    for pattern in CREDENTIAL_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            credentials.append(pattern)
    lowered = text.lower()
    for word in KEYWORD_PATTERNS:
        if word.lower() in lowered:
            keywords.append(word)
    return credentials, keywords


def homograph_context(text: str, word: str, window: int = 46) -> list:
    """列出同形词所在上下文，供人工复核。"""
    found, lowered = [], text.lower()
    start = lowered.find(word.lower())
    while start >= 0 and len(found) < 6:
        found.append(text[max(0, start - window):start + len(word) + window].replace('\n', ' ⏎ '))
        start = lowered.find(word.lower(), start + 1)
    return found


# --------------------------------------------------------------------------- #
# E01 数据来源与岗位详情页（附件2 页面截图）
# --------------------------------------------------------------------------- #
def build_e01():
    menu = crop_image(RAW_SRC / 'A2_orig2_13_menu_expanded.png')
    detail = crop_image(RAW_SRC / 'A2_orig2_34_job_detail_page.png', top=150, bottom=1200)
    menu = labelled(resize_w(menu, 1800), '(a) “互联网 IT”菜单展开页面')
    detail = labelled(resize_w(detail, 1800), '(b) 岗位详情页面')
    out = EV / '01_source' / 'E01_source_pages.png'
    stack([menu, detail], out)
    record('E01', '3', '3.1', '平台页面截图', '附件2 网页截图（裁切去浏览器外框）',
           'docs/reference/基于实习僧平台的互联网 IT 类实习岗位的数据采集与分析 - 修订版.docx',
           '原图2.13 / 原图2.34',
           'docx media image13 / image34', '菜单展开页 + 岗位详情页（薪资、城市、学历、发布时间、职位描述）',
           '图 3-1 实习僧平台数据来源与岗位详情页面示例',
           '子图 (a) 为“互联网 IT”菜单展开页面，子图 (b) 为岗位详情页面；均来自真实平台页面截图，未做AI重绘',
           'REUSE_ATTACHMENT2', 'PASS', 'YES', 'YES', 'HIGH', 'GENERATED', str(out),
           '附件2 原图2.13（菜单展开）、原图2.34（职位详情页，裁切 top=150px 去浏览器标签页/地址栏/书签栏）')
    return out


# --------------------------------------------------------------------------- #
# E02 详情链接规范化与重复抓取控制
# --------------------------------------------------------------------------- #
def build_e02():
    a = excerpt('src/url_utils.py', 19, 41)
    b = excerpt('scripts/03_build_job_observations.py', 236, 244)
    p1 = code_panel(a, '(a) 岗位详情链接规范化：保留岗位身份 path，删除 fragment 与无业务含义的 query 参数')
    p2 = code_panel(b, '(b) 规范化链接与岗位标识的一一对应校验（去重与重复抓取控制的事实基础）')
    out = EV / '01_source' / 'E02_url_dedup_code.png'
    stack([p1, p2], out)
    record('E02', '3', '3.2', '核心实现代码', '当前正式代码',
           'src/url_utils.py', 'normalize_detail_url', '28-41',
           'TRACKING_PARAMS / DEFAULT_SCHEME（19-25）、url_utils.normalize_series 调用',
           '图 3-3 岗位详情链接规范化与重复抓取控制核心代码',
           '代码为当前正式实现片段，逐字符取自 src/url_utils.py 与 scripts/03_build_job_observations.py',
           'REGENERATE', 'PASS', 'YES', 'YES', 'HIGH', 'GENERATED', str(out),
           'src/url_utils.py 19-41；scripts/03_build_job_observations.py 236-244')
    return out


# --------------------------------------------------------------------------- #
# E03 采集运行界面（附件2 双标签页运行界面图）
# --------------------------------------------------------------------------- #
def build_e03():
    runtime = crop_image(RAW_SRC / 'A2_orig2_25_dual_tab_runtime.png')
    # 遮盖地址栏色带（保留标签页，遮蔽完整详情 URL）：
    # 词级 OCR 定位显示标签页文字位于 y≈10-35，地址栏文字位于 y≈70-89，
    # 页面内容自 y≈120 起，因此覆盖 y∈[55,104] 只抹去地址栏。
    draw = ImageDraw.Draw(runtime)
    draw.rectangle([0, 55, runtime.width, 104], fill=(0xEC, 0xEC, 0xEC))
    runtime = labelled(resize_w(runtime, 1700), '(a) 双标签页并行采集运行界面')
    out = EV / '01_source' / 'E03_crawl_runtime.png'
    stack([runtime], out)
    record('E03', '3', '3.2', '采集运行界面截图', '附件2 运行界面截图',
           'docs/reference/基于实习僧平台的互联网 IT 类实习岗位的数据采集与分析 - 修订版.docx',
           '原图2.25 双标签页运行界面图', 'docx media image25',
           '浏览器双标签页并行采集运行状态（标签页保留，地址栏色带已遮盖）',
           '图 3-4 岗位采集运行界面（双标签页协同）',
           '仅复用仍与当前采集口径一致的运行界面素材；未使用旧版processed_job_details截图；'
           '浏览器地址栏按“详情URL不完整展示”原则遮盖',
           'REUSE_ATTACHMENT2', 'PASS', 'YES', 'YES', 'MEDIUM', 'GENERATED', str(out),
           '附件2 原图2.25；覆盖区域 rows 34-52（地址栏色带）。'
           '图2.40 为桌面级多窗口叠加截图（含多窗口边框、完整地址与任务栏区域），'
           '按提示词 §12 判定不适合，改用图2.25')
    return out


# --------------------------------------------------------------------------- #
# E04 原始岗位观测数据片段
# --------------------------------------------------------------------------- #
E04_COLUMNS = ['intern_id', 'job_title_detail', 'item_text', 'city_detail', 'salary_detail',
               'degree_detail', 'intern_months_detail', 'publish_time_detail',
               'deadline_detail', 'company_name_detail']


def pick_e04_rows(raw: pd.DataFrame, n: int = 6):
    frame = raw[E04_COLUMNS].copy()
    frame = frame[frame['job_title_detail'].str.len() <= 16]
    frame = frame[frame['city_detail'].str.len() <= 4]
    frame = frame[frame['intern_months_detail'].str.len() <= 8]
    frame = frame.sort_values('intern_id', kind='stable').reset_index(drop=True)
    step = max(len(frame) // (n + 1), 1)
    return frame.iloc[[i * step for i in range(1, n + 1)]].reset_index(drop=True)


def build_e04(raw: pd.DataFrame):
    anon = Anonymizer()
    rows = pick_e04_rows(raw)
    base = [(anon.ident(r['intern_id']), r['job_title_detail'], r['item_text'],
             r['city_detail'], r['salary_detail']) for _, r in rows.iterrows()]
    tail = [(anon.ident(r['intern_id']), r['degree_detail'], r['intern_months_detail'],
             hhmm(r['publish_time_detail']), str(r['deadline_detail']),
             anon.comp(r['company_name_detail'])) for _, r in rows.iterrows()]

    t1 = table_panel('(a) 岗位身份、岗位名称、搜索分类、工作城市与薪资原文', 
                     ['岗位标识（掩码）', '岗位名称', '搜索分类', '工作城市', '薪资原文'],
                     base, align=['l', 'l', 'l', 'l', 'l'],
                     header_sub='来源：原始岗位观测表（172,063 条观测）')
    t2 = table_panel('(b) 学历要求、实习时长、发布时间、投递截止日期与公司名称',
                     ['岗位标识（掩码）', '学历要求', '实习时长', '发布时间', '投递截止日期',
                      '公司名称（匿名）'],
                     tail, align=['l', 'l', 'l', 'l', 'l', 'l'],
                     note=['注：公司名称与岗位标识已在展示副本中匿名化，原始冻结数据未做任何修改；'
                           '采集时间等数据治理字段不进入正文数据片段。'])
    out = EV / '02_preprocess' / 'E04_raw_observations.png'
    stack([t1, t2], out)
    record('E04', '3', '3.3', '真实数据片段',
           '冻结原始观测表（只读）', 'data/raw/shixiseng_job_details.parquet',
           '表 shixiseng_job_details（172,063 行 × 30 列）',
           '抽样行：按 intern_id 稳定排序后等距取 6 行',
           'intern_id / job_title_detail / item_text / city_detail / salary_detail / '
           'degree_detail / intern_months_detail / publish_time_detail / deadline_detail / '
           'company_name_detail',
           '图 3-5 原始岗位观测数据片段',
           '仅展示岗位身份、岗位属性与业务时间字段，不展示爬取时间、数据创建时间与数据更新时间',
           'REGENERATE', 'PASS', 'YES', 'YES', 'HIGH', 'GENERATED', str(out),
           'data/raw/shixiseng_job_details.parquet 等距抽样 6 行（展示级匿名化）')
    return out


# --------------------------------------------------------------------------- #
# E05 + E06 观测压缩与唯一实体（代码 + 真实案例）
# --------------------------------------------------------------------------- #
E06_JOB = 'inn_1bqlo3kqf64p'


def build_e05():
    a = excerpt('src/versioning.py', 241, 260)
    b = excerpt('src/dedup.py', 450, 467)
    p1 = code_panel(a, '(a) 核心业务签名 / 完整页面签名与连续版本压缩（A→B→A 保留三段）')
    p2 = code_panel(b, '(b) 唯一实体代表观测选择：版本优先、完整度其次')
    out = EV / '02_preprocess' / 'E05_entity_code.png'
    stack([p1, p2], out)
    record('E05', '3', '3.4', '核心实现代码', '当前正式代码',
           'src/versioning.py + src/dedup.py',
           'assign_snapshot_versions / select_representatives', '241-260；450-467',
           'CORE_SIGNATURE_FIELDS / FULL_SIGNATURE_FIELDS / 核心版本号 / 完整页面版本号 / '
           '代表记录选择依据',
           '图 3-7 岗位观测压缩与唯一实体生成核心代码',
           '代码为当前正式实现片段，逐字符取自 src/versioning.py 与 src/dedup.py；'
           '与 E06 示例互为“实现—实例”证据',
           'REGENERATE', 'PASS', 'YES', 'YES', 'HIGH', 'GENERATED', str(out),
           'src/versioning.py 241-260；src/dedup.py 450-467')
    return out


def build_e06(raw: pd.DataFrame):
    obs = raw[raw['intern_id'] == E06_JOB].copy().reset_index(drop=True)
    anon = Anonymizer()
    company = anon.comp(obs['company_name_detail'].iloc[0])
    ident = anon.ident(E06_JOB)
    rows = []
    for i, r in obs.iterrows():
        v = 1 if i < 6 else 2
        rows.append(['原始观测', str(i + 1), r['salary_detail'], r['item_text'],
                     r['city_detail'], hhmm(r['publish_time_detail']),
                     str(r['deadline_detail']),
                     '并入核心版本 %d' % v])
    for v, sub in obs.groupby(obs.index.map(lambda i: 1 if i < 6 else 2)):
        changed = '—' if v == 1 else '薪资信息'
        rows.append(['核心业务版本', str(v), sub['salary_detail'].iloc[0], '—',
                     sub['city_detail'].iloc[0], hhmm(sub['publish_time_detail'].iloc[0]),
                     str(sub['deadline_detail'].iloc[0]),
                     '%d 次观测压缩' % len(sub) if v == 1 else '相对上一版本变化：%s' % changed])
    final = obs.iloc[-1]
    rows.append(['唯一岗位实体', '最终', final['salary_detail'], '多分类合并',
                 final['city_detail'], hhmm(final['publish_time_detail']),
                 str(final['deadline_detail']), '取最终核心版本代表观测'])

    t = table_panel('(a) 真实岗位从多次观测到唯一实体的处理示例（岗位标识与公司名称已匿名化）',
                    ['层级', '编号', '薪资原文', '搜索分类', '工作城市', '发布时间',
                     '投递截止日期', '处理结果'],
                    rows, align=['l', 'c', 'l', 'l', 'l', 'l', 'l', 'l'],
                    note=['注：岗位标识 %s；公司名称 %s；观测序号按观测先后排列，'
                          '采集时间不作为正文分析维度。' % (ident, company),
                          '该岗位的城市与公司全程未变，薪资由 100-200 元/天变为 200-300 元/天，'
                          '版本层保留这一变化，实体层取最终核心版本作为代表状态。'])
    out = EV / '02_preprocess' / 'E06_entity_example.png'
    stack([t], out)
    record('E06', '3', '3.4.4', '真实数据片段', '冻结原始观测表（只读）',
           'data/raw/shixiseng_job_details.parquet + '
           'data/processed/job_version_history.parquet',
           '源实体 id = %s（真实存在，展示层匿名化）' % E06_JOB,
           '该 intern_id 的 7 条原始观测；核心版本号 1-2；是否最终核心版本',
           '薪资信息 / 工作城市 / 发布时间 / 投递截止日期 / 核心版本号 / 是否最终核心版本',
           '图 3-8 同一岗位从多次观测到唯一实体的处理示例',
           '示例为真实冻结记录，岗位标识与公司名称仅在展示副本中匿名化；'
           '不使用采集时间作为分析维度',
           'REGENERATE', 'PASS', 'YES', 'YES', 'HIGH', 'GENERATED', str(out),
           'source entity id = %s（7 条原始观测 → 2 个核心业务版本 → 1 个唯一实体）' % E06_JOB)
    return out


# --------------------------------------------------------------------------- #
# E07 公司属性语义槽位异常识别与修复
# --------------------------------------------------------------------------- #
def build_e07(xls: pd.ExcelFile):
    a = excerpt('src/field_repair.py', 126, 140)
    p1 = code_panel(a, '(a) 源记录公司属性语义槽位异常的三类判定规则（确定性，无估计或插补）')

    cand = xls.parse('03_全部候选行')
    stats = xls.parse('02_异常类型统计').set_index('指标')['数值']
    residual = xls.parse('08_残留异常').set_index('检查项')['结果']
    anon = Anonymizer()
    picks = []
    for kind in ['FULL_SHIFT', 'NATURE_ONLY', 'SIZE_ONLY']:
        row = cand[cand['异常类型'] == kind].iloc[0]
        picks.append([anon.comp(row['公司名称']), kind,
                      '空' if pd.isna(row['公司性质_原值']) else str(row['公司性质_原值']),
                      '空' if pd.isna(row['公司规模_原值']) else str(row['公司规模_原值']),
                      '空' if pd.isna(row['公司所在地_原值']) else str(row['公司所在地_原值']),
                      '缺失' if pd.isna(row['公司性质_修复值']) else str(row['公司性质_修复值']),
                      '缺失' if pd.isna(row['公司规模_修复值']) else str(row['公司规模_修复值']),
                      '缺失' if pd.isna(row['公司所在地_修复值']) else str(row['公司所在地_修复值'])])
    t1 = table_panel('(b) 真实记录的修复前后对照（公司名称已匿名化）',
                     ['公司（匿名）', '异常类型', '性质(前)', '规模(前)', '所在地(前)',
                      '性质(后)', '规模(后)', '所在地(后)'],
                     picks, align=['l', 'l', 'l', 'l', 'l', 'l', 'l', 'l'],
                     note=['注：三类异常的触发条件均要求公司所在地栏位为空；'
                           '公司性质一旦无法从本行还原即置为缺失，不做众数填充或按行业推断。'])
    t2 = table_panel('(c) 异常规模与确定性修复校验（真实统计结果）',
                     ['指标', '数值'],
                     [['公司性质栏位为人数区间', '%d 行' % int(stats['性质栏位为人数区间行数'])],
                      ['公司规模栏位为地点文本', '%d 行' % int(stats['规模栏位为地点文本行数'])],
                      ['两类症状同时出现', '%d 行' % int(stats['两类症状同时出现行数'])],
                      ['异常候选并集', '%d 行' % int(stats['异常候选并集行数'])],
                      ['FULL_SHIFT / NATURE_ONLY / SIZE_ONLY',
                       '%d / %d / %d 行' % (int(stats['FULL_SHIFT 行数']),
                                            int(stats['NATURE_ONLY 行数']),
                                            int(stats['SIZE_ONLY 行数']))],
                      ['修复后残留异常行数', '%d 行' % int(residual['修复后残留异常行数'])],
                      ['MySQL 与原始 Parquet 同 ID 30 列完全一致',
                       '%d / %d 行' % (int((xls.parse('04_MySQL_vs_Parquet证据')[
                           '30字段是否完全一致'] == '是').sum()), int(len(cand)))]],
                     align=['l', 'r'], col_max_px=[640, None])
    out = EV / '02_preprocess' / 'E07_company_slot_repair.png'
    stack([p1, t1, t2], out)
    record('E07', '3', '3.5', '核心实现代码 + 真实数据片段', '当前正式代码 + 冻结审计表（只读）',
           'src/field_repair.py + outputs/tables/22_company_attribute_semantic_anomaly_audit.xlsx',
           'classify_company_attribute_slot_anomaly', '126-140',
           '公司性质 / 公司规模 / 公司所在地 / 异常类型 / 检测证据',
           '图 3-9 公司属性语义槽位异常识别与修复示例',
           '统一使用正式术语“源记录公司属性语义槽位异常”，不写作“字段错位/槽位前移”，不推断丢失的公司性质',
           'REGENERATE', 'PASS', 'YES', 'YES', 'HIGH', 'GENERATED', str(out),
           'src/field_repair.py 126-140；22 号审计表 Sheet 02/03/04/08')
    return out


# --------------------------------------------------------------------------- #
# E08 薪资字段解析与正式目标构造
# --------------------------------------------------------------------------- #
def build_e08(salary: pd.DataFrame):
    a = excerpt('scripts/11_clean_structured_fields.py', 80, 107)
    p1 = code_panel(a, '(a) 区间 / 单值 / 面议判定、上下界逻辑检查与中点构造')

    def pick(cond, cols=None):
        return salary[cond].iloc[0]

    interval = pick(salary['薪资信息'].eq('120-150/天'))
    single = pick(salary['薪资信息'].eq('100/天'))
    negotiable = pick(salary['薪资信息'].eq('薪资面议'))
    anomaly = pick(salary['薪资异常标志'].fillna('').str.contains('上下限倒置'))

    def fmt(v):
        return '缺失' if pd.isna(v) else ('%g' % v)

    rows = [
        [str(interval['薪资信息']), fmt(interval['薪资下限']), fmt(interval['薪资上限']),
         fmt(interval['薪资中点']), '0', str(interval['薪资解析状态']),
         str(interval['薪资异常标志']) or '—'],
        [str(single['薪资信息']), fmt(single['薪资下限']), fmt(single['薪资上限']),
         fmt(single['薪资中点']), '0', str(single['薪资解析状态']),
         str(single['薪资异常标志']) or '—'],
        [str(negotiable['薪资信息']), fmt(negotiable['薪资下限']), fmt(negotiable['薪资上限']),
         fmt(negotiable['薪资中点']), '1', str(negotiable['薪资解析状态']),
         str(negotiable['薪资异常标志']) or '—'],
        [str(anomaly['薪资信息']), fmt(anomaly['薪资下限']), fmt(anomaly['薪资上限']),
         fmt(anomaly['薪资中点']), '0', str(anomaly['薪资解析状态']),
         str(anomaly['薪资异常标志'])],
    ]
    t1 = table_panel('(b) 真实薪资文本的解析前后对照（区间 / 单值 / 面议 / 异常区间）',
                     ['薪资原文', '薪资下限', '薪资上限', '薪资中点', '是否面议',
                      '解析状态', '异常标志'],
                     rows, align=['l', 'r', 'r', 'r', 'c', 'c', 'l'],
                     note=['注：面议岗位在薪资分析中保持缺失、不插补；逻辑异常记录直接剔除并留档。',
                           '150-200 元/天 → 下限 150、上限 200、中点 175；200 元/天 → 三列同为 200。'])
    t2 = table_panel('(c) 薪资目标构造的正式口径规模（真实结果）',
                     ['处理环节', '岗位数'],
                     [['唯一岗位实体', '17,144'], ['其中：薪资标注为面议', '2,245'],
                      ['明确薪资可解析', '14,899'], ['逻辑异常（区间颠倒、数值越界等）', '16'],
                      ['正式薪资样本', '14,883']], align=['l', 'r'])
    out = EV / '02_preprocess' / 'E08_salary_parse.png'
    stack([p1, t1, t2], out)
    record('E08', '3', '3.6', '核心实现代码 + 真实数据片段', '当前正式代码 + 冻结薪资目标表（只读）',
           'scripts/11_clean_structured_fields.py + data/processed/job_salary_targets.parquet',
           'parse_salary', '68-114（展示 80-107）',
           '薪资信息 / 薪资下限 / 薪资上限 / 薪资中点 / 是否面议 / 薪资解析状态 / 薪资异常标志',
           '图 3-10 薪资字段解析与正式目标构造示例',
           '正式口径规模保持 17,144 / 2,245 / 14,899 / 16 / 14,883 不变',
           'REGENERATE', 'PASS', 'YES', 'YES', 'HIGH', 'GENERATED', str(out),
           'scripts/11_clean_structured_fields.py 80-107；job_salary_targets.parquet 真实记录')
    return out


# --------------------------------------------------------------------------- #
# E09 岗位文本安全处理与技能抽取
# --------------------------------------------------------------------------- #
E09_JOB = 'inn_4fezolxa9tsh'


def build_e09(corpus: pd.DataFrame, membership: pd.DataFrame, anon: Anonymizer):
    a = excerpt('src/text_utils.py', 344, 360)
    p1 = code_panel(a, '(a) 模型安全版构造：多轮清除薪资表达，保留技术版本号 / 实习月数与到岗天数')

    row = corpus[(corpus['实习岗位ID'] == E09_JOB)
                 & (corpus['是否最终核心版本'] == 1)].iloc[0]
    raw_tail = str(row['岗位描述_原始'])[-190:]
    safe_tail = str(row['岗位描述_模型安全版'])[-190:]
    req = str(row['任职要求文本'])
    p2 = text_panel('(b) 真实岗位文本：原始岗位描述 → 模型安全版（局部）',
                    [('原始岗位描述（薪资表达保留）', raw_tail, (0x60, 0x60, 0x60)),
                     ('模型安全版（薪资表达已清除，其余业务数字保留）', safe_tail,
                      (0x1F, 0x7A, 0x33)),
                     (' REQUIREMENT_SECTION（要求段落，字段：任职要求文本）', req[:260],
                      (0x1A, 0x3D, 0x6B))],
                    width_px=1560,
                    note=['注：只删除薪资表达本身，不删除技术版本号、实习月数与每周到岗天数等业务数字；'
                          '公司名称已匿名化。'])

    sub = membership[(membership['intern_id'] == E09_JOB)]
    req_rows = sub[sub['match_scope'] == 'REQUIREMENT_SECTION'].sort_values(
        ['feature_family', 'canonical_skill'])
    p3 = table_panel('(c) 要求段落口径下抽取的规范技能（scope = REQUIREMENT_SECTION）',
                     ['规范技能', '技能族', '技能组', '匹配口径', '命中次数'],
                     [[r['canonical_skill'], r['feature_family'], r['group'],
                       r['match_scope'], str(int(r['hit_count']))]
                      for _, r in req_rows.iterrows()],
                     align=['l', 'l', 'l', 'l', 'r'],
                     note=['注：技能词典按技能族、技能组、标准技能名三级组织；'
                           '此处只展示该岗位命中的规范技能，不展示完整技能词典。'])
    out = EV / '02_preprocess' / 'E09_text_skill.png'
    stack([p1, p2, p3], out)
    record('E09', '3', '3.7', '核心实现代码 + 真实文本与技能片段',
           '当前正式代码 + 冻结语料/技能成员表（只读）',
           'src/text_utils.py + data/interim/job_text_version_corpus.parquet + '
           'data/features/job_skill_membership.parquet',
           'build_model_safe_version / extract_requirement_section', '344-360',
           '岗位描述_原始 / 岗位描述_模型安全版 / 任职要求文本 / canonical_skill / match_scope',
           '图 3-11 岗位文本安全处理与技能抽取示例',
           '体现原始岗位描述、REQUIREMENT_SECTION、规范技能与薪资泄漏清理四者关系',
           'REGENERATE', 'PASS', 'YES', 'YES', 'HIGH', 'GENERATED', str(out),
           'src/text_utils.py 344-360；源实体 id = %s（真实岗位，公司名称展示层匿名化）' % E09_JOB)
    return out


# --------------------------------------------------------------------------- #
# E10 最终建模数据集代表性字段片段
# --------------------------------------------------------------------------- #
E10_CALIBER = 'ALL_USABLE'


def flat_cell(value) -> str:
    """把 list / ndarray 单元格渲染为可读文本；空值返回 '—'。"""
    if value is None:
        return '—'
    if isinstance(value, (list, tuple, set)):
        items = [str(v) for v in value if str(v).strip()]
        return '、'.join(items) if items else '—'
    if isinstance(value, np.ndarray):
        items = [str(v) for v in value.tolist() if str(v).strip()]
        return '、'.join(items) if items else '—'
    text = str(value).strip()
    if text in ('', 'nan', 'None', '[]', "['']"):
        return '—'
    return text


def build_e10(model: pd.DataFrame, unique: pd.DataFrame, membership: pd.DataFrame,
              anon: Anonymizer):
    frame = model.copy()
    frame = frame.merge(unique[['实习岗位ID', '公司认证标签']], on='实习岗位ID', how='left')
    usable = membership[membership['match_scope'].isin(['REQUIREMENT_SECTION',
                                                       'FULL_TEXT_FALLBACK'])]
    skill_sets = usable.groupby('intern_id')['canonical_skill'].apply(set)

    picks = frame.iloc[[7, 1500, 3600, 6200, 9500, 13200]].reset_index(drop=True)
    rows1, rows2 = [], []
    for _, r in picks.iterrows():
        ident = anon.ident(r['实习岗位ID'])
        skills = skill_sets.get(r['实习岗位ID'], set())
        rows1.append([ident, '%g' % r['薪资中点'], flat_cell(r['岗位大类集合']),
                      str(r['工作城市_规范']), str(r['学历要求']),
                      str(r['实习时长要求']), str(r['每周到岗要求'])])
        rows2.append([ident, str(r['公司规模']), str(r['公司性质']), str(r['所属行业']),
                      flat_cell(r['公司认证标签']), '1' if 'Python' in skills else '0',
                      '1' if 'SQL' in skills else '0',
                      str(int(r['技能数量'])), str(int(r['岗位描述字符数']))])

    t1 = table_panel('(a) 目标变量与岗位/地域字段（字段级宽表，非编码后矩阵）',
                     ['岗位标识（掩码）', '薪资中点', '岗位大类', '工作城市', '学历要求',
                      '实习时长', '每周到岗'],
                     rows1, align=['l', 'r', 'l', 'l', 'l', 'l', 'l'],
                     header_sub='正式建模样本：14,883 个岗位')
    t2 = table_panel('(b) 公司侧字段、技能指示列与文本长度',
                     ['岗位标识（掩码）', '公司规模', '公司性质', '所属行业', '公司认证',
                      'skill_Python', 'skill_SQL', '技能计数', '描述字符数'],
                     rows2, align=['l', 'l', 'l', 'l', 'l', 'c', 'c', 'r', 'r'],
                     note=['注：skill_Python 与 skill_SQL 为技能扩展口径（REQUIREMENT_SECTION + '
                           'FULL_TEXT_FALLBACK）下的技能指示值，由冻结技能成员表按该口径派生。',
                           '图中仅展示代表性字段；经训练集内编码、技能列筛选与文本降维后形成 '
                           '288 维正式模型输入，编码矩阵不在图中展开。'])
    out = EV / '02_preprocess' / 'E10_model_dataset.png'
    stack([t1, t2], out)
    record('E10', '3', '3.8', '真实数据片段', '冻结建模数据集（只读）',
           'data/processed/job_salary_model_dataset.parquet', '表 job_salary_model_dataset',
           '抽样行：索引 7 / 1500 / 3600 / 6200 / 9500 / 13200',
           '薪资中点 / 岗位大类集合 / 工作城市_规范 / 学历要求 / 实习时长要求 / 每周到岗要求 / '
           '公司规模 / 公司性质 / 所属行业 / 公司认证标签 / skill_Python / skill_SQL / '
           '技能数量 / 岗位描述字符数',
           '图 3-12 最终建模数据集代表性字段片段',
           '图注说明：正式建模样本为 14,883 个岗位；图中仅展示部分代表性字段，'
           '经训练集内编码、技能筛选及文本降维后形成正式模型输入',
           'REGENERATE', 'PASS', 'YES', 'YES', 'HIGH', 'GENERATED', str(out),
           'job_salary_model_dataset.parquet + job_details_unique.parquet（公司认证标签）+ '
           'job_skill_membership.parquet（skill_Python / skill_SQL 指示值）')
    return out


# --------------------------------------------------------------------------- #
# E11 分组薪资差异检验与多重校正核心代码
# --------------------------------------------------------------------------- #
def build_e11():
    a = excerpt('src/eda_analysis.py', 149, 164)
    b = excerpt('src/eda_analysis.py', 183, 194)
    c = excerpt('src/eda_analysis.py', 197, 212)
    p1 = code_panel(a, '(a) 单值互斥因素：Kruskal–Wallis H 与 ε² 效应量')
    p2 = code_panel(b, '(b) 二元命中因素：Cliff\'s δ 与 Mann–Whitney U 检验')
    p3 = code_panel(c, '(c) 同一因素内的多重比较校正：Benjamini–Hochberg FDR')
    out = EV / '03_analysis' / 'E11_stats_test_code.png'
    stack([p1, p2, p3], out)
    record('E11', '5', '5.1', '核心实现代码', '当前正式代码', 'src/eda_analysis.py',
           'kruskal_wallis / two_group_stats / add_fdr', '149-164；183-194；197-212',
           '分组薪资中点 / H 统计量 / ε² / Cliff\'s δ / Mann–Whitney U / p_raw / p_adjusted_bh',
           '图 5-1 分组薪资差异检验与多重校正核心代码',
           '统一检验函数用于城市、学历、行业、认证、技能等全部因素，不为单一因素重复生成代码图',
           'REGENERATE', 'PASS', 'YES', 'YES', 'HIGH', 'GENERATED', str(out),
           'src/eda_analysis.py 149-164 / 183-194 / 197-212')
    return out


# --------------------------------------------------------------------------- #
# E12 训练验证测试划分与训练集拟合
# --------------------------------------------------------------------------- #
def build_e12():
    a = excerpt('src/model_training.py', 46, 66)
    b = excerpt('scripts/14_train_salary_model.py', 507, 514)
    p1 = code_panel(a, '(a) 70% / 15% / 15% 分层随机划分（按薪资十分位分层，random_state = 42）')
    p2 = code_panel(b, '(b) 预处理器只在训练集拟合，再分别变换训练集、验证集与测试集')
    out = EV / '04_model' / 'E12_split_code.png'
    stack([p1, p2], out)
    record('E12', '7', '7.2', '核心实现代码', '当前正式代码',
           'src/model_training.py + scripts/14_train_salary_model.py',
           'build_splits / assemble', '46-66；507-514',
           '薪资中点（分层变量）/ split / 训练集 10,418、验证集 2,232、测试集 2,233',
           '图 7-2 训练验证测试划分与训练集拟合核心代码',
           '体现测试集不参与预处理拟合；正式样本量保持 10,418 / 2,232 / 2,233',
           'REGENERATE', 'PASS', 'YES', 'YES', 'HIGH', 'GENERATED', str(out),
           'src/model_training.py 46-66；scripts/14_train_salary_model.py 507-514')
    return out


# --------------------------------------------------------------------------- #
# E13 A/B/C/D/E 异构特征构建与拼接
# --------------------------------------------------------------------------- #
def build_e13():
    a = excerpt('src/model_training.py', 178, 191)
    b = excerpt('src/model_training.py', 155, 176)
    p1 = code_panel(a, '(a) 五组主体特征的编码与拼接：数值 / 类别 / 多值 / 技能 / 文本语义')
    p2 = code_panel(b, '(b) 技能列按训练集频次筛选（≥100）与文本语义截断奇异值分解（16 维）')
    out = EV / '04_model' / 'E13_feature_pipeline_code.png'
    stack([p1, p2], out)
    record('E13', '7', '7.3', '核心实现代码', '当前正式代码', 'src/model_training.py',
           'SalaryFeatureAssembler.transform / .fit', '178-191；155-176',
           'A 岗位基础 / B 地域 / C 公司 / D 技能 / E 文本语义；A+B+C = 216、D = 56、'
           'E = 16、Full = 288、加发布时间位置扩展特征 = 290',
           '图 7-3 A/B/C/D/E 异构特征构建与拼接核心代码',
           '正式维度 216 / 56 / 16 / 288 / 290 不变；不展开 288 个字段明细',
           'REGENERATE', 'PASS', 'YES', 'YES', 'HIGH', 'GENERATED', str(out),
           'src/model_training.py 178-191 / 155-176')
    return out


# --------------------------------------------------------------------------- #
# E14 候选回归模型训练与验证集选择
# --------------------------------------------------------------------------- #
def build_e14():
    a = excerpt('src/model_training.py', 324, 348)
    b = excerpt('scripts/14_train_salary_model.py', 595, 602)
    c = excerpt('scripts/14_train_salary_model.py', 628, 632)
    p1 = code_panel(a, '(a) 候选模型：Dummy 基线、Ridge、RandomForest、CatBoost、LightGBM')
    p2 = code_panel(b, '(b) 小网格超参数按验证集平均绝对误差取最优')
    p3 = code_panel(c, '(c) 在非基线模型中按验证集平均绝对误差锁定主模型')
    out = EV / '04_model' / 'E14_model_selection_code.png'
    stack([p1, p2, p3], out)
    record('E14', '7', '7.4', '核心实现代码', '当前正式代码',
           'src/model_training.py + scripts/14_train_salary_model.py',
           'make_model / 模型网格循环 / 主模型锁定', '324-348；595-602；628-632',
           '候选模型 / 超参数 / validation_MAE / validation_RMSE / validation_R²',
           '图 7-4 候选回归模型训练与验证集选择核心代码',
           '模型选择依据为验证集平均绝对误差，测试集不参与模型选择；正式指标引用既有冻结结果',
           'REGENERATE', 'PASS', 'YES', 'YES', 'HIGH', 'GENERATED', str(out),
           'src/model_training.py 324-348；scripts/14_train_salary_model.py 595-602 / 628-632')
    return out


# --------------------------------------------------------------------------- #
# E15 特征组消融与替代泛化划分
# --------------------------------------------------------------------------- #
def build_e15():
    a = excerpt('src/ablation_shap.py', 32, 40)
    b = excerpt('src/model_training.py', 72, 87)
    c = excerpt('scripts/26b_stage26_1_temporal_tightening.py', 945, 960)
    p1 = code_panel(a, '(a) 正式消融配置：Base = A+B+C、Base+Skill、Full-Skill、Full')
    p2 = code_panel(b, '(b) 按公司实体分组划分：同一公司不跨训练 / 验证 / 测试子集')
    p3 = code_panel(c, '(c) 按业务发布时间排序的回顾性时间划分（同一天不拆分）')
    out = EV / '04_model' / 'E15_ablation_generalization_code.png'
    stack([p1, p2, p3], out)
    record('E15', '8', '8.1-8.2', '核心实现代码', '当前正式代码',
           'src/ablation_shap.py + src/model_training.py + '
           'scripts/26b_stage26_1_temporal_tightening.py',
           'ABLATION_CONFIGS / build_company_group_split / run_temporal_split',
           '32-40；72-87；945-960',
           '特征组 A/B/C/D/E / company_entity_id / company_group_split / 业务发布日期 / '
           'temporal_split',
           '图 8-2 特征组消融与替代泛化划分核心代码',
           '消融配置与当前正式组合一致；公司分组划分与发布时间排序划分口径不变',
           'REGENERATE', 'PASS', 'YES', 'YES', 'HIGH', 'GENERATED', str(out),
           'src/ablation_shap.py 32-40；src/model_training.py 72-87；'
           'scripts/26b_stage26_1_temporal_tightening.py 945-960')
    return out


# --------------------------------------------------------------------------- #
# E16 TreeSHAP 模型解释与技能方向统计
# --------------------------------------------------------------------------- #
def build_e16():
    a = excerpt('src/ablation_shap.py', 124, 136)
    b = excerpt('src/ablation_shap.py', 206, 231)
    p1 = code_panel(a, '(a) LightGBM 原生 TreeSHAP：pred_contrib 贡献矩阵（去掉基准值列）')
    p2 = code_panel(b, '(b) 技能存在时的平均贡献方向与平均绝对 SHAP 值统计')
    out = EV / '05_explain' / 'E16_shap_code.png'
    stack([p1, p2], out)
    record('E16', '8', '8.4', '核心实现代码', '当前正式代码', 'src/ablation_shap.py',
           'tree_shap_values / shap_skill_table', '124-136；206-231',
           '技能列 / present 与 absent 划分 / mean_abs_SHAP / mean_SHAP_present / '
           'presence_direction',
           '图 8-3 TreeSHAP 模型解释与技能方向统计核心代码',
           '只解释既有模型结果；不重新运行 SHAP，不覆盖任何冻结结果',
           'REGENERATE', 'PASS', 'YES', 'YES', 'HIGH', 'GENERATED', str(out),
            'src/ablation_shap.py 124-136 / 206-231')
    return out


# =========================================================================== #
# 主流程
# =========================================================================== #
REGISTRY_COLUMNS = [
    'evidence_id', 'chapter', 'section', 'evidence_type', 'source_type', 'source_file',
    'source_function_or_table', 'source_line_range', 'data_columns', 'formal_title',
    'caption_note', 'reuse_or_regenerate', 'privacy_checked', 'current_pipeline_consistent',
    'insert_to_word', 'priority', 'status', 'figure_file', 'panel_source',
]


def write_registry(extra_sheets: dict):
    path = ROOT / 'outputs' / 'tables' / '34_visual_evidence_registry.xlsx'
    frame = pd.DataFrame(REGISTRY)[REGISTRY_COLUMNS]
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        frame.to_excel(writer, sheet_name='01_visual_evidence_registry', index=False)
        for sheet, data in extra_sheets.items():
            data.to_excel(writer, sheet_name=sheet, index=False)
        sheet = writer.sheets['01_visual_evidence_registry']
        widths = {'evidence_id': 11, 'chapter': 8, 'section': 10, 'evidence_type': 24,
                  'source_type': 26, 'source_file': 46, 'source_function_or_table': 26,
                  'source_line_range': 20, 'data_columns': 52, 'formal_title': 40,
                  'caption_note': 46, 'reuse_or_regenerate': 20, 'privacy_checked': 15,
                  'current_pipeline_consistent': 26, 'insert_to_word': 15,
                  'priority': 10, 'status': 12, 'figure_file': 56, 'panel_source': 56}
        for idx, col in enumerate(REGISTRY_COLUMNS, start=1):
            sheet.column_dimensions[
                sheet.cell(row=1, column=idx).column_letter].width = widths.get(col, 18)
        sheet.freeze_panes = 'A2'
    return path


def main() -> int:
    frozen = load_frozen()
    anon = Anonymizer()
    for sub in ('01_source', '02_preprocess', '03_analysis', '04_model', '05_explain'):
        (EV / sub).mkdir(parents=True, exist_ok=True)

    outputs = []
    outputs.append(build_e01())
    outputs.append(build_e02())
    outputs.append(build_e03())
    outputs.append(build_e04(frozen['raw']))
    outputs.append(build_e05())
    outputs.append(build_e06(frozen['raw']))
    outputs.append(build_e07(frozen['anomaly']))
    outputs.append(build_e08(frozen['salary']))
    outputs.append(build_e09(frozen['corpus'], frozen['membership'], anon))
    outputs.append(build_e10(frozen['model'], frozen['unique'], frozen['membership'], anon))
    outputs.append(build_e11())
    outputs.append(build_e12())
    outputs.append(build_e13())
    outputs.append(build_e14())
    outputs.append(build_e15())
    outputs.append(build_e16())

    # 隐私扫描（对渲染进图的全部文本）
    joined = '\n'.join(RENDERED_TEXT)
    credentials, keywords = privacy_scan(joined)
    homographs = []
    for word in keywords:
        for context in homograph_context(joined, word):
            homographs.append({'同形词': word, '上下文': context})
    forbidden_engineering = [w for w in ('Stage', 'Gate', 'PASS', 'STRICT', 'RECORD')
                             if w in joined]
    image_rows = []
    for out in outputs:
        with Image.open(out) as im:
            image_rows.append({'figure_file': out.name, 'width_px': im.width,
                               'height_px': im.height,
                               'aspect_h_over_w': round(im.height / im.width, 3),
                               'size_kb': round(out.stat().st_size / 1024, 1)})

    sheet_privacy = pd.DataFrame([
        {'检查项': '凭证级命中（password / passwd / pwd 键值）', '命中数': sum(
            1 for p in credentials if 'pass' in p or 'pwd' in p)},
        {'检查项': '凭证级命中（token / api_key / secret / authorization / cookie 键值）',
         '命中数': sum(1 for p in credentials if any(
             k in p for k in ('token', 'api_key', 'secret', 'authorization', 'cookie')))},
        {'检查项': '数据库连接串 / 端口 / root@ / 127.0.0.1 命中', '命中数': sum(
            1 for p in credentials if any(k in p for k in (
                'pymysql', 'root@', '127', '3306')))},
        {'检查项': '本地绝对路径命中（C:\\ / D:\\ / E:\\ / C:\\Users\\）', '命中数': sum(
            1 for p in credentials if '\\\\' in p)},
        {'检查项': '敏感词根同形词（变量名 / 跟踪参数名，已逐条复核）', '命中数': len(homographs)},
        {'检查项': '渲染文本中的工程内部词命中（Stage/Gate/PASS）', '命中数':
            len(forbidden_engineering)},
        {'检查项': '渲染文本总条目数（代码行 + 表格单元 + 文本块）', '命中数': len(RENDERED_TEXT)},
    ])
    sheet_homograph = pd.DataFrame(homographs or [{'同形词': '无', '上下文': '无'}])
    sheet_images = pd.DataFrame(image_rows)
    sheet_code = pd.DataFrame([
        {'evidence_id': r['evidence_id'], 'source_file': r['source_file'],
         'source_function_or_table': r['source_function_or_table'],
         'source_line_range': r['source_line_range'], 'figure_file': Path(r['figure_file']).name}
        for r in REGISTRY if r['evidence_type'].startswith('核心实现代码')])

    registry = write_registry({
        '02_privacy_scan': sheet_privacy,
        '03_homograph_review': sheet_homograph,
        '04_image_geometry': sheet_images,
        '05_code_traceability': sheet_code,
    })

    print('=' * 78)
    for out in outputs:
        with Image.open(out) as im:
            print('  %-52s %5dx%-5d %8.1f KB' % (out.name, im.width, im.height,
                                                 out.stat().st_size / 1024))
    print('-' * 78)
    print('证据资产数:', len(outputs))
    print('登记条目数:', len(REGISTRY))
    print('凭证级隐私命中:', credentials or '无')
    print('同形词命中:', keywords or '无')
    for row in homographs:
        print('   ', row['同形词'], '->', row['上下文'][:110])
    print('渲染文本工程内部词命中:', forbidden_engineering or '无')
    print('Registry:', registry)
    print('=' * 78)
    return 0


if __name__ == '__main__':
    sys.exit(main())


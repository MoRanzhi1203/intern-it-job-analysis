# -*- coding: utf-8 -*-
"""Stage24 论文 QA 工具链：Word → PDF → 逐页渲染 → 程序化视觉检查。

本脚本**只做工具与检查**，绝不修改论文内容、正文、数据与图件：

- ``open-update-export``：Word COM 打开 docx → 更新全部域（Fields / TOC /
  StoryRanges / 页眉页脚）→ 保存回同一路径 → ``ExportAsFixedFormat`` 导出 PDF。
- ``page-map``：输出「逐段落 → 页码」映射（CSV + JSON），含段落样式 / 页码 /
  是否含图；单独输出 InlineShape、Table、节与文档级统计。
- ``render``：调用 ``pdftoppm``（默认 ``D:\\texlive\\2024\\bin\\windows\\pdftoppm.exe``）
  以 ``-r 110 -png`` 逐页渲染，文件名规范 ``page-01.png``、``page-02.png`` …。
- ``inspect``：对渲染 PNG + page-map 做程序化视觉检查（空白页、边缘越界、
  文字重叠、图片/表格越界、图题表题分离、孤立标题、题注序号连续性）。
- ``textscan``：PyPDF2 逐页抽取文本，扫描必须为 0 的旧串/工程串与必须存在的关键值。
- ``report``：汇总为 ``qa_report.json`` + ``qa_summary.md`` 并打印关键结论。
- ``all``：串联以上全部步骤（可 ``--copy-docx`` 先在 work-dir 内做副本，
  保证源 docx 不被修改）。

环境无法目视看图，因此视觉 QA 全部为程序化检查（页码取自 Word COM，
像素分析取自 PIL/NumPy），不依赖人工看图。

用法::

    # 在既有 Stage21 docx 的临时副本上验证整条工具链
    python scripts/24b_qa_render.py all \\
        --docx "outputs/paper/课程设计论文_终稿_Stage21.docx" \\
        --work-dir "outputs/paper/qa/stage21_verify" --copy-docx

    # Stage24 正式成品（无 --work-dir 时默认写到 outputs/paper/qa 下）
    python scripts/24b_qa_render.py all \\
        --docx "outputs/paper/课程设计论文_终稿_Stage24.docx" --copy-docx

    # 子命令单独使用
    python scripts/24b_qa_render.py open-update-export --docx X.docx --pdf X.pdf
    python scripts/24b_qa_render.py page-map --docx X.docx --out-json page_map.json
    python scripts/24b_qa_render.py render --pdf X.pdf --out-dir rendered --dpi 110
    python scripts/24b_qa_render.py inspect --page-map-json page_map.json --png-dir rendered
    python scripts/24b_qa_render.py textscan --pdf X.pdf --out-json textscan.json
    python scripts/24b_qa_render.py report --work-dir outputs/paper/qa

受保护路径（本脚本只读、绝不写入）：
``data/``、``outputs/tables/``、``outputs/models/``、``outputs/figures/**``、
``src/plot_style.py``、``docs/paper/stage23/**``、``docs/records/**``。
所有写盘路径只允许落在 ``outputs/paper/qa/`` 之下（UTF-8 编码）。
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QA_ROOT = PROJECT_ROOT / 'outputs' / 'paper' / 'qa'
DEFAULT_DOCX_STAGE21 = PROJECT_ROOT / 'outputs' / 'paper' / '课程设计论文_终稿_Stage21.docx'
DEFAULT_DOCX_STAGE24 = PROJECT_ROOT / 'outputs' / 'paper' / '课程设计论文_终稿_Stage24.docx'
DEFAULT_PDFTOPPM = Path(r'D:\texlive\2024\bin\windows\pdftoppm.exe')

# ---------------------------------------------------------------- 常量

PT_PER_CM = 72.0 / 2.54                      # 28.3465 pt/cm
A4_CONTENT_W_PT = 15.5 * PT_PER_CM           # 21.0 - 3.0 - 2.5 = 15.5 cm ≈ 439.37 pt
A4_CONTENT_H_PT = 24.7 * PT_PER_CM           # 29.7 - 2.5 - 2.5 = 24.7 cm ≈ 700.16 pt

# Word 常量（避免依赖 gencache 生成的命名常量）
WD_ACTIVE_END_PAGE = 3          # wdActiveEndPageNumber
WD_WITH_IN_TABLE = 12           # wdWithInTable
WD_STAT_PAGES = 2               # wdStatisticPages
WD_STAT_WORDS = 0               # wdStatisticWords
WD_EXPORT_FORMAT_PDF = 17       # wdExportFormatPDF
WD_EXPORT_ALL_DOCUMENT = 0      # wdExportAllDocument
WD_EXPORT_DOCUMENT_CONTENT = 0  # wdExportDocumentContent
WD_EXPORT_NO_BOOKMARKS = 0      # wdExportCreateNoBookmarks
WD_EXPORT_OPTIMIZE_PRINT = 0    # wdExportOptimizeForPrint
HF_PRIMARY, HF_FIRST, HF_EVEN = 1, 2, 3

INLINE_SHAPE_TYPE_NAMES = {
    1: 'EmbeddedOLEObject', 2: 'LinkedOLEObject', 3: 'Picture', 4: 'LinkedPicture',
    5: 'OLEControlObject', 6: 'HorizontalLine', 7: 'PictureHorizontalLine',
    8: 'LinkedPictureHorizontalLine', 9: 'ContentControl', 10: 'Placeholder',
    11: 'Chart', 12: 'Chart', 13: 'Group', 14: 'Slicer',
    15: 'SmartArt', 16: 'WebVideo', 17: 'Model3D', 23: 'ContentControl',
}

PAGE_NUMBER_STYLE_NAMES = {
    0: 'Arabic', 1: 'UpperRoman', 2: 'LowerRoman', 3: 'UpperLetter', 4: 'LowerLetter',
    5: 'Ordinal', 6: 'CardinalText', 7: 'OrdinalText', 8: 'Hex',
    10: 'IdeographDigital', 14: 'DecimalFullWidth', 15: 'DecimalHalfWidth',
    18: 'DecimalEnclosedCircle', 19: 'DecimalFullWidth2', 20: 'AiueoFullWidth',
    21: 'IrohaFullWidth', 22: 'DecimalZero',
}

CAPTION_STYLES = ('Figure Caption', 'Table Caption')
# Heading 样式名（英文 + 中文 Word 本地化名）；实际判定以 OutlineLevel 为准
HEADING_STYLES = ('Heading 1', 'Heading 2', 'Heading 3',
                  '标题 1', '标题 2', '标题 3')

# 必须为 0 的字符串（Stage24 提示词 §23/§24；Stage26.6 追加工程命名与已删图号）
FORBIDDEN_STRINGS = [
    '172050', '0.471492', '369组', '369 组', '图17', '模型记忆公司', '对公司的熟悉',
    '可迁移能力的下限估计', 'Stage23', 'Stage24', 'Gate', 'PASS', 'CONTENT_FREEZE',
    'outputs/', 'data/', '.py', 'source:', 'Error!',
    # Stage26.6：工程组名、伪对称误差线、已删除图号与一次评估表述
    'Safe-F', 'SafeF', 'IQR/2', 'IQR / 2',
    '测试集只使用一次', '一次性评估', '完全不再使用测试集',
    '图 3-3', '图 4-7', '图 4-8', '图 5-4', '图 6-4', '中文预训练文本嵌入模型',
]

# ---------------------------------------------------------------- 附录门禁
# Stage24 最终论文取消全部附录：以下计数全部必须为 0
APPENDIX_FIGURE_COUNT = 0       # 附图 X-n 题注数
APPENDIX_TABLE_COUNT = 0        # 附表 X-n 题注数
APPENDIX_HEADING_COUNT = 0      # Heading 1/2/3 中含“附录”（即导航窗格大纲）的标题数
APPENDIX_REFERENCE_COUNT = 0    # 正文中对附录的引用（附图/附表/见附图/见附表/附录 A/B）数

# 异常页数门禁：总页数 >= 该值即 FAIL（提示可能仍有全量表被错误插入）
MAX_TOTAL_PAGE_COUNT = 100

# TOC / 导航（Heading 大纲）中必须为 0 的附录残留串
APPENDIX_TOC_FORBIDDEN = ['附录 A', '附录 B', '附录A', '附录B']

# 可见文本（PDF 抽取）中必须为 0 的附录残留串
APPENDIX_TEXT_FORBIDDEN = ['附图 A-', '附图A-', '附表 B-', '附表B-',
                           '见附图', '见附表', '附录 A', '附录A', '附录 B', '附录B']

# 附录引用 token（用于 APPENDIX_REFERENCE_COUNT 的结构化扫描）
APPENDIX_REFERENCE_TOKENS = ('附图 A-', '附图A-', '附表 B-', '附表B-',
                             '见附图', '见附表', '附录A', '附录 A', '附录B', '附录 B')

# 附录残留串一并纳入 PDF 文本扫描的“必须为 0”清单
FORBIDDEN_STRINGS = FORBIDDEN_STRINGS + [s for s in APPENDIX_TEXT_FORBIDDEN
                                        if s not in FORBIDDEN_STRINGS]

# 必须存在的关键值（Stage26.6 最终一致性修订版：正式模型与全链结果沿用 Stage26.5 复核后的
# 口径，关键值不变；另补入本轮新增的省域/城市两口径值、D 组维度与文本嵌入模型名）
REQUIRED_VALUES = {
    '172063': ['172063', '172,063'],
    '17144': ['17144', '17,144'],
    '17040': ['17040', '17,040'],
    '14883': ['14883', '14,883'],
    '4328': ['4328', '4,328'],
    '4253': ['4253', '4,253'],
    '4795': ['4795', '4,795'],
    '288': ['288'],
    '290': ['290'],
    '56': ['56 维', '56维'],
    '35.48': ['35.48'],
    '64.81': ['64.81'],
    '0.582': ['0.582'],
    '36.00': ['36.00'],
    '51.98': ['51.98'],
    '31.72': ['31.72'],
    '1.68': ['1.68'],
    '1,000': ['1,000'],
    '17131': ['17131', '17,131'],
    '14.88': ['14.88'],
    '1.726': ['1.726'],
    '0.609': ['0.609'],
    '61.96': ['61.96'],
    '0.6389': ['0.6389'],
    'bge-small-zh-v1.5': ['bge-small-zh-v1.5'],
}


# 低特异性字符串（子串计数会被年份/小数污染，额外给“定界计数”）
LOW_SPECIFICITY = ('288', '290', '1.68')

TEXT_OVERLAP_LIMITATION = (
    '实现方法：把页面按行投影切成连续墨迹带（相邻两行之间必须有整行空白才会被切分），'
    '以全文中位行高为基准（跨页估计，可用 --line-height-px 覆盖），输出两类候选：'
    '① waist_merged_pair——带高 ≥1.6×行高、带内存在墨迹“腰”（峰值 8%~35% 的谷），'
    '腰上下两半各自高度 ≥0.6×行高且水平重叠 >30%；'
    '② dense_merged_block——带高 1.6~3.5×行高、带内无腰（逐行墨迹均 ≥40% 峰值）'
    '且宽度 ≥50% 页宽，即多行墨迹完全糊成一片。'
    '高度 >3.5×行高的带按整幅图片/公式块排除（数量单独计数）。'
    '局限：① 命中只说明相邻文本行的墨迹在垂直方向连成一片（行距过密、括号/上下标、'
    '西文升降部、表格密排、公式都可能造成），**不等价于真实文字互相覆盖**；'
    '② 无法发现水平方向的字距碰撞、文字与图片/表格/页眉页脚的重叠、'
    '同带内三条以上行部分互叠、浮动对象覆盖；'
    '③ 阈值依赖渲染 dpi（默认 110），改 dpi 需同步调整像素阈值。'
    '命中项均为候选，需结合 page-map 的段落/表格页码复核。'
)


class QaError(RuntimeError):
    """工具链可预期的失败（参数/文件缺失等），给出清晰报错而非堆栈。"""


# ---------------------------------------------------------------- 通用工具

def log(msg):
    print(str(msg), flush=True)


def now_iso():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def require_file(path, what='文件'):
    p = Path(path).resolve()
    if not p.is_file():
        raise QaError('%s不存在：%s\n（请检查路径；若为 Stage24 docx，需先由 Stage24 生成步骤产出）' % (what, p))
    return p


def ensure_under_qa_dir(path):
    """只允许把产物写到 outputs/paper/qa/ 之下。"""
    p = Path(path).resolve()
    qa = QA_ROOT.resolve()
    try:
        p.relative_to(qa)
    except ValueError:
        raise QaError('只允许在 %s 下写盘，收到：%s' % (qa, p))
    return p


def derive_qa_pdf_path(work_dir, docx_path):
    """QA PDF 默认命名：<源 docx 名>_QA.pdf（去掉验证副本的 _qa_copy 后缀）。"""
    stem = Path(docx_path).stem
    if stem.endswith('_qa_copy'):
        stem = stem[: -len('_qa_copy')]
    return Path(work_dir) / (stem + '_QA.pdf')


def dump_json(path, obj):
    p = ensure_under_qa_dir(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    return p


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def trim_text(text, n=60):
    t = (text or '')
    t = t.replace('\r', ' ').replace('\x07', ' ').replace('\x0b', ' ').replace('\x0c', ' ')
    t = re.sub(r'[\x00-\x08\x0e-\x1f]', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t[:n]


# ---------------------------------------------------------------- Word COM 封装

def open_word_app():
    try:
        import win32com.client as win32
    except Exception as exc:  # noqa: BLE001
        raise QaError('无法导入 win32com.client（需要 pywin32）：%r' % (exc,))
    try:
        word = win32.gencache.EnsureDispatch('Word.Application')
    except Exception:
        word = win32.Dispatch('Word.Application')  # 缓存损坏时退化为后期绑定
    word.Visible = False
    word.DisplayAlerts = 0
    try:
        word.ScreenUpdating = False
    except Exception:
        pass
    return word


def _quit_word(word):
    """释放 Word 会话：关闭进程并回收 COM 引用（异常时也必须调用）。"""
    try:
        if word is not None:
            word.Quit()
    except Exception:
        pass
    gc.collect()


def _is_rpc_error(exc):
    text = str(exc)
    return ('RPC' in text) or ('-2147023174' in text) or ('-2147417848' in text)


def run_in_word_session(fn, attempts=3):
    """为 fn(word) 提供可用 Word 会话；会话掉线（RPC 不可用）时重开重试。

    Word 是单实例 COM 服务器，前一次 Quit() 尚未结束时立刻新建会话会出现
    “RPC 服务器不可用”，因此这里在异常时等待并重建会话。
    """
    last = None
    for attempt in range(1, attempts + 1):
        word = None
        try:
            word = open_word_app()
            return fn(word)
        except Exception as exc:  # noqa: BLE001
            last = exc
            if not _is_rpc_error(exc) or attempt == attempts:
                raise
            log('[word] Word 会话中断（第 %d 次），%d 秒后重开会话…' % (attempt, 2 * attempt))
            time.sleep(2 * attempt)
        finally:
            _quit_word(word)
    raise last


def _story_chain(story_range):
    """StoryRange 及其 NextStoryRange 链（正文/页眉/页脚/文本框）。"""
    seen = 0
    rng = story_range
    while rng is not None and seen < 12:
        yield rng
        try:
            rng = rng.NextStoryRange
        except Exception:
            return
        seen += 1


def update_all_fields(doc):
    """更新全部域：doc.Fields → TOC/TOF → StoryRanges → 页眉页脚 → 二次更新。"""
    stats = {'doc_fields': 0, 'toc': 0, 'tof': 0, 'story_ranges': 0,
             'header_footer': 0, 'per_field': 0, 'errors': []}

    def _try(label, fn):
        try:
            fn()
            return True
        except Exception as exc:  # noqa: BLE001
            stats['errors'].append('%s: %r' % (label, exc)[:220])
            return False

    if _try('doc.Fields.Update', lambda: doc.Fields.Update()):
        stats['doc_fields'] = 1

    for i in range(1, doc.TablesOfContents.Count + 1):
        if _try('TablesOfContents(%d).Update' % i, lambda i=i: doc.TablesOfContents(i).Update()):
            stats['toc'] += 1
    try:
        tof_count = doc.TablesOfFigures.Count
    except Exception:
        tof_count = 0
    for i in range(1, tof_count + 1):
        if _try('TablesOfFigures(%d).Update' % i, lambda i=i: doc.TablesOfFigures(i).Update()):
            stats['tof'] += 1

    # StoryRanges（含链接的 NextStoryRange）内域二次更新
    try:
        story_ranges = list(doc.StoryRanges)
    except Exception:
        story_ranges = []
    for story in story_ranges:
        for rng in _story_chain(story):
            if _try('StoryRange.Fields.Update', lambda rng=rng: rng.Fields.Update()):
                stats['story_ranges'] += 1

    # 页眉页脚内域（正文 StoryRanges 已含，但显式再走一遍更稳）
    for i in range(1, doc.Sections.Count + 1):
        for coll_name in ('Headers', 'Footers'):
            coll = getattr(doc.Sections(i), coll_name)
            for j in (HF_PRIMARY, HF_FIRST, HF_EVEN):
                if _try('%s(%d).Range.Fields.Update' % (coll_name, j),
                        lambda coll=coll, j=j: coll(j).Range.Fields.Update()):
                    stats['header_footer'] += 1

    # 兜底：逐域更新（doc.Fields.Update 可能跳过锁定域）
    n_ok = 0
    try:
        fields = list(doc.Fields)
    except Exception:
        fields = []
    for f in fields:
        try:
            f.Update()
            n_ok += 1
        except Exception:  # noqa: BLE001
            pass
    stats['per_field'] = n_ok

    # 更新完成后再次整体更新（TOC 页码已稳定）
    _try('doc.Fields.Update(2nd)', lambda: doc.Fields.Update())
    return stats


def count_field_errors(doc):
    bad = []
    try:
        fields = list(doc.Fields)
    except Exception:
        return bad
    for i, f in enumerate(fields, start=1):
        try:
            text = f.Result.Text or ''
        except Exception:
            text = ''
        if ('Error' in text) or ('错误' in text) or ('未找到' in text):
            bad.append({'field_index': i, 'result': trim_text(text, 80)})
    return bad


def doc_basic_stats(doc):
    def _safe(fn, default=None):
        try:
            return fn()
        except Exception:
            return default

    return {
        'pages': _safe(lambda: doc.ComputeStatistics(WD_STAT_PAGES)),
        'words': _safe(lambda: doc.ComputeStatistics(WD_STAT_WORDS)),
        'sections': _safe(lambda: doc.Sections.Count),
        'paragraphs': _safe(lambda: doc.Paragraphs.Count),
        'tables': _safe(lambda: doc.Tables.Count),
        'inline_shapes': _safe(lambda: doc.InlineShapes.Count),
        'floating_shapes': _safe(lambda: doc.Shapes.Count),
        'fields': _safe(lambda: doc.Fields.Count),
        'toc_count': _safe(lambda: doc.TablesOfContents.Count),
    }


# ---------------------------------------------------------------- 1) open-update-export

def do_export(word, target_docx, pdf, source_docx=None):
    """在既有 Word 会话内：打开 → 更新全部域 → 保存回原路径 → 导出 PDF。

    只更新域与页码，不触碰任何正文内容。
    """
    target_docx = Path(target_docx)
    pdf = Path(pdf)
    info = {
        'generated_at': now_iso(),
        'source_docx': str(source_docx or target_docx),
        'working_docx': str(target_docx),
        'pdf': str(pdf),
    }
    doc = None
    try:
        doc = word.Documents.Open(str(target_docx), ReadOnly=False, AddToRecentFiles=False)
        info['field_update'] = update_all_fields(doc)
        doc.Repaginate()
        info['doc_stats'] = doc_basic_stats(doc)
        info['field_errors'] = count_field_errors(doc)
        info['field_error_count'] = len(info['field_errors'])
        doc.Save()
        info['saved_docx'] = True
        doc.ExportAsFixedFormat(
            OutputFileName=str(pdf), ExportFormat=WD_EXPORT_FORMAT_PDF,
            OpenAfterExport=False, OptimizeFor=WD_EXPORT_OPTIMIZE_PRINT,
            Range=WD_EXPORT_ALL_DOCUMENT, Item=WD_EXPORT_DOCUMENT_CONTENT,
            IncludeDocProps=True, KeepIRM=True, CreateBookmarks=WD_EXPORT_NO_BOOKMARKS,
            DocStructureTags=True, BitmapMissingFonts=True, UseISO19005_1=False,
        )
        info['exported_pdf'] = True
    finally:
        try:
            if doc is not None:
                doc.Close(SaveChanges=0)
        except Exception:
            pass
        doc = None
        gc.collect()

    if not pdf.is_file():
        raise QaError('PDF 导出失败（未生成文件）：%s' % pdf)
    info['pdf_size'] = pdf.stat().st_size
    info['docx_size'] = target_docx.stat().st_size
    return info


def cmd_open_update_export(args):
    docx = require_file(args.docx, 'docx')
    pdf = Path(args.pdf).resolve() if args.pdf else derive_qa_pdf_path(args.work_dir, docx)
    ensure_under_qa_dir(pdf)

    if args.copy_docx and not docx.stem.endswith('_qa_copy'):
        work = Path(args.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        target_docx = ensure_under_qa_dir(work / (docx.stem + '_qa_copy.docx'))
        shutil.copy2(str(docx), str(target_docx))
    else:
        target_docx = docx

    info = run_in_word_session(lambda word: do_export(word, target_docx, pdf, source_docx=docx))
    info['generated_at'] = now_iso()
    out = dump_json(Path(args.work_dir) / 'export_info.json', info)
    log('[open-update-export] docx=%s' % target_docx)
    log('[open-update-export] pdf=%s (%d bytes) pages=%s' % (
        pdf, info['pdf_size'], (info.get('doc_stats') or {}).get('pages')))
    log('[open-update-export] 域更新：%s，域错误 %d 处' % (
        {k: v for k, v in info['field_update'].items() if k != 'errors'}, info['field_error_count']))
    log('[open-update-export] info=%s' % out)
    return info


# ---------------------------------------------------------------- 2) page-map

def _pnum_style_name(value):
    if value is None:
        return None
    return PAGE_NUMBER_STYLE_NAMES.get(int(value), 'Style%d' % int(value))


def _section_info(doc, idx):
    sec = doc.Sections(idx)
    ps = sec.PageSetup
    info = {'index': idx}
    try:
        start_rng = doc.Range(sec.Range.Start, sec.Range.Start)
        info['start_page'] = int(start_rng.Information(WD_ACTIVE_END_PAGE))
    except Exception:
        info['start_page'] = None
    try:
        width_pt = float(ps.PageWidth)
        height_pt = float(ps.PageHeight)
        left = float(ps.LeftMargin)
        right = float(ps.RightMargin)
        top = float(ps.TopMargin)
        bottom = float(ps.BottomMargin)
        info.update({
            'page_width_cm': round(width_pt / PT_PER_CM, 3),
            'page_height_cm': round(height_pt / PT_PER_CM, 3),
            'margin_left_cm': round(left / PT_PER_CM, 3),
            'margin_right_cm': round(right / PT_PER_CM, 3),
            'margin_top_cm': round(top / PT_PER_CM, 3),
            'margin_bottom_cm': round(bottom / PT_PER_CM, 3),
            'content_width_pt': round(width_pt - left - right, 3),
            'content_height_pt': round(height_pt - top - bottom, 3),
            'orientation': 'landscape' if int(getattr(ps, 'Orientation', 0)) == 1 else 'portrait',
        })
    except Exception:
        info['page_setup_error'] = True
    try:
        pn = sec.Footers(HF_PRIMARY).PageNumbers
        info['page_number_style'] = _pnum_style_name(getattr(pn, 'NumberStyle', None))
        info['restart_numbering'] = bool(pn.RestartNumberingAtSection)
        info['starting_number'] = int(pn.StartingNumber)
    except Exception:
        info['page_numbering_error'] = True
    try:
        info['different_first_page'] = bool(ps.DifferentFirstPageHeaderFooter)
    except Exception:
        pass
    return info


def collect_page_map(docx_path, text_head_len=60):
    docx_path = require_file(docx_path, 'docx')
    return run_in_word_session(lambda word: do_page_map(word, docx_path, text_head_len))


def do_page_map(word, docx_path, text_head_len=60):
    """在既有 Word 会话内以只读方式构建「逐段落 → 页码」映射。"""
    docx_path = Path(docx_path)
    doc = None
    try:
        doc = word.Documents.Open(str(docx_path), ReadOnly=True, AddToRecentFiles=False)
        doc.Repaginate()

        result = {
            'generated_at': now_iso(),
            'docx': str(docx_path),
            'doc_stats': doc_basic_stats(doc),
            'sections': [],
            'inline_shapes': [],
            'tables': [],
            'paragraphs': [],
        }

        for i in range(1, doc.Sections.Count + 1):
            result['sections'].append(_section_info(doc, i))

        # InlineShape：页码 / 尺寸(磅) / 类型
        for i in range(1, doc.InlineShapes.Count + 1):
            s = doc.InlineShapes(i)
            row = {'index': i}
            # Information(3) 取的是 range 的 active end 所在页，故用起点页更准确
            try:
                row['page'] = int(doc.Range(s.Range.Start, s.Range.Start)
                                  .Information(WD_ACTIVE_END_PAGE))
            except Exception:
                row['page'] = None
            for key, attr in (('width_pt', 'Width'), ('height_pt', 'Height')):
                try:
                    row[key] = round(float(getattr(s, attr)), 2)
                except Exception:
                    row[key] = None
            try:
                t = int(s.Type)
                row['type'] = t
                row['type_name'] = INLINE_SHAPE_TYPE_NAMES.get(t, 'Type%d' % t)
            except Exception:
                row['type'] = None
            try:
                row['alt_text'] = trim_text(s.AlternativeText or '', 60)
            except Exception:
                pass
            result['inline_shapes'].append(row)

        # Table：页码 / 行列 / 宽度(磅) / 标题行重复
        for i in range(1, doc.Tables.Count + 1):
            t = doc.Tables(i)
            row = {'index': i}
            # page = 表格起始页；end_page = 表格结束页（跨页表两者不同）
            try:
                row['page'] = int(doc.Range(t.Range.Start, t.Range.Start)
                                  .Information(WD_ACTIVE_END_PAGE))
            except Exception:
                row['page'] = None
            try:
                row['end_page'] = int(t.Range.Information(WD_ACTIVE_END_PAGE))
            except Exception:
                row['end_page'] = None
            for key, fn in (('rows', lambda: int(t.Rows.Count)),
                            ('cols', lambda: int(t.Columns.Count))):
                try:
                    row[key] = fn()
                except Exception:
                    row[key] = None
            width = None
            pref_raw = None
            pref_type = None
            try:
                pref_raw = float(t.PreferredWidth)
                pref_type = int(t.PreferredWidthType)
                # 9999999 是 Word 的“自动宽度”哨兵值，不能当作实际磅宽
                if pref_type == 3 and 0 < pref_raw < 1e6:
                    width = pref_raw
            except Exception:
                pass
            if not width:
                try:
                    width = float(sum(float(t.Columns(c).Width) for c in range(1, t.Columns.Count + 1)))
                except Exception:
                    width = None
            row['width_pt'] = round(width, 2) if width else None
            if pref_raw is not None:
                row['preferred_width_raw'] = pref_raw
                row['preferred_width_undefined'] = bool(pref_raw >= 1e6)
            if pref_type is not None:
                row['preferred_width_type'] = pref_type
            try:
                row['heading_format'] = bool(t.Rows(1).HeadingFormat)
            except Exception:
                row['heading_format'] = None
            try:
                row['first_row_text'] = trim_text(t.Cell(1, 1).Range.Text, 40)
            except Exception:
                pass
            result['tables'].append(row)

        # 段落 → 页码
        shape_cursor = 0
        table_cursor = 0
        prev_in_table = False
        for idx, p in enumerate(doc.Paragraphs, start=1):
            row = {'index': idx}
            try:
                row['style'] = p.Style.NameLocal
            except Exception:
                row['style'] = None
            # OutlineLevel：1~3 = Heading 1~3（与导航窗格大纲一致，不受界面语言影响）
            try:
                ol = int(p.OutlineLevel)
            except Exception:
                ol = None
            row['outline_level'] = ol
            try:
                page = p.Range.Information(WD_ACTIVE_END_PAGE)
                row['page'] = int(page) if page else None
            except Exception:
                row['page'] = None
            # start_page：段落“起始页”。Information(wdActiveEndPageNumber) 取的是 range
            # 的 active end（末页），跨页段落会被记到末页，判定“孤立标题/同页”时需用起始页。
            try:
                sp = doc.Range(p.Range.Start, p.Range.Start).Information(WD_ACTIVE_END_PAGE)
                row['start_page'] = int(sp) if sp else None
            except Exception:
                row['start_page'] = None
            try:
                in_table = bool(p.Range.Information(WD_WITH_IN_TABLE))
            except Exception:
                in_table = False
            row['in_table'] = in_table
            if in_table and not prev_in_table:
                table_cursor += 1
            row['table_index'] = table_cursor if in_table else None
            prev_in_table = in_table
            try:
                text = trim_text(p.Range.Text, text_head_len)
            except Exception:
                text = ''
            row['text_head'] = text
            row['empty'] = (text == '')
            try:
                n_shapes = int(p.Range.InlineShapes.Count)
            except Exception:
                n_shapes = 0
            row['shape_indices'] = list(range(shape_cursor + 1, shape_cursor + n_shapes + 1))
            row['has_inline_shape'] = n_shapes > 0
            shape_cursor += n_shapes
            row['is_figure_caption'] = bool(row['style'] and str(row['style']).startswith(CAPTION_STYLES[0]))
            row['is_table_caption'] = bool(row['style'] and str(row['style']).startswith(CAPTION_STYLES[1]))
            row['is_heading'] = bool(
                (ol is not None and 1 <= ol <= 3)
                or (row['style'] and str(row['style']) in HEADING_STYLES))
            result['paragraphs'].append(row)

        # 文档级版心（取出现最多的内容宽/高，作为正文版心基准）
        widths = [s.get('content_width_pt') for s in result['sections'] if s.get('content_width_pt')]
        heights = [s.get('content_height_pt') for s in result['sections'] if s.get('content_height_pt')]
        result['content_box_pt'] = {
            'width': max(widths) if widths else round(A4_CONTENT_W_PT, 2),
            'height': max(heights) if heights else round(A4_CONTENT_H_PT, 2),
            'portrait_width': min(widths) if widths else round(A4_CONTENT_W_PT, 2),
            'a4_reference_width': round(A4_CONTENT_W_PT, 2),
            'a4_reference_height': round(A4_CONTENT_H_PT, 2),
            'note': 'width/height 取全部节中的最大值（兼容横向 section）；portrait_width 为最小节宽（纵向版心）。',
        }
        return result
    finally:
        try:
            if doc is not None:
                doc.Close(SaveChanges=0)
        except Exception:
            pass
        doc = None
        gc.collect()


PARAGRAPH_CSV_FIELDS = ['index', 'page', 'style', 'in_table', 'table_index',
                        'has_inline_shape', 'empty', 'is_figure_caption',
                        'is_table_caption', 'is_heading', 'text_head']


def write_page_map_csv(path, page_map):
    p = ensure_under_qa_dir(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8-sig', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=PARAGRAPH_CSV_FIELDS, extrasaction='ignore')
        writer.writeheader()
        for row in page_map['paragraphs']:
            writer.writerow(row)
    return p


def cmd_page_map(args):
    page_map = collect_page_map(args.docx, text_head_len=args.text_head_len)
    out_json = dump_json(args.out_json, page_map)
    out_csv = write_page_map_csv(args.out_csv, page_map)
    log('[page-map] docx=%s' % page_map['docx'])
    log('[page-map] 段落=%d 图(InlineShape)=%d 表=%d 节=%d 页数=%s' % (
        len(page_map['paragraphs']), len(page_map['inline_shapes']),
        len(page_map['tables']), len(page_map['sections']),
        (page_map.get('doc_stats') or {}).get('pages')))
    log('[page-map] json=%s' % out_json)
    log('[page-map] csv=%s' % out_csv)
    return page_map


# ---------------------------------------------------------------- 3) render

def render_pdf_to_png(pdf, out_dir, dpi=110, pdftoppm=None, prefix='page'):
    pdf = require_file(pdf, 'PDF')
    exe = Path(pdftoppm or DEFAULT_PDFTOPPM)
    if not exe.is_file():
        raise QaError('未找到 pdftoppm：%s（可用 --pdftoppm 指定绝对路径）' % exe)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob('%s-*.png' % prefix):
        old.unlink()
    cmd = [str(exe), '-r', str(dpi), '-png', str(pdf), str(out_dir / prefix)]
    proc = subprocess.run(cmd, capture_output=True, text=True, errors='replace')
    if proc.returncode != 0:
        raise QaError('pdftoppm 渲染失败（exit=%d）：%s\n%s' % (
            proc.returncode, ' '.join(cmd), (proc.stderr or '')[:500]))

    # pdftoppm 的零填充位数随页数变化，统一规范为 page-01.png / page-02.png …
    files = sorted(out_dir.glob('%s-*.png' % prefix),
                   key=lambda x: int(re.findall(r'(\d+)$', x.stem)[0]))
    total = len(files)
    width = max(2, len(str(total)))
    for f in files:
        num = int(re.findall(r'(\d+)$', f.stem)[0])
        target = out_dir / ('%s-%0*d.png' % (prefix, width, num))
        if f.name != target.name:
            f.replace(target)
    files = sorted(out_dir.glob('%s-*.png' % prefix),
                   key=lambda x: int(re.findall(r'(\d+)$', x.stem)[0]))
    return {'pdf': str(pdf), 'out_dir': str(out_dir), 'dpi': dpi, 'cmd': cmd,
            'count': len(files), 'files': [f.name for f in files]}


def cmd_render(args):
    info = render_pdf_to_png(args.pdf, args.out_dir, dpi=args.dpi, pdftoppm=args.pdftoppm)
    info['generated_at'] = now_iso()
    log('[render] %d 张 PNG → %s（dpi=%d）' % (info['count'], info['out_dir'], info['dpi']))
    return info


# ---------------------------------------------------------------- 4) inspect

def _load_png_pages(png_dir, prefix='page'):
    png_dir = Path(png_dir)
    if not png_dir.is_dir():
        raise QaError('渲染目录不存在：%s（请先运行 render）' % png_dir)
    files = sorted(png_dir.glob('%s-*.png' % prefix),
                   key=lambda x: int(re.findall(r'(\d+)$', x.stem)[0]))
    if not files:
        raise QaError('渲染目录内没有 %s-*.png：%s' % (prefix, png_dir))
    pages = []
    for f in files:
        num = int(re.findall(r'(\d+)$', f.stem)[0])
        pages.append((num, f))
    return pages


def _ink_array(gray_arr, ink_threshold):
    return gray_arr < ink_threshold


def _bands(row_mask):
    """按行投影切分连续墨迹带，返回 [(y0, y1), ...]（闭区间）。"""
    bands = []
    y = 0
    n = len(row_mask)
    while y < n:
        if not row_mask[y]:
            y += 1
            continue
        y0 = y
        while y < n and row_mask[y]:
            y += 1
        bands.append((y0, y - 1))
    return bands


def page_pixel_metrics(png_path, ink_threshold=250, edge_frac=0.01):
    img = Image.open(png_path)
    gray = np.asarray(img.convert('L'))
    h, w = gray.shape
    ink = _ink_array(gray, ink_threshold)
    total = float(h * w)
    metrics = {
        'file': Path(png_path).name,
        'width_px': int(w), 'height_px': int(h),
        'ink_pixels': int(ink.sum()),
        'ink_ratio': round(float(ink.sum()) / total, 6),
    }
    if ink.any():
        ys, xs = np.where(ink)
        metrics['ink_bbox'] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    else:
        metrics['ink_bbox'] = None
    ey = max(1, int(round(edge_frac * h)))
    ex = max(1, int(round(edge_frac * w)))
    metrics['edge_band_px'] = {'top_bottom': ey, 'left_right': ex}
    metrics['edge_ink'] = {
        'top': int(ink[:ey, :].sum()),
        'bottom': int(ink[h - ey:, :].sum()),
        'left': int(ink[:, :ex].sum()),
        'right': int(ink[:, w - ex:].sum()),
    }
    return metrics, gray


def estimate_line_height_px(pages, ink_threshold=250, edge_frac=0.01, min_bands=5):
    """跨页估计全文单行墨迹高度（中位行高的中位数），作为行距先验。"""
    meds = []
    for _num, path in pages:
        gray = np.asarray(Image.open(path).convert('L'))
        ink = _ink_array(gray, ink_threshold)
        h = ink.shape[0]
        m = max(1, int(round(edge_frac * h)))
        core = ink[m:h - m, :]
        if core.size == 0:
            continue
        row_cnt = core.sum(axis=1)
        hs = [b[1] - b[0] + 1 for b in _bands(row_cnt > 0) if (b[1] - b[0] + 1) >= 6]
        if len(hs) >= min_bands:
            meds.append(float(np.median(hs)))
    return float(np.median(meds)) if meds else None


def check_text_overlap(gray, page_no, ink_threshold=250, edge_frac=0.01,
                       min_band_h_px=6, band_factor=1.6, max_band_factor=3.5,
                       min_subband_frac=0.6, waist_low=0.08,
                       waist_high=0.35, overlap_min=0.30, min_width_px=40,
                       min_row_ink=20, line_height_px=None,
                       dense_min_ratio=0.40, dense_width_frac=0.50):
    """行投影墨迹带切分 → 检出「两行粘连」候选。

    返回 ``(hits, excluded_large_blocks)``。

    两类候选：
    1. ``waist_merged_pair``：带高 ≥ band_factor×行高、带内存在墨迹“腰”，
       腰上下两半各自高度 ≥ min_subband_frac×行高，且水平重叠 > overlap_min。
    2. ``dense_merged_block``：带高在 [1.9, max_band_factor]×行高、带内**无腰**
       （内部逐行墨迹均 ≥ dense_min_ratio×峰值），且宽度 ≥ dense_width_frac×页宽——
       典型的“多行墨迹完全糊成一片”（真实压盖或极端紧凑行距）。

    行高优先用 ``line_height_px``（全文先验），否则退回本页中位行高。
    高度 > max_band_factor×行高的带按整幅图片/公式块排除，单独计数。
    """
    ink = _ink_array(gray, ink_threshold)
    h, w = ink.shape
    m = max(1, int(round(edge_frac * h)))
    core = ink[m:h - m, :]
    if core.size == 0:
        return [], []
    row_cnt = core.sum(axis=1)
    bands = _bands(row_cnt > 0)
    heights = [b[1] - b[0] + 1 for b in bands if (b[1] - b[0] + 1) >= min_band_h_px]
    if not bands or not heights:
        return [], []
    if line_height_px:
        med_h = float(line_height_px)
    else:
        if len(bands) < 3:
            return [], []
        med_h = float(np.median(heights))
    hits = []
    excluded_large = []

    def _x_extent(sub):
        xs = np.where(sub.any(axis=0))[0]
        if xs.size == 0:
            return None
        return int(xs.min()), int(xs.max())

    def _overlap_ratio(ext_a, ext_b):
        if ext_a is None or ext_b is None:
            return None
        a0, a1 = ext_a
        b0, b1 = ext_b
        aw, bw = a1 - a0 + 1, b1 - b0 + 1
        if aw < min_width_px or bw < min_width_px:
            return None
        ov = min(a1, b1) - max(a0, b0) + 1
        if ov <= 0:
            return None
        return ov / float(min(aw, bw))

    for (y0, y1) in bands:
        hh = y1 - y0 + 1
        if hh < max(min_band_h_px, band_factor * med_h):
            continue
        if hh > max_band_factor * med_h:
            ext = _x_extent(core[y0:y1 + 1, :])
            excluded_large.append({
                'page': page_no,
                'band_top_px': int(y0 + m),
                'band_bottom_px': int(y1 + m),
                'band_height_px': int(hh),
                'band_height_over_line': round(hh / med_h, 2),
                'band_width_px': (ext[1] - ext[0] + 1) if ext else None,
                'note': '按整幅图片/公式块排除；若高度≈整数倍行高且宽度≈正文宽，可能是多行文本完全糊合',
            })
            continue
        seg = row_cnt[y0:y1 + 1]
        mx = int(seg.max())
        if mx < min_row_ink:
            continue
        lo_k = max(1, int(0.25 * hh))
        hi_k = min(hh - 2, int(0.75 * hh))
        waist = None
        for k in range(lo_k, hi_k + 1):
            v = int(seg[k])
            if waist_low * mx <= v <= waist_high * mx:
                if waist is None or v < int(seg[waist]):
                    waist = k

        if waist is not None:
            upper = core[y0:y0 + waist + 1, :]
            lower = core[y0 + waist + 1:y1 + 1, :]
            if upper.shape[0] < min_subband_frac * med_h \
                    or lower.shape[0] < min_subband_frac * med_h:
                continue
            ratio = _overlap_ratio(_x_extent(upper), _x_extent(lower))
            if ratio is None or ratio < overlap_min:
                continue
            hits.append({
                'page': page_no,
                'sub_type': 'waist_merged_pair',
                'band_top_px': int(y0 + m),
                'band_bottom_px': int(y1 + m),
                'band_height_px': int(hh),
                'line_height_px': round(med_h, 1),
                'band_height_over_line': round(hh / med_h, 2),
                'waist_row_px': int(y0 + waist + m),
                'upper_height_px': int(upper.shape[0]),
                'lower_height_px': int(lower.shape[0]),
                'overlap_ratio': round(ratio, 3),
            })
            continue

        # 无腰但整带墨迹饱和：疑似多行完全糊成一片
        inner = seg[2:-2] if hh > 4 else seg
        if inner.size == 0 or int(inner.min()) < dense_min_ratio * mx:
            continue
        ext = _x_extent(core[y0:y1 + 1, :])
        if ext is None or (ext[1] - ext[0] + 1) < dense_width_frac * w:
            continue
        mid = y0 + hh // 2
        ratio = _overlap_ratio(_x_extent(core[y0:mid, :]), _x_extent(core[mid:y1 + 1, :]))
        hits.append({
            'page': page_no,
            'sub_type': 'dense_merged_block',
            'band_top_px': int(y0 + m),
            'band_bottom_px': int(y1 + m),
            'band_height_px': int(hh),
            'line_height_px': round(med_h, 1),
            'band_height_over_line': round(hh / med_h, 2),
            'estimated_line_count': round(hh / med_h, 2),
            'min_interior_row_ink': int(inner.min()),
            'max_row_ink': mx,
            'band_width_px': ext[1] - ext[0] + 1,
            'overlap_ratio': round(ratio, 3) if ratio is not None else None,
        })
    return hits, excluded_large


CAPTION_PATTERNS = (
    ('main_figure', re.compile(r'^图(\d+)[-–—](\d+)')),
    ('appendix_figure', re.compile(r'^附图([A-Za-z])[-–—](\d+)')),
    ('main_table', re.compile(r'^表(\d+)[-–—](\d+)')),
    ('appendix_table', re.compile(r'^附表([A-Za-z])[-–—](\d+)')),
    ('audit_figure', re.compile(r'^图S(\d+)')),
)


def parse_caption(text):
    t = re.sub(r'\s+', '', text or '')
    if not t:
        return None
    for kind, pattern in CAPTION_PATTERNS:
        m = pattern.match(t)
        if not m:
            continue
        if kind == 'audit_figure':
            return {'kind': kind, 'group': 'S', 'number': int(m.group(1)), 'token': m.group(0)}
        if kind in ('main_figure', 'main_table'):
            return {'kind': kind, 'group': m.group(1), 'number': int(m.group(2)), 'token': m.group(0)}
        return {'kind': kind, 'group': m.group(1).upper(), 'number': int(m.group(2)), 'token': m.group(0)}
    return None


def check_caption_numbering(page_map):
    series = {}
    entries = []
    other = []
    for row in page_map['paragraphs']:
        if not (row.get('is_figure_caption') or row.get('is_table_caption')):
            continue
        parsed = parse_caption(row.get('text_head') or '')
        entry = {
            'paragraph_index': row['index'],
            'page': row.get('page'),
            'style': row.get('style'),
            'text': row.get('text_head'),
        }
        if parsed is None:
            other.append(entry)
            continue
        entry.update(parsed)
        entries.append(entry)
        series.setdefault(parsed['kind'], {}).setdefault(parsed['group'], []).append(entry)

    issues = []
    summary = {}
    for kind, groups in series.items():
        summary[kind] = {}
        for group, items in groups.items():
            numbers = [it['number'] for it in items]
            duplicates = sorted({n for n in numbers if numbers.count(n) > 1})
            expected = set(range(1, max(numbers) + 1))
            missing = sorted(expected - set(numbers))
            out_of_order = [items[i]['number'] for i in range(1, len(items))
                            if items[i]['number'] < items[i - 1]['number']]
            summary[kind][group] = {
                'order': numbers,
                'count': len(numbers),
                'duplicates': duplicates,
                'missing': missing,
                'out_of_order': out_of_order,
            }
            if duplicates:
                issues.append({'kind': kind, 'group': group, 'issue': 'duplicate',
                               'numbers': duplicates})
            if missing:
                issues.append({'kind': kind, 'group': group, 'issue': 'missing',
                               'numbers': missing})
            if out_of_order:
                issues.append({'kind': kind, 'group': group, 'issue': 'out_of_order',
                               'numbers': out_of_order})
    if other:
        issues.append({'issue': 'unparsed_caption', 'count': len(other),
                       'items': [{'paragraph_index': o['paragraph_index'], 'text': o['text']}
                                 for o in other]})
    return {'count': len(issues), 'series': summary, 'issues': issues,
            'caption_count': len(entries), 'unparsed_count': len(other),
            'entries': entries}


def _compact(text):
    return re.sub(r'\s+', '', trim_text(text, 200) or '')


def _is_toc_style(style):
    """TOC 段落样式（英文 toc 1/2/3 与中文 Word 的本地化名）。"""
    s = str(style or '').lower()
    return s.startswith('toc') or s.startswith('目录')


def check_appendix_residue(page_map):
    """附录残留检查：题注 / 标题（导航大纲）/ TOC / 正文引用。

    - ``figure_caption_count``：``附图 X-n`` 形式图题数（门禁 0）
    - ``table_caption_count``：``附表 X-n`` 形式表题数（门禁 0）
    - ``heading_count``：Heading 1/2/3 文本含“附录”的标题数（即导航窗格大纲）
    - ``toc_count``：目录（toc 1/2/3）段落文本含“附录”的段落数
    - ``reference_count``：正文段落中对附录的引用 token 数（见附图/见附表/附录 A …）
    """
    paras = page_map.get('paragraphs') or []
    fig_hits, tbl_hits, head_hits, toc_hits, ref_hits = [], [], [], [], []
    for row in paras:
        style = str(row.get('style') or '')
        raw = row.get('text_head') or ''
        compact = _compact(raw)
        entry = {'paragraph_index': row.get('index'), 'page': row.get('page'),
                 'style': style, 'text': trim_text(raw, 60)}
        if row.get('is_figure_caption') or row.get('is_table_caption'):
            parsed = parse_caption(raw)
            if parsed is not None and parsed['kind'] == 'appendix_figure':
                fig_hits.append(entry)
            elif parsed is not None and parsed['kind'] == 'appendix_table':
                tbl_hits.append(entry)
        if row.get('is_heading') and any(_compact(t) in compact
                                         for t in APPENDIX_TOC_FORBIDDEN):
            head_hits.append(entry)
        if _is_toc_style(style) and any(_compact(t) in compact
                                        for t in APPENDIX_TOC_FORBIDDEN):
            toc_hits.append(entry)
        if any(_compact(t) in compact for t in APPENDIX_REFERENCE_TOKENS) \
                and not (row.get('is_heading') or row.get('is_figure_caption')
                         or row.get('is_table_caption') or _is_toc_style(style)):
            ref_hits.append(entry)
    return {
        'figure_caption_count': len(fig_hits),
        'table_caption_count': len(tbl_hits),
        'heading_count': len(head_hits),
        'toc_count': len(toc_hits),
        'reference_count': len(ref_hits),
        'count': len(fig_hits) + len(tbl_hits) + len(head_hits) + len(toc_hits) + len(ref_hits),
        'figure_caption_hits': fig_hits,
        'table_caption_hits': tbl_hits,
        'heading_hits': head_hits,
        'toc_hits': toc_hits,
        'reference_hits': ref_hits,
        'note': '标题扫描基于 Heading 1/2/3 段落（导航窗格大纲来源）；'
                'TOC 扫描基于 toc 1/2/3 段落（TOC 域更新后的目录文本）；'
                '题注扫描基于 Figure/Table Caption 样式段落。'
                '门禁：上述计数全部必须为 0。',
    }


def check_caption_separation(page_map, lookback=6):
    paras = page_map['paragraphs']
    by_index = {p['index']: p for p in paras}
    shape_page = {s['index']: s.get('page') for s in page_map['inline_shapes']}
    table_page = {t['index']: t.get('page') for t in page_map['tables']}

    hits = []
    no_object = []

    def shapes_of(row):
        return [i for i in (row.get('shape_indices') or [])]

    for row in paras:
        if row.get('is_figure_caption'):
            found = None
            for j in range(row['index'] - 1, max(0, row['index'] - 1 - lookback), -1):
                cand = by_index.get(j)
                if cand is None:
                    continue
                if cand.get('in_table'):
                    continue
                if shapes_of(cand):
                    found = cand
                    break
            if found is None and shapes_of(row):
                found = row
            if found is None:
                no_object.append({'type': 'figure', 'caption_index': row['index'],
                                  'caption_page': row.get('page'),
                                  'caption_text': row.get('text_head')})
                continue
            for si in shapes_of(found):
                if shape_page.get(si) is not None and row.get('page') is not None \
                        and shape_page[si] != row['page']:
                    hits.append({
                        'type': 'figure_vs_caption', 'caption_index': row['index'],
                        'caption_page': row.get('page'), 'caption_text': row.get('text_head'),
                        'shape_index': si, 'shape_page': shape_page[si],
                        'shape_paragraph_index': found['index'],
                    })
        elif row.get('is_table_caption'):
            found = None
            for j in range(row['index'] + 1, row['index'] + 1 + lookback):
                cand = by_index.get(j)
                if cand is None:
                    continue
                if cand.get('in_table'):
                    found = cand
                    break
            if found is None:
                no_object.append({'type': 'table', 'caption_index': row['index'],
                                  'caption_page': row.get('page'),
                                  'caption_text': row.get('text_head')})
                continue
            ti = found.get('table_index')
            if ti is not None and table_page.get(ti) is not None and row.get('page') is not None \
                    and table_page[ti] != row['page']:
                hits.append({
                    'type': 'table_vs_caption', 'caption_index': row['index'],
                    'caption_page': row.get('page'), 'caption_text': row.get('text_head'),
                    'table_index': ti, 'table_page': table_page[ti],
                })
    return {'count': len(hits), 'hits': hits,
            'caption_without_adjacent_object': {'count': len(no_object), 'items': no_object},
            'note': '图题按“向前 lookback 段内最近含图段落”定位；表题按“向后最近进入表格的段落”'
                    '定位并据段落顺序映射表序号。浮动图片(doc.Shapes)不参与本检查。'}


def check_heading_orphan(page_map):
    """孤立标题：该标题是本页最后一个内容段落，且其后继内容的「起始页」在下一页。

    用 start_page（段落起始页）而非 page（段落末页）判定后继内容，避免把“跨页长段落”
    误判成孤立标题（其首行其实与标题同页）。"""
    paras = page_map['paragraphs']
    content = [p for p in paras if (not p.get('empty')) or p.get('has_inline_shape')]
    last_content_idx = {}
    for p in content:
        page = p.get('start_page') or p.get('page')
        if page is None:
            continue
        if page not in last_content_idx or p['index'] > last_content_idx[page]:
            last_content_idx[page] = p['index']
    by_index = {p['index']: p for p in paras}
    hits = []
    for p in paras:
        if not p.get('is_heading') or p.get('page') is None:
            continue
        if last_content_idx.get(p.get('start_page') or p.get('page')) != p['index']:
            continue
        nxt = by_index.get(p['index'] + 1)
        nxt_page = None if nxt is None else (nxt.get('start_page') or nxt.get('page'))
        if nxt is not None and nxt_page == p['page'] + 1:
            hits.append({
                'paragraph_index': p['index'], 'style': p.get('style'),
                'page': p['page'], 'text': p.get('text_head'),
                'next_paragraph_index': nxt['index'], 'next_page': nxt_page,
            })
    return {'count': len(hits), 'hits': hits,
            'note': '判定“孤立标题”＝该标题是本页最后一个内容段落，且其紧邻下一段落的'
                    '“起始页”在下一页（Word 的 Keep with next 未被破坏时会自然为 0）。'}


def check_object_overflow(page_map, tol_pt=2.0):
    box = page_map.get('content_box_pt') or {}
    box_w = float(box.get('width') or A4_CONTENT_W_PT)
    portrait_w = float(box.get('portrait_width') or A4_CONTENT_W_PT)
    box_h = float(box.get('height') or A4_CONTENT_H_PT)

    image_hits = []
    for s in page_map['inline_shapes']:
        w, h = s.get('width_pt'), s.get('height_pt')
        if w is None or h is None:
            continue
        if w > box_w + tol_pt or h > box_h + tol_pt:
            image_hits.append({
                'shape_index': s['index'], 'page': s.get('page'),
                'width_pt': w, 'height_pt': h, 'type': s.get('type_name'),
                'overflow_width_pt': round(max(0.0, w - box_w), 2),
                'overflow_height_pt': round(max(0.0, h - box_h), 2),
                'over_portrait_only': bool(w > portrait_w + tol_pt and w <= box_w + tol_pt),
            })
    table_hits = []
    for t in page_map['tables']:
        w = t.get('width_pt')
        if w is None:
            continue
        if w > box_w + tol_pt:
            table_hits.append({
                'table_index': t['index'], 'page': t.get('page'),
                'width_pt': w, 'rows': t.get('rows'), 'cols': t.get('cols'),
                'overflow_pt': round(w - box_w, 2),
                'first_row_text': t.get('first_row_text'),
                'heading_format': t.get('heading_format'),
                'over_portrait_only': bool(w > portrait_w + tol_pt and w <= box_w + tol_pt),
            })
    return {
        'content_box_pt': {'width': round(box_w, 2), 'portrait_width': round(portrait_w, 2),
                           'height': round(box_h, 2),
                           'a4_reference': {'width': round(A4_CONTENT_W_PT, 2),
                                            'height': round(A4_CONTENT_H_PT, 2)}},
        'image': {'count': len(image_hits), 'hits': image_hits},
        'table': {'count': len(table_hits), 'hits': table_hits},
        'note': '版心取 page-map 中各节内容框的最大值（兼容横向 section）；'
                'over_portrait_only=True 表示仅超过纵向版心，可能位于横向 section。',
    }


def cmd_inspect(args):
    page_map = load_json(args.page_map_json) if args.page_map_json else None
    pages = _load_png_pages(args.png_dir, prefix=args.prefix)
    cover = set(args.cover_pages or [])
    ink_threshold = args.ink_threshold

    page_reports = []
    blank_pages, near_blank_pages, edge_pages = [], [], []
    overlap_hits = []
    overlap_excluded_blocks = []
    if args.line_height_px:
        line_h = float(args.line_height_px)
        line_h_source = 'cli'
    else:
        line_h = estimate_line_height_px(pages, ink_threshold=ink_threshold,
                                         edge_frac=args.edge_frac)
        line_h_source = 'estimated_document_median'
    for num, path in pages:
        metrics, gray = page_pixel_metrics(path, ink_threshold=ink_threshold,
                                           edge_frac=args.edge_frac)
        metrics['page'] = num
        metrics['is_cover'] = num in cover
        ratio = metrics['ink_ratio']
        if num in cover:
            metrics['blank_status'] = 'cover_excluded'
        elif ratio < args.blank_ratio:
            metrics['blank_status'] = 'blank'
            blank_pages.append({'page': num, 'ink_ratio': ratio})
        elif ratio < args.near_blank_ratio:
            metrics['blank_status'] = 'near_blank'
            near_blank_pages.append({'page': num, 'ink_ratio': ratio})
        else:
            metrics['blank_status'] = 'ok'
        edge_hit = {k: v for k, v in metrics['edge_ink'].items() if v >= args.edge_min_ink}
        metrics['edge_hit_sides'] = sorted(edge_hit)
        if edge_hit and num not in cover:
            edge_pages.append({'page': num, 'sides': edge_hit,
                               'band_px': metrics['edge_band_px']})
        page_reports.append(metrics)
        hits, excluded = check_text_overlap(
            gray, num, ink_threshold=ink_threshold, edge_frac=args.edge_frac,
            band_factor=args.overlap_band_factor,
            max_band_factor=args.overlap_max_band_factor,
            min_subband_frac=args.overlap_min_subband_frac,
            overlap_min=args.overlap_min, line_height_px=line_h)
        overlap_hits.extend(hits)
        overlap_excluded_blocks.extend(excluded)

    result = {
        'generated_at': now_iso(),
        'png_dir': str(Path(args.png_dir).resolve()),
        'page_count': len(pages),
        'config': {
            'dpi_note': '像素阈值均按渲染 dpi 的像素数表达，与 render 的 --dpi 对应',
            'ink_threshold': ink_threshold,
            'blank_ratio': args.blank_ratio,
            'near_blank_ratio': args.near_blank_ratio,
            'edge_frac': args.edge_frac,
            'edge_min_ink': args.edge_min_ink,
            'overlap_min': args.overlap_min,
            'overlap_band_factor': args.overlap_band_factor,
            'overlap_max_band_factor': args.overlap_max_band_factor,
            'overlap_min_subband_frac': args.overlap_min_subband_frac,
            'line_height_px': round(line_h, 2) if line_h else None,
            'line_height_px_source': line_h_source,
            'cover_pages_excluded': sorted(cover),
        },
        'pages': page_reports,
        'checks': {
            'blank_pages': {'count': len(blank_pages), 'pages': blank_pages,
                            'threshold_ink_ratio': args.blank_ratio},
            'near_blank_pages': {'count': len(near_blank_pages), 'pages': near_blank_pages,
                                 'threshold_ink_ratio': args.near_blank_ratio},
            'edge_overflow': {'count': len(edge_pages), 'pages': edge_pages,
                              'edge_band_frac': args.edge_frac,
                              'min_ink_px': args.edge_min_ink,
                              'note': '检测页面最外 1% 边带内的非白墨迹（封面页已排除）。'},
            'text_overlap': {'count': len(overlap_hits), 'hits': overlap_hits,
                             'excluded_large_blocks': {
                                 'count': len(overlap_excluded_blocks),
                                 'items': overlap_excluded_blocks[:200],
                                 'definition': '高度 > %s×行高的墨迹带（按整幅图片/公式块排除）'
                                               % args.overlap_max_band_factor,
                             },
                             'limitation': TEXT_OVERLAP_LIMITATION},
        },
    }

    if page_map is not None:
        overflow = check_object_overflow(page_map)
        result['checks']['image_overflow'] = overflow['image']
        result['checks']['table_overflow'] = overflow['table']
        result['checks']['overflow_content_box'] = overflow['content_box_pt']
        result['checks']['caption_separation'] = check_caption_separation(page_map)
        result['checks']['heading_orphan'] = check_heading_orphan(page_map)
        result['checks']['caption_numbering'] = check_caption_numbering(page_map)
        result['checks']['appendix_residue'] = check_appendix_residue(page_map)
        expected = (page_map.get('doc_stats') or {}).get('pages')
        result['page_count_match'] = {
            'word_pages': expected, 'rendered_pages': len(pages),
            'match': (expected == len(pages)),
        }
        result['page_count_gate'] = {
            'total_page_count': expected,
            'rendered_page_count': len(pages),
            'max_allowed': MAX_TOTAL_PAGE_COUNT,
            'status': ('PASS' if (expected is not None and expected < MAX_TOTAL_PAGE_COUNT)
                       else 'FAIL'),
            'note': '总页数 >= %d 即 FAIL：可能仍有全量表（附录附表）被错误插入。'
                    % MAX_TOTAL_PAGE_COUNT,
        }
    else:
        result['checks']['skipped'] = ['image_overflow', 'table_overflow',
                                       'caption_separation', 'heading_orphan',
                                       'caption_numbering', 'appendix_residue']

    out = dump_json(args.out_json, result)
    checks = result['checks']
    log('[inspect] 页面=%d（空白 %d / 近空白 %d / 边缘越界 %d / 文字重叠候选 %d，另有 %d 个高墨迹块按图片/公式排除）' % (
        len(pages), checks['blank_pages']['count'], checks['near_blank_pages']['count'],
        checks['edge_overflow']['count'], checks['text_overlap']['count'],
        (checks['text_overlap'].get('excluded_large_blocks') or {}).get('count', 0)))
    for key in ('image_overflow', 'table_overflow', 'caption_separation',
                'heading_orphan', 'caption_numbering'):
        if key in checks:
            log('[inspect] %s 命中=%s' % (key, checks[key]['count']))
    ap = checks.get('appendix_residue')
    if ap:
        log('[inspect] appendix_residue 附图=%d 附表=%d 标题=%d 目录=%d 正文引用=%d'
            % (ap['figure_caption_count'], ap['table_caption_count'],
               ap['heading_count'], ap['toc_count'], ap['reference_count']))
    pg = result.get('page_count_gate')
    if pg:
        log('[inspect] TOTAL_PAGE_COUNT=%s（门禁 <%d → %s）'
            % (pg['total_page_count'], pg['max_allowed'], pg['status']))
    log('[inspect] json=%s' % out)
    return result


# ---------------------------------------------------------------- 5) textscan

def extract_pdf_text(pdf_path):
    pdf_path = require_file(pdf_path, 'PDF')
    try:
        from PyPDF2 import PdfReader
    except Exception:
        try:
            from pypdf import PdfReader          # PyPDF2 的官方继任包，API 兼容
        except Exception as exc:  # noqa: BLE001
            raise QaError('无法导入 PyPDF2 / pypdf：%r' % (exc,))
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ''
        except Exception:
            text = ''
        pages.append({'page': i, 'text': text})
    return pages


def _locate(pages, needle, use_regex=False, pattern=None):
    hits = []
    for pg in pages:
        if use_regex:
            n = len(re.findall(pattern, pg['text']))
        else:
            n = pg['text'].count(needle)
        if n:
            hits.append({'page': pg['page'], 'count': n})
    return hits


def scan_text(pages, forbidden, required, low_specificity=LOW_SPECIFICITY):
    total_chars = sum(len(p['text']) for p in pages)
    cjk_pages = [p['page'] for p in pages if re.search(r'[\u4e00-\u9fff]', p['text'])]
    empty_pages = [p['page'] for p in pages if not p['text'].strip()]

    forbidden_hits = []
    for s in forbidden:
        hits = _locate(pages, s)
        forbidden_hits.append({'string': s, 'count': sum(h['count'] for h in hits), 'pages': hits})

    required_hits = []
    for label, variants in required.items():
        variant_rows = []
        present = False
        for v in variants:
            hits = _locate(pages, v)
            cnt = sum(h['count'] for h in hits)
            present = present or cnt > 0
            row = {'variant': v, 'count': cnt, 'pages': hits}
            if v in low_specificity:
                pattern = r'(?<![0-9.,])%s(?![0-9.,])' % re.escape(v)
                row['count_delimited'] = sum(
                    len(re.findall(pattern, p['text'])) for p in pages)
            variant_rows.append(row)
        required_hits.append({'value': label, 'present': present, 'variants': variant_rows})

    informational = {}
    for s in ('Gate', 'PASS', 'Stage23', 'Stage24', 'gate', 'pass'):
        hits = _locate(pages, s)
        informational[s] = {'count': sum(h['count'] for h in hits), 'pages': hits}

    return {
        'generated_at': now_iso(),
        'page_count': len(pages),
        'extraction_quality': {
            'total_chars': total_chars,
            'chars_per_page_avg': round(total_chars / max(1, len(pages)), 1),
            'pages_with_cjk': len(cjk_pages),
            'pages_without_cjk': [p['page'] for p in pages if p['page'] not in set(cjk_pages)],
            'empty_pages': empty_pages,
            'note': 'PyPDF2 对中文字体（含子集化字体）抽取可能有噪声或丢字，'
                    'total_chars 与 pages_with_cjk 用于判断抽取是否有效；'
                    '未命中不等于正文缺失，需结合 Word 域/文本核对。',
        },
        'forbidden': {
            'count_violations': sum(1 for r in forbidden_hits if r['count'] > 0),
            'results': forbidden_hits,
            'expected_all_zero': True,
        },
        'required': {
            'count_missing': sum(1 for r in required_hits if not r['present']),
            'results': required_hits,
            'note': '低特异性数字（46/144/469）同时给出“定界计数”'
                    '（排除紧邻数字/逗号/小数点的子串）。',
        },
        'informational_case_variants': informational,
    }


def cmd_textscan(args):
    pdf = require_file(args.pdf, 'PDF')
    pages = extract_pdf_text(pdf)
    result = scan_text(pages, FORBIDDEN_STRINGS, REQUIRED_VALUES)
    result['pdf'] = str(pdf)
    out = dump_json(args.out_json, result)
    fz = result['forbidden']
    rq = result['required']
    log('[textscan] pdf=%s 页数=%d 抽取总字符=%d 含中文页数=%d' % (
        pdf, result['page_count'], result['extraction_quality']['total_chars'],
        result['extraction_quality']['pages_with_cjk']))
    log('[textscan] 必须为 0 的字符串：违规 %d / %d' % (fz['count_violations'], len(fz['results'])))
    for r in fz['results']:
        if r['count']:
            log('    ! %r × %d @ %s' % (r['string'], r['count'],
                                        [h['page'] for h in r['pages']][:12]))
    log('[textscan] 必须存在的关键值：缺失 %d / %d' % (rq['count_missing'], len(rq['results'])))
    for r in rq['results']:
        if not r['present']:
            log('    ! 缺失 %s（候选写法 %s）' % (r['value'], [v['variant'] for v in r['variants']]))
    log('[textscan] json=%s' % out)
    return result


# ---------------------------------------------------------------- 6) report

def _md_table(rows, header):
    lines = ['| ' + ' | '.join(header) + ' |',
             '| ' + ' | '.join(['---'] * len(header)) + ' |']
    for r in rows:
        lines.append('| ' + ' | '.join(str(c) for c in r) + ' |')
    return '\n'.join(lines)


def _gate(name, value, expected, status, note=''):
    return {'gate': name, 'value': value, 'expected': expected,
            'status': status, 'note': note}


def evaluate_gates(doc_stats, inspect_result, textscan_result):
    """Stage24 门禁：附录全部移除 + 总页数异常检查（附录引用清零）。"""
    checks = (inspect_result or {}).get('checks') or {}
    ap = checks.get('appendix_residue') or {}
    pages = (doc_stats or {}).get('pages')

    def eq_gate(name, value, expected):
        ok = (value == expected)
        return _gate(name, value, expected, 'PASS' if ok else 'FAIL',
                     '' if ok else '应为 %s，实际 %s' % (expected, value))

    gates = [
        eq_gate('APPENDIX_FIGURE_COUNT', ap.get('figure_caption_count'), APPENDIX_FIGURE_COUNT),
        eq_gate('APPENDIX_TABLE_COUNT', ap.get('table_caption_count'), APPENDIX_TABLE_COUNT),
        eq_gate('APPENDIX_HEADING_COUNT', ap.get('heading_count'), APPENDIX_HEADING_COUNT),
        eq_gate('APPENDIX_TOC_COUNT', ap.get('toc_count'), 0),
        eq_gate('APPENDIX_REFERENCE_COUNT', ap.get('reference_count'),
                APPENDIX_REFERENCE_COUNT),
    ]

    appendix_text_hits = []
    if textscan_result is not None:
        for r in ((textscan_result.get('forbidden') or {}).get('results') or []):
            if r['string'] in APPENDIX_TEXT_FORBIDDEN and r['count']:
                appendix_text_hits.append({'string': r['string'], 'count': r['count'],
                                           'pages': [h['page'] for h in r['pages']]})
        gates.append(eq_gate('APPENDIX_TEXT_RESIDUE',
                             sum(h['count'] for h in appendix_text_hits), 0))

    page_ok = pages is not None and pages < MAX_TOTAL_PAGE_COUNT
    gates.append(_gate(
        'TOTAL_PAGE_COUNT', pages, '< %d' % MAX_TOTAL_PAGE_COUNT,
        'PASS' if page_ok else 'FAIL',
        '' if page_ok else '总页数 >= %d：可能仍有全量表（附录附表）被错误插入'
                          % MAX_TOTAL_PAGE_COUNT))

    failed = [g['gate'] for g in gates if g['status'] == 'FAIL']
    return {'gates': gates, 'count_failed': len(failed), 'failed': failed,
            'appendix_text_hits': appendix_text_hits,
            'note': '门禁失败即视为交付不通过（Gate 字样不会写入 docx，仅出现在 QA 产物中）。'}


def cmd_report(args):
    work = Path(args.work_dir)
    paths = {
        'export': Path(args.export_info or (work / 'export_info.json')),
        'page_map': Path(args.page_map_json or (work / 'page_map.json')),
        'inspect': Path(args.inspect_json or (work / 'inspect.json')),
        'textscan': Path(args.textscan_json or (work / 'textscan.json')),
    }
    loaded = {}
    for key, path in paths.items():
        loaded[key] = load_json(path) if path.is_file() else None

    page_map, inspect, textscan, export = (loaded['page_map'], loaded['inspect'],
                                          loaded['textscan'], loaded['export'])
    if not any(loaded.values()):
        raise QaError('report 找不到任何输入 JSON（期望在 %s 下）' % work)

    docx = (export or {}).get('working_docx') or (page_map or {}).get('docx')
    doc_stats = (page_map or {}).get('doc_stats') or (export or {}).get('doc_stats') or {}

    summary = {
        'generated_at': now_iso(),
        'source_docx': (export or {}).get('source_docx'),
        'working_docx': docx,
        'pdf': (export or {}).get('pdf') or (textscan or {}).get('pdf'),
        'png_dir': (inspect or {}).get('png_dir'),
        'doc_stats': doc_stats,
        'page_count': {
            'word_pages': doc_stats.get('pages'),
            'rendered_pages': (inspect or {}).get('page_count'),
            'page_map_match': (inspect or {}).get('page_count_match'),
        },
        'field_errors': (export or {}).get('field_errors'),
        'issue_counts': {},
        'issues': {},
        'textscan': None,
    }

    if inspect:
        checks = inspect.get('checks') or {}
        for key in ('blank_pages', 'near_blank_pages', 'edge_overflow', 'text_overlap',
                    'image_overflow', 'table_overflow', 'heading_orphan',
                    'caption_separation', 'caption_numbering'):
            if key in checks:
                summary['issue_counts'][key] = checks[key].get('count', 0)
        for key in ('blank_pages', 'near_blank_pages', 'edge_overflow', 'text_overlap',
                    'image_overflow', 'table_overflow', 'caption_separation',
                    'heading_orphan', 'caption_numbering'):
            if key in checks:
                summary['issues'][key] = checks[key]
        summary['overlap_limitation'] = (checks.get('text_overlap') or {}).get('limitation')
        overlap = checks.get('text_overlap') or {}
        if overlap.get('hits') is not None:
            summary['issue_counts']['text_overlap_dense_merged_block'] = sum(
                1 for h in overlap['hits'] if h.get('sub_type') == 'dense_merged_block')
            summary['issue_counts']['text_overlap_waist_merged_pair'] = sum(
                1 for h in overlap['hits'] if h.get('sub_type') == 'waist_merged_pair')
            summary['issue_counts']['text_overlap_excluded_large_blocks'] = \
                (overlap.get('excluded_large_blocks') or {}).get('count', 0)

    if textscan:
        fz = textscan.get('forbidden') or {}
        rq = textscan.get('required') or {}
        summary['textscan'] = {
            'extraction_quality': textscan.get('extraction_quality'),
            'forbidden_violations': [
                {'string': r['string'], 'count': r['count'],
                 'pages': [h['page'] for h in r['pages']]}
                for r in (fz.get('results') or []) if r['count'] > 0],
            'required_missing': [r['value'] for r in (rq.get('results') or []) if not r['present']],
            'required_present': [
                {'value': r['value'], 'count': sum(v['count'] for v in r['variants']),
                 'pages': sorted({h['page'] for v in r['variants'] for h in v['pages']})}
                for r in (rq.get('results') or []) if r['present']],
        }
        summary['issue_counts']['forbidden_string_violations'] = len(summary['textscan']['forbidden_violations'])
        summary['issue_counts']['required_values_missing'] = len(summary['textscan']['required_missing'])

    gates = evaluate_gates(doc_stats, inspect, textscan)
    summary['gates'] = gates
    summary['issue_counts']['gate_failures'] = gates['count_failed']
    if inspect:
        ap = (inspect.get('checks') or {}).get('appendix_residue')
        if ap:
            summary['issue_counts']['appendix_residue_figure_captions'] = ap['figure_caption_count']
            summary['issue_counts']['appendix_residue_table_captions'] = ap['table_caption_count']
            summary['issue_counts']['appendix_residue_headings'] = ap['heading_count']
            summary['issue_counts']['appendix_residue_toc'] = ap['toc_count']
            summary['issue_counts']['appendix_reference_count'] = ap['reference_count']
        pg = inspect.get('page_count_gate')
        if pg:
            summary['page_count_gate'] = pg

    json_path = dump_json(args.report_json or (work / 'qa_report.json'), summary)

    # ---- Markdown 摘要
    lines = ['# Stage24 论文 QA 摘要', '']
    lines.append('- 生成时间：%s' % summary['generated_at'])
    lines.append('- 源 docx：`%s`' % summary['source_docx'])
    lines.append('- 工作 docx：`%s`' % summary['working_docx'])
    lines.append('- QA PDF：`%s`' % summary['pdf'])
    lines.append('- 渲染 PNG 目录：`%s`' % summary['png_dir'])
    lines.append('- Word 页数：%s；渲染页数：%s' % (
        doc_stats.get('pages'), (inspect or {}).get('page_count')))
    lines.append('')

    lines.append('## 1. 文档级统计')
    lines.append('')
    lines.append(_md_table(
        [[k, v] for k, v in sorted(doc_stats.items())], ['项目', '值']))
    lines.append('')
    if summary.get('field_errors') is not None:
        lines.append('- 域错误（Error!/未找到）数量：%d' % len(summary['field_errors'] or []))
    lines.append('')

    lines.append('## 2. 检查命中数')
    lines.append('')
    rows = [[k, v] for k, v in summary['issue_counts'].items()]
    lines.append(_md_table(rows or [['（无）', '']], ['检查项', '命中数']))
    lines.append('')

    lines.append('## 3. 命中清单')
    lines.append('')

    def _dump(key, title, formatter):
        node = (summary['issues'] or {}).get(key)
        if node is None:
            return
        lines.append('### %s（命中 %s）' % (title, node.get('count')))
        lines.append('')
        items = node.get('hits') or node.get('pages') or []
        if not items:
            lines.append('- 无')
        else:
            for it in items[:200]:
                lines.append('- ' + formatter(it))
        note = node.get('note') or node.get('limitation')
        if note:
            lines.append('')
            lines.append('> 说明：%s' % note)
        lines.append('')

    _dump('blank_pages', '空白页', lambda it: 'p%s 墨迹占比 %.4f%%' % (it['page'], it['ink_ratio'] * 100))
    _dump('near_blank_pages', '近空白页', lambda it: 'p%s 墨迹占比 %.4f%%' % (it['page'], it['ink_ratio'] * 100))
    _dump('edge_overflow', '页面内容越界（边缘 1% 带内有墨迹）',
          lambda it: 'p%s 命中边：%s（%s）' % (it['page'], ','.join(it['sides']), it['sides']))
    _dump('text_overlap', '文字重叠候选（墨迹粘连）',
          lambda it: 'p%s [%s] 行带 y=%s~%s（高 %spx = %s×行高%s，腰 y=%s，水平重叠 %s）' % (
              it['page'], it.get('sub_type'),
              it['band_top_px'], it['band_bottom_px'], it['band_height_px'],
              it.get('band_height_over_line'), it.get('line_height_px'),
              it.get('waist_row_px', '-'),
              ('%.0f%%' % (it['overlap_ratio'] * 100)) if it.get('overlap_ratio') is not None else 'n/a'))
    excluded = ((summary['issues'].get('text_overlap') or {})
                .get('excluded_large_blocks') or {})
    if excluded.get('count'):
        items = excluded.get('items') or []
        lines.append('- 另有 %d 个高墨迹块按整幅图片/公式块排除（%s），明细见 '
                     '`inspect.json` → `checks.text_overlap.excluded_large_blocks`：' % (
                         excluded['count'], excluded.get('definition', '超出高度上限')))
        lines.append('  - %s' % '；'.join(
            'p%s y=%s~%s 高%spx(%.2f×行高) 宽%spx' % (
                i['page'], i['band_top_px'], i['band_bottom_px'], i['band_height_px'],
                i['band_height_over_line'], i.get('band_width_px') or 0)
            for i in items[:30]))
        lines.append('')
    _dump('image_overflow', '图片越界',
          lambda it: 'InlineShape#%s p%s %s×%s pt（超出版心 %.1f×%.1f pt）' % (
              it['shape_index'], it['page'], it['width_pt'], it['height_pt'],
              it['overflow_width_pt'], it['overflow_height_pt']))
    _dump('table_overflow', '表格越界',
          lambda it: 'Table#%s p%s 宽 %s pt（超出 %.1f pt，%s 行×%s 列）' % (
              it['table_index'], it['page'], it['width_pt'], it['overflow_pt'],
              it['rows'], it['cols']))
    cs = (summary['issues'] or {}).get('caption_separation')
    if cs:
        lines.append('### 图题/表题与对象分离（命中 %s）' % cs.get('count'))
        lines.append('')
        if cs.get('hits'):
            for it in cs['hits']:
                if it['type'] == 'figure_vs_caption':
                    lines.append('- 图：题注（段 %s，p%s「%s」）与图片（InlineShape#%s，p%s）不同页' % (
                        it['caption_index'], it['caption_page'], it['caption_text'],
                        it['shape_index'], it['shape_page']))
                else:
                    lines.append('- 表：题注（段 %s，p%s「%s」）与表格（Table#%s，p%s）不同页' % (
                        it['caption_index'], it['caption_page'], it['caption_text'],
                        it['table_index'], it['table_page']))
        else:
            lines.append('- 无')
        miss = cs.get('caption_without_adjacent_object') or {}
        lines.append('')
        lines.append('- 未找到相邻对象的题注：%s 处' % miss.get('count'))
        for it in (miss.get('items') or [])[:50]:
            lines.append('  - %s 题注（段 %s，p%s「%s」）' % (
                it['type'], it['caption_index'], it['caption_page'], it['caption_text']))
        if cs.get('note'):
            lines.append('')
            lines.append('> 说明：%s' % cs['note'])
        lines.append('')

    _dump('heading_orphan', '孤立标题（页底标题，正文在下一页）',
          lambda it: 'p%s %s「%s」（段 %s，下一段 %s 在 p%s）' % (
              it['page'], it['style'], it['text'], it['paragraph_index'],
              it['next_paragraph_index'], it['next_page']))

    cn = (summary['issues'] or {}).get('caption_numbering')
    if cn:
        lines.append('### 题注序号连续性（问题 %s 项，题注总数 %s）' % (
            cn.get('count'), cn.get('caption_count')))
        lines.append('')
        if cn.get('series'):
            rows = []
            for kind, groups in cn['series'].items():
                for group, info in groups.items():
                    rows.append([kind, group, ','.join(str(n) for n in info['order']),
                                 info['count'], info['missing'] or '-', info['duplicates'] or '-',
                                 info['out_of_order'] or '-'])
            lines.append(_md_table(rows, ['类型', '分组', '出现序号', '个数', '缺号', '重号', '乱序']))
            lines.append('')
        if cn.get('issues'):
            for it in cn['issues']:
                lines.append('- %s' % json.dumps(it, ensure_ascii=False)[:300])
        else:
            lines.append('- 无')
        lines.append('')

    ap = (summary['issues'] or {}).get('appendix_residue')
    if ap:
        lines.append('### 附录残留（合计 %s：附图题注 %s / 附表题注 %s / 标题 %s / 目录 %s / 正文引用 %s）'
                     % (ap.get('count'), ap.get('figure_caption_count'),
                        ap.get('table_caption_count'), ap.get('heading_count'),
                        ap.get('toc_count'), ap.get('reference_count')))
        lines.append('')
        for key, title in (('figure_caption_hits', '附图题注'),
                           ('table_caption_hits', '附表题注'),
                           ('heading_hits', '标题（导航大纲）'),
                           ('toc_hits', '目录（TOC）'),
                           ('reference_hits', '正文附录引用')):
            items = ap.get(key) or []
            lines.append('- %s：%d 处%s' % (
                title, len(items),
                '' if not items else ' → ' + '；'.join(
                    'p%s 段%s「%s」' % (i['page'], i['paragraph_index'], i['text'])
                    for i in items[:20])))
        if ap.get('note'):
            lines.append('')
            lines.append('> 说明：%s' % ap['note'])
        lines.append('')

    if summary['textscan']:
        ts = summary['textscan']
        lines.append('## 4. PDF 文本扫描')
        lines.append('')
        eq = ts['extraction_quality'] or {}
        lines.append('- 抽取总字符数：%s；平均每页 %s；含中文页数：%s/%s' % (
            eq.get('total_chars'), eq.get('chars_per_page_avg'),
            eq.get('pages_with_cjk'), (textscan or {}).get('page_count')))
        lines.append('')
        lines.append('### 4.1 必须为 0 的字符串（违规 %d）' % len(ts['forbidden_violations']))
        lines.append('')
        if ts['forbidden_violations']:
            lines.append(_md_table(
                [[r['string'], r['count'], ','.join(str(p) for p in r['pages'][:20])]
                 for r in ts['forbidden_violations']], ['字符串', '命中次数', '页码']))
        else:
            lines.append('- 全部为 0')
        lines.append('')
        lines.append('### 4.2 必须存在的关键值（缺失 %d）' % len(ts['required_missing']))
        lines.append('')
        if ts['required_missing']:
            lines.append('- 缺失：%s' % '、'.join(ts['required_missing']))
        lines.append('')
        lines.append(_md_table(
            [[r['value'], r['count'], ','.join(str(p) for p in r['pages'][:20])]
             for r in ts['required_present']], ['关键值', '命中次数', '页码']))
        delimited = [
            {'value': r['value'], 'count': sum(v['count'] for v in r['variants']),
             'delimited': sum(v.get('count_delimited', 0) for v in r['variants'])}
            for r in (textscan.get('required') or {}).get('results', [])
            if any('count_delimited' in v for v in r['variants'])
        ]
        if delimited:
            lines.append('')
            lines.append('- 低特异性数字的“定界出现次数”（排除紧邻数字/逗号/小数点的子串）：%s' % '；'.join(
                '%s 原始 %d / 定界 %d' % (d['value'], d['count'], d['delimited']) for d in delimited))
        lines.append('')

    lines.append('## 5. 门禁（Gate）')
    lines.append('')
    g = summary.get('gates') or {}
    lines.append(_md_table(
        [[r['gate'], r['value'], r['expected'], r['status'], r['note'] or '-']
         for r in (g.get('gates') or [])],
        ['门禁项', '实测值', '期望', '结果', '说明']))
    lines.append('')
    if g.get('count_failed'):
        lines.append('- **FAIL %d 项：%s**' % (g['count_failed'], '、'.join(g['failed'])))
    else:
        lines.append('- 全部门禁 PASS')
    if g.get('appendix_text_hits'):
        lines.append('- PDF 可见文本中的附录残留串：%s' % json.dumps(
            g['appendix_text_hits'], ensure_ascii=False))
    if g.get('note'):
        lines.append('')
        lines.append('> 说明：%s' % g['note'])
    lines.append('')

    lines.append('## 6. 已知局限')
    lines.append('')
    lines.append('- %s' % TEXT_OVERLAP_LIMITATION)
    lines.append('- PyPDF2 中文抽取可能丢字/乱序，文本扫描以“整串命中”为准，'
                 '页码为抽取页定位，可能有 ±1 页偏差（跨页表格/图片文字）。')
    lines.append('- 边缘越界检查仅判断页面最外 1% 边带是否有墨迹，'
                 '页眉页脚若被设置在极小边距内也会命中。')
    lines.append('- 图片截断仅由 page-map 的尺寸与版心比对判定，'
                 '无法判定图片内部是否被裁切。')
    lines.append('')

    md_path = ensure_under_qa_dir(args.report_md or (work / 'qa_summary.md'))
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text('\n'.join(lines), encoding='utf-8')

    log('=' * 72)
    log('[report] docx=%s' % summary['working_docx'])
    log('[report] Word 页数=%s 渲染页数=%s' % (
        doc_stats.get('pages'), (inspect or {}).get('page_count')))
    log('[report] 命中数：%s' % json.dumps(summary['issue_counts'], ensure_ascii=False))
    log('[report] 门禁：%s（FAIL %d 项%s）' % (
        'PASS' if not gates['count_failed'] else 'FAIL',
        gates['count_failed'], ('：%s' % '、'.join(gates['failed'])) if gates['count_failed'] else ''))
    if summary['textscan']:
        log('[report] textscan：违规串 %d，关键值缺失 %d' % (
            len(summary['textscan']['forbidden_violations']),
            len(summary['textscan']['required_missing'])))
    log('[report] json=%s' % json_path)
    log('[report] md=%s' % md_path)
    log('=' * 72)
    return summary


# ---------------------------------------------------------------- all

def cmd_all(args):
    work = ensure_under_qa_dir(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    docx = require_file(args.docx, 'docx')
    working_docx = docx
    if args.copy_docx and not docx.stem.endswith('_qa_copy'):
        working_docx = ensure_under_qa_dir(work / (docx.stem + '_qa_copy.docx'))
        shutil.copy2(str(docx), str(working_docx))
        log('[all] 已复制副本（源文件保持不动）：%s' % working_docx)

    pdf = Path(args.pdf) if args.pdf else derive_qa_pdf_path(work, docx)
    pdf = ensure_under_qa_dir(pdf)
    rendered = Path(args.out_dir) if args.out_dir else (work / 'rendered')

    # export 与 page-map 共用同一个 Word 会话（Word 为单实例 COM 服务器，
    # 前一个 Quit() 未结束时立刻新建会话会报 “RPC 服务器不可用”）
    def _word_stage(word):
        export_info = None
        page_map = None
        if not args.skip_export:
            export_info = do_export(word, working_docx, pdf, source_docx=docx)
            log('[all] export 完成：pdf=%s pages=%s' % (
                pdf, (export_info.get('doc_stats') or {}).get('pages')))
        if not args.reuse_page_map:
            page_map = do_page_map(word, working_docx, text_head_len=args.text_head_len)
            log('[all] page-map 完成：段落=%d 图=%d 表=%d' % (
                len(page_map['paragraphs']), len(page_map['inline_shapes']),
                len(page_map['tables'])))
        return export_info, page_map

    if args.skip_export and args.reuse_page_map:
        export_info, page_map = None, None
    else:
        export_info, page_map = run_in_word_session(_word_stage)

    if args.skip_export:
        require_file(pdf, 'PDF')
        log('[all] 跳过 export，复用 %s' % pdf)
    if export_info is not None:
        export_info['generated_at'] = now_iso()
        dump_json(work / 'export_info.json', export_info)
        log('[open-update-export] 域更新：%s，域错误 %d 处' % (
            {k: v for k, v in export_info['field_update'].items() if k != 'errors'},
            export_info['field_error_count']))
    elif not (work / 'export_info.json').is_file():
        log('[all] 未产生 export_info.json（--skip-export）')
    if page_map is not None:
        dump_json(work / 'page_map.json', page_map)
        write_page_map_csv(work / 'page_map_paragraphs.csv', page_map)
    elif (work / 'page_map.json').is_file():
        log('[all] 复用已有 page-map')

    cmd_render(argparse.Namespace(pdf=str(pdf), out_dir=str(rendered),
                                  dpi=args.dpi, pdftoppm=args.pdftoppm))

    page_map_json = work / 'page_map.json'
    cmd_inspect(argparse.Namespace(
        page_map_json=str(page_map_json) if page_map_json.is_file() else None,
        png_dir=str(rendered),
        out_json=str(work / 'inspect.json'), prefix='page',
        cover_pages=args.cover_pages, ink_threshold=args.ink_threshold,
        edge_frac=args.edge_frac, edge_min_ink=args.edge_min_ink,
        blank_ratio=args.blank_ratio, near_blank_ratio=args.near_blank_ratio,
        overlap_min=args.overlap_min,
        overlap_band_factor=args.overlap_band_factor,
        overlap_max_band_factor=args.overlap_max_band_factor,
        overlap_min_subband_frac=args.overlap_min_subband_frac,
        line_height_px=args.line_height_px))

    cmd_textscan(argparse.Namespace(pdf=str(pdf), out_json=str(work / 'textscan.json')))

    return cmd_report(argparse.Namespace(
        work_dir=str(work), export_info=str(work / 'export_info.json'),
        page_map_json=str(work / 'page_map.json'),
        inspect_json=str(work / 'inspect.json'),
        textscan_json=str(work / 'textscan.json'),
        report_json=str(work / 'qa_report.json'),
        report_md=str(work / 'qa_summary.md')))


# ---------------------------------------------------------------- CLI

def build_parser():
    parser = argparse.ArgumentParser(
        prog='24b_qa_render.py',
        description='Stage24 论文 Word → PDF → 逐页渲染 → 程序化视觉 QA 工具链',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='示例：\n'
               '  python scripts/24b_qa_render.py all --docx "outputs/paper/课程设计论文_终稿_Stage21.docx"'
               ' --work-dir "outputs/paper/qa/stage21_verify" --copy-docx\n')
    sub = parser.add_subparsers(dest='command', required=True)

    def add_work(p, default=QA_ROOT):
        p.add_argument('--work-dir', default=str(default),
                       help='QA 产物目录（必须在 outputs/paper/qa 下，默认 outputs/paper/qa）')

    def add_common_inspect(p):
        p.add_argument('--ink-threshold', type=int, default=250,
                       help='灰度值 < 该值视作墨迹（默认 250）')
        p.add_argument('--edge-frac', type=float, default=0.01,
                       help='边缘检查带宽比例（默认 0.01＝1%%）')
        p.add_argument('--edge-min-ink', type=int, default=20,
                       help='边缘带内像素数达到该值才报告越界（默认 20）')
        p.add_argument('--blank-ratio', type=float, default=0.002,
                       help='墨迹占比 < 该值视为空白页（默认 0.2%%）')
        p.add_argument('--near-blank-ratio', type=float, default=0.01,
                       help='墨迹占比 < 该值视为近空白页（默认 1.0%%）')
        p.add_argument('--overlap-min', type=float, default=0.30,
                       help='文字重叠检查的水平重叠阈值（默认 0.30）')
        p.add_argument('--overlap-band-factor', type=float, default=1.6,
                       help='墨迹带高于中位行高该倍数才视为“两行粘连”候选（默认 1.6）')
        p.add_argument('--overlap-max-band-factor', type=float, default=3.5,
                       help='高于该倍数（默认 3.5）的带按图片/公式块排除')
        p.add_argument('--overlap-min-subband-frac', type=float, default=0.6,
                       help='腰上下两半各自高度 ≥ 中位行高该比例（默认 0.6）')
        p.add_argument('--line-height-px', type=float, default=None,
                       help='单行墨迹高度先验（像素）；缺省时按跨页中位行高自动估计')
        p.add_argument('--cover-pages', type=int, nargs='*', default=[1],
                       help='排除空白判定的封面页（默认 1）')

    p = sub.add_parser('open-update-export',
                       help='Word COM 打开 docx → 更新全部域 → 保存回 docx → 导出 PDF')
    p.add_argument('--docx', required=True)
    p.add_argument('--pdf', default=None, help='PDF 输出路径（默认 <work-dir>/<docx_stem>_QA.pdf）')
    p.add_argument('--copy-docx', action='store_true',
                   help='先把 docx 复制到 work-dir 再加域改名操作（保护源文件）')
    add_work(p)
    p.set_defaults(func=cmd_open_update_export)

    p = sub.add_parser('page-map', help='输出逐段落→页码映射（CSV + JSON）')
    p.add_argument('--docx', required=True)
    p.add_argument('--out-json', default=None)
    p.add_argument('--out-csv', default=None)
    p.add_argument('--text-head-len', type=int, default=60, help='段落文本截断长度（默认 60 字）')
    add_work(p)
    p.set_defaults(func=cmd_page_map)

    p = sub.add_parser('render', help='pdftoppm 逐页渲染 PNG（page-01.png …）')
    p.add_argument('--pdf', required=True)
    p.add_argument('--out-dir', default=None)
    p.add_argument('--dpi', type=int, default=110)
    p.add_argument('--pdftoppm', default=str(DEFAULT_PDFTOPPM))
    add_work(p)
    p.set_defaults(func=cmd_render)

    p = sub.add_parser('inspect', help='对渲染 PNG + page-map 做程序化视觉检查')
    p.add_argument('--page-map-json', default=None)
    p.add_argument('--png-dir', default=None)
    p.add_argument('--out-json', default=None)
    p.add_argument('--prefix', default='page')
    add_common_inspect(p)
    add_work(p)
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser('textscan', help='PyPDF2 抽取每页文本并扫描禁用串/关键值')
    p.add_argument('--pdf', required=True)
    p.add_argument('--out-json', default=None)
    add_work(p)
    p.set_defaults(func=cmd_textscan)

    p = sub.add_parser('report', help='汇总为 qa_report.json + qa_summary.md')
    p.add_argument('--export-info', default=None)
    p.add_argument('--page-map-json', default=None)
    p.add_argument('--inspect-json', default=None)
    p.add_argument('--textscan-json', default=None)
    p.add_argument('--report-json', default=None)
    p.add_argument('--report-md', default=None)
    add_work(p)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser('all', help='串联 export → page-map → render → inspect → textscan → report')
    p.add_argument('--docx', required=True)
    p.add_argument('--pdf', default=None)
    p.add_argument('--out-dir', default=None, help='PNG 输出目录（默认 <work-dir>/rendered）')
    p.add_argument('--dpi', type=int, default=110)
    p.add_argument('--pdftoppm', default=str(DEFAULT_PDFTOPPM))
    p.add_argument('--copy-docx', action='store_true', help='先在 work-dir 内做 docx 副本再操作')
    p.add_argument('--skip-export', action='store_true', help='跳过 Word 导出，复用既有 PDF')
    p.add_argument('--reuse-page-map', action='store_true', help='复用既有 page_map.json')
    p.add_argument('--text-head-len', type=int, default=60)
    add_common_inspect(p)
    add_work(p)
    p.set_defaults(func=cmd_all)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # 各子命令的默认输出路径（统一落在 work-dir 下）
    work = Path(getattr(args, 'work_dir', QA_ROOT))
    if getattr(args, 'out_json', None) is None:
        defaults = {
            'page-map': work / 'page_map.json',
            'inspect': work / 'inspect.json',
            'textscan': work / 'textscan.json',
        }
        if args.command in defaults:
            args.out_json = str(defaults[args.command])
    if args.command == 'page-map' and getattr(args, 'out_csv', None) is None:
        args.out_csv = str(work / 'page_map_paragraphs.csv')
    if args.command == 'inspect' and getattr(args, 'png_dir', None) is None:
        args.png_dir = str(work / 'rendered')
    if args.command == 'inspect' and getattr(args, 'page_map_json', None) is None:
        candidate = work / 'page_map.json'
        args.page_map_json = str(candidate) if candidate.is_file() else None
    if args.command == 'render' and getattr(args, 'out_dir', None) is None:
        args.out_dir = str(work / 'rendered')

    try:
        ensure_under_qa_dir(work)
        args.func(args)
    except QaError as exc:
        log('[ERROR] %s' % exc)
        return 2
    except Exception as exc:  # noqa: BLE001
        log('[ERROR] 未预期异常：%r' % (exc,))
        traceback.print_exc()
        return 3
    return 0


if __name__ == '__main__':
    sys.exit(main())

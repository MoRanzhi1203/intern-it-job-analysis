# -*- coding: utf-8 -*-
"""文本预处理模块：清洗、分段、语义分析版、模型安全版、分词与词面相似度。

本模块是所有文本处理的唯一实现，禁止在业务脚本中重复实现同类逻辑。

四类文本严格区分（禁止混用）：

- ``原始文本``      ：数据库原样文本，衍生字段一律新增列，绝不回写覆盖；
- ``清洗文本``      ：仅做低风险清洗（HTML / 零宽字符 / 换行 / 空白 / 项目符号）；
- ``语义分析版``    ：在清洗基础上剔除招聘平台模板与联系方式，保留职责与要求；
- ``模型安全版``    ：在语义分析版基础上彻底删除薪资数字、区间、元/天、元/月等直接目标泄漏信息。

语义分析版与模型安全版是两列；进入后续薪资模型的文本必须来自模型安全版。
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from . import project_paths

# ============================================================================
# 1. 低风险清洗（清洗文本）
# ============================================================================

HTML_TAG_PATTERN = re.compile(r'<[^>]{1,200}>')
HTML_ENTITY_MAP = {
    '&nbsp;': ' ', '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"',
    '&#39;': "'", '&ldquo;': '“', '&rdquo;': '”', '&hellip;': '…',
}
ZERO_WIDTH_PATTERN = re.compile(r'[\u200b-\u200f\u202a-\u202e\u2060\ufeff]')
BULLET_PATTERN = re.compile(r'(?m)^[\s]*[·•●○◆◇■□▪◦*\-–—+]{1,2}[\s]*')
WHITESPACE_RUN_PATTERN = re.compile(r'[ \t\u00a0\u3000]{2,}')
MULTI_NEWLINE_PATTERN = re.compile(r'\n{3,}')
NUMBERED_BULLET_PATTERN = re.compile(r'(?m)^[\s]*(\d{1,2})[、.)）]')


def strip_html(text: str) -> str:
    """去除 HTML 标签与常见实体。"""
    if not isinstance(text, str):
        return ''
    cleaned = HTML_TAG_PATTERN.sub(' ', text)
    for entity, replacement in HTML_ENTITY_MAP.items():
        cleaned = cleaned.replace(entity, replacement)
    return cleaned


def clean_text(text) -> str:
    """低风险清洗：仅处理格式噪声，保留技术名词、数字、版本号与职责信息。"""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ''
    cleaned = strip_html(str(text))
    cleaned = ZERO_WIDTH_PATTERN.sub('', cleaned)
    cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n')
    cleaned = BULLET_PATTERN.sub('', cleaned)
    cleaned = NUMBERED_BULLET_PATTERN.sub(lambda match: f'{match.group(1)}、', cleaned)
    cleaned = WHITESPACE_RUN_PATTERN.sub(' ', cleaned)
    cleaned = MULTI_NEWLINE_PATTERN.sub('\n\n', cleaned)
    lines = [line.strip() for line in cleaned.split('\n')]
    return '\n'.join(lines).strip()


# ============================================================================
# 2. 岗位描述结构化分段
# ============================================================================

# 小标题关键词 → 语义段类型
SECTION_HEADINGS = {
    '职责': ['岗位职责', '职位职责', '工作职责', '职责描述', '主要职责', '岗位描述', '职位描述',
             '工作内容', '工作描述', '你将负责', '你会做', '工作范围', '职责说明'],
    '要求': ['任职要求', '岗位要求', '职位要求', '任职资格', '职位资格', '任职条件', '岗位条件',
             '岗位需求', '申请要求', '我们希望你', '我们期待你', '我们需要你', '希望你',
             '你需要具备', '应聘要求', '人员要求', '任职条件要求'],
    '技能': ['技能要求', '技术要求', '专业技能', '技术栈', '岗位技能', '技术能力', '技能需求'],
    '加分': ['加分项', '加分点', '优先考虑', '具备以下经验优先', '有以下经验优先', '优先条件',
             '满足以下条件优先', '加分'],
    '福利': ['福利', '你将获得', '岗位福利', '薪资福利', '公司福利', '团队福利', '我们提供',
             '实习福利', '待遇', '实习收获', '你将得到'],
    '其他': ['公司介绍', '关于我们', '团队介绍', '联系方式', '其他说明', '备注', '工作地点说明',
             '招聘流程', '投递说明', '实习说明'],
}

ALL_HEADINGS = [heading for headings in SECTION_HEADINGS.values() for heading in headings]
HEADING_PATTERN = re.compile(
    r'(?:^|[\n。；;，,、\s])'
    r'(?:\d{1,2}\s*[、.)）]\s*)?'
    r'(?P<heading>' + '|'.join(sorted((re.escape(item) for item in ALL_HEADINGS),
                                      key=len, reverse=True)) + r')'
    r'\s*(?:[:：]|(?=\s*[\n])|$)')

HEADING_TO_SECTION = {heading: section
                      for section, headings in SECTION_HEADINGS.items()
                      for heading in headings}

SECTION_TEXT_FIELDS = {
    '职责': '职责文本',
    '要求': '任职要求文本',
    '技能': '技能要求文本',
    '加分': '加分项文本',
    '福利': '福利文本',
    '其他': '其他文本',
}

SECTION_FLAG_FIELDS = {
    '职责': '职责段存在标志',
    '要求': '要求段存在标志',
    '技能': '技能段存在标志',
    '加分': '加分段存在标志',
    '福利': '福利段存在标志',
}

SECTION_SPLIT_STATUS_FIELD = '岗位描述分段状态'
SECTION_STATUS_VALUES = ['完整', '部分', '未识别']

# ---- 岗位要求段落（技能提取优先使用） ----
# 参与技能提取的高可信段落：要求 / 技能 / 加分（复用上方 SECTION_HEADINGS，单一来源）
REQUIREMENT_SECTION_KEYS = ['要求', '技能', '加分']

# 语义分析版需要剔除的招聘平台模板 / 联系方式行
TEMPLATE_LINE_PATTERNS = [
    re.compile(r'(简历|投递|应聘).{0,6}(邮箱|链接|地址|方式|渠道|发送|邮件)'),
    re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+'),
    re.compile(r'(微信|QQ|电话|手机|联系方式|联系人)\s*[:：]?\s*[\w\-+]{5,}'),
    re.compile(r'https?://\S+'),
    re.compile(r'(版权|著作权|©|All Rights Reserved|保留所有权利)', re.IGNORECASE),
    re.compile(r'(实习僧|招聘平台|本平台|平台声明|免责声明)'),
    re.compile(r'(扫码|二维码|点击链接|点击投递)'),
    re.compile(r'^(公司地址|办公地址|公司官网)\s*[:：]'),
]
MARKETING_NOISE_PATTERNS = [
    re.compile(r'(我们是|我们是一家|本公司).{0,40}(行业|领先|独角兽|上市|五百强|第一)'),
]


def split_job_description(cleaned_text: str) -> dict:
    """把清洗后的岗位描述拆分为结构化段落。

    无法稳定识别小标题时**不强拆**：分段文本留空，只记录分段状态与标志位。
    """
    result: dict = {field: None for field in SECTION_TEXT_FIELDS.values()}
    result.update({field: 0 for field in SECTION_FLAG_FIELDS.values()})
    result[SECTION_SPLIT_STATUS_FIELD] = '未识别'
    result['_heading_count'] = 0
    if not cleaned_text:
        return result

    matches = list(HEADING_PATTERN.finditer(cleaned_text))
    if not matches:
        return result

    buckets: dict = {key: [] for key in SECTION_TEXT_FIELDS}
    for index, match in enumerate(matches):
        section = HEADING_TO_SECTION.get(match.group('heading'))
        if section is None:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(cleaned_text)
        content = cleaned_text[start:end].strip()
        if content:
            buckets[section].append(content)
    result['_heading_count'] = len(matches)

    recognized = 0
    for section, pieces in buckets.items():
        text = '\n'.join(pieces).strip()
        if text:
            result[SECTION_TEXT_FIELDS[section]] = text
            if section in SECTION_FLAG_FIELDS:
                result[SECTION_FLAG_FIELDS[section]] = 1
                recognized += 1
    has_duty = result[SECTION_FLAG_FIELDS['职责']] == 1
    has_require = result[SECTION_FLAG_FIELDS['要求']] == 1
    if has_duty and has_require:
        result[SECTION_SPLIT_STATUS_FIELD] = '完整'
    elif recognized > 0:
        result[SECTION_SPLIT_STATUS_FIELD] = '部分'
    return result


def extract_requirement_section(text) -> dict:
    """提取「岗位要求」段落，供技能提取优先使用。

    识别成功 → REQUIREMENT_SECTION（返回要求 / 技能 / 加分三段拼接文本）
    无法识别 → FULL_TEXT_FALLBACK（返回全文，仍允许提取技能，不丢弃岗位）
    空文本   → EMPTY_TEXT

    标题词表复用 ``SECTION_HEADINGS``（与 Stage 06 分段同一来源），不新增第二套标题规则。
    """
    if not isinstance(text, str) or not text.strip():
        return {'text': '', 'scope': 'EMPTY_TEXT', 'matched_headings': []}
    section_result = split_job_description(text)
    pieces = []
    headings = []
    for key in REQUIREMENT_SECTION_KEYS:
        piece = section_result.get(SECTION_TEXT_FIELDS[key])
        if piece:
            pieces.append(piece)
            headings.append(key)
    combined = '\n'.join(pieces).strip()
    if combined:
        return {'text': combined, 'scope': 'REQUIREMENT_SECTION', 'matched_headings': headings}
    return {'text': text, 'scope': 'FULL_TEXT_FALLBACK', 'matched_headings': []}


# ============================================================================
# 3. 语义分析版
# ============================================================================

def build_semantic_version(cleaned_text: str) -> str:
    """构建语义分析版：剔除联系方式、平台模板与与岗位业务无关的固定话术。"""
    if not cleaned_text:
        return ''
    kept_lines = []
    for line in cleaned_text.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue
        if any(pattern.search(stripped) for pattern in TEMPLATE_LINE_PATTERNS):
            continue
        if len(stripped) <= 120 and any(pattern.search(stripped)
                                        for pattern in MARKETING_NOISE_PATTERNS):
            continue
        kept_lines.append(stripped)
    return '\n'.join(kept_lines).strip()


# ============================================================================
# 4. 模型安全版（防目标泄漏）
# ============================================================================

SALARY_AMOUNT_UNIT = r'(?:元|块|人民币|k|K|千|万|w|W|亿)'
SALARY_KEYWORDS = r'(?:薪资|工资|薪酬|月薪|日薪|时薪|年薪|实习薪资|实习工资|补贴|津贴|奖金|绩效|报酬|待遇)'
SALARY_BENEFIT_KEYWORDS = r'(?:餐补|交通补贴|住房补贴|话费补贴|全勤奖|绩效奖金|项目奖金|加班费|报销)'
# 列表序号保护：数字后面紧跟 “.”、“、”、“)” 时属于条目编号，不是金额
SALARY_LIST_GUARD = r'(?!\s*[.、)）])'
# 金额区间连接符（含全角波浪线、长横线、至/到）
SALARY_RANGE_DASH = r'[-~～—－至到]'
# 括号包裹（方括号 / 中文书名号式方括号 / 圆括号 / 全角圆括号）
SALARY_BRACKET_OPEN = r'[\[【（(]'
SALARY_BRACKET_CLOSE = r'[\]】）)]'
SALARY_TIME_UNIT = r'(?:天|日|月|年|时|小时|周)'

# 正式薪资样式规则源（唯一）：Stage 06 清洗、Stage 06/07 门禁检测与 tests 全部复用
SALARY_LEAKAGE_PATTERNS = [
    # 0) 括号包裹的金额区间 + 时间单位：[200-300] 元/天、【200-300】元/天、（200-300）元/天
    re.compile(SALARY_BRACKET_OPEN + r'\s*\d+(?:\.\d+)?\s*' + SALARY_RANGE_DASH
               + r'\s*\d+(?:\.\d+)?\s*' + SALARY_BRACKET_CLOSE
               + r'\s*(?:元|块|人民币)?\s*/\s*' + SALARY_TIME_UNIT),
    # 0b) 括号包裹的金额区间 + 货币单位（无时间单位）：[200-300]元、（200-300）元
    re.compile(SALARY_BRACKET_OPEN + r'\s*\d+(?:\.\d+)?\s*' + SALARY_RANGE_DASH
               + r'\s*\d+(?:\.\d+)?\s*' + SALARY_BRACKET_CLOSE + r'\s*(?:元|块|人民币)'
               + SALARY_LIST_GUARD),
    # 1) 金额区间 + 时间单位：150-300/天、1.5k-2k/月、200～300元/天、200—300元/天
    re.compile(r'\d+(?:\.\d+)?\s*(?:[kK千万元wW])?\s*' + SALARY_RANGE_DASH
               + r'\s*\d+(?:\.\d+)?\s*(?:[kK千万元wW])?\s*(?:元|块|人民币)?\s*/\s*'
               + SALARY_TIME_UNIT),
    # 2) 金额 + 时间单位：200元/天、3000元/月、300 元/日
    re.compile(r'\d+(?:\.\d+)?\s*' + SALARY_AMOUNT_UNIT + r'\s*/\s*' + SALARY_TIME_UNIT),
    # 3) 金额区间 + 单位：150-300元、1-2万、200～300元
    re.compile(r'\d+(?:\.\d+)?\s*' + SALARY_RANGE_DASH + r'\s*\d+(?:\.\d+)?\s*'
               + SALARY_AMOUNT_UNIT),
    # 4) 货币符号 + 金额：¥1500、￥200
    re.compile(r'[¥￥]\s*\d+(?:\.\d+)?\s*(?:元|块|k|K|千|万)?'),
    # 5) 薪资关键词 + 金额（带单位，允许冒号后紧跟括号）
    re.compile(SALARY_KEYWORDS + r'\s*[:：]?\s*' + SALARY_BRACKET_OPEN + r'?\s*\d+(?:\.\d+)?\s*'
               + SALARY_AMOUNT_UNIT + SALARY_LIST_GUARD),
    # 6) 薪资关键词 + 多位数字（无单位，排除列表序号）
    re.compile(SALARY_KEYWORDS + r'\s*[:：]?\s*' + SALARY_BRACKET_OPEN + r'?\s*\d{2,}(?:\.\d+)?'
               + SALARY_LIST_GUARD),
    # 7) 福利类金额关键词 + 金额
    re.compile(SALARY_BENEFIT_KEYWORDS + r'\s*[:：]?\s*' + SALARY_BRACKET_OPEN + r'?\s*'
               + r'\d+(?:\.\d+)?\s*(?:元|块)?' + SALARY_LIST_GUARD),
    # 8) 纯数字 + 元：3000元、200元、3元以上（与严格检查统计范围一致）
    re.compile(r'\d+(?:\.\d+)?\s*元(?:/|\s*每)?(?:天|日|月|年|时|小时)?'),
    # 9) k / 千 / 万 金额缩写：8k、1.5万（排除数量单位与英文后缀）
    re.compile(r'\d+(?:\.\d+)?\s*[kKwW](?![A-Za-z])'),
    re.compile(r'\d+(?:\.\d+)?\s*[千万](?!\s*[人个次名条台份年月周天日户条款项])'),
    # 10) 薪资面议类
    re.compile(r'(?:' + SALARY_KEYWORDS + r')?\s*面议'),
    # 11) 裸币种单位 + 时间单位：元/天（无金额，如「计薪单位：元/天」）
    re.compile(r'(?:元|块|人民币)\s*/\s*' + SALARY_TIME_UNIT),
]
SALARY_REMOVAL_TOKEN = ' '
# 多轮清除：单轮替换可能使相邻数字重新组成新的薪资形态，故循环清除直到无命中
SALARY_CLEAN_ROUNDS = 3

# 严格薪资样式检查（残留必须为 0）：只覆盖「明确具有薪资语义」的表达。
# 刻意不把「纯括号数字区间」列为薪资样式，避免误伤 [4-5] 天到岗 / （3-5）天每周 /
# （2026.03~2026.07） 等实习天数与日期；「块」仅在紧跟时间单位时才视为币种。
STRICT_SALARY_PATTERNS = [
    # 1) 金额（可带区间与右括号）+ 元（可带时间单位）：200元、200-300元/天、[200-300] 元/天
    re.compile(r'\d+(?:\.\d+)?\s*(?:' + SALARY_RANGE_DASH + r'\s*\d+(?:\.\d+)?)?\s*'
               + SALARY_BRACKET_CLOSE + r'?\s*元\s*(?:/\s*' + SALARY_TIME_UNIT + r')?'),
    # 2) 金额（可带区间与右括号）+ 块 + 时间单位：300块/天
    re.compile(r'\d+(?:\.\d+)?\s*(?:' + SALARY_RANGE_DASH + r'\s*\d+(?:\.\d+)?)?\s*'
               + SALARY_BRACKET_CLOSE + r'?\s*块\s*/\s*' + SALARY_TIME_UNIT),
    # 3) 裸币种单位 + 时间单位：元/天、块/日
    re.compile(r'(?:元|块|人民币)\s*/\s*' + SALARY_TIME_UNIT),
    # 4) 货币符号 + 金额：¥1500、￥200
    re.compile(r'[¥￥]\s*\d+(?:\.\d+)?'),
    # 5) k / 千 / 万 金额缩写：8k、1.5万
    re.compile(r'\d+(?:\.\d+)?\s*[kKwW](?![A-Za-z])'),
    # 6) 薪资面议类
    re.compile(r'(?:薪资|工资|薪酬|月薪|日薪|时薪|年薪|待遇|补贴|津贴|奖金|报酬)\s*[:：]?\s*面议'),
    # 7) 金额区间 + 时间单位（不带元）：150-300/天
    re.compile(r'\d+(?:\.\d+)?\s*(?:' + SALARY_RANGE_DASH + r'\s*\d+(?:\.\d+)?)\s*/\s*'
               + SALARY_TIME_UNIT),
]


def strict_salary_hits(text) -> dict:
    """严格薪资样式命中统计（按规则逐条计数，供门禁与审计表使用）。"""
    if not isinstance(text, str) or not text:
        return {}
    detail = {}
    for index, pattern in enumerate(STRICT_SALARY_PATTERNS, start=1):
        hits = pattern.findall(text)
        if hits:
            detail[f'strict_{index}'] = len(hits)
    return detail


def detect_strict_salary_leakage(text) -> int:
    """严格薪资样式残留次数（要求为 0）。"""
    return int(sum(strict_salary_hits(text).values()))


def build_model_safe_version(semantic_text: str) -> tuple:
    """删除直接目标泄漏信息"""
    if not semantic_text:
        return '', 0
    safe = semantic_text
    hits = 0
    for _round in range(SALARY_CLEAN_ROUNDS):
        round_hits = 0
        for pattern in SALARY_LEAKAGE_PATTERNS:
            safe, count = pattern.subn(SALARY_REMOVAL_TOKEN, safe)
            round_hits += count
        hits += round_hits
        if round_hits == 0:
            break
    safe = re.sub(r'[ \t]{2,}', ' ', safe)
    safe = re.sub(r'\n{2,}', '\n', safe)
    return safe.strip(), hits


def detect_salary_leakage(text: str) -> int:
    """检查文本中是否仍残留薪资泄漏信息，返回命中次数。"""
    if not text:
        return 0
    return sum(len(pattern.findall(text)) for pattern in SALARY_LEAKAGE_PATTERNS)


# ============================================================================
# 5. 分词、段落与词面相似度
# ============================================================================

# 技术短词保护清单（禁止被停用词过滤）
PROTECTED_SHORT_TOKENS = {'r', 'go', 'c', 'ai', 'ml', 'dl', 'cv', 'sql', 'js', 'ts', 'bi',
                          'c++', 'c#', '.net', 'r语言'}

LATIN_TOKEN_PATTERN = re.compile(r'[A-Za-z][A-Za-z0-9+#._\-]{0,30}')
STOPWORDS_PATH = project_paths.STOPWORDS_PATH
_STOPWORDS_CACHE: set | None = None
_JIEBA_READY = False


def load_stopwords(path: Path | None = None) -> set:
    """读取停用词表（跳过注释与空行），并强制保护技术短词。"""
    global _STOPWORDS_CACHE
    if _STOPWORDS_CACHE is not None and path is None:
        return _STOPWORDS_CACHE
    target = Path(path or STOPWORDS_PATH)
    stopwords: set = set()
    if target.exists():
        for line in target.read_text(encoding='utf-8').splitlines():
            token = line.strip()
            if not token or token.startswith('#'):
                continue
            stopwords.add(token.lower())
    stopwords -= PROTECTED_SHORT_TOKENS
    if path is None:
        _STOPWORDS_CACHE = stopwords
    return stopwords


def tokenize(text: str) -> list:
    """分词：优先 jieba（若可用），否则退化为「英文词 + 中文双字」确定性切分。"""
    global _JIEBA_READY
    if not text:
        return []
    used = 'regex_fallback'
    tokens: list = []
    try:
        import jieba  # noqa: PLC0415

        if not _JIEBA_READY:
            jieba.initialize()
            _JIEBA_READY = True
        tokens = [token.strip().lower() for token in jieba.lcut(text) if token.strip()]
        used = 'jieba'
    except ImportError:
        latin = LATIN_TOKEN_PATTERN.findall(text)
        cjk_chars = re.findall(r'[\u4e00-\u9fff]', text)
        bigrams = [cjk_chars[index] + cjk_chars[index + 1]
                   for index in range(len(cjk_chars) - 1)]
        tokens = [token.lower() for token in latin] + bigrams
    _TOKENIZER_USED.add(used)
    stopwords = load_stopwords()
    return [token for token in tokens
            if token not in stopwords and (len(token) > 1 or token in PROTECTED_SHORT_TOKENS)]


_TOKENIZER_USED: set = set()


def tokenizer_used() -> str:
    """返回本次运行实际使用的分词器名称（写入审计表）。"""
    if not _TOKENIZER_USED:
        tokenize('测试 tokenizer Python 分词')
    return '+'.join(sorted(_TOKENIZER_USED))


def count_tokens(text: str) -> int:
    """分词数。"""
    return len(tokenize(text))


def count_paragraphs(text: str) -> int:
    """段落数：非空行数量。"""
    if not text:
        return 0
    return sum(1 for line in text.split('\n') if line.strip())


def char_length(text) -> int:
    """字符数（去除首尾空白后的 Unicode 字符长度）。"""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return 0
    return len(str(text).strip())


def normalize_for_compare(text) -> str:
    """词面对比归一化：去空白与常见标点差异。"""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ''
    normalized = str(text).lower()
    normalized = re.sub(r'[\s\u3000]+', '', normalized)
    normalized = re.sub(r'[，。、；：,.;:!！?？"“”\'‘’()（）\[\]【】~～—\-_]', '', normalized)
    return normalized


def token_jaccard(text_a: str, text_b: str) -> float:
    """Token Jaccard 相似度（词面指标，不等于深层语义）。"""
    set_a = set(tokenize(text_a))
    set_b = set(tokenize(text_b))
    if not set_a and not set_b:
        return float('nan')
    union = set_a | set_b
    if not union:
        return float('nan')
    return len(set_a & set_b) / len(union)


def edit_similarity(text_a: str, text_b: str) -> float:
    """编辑相似度（difflib 比值的另一种表述：1 - 距离/最大长度）。"""
    import difflib  # noqa: PLC0415

    norm_a = normalize_for_compare(text_a)
    norm_b = normalize_for_compare(text_b)
    if not norm_a and not norm_b:
        return float('nan')
    ratio = difflib.SequenceMatcher(None, norm_a, norm_b).ratio()
    return float(ratio)


def texts_identical(text_a, text_b) -> bool:
    """归一化后文本是否完全一致。"""
    return normalize_for_compare(text_a) == normalize_for_compare(text_b)


# ============================================================================
# 6. 标签解析
# ============================================================================

def split_tags(raw) -> list:
    """把顿号/逗号/竖线分隔的标签原文解析为列表。"""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    text = str(raw).strip()
    if not text:
        return []
    parts = re.split(r'[、,，|｜;；]+', text)
    return [item.strip() for item in parts if item.strip()]


def as_list(value) -> list:
    """把列表型字段统一转为 Python list（兼容 ndarray / None / 标量 / NaN）。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [item for item in value]
    if isinstance(value, float) and pd.isna(value):
        return []
    if hasattr(value, 'tolist'):
        try:
            return list(value.tolist())
        except (ValueError, AttributeError):
            return []
    return [value]


# ============================================================================
# 7. 岗位方向识别（文本侧交叉验证，不替换平台岗位大类/细分类）
# ============================================================================

# 方向识别规则：按顺序匹配，先命中者优先（大模型 / 算法等更具体的方向优先）
JOB_DIRECTION_TITLE_RULES = [
    ('大模型', ['大模型', 'LLM', 'AIGC', 'AIGC', 'RAG', 'Agent', '智能体', 'prompt', '提示词',
                '生成式', '语言模型', '多模态']),
    ('算法', ['算法', '机器学习', '深度学习', 'AI', '人工智能', '数据挖掘', '推荐', 'cv', 'nlp',
             '自然语言', '计算机视觉', '视觉', '语音', '预测', '建模']),
    ('前端开发', ['前端', 'web', 'h5', 'vue', 'react', 'javascript', 'html', 'css', '小程序',
                '页面', 'ui开发']),
    ('客户端', ['客户端', 'android', 'ios', '移动端', 'app开发', '鸿蒙', 'flutter']),
    ('后端开发', ['后端', '服务端', 'java', 'golang', 'go开发', 'python开发', 'php', 'node',
                '微服务', 'api', '架构', '平台开发', '中台', 'framework', '框架开发']),
    ('数据开发', ['数据开发', '数仓', '数据仓库', 'etl', '大数据', 'spark', 'flink', 'hadoop',
                'hive', '数据平台', '数据治理', '数据工程']),
    ('数据科学', ['数据科学', 'data scientist', '数据科学家', '定量分析']),
    ('数据分析', ['数据分析', '商业分析', '业务分析', '数据运营', 'bi', '数据洞察', '分析师']),
    ('测试', ['测试', 'qa', '质量保障', '自动化测试', '测试开发']),
    ('运维', ['运维', 'sre', 'devops', '系统工程师', '网络工程', 'it支持', '技术支持', '实施']),
    ('安全', ['安全', '渗透', '漏洞', '风控安全', '网络安全']),
    ('产品', ['产品经理', '产品运营', '产品设计', '产品助理', '产品专员', '产品岗', '产品']),
    ('设计', ['设计', '视觉', 'ui', 'ue', '交互', '平面', '美术', '插画', '美工']),
    ('运营', ['运营', '市场', '营销', '新媒体', '内容', '社群', '商务', '增长', '编辑']),
]

# 方向 → 平台岗位大类的对应关系（用于一致标志）
DIRECTION_TO_PLATFORM_GROUP = {
    '大模型': ['人工智能'],
    '算法': ['人工智能'],
    '前端开发': ['前端开发'],
    '客户端': ['客户端开发'],
    '后端开发': ['后端开发'],
    '数据开发': ['数据开发', '后端开发'],
    '数据科学': ['数据分析', '人工智能'],
    '数据分析': ['数据分析'],
    '测试': ['测试'],
    '运维': ['运维'],
    '安全': ['安全'],
    '产品': ['产品'],
    '设计': ['设计', '运营'],
    '运营': ['运营'],
    '其他': [],
}


def identify_job_direction(title, skills=None, description='') -> str:
    """结合岗位标题、技能集合与岗位描述识别岗位方向（文本侧）。"""
    title_text = ('' if title is None else str(title)).strip()
    description_text = ('' if description is None else str(description)).strip()
    skill_text = ' '.join(str(item) for item in as_list(skills))
    for direction, keywords in JOB_DIRECTION_TITLE_RULES:
        lowered = title_text.lower()
        if any(keyword.lower() in lowered for keyword in keywords):
            return direction
    for direction, keywords in JOB_DIRECTION_TITLE_RULES:
        combined = f'{description_text} {skill_text}'.lower()
        if any(keyword.lower() in combined for keyword in keywords):
            return direction
    return '其他'


def direction_consistent_with_group(direction: str, group_set) -> int:
    """判断文本方向与平台岗位大类是否一致（1 一致 / 0 不一致）。"""
    expected = DIRECTION_TO_PLATFORM_GROUP.get(direction, [])
    if not expected:
        return 0
    return int(bool(set(expected) & set(as_list(group_set))))

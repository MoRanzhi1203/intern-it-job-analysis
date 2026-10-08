# -*- coding: utf-8 -*-
"""公司认证标签模块：把「公司介绍图片链接」语义映射为「公司认证标签」。

原字段形态不可控，可能是：

- 单个 URL；
- 多个 URL（当前以 ``" | "`` 分隔）；
- JSON 列表字符串 / Python 列表字符串；
- 逗号 / 分号 / 空格 / 换行分隔；
- 空字符串或 NULL。

因此解析函数必须容错，且**必须先枚举全部真实文件名**，禁止只针对已确认图标
硬编码后静默忽略未知图标。

当前已确认映射：

    icon-best-employer.png → 最佳雇主
    icon-hyrz.png          → 行业认证

若出现任何未知非空文件名，调用方必须让 ``COMPANY_CERT_MAPPING`` 门禁失败，
停止删除原字段与后续去重。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from urllib.parse import urlsplit

import pandas as pd

# 当前已确认的图标语义映射（不代表允许忽略其他真实取值）
COMPANY_CERT_ICON_MAP = {
    'icon-best-employer.png': '最佳雇主',
    'icon-hyrz.png': '行业认证',
}

# 标签固定排序
CERT_TAG_ORDER = ['最佳雇主', '行业认证']

# URL 抽取（避免把分隔符/引号吞进链接）
_URL_PATTERN = re.compile(r'https?://[^\s|,;"\'\[\]()]+')

# 兜底分隔符：竖线 / 逗号 / 分号 / 空白（含换行、制表符）
_SPLIT_PATTERN = re.compile(r'[|,;，；\s]+')


def _parse_literal(text: str):
    """尝试把文本解析为 JSON / Python 字面量，失败返回 None。"""
    for loader in (json.loads,):
        try:
            return loader(text)
        except (ValueError, TypeError):
            pass
    try:
        import ast
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError):
        return None


def _iter_tokens(value) -> list:
    """把任意形态的取值拆分为 token 列表（保持顺序、保留重复）。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        tokens: list = []
        for item in value:
            tokens.extend(_iter_tokens(item))
        return tokens
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text[0] in '[({':
            parsed = _parse_literal(text)
            if isinstance(parsed, (list, tuple, set, frozenset)):
                return _iter_tokens(parsed)
        urls = _URL_PATTERN.findall(text)
        if urls:
            return urls
        return [part for part in _SPLIT_PATTERN.split(text) if part]
    return [str(value)]


def parse_company_cert_urls(value) -> list:
    """容错解析原始取值为 URL token 列表；空值返回空列表。"""
    if value is None:
        return []
    if not isinstance(value, (str, list, tuple, set, frozenset)):
        try:
            if pd.isna(value):
                return []
        except (TypeError, ValueError):
            pass
    return _iter_tokens(value)


def extract_cert_icon_name(url) -> str:
    """取 token 的 path 末段作为图标文件名（小写）；无法解析返回空字符串。"""
    if not isinstance(url, str) or not url.strip():
        return ''
    path = urlsplit(url.strip()).path or url.strip()
    return path.rsplit('/', 1)[-1].lower()


def sort_cert_tags(tags) -> list:
    """去重并按固定顺序排序公司认证标签。"""
    unique = set(tags)
    return sorted(unique, key=lambda tag: (CERT_TAG_ORDER.index(tag)
                                          if tag in CERT_TAG_ORDER else len(CERT_TAG_ORDER), tag))


def map_company_cert_tags(value) -> list:
    """把原始图片链接取值映射为固定排序的公司认证标签列表（空来源 → []）。"""
    tags = []
    for token in parse_company_cert_urls(value):
        tag = COMPANY_CERT_ICON_MAP.get(extract_cert_icon_name(token))
        if tag is not None:
            tags.append(tag)
    return sort_cert_tags(tags)


def _token_frame(series: pd.Series) -> pd.DataFrame:
    """展开全部 token，返回（图标文件名, token, 记录序号）明细表。"""
    rows = []
    for position, value in enumerate(series.tolist()):
        for token in parse_company_cert_urls(value):
            rows.append({
                '记录序号': position,
                '图标文件名': extract_cert_icon_name(token),
                'URL': token,
            })
    return pd.DataFrame(rows, columns=['记录序号', '图标文件名', 'URL'])


def audit_company_cert_values(series: pd.Series) -> dict:
    """公司介绍图片链接全量取值审计（不抽样，覆盖全部记录）。"""
    text = series.astype('object').map(lambda v: '' if v is None else str(v).strip())
    blank = text.eq('')
    tokens = _token_frame(series)
    filenames = tokens['图标文件名'].value_counts()
    known = set(COMPANY_CERT_ICON_MAP)
    unknown_names = sorted({name for name in tokens['图标文件名']
                            if name and name not in known})
    unknown_tokens = int(tokens['图标文件名'].isin(unknown_names).sum()) if unknown_names else 0
    empty_name_tokens = int(tokens['图标文件名'].eq('').sum())
    return {
        '总记录数': int(len(series)),
        '非空记录数': int((~blank).sum()),
        '空值记录数': int(blank.sum()),
        '原始唯一值数量': int(series.astype('object').map(
            lambda v: '' if v is None else str(v)).nunique()),
        'URL_token数量': int(len(tokens)),
        'URL文件名唯一值数量': int(filenames.shape[0]),
        '无法解析值数量': int((~blank & series.map(
            lambda v: len(parse_company_cert_urls(v)) == 0)).sum()),
        '未知文件名数量': int(len(unknown_names)),
        '未知URL数量': int(unknown_tokens + empty_name_tokens),
    }


def build_value_table(series: pd.Series) -> pd.DataFrame:
    """原始取值统计：原始值 / 记录数 / 占比。"""
    text = series.astype('object').map(lambda v: '' if v is None else str(v))
    table = text.value_counts(dropna=False).rename('记录数').reset_index()
    table.columns = ['原始取值', '记录数']
    table['占比'] = (table['记录数'] / len(series)).round(6) if len(series) else 0.0
    return table


def build_icon_table(series: pd.Series) -> pd.DataFrame:
    """URL 文件名统计：图标文件名 / token 数量 / 认证标签 / 是否已确认。"""
    tokens = _token_frame(series)
    if tokens.empty:
        return pd.DataFrame(columns=['图标文件名', 'token数量', '认证标签', '是否已确认'])
    table = (tokens.groupby('图标文件名').size().rename('token数量')
             .reset_index().sort_values('token数量', ascending=False))
    table['认证标签'] = table['图标文件名'].map(
        lambda name: COMPANY_CERT_ICON_MAP.get(name, ''))
    table['是否已确认'] = table['图标文件名'].map(
        lambda name: '是' if name in COMPANY_CERT_ICON_MAP else '否')
    return table.reset_index(drop=True)


def find_unknown_icons(series: pd.Series) -> pd.DataFrame:
    """未知图标检查：未知文件名 / 示例 URL / 出现次数。"""
    tokens = _token_frame(series)
    if tokens.empty:
        return pd.DataFrame(columns=['图标文件名', '示例URL', '出现次数'])
    unknown = tokens[~tokens['图标文件名'].isin(set(COMPANY_CERT_ICON_MAP))].copy()
    unknown = unknown[~unknown['图标文件名'].eq('')]
    if unknown.empty:
        return pd.DataFrame(columns=['图标文件名', '示例URL', '出现次数'])
    table = (unknown.groupby('图标文件名')
             .agg(示例URL=('URL', 'first'), 出现次数=('URL', 'size'))
             .reset_index().sort_values('出现次数', ascending=False))
    return table.reset_index(drop=True)


def build_mapping_table() -> pd.DataFrame:
    """映射字典表：图标文件名 / 认证标签。"""
    rows = [{'图标文件名': name, '认证标签': tag}
            for name, tag in COMPANY_CERT_ICON_MAP.items()]
    return pd.DataFrame(rows).sort_values('图标文件名').reset_index(drop=True)


def tag_combination_series(series: pd.Series) -> pd.Series:
    """生成标签组合的稳定文本表示（JSON，中文不转义）。"""
    return series.map(lambda tags: json.dumps(list(tags), ensure_ascii=False))


def build_tag_combination_table(series: pd.Series) -> pd.DataFrame:
    """标签组合统计：认证标签组合 / 记录数 / 占比。"""
    text = tag_combination_series(series)
    table = text.value_counts(dropna=False).rename('记录数').reset_index()
    table.columns = ['认证标签组合', '记录数']
    table['占比'] = (table['记录数'] / len(series)).round(6) if len(series) else 0.0
    return table


def mapped_token_count(series: pd.Series) -> tuple:
    """返回（token 总数, 已成功映射 token 数），用于 100% 映射门禁。"""
    tokens = _token_frame(series)
    total = int(len(tokens))
    mapped = int(tokens['图标文件名'].isin(set(COMPANY_CERT_ICON_MAP)).sum())
    return total, mapped


def cert_tag_counter(series: pd.Series) -> Counter:
    """统计各认证标签被命中的记录数。"""
    counter: Counter = Counter()
    for tags in series:
        for tag in tags:
            counter[tag] += 1
    return counter

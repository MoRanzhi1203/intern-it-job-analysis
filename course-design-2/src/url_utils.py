# -*- coding: utf-8 -*-
"""URL 工具模块：岗位详情链接规范化与审计。

规范化要求：
- 去首尾空格；
- 协议小写；
- 域名小写；
- 完整保留岗位身份 path（含 intern_id）；
- 删除 fragment；
- 删除无业务意义的跟踪 query 参数（当前已知至少包含 pcm）。
"""

from __future__ import annotations

import pandas as pd
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# 已知跟踪参数（无业务含义，规范化时删除）
TRACKING_PARAMS = frozenset({
    'pcm', 'from', 'utm_source', 'utm_medium', 'utm_campaign',
    'utm_term', 'utm_content', 'spm', 'ref', 'referer', 'share_token',
})

# 默认协议（缺失协议时补齐）
DEFAULT_SCHEME = 'https'


def normalize_detail_url(value: object) -> str:
    """规范化岗位详情链接；空值返回空字符串。"""
    if not isinstance(value, str) or not value.strip():
        return ''
    parts = urlsplit(value.strip())
    kept_query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                  if k.lower() not in TRACKING_PARAMS]
    return urlunsplit((
        (parts.scheme or DEFAULT_SCHEME).lower(),
        parts.netloc.lower(),
        parts.path,
        urlencode(kept_query),
        '',
    ))


def normalize_series(series: pd.Series) -> pd.Series:
    """对整列链接做规范化。"""
    return series.map(normalize_detail_url)


def extract_intern_id_from_url(value: object) -> str:
    """从详情链接 path 末段提取岗位 ID；无法提取时返回空字符串。"""
    if not isinstance(value, str) or not value.strip():
        return ''
    path = urlsplit(value.strip()).path
    return path.rsplit('/', 1)[-1] if path else ''


def audit_url_queries(series: pd.Series) -> pd.DataFrame:
    """统计 URL query 参数分布，返回 DataFrame（原始query / 记录数）。"""
    queries = series.map(lambda u: urlsplit(u).query if isinstance(u, str) else '')
    table = queries.value_counts().rename('记录数').reset_index()
    table.columns = ['query 参数', '记录数']
    return table


def audit_url_structure(series: pd.Series) -> dict:
    """统计协议、域名、fragment、首尾空格等结构信息。"""
    schemes = series.map(lambda u: urlsplit(u).scheme if isinstance(u, str) else '')
    netlocs = series.map(lambda u: urlsplit(u).netloc if isinstance(u, str) else '')
    fragments = series.map(lambda u: urlsplit(u).fragment if isinstance(u, str) else '')
    stripped = series.astype(str)
    return {
        'scheme_distribution': schemes.value_counts().to_dict(),
        'netloc_distribution': netlocs.value_counts().to_dict(),
        'fragment_rows': int(fragments.ne('').sum()),
        'leading_trailing_space_rows': int((series != stripped.str.strip()).sum()),
    }

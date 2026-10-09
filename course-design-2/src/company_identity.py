# -*- coding: utf-8 -*-
"""公司实体识别模块（保守策略，禁止仅凭文本相似度强行合并）。

核心原则（提示词第 27~31 节）：

1. 公司简介**不能按岗位ID**做时序，必须先建立公司实体映射；
2. 实体识别必须保守：公司改名、简称、品牌名不能只凭相似度合并；
3. 只有 ``EXACT_NORMALIZED_NAME`` / ``EXACT_NAME_LOCATION`` / ``APPROVED_ALIAS``
   三种映射方式允许进入正式公司简介时序；
4. 人工确认的别名统一写在 ``config/company_aliases.yml``，禁止散落硬编码。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd
import yaml

from . import project_paths, schema

COMPANY_ALIAS_PATH = project_paths.PROJECT_ROOT / 'config' / 'company_aliases.yml'

# ---- 映射方式（与提示词第 18 / 29 节严格一致） ----
METHOD_EXACT_NORMALIZED_NAME = 'EXACT_NORMALIZED_NAME'
METHOD_EXACT_NAME_LOCATION = 'EXACT_NAME_LOCATION'
METHOD_APPROVED_ALIAS = 'APPROVED_ALIAS'
METHOD_MULTI_LOCATION_AMBIGUOUS = 'MULTI_LOCATION_AMBIGUOUS'
METHOD_HIGH_CONFIDENCE_MATCH = 'HIGH_CONFIDENCE_MATCH'
METHOD_UNRESOLVED = 'UNRESOLVED'
MAPPING_METHODS = [METHOD_EXACT_NORMALIZED_NAME, METHOD_MULTI_LOCATION_AMBIGUOUS,
                   METHOD_APPROVED_ALIAS, METHOD_HIGH_CONFIDENCE_MATCH, METHOD_UNRESOLVED,
                   METHOD_EXACT_NAME_LOCATION]
# 正式映射方式（Refinement R1 收紧）：所在地不再作为自动拆分正式实体的充分条件
FORMAL_MAPPING_METHODS = [METHOD_EXACT_NORMALIZED_NAME, METHOD_APPROVED_ALIAS]
# 各映射方式的默认置信度
METHOD_CONFIDENCE = {
    METHOD_EXACT_NORMALIZED_NAME: 1.00,
    METHOD_APPROVED_ALIAS: 0.95,
    METHOD_HIGH_CONFIDENCE_MATCH: 0.90,
    METHOD_EXACT_NAME_LOCATION: 0.80,
    METHOD_MULTI_LOCATION_AMBIGUOUS: 0.70,
    METHOD_UNRESOLVED: 0.00,
}
# 不可进入正式公司时序的原因（写入审计字段）
METHOD_EXCLUDE_REASON = {
    METHOD_MULTI_LOCATION_AMBIGUOUS: '同名跨地域，所在地不足以作为自动拆分正式实体的充分条件，需人工确认',
    METHOD_HIGH_CONFIDENCE_MATCH: '不同名称但公司简介指纹一致，未经人工确认不得合并',
    METHOD_EXACT_NAME_LOCATION: '旧统计范围按「名称+所在地」拆分，本轮不再作为正式进入条件',
    METHOD_UNRESOLVED: '公司名称为空，无法解析',
}

FULL_WIDTH_OFFSET = 0xFEE0


def to_half_width(text: str) -> str:
    """全角转半角（仅处理 ASCII 可见区间）。"""
    return ''.join(
        chr(ord(char) - FULL_WIDTH_OFFSET) if 0xFF01 <= ord(char) <= 0xFF5E else char
        for char in text
    )


def normalize_company_name(name) -> str:
    """公司名称规范化：仅做低风险归一化，不做任何模糊合并。"""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ''
    text = to_half_width(str(name)).strip()
    text = re.sub(r'[\u3000\s]+', '', text)
    text = re.sub(r'[·・．.。，“”"\'（）()\[\]【】<>《》]+', '', text)
    text = text.replace('（', '').replace('）', '')
    return text


def company_fingerprint(profile_text) -> str:
    """公司简介指纹：用于判定「同一简介被多个名称引用」。"""
    if profile_text is None or (isinstance(profile_text, float) and pd.isna(profile_text)):
        return ''
    text = re.sub(r'[\s\u3000]+', '', str(profile_text))
    if not text:
        return ''
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]


def load_company_aliases(path: Path | None = None) -> tuple:
    """加载公司别名配置，返回 (approved 映射, 全部条目, 原始配置)。

    approved 映射为 ``别名规范化名称 -> 规范名称``，只包含 status=approved 的条目。
    """
    target = Path(path or COMPANY_ALIAS_PATH)
    if not target.exists():
        raise FileNotFoundError(f'公司别名配置不存在: {target}')
    payload = yaml.safe_load(target.read_text(encoding='utf-8')) or {}
    entries = payload.get('companies') or []
    approved: dict = {}
    rows = []
    for entry in entries:
        canonical = str(entry.get('canonical', '')).strip()
        if not canonical:
            continue
        status = str(entry.get('status', 'pending')).strip().lower()
        aliases = [str(item).strip() for item in entry.get('aliases') or [canonical]]
        if canonical not in aliases:
            aliases.insert(0, canonical)
        rows.append({
            '规范名称': canonical,
            '别名数量': len(aliases),
            '别名列表': '、'.join(aliases),
            '状态': 'approved' if status == 'approved' else 'pending',
            '确认依据': str(entry.get('evidence', '')).strip(),
            '备注': str(entry.get('note', '')).strip(),
        })
        if status != 'approved':
            continue
        canonical_normalized = normalize_company_name(canonical)
        approved[canonical_normalized] = canonical_normalized
        for alias in aliases:
            approved[normalize_company_name(alias)] = canonical_normalized
    return approved, pd.DataFrame(rows), payload


def build_company_entity_map(name_location_stats: pd.DataFrame,
                             profile_fingerprint_stats: pd.DataFrame | None = None,
                             alias_path: Path | None = None) -> tuple:
    """构建公司实体映射表（保守策略，禁止仅凭所在地或相似度自动拆分/合并）。

    输入 ``name_location_stats`` 至少包含：公司名称、公司所在地、岗位数、观测记录数。
    返回 ``(entity_map, diagnostics)``。
    """
    approved, alias_table, _payload = load_company_aliases(alias_path)
    frame = name_location_stats.copy()
    frame['公司名称'] = frame['公司名称'].fillna('').astype(str)
    frame['公司所在地'] = frame['公司所在地'].fillna('').astype(str)
    frame['规范化名称'] = frame['公司名称'].map(normalize_company_name)
    frame['所在地'] = frame['公司所在地'].str.strip()

    # ---- 1) 空名称 → UNRESOLVED ----
    unresolved = frame[frame['规范化名称'] == '']
    resolved = frame[frame['规范化名称'] != '']

    # ---- 2) 高置信度候选：不同名称共享完全相同的公司简介指纹 ----
    fingerprint_by_name: dict = {}
    fingerprint_count: dict = {}
    if profile_fingerprint_stats is not None and not profile_fingerprint_stats.empty:
        profiles = profile_fingerprint_stats.copy()
        profiles['规范化名称'] = profiles['公司名称'].map(normalize_company_name)
        counts = (profiles[profiles['规范化名称'] != '']
                  .groupby('规范化名称')['公司简介指纹'].nunique())
        fingerprint_count = {name: int(value) for name, value in counts.items()}
        grouped_profiles = (profiles.groupby('公司简介指纹')['规范化名称']
                            .apply(lambda names: sorted(set(names))))
        for fingerprint, names in grouped_profiles.items():
            if not fingerprint or len(names) < 2:
                continue
            for name in names:
                fingerprint_by_name.setdefault(name, set()).update(
                    item for item in names if item != name)

    # ---- 3) 同名跨地域：不再自动拆分正式实体，统一标记 MULTI_LOCATION_AMBIGUOUS ----
    location_map: dict = {}
    for normalized, group in resolved.groupby('规范化名称', sort=True):
        location_map[normalized] = sorted({item for item in group['所在地'] if item})
    multi_location_names = {name: locations
                            for name, locations in location_map.items() if len(locations) > 1}

    rows = []
    for record in unresolved.to_dict('records'):
        rows.append({
            'company_entity_id': '',
            'company_name_original': record['公司名称'],
            'company_name_normalized': '',
            '公司所在地': record['所在地'],
            '映射方式': METHOD_UNRESOLVED,
            '映射置信度': METHOD_CONFIDENCE[METHOD_UNRESOLVED],
            '是否需人工复核': 1,
            '是否进入正式公司时序': 0,
            '岗位数': int(record.get('岗位数', 0) or 0),
            '观测记录数': int(record.get('观测记录数', 0) or 0),
            '所在地数量': 0,
            '所在地集合': [],
            '是否跨地域': 0,
            '公司简介指纹数量': 0,
            '是否存在同名跨地域歧义': 0,
            '旧映射方式_EXACT_NAME_LOCATION适用性': 0,
            '不可进入正式公司时序原因': METHOD_EXCLUDE_REASON[METHOD_UNRESOLVED],
        })

    for record in resolved.to_dict('records'):
        normalized = record['规范化名称']
        location = record['所在地']
        locations = location_map.get(normalized, [])
        cross_location = int(len(locations) > 1)
        if normalized in approved:
            method = METHOD_APPROVED_ALIAS
            entity_id = approved[normalized]
        elif cross_location:
            # 名称相同、所在地不同：可能是同一企业不同办公地点，禁止自动拆分
            method = METHOD_MULTI_LOCATION_AMBIGUOUS
            entity_id = normalized
        elif normalized in fingerprint_by_name:
            # 名称不同但公司简介指纹完全一致：人工确认前不合并，仅作候选
            method = METHOD_HIGH_CONFIDENCE_MATCH
            entity_id = normalized
        else:
            method = METHOD_EXACT_NORMALIZED_NAME
            entity_id = normalized
        rows.append({
            'company_entity_id': entity_id,
            'company_name_original': record['公司名称'],
            'company_name_normalized': normalized,
            '公司所在地': location,
            '映射方式': method,
            '映射置信度': METHOD_CONFIDENCE[method],
            '是否需人工复核': int(method not in FORMAL_MAPPING_METHODS),
            '是否进入正式公司时序': int(method in FORMAL_MAPPING_METHODS),
            '岗位数': int(record.get('岗位数', 0) or 0),
            '观测记录数': int(record.get('观测记录数', 0) or 0),
            '所在地数量': len(locations),
            '所在地集合': locations,
            '是否跨地域': cross_location,
            '公司简介指纹数量': int(fingerprint_count.get(normalized, 0)),
            '是否存在同名跨地域歧义': cross_location,
            '旧映射方式_EXACT_NAME_LOCATION适用性': cross_location,
            '不可进入正式公司时序原因': ('' if method in FORMAL_MAPPING_METHODS
                                 else METHOD_EXCLUDE_REASON.get(method, '')),
        })

    entity_map = pd.DataFrame(rows)
    if entity_map.empty:
        entity_map = pd.DataFrame(columns=schema.COMPANY_ENTITY_MAP_COLUMNS)
    entity_map = entity_map[schema.COMPANY_ENTITY_MAP_COLUMNS]

    # ---- 4) 一对多 / 多对一检查 ----
    formal_map = entity_map[entity_map['是否进入正式公司时序'] == 1]
    one_to_many = (formal_map.groupby('company_entity_id')['company_name_normalized']
                   .agg(来源名称数='nunique', 来源名称='、'.join)
                   .reset_index().sort_values('来源名称数', ascending=False))
    one_to_many = one_to_many[one_to_many['来源名称数'] > 1]
    many_to_one = (entity_map.groupby('company_name_normalized')['company_entity_id']
                   .agg(实体数='nunique', 实体列表='、'.join)
                   .reset_index().sort_values('实体数', ascending=False))
    many_to_one = many_to_one[many_to_one['实体数'] > 1]

    diagnostics = {
        '别名配置表': alias_table,
        '一对多映射': one_to_many,
        '多对一映射': many_to_one,
        '同名跨地域': pd.DataFrame(
            [{'公司规范名称': name, '所在地数量': len(locations),
              '所在地集合': '、'.join(locations),
              '映射方式': METHOD_MULTI_LOCATION_AMBIGUOUS,
              '是否进入正式公司时序': 0}
             for name, locations in sorted(multi_location_names.items())]),
        '简介指纹共享候选': pd.DataFrame(
            [{'公司规范名称': name, '同简介其他名称数': len(others),
              '同简介其他名称': '、'.join(sorted(others))}
             for name, others in sorted(fingerprint_by_name.items())]),
        '旧EXACT_NAME_LOCATION影响': pd.DataFrame(
            [{'公司规范名称': name, '所在地数量': len(locations),
              '旧实体数（按名称@所在地拆分）': len(locations),
              '新实体数（不再拆分）': 1,
              '是否进入正式公司时序': 0}
             for name, locations in sorted(multi_location_names.items())]),
    }
    return entity_map, diagnostics


def resolve_formal_entities(entity_map: pd.DataFrame,
                            require_review_ok: bool = True) -> pd.DataFrame:
    """返回允许进入正式公司简介时序的实体映射行。

    Refinement R1 起正式统计范围只允许 ``EXACT_NORMALIZED_NAME`` / ``APPROVED_ALIAS``，
    且默认要求「无需人工复核」。
    """
    formal = entity_map[entity_map['映射方式'].isin(FORMAL_MAPPING_METHODS)].copy()
    if require_review_ok:
        formal = formal[formal['是否需人工复核'] == 0]
    return formal

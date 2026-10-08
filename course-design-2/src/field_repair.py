# -*- coding: utf-8 -*-
"""源记录公司属性语义槽位异常：检测、分型、确定性修复与校验（唯一正式实现）。

## 正式名称

本模块处理的问题统一称为：

    source-record company-attribute semantic slot anomaly
    源记录公司属性语义槽位异常

禁止再使用「平台字段错位」「MySQL 迁移错位」「Parquet 迁移错位」等表述
（依据 docs/records/15_company_attribute_semantic_anomaly_record.md 的取证结论：
MySQL 与 Stage 00 raw Parquet 同 ID 30 列完全一致 467/467，异常来自源记录本身）。

## 取证结论（真实数据，172,063 条原始观测）

| 症状 | 行数 |
| --- | --- |
| 公司性质栏位写着人数区间（如 `150-500人`） | 451 |
| 公司规模栏位写着地点文本（如 `上海`、`北京市/北京`） | 356 |
| 两类症状同时出现 | 340 |
| 三类异常候选并集 | 467 |
| 上述候选行的「公司所在地」为空 | 467 |

## 三类异常与确定性修复（只在 Stage 01 工作副本执行一次）

| 类型 | 条件 | 修复 |
| --- | --- | --- |
| FULL_SHIFT | 性质像人数规模 且 规模像地点 且 所在地为空 | 规模 ← 原性质；所在地 ← 原规模；性质 ← 缺失 |
| NATURE_ONLY | 性质像人数规模 且 规模非地点 且 所在地为空 | 规模 ← 原性质；性质 ← 缺失；所在地保持原值 |
| SIZE_ONLY | 性质正常 且 规模像地点 且 所在地为空 | 所在地 ← 原规模；规模 ← 缺失；性质保持原值 |

原公司性质一旦丢失即**置为缺失**，禁止众数填充 / 同公司回填 / 按行业或规模推断；
审计辅助列（异常类型、原值、修复值、检测证据、参考行 ID）只进入审计表，
不进入 `data/interim` 正式宽表。
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import yaml

from src import project_paths

NATURE_FIELD = '公司性质'
SIZE_FIELD = '公司规模'
LOCATION_FIELD = '公司所在地'
CITY_FIELD = '工作城市'
JOB_ID_FIELD = '实习岗位ID'
COMPANY_NAME_FIELD = '公司名称'

# 三类异常（条件均要求「公司所在地为空」）
FULL_SHIFT = 'FULL_SHIFT'
NATURE_ONLY = 'NATURE_ONLY'
SIZE_ONLY = 'SIZE_ONLY'
ANOMALY_TYPES = [FULL_SHIFT, NATURE_ONLY, SIZE_ONLY]

CONFIG_SECTION = 'company_attribute_slot_anomaly'

# 审计表明细列（仅进入审计表，不进入正式宽表）
AUDIT_COLUMNS = [
    '源记录序号', JOB_ID_FIELD, COMPANY_NAME_FIELD, '异常类型',
    '公司性质_原值', '公司性质_修复值',
    '公司规模_原值', '公司规模_修复值',
    '公司所在地_原值', '公司所在地_修复值',
    '检测证据', '性质是否可还原',
]


def load_company_attribute_anomaly_config() -> dict:
    """读取结构化字段配置，并校验公司属性槽位异常小节存在。"""
    path = project_paths.STRUCTURED_FIELDS_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f'缺少结构化字段配置: {path}')
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    if CONFIG_SECTION not in config:
        raise KeyError(f'配置缺少 {CONFIG_SECTION} 小节')
    return config


def _clean(series: pd.Series) -> pd.Series:
    return series.fillna('').astype(str).str.strip()


def build_city_vocabulary(frame: pd.DataFrame, separators: list) -> set:
    """由真实「工作城市」列构建城市词表（含多取值首段），作为地点文本的证据来源。"""
    values = _clean(frame[CITY_FIELD]) if CITY_FIELD in frame.columns else pd.Series(dtype=str)
    vocabulary = set(values.unique())
    pattern = '|'.join(re.escape(item) for item in separators) if separators else None
    for value in values.unique():
        if not value:
            continue
        segments = re.split(pattern, value) if pattern else [value]
        vocabulary.update(segment.strip() for segment in segments if segment.strip())
    vocabulary.discard('')
    return vocabulary


def is_scale_text(series: pd.Series, config: dict) -> pd.Series:
    """判断取值是否为人数区间形态（含无单位形态，如 100-200）。"""
    scale = config['company_scale']
    cleaned = _clean(series)
    patterns = [scale['range_pattern'], scale['range_pattern_no_unit'], scale['less_pattern'],
                scale['less_suffix_pattern'], scale['more_pattern'], scale['single_pattern']]
    result = pd.Series(False, index=series.index)
    for pattern in patterns:
        result = result | cleaned.str.match(pattern)
    return result


def is_location_text(series: pd.Series, city_vocabulary: set, config: dict) -> pd.Series:
    """判断取值是否为地点文本：命中工作城市词表，或含省/市/区/县/州等行政区字样。"""
    cleaned = _clean(series)
    in_vocabulary = cleaned.isin(city_vocabulary)
    tokens = list(config[CONFIG_SECTION].get('location_tokens', []))
    if tokens:
        has_admin = cleaned.str.contains('|'.join(re.escape(token) for token in tokens), regex=True)
    else:
        has_admin = pd.Series(False, index=series.index)
    return cleaned.ne('') & (in_vocabulary | has_admin)


def classify_company_attribute_slot_anomaly(nature_scale: pd.Series, size_location: pd.Series,
                                            location_empty: pd.Series) -> pd.Series:
    """按三类规则判定异常类型；命中「症状但所在地非空」的行返回 NOT_REPAIRABLE（不修复）。"""
    nature_hit = nature_scale.to_numpy()
    size_hit = size_location.to_numpy()
    empty = location_empty.to_numpy()
    types = np.full(len(nature_scale), '', dtype=object)
    both = nature_hit & size_hit & empty
    nature_only = nature_hit & (~size_hit) & empty
    size_only = (~nature_hit) & size_hit & empty
    types[both] = FULL_SHIFT
    types[nature_only] = NATURE_ONLY
    types[size_only] = SIZE_ONLY
    types[(nature_hit | size_hit) & (~empty)] = 'NOT_REPAIRABLE'
    return pd.Series(types, index=nature_scale.index)


def detect_company_attribute_slot_anomaly(frame: pd.DataFrame, city_vocabulary: set = None,
                                          config: dict = None) -> tuple:
    """检测源记录公司属性语义槽位异常，返回（异常明细表, 诊断信息）。

    只做检测与留痕，不修改入参；修复由 apply_deterministic_company_attribute_repair 完成。
    """
    config = config or load_company_attribute_anomaly_config()
    city_vocabulary = (city_vocabulary if city_vocabulary is not None else
                       build_city_vocabulary(frame, config['city_normalization']['separators']))

    nature = _clean(frame[NATURE_FIELD])
    size = _clean(frame[SIZE_FIELD])
    location = _clean(frame[LOCATION_FIELD])
    nature_scale = is_scale_text(nature, config)
    size_scale = is_scale_text(size, config)
    size_location = (~size_scale) & is_location_text(size, city_vocabulary, config)
    location_empty = location.eq('')
    anomaly_type = classify_company_attribute_slot_anomaly(nature_scale, size_location,
                                                           location_empty)
    repairable = anomaly_type.isin(ANOMALY_TYPES)
    not_repairable = anomaly_type.eq('NOT_REPAIRABLE')

    records = []
    for position in np.flatnonzero(repairable.to_numpy()):
        current_type = anomaly_type.iloc[position]
        nature_hit = bool(nature_scale.iloc[position])
        size_hit = bool(size_location.iloc[position])
        evidence = []
        if nature_hit:
            evidence.append('公司性质栏位为人数区间')
        if size_hit:
            evidence.append('公司规模栏位为地点文本')
        evidence.append('公司所在地栏位为空')
        recoverable = not nature_hit
        records.append({
            '源记录序号': int(position),
            JOB_ID_FIELD: frame[JOB_ID_FIELD].iloc[position] if JOB_ID_FIELD in frame.columns else '',
            COMPANY_NAME_FIELD: (frame[COMPANY_NAME_FIELD].iloc[position]
                                 if COMPANY_NAME_FIELD in frame.columns else ''),
            '异常类型': current_type,
            '公司性质_原值': nature.iloc[position],
            '公司性质_修复值': '' if nature_hit else nature.iloc[position],
            '公司规模_原值': size.iloc[position],
            '公司规模_修复值': nature.iloc[position] if nature_hit else '',
            '公司所在地_原值': location.iloc[position],
            '公司所在地_修复值': size.iloc[position] if size_hit else location.iloc[position],
            '检测证据': '；'.join(evidence),
            '性质是否可还原': '是（性质栏位本身正常，未被前移覆盖）' if recoverable
            else '否（源记录槽位缺位，无法从本行三字段恢复）',
        })
    audit_table = pd.DataFrame(records, columns=AUDIT_COLUMNS)

    type_counts = audit_table['异常类型'].value_counts().to_dict() if len(audit_table) else {}
    diagnostics = {
        '总观测行数': int(len(frame)),
        '性质栏位为人数区间行数': int(nature_scale.sum()),
        '规模栏位为地点文本行数': int(size_location.sum()),
        '两类症状同时出现行数': int((nature_scale & size_location).sum()),
        '异常候选并集行数': int(repairable.sum()),
        '症状命中但所在地非空（不修复）行数': int(not_repairable.sum()),
        'FULL_SHIFT 行数': int(type_counts.get(FULL_SHIFT, 0)),
        'NATURE_ONLY 行数': int(type_counts.get(NATURE_ONLY, 0)),
        'SIZE_ONLY 行数': int(type_counts.get(SIZE_ONLY, 0)),
        '修复后规模可还原行数': int(audit_table['公司规模_修复值'].ne('').sum()) if len(audit_table) else 0,
        '修复后所在地可还原行数': (int(audit_table['公司所在地_修复值'].ne('').sum())
                          if len(audit_table) else 0),
        '性质被置为缺失行数': (int(audit_table['公司性质_修复值'].eq('').sum())
                       if len(audit_table) else 0),
        '涉及岗位数': (int(audit_table[JOB_ID_FIELD].replace('', pd.NA).nunique())
                   if len(audit_table) else 0),
        '涉及公司数': (int(audit_table[COMPANY_NAME_FIELD].replace('', pd.NA).nunique())
                   if len(audit_table) else 0),
    }
    return audit_table, diagnostics


def apply_deterministic_company_attribute_repair(frame: pd.DataFrame,
                                                 audit_table: pd.DataFrame) -> pd.DataFrame:
    """按异常明细回填三个公司属性字段，返回新的 DataFrame（不修改入参）。

    无法从本行恢复的槽位一律置为缺失（NaN），不做任何推断填充。
    """
    repaired = frame.copy()
    if audit_table.empty:
        return repaired
    positions = audit_table['源记录序号'].to_numpy()
    for source_column, target_column in [('公司性质_修复值', NATURE_FIELD),
                                         ('公司规模_修复值', SIZE_FIELD),
                                         ('公司所在地_修复值', LOCATION_FIELD)]:
        values = audit_table[source_column].to_numpy(dtype=object)
        values = np.where(pd.Series(values).fillna('').astype(str).str.strip().eq(''),
                          np.nan, values)
        repaired.iloc[positions, repaired.columns.get_loc(target_column)] = values
    return repaired


def validate_company_attribute_repair(before: pd.DataFrame, after: pd.DataFrame,
                                      audit_table: pd.DataFrame,
                                      city_vocabulary: set = None,
                                      config: dict = None) -> dict:
    """修复后校验：残留为零、修复行仅三字段变化、未修复行逐格不变。"""
    config = config or load_company_attribute_anomaly_config()
    _, residual = detect_company_attribute_slot_anomaly(after, city_vocabulary, config)
    positions = audit_table['源记录序号'].to_numpy() if len(audit_table) else np.array([], dtype=int)
    touched = set(positions.tolist())
    untouched_mask = ~np.isin(np.arange(len(before)), list(touched))
    repaired_fields = [NATURE_FIELD, SIZE_FIELD, LOCATION_FIELD]
    other_fields = [column for column in before.columns if column not in repaired_fields]

    untouched_before = before.loc[untouched_mask, repaired_fields].astype(str).map(str).to_numpy()
    untouched_after = after.loc[untouched_mask, repaired_fields].astype(str).map(str).to_numpy()
    other_before = before[other_fields].astype(str).map(str).to_numpy()
    other_after = after[other_fields].astype(str).map(str).to_numpy()

    checks = {
        '行数与列序一致': bool(len(before) == len(after)
                          and list(before.columns) == list(after.columns)),
        '修复后残留异常行数': int(residual['异常候选并集行数']),
        '修复后性质像规模行数': int(residual['性质栏位为人数区间行数']),
        '修复后规模像地点行数': int(residual['规模栏位为地点文本行数']),
        '未修复行三字段逐格未变': bool((untouched_before == untouched_after).all()),
        '修复行仅三字段变化': bool((other_before == other_after).all()),
        '修复行数一致': bool(len(audit_table) == int(
            (before.loc[positions, repaired_fields].astype(str).map(str).to_numpy()
             != after.loc[positions, repaired_fields].astype(str).map(str).to_numpy())
            .any(axis=1).sum()) if len(audit_table) else True),
    }
    checks['是否通过'] = bool(checks['行数与列序一致']
                          and checks['修复后残留异常行数'] == 0
                          and checks['未修复行三字段逐格未变']
                          and checks['修复行仅三字段变化']
                          and checks['修复行数一致'])
    return checks


def build_nature_lost_reference_hint(frame: pd.DataFrame,
                                     audit_table: pd.DataFrame) -> pd.DataFrame:
    """性质已丢失行的人工参考提示：同公司其它（未异常）记录中的唯一性质取值。

    仅作人工复核参考，**不参与任何自动填充**，也不写入正式宽表。
    """
    if audit_table.empty:
        return pd.DataFrame(columns=[COMPANY_NAME_FIELD, '性质缺失行数', '同公司唯一性质取值',
                                     '参考行ID', '参考依据'])
    lost = audit_table[audit_table['公司性质_修复值'].eq('')]
    if lost.empty:
        return pd.DataFrame(columns=[COMPANY_NAME_FIELD, '性质缺失行数', '同公司唯一性质取值',
                                     '参考行ID', '参考依据'])
    config = load_company_attribute_anomaly_config()
    aligned = frame.loc[~frame.index.isin(audit_table['源记录序号'].to_numpy())]
    aligned = aligned[aligned[NATURE_FIELD].fillna('').astype(str).str.strip().ne('')]
    aligned = aligned[~is_scale_text(aligned[NATURE_FIELD], config)]
    rows = []
    for company, group in lost.groupby(COMPANY_NAME_FIELD):
        same_company = aligned[aligned[COMPANY_NAME_FIELD] == company]
        values = sorted({str(item).strip() for item in same_company[NATURE_FIELD]
                         if str(item).strip()})
        reference_id = ''
        if len(values) == 1 and '记录ID' in same_company.columns:
            reference_id = same_company.loc[
                same_company[NATURE_FIELD].astype(str).str.strip() == values[0], '记录ID'].iloc[0]
        rows.append({
            COMPANY_NAME_FIELD: company,
            '性质缺失行数': int(len(group)),
            '同公司唯一性质取值': values[0] if len(values) == 1 else '',
            '参考行ID': reference_id,
            '参考依据': ('同公司其它记录唯一性质取值（待人工确认，未自动填充）' if len(values) == 1
                     else ('同公司无其它正常记录' if not values
                           else '同公司存在多个性质取值，无法唯一确定')),
        })
    frame_table = pd.DataFrame(rows, columns=[COMPANY_NAME_FIELD, '性质缺失行数',
                                              '同公司唯一性质取值', '参考行ID', '参考依据'])
    return frame_table.sort_values('性质缺失行数', ascending=False).reset_index(drop=True)


def build_repair_rule_table(config: dict = None) -> pd.DataFrame:
    """确定性修复规则表（写入审计表，供人工核对规则本身）。"""
    config = config or load_company_attribute_anomaly_config()
    section = config[CONFIG_SECTION]
    rows = []
    for anomaly_type in ANOMALY_TYPES:
        rows.append({
            '异常类型': anomaly_type,
            '触发条件': section['anomaly_types'][anomaly_type],
            '确定性修复': section['repair_policy'][anomaly_type],
        })
    rows.append({
        '异常类型': '性质不可还原',
        '触发条件': '公司性质栏位被前移覆盖（FULL_SHIFT / NATURE_ONLY）',
        '确定性修复': section['repair_policy']['nature_unrecoverable_action'],
    })
    return pd.DataFrame(rows)

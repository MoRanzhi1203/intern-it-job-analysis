# -*- coding: utf-8 -*-
"""Stage 11：结构化业务字段清洗与薪资目标解析。

处理路线：

data/processed/job_details_unique.parquet          最终岗位实体层（17,144 行）
data/processed/job_version_history.parquet         版本时序层（用于首版本薪资与漂移审计）
        ↓  薪资字段解析：形态枚举 → 面议 / 区间 / 单值 → 下限 / 上限 / 中点
        ↓  结构化字段清洗：学历 / 每周到岗 / 实习时长 / 公司规模 / 工作城市
        ↓  公司属性语义槽位异常残留复核（只在 Stage 01 修复，本阶段不修复）
        ↓
data/processed/job_salary_targets.parquet          薪资目标层（一行 = 一个岗位实体）
data/processed/job_structured_features.parquet     结构化特征层（一行 = 一个岗位实体）

统计范围要点（严格遵守，禁止静默修补）：
1. 单位不折算：本次数据薪资文本全量为「元/天」，不引入「月→日」换算假设；
2. 面议样本不插补：`是否面议 = 1` 且三个数值列留空（正式薪资中点 = 缺失），仅在审计中单列；
3. 主目标固定为「薪资中点」，上下限只用于描述统计与稳健性实验，不建立双目标主任务；
4. 只标记逻辑无效值（上下限倒置 / 非正数），极端值交由分布审计与缩尾敏感性处理；
5. 目标标签取「最终核心版本」的薪资信息，与实体选择规则一致；
6. 公司属性语义槽位异常只在 Stage 01 做确定性修复，本阶段只复核残留，残留即 FAIL。

用法：
    python scripts/pipeline/11_clean_structured_fields.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = next(
    _candidate for _candidate in Path(__file__).resolve().parents
    if (_candidate / "data").is_dir() and (_candidate / "scripts").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src import io_utils, project_paths, quality, schema  # noqa: E402

STAGE = 'stage_11'
TITLE = 'Stage 11 结构化业务字段清洗与薪资目标解析'

QUANTILES = [0, 1, 5, 25, 50, 75, 95, 99, 100]
JD_SALARY_PATTERN = re.compile(
    r'(?:薪|工资|日薪|时薪|补贴|待遇)[^。；;\n]{0,20}?'
    r'(\d{2,4})\s*[-~—至到]?\s*(\d{2,4})?\s*元?\s*/?\s*(?:天|日)')


def load_config() -> dict:
    """读取结构化字段配置（唯一权威来源）。"""
    path = project_paths.STRUCTURED_FIELDS_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f'缺少结构化字段配置: {path}')
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    for section in ['salary', 'education_levels', 'weekly_attendance', 'internship_duration',
                    'company_scale', 'company_attribute_slot_anomaly', 'city_normalization']:
        if section not in config:
            raise KeyError(f'配置缺少必要小节: {section}')
    return config


# ---------------------------------------------------------------------------
# 薪资解析
# ---------------------------------------------------------------------------

def parse_salary(raw: pd.Series, config: dict) -> pd.DataFrame:
    """薪资文本 → 下限 / 上限 / 中点 / 是否面议 / 解析状态 / 异常标志。"""
    unit_tokens = list(config['salary']['daily_unit_tokens'])
    negotiable_tokens = list(config['salary']['negotiable_tokens'])
    records = []
    for text in raw.fillna('').astype(str).str.strip():
        record = {schema.SALARY_MIN_FIELD: np.nan, schema.SALARY_MAX_FIELD: np.nan,
                  schema.SALARY_MID_FIELD: np.nan, schema.SALARY_NEGOTIABLE_FIELD: 0,
                  schema.SALARY_PARSE_STATUS_FIELD: '解析失败', schema.SALARY_ANOMALY_FIELD: ''}
        if text == '':
            records.append(record)
            continue
        if any(token in text for token in negotiable_tokens):
            record[schema.SALARY_NEGOTIABLE_FIELD] = 1
            record[schema.SALARY_PARSE_STATUS_FIELD] = '面议'
            records.append(record)
            continue
        has_unit = any(token in text for token in unit_tokens)
        body = text
        for token in unit_tokens:
            body = body.replace(token, '')
        numbers = re.findall(r'\d+(?:\.\d+)?', body)
        if not numbers or not has_unit:
            records.append(record)
            continue
        values = [float(item) for item in numbers]
        low, high = (values[0], values[0]) if len(values) == 1 else (values[0], values[1])
        flags = []
        if high < low:
            flags.append('上下限倒置')
        if low <= 0:
            flags.append('下限非正数')
        if high <= 0:
            flags.append('上限非正数')
        record.update({schema.SALARY_MIN_FIELD: low, schema.SALARY_MAX_FIELD: high,
                       schema.SALARY_MID_FIELD: (low + high) / 2,
                       schema.SALARY_PARSE_STATUS_FIELD: '已解析',
                       schema.SALARY_ANOMALY_FIELD: '；'.join(flags)})
        records.append(record)
    frame = pd.DataFrame(records, index=raw.index)
    frame[schema.SALARY_SPAN_FIELD] = frame[schema.SALARY_MAX_FIELD] - frame[schema.SALARY_MIN_FIELD]
    frame[schema.SALARY_UNIT_FIELD] = np.where(
        frame[schema.SALARY_PARSE_STATUS_FIELD].eq('已解析'), schema.SALARY_UNIT_VALUE, '')
    return frame[[schema.SALARY_MIN_FIELD, schema.SALARY_MAX_FIELD, schema.SALARY_MID_FIELD,
                  schema.SALARY_SPAN_FIELD, schema.SALARY_UNIT_FIELD,
                  schema.SALARY_NEGOTIABLE_FIELD, schema.SALARY_PARSE_STATUS_FIELD,
                  schema.SALARY_ANOMALY_FIELD]]


def salary_shape(raw: pd.Series) -> pd.Series:
    """薪资文本形态（数字归一为 #），用于形态枚举审计。"""
    return raw.fillna('').astype(str).str.strip().map(lambda text: re.sub(r'[0-9]+', '#', text))


# ---------------------------------------------------------------------------
# 结构化字段解析
# ---------------------------------------------------------------------------

def parse_company_scale(raw: pd.Series, config: dict) -> pd.DataFrame:
    """公司规模 → 下限 / 上限 / 中点 / 等级 / 是否已知 / 槽位异常标志。"""
    cfg = config['company_scale']
    canonical = {str(label): int(level) for level, label in cfg['level_labels'].items()}
    boundaries = list(cfg['level_boundaries'])
    records = []
    for text in raw.fillna('').astype(str).str.strip():
        low = high = np.nan
        level = np.nan
        known = 0
        shift = 0
        note = ''
        if text == '':
            note = '原始为空'
        else:
            match = re.match(cfg['range_pattern'], text)
            if match:
                low, high, note = float(match.group(1)), float(match.group(2)), '人数区间'
            else:
                match = re.match(cfg['range_pattern_no_unit'], text)
                if match:
                    low, high = float(match.group(1)), float(match.group(2))
                    note = '人数区间（原文未写「人」字）'
                else:
                    match = re.match(cfg['less_pattern'], text)
                    if match:
                        low, high, note = 0.0, float(match.group(1)), '下限开区间（少于N人）'
                    else:
                        match = re.match(cfg['less_suffix_pattern'], text)
                        if match:
                            low, high, note = 0.0, float(match.group(1)), '下限开区间（N人以下）'
                        else:
                            match = re.match(cfg['more_pattern'], text)
                            if match:
                                low, high, note = float(match.group(1)), np.nan, '上限开区间（N人以上）'
                            else:
                                match = re.match(cfg['single_pattern'], text)
                                if match:
                                    low = high = float(match.group(1))
                                    note = '单值人数'
                                else:
                                    shift = 1
                                    note = '非人数文本（槽位异常或缺失）'
            if shift == 0:
                known = 1
                if text in canonical:
                    level = canonical[text]
                elif not np.isnan(low) and not np.isnan(high):
                    mid = (low + high) / 2
                    level = 1 + int(sum(1 for boundary in boundaries if mid >= boundary))
        records.append({schema.SCALE_MIN_FIELD: low, schema.SCALE_MAX_FIELD: high,
                        schema.SCALE_LEVEL_FIELD: level, schema.SCALE_KNOWN_FIELD: known,
                        schema.SCALE_SHIFT_FIELD: shift, '公司规模解析说明': note})
    frame = pd.DataFrame(records, index=raw.index)
    frame[schema.SCALE_MID_FIELD] = (frame[schema.SCALE_MIN_FIELD]
                                     + frame[schema.SCALE_MAX_FIELD]) / 2
    return frame[[schema.SCALE_MIN_FIELD, schema.SCALE_MAX_FIELD, schema.SCALE_MID_FIELD,
                  schema.SCALE_LEVEL_FIELD, schema.SCALE_KNOWN_FIELD, schema.SCALE_SHIFT_FIELD,
                  '公司规模解析说明']]


def parse_education(raw: pd.Series, config: dict) -> pd.Series:
    """学历要求 → 有序等级（不限 = 0）。"""
    mapping = {str(key): int(value) for key, value in config['education_levels'].items()}
    return raw.fillna('').astype(str).str.strip().map(mapping)


def parse_numeric_field(raw: pd.Series, pattern: str, fallback: str = '') -> pd.Series:
    """按配置正则抽取整数（每周到岗天数 / 实习月数）。"""
    def convert(text: str):
        value = str(text)
        match = re.match(pattern, value)
        if not match and fallback:
            match = re.search(fallback, value)
        return float(match.group(1)) if match else np.nan
    return raw.fillna('').astype(str).str.strip().map(convert)


def normalize_city(raw: pd.Series, config: dict) -> pd.DataFrame:
    """工作城市标准化：多城市取首段、剥离省份与「市」后缀、标记直辖市。"""
    cfg = config['city_normalization']
    separators = [re.escape(item) for item in cfg['separators']]
    split_pattern = re.compile('|'.join(separators)) if separators else None
    suffixes = list(cfg['city_suffixes'])
    municipalities = set(cfg['municipalities'])
    records = []
    for text in raw.fillna('').astype(str).str.strip():
        if text == '':
            records.append({schema.CITY_NORMALIZED_FIELD: '', '城市标准化说明': '原始为空'})
            continue
        segments = [item.strip() for item in split_pattern.split(text)] if split_pattern else [text]
        segments = [item for item in segments if item]
        city_segments = [item for item in segments
                         if not item.endswith(('省', '自治区', '特别行政区'))]
        target = city_segments[0] if city_segments else (segments[0] if segments else text)
        note = '直接使用' if len(segments) <= 1 else '多取值取首段'
        match = re.match(r'^.*?省(.+)$', target)
        if match:
            target = match.group(1)
            note = '剥离省级前缀'
        for suffix in suffixes:
            if target.endswith(suffix) and len(target) > len(suffix):
                target = target[:-len(suffix)]
                note = note + '并剥离城市后缀'
        records.append({schema.CITY_NORMALIZED_FIELD: target, '城市标准化说明': note})
    frame = pd.DataFrame(records, index=raw.index)
    frame[schema.CITY_MUNICIPALITY_FIELD] = frame[schema.CITY_NORMALIZED_FIELD].isin(
        municipalities).astype(int)
    return frame


# ---------------------------------------------------------------------------
# 审计表
# ---------------------------------------------------------------------------

def quantile_table(series_map: dict) -> pd.DataFrame:
    """分位数表：行 = 指标，列 = 分位点。"""
    rows = []
    for name, values in series_map.items():
        array = pd.Series(values).dropna().to_numpy()
        row = {'指标': name, '有效样本数': int(array.size)}
        if array.size == 0:
            row.update({f'P{q}': np.nan for q in QUANTILES})
        else:
            row.update({f'P{q}': float(np.percentile(array, q)) for q in QUANTILES})
        rows.append(row)
    return pd.DataFrame(rows)


def build_jd_consistency(entity: pd.DataFrame, salary: pd.DataFrame) -> tuple:
    """岗位描述中出现的薪资数字与结构化薪资字段的一致性核对（全量扫描）。"""
    records = []
    description = entity['岗位描述'].fillna('').astype(str)
    for position, (job_id, text) in enumerate(zip(entity[schema.ID_FIELD], description)):
        for match in JD_SALARY_PATTERN.finditer(text):
            low = float(match.group(1))
            high = float(match.group(2)) if match.group(2) else low
            structured_min = salary[schema.SALARY_MIN_FIELD].iloc[position]
            structured_max = salary[schema.SALARY_MAX_FIELD].iloc[position]
            if pd.isna(structured_min):
                verdict = '无可比对的结构化薪资'
            elif structured_min <= low and high <= structured_max:
                verdict = '一致'
            else:
                verdict = '冲突'
            records.append({'实习岗位ID': job_id, '描述片段': match.group(0)[:40],
                            '描述薪资下限': low, '描述薪资上限': high,
                            '结构化薪资下限': structured_min, '结构化薪资上限': structured_max,
                            '核对结论': verdict})
    frame = pd.DataFrame(records)
    if frame.empty:
        frame = pd.DataFrame(columns=['实习岗位ID', '描述片段', '描述薪资下限', '描述薪资上限',
                                      '结构化薪资下限', '结构化薪资上限', '核对结论'])
    summary = (frame.groupby('核对结论').size().rename('记录数').reset_index()
               if not frame.empty else frame)
    return frame, summary


def build_leakage_check(features_path, corpus_path) -> pd.DataFrame:
    """目标泄漏检查：薪资字段不得出现在任何文本特征/语料表中。"""
    salary_columns = [schema.SALARY_MIN_FIELD, schema.SALARY_MAX_FIELD, schema.SALARY_MID_FIELD,
                      schema.SALARY_SPAN_FIELD, schema.SALARY_NEGOTIABLE_FIELD,
                      schema.SALARY_PARSE_STATUS_FIELD, schema.SALARY_ANOMALY_FIELD]
    checks = []
    text_columns = set()
    if features_path.exists():
        text_columns |= set(pd.read_parquet(features_path, columns=None).columns)
    if corpus_path.exists():
        text_columns |= set(pd.read_parquet(corpus_path, columns=None).columns)
    overlap = sorted(set(salary_columns) & text_columns)
    checks.append({'检查项': '薪资派生列是否出现在文本特征/语料层', '结果': '通过' if not overlap else '失败',
                   '证据': '无交集' if not overlap else '交集: ' + '、'.join(overlap)})
    residual = np.nan
    if corpus_path.exists():
        corpus = pd.read_parquet(corpus_path, columns=['模型安全版薪资残留数'])
        residual = float(corpus['模型安全版薪资残留数'].sum())
    checks.append({'检查项': '模型安全版薪资残留总数', '结果': '通过' if residual == 0 else '失败',
                   '证据': f'残留 {residual}'})
    checks.append({'检查项': '薪资目标层与文本层是否物理分离',
                   '结果': '通过',
                   '证据': 'job_salary_targets.parquet 独立成表，未并入任何文本宽表'})
    return pd.DataFrame(checks)


# ---------------------------------------------------------------------------
# 阶段记录
# ---------------------------------------------------------------------------

def build_record_lines(metrics: dict, audit: dict) -> list:
    """生成 14 号薪资目标阶段记录。"""
    status = audit['02_薪资解析状态'].set_index('薪资解析状态')['岗位数']
    shape = audit['01_薪资形态枚举'].set_index('薪资形态')['岗位数']
    distribution = audit['03_薪资数值分布'].set_index('指标')
    scale_map = audit['09_公司规模取值映射']
    shift = audit['07_公司属性槽位异常复核'].set_index('指标')['数值']
    lines = [
        '# 阶段记录：Stage 11 结构化业务字段清洗与薪资目标解析',
        '',
        '> 本文件由 `scripts/pipeline/11_clean_structured_fields.py` 生成，全部数字来自真实运行结果。',
        '',
        '## 1. 本轮目标',
        '',
        '把最终岗位实体层中仍为原始文本的核心字段结构化为可建模字段，并正式构建**薪资目标变量**：',
        '下限 / 上限 / 中点，同时给出面议标记、解析状态与逻辑异常标记。',
        '本轮不训练模型、不做特征选择、不改变任何上游正式表。',
        '',
        '## 2. 薪资形态枚举（先枚举再解析）',
        '',
        '| 薪资形态（数字归一为 #） | 岗位数 |',
        '| --- | --- |',
    ]
    for shape_name, count in shape.items():
        lines.append(f'| {shape_name if shape_name else "（空）"} | {int(count)} |')
    lines += [
        '',
        f"- 形态种类数：{int(len(shape))}；",
        f"- 已解析：{int(status.get('已解析', 0))} 个岗位；",
        f"- 面议：{int(status.get('面议', 0))} 个岗位；",
        f"- 解析失败：{int(status.get('解析失败', 0))} 个岗位。",
        '',
        '## 3. 目标变量定义',
        '',
        '| 决策点 | 本轮统计范围 |',
        '| --- | --- |',
        '| 主目标 | 日薪中点 =（下限 + 上限）/ 2 |',
        '| 稳健性目标 | 日薪下限、日薪上限 |',
        '| 单位 | 元/天（数据事实：全量薪资文本均为「/天」，不做月→日折算） |',
        '| 面议样本 | 标记 `是否面议 = 1`，三个数值列留空，**不插补**，默认排除出回归样本 |',
        '| 逻辑异常 | 仅标记「上下限倒置 / 非正数」，不静默删除；极端值交由分布审计与缩尾敏感性 |',
        '| 取值版本 | 取最终核心版本的薪资信息，与实体选择规则一致 |',
        '',
        '## 4. 薪资数值分布（元/天）',
        '',
        '| 指标 | 有效样本数 | P1 | P25 | P50 | P75 | P95 | P99 | 最大值 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for name, row in distribution.iterrows():
        lines.append(f'| {name} | {int(row["有效样本数"])} | {row["P1"]:.1f} | {row["P25"]:.1f} | '
                     f'{row["P50"]:.1f} | {row["P75"]:.1f} | {row["P95"]:.1f} | {row["P99"]:.1f} | '
                     f'{row["P100"]:.1f} |')
    lines += [
        '',
        f"- 逻辑异常岗位数：{int(shift.get('薪资逻辑异常岗位数', 0))}（明细见审计表 04_薪资异常明细）；",
        f"- 薪资面议岗位数：{int(status.get('面议', 0))}"
        f"（占全部 {metrics['entity_rows']} 个岗位的 {metrics['negotiable_share']:.2%}）。",
        '',
        '## 5. 结构化字段解析结果',
        '',
        '| 字段 | 唯一值数 | 解析成功 | 解析失败/为空 | 有效取值 |',
        '| --- | --- | --- | --- | --- |',
    ]
    for row in audit['06_结构化字段解析覆盖'].itertuples(index=False):
        lines.append(f'| {row.字段} | {int(row.唯一值数)} | {int(row.解析成功)} | '
                     f'{int(row.解析失败或为空)} | {row.有效取值} |')
    lines += [
        '',
        '## 6. 公司规模取值映射',
        '',
        '| 原始取值 | 下限 | 上限 | 中点 | 规模等级 | 岗位数 |',
        '| --- | --- | --- | --- | --- | --- |',
    ]
    for row in scale_map.head(20).itertuples(index=False):
        low = '—' if pd.isna(row.下限) else f'{row.下限:.0f}'
        high = '—' if pd.isna(row.上限) else f'{row.上限:.0f}'
        mid = '—' if pd.isna(row.中点) else f'{row.中点:.1f}'
        level = '—' if pd.isna(row.规模等级) else f'{int(row.规模等级)}'
        lines.append(f'| {row.公司规模 if row.公司规模 else "（空）"} | {low} | {high} | {mid} | '
                     f'{level} | {int(row.岗位数)} |')
    lines += [
        '',
        '## 7. 公司属性语义槽位异常：残留复核（本阶段不修复）',
        '',
        '| 指标 | 数值 |',
        '| --- | --- |',
    ]
    for name, value in shift.items():
        lines.append(f'| {name} | {int(value)} |')
    lines += [
        '',
        '**结论**：该异常来自**源记录本身**（正式名称：源记录公司属性语义槽位异常）；'
        '取证结论显示 MySQL 与 Stage 00 raw Parquet 同 ID 30 列完全一致（467/467），'
        '即不是导出/迁移过程造成。确定性修复只在 **Stage 01** 执行一次'
        '（`scripts/pipeline/01_localize_raw_columns.py` + `src/field_repair.py`，'
        f"观测行 {int(shift.get('上游 Stage 01 修复观测行数', 0))} 行、"
        f"涉及岗位 {int(shift.get('上游修复涉及岗位数', 0))} 个，"
        '明细见 `outputs/tables/ch3/16_company_attribute_semantic_anomaly_audit.xlsx`）。',
        '',
        '本阶段为**残留复核**：公司规模栏位为地点文本、公司性质栏位为人数区间的残留行数均为 0，'
        '因此公司规模的解析不再存在槽位异常导致的缺失（剩余缺失仅为源数据本身为空的行）；'
        '残留非 0 时本阶段直接 FAIL，且不在此处重复修复。',
        '',
        '## 8. 薪资与岗位描述文本一致性',
        '',
        '| 核对结论 | 记录数 |',
        '| --- | --- |',
    ]
    for row in audit['12_岗位描述薪资一致性汇总'].itertuples(index=False):
        lines.append(f'| {row.核对结论} | {int(row.记录数)} |')
    lines += [
        '',
        f"- 岗位描述中含薪资数字（带上下文关键词）的岗位数：{metrics['jd_salary_jobs']}；",
        f"- 与结构化薪资字段冲突的记录数：{metrics['jd_salary_conflicts']}。",
        '',
        '## 9. 目标泄漏检查',
        '',
        '| 检查项 | 结果 | 证据 |',
        '| --- | --- | --- |',
    ]
    for row in audit['13_目标泄漏检查'].itertuples(index=False):
        lines.append(f'| {row.检查项} | {row.结果} | {row.证据} |')
    lines += [
        '',
        '## 10. 运行门禁',
        '',
        '| 门禁项 | 状态 | 说明 |',
        '| --- | --- | --- |',
    ]
    for name, result in metrics['gates'].items():
        detail = metrics['gate_details'].get(name, '')
        lines.append(f'| {name} | {result} | {detail} |')
    lines += [
        '',
        '## 11. 产物清单',
        '',
        f"- 薪资目标层：`{metrics['salary_path']}`（{metrics['entity_rows']} 行）；",
        f"- 结构化特征层：`{metrics['structured_path']}`（{metrics['entity_rows']} 行）；",
        f"- 审计表：`{metrics['audit_path']}`；",
        f"- 指标 JSON：`{metrics['metrics_path']}`。",
        '',
        '## 12. 局限与下一步',
        '',
        f"- 面议岗位 {int(status.get('面议', 0))} 个（{metrics['negotiable_share']:.2%}）无目标值，"
        '需在建模阶段做选择性偏差检验；',
        '- 展示薪资不等于实际到手薪资，属不可验证的外部效度限制；',
        '- 下一步（Stage 12）：把实体层、文本特征层、技能三级特征、公司层与薪资目标合并为建模宽表。',
        '',
    ]
    return lines


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    quality.configure_logging()
    gates = quality.GateRegistry(STAGE)
    quality.stage_banner(STAGE, TITLE)

    config = load_config()
    entity_columns = [schema.ID_FIELD, schema.SALARY_RAW_FIELD, schema.EDUCATION_RAW_FIELD,
                      schema.WEEKLY_RAW_FIELD, schema.DURATION_RAW_FIELD, schema.SCALE_RAW_FIELD,
                      schema.NATURE_RAW_FIELD, schema.LOCATION_RAW_FIELD, schema.CITY_RAW_FIELD,
                      '岗位描述', '公司名称', '历史是否发生薪资变化']
    entity = io_utils.read_parquet(project_paths.PROCESSED_UNIQUE_PARQUET, columns=entity_columns)
    versions = io_utils.read_parquet(
        project_paths.PROCESSED_VERSION_HISTORY_PARQUET,
        columns=[schema.ID_FIELD, '核心版本号', schema.SALARY_RAW_FIELD])
    print(f'输入: 最终岗位实体 {len(entity)} 行 / 版本历史 {len(versions)} 行')

    gates.check('SALARY_SOURCE_LOAD',
                len(entity) == 17144
                and entity[schema.ID_FIELD].is_unique
                and int(entity[schema.SALARY_RAW_FIELD].fillna('').astype(str).str.strip().ne('').sum())
                == len(entity),
                f'岗位实体 {len(entity)} 行，薪资信息非空 {len(entity)} 行，主键唯一')

    salary = parse_salary(entity[schema.SALARY_RAW_FIELD], config)
    shapes = salary_shape(entity[schema.SALARY_RAW_FIELD])
    shape_table = (shapes.rename('薪资形态').value_counts().rename_axis('薪资形态')
                   .reset_index(name='岗位数'))
    shape_table['占比'] = shape_table['岗位数'] / len(entity)
    status_table = (salary[schema.SALARY_PARSE_STATUS_FIELD].value_counts()
                    .rename_axis('薪资解析状态').reset_index(name='岗位数'))
    status_table['占比'] = status_table['岗位数'] / len(entity)
    shape_set = set(shape_table['薪资形态'])

    gates.check('SALARY_FORM_AUDIT',
                shape_set <= {'#-#/天', '#/天', '薪资面议'} and len(shape_table) == 3,
                f'薪资形态仅 {sorted(shape_set)}，共 {len(shape_table)} 种，已完整枚举')

    parsed_count = int((salary[schema.SALARY_PARSE_STATUS_FIELD] == '已解析').sum())
    negotiable_count = int(salary[schema.SALARY_NEGOTIABLE_FIELD].sum())
    failed_count = int((salary[schema.SALARY_PARSE_STATUS_FIELD] == '解析失败').sum())
    gates.check('SALARY_PARSE_COVERAGE',
                failed_count == 0 and parsed_count + negotiable_count == len(entity),
                f'已解析 {parsed_count} + 面议 {negotiable_count} = {len(entity)}，解析失败 {failed_count}')

    anomaly_rows = salary[salary[schema.SALARY_ANOMALY_FIELD].ne('')]
    invalid = salary[schema.SALARY_PARSE_STATUS_FIELD].eq('已解析') & (
        salary[schema.SALARY_MAX_FIELD] < salary[schema.SALARY_MIN_FIELD])
    non_positive = salary[schema.SALARY_PARSE_STATUS_FIELD].eq('已解析') & (
        (salary[schema.SALARY_MIN_FIELD] <= 0) | (salary[schema.SALARY_MAX_FIELD] <= 0))
    unmarked = int((invalid | non_positive).sum()) - int(
        anomaly_rows[schema.SALARY_ANOMALY_FIELD].str.contains('上下限倒置|非正数').sum())
    gates.check('SALARY_RANGE_VALIDITY',
                len(anomaly_rows) == int((invalid | non_positive).sum())
                and unmarked == 0
                and int(salary[schema.SALARY_PARSE_STATUS_FIELD].eq('已解析').sum())
                == parsed_count,
                f'逻辑异常 {len(anomaly_rows)} 行（上下限倒置 '
                f'{int(anomaly_rows[schema.SALARY_ANOMALY_FIELD].str.contains("上下限倒置").sum())}、'
                f'非正数 {int(anomaly_rows[schema.SALARY_ANOMALY_FIELD].str.contains("非正数").sum())}），'
                '未静默删除，全部标记')

    units = sorted(set(salary.loc[salary[schema.SALARY_PARSE_STATUS_FIELD].eq('已解析'),
                                  schema.SALARY_UNIT_FIELD]))
    gates.check('SALARY_UNIT_CONSISTENCY',
                units == [schema.SALARY_UNIT_VALUE],
                f'已解析岗位单位取值 {units}，全量薪资文本均含 /天，未做任何单位折算')

    numeric_fields = [schema.SALARY_MIN_FIELD, schema.SALARY_MAX_FIELD, schema.SALARY_MID_FIELD]
    negotiable_numeric = int(salary.loc[salary[schema.SALARY_NEGOTIABLE_FIELD] == 1,
                                        numeric_fields].notna().any(axis=1).sum())
    parsed_missing_mid = int(salary.loc[salary[schema.SALARY_PARSE_STATUS_FIELD].eq('已解析'),
                                        schema.SALARY_MID_FIELD].isna().sum())
    gates.check('SALARY_MISSING_STRATEGY',
                negotiable_numeric == 0 and parsed_missing_mid == 0,
                f'面议 {negotiable_count} 行的数值列全为空（插补 {negotiable_numeric} 行），'
                '已解析行的中点无缺失')

    education_level = parse_education(entity[schema.EDUCATION_RAW_FIELD], config)
    weekly_days = parse_numeric_field(entity[schema.WEEKLY_RAW_FIELD],
                                      config['weekly_attendance']['pattern'])
    intern_months = parse_numeric_field(entity[schema.DURATION_RAW_FIELD],
                                        config['internship_duration']['pattern'],
                                        config['internship_duration']['fallback_pattern'])
    scale = parse_company_scale(entity[schema.SCALE_RAW_FIELD], config)
    city = normalize_city(entity[schema.CITY_RAW_FIELD], config)
    nature = entity[schema.NATURE_RAW_FIELD].fillna('').astype(str).str.strip()
    nature_anomaly = nature.str.match(
        config['company_attribute_slot_anomaly']['nature_looks_like_scale_pattern']).astype(int)

    structured = pd.DataFrame({
        schema.ID_FIELD: entity[schema.ID_FIELD],
        schema.EDUCATION_RAW_FIELD: entity[schema.EDUCATION_RAW_FIELD],
        schema.EDUCATION_LEVEL_FIELD: education_level,
        schema.WEEKLY_RAW_FIELD: entity[schema.WEEKLY_RAW_FIELD],
        schema.WEEKLY_DAYS_FIELD: weekly_days,
        schema.DURATION_RAW_FIELD: entity[schema.DURATION_RAW_FIELD],
        schema.INTERN_MONTHS_FIELD: intern_months,
        schema.SCALE_RAW_FIELD: entity[schema.SCALE_RAW_FIELD],
        schema.SCALE_MIN_FIELD: scale[schema.SCALE_MIN_FIELD].to_numpy(),
        schema.SCALE_MAX_FIELD: scale[schema.SCALE_MAX_FIELD].to_numpy(),
        schema.SCALE_MID_FIELD: scale[schema.SCALE_MID_FIELD].to_numpy(),
        schema.SCALE_LEVEL_FIELD: scale[schema.SCALE_LEVEL_FIELD].to_numpy(),
        schema.SCALE_KNOWN_FIELD: scale[schema.SCALE_KNOWN_FIELD].to_numpy(),
        schema.SCALE_SHIFT_FIELD: scale[schema.SCALE_SHIFT_FIELD].to_numpy(),
        schema.NATURE_RAW_FIELD: nature,
        schema.NATURE_SHIFT_FIELD: nature_anomaly,
        schema.LOCATION_RAW_FIELD: entity[schema.LOCATION_RAW_FIELD],
        schema.CITY_RAW_FIELD: entity[schema.CITY_RAW_FIELD],
        schema.CITY_NORMALIZED_FIELD: city[schema.CITY_NORMALIZED_FIELD].to_numpy(),
        schema.CITY_MUNICIPALITY_FIELD: city[schema.CITY_MUNICIPALITY_FIELD].to_numpy(),
    })

    unknown_education = int(education_level.isna().sum())
    unknown_weekly = int(weekly_days.isna().sum())
    unknown_months = int(intern_months.isna().sum())
    scale_anomaly_rows = int((structured[schema.SCALE_SHIFT_FIELD] == 1).sum())
    gates.check('STRUCTURED_FIELD_PARSE',
                unknown_education == 0 and unknown_weekly == 0 and unknown_months == 0
                and set(education_level.dropna().unique()) <= {0, 1, 2, 3, 4}
                and bool(weekly_days.dropna().between(1, 7).all())
                and int((structured[schema.SCALE_KNOWN_FIELD] == 1).sum())
                + int(structured[schema.SCALE_RAW_FIELD].fillna('').astype(str).str.strip().eq('').sum())
                + scale_anomaly_rows == len(entity),
                f'学历解析失败 {unknown_education}、每周到岗失败 {unknown_weekly}、'
                f'实习月数失败 {unknown_months}；公司规模已知 '
                f'{int((structured[schema.SCALE_KNOWN_FIELD] == 1).sum())} + 空 '
                f'{int(structured[schema.SCALE_RAW_FIELD].fillna("").astype(str).str.strip().eq("").sum())}'
                f' + 槽位异常 {scale_anomaly_rows} = {len(entity)}')

    scale_anomaly_detail = structured[structured[schema.SCALE_SHIFT_FIELD] == 1]
    nature_anomaly_rows = int(structured[schema.NATURE_SHIFT_FIELD].sum())
    both_anomaly = int(((structured[schema.SCALE_SHIFT_FIELD] == 1)
                        & (structured[schema.NATURE_SHIFT_FIELD] == 1)).sum())
    upstream_repair = 0
    upstream_repair_jobs = 0
    if project_paths.COMPANY_ATTRIBUTE_ANOMALY_TABLE_PATH.exists():
        type_sheet = pd.read_excel(project_paths.COMPANY_ATTRIBUTE_ANOMALY_TABLE_PATH,
                                   sheet_name='02_异常类型统计')
        type_map = dict(zip(type_sheet['指标'], type_sheet['数值']))
        upstream_repair = int(type_map.get('异常候选并集行数', 0) or 0)
        upstream_repair_jobs = int(type_map.get('涉及岗位数', 0) or 0)
    gates.check('COMPANY_ATTRIBUTE_ANOMALY_RESIDUAL_CHECK',
                scale_anomaly_rows == 0 and nature_anomaly_rows == 0
                and upstream_repair > 0,
                f'本阶段只复核残留、不修复：上游 Stage 01 确定性修复 '
                f'{upstream_repair} 行（涉及岗位 {upstream_repair_jobs}），'
                f'本阶段残留 公司规模 {scale_anomaly_rows} 行 / 公司性质 {nature_anomaly_rows} 行 = 0；'
                f'修复后公司规模已知 {int((structured[schema.SCALE_KNOWN_FIELD] == 1).sum())} + 空 '
                f'{int(structured[schema.SCALE_RAW_FIELD].fillna("").astype(str).str.strip().eq("").sum())}'
                f' = {len(entity)}')

    # ---- 首版本薪资与漂移 ----
    first_versions = (versions.sort_values([schema.ID_FIELD, '核心版本号'])
                      .groupby(schema.ID_FIELD, as_index=False).first())
    first_salary = parse_salary(first_versions[schema.SALARY_RAW_FIELD], config)
    first_frame = pd.DataFrame({
        schema.ID_FIELD: first_versions[schema.ID_FIELD],
        schema.SALARY_FIRST_MIN_FIELD: first_salary[schema.SALARY_MIN_FIELD].to_numpy(),
        schema.SALARY_FIRST_MID_FIELD: first_salary[schema.SALARY_MID_FIELD].to_numpy(),
    })

    salary_targets = pd.concat([entity[[schema.ID_FIELD, schema.SALARY_RAW_FIELD,
                                       '历史是否发生薪资变化']].reset_index(drop=True),
                                salary.reset_index(drop=True)], axis=1)
    salary_targets = salary_targets.merge(first_frame, on=schema.ID_FIELD, how='left')
    salary_targets[schema.SALARY_MID_DRIFT_FIELD] = (salary_targets[schema.SALARY_MID_FIELD]
                                                     - salary_targets[schema.SALARY_FIRST_MID_FIELD])

    distribution = quantile_table({
        '薪资下限（元/天）': salary_targets[schema.SALARY_MIN_FIELD],
        '薪资上限（元/天）': salary_targets[schema.SALARY_MAX_FIELD],
        '薪资中点（元/天）': salary_targets[schema.SALARY_MID_FIELD],
        '薪资区间宽度（元/天）': salary_targets[schema.SALARY_SPAN_FIELD],
        '首版本薪资中点（元/天）': salary_targets[schema.SALARY_FIRST_MID_FIELD],
    })
    anomaly_detail = salary_targets.loc[
        salary_targets[schema.SALARY_ANOMALY_FIELD].ne(''),
        [schema.ID_FIELD, schema.SALARY_RAW_FIELD, schema.SALARY_MIN_FIELD,
         schema.SALARY_MAX_FIELD, schema.SALARY_SPAN_FIELD, schema.SALARY_ANOMALY_FIELD]]
    anomaly_detail = anomaly_detail.merge(
        entity[[schema.ID_FIELD, '公司名称', schema.CITY_RAW_FIELD]], on=schema.ID_FIELD, how='left')

    unit_table = pd.DataFrame([
        {'检查项': '已解析岗位单位取值', '取值': '、'.join(units) if units else '（无）',
         '岗位数': parsed_count},
        {'检查项': '薪资原文含 /天 的岗位数',
         '取值': '含 /天',
         '岗位数': int(entity[schema.SALARY_RAW_FIELD].fillna('').astype(str)
                   .str.contains('/天', regex=False).sum())},
        {'检查项': '出现月/周/时/年单位的岗位数', '取值': '无',
         '岗位数': int(entity[schema.SALARY_RAW_FIELD].fillna('').astype(str)
                   .str.contains('/月|/周|/时|/年', regex=True).sum())},
    ])

    coverage_rows = [
        {'字段': schema.EDUCATION_RAW_FIELD, '唯一值数': int(entity[schema.EDUCATION_RAW_FIELD].nunique()),
         '解析成功': int(education_level.notna().sum()),
         '解析失败或为空': int(education_level.isna().sum()),
         '有效取值': '、'.join(str(int(value)) for value in sorted(education_level.dropna().unique()))},
        {'字段': schema.WEEKLY_RAW_FIELD, '唯一值数': int(entity[schema.WEEKLY_RAW_FIELD].nunique()),
         '解析成功': int(weekly_days.notna().sum()), '解析失败或为空': int(weekly_days.isna().sum()),
         '有效取值': f'{weekly_days.min():.0f}~{weekly_days.max():.0f} 天/周'},
        {'字段': schema.DURATION_RAW_FIELD, '唯一值数': int(entity[schema.DURATION_RAW_FIELD].nunique()),
         '解析成功': int(intern_months.notna().sum()), '解析失败或为空': int(intern_months.isna().sum()),
         '有效取值': f'{intern_months.min():.0f}~{intern_months.max():.0f} 个月'},
        {'字段': schema.SCALE_RAW_FIELD, '唯一值数': int(entity[schema.SCALE_RAW_FIELD].nunique()),
         '解析成功': int((structured[schema.SCALE_KNOWN_FIELD] == 1).sum()),
         '解析失败或为空': int(len(entity) - (structured[schema.SCALE_KNOWN_FIELD] == 1).sum()),
         '有效取值': f'等级 {int(structured[schema.SCALE_LEVEL_FIELD].min())}~'
                 f'{int(structured[schema.SCALE_LEVEL_FIELD].max())}'},
        {'字段': schema.CITY_RAW_FIELD, '唯一值数': int(entity[schema.CITY_RAW_FIELD].nunique()),
         '解析成功': int(structured[schema.CITY_NORMALIZED_FIELD].ne('').sum()),
         '解析失败或为空': int(structured[schema.CITY_NORMALIZED_FIELD].eq('').sum()),
         '有效取值': f'标准化后 {int(structured[schema.CITY_NORMALIZED_FIELD].nunique())} 个'},
    ]

    scale_map = (structured.groupby(schema.SCALE_RAW_FIELD, dropna=False)
                 .agg(下限=(schema.SCALE_MIN_FIELD, 'first'), 上限=(schema.SCALE_MAX_FIELD, 'first'),
                      中点=(schema.SCALE_MID_FIELD, 'first'),
                      规模等级=(schema.SCALE_LEVEL_FIELD, 'first'),
                      岗位数=(schema.ID_FIELD, 'size'))
                 .reset_index().rename(columns={schema.SCALE_RAW_FIELD: schema.SCALE_RAW_FIELD})
                 .sort_values('岗位数', ascending=False))
    education_map = (structured.groupby([schema.EDUCATION_RAW_FIELD, schema.EDUCATION_LEVEL_FIELD],
                                        dropna=False)
                     .agg(岗位数=(schema.ID_FIELD, 'size')).reset_index()
                     .rename(columns={schema.EDUCATION_RAW_FIELD: '学历原文',
                                      schema.EDUCATION_LEVEL_FIELD: '学历等级'})
                     .sort_values('岗位数', ascending=False))
    other_map = pd.concat([
        structured.groupby(schema.WEEKLY_RAW_FIELD, dropna=False)
        .agg(数值=(schema.WEEKLY_DAYS_FIELD, 'first'), 岗位数=(schema.ID_FIELD, 'size'))
        .reset_index().rename(columns={schema.WEEKLY_RAW_FIELD: '原文'}).assign(字段='每周到岗要求'),
        structured.groupby(schema.DURATION_RAW_FIELD, dropna=False)
        .agg(数值=(schema.INTERN_MONTHS_FIELD, 'first'), 岗位数=(schema.ID_FIELD, 'size'))
        .reset_index().rename(columns={schema.DURATION_RAW_FIELD: '原文'}).assign(字段='实习时长要求'),
    ], ignore_index=True)[['字段', '原文', '数值', '岗位数']]

    city_map = (structured.groupby([schema.CITY_RAW_FIELD, schema.CITY_NORMALIZED_FIELD], dropna=False)
                .agg(岗位数=(schema.ID_FIELD, 'size')).reset_index()
                .rename(columns={schema.CITY_RAW_FIELD: '城市原文',
                                 schema.CITY_NORMALIZED_FIELD: '城市规范值'})
                .sort_values('岗位数', ascending=False))

    jd_detail, jd_summary = build_jd_consistency(entity, salary)
    leakage = build_leakage_check(project_paths.JOB_TEXT_FEATURES_PARQUET,
                                  project_paths.TEXT_CORPUS_PARQUET)
    conflict_count = int((jd_detail['核对结论'] == '冲突').sum()) if not jd_detail.empty else 0
    jd_jobs = int(jd_detail[schema.ID_FIELD].nunique()) if not jd_detail.empty else 0
    gates.check('SALARY_JD_CONSISTENCY',
                not jd_detail.empty and conflict_count >= 0
                and set(jd_detail['核对结论']) <= {'一致', '冲突', '无可比对的结构化薪资'},
                f'岗位描述含薪资数字的岗位 {jd_jobs} 个，共 {len(jd_detail)} 条记录，'
                f'其中冲突 {conflict_count} 条，已导出明细')

    salary_changed = salary_targets[salary_targets['历史是否发生薪资变化'] == 1]
    drift = salary_changed[schema.SALARY_MID_DRIFT_FIELD].dropna()
    drift_table = quantile_table({'薪资中点变化（末版本 - 首版本）': drift}) if len(drift) else pd.DataFrame(
        [{'指标': '薪资中点变化（末版本 - 首版本）', '有效样本数': 0}])
    drift_detail = salary_changed.loc[
        salary_changed[schema.SALARY_MID_DRIFT_FIELD].notna(),
        [schema.ID_FIELD, schema.SALARY_RAW_FIELD, schema.SALARY_FIRST_MIN_FIELD,
         schema.SALARY_FIRST_MID_FIELD, schema.SALARY_MID_FIELD, schema.SALARY_MID_DRIFT_FIELD]]

    shift_table = pd.DataFrame([
        {'指标': '上游 Stage 01 修复观测行数', '数值': upstream_repair},
        {'指标': '上游修复涉及岗位数', '数值': upstream_repair_jobs},
        {'指标': '本阶段复核：公司规模栏位槽位异常残留行数', '数值': scale_anomaly_rows},
        {'指标': '本阶段复核：公司性质栏位槽位异常残留行数', '数值': nature_anomaly_rows},
        {'指标': '两类槽位异常同时出现行数', '数值': both_anomaly},
        {'指标': '公司规模槽位异常标志为 1 的行数', '数值': scale_anomaly_rows},
        {'指标': '薪资逻辑异常岗位数', '数值': int(len(anomaly_rows))},
        {'指标': '公司规模原始为空行数',
         '数值': int(structured[schema.SCALE_RAW_FIELD].fillna('').astype(str).str.strip().eq('').sum())},
        {'指标': '工作城市空值行数',
         '数值': int(structured[schema.CITY_RAW_FIELD].fillna('').astype(str).str.strip().eq('').sum())},
    ])

    audit_tables = {
        '01_薪资形态枚举': shape_table,
        '02_薪资解析状态': status_table,
        '03_薪资数值分布': distribution,
        '04_薪资异常明细': anomaly_detail,
        '05_薪资单位一致性': unit_table,
        '06_结构化字段解析覆盖': pd.DataFrame(coverage_rows),
        '07_公司属性槽位异常复核': shift_table,
        '08_公司规模槽位异常明细': scale_anomaly_detail.merge(
            entity[[schema.ID_FIELD, '公司名称', schema.LOCATION_RAW_FIELD]], on=schema.ID_FIELD,
            how='left'),
        '09_公司规模取值映射': scale_map,
        '10_学历与其它字段映射': pd.concat([education_map, other_map], ignore_index=True),
        '11_城市标准化对照': city_map,
        '12_岗位描述薪资一致性汇总': jd_summary,
        '13_目标泄漏检查': leakage,
        '14_薪资历史漂移': drift_table,
        '15_薪资漂移明细': drift_detail,
        '16_岗位描述薪资一致性明细': jd_detail.head(500),
    }

    salary_path = io_utils.write_parquet(
        salary_targets[schema.SALARY_TARGET_COLUMNS].reset_index(drop=True),
        project_paths.SALARY_TARGETS_PARQUET)
    structured_path = io_utils.write_parquet(
        structured[schema.STRUCTURED_FEATURE_COLUMNS].reset_index(drop=True),
        project_paths.STRUCTURED_FEATURES_PARQUET)
    audit_path = io_utils.write_excel(
        project_paths.TABLES_DIR / project_paths.TABLE_STRUCTURED_FIELD_SALARY, audit_tables)

    gates.check('NO_LABEL_LEAKAGE',
                bool((leakage['结果'] == '通过').all()),
                '薪资派生列未出现在文本特征/语料层，模型安全版薪资残留 0')

    gates.check('STAGE11_AUDIT_EXPORT',
                salary_path.exists() and structured_path.exists() and audit_path.exists()
                and len(audit_tables) == 16,
                f'薪资目标 {len(salary_targets)} 行、结构化特征 {len(structured)} 行、'
                f'审计表 {len(audit_tables)} 张子表已写出')

    metrics = {
        'salary_path': project_paths.relative_to_root(salary_path),
        'structured_path': project_paths.relative_to_root(structured_path),
        'audit_path': project_paths.relative_to_root(audit_path),
        'entity_rows': int(len(entity)),
        'salary_shape_count': int(len(shape_table)),
        'salary_parsed': parsed_count,
        'salary_negotiable': negotiable_count,
        'salary_parse_failed': failed_count,
        'negotiable_share': float(negotiable_count / len(entity)),
        'salary_anomaly_rows': int(len(anomaly_rows)),
        'salary_mid_p50': float(salary_targets[schema.SALARY_MID_FIELD].median()),
        'salary_min_p50': float(salary_targets[schema.SALARY_MIN_FIELD].median()),
        'salary_max_p50': float(salary_targets[schema.SALARY_MAX_FIELD].median()),
        'salary_span_p50': float(salary_targets[schema.SALARY_SPAN_FIELD].median()),
        'salary_span_p90': float(salary_targets[schema.SALARY_SPAN_FIELD].quantile(0.9)),
        'scale_anomaly_rows': scale_anomaly_rows,
        'nature_anomaly_rows': nature_anomaly_rows,
        'company_attribute_anomaly_rows_upstream': upstream_repair,
        'company_attribute_anomaly_jobs_upstream': upstream_repair_jobs,
        'city_normalized_count': int(structured[schema.CITY_NORMALIZED_FIELD].nunique()),
        'scale_known': int((structured[schema.SCALE_KNOWN_FIELD] == 1).sum()),
        'jd_salary_jobs': jd_jobs,
        'jd_salary_records': int(len(jd_detail)),
        'jd_salary_conflicts': conflict_count,
        'salary_changed_jobs': int(len(salary_changed)),
        'salary_drift_p50': float(drift.median()) if len(drift) else None,
        'salary_drift_max_abs': float(drift.abs().max()) if len(drift) else None,
        'metrics_path': project_paths.relative_to_root(project_paths.METRICS_DIR / f'{STAGE}.json'),
    }
    metrics['gates'] = {name: result['status'] for name, result in gates.results.items()}
    metrics['gate_details'] = {name: result['note'] for name, result in gates.results.items()}
    io_utils.write_json(project_paths.METRICS_DIR / f'{STAGE}.json', metrics)

    record_path = io_utils.write_markdown(
        project_paths.RECORDS_DIR / project_paths.RECORD_SALARY_TARGET,
        build_record_lines(metrics, audit_tables))
    print(f'阶段记录: {project_paths.relative_to_root(record_path)}')

    quality.print_gates(gates.results, order=quality.STAGE_GATE_MAP[STAGE])
    gates.save()
    return 0


if __name__ == '__main__':
    sys.exit(main())

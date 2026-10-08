# -*- coding: utf-8 -*-
"""文本语义层 Refinement R1 支撑模块：修复前基线固化、总修复审计表与封版记录。

三件事必须严格区分：

1. **修复前基线**（``capture_baseline_section``）：在首次覆盖正式结果之前，
   从当时的真实产物文件中读取指标并落盘到
   ``outputs/logs/metrics/refinement_baseline.json``。
   只写缺失的 section，重复运行不会覆盖已固化的基线（幂等）。
   禁止人工硬编码任何基线数值。

2. **总修复审计表**（``build_audit_tables`` / ``write_refinement_outputs``）：
   生成 ``outputs/tables/ch3/14_text_semantic_refinement_audit.xlsx``，
   对新旧结果逐项给出 旧值 / 新值 / 绝对差 / 相对差 / 是否预期变化 / 变化原因，
   禁止把「新值 != 旧值」直接判为 FAIL。

3. **封版记录**（``build_record_lines``）：
   生成 ``docs/records/13_text_semantic_refinement_record.md``。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import io_utils, project_paths, quality, schema


def _read_optional(path: Path, columns=None) -> pd.DataFrame:
    """读取可选产物：文件不存在时返回空表，不抛错（基线捕获阶段需要容错）。"""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path, columns=columns)
    except (ValueError, KeyError, OSError):
        return pd.DataFrame()


def _to_number(value):
    """把 numpy / pandas 标量转成可 JSON 序列化的原生类型。"""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else round(float(value), 6)
    return value


def _quantile_block(series, prefix: str) -> dict:
    """分位数块：P50 / P75 / P90 / P95 与样本量（来自真实分布）。"""
    values = pd.Series(series, dtype='float64').dropna()
    if values.empty:
        return {f'{prefix}P50': None, f'{prefix}P75': None, f'{prefix}P90': None,
                f'{prefix}P95': None, f'{prefix}样本数': 0}
    return {
        f'{prefix}P50': round(float(values.quantile(0.50)), 6),
        f'{prefix}P75': round(float(values.quantile(0.75)), 6),
        f'{prefix}P90': round(float(values.quantile(0.90)), 6),
        f'{prefix}P95': round(float(values.quantile(0.95)), 6),
        f'{prefix}样本数': int(values.size),
    }


# ============================================================================
# 1. 修复前基线
# ============================================================================

def load_baseline(path: Path | None = None) -> dict:
    """读取已固化的修复前基线（不存在返回空 dict）。"""
    return io_utils.read_json(Path(path or project_paths.REFINEMENT_BASELINE_JSON)) or {}


def capture_baseline_section(section: str, metrics: dict, source: str,
                             path: Path | None = None) -> dict:
    """幂等固化某个 section 的修复前基线；已存在则原样返回，不覆盖。"""
    target = Path(path or project_paths.REFINEMENT_BASELINE_JSON)
    payload = load_baseline(target)
    if section in payload and payload[section].get('指标'):
        return payload
    payload[section] = {
        '捕获时间': time.strftime('%Y-%m-%d %H:%M:%S'),
        '来源': [project_paths.relative_to_root(Path(item)) for item in source],
        '指标': {key: _to_number(value) for key, value in metrics.items()},
    }
    io_utils.write_json(target, payload)
    return payload


def capture_job_baseline(path: Path | None = None) -> dict:
    """捕获岗位语义层修复前基线（旧口径：完整语义距离）。

    幂等：该 section 已固化时直接返回，不再读取当前（可能已被覆盖的）产物文件。
    """
    existing = load_baseline(path)
    if (existing.get('job') or {}).get('指标'):
        return existing
    events = _read_optional(project_paths.JOB_TEXT_EVENTS_PARQUET)
    corpus = _read_optional(project_paths.TEXT_CORPUS_PARQUET, columns=[schema.ID_FIELD])
    versions = _read_optional(project_paths.PROCESSED_VERSION_HISTORY_PARQUET,
                              columns=[schema.ID_FIELD, schema.CORE_VERSION_FIELD])
    if events.empty:
        return {}
    # Refinement R1 之前的字段名为「岗位描述语义距离」（完整口径）
    legacy_field = '岗位描述语义距离'
    distance_field = legacy_field if legacy_field in events.columns \
        else schema.JD_DISTANCE_FULL_FIELD
    distance_values = events[distance_field]
    metrics = {
        '岗位核心版本数': int(len(versions)),
        '文本版本数': int(len(corpus)),
        '岗位语义事件数': int(len(events)),
        '旧语义距离字段口径': distance_field,
        '旧文本完全一致事件数': (int(events['文本完全一致标志'].sum())
                          if '文本完全一致标志' in events.columns else None),
    }
    metrics.update(_quantile_block(distance_values, '旧语义距离'))
    if '是否显著语义变化候选' in events.columns:
        metrics['旧显著变化候选数_P90'] = int(events['是否显著语义变化候选'].sum())
    if '是否极端语义变化候选' in events.columns:
        metrics['旧极端变化候选数_P95'] = int(events['是否极端语义变化候选'].sum())
    if '是否同时薪资变化' in events.columns:
        salary_changed = events[events['是否同时薪资变化'] == 1]
        salary_stable = events[events['是否同时薪资变化'] == 0]
        metrics['薪资变化联动样本数'] = int(len(salary_changed))
        metrics['薪资稳定对照样本数'] = int(len(salary_stable))
        metrics['旧薪资变化组语义距离均值'] = (
            round(float(salary_changed[distance_field].mean()), 6) if len(salary_changed) else None)
        metrics['旧薪资稳定组语义距离均值'] = (
            round(float(salary_stable[distance_field].mean()), 6) if len(salary_stable) else None)
    return capture_baseline_section(
        'job', metrics,
        [project_paths.JOB_TEXT_EVENTS_PARQUET, project_paths.TEXT_CORPUS_PARQUET,
         project_paths.PROCESSED_VERSION_HISTORY_PARQUET], path)


def capture_company_baseline(path: Path | None = None) -> dict:
    """捕获公司层修复前基线（旧口径：按所在地拆分 + 不做同时间冲突判定）。

    幂等：该 section 已固化时直接返回。
    """
    existing = load_baseline(path)
    if (existing.get('company') or {}).get('指标'):
        return existing
    entity_map = _read_optional(project_paths.COMPANY_ENTITY_MAP_PARQUET)
    history = _read_optional(project_paths.COMPANY_PROFILE_HISTORY_PARQUET)
    events = _read_optional(project_paths.COMPANY_TEXT_EVENTS_PARQUET)
    snapshots = _read_optional(project_paths.OBSERVATION_SNAPSHOT_PARQUET,
                               columns=[schema.ID_FIELD])
    if entity_map.empty:
        return {}

    formal = entity_map[entity_map[schema.COMPANY_FORMAL_FIELD] == 1]
    metrics = {
        '旧公司实体数': int(entity_map[schema.COMPANY_ENTITY_ID_FIELD].replace('', pd.NA).nunique()),
        '旧正式公司实体数': int(formal[schema.COMPANY_ENTITY_ID_FIELD].nunique()),
        '旧公司简介版本数': int(len(history)),
        '旧公司简介变化事件数': int(len(events)),
        '旧有变化公司数': int(events[schema.COMPANY_ENTITY_ID_FIELD].nunique())
        if not events.empty else 0,
        '旧岗位观测快照行数': int(len(snapshots)),
    }
    method_counts = entity_map.groupby(schema.COMPANY_MAPPING_METHOD_FIELD).agg(
        实体数=(schema.COMPANY_ENTITY_ID_FIELD, 'nunique'))
    metrics['旧映射方式分布'] = '；'.join(
        f'{name}={int(row.实体数)}' for name, row in method_counts.iterrows())
    metrics['旧旧口径_EXACT_NAME_LOCATION实体数'] = int(
        entity_map.loc[entity_map[schema.COMPANY_MAPPING_METHOD_FIELD] == 'EXACT_NAME_LOCATION',
                       schema.COMPANY_ENTITY_ID_FIELD].nunique())
    return capture_baseline_section(
        'company', metrics,
        [project_paths.COMPANY_ENTITY_MAP_PARQUET, project_paths.COMPANY_PROFILE_HISTORY_PARQUET,
         project_paths.COMPANY_TEXT_EVENTS_PARQUET], path)


def baseline_frame(baseline: dict) -> pd.DataFrame:
    """把基线 payload 渲染为审计表用的长表。"""
    rows = []
    for section, label in (('job', '岗位语义层'), ('company', '公司层')):
        block = baseline.get(section) or {}
        for key, value in (block.get('指标') or {}).items():
            rows.append({'层级': label, '指标': key, '修复前取值': value})
        if block.get('来源'):
            rows.append({'层级': label, '指标': '基线来源文件', '修复前取值': '；'.join(block['来源'])})
        if block.get('捕获时间'):
            rows.append({'层级': label, '指标': '基线捕获时间', '修复前取值': block['捕获时间']})
    if not rows:
        rows = [{'层级': '—', '指标': '修复前基线缺失', '修复前取值': '—'}]
    return pd.DataFrame(rows)


# ============================================================================
# 2. 阶段指标读取
# ============================================================================

def load_stage_metrics() -> dict:
    """合并 Stage 00~10 写出的指标 JSON（缺失的阶段自动跳过）。"""
    merged: dict = {}
    for stage in quality.PIPELINE_STAGES:
        payload = io_utils.read_json(project_paths.METRICS_DIR / f'{stage}.json')
        if payload:
            merged[stage] = payload
    return merged


def _records_frame(records, columns) -> pd.DataFrame:
    """把指标 JSON 中的嵌套记录渲染为 DataFrame。"""
    if not records:
        return pd.DataFrame(columns=list(columns))
    frame = pd.DataFrame(records)
    return frame


# ============================================================================
# 3. 新旧指标对比
# ============================================================================

# 本轮预期变化指标（禁止强行对齐旧数值）
EXPECTED_CHANGE_REASONS = {
    '岗位语义事件数': '事件口径不变（相邻核心版本一对），仅列口径与判定阈值来源改变',
    '岗位显著语义变化候选数': '主判据由完整语义距离改为去薪资语义距离，阈值随真实分布重算',
    '公司正式实体数': '跨地域不再作为自动拆分正式实体的充分条件，同名跨地域改判 MULTI_LOCATION_AMBIGUOUS',
    '公司简介版本数': '只允许 CONSISTENT 快照进入正式版本历史，同时间多简介不再串成时间版本',
    '公司简介变化事件数': '随版本层重算，并新增「跨越歧义快照」说明字段',
    '有变化公司数': '随版本层与事件层重算',
}


def compare_metric_rows(pairs: list, unit: str = '') -> pd.DataFrame:
    """构造「旧值 / 新值 / 绝对差 / 相对差 / 是否预期变化 / 变化原因」对比表。"""
    rows = []
    for name, old_value, new_value, reason in pairs:
        old_number = pd.to_numeric(old_value, errors='coerce')
        new_number = pd.to_numeric(new_value, errors='coerce')
        if pd.notna(old_number) and pd.notna(new_number):
            difference = round(float(new_number) - float(old_number), 6)
            relative = (round(float(difference) / float(old_number), 6)
                        if float(old_number) != 0 else None)
            changed = bool(not np.isclose(float(new_number), float(old_number),
                                          rtol=1e-9, atol=1e-9))
        else:
            difference, relative = None, None
            changed = (old_value != new_value)
        rows.append({
            '指标': name,
            '旧值': old_value,
            '新值': new_value,
            '单位': unit,
            '绝对差': difference,
            '相对差': relative,
            '是否变化': int(changed),
            '是否预期变化': int(bool(reason)),
            '变化原因': reason or '不允许变化的核心回归项',
        })
    return pd.DataFrame(rows)


# ============================================================================
# 4. 总修复审计表
# ============================================================================

def build_audit_tables(baseline: dict, stage_metrics: dict,
                       comparison: pd.DataFrame | None = None) -> dict:
    """构建 20 号总修复审计表（12 张子表）。"""
    job_metrics = stage_metrics.get('stage_08', {})
    company_metrics = stage_metrics.get('stage_10', {})
    identity_metrics = stage_metrics.get('stage_09', {})
    baseline_company = (baseline.get('company') or {}).get('指标') or {}

    overview_rows = [
        {'指标': '岗位核心版本数', '数值': job_metrics.get('corpus_rows')},
        {'指标': '岗位文本语义事件数', '数值': job_metrics.get('event_rows')},
        {'指标': '去薪资口径显著变化候选数_P90', '数值': job_metrics.get('significant_events')},
        {'指标': '去薪资口径极端变化候选数_P95', '数值': job_metrics.get('extreme_events')},
        {'指标': '旧完整口径显著候选数_P90', '数值': job_metrics.get('significant_full_caliber')},
        {'指标': '旧完整口径极端候选数_P95', '数值': job_metrics.get('extreme_full_caliber')},
        {'指标': '因薪资文本影响退出显著候选的事件数',
         '数值': job_metrics.get('dropped_by_salary_text')},
        {'指标': '因新口径进入显著候选的事件数',
         '数值': job_metrics.get('entered_by_safe_caliber')},
        {'指标': '模型安全版薪资泄漏残留', '数值': job_metrics.get('salary_residual')},
        {'指标': '公司正式时序可用实体数', '数值': company_metrics.get('formal_entities')},
        {'指标': '公司简介快照总数', '数值': company_metrics.get('snapshot_rows')},
        {'指标': 'CONSISTENT 快照数', '数值': company_metrics.get('consistent_snapshots')},
        {'指标': 'AMBIGUOUS 快照数', '数值': company_metrics.get('ambiguous_snapshots')},
        {'指标': 'MISSING 快照数', '数值': company_metrics.get('missing_snapshots')},
        {'指标': '公司简介版本数', '数值': company_metrics.get('history_rows')},
        {'指标': '公司简介语义事件数', '数值': company_metrics.get('event_rows')},
        {'指标': '跨歧义快照事件数', '数值': company_metrics.get('ambiguous_cross_events')},
    ]

    company_pairs = [
        ('公司正式实体数', baseline_company.get('旧正式公司实体数'),
         identity_metrics.get('formal_entities'), EXPECTED_CHANGE_REASONS['公司正式实体数']),
        ('公司实体总数', baseline_company.get('旧公司实体数'),
         identity_metrics.get('entity_candidates'), EXPECTED_CHANGE_REASONS['公司正式实体数']),
        ('公司简介版本数', baseline_company.get('旧公司简介版本数'),
         company_metrics.get('history_rows'), EXPECTED_CHANGE_REASONS['公司简介版本数']),
        ('公司简介变化事件数', baseline_company.get('旧公司简介变化事件数'),
         company_metrics.get('event_rows'), EXPECTED_CHANGE_REASONS['公司简介变化事件数']),
        ('有变化公司数', baseline_company.get('旧有变化公司数'),
         company_metrics.get('entities_with_change'), EXPECTED_CHANGE_REASONS['有变化公司数']),
    ]

    tables = {
        '01_修复前基线': baseline_frame(baseline),
        '02_修复后总体': pd.DataFrame(overview_rows),
        '03_完整vs去薪资语义': _records_frame(
            job_metrics.get('caliber_summary'),
            ['指标', '完整口径', '去薪资口径', '完整减去薪资安全语义距离差']),
        '04_去薪资语义分位数': _records_frame(
            job_metrics.get('safe_thresholds'), ['指标', '阈值', '样本数', '来源', '用途']),
        '05_BGE截断统计': _records_frame(
            job_metrics.get('token_stats'),
            ['文本类型', '文本数', 'Token_P50', 'Token_P95', 'Token_P99', 'Token_Max',
             '模型最大token数', '截断文本数', '截断比例', '平均理论保留比例']),
        '06_岗位大类技能覆盖': _records_frame(
            stage_metrics.get('stage_07', {}).get('category_skill_coverage'),
            ['岗位大类', '样本岗位数', '提取到至少1项技术技能的岗位数', '技术技能覆盖率',
             '提取到任意技能或业务能力的岗位数', '总体实体覆盖率', '平均技术技能数', '平均业务能力数']),
        '07_技能一级类型': _records_frame(
            stage_metrics.get('stage_07', {}).get('skill_family_stats'),
            ['feature_family', '技能标准名数', '岗位数', '岗位覆盖率', '技能项次']),
        '08_公司身份调整影响': _records_frame(
            identity_metrics.get('identity_adjustment'),
            ['指标', '数值', '说明']),
        '09_公司同时间简介冲突': _records_frame(
            company_metrics.get('snapshot_conflict_table'),
            ['公司简介快照状态', '快照数', '占比', '涉及公司数']),
        '10_公司版本重构前后': compare_metric_rows(company_pairs[:3]),
        '11_公司事件重构前后': compare_metric_rows(company_pairs[3:]),
        '12_回归检查': comparison if comparison is not None else pd.DataFrame(),
    }
    return tables


def write_refinement_outputs(baseline: dict | None = None,
                             comparison: pd.DataFrame | None = None) -> dict:
    """写出 20 号修复审计表与 13 号封版记录，返回产物路径与关键指标。"""
    baseline = baseline if baseline is not None else load_baseline()
    stage_metrics = load_stage_metrics()
    if comparison is None:
        comparison = quality.compare_with_baseline(_regression_metrics(stage_metrics))
    tables = build_audit_tables(baseline, stage_metrics, comparison)
    audit_path = io_utils.write_excel(
        project_paths.TABLES_DIR / project_paths.TABLE_TEXT_SEMANTIC_REFINEMENT, tables)
    record_path = io_utils.write_markdown(
        project_paths.RECORDS_DIR / project_paths.RECORD_TEXT_SEMANTIC_REFINEMENT,
        build_record_lines(baseline, stage_metrics, tables, comparison))
    return {
        'audit_path': project_paths.relative_to_root(audit_path),
        'record_path': project_paths.relative_to_root(record_path),
        'sheet_count': int(len(tables)),
    }


def _regression_metrics(stage_metrics: dict) -> dict:
    """把 Stage 指标合并为 compare_with_baseline 需要的扁平 dict。"""
    merged: dict = {}
    for payload in stage_metrics.values():
        merged.update({key: value for key, value in payload.items()
                       if not isinstance(value, (list, dict))})
    return merged


def _diff_reason_lines() -> list:
    """定量解释本轮新旧差异的来源（全部由真实产物文件计算）。"""
    lines: list = []
    job_events = _read_optional(project_paths.JOB_TEXT_EVENTS_PARQUET)
    job_metrics = io_utils.read_json(project_paths.METRICS_DIR / 'stage_08.json') or {}
    if not job_events.empty:
        gap = (job_events[schema.JD_DISTANCE_FULL_FIELD]
               - job_events[schema.JD_DISTANCE_SAFE_FIELD]).abs()
        affected = int((gap > 1e-9).sum())
        identical = int(job_events['文本完全一致标志'].sum())
        lines += [
            f'- **岗位语义层**：{len(job_events)} 条版本切换中，完整口径与去薪资口径语义距离'
            f'存在差异的事件 **{affected}** 条；其余事件两种文本完全一致'
            f'（文本完全一致事件 {identical} 条），距离逐位相同，因此分位数与显著候选数不变；',
            f"  去薪资文本完全相同的版本对 {job_metrics.get('safe_identical_pairs', '—')} 条，"
            '其去薪资语义距离全部为 0，说明「纯薪资文字调整」在正式口径下不再被计为岗位语义变化。',
        ]
    snapshots = _read_optional(project_paths.COMPANY_PROFILE_SNAPSHOTS_PARQUET,
                               columns=[schema.COMPANY_ENTITY_ID_FIELD,
                                        schema.COMPANY_SNAPSHOT_STATE_FIELD])
    if not snapshots.empty:
        consistent_entities = set(snapshots.loc[
            snapshots[schema.COMPANY_SNAPSHOT_STATE_FIELD] == 'CONSISTENT',
            schema.COMPANY_ENTITY_ID_FIELD])
        missing_entities = set(snapshots.loc[
            snapshots[schema.COMPANY_SNAPSHOT_STATE_FIELD] == 'MISSING',
            schema.COMPANY_ENTITY_ID_FIELD])
        pseudo = consistent_entities & missing_entities
        lines += [
            f'- **公司层版本数下降**：旧口径把「空简介 ↔ 非空简介」也压缩成时间版本（伪版本），'
            f'本轮有 {len(pseudo)} 个公司同时存在 CONSISTENT 与 MISSING 快照，'
            '这些公司在新口径下只保留 CONSISTENT 版本，空简介不再生成伪版本；',
            f'  叠加跨地域歧义公司不再进入正式公司时序'
            f'（MULTI_LOCATION_AMBIGUOUS {_multi_location_rows()} 行），'
            '共同导致公司简介版本数与事件数下降。',
        ]
    if not lines:
        lines.append('- 缺资产物文件，无法计算差异来源。')
    return lines


def _multi_location_rows() -> int:
    """统计公司实体映射表中 MULTI_LOCATION_AMBIGUOUS 行数（缺文件返回 0）。"""
    frame = _read_optional(project_paths.COMPANY_ENTITY_MAP_PARQUET,
                           columns=[schema.COMPANY_MAPPING_METHOD_FIELD])
    if frame.empty:
        return 0
    return int((frame[schema.COMPANY_MAPPING_METHOD_FIELD] == 'MULTI_LOCATION_AMBIGUOUS').sum())


def build_record_lines(baseline: dict, stage_metrics: dict, tables: dict,
                       comparison: pd.DataFrame) -> list:
    """生成 13 号文本语义层封版记录。"""
    job_metrics = stage_metrics.get('stage_08', {})
    company_metrics = stage_metrics.get('stage_10', {})
    identity_metrics = stage_metrics.get('stage_09', {})
    baseline_job = (baseline.get('job') or {}).get('指标') or {}
    baseline_company = (baseline.get('company') or {}).get('指标') or {}

    lines = [
        '# 阶段记录：Stage 06~10 Refinement R1 修复与数据封版',
        '',
        '> 本文件由 `src/refinement.py` 生成，全部数字来自真实运行结果与真实产物文件，'
        '禁止人工硬编码基线。',
        '',
        '## 1. 为什么修',
        '',
        '| 问题 | 风险 | 本轮处理 |',
        '| --- | --- | --- |',
        '| 岗位语义距离与薪资文本自相关 | 纯薪资文字调整会被误判为岗位实质变化 | 新增「去薪资」语义口径并作为正式主判据 |',
        '| BGE 长文本可能被截断 | 长 JD 的语义距离不可比 | 用真实 tokenizer 做 token 长度与截断审计，并做截断敏感性对照 |',
        '| 技能/业务能力/技术领域混在同一层 | 覆盖率与技能数不可解释 | 技能字典升级为 feature_family → group → canonical 三级 |',
        '| 同名跨地域被自动拆成多个正式公司实体 | 同一企业被过度拆分，公司时序失真 | 所在地不再作为自动拆分充分条件，改判 MULTI_LOCATION_AMBIGUOUS |',
        '| 同一公司同一时间存在多个简介 | 并行简介被强行串成 A → B 时间版本 | 新增公司简介快照层，只有 CONSISTENT 快照进入正式版本历史 |',
        '',
        '## 2. 修了什么（口径变化清单）',
        '',
        '| 字段 / 产物 | 旧口径 | 新口径 |',
        '| --- | --- | --- |',
        '| 岗位描述语义距离 | 单一字段，基于语义分析版全文 | 拆为 `_完整` 与 `_去薪资` 两列，正式主字段为 `_去薪资` |',
        '| 是否显著语义变化候选 | 基于完整语义距离 P90 | 基于去薪资语义距离 P90（完整口径保留为对照字段） |',
        '| 完整减去薪资安全语义距离差 | 不存在 | 新增，仅作诊断；**禁止**解释为「薪资文本贡献率」 |',
        '| 相对上一版本JD语义距离 | 完整语义距离 | 兼容字段，口径切换为**去薪资语义距离** |',
        '| 技能标准实体 | 单一 group | 新增 feature_family 一级类型，group 保留为细粒度类别 |',
        '| 公司正式时序准入 | EXACT_NORMALIZED_NAME / EXACT_NAME_LOCATION / APPROVED_ALIAS | 仅 EXACT_NORMALIZED_NAME / APPROVED_ALIAS |',
        '| 公司简介版本 | 直接按「实体 + 时间 + 简介」折叠 | 先建快照层，仅 CONSISTENT 快照进入版本层 |',
        '| 公司变化事件 | 无跨歧义说明 | 新增「中间存在歧义快照」「跨越歧义快照数量」等字段 |',
        '',
        '## 3. 岗位语义层：旧 → 新',
        '',
        '| 指标 | 修复前 | 修复后 |',
        '| --- | --- | --- |',
        f"| 岗位语义事件数 | {baseline_job.get('岗位语义事件数', '—')} | "
        f"{job_metrics.get('event_rows', '—')} |",
        f"| 完整口径语义距离 P90 | {baseline_job.get('旧语义距离P90', '—')} | "
        f"{job_metrics.get('semantic_p90_full', '—')} |",
        f"| 去薪资口径语义距离 P90 | —（本轮新增） | {job_metrics.get('semantic_p90', '—')} |",
        f"| 显著变化候选数（P90） | {baseline_job.get('旧显著变化候选数_P90', '—')} | "
        f"{job_metrics.get('significant_events', '—')} |",
        f"| 极端变化候选数（P95） | {baseline_job.get('旧极端变化候选数_P95', '—')} | "
        f"{job_metrics.get('extreme_events', '—')} |",
        f"| 完整=安全文本的版本数 | —（本轮统计） | {job_metrics.get('identical_text_versions', '—')} |",
        f"| 安全文本总数（非空） | —（本轮统计） | {job_metrics.get('safe_text_total', '—')} |",
        f"| 缓存整体复用安全文本数 | —（本轮统计） | {job_metrics.get('safe_cache_reused', '—')} |",
        f"| 复用完整版向量的安全文本数 | —（本轮统计） | "
        f"{job_metrics.get('reused_safe_vectors', '—')} |",
        f"| 哈希去重后新增编码安全文本数 | —（本轮统计） | "
        f"{job_metrics.get('new_safe_encodings', '—')} |",
        '',
        '> 完整口径只作页面整体变化辅助指标；薪资变化与 JD 语义变化的对照全部改用去薪资口径。',
        '',
        '## 4. BGE token 与截断审计',
        '',
        '| 文本类型 | 文本数 | Token P50 | Token P95 | Token P99 | Token Max | 截断文本数 | 截断比例 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for row in tables['05_BGE截断统计'].itertuples(index=False):
        lines.append(f"| {row.文本类型} | {row.文本数} | {row.Token_P50} | {row.Token_P95} | "
                     f"{row.Token_P99} | {row.Token_Max} | {row.截断文本数} | {row.截断比例} |")
    lines += [
        '',
        f"- 模型最大 token 数：{job_metrics.get('model_max_tokens', '—')}"
        '（由模型 tokenizer 真实计算，禁止用字符数代替）；',
        f"- 非截断样本敏感性分析：{job_metrics.get('truncation_sensitivity_note', '见 17 号审计表 19 号子表')}。",
        '',
        '## 5. 技能三级分类',
        '',
        '| feature_family | 技能标准名数 | 岗位数 | 岗位覆盖率 | 技能项次 |',
        '| --- | --- | --- | --- | --- |',
    ]
    for row in tables['07_技能一级类型'].itertuples(index=False):
        lines.append(f"| {row.feature_family} | {row.技能标准名数} | {row.岗位数} | "
                     f"{row.岗位覆盖率} | {row.技能项次} |")
    lines += [
        '',
        '> feature_family → group → canonical 全部来自 `config/skills.yml` 显式配置，'
        '禁止运行时猜类别；未显式归属的标准实体一律落在「其他」并由门禁拦截。',
        '',
        '## 6. 岗位大类 × 技能覆盖率',
        '',
        '| 岗位大类 | 样本岗位数 | 技术技能覆盖率 | 总体实体覆盖率 | 平均技术技能数 |',
        '| --- | --- | --- | --- | --- |',
    ]
    for row in tables['06_岗位大类技能覆盖'].itertuples(index=False):
        lines.append(f"| {row.岗位大类} | {int(row.样本岗位数)} | {row.技术技能覆盖率} | "
                     f"{row.总体实体覆盖率} | {row.平均技术技能数} |")
    lines += [
        '',
        '> 覆盖率为真实统计结果，未为提高覆盖率机械扩词典；低覆盖岗位大类只作排序展示。',
        '',
        '## 7. 公司身份调整',
        '',
        f"- 修复前正式公司实体数：{baseline_company.get('旧正式公司实体数', '—')}；",
        f"- 修复后正式时序可用实体数：{identity_metrics.get('formal_entities', '—')}；",
        f"- MULTI_LOCATION_AMBIGUOUS 行数：{identity_metrics.get('multi_location_rows', '—')}"
        f"（涉及规范名称 {identity_metrics.get('multi_location_names', '—')} 个）；",
        f"- 人工复核候选行数：{identity_metrics.get('review_rows', '—')}。",
        '',
        '## 8. 公司快照与版本重构',
        '',
        f"- 公司简介快照总数：{company_metrics.get('snapshot_rows', '—')}；",
        f"- CONSISTENT / AMBIGUOUS / MISSING：{company_metrics.get('consistent_snapshots', '—')} / "
        f"{company_metrics.get('ambiguous_snapshots', '—')} / {company_metrics.get('missing_snapshots', '—')}；",
        f"- 同时间多简介涉及公司数：{company_metrics.get('ambiguous_entities', '—')}；",
        f"- 公司简介版本数：{baseline_company.get('旧公司简介版本数', '—')} → "
        f"{company_metrics.get('history_rows', '—')}；",
        f"- 公司简介语义事件数：{baseline_company.get('旧公司简介变化事件数', '—')} → "
        f"{company_metrics.get('event_rows', '—')}；",
        f"- 跨歧义快照的变化事件数：{company_metrics.get('ambiguous_cross_events', '—')}。",
        '',
        '> A → B 之间若存在 AMBIGUOUS 快照，事件中必须记录跨越歧义快照数量，'
        '避免把变化时间读成精确发生时刻（变化时间仍定义为新版本首次有效观测时间）。',
        '',
        '## 9. 新旧结果比较原则',
        '',
        '| 指标 | 旧值 | 新值 | 绝对差 | 相对差 | 是否预期变化 | 变化原因 |',
        '| --- | --- | --- | --- | --- | --- | --- |',
    ]
    for row in tables['10_公司版本重构前后'].itertuples(index=False):
        lines.append(f"| {row.指标} | {row.旧值} | {row.新值} | {row.绝对差} | {row.相对差} | "
                     f"{row.是否预期变化} | {row.变化原因} |")
    for row in tables['11_公司事件重构前后'].itertuples(index=False):
        lines.append(f"| {row.指标} | {row.旧值} | {row.新值} | {row.绝对差} | {row.相对差} | "
                     f"{row.是否预期变化} | {row.变化原因} |")
    lines += [
        '',
        '> 本轮预期变化不得被判为 FAIL；核心回归项（raw = 172,063、唯一岗位 ID = 17,144、'
        'ID↔URL 1:1、最终实体 = 17,144、分类关系集合一致）必须保持不变。',
        '',
        '## 9.1 差异原因定量说明（由真实产物计算）',
        '',
    ]
    for line in _diff_reason_lines():
        lines.append(line)
    lines += [
        '',
        '## 10. 核心回归检查',
        '',
        '| 指标 | 历史基线 | 新结果 | 绝对差异 | 是否一致 | 变化性质 |',
        '| --- | --- | --- | --- | --- | --- |',
    ]
    for row in comparison.itertuples(index=False):
        lines.append(f'| {row.指标} | {row.历史基线} | {row.新流水线结果} | {row.绝对差异} | '
                     f'{row.是否一致} | {row.变化性质} |')
    lines += [
        '',
        '## 11. 产物清单',
        '',
        f"- 公司简介快照：`{project_paths.relative_to_root(project_paths.COMPANY_PROFILE_SNAPSHOTS_PARQUET)}`；",
        f"- 总修复审计表：`{project_paths.relative_to_root(project_paths.TABLES_DIR / project_paths.TABLE_TEXT_SEMANTIC_REFINEMENT)}`；",
        f"- 封版记录：`{project_paths.relative_to_root(project_paths.RECORDS_DIR / project_paths.RECORD_TEXT_SEMANTIC_REFINEMENT)}`；",
        f"- 修复前基线：`{project_paths.relative_to_root(project_paths.REFINEMENT_BASELINE_JSON)}`。",
        '',
        '## 12. 最终正式口径（封版声明）',
        '',
        '- 岗位语义：`岗位描述语义距离_去薪资` 为**唯一正式主判据**，完整口径仅作页面整体变化辅助；',
        '- 技能：`feature_family → group → canonical` 三级，AI 技能数与大模型技能数分开统计；',
        '- 公司：仅 `EXACT_NORMALIZED_NAME` / `APPROVED_ALIAS` 进入正式公司时序；',
        '- 公司时序：仅 `CONSISTENT` 快照进入正式公司简介版本历史；',
        '- 高维向量不写入任何主业务宽表，模型安全版向量与完整版向量逐行对齐。',
        '',
    ]
    return lines

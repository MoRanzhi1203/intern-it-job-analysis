# -*- coding: utf-8 -*-
"""质量与门禁模块：真实检查驱动的运行门禁、字段质量统计、阶段日志与回归核验。

门禁状态必须由真实检查决定：
- 检查不通过时直接抛错，禁止"直接 set PASS"或捕获异常后继续。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import io_utils, project_paths

# ---- Stage 03 门禁：岗位观测快照（含公司认证标签前置语义映射与分类关系层） ----
STAGE03_GATES = [
    'CN_DATA_LOAD', 'ITEM_URL_DROP',
    'COMPANY_CERT_VALUE_AUDIT', 'COMPANY_CERT_MAPPING',
    'COMPANY_CERT_UNKNOWN_CHECK', 'COMPANY_CERT_SOURCE_DROP',
    'DETAIL_URL_NORMALIZE', 'ID_URL_ONE_TO_ONE',
    'CATEGORY_RELATION_BUILD', 'CATEGORY_RELATION_RECONCILE', 'SOURCE_CATEGORY_AGG',
    'OBSERVATION_BUILD', 'CATEGORY_DUPLICATE_COLLAPSE', 'OBSERVATION_TIME_CHECK',
    'OBSERVATION_AUDIT_EXPORT',
]

# ---- Stage 04 门禁：岗位版本时序与变化事件 ----
STAGE04_GATES = [
    'CORE_SIGNATURE_BUILD', 'FULL_SIGNATURE_BUILD',
    'VERSION_SEQUENCE_BUILD', 'VERSION_TIME_ORDER', 'VERSION_CONTIGUITY',
    'CHANGE_EVENT_BUILD',
    'FIELD_CONSISTENCY_AUDIT', 'BUSINESS_CONFLICT_EXPORT', 'VERSION_AUDIT_EXPORT',
]

# ---- Stage 05 门禁：最终岗位实体与还原校验 ----
STAGE05_GATES = [
    'FINAL_VERSION_SELECT', 'UNIQUE_ENTITY_BUILD',
    'UNIQUE_ENTITY_ID_CHECK', 'UNIQUE_ENTITY_URL_CHECK',
    'VERSION_TO_ENTITY_RECONCILE', 'CATEGORY_TO_ENTITY_RECONCILE',
    'COMPANY_CERT_FIELD_CHECK', 'ENTITY_AUDIT_EXPORT',
]

RUNNER_GATES = ['ARCHIVE_PRE_REFACTOR', 'BASELINE_RECORD', 'PROJECT_STRUCTURE', 'SHARED_MODULES',
                'REGRESSION_CHECK', 'PIPELINE_ENTRY', 'INDEPENDENT_SCRIPT_RUN',
                'DOCUMENTATION_UPDATE', 'REFINEMENT_BASELINE_CAPTURE',
                'REFINEMENT_REGRESSION_CHECK', 'FINAL_CHECK']

# ---- Stage 06 门禁：岗位版本文本语料准备 ----
STAGE06_GATES = [
    'TEXT_SOURCE_LOAD', 'TEXT_RAW_PRESERVE', 'TEXT_CLEAN_BUILD', 'TEXT_SECTION_SPLIT',
    'SALARY_LEAKAGE_REMOVAL', 'MODEL_SAFE_STRICT_SALARY_PATTERN_CHECK',
    'SKILL_CONFIG_LOAD', 'TEXT_AUDIT_EXPORT',
]

# ---- Stage 07 门禁：岗位技能与文本特征（含技能词典契约与歧义防护） ----
STAGE07_GATES = [
    'SKILL_CONFIG_SCHEMA', 'SKILL_ALIAS_UNIQUENESS', 'SKILL_CATEGORY_HIERARCHY',
    'SKILL_REQUIREMENT_SECTION', 'SKILL_MODEL_SAFE_TEXT', 'SKILL_FINAL_VERSION_SELECTION',
    'SKILL_EXTRACTION', 'SKILL_ALIAS_NORMALIZE', 'SKILL_FAMILY_MAPPING',
    'SKILL_AMBIGUOUS_TERM_GUARD', 'SKILL_JOB_LEVEL_BUILD',
    'TEXT_FEATURE_BUILD', 'TEXT_FEATURE_COVERAGE',
    'CATEGORY_SKILL_COVERAGE_BUILD', 'SKILL_AUDIT_EXPORT', 'TEXT_FEATURE_AUDIT_EXPORT',
]

# ---- Stage 08 门禁：岗位描述语义时序（Refinement R1 增加双口径与截断审计） ----
STAGE08_GATES = [
    'TFIDF_BASELINE_BUILD', 'EMBEDDING_MODEL_LOAD', 'EMBEDDING_BUILD',
    'SEMANTIC_DISTANCE_CHECK',
    'SAFE_TEXT_EMBEDDING_BUILD', 'FULL_SAFE_EMBEDDING_REUSE',
    'SAFE_SEMANTIC_DISTANCE_CHECK', 'SAFE_SALARY_ALIGNMENT',
    'TOKEN_LENGTH_AUDIT', 'TRUNCATION_SENSITIVITY',
    'JOB_TEXT_EVENT_BUILD', 'JOB_TEXT_VERSION_RECONCILE',
    'JOB_TEXT_VERSION_SUMMARY', 'JOB_TEXT_AUDIT_EXPORT',
]

# ---- Stage 09 门禁：公司实体识别（Refinement R1 禁止按所在地自动拆分正式实体） ----
STAGE09_GATES = [
    'COMPANY_IDENTITY_BUILD', 'COMPANY_IDENTITY_AMBIGUITY_CHECK',
    'MULTI_LOCATION_COMPANY_CHECK', 'FORMAL_COMPANY_IDENTITY_FILTER',
    'COMPANY_IDENTITY_AUDIT_EXPORT',
]

# ---- Stage 10 门禁：公司简介快照 / 版本 / 语义时序（Refinement R1 新增快照层） ----
STAGE10_GATES = [
    'COMPANY_PROFILE_SNAPSHOT_BUILD', 'COMPANY_PROFILE_SNAPSHOT_UNIQUE',
    'COMPANY_SAME_TIME_CONFLICT_AUDIT', 'AMBIGUOUS_SNAPSHOT_EXCLUSION',
    'COMPANY_VERSION_REBUILD', 'COMPANY_PROFILE_VERSION_BUILD',
    'COMPANY_EVENT_REBUILD', 'COMPANY_TEXT_EVENT_BUILD', 'COMPANY_TEXT_AUDIT_EXPORT',
]

# ---- Stage 11 门禁：结构化业务字段清洗与薪资目标解析 ----
STAGE11_GATES = [
    'SALARY_SOURCE_LOAD', 'SALARY_FORM_AUDIT', 'SALARY_PARSE_COVERAGE',
    'SALARY_RANGE_VALIDITY', 'SALARY_UNIT_CONSISTENCY', 'SALARY_MISSING_STRATEGY',
    'STRUCTURED_FIELD_PARSE', 'COMPANY_ATTRIBUTE_ANOMALY_RESIDUAL_CHECK',
    'SALARY_JD_CONSISTENCY', 'NO_LABEL_LEAKAGE', 'STAGE11_AUDIT_EXPORT',
]

# ---- Stage 01 门禁：源记录标准化 + 公司属性语义槽位异常修复 ----
STAGE01_GATES = ['COLUMN_LOCALIZATION', 'MISSING_STANDARDIZATION',
                 'COMPANY_ATTRIBUTE_ANOMALY_DETECT', 'COMPANY_ATTRIBUTE_ANOMALY_REPAIR',
                 'COMPANY_ATTRIBUTE_ANOMALY_AUDIT_EXPORT']

# ---- 全部门禁顺序（Stage 00~11） ----
FULL_GATE_ORDER = list(dict.fromkeys([
    'ARCHIVE_PRE_REFACTOR', 'BASELINE_RECORD', 'PROJECT_STRUCTURE', 'SHARED_MODULES',
    'RAW_DATA_CHECK', *STAGE01_GATES, 'IDENTITY_AUDIT',
    *STAGE03_GATES, *STAGE04_GATES, *STAGE05_GATES,
    *STAGE06_GATES, *STAGE07_GATES, *STAGE08_GATES, *STAGE09_GATES, *STAGE10_GATES,
    *STAGE11_GATES,
    'REGRESSION_CHECK', 'PIPELINE_ENTRY', 'INDEPENDENT_SCRIPT_RUN',
    'DOCUMENTATION_UPDATE', 'FINAL_CHECK',
]))

# ---- Stage 12 建模数据集门禁（独立脚本，不进入 Stage 00~11 预处理流水线） ----
STAGE12_MODELING_GATES = [
    'MODEL_DATASET_TARGET_LEAKAGE', 'MODEL_DATASET_TEXT_LEAKAGE',
    'MODEL_DATASET_SALARY_DERIVED_FEATURE_BLOCK', 'MODEL_DATASET_ONE_JOB_ONE_ROW',
    'MODEL_DATASET_TARGET_SAMPLE', 'MODEL_DATASET_FEATURE_MANIFEST',
    'MODEL_DATASET_AUDIT_EXPORT',
]

# ---- Stage 13 正式 EDA 与统计检验门禁（独立脚本，不进入 Stage 00~11 流水线） ----
STAGE13_EDA_GATES = [
    'EDA_UNIQUE_JOB_UNIT', 'EDA_SALARY_SAMPLE_14883', 'EDA_DESCRIPTIVE_STATS',
    'EDA_GROUP_TESTS', 'EDA_MULTIPLE_TESTING_FDR', 'EDA_SKILL_SCOPE_DENOMINATOR',
    'EDA_SKILL_LAYER_SPLIT', 'EDA_COOCCURRENCE_THRESHOLD', 'EDA_COMPANY_FACTOR_SEMANTICS',
    'EDA_COMPANY_TAG_NO_OVERLAP_KW', 'EDA_COMPANY_TAG_BINARY_INFERENCE',
    'EDA_NO_CAUSAL_LABEL', 'EDA_FIGURE_STYLE', 'EDA_AUDIT_EXPORT',
]

# ---- Stage 14 薪资预测模型对比门禁（独立脚本，不进入 Stage 00~11 流水线） ----
STAGE14_MODEL_GATES = [
    'MODEL_SPLIT', 'MODEL_SPLIT_DISJOINT', 'MODEL_TRAIN_ONLY_PREPROCESS',
    'MODEL_SKILL_THRESHOLD_TRAIN_ONLY', 'MODEL_VALIDATION_SELECTION',
    'MODEL_TEST_SINGLE_EVAL', 'MODEL_COMPARISON_EXPORT', 'MODEL_ARTIFACT_EXPORT',
]

# ---- Stage 15 消融、稳健性与 SHAP 门禁（独立脚本，不进入 Stage 00~11 流水线） ----
STAGE15_ABLATION_GATES = [
    'ABLATION_SAME_SPLIT', 'ABLATION_FEATURE_GROUP_ONLY', 'SKILL_INCREMENT_EVAL',
    'TEXT_INCREMENT_EVAL', 'COMPANY_GROUP_SPLIT', 'ROBUSTNESS_EXPORT',
    'SHAP_FINAL_MODEL_ONLY', 'SHAP_EXPORT', 'SHAP_SKILL_PRESENCE',
]

# ---- Stage 16 公司字段语义核查门禁（只读取证脚本） ----
STAGE16_COMPANY_FIELD_GATES = [
    'COMPANY_FIELD_TRACE', 'COMPANY_CERTIFICATION_ATOMS', 'COMPANY_TAG_FIELD_DISTINCT',
    'COMPANY_FACTOR_MISLABEL_CONFIRMED', 'COMPANY_FIELD_AUDIT_EXPORT',
]

# ---- Stage 17 最终解释封版门禁（字段语义修正 + 技能 SHAP presence 口径） ----
STAGE17_INTERPRETATION_GATES = [
    'INTERPRETATION_FIELD_SEMANTICS', 'INTERPRETATION_SHAP_PRESENCE',
    'INTERPRETATION_MODEL_UNCHANGED', 'FINAL_AUDIT_SHEET_STRUCTURE', 'ANALYSIS_FREEZE',
    'INTERPRETATION_EXPORT',
]

# ---- Stage 13 技能 EDA 口径审计门禁（独立脚本，不进入 Stage 00~11 流水线） ----
STAGE13_SKILL_EDA_GATES = [
    'SKILL_EDA_SCOPE_SAMPLE', 'SKILL_EDA_LAYER_SPLIT', 'SKILL_EDA_GROUP_DEDUP',
    'SKILL_EDA_ROBUSTNESS', 'SKILL_EDA_SALARY_SCOPE', 'SKILL_EDA_AUDIT_EXPORT',
]

STAGE_GATE_MAP = {
    'stage_00': ['RAW_DATA_CHECK'],
    'stage_01': STAGE01_GATES,
    'stage_02': ['IDENTITY_AUDIT', 'DETAIL_URL_NORMALIZE', 'ID_URL_ONE_TO_ONE'],
    'stage_03': STAGE03_GATES,
    'stage_04': STAGE04_GATES,
    'stage_05': STAGE05_GATES,
    'stage_06': STAGE06_GATES,
    'stage_07': STAGE07_GATES,
    'stage_08': STAGE08_GATES,
    'stage_09': STAGE09_GATES,
    'stage_10': STAGE10_GATES,
    'stage_11': STAGE11_GATES,
    'stage_12_modeling': STAGE12_MODELING_GATES,
    'stage_13_skill_eda': STAGE13_SKILL_EDA_GATES,
    'stage_13_eda': STAGE13_EDA_GATES,
    'stage_14_model': STAGE14_MODEL_GATES,
    'stage_15_ablation': STAGE15_ABLATION_GATES,
    'stage_16_company_field': STAGE16_COMPANY_FIELD_GATES,
    'stage_17_interpretation': STAGE17_INTERPRETATION_GATES,
    'runner': RUNNER_GATES,
}

PIPELINE_STAGES = ('stage_00', 'stage_01', 'stage_02', 'stage_03', 'stage_04', 'stage_05',
                   'stage_06', 'stage_07', 'stage_08', 'stage_09', 'stage_10', 'stage_11')

# ---- 返工基线（来自 docs/records/00_refactor_baseline.md，仅用于回归核验） ----
BASELINE_METRICS = {
    'raw_rows': 172063,
    'raw_columns': 30,
    'unique_intern_id': 17144,
    'raw_detail_url_unique': 17148,
    'normalized_detail_url_unique': 17144,
    'id_url_unique_pairs': 17144,
    'unique_job_rows': 17144,
    'unique_job_columns': 31,
    'merged_duplicate_rows': 154919,
    'avg_records_per_job': 10.0363,
    'duplicate_groups': 15311,
    'single_record_groups': 1833,
    'multi_value_business_fields': 21,
    'business_conflict_rows': 4237,
    'key_conflict_jobs': 265,
    'records_per_id_p50': 9,
    'records_per_id_p95': 21,
    'records_per_id_max': 30,
    'top_conflict_field': '发布时间',
    'top_conflict_field_groups': 2399,
}

# ---- 本轮有意改变口径的指标：标记为「预期变化」，不参与回归失败判定 ----
EXPECTED_CHANGE_METRICS = {
    'unique_job_columns': '实体选择规则由「全历史完整度优先」改为「最终核心版本优先」，'
                          '并新增岗位版本摘要字段（核心版本数/完整页面版本数/历史是否发生X变化），'
                          '原「公司介绍图片链接」已在去重前完成语义映射并被删除',
    'multi_value_business_fields': '字段多值审计改用版本签名的归一化口径（去首尾空格、'
                                  '空字符串统一归一为缺失）并采用 nunique(dropna=True)，'
                                  '「有值 vs 空值」不再计为字段冲突（仍会在版本签名中体现为不同状态）',
    'business_conflict_rows': '同上：冲突留档口径迁移到变化事件表与 14 号版本审计表；'
                              '另：Stage 01 已确定性修复源记录的公司属性语义槽位异常'
                              '（公司性质/公司规模/公司所在地），相关字段的伪冲突随之减少',
    'key_conflict_jobs': '同上：空串归一为缺失后，1 个岗位的关键字段差异实际为「有值 vs 空值」，'
                         '不计为值冲突（该岗位的历史状态变化仍完整保留在版本与变化事件中）；'
                         '另：Stage 01 槽位异常修复后，公司性质/规模/所在地的伪冲突消失',
}

BASELINE_METRIC_LABELS = {
    'raw_rows': 'raw 行数',
    'raw_columns': 'raw 列数',
    'unique_intern_id': '实习岗位ID唯一值',
    'raw_detail_url_unique': '原始详情链接唯一值',
    'normalized_detail_url_unique': '规范化详情链接唯一值',
    'id_url_unique_pairs': 'ID—规范化URL唯一组合',
    'unique_job_rows': '唯一岗位数',
    'unique_job_columns': '最终唯一岗位表字段数',
    'merged_duplicate_rows': '合并重复记录数',
    'avg_records_per_job': '平均每岗位原始记录数',
    'duplicate_groups': '重复组数量',
    'single_record_groups': '单记录组数量',
    'multi_value_business_fields': '存在多值的业务字段数',
    'business_conflict_rows': '业务字段冲突明细行数',
    'key_conflict_jobs': '关键业务字段冲突岗位数',
    'records_per_id_p50': '原始重复记录数 P50',
    'records_per_id_p95': '原始重复记录数 P95',
    'records_per_id_max': '原始重复记录数 Max',
    'top_conflict_field': '冲突最多的字段',
    'top_conflict_field_groups': '冲突最多的字段对应岗位组数',
}


class GateError(RuntimeError):
    """门禁检查失败。"""


class GateRegistry:
    """运行门禁登记器：只接受通过真实检查的 PASS。"""

    def __init__(self, stage: str):
        if stage not in STAGE_GATE_MAP:
            raise KeyError(f'未知阶段: {stage}')
        self.stage = stage
        self.allowed = set(STAGE_GATE_MAP[stage])
        self.gates: dict = {}

    def check(self, name: str, condition: bool, note: str = '') -> None:
        """真实检查：条件为真记 PASS，否则记 FAIL 并抛错。"""
        if name not in self.allowed:
            raise KeyError(f'阶段 {self.stage} 无权登记门禁 {name}')
        if condition:
            self.gates[name] = {'status': 'PASS', 'note': note}
            print(f'GATE {name} = PASS' + (f'  [{note}]' if note else ''))
            return
        self.gates[name] = {'status': 'FAIL', 'note': note or '检查未通过'}
        self.save()
        raise GateError(f'门禁失败: {name}（{note}）')

    def mark_not_run(self, name: str, reason: str = '') -> None:
        """显式登记 NOT_RUN（模块未执行 / 依赖不可用），禁止伪造 PASS。"""
        if name not in self.allowed:
            raise KeyError(f'阶段 {self.stage} 无权登记门禁 {name}')
        self.gates[name] = {'status': 'NOT_RUN', 'note': reason or '本模块本轮未执行'}
        print(f'GATE {name} = NOT_RUN' + (f'  [{reason}]' if reason else ''))

    def assert_frame(self, name: str, frame: pd.DataFrame, expected_rows: int | None = None,
                     expected_columns=None, required_columns=None, note: str = '') -> None:
        """通用 DataFrame 一致性检查：行数 / 列名 / 必需字段。"""
        problems = []
        if expected_rows is not None and len(frame) != expected_rows:
            problems.append(f'行数 {len(frame)} != 期望 {expected_rows}')
        if expected_columns is not None and list(frame.columns) != list(expected_columns):
            problems.append('列名或列顺序不一致')
        if required_columns is not None:
            missing = [c for c in required_columns if c not in frame.columns]
            if missing:
                problems.append(f'缺少必需字段 {missing}')
        detail = '; '.join(problems) if problems else note
        self.check(name, not problems, detail)

    @property
    def results(self) -> dict:
        return dict(self.gates)

    def save(self, path: Path | None = None) -> Path:
        target = path or (project_paths.GATES_DIR / f'{self.stage}.json')
        return io_utils.write_json(target, {'stage': self.stage, 'gates': self.gates})

    @staticmethod
    def load_all() -> dict:
        """读取 outputs/logs/gates/ 下全部阶段门禁结果。"""
        merged: dict = {}
        for path in sorted(project_paths.GATES_DIR.glob('*.json')):
            payload = io_utils.read_json(path) or {}
            merged.update(payload.get('gates', {}))
        return merged


def configure_logging(log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """配置根日志：终端输出 + 可选文件输出。"""
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%H:%M:%S')

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    return logging.getLogger('pipeline')


def stage_banner(stage: str, title: str) -> None:
    """打印阶段标题。"""
    print()
    print('=' * 78)
    print(f'[{stage}] {title}')
    print('=' * 78)


def _is_blank(value) -> bool:
    """单值缺失判定：NaN / 空字符串 / 空 list（公司认证标签 []）均视为缺失。"""
    if isinstance(value, (list, tuple, set, frozenset)):
        return len(value) == 0
    if isinstance(value, str):
        return value.strip() == ''
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def count_blank(frame: pd.DataFrame, columns) -> pd.Series:
    """统计缺失数量：NaN、空字符串与空 list 均计为缺失。"""
    total = pd.Series(0, index=frame.index, dtype='int64')
    for column in columns:
        blank = frame[column].map(_is_blank)
        total = total + blank.astype('int64')
    return total


def field_quality_table(frame: pd.DataFrame, name_map: dict | None = None,
                        description_map: dict | None = None) -> pd.DataFrame:
    """字段质量统计：非空 / 缺失 / 缺失率 / 唯一值 / 空字符串。"""
    rows = []
    for index, column in enumerate(frame.columns, start=1):
        series = frame[column]
        non_null = int(series.notna().sum())
        blank = count_blank(frame[[column]], [column]).sum()
        rows.append({
            '序号': index,
            '字段名': column,
            '中文名称': (name_map or {}).get(column, ''),
            '数据类型': str(series.dtype),
            '非空数量': non_null,
            '缺失数量': int(series.isna().sum()),
            '空字符串数量': int(blank) - int(series.isna().sum()),
            '唯一值数量': int(series.nunique(dropna=True)),
        })
    table = pd.DataFrame(rows)
    table['缺失率'] = (table['缺失数量'] / len(frame)).round(6) if len(frame) else 0.0
    if description_map is not None:
        table['字段说明'] = [description_map.get(c, '') for c in frame.columns]
    return table


def identity_field_stats(frame: pd.DataFrame, fields) -> pd.DataFrame:
    """身份字段统计：总记录数 / 非空 / 缺失 / 空字符串 / 唯一值。"""
    rows = []
    for field in fields:
        series = frame[field]
        blank = count_blank(frame[[field]], [field]).sum()
        rows.append({
            '字段': field,
            '总记录数': len(frame),
            '非空数量': int(series.notna().sum()),
            '缺失数量': int(series.isna().sum()),
            '空字符串数量': int(blank) - int(series.isna().sum()),
            '唯一值数量': int(series.nunique(dropna=True)),
        })
    return pd.DataFrame(rows)


def compare_with_baseline(new_metrics: dict) -> pd.DataFrame:
    """新结果与历史基线逐项对比，输出回归核验表。

    本轮属于有意改变 Stage 03 业务语义，EXPECTED_CHANGE_METRICS 中的指标
    标记为「预期变化」，不参与回归失败判定。
    """
    rows = []
    for key, baseline_value in BASELINE_METRICS.items():
        new_value = new_metrics.get(key, None)
        if isinstance(baseline_value, (int, float)) and isinstance(new_value, (int, float)):
            difference = round(float(new_value) - float(baseline_value), 4)
            consistent = bool(np.isclose(float(new_value), float(baseline_value),
                                         rtol=1e-6, atol=1e-4))
        else:
            difference = ''
            consistent = (new_value == baseline_value)
        expected_change = key in EXPECTED_CHANGE_METRICS
        if consistent:
            nature, note = '需一致', '一致'
        elif expected_change:
            nature = '预期变化'
            note = f'预期变化：{EXPECTED_CHANGE_METRICS[key]}（禁止为匹配基线恢复原链接字段）'
        else:
            nature = '需一致'
            note = '差异需人工确认（不得修改真实结果去匹配基线）'
        rows.append({
            '指标': BASELINE_METRIC_LABELS.get(key, key),
            '历史基线': baseline_value,
            '新流水线结果': new_value,
            '绝对差异': difference,
            '是否一致': consistent,
            '变化性质': nature,
            '备注': note,
        })
    return pd.DataFrame(rows)


def summarize_gates(gate_results: dict, order=None) -> pd.DataFrame:
    """汇总门禁结果，缺失项标记为 NOT_RUN。"""
    order = order or FULL_GATE_ORDER
    rows = []
    for name in order:
        item = gate_results.get(name)
        if item is None:
            rows.append({'门禁项': name, '状态': 'NOT_RUN', '说明': ''})
        else:
            rows.append({'门禁项': name, '状态': item.get('status', ''),
                         '说明': item.get('note', '')})
    return pd.DataFrame(rows)


def print_gates(gate_results: dict, order=None) -> None:
    """按固定顺序打印门禁状态。"""
    table = summarize_gates(gate_results, order=order)
    print()
    print('=' * 78)
    for row in table.itertuples(index=False):
        note = f'  ({row.说明})' if row.说明 else ''
        print(f'{row.门禁项} = {row.状态}{note}')
    print('=' * 78)
    pass_count = int((table['状态'] == 'PASS').sum())
    fail_count = int((table['状态'] == 'FAIL').sum())
    not_run = int((table['状态'] == 'NOT_RUN').sum())
    print(f'门禁统计: PASS {pass_count} / FAIL {fail_count} / NOT_RUN {not_run}（共 {len(table)} 项）')

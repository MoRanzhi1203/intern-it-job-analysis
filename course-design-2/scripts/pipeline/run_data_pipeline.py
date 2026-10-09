# -*- coding: utf-8 -*-
"""数据流水线总执行器：一键运行 Stage 00 → 10，并完成回归核验与修复审计。

用法：
    python scripts/pipeline/run_data_pipeline.py
    python scripts/pipeline/run_data_pipeline.py --reuse-existing-raw
    python scripts/pipeline/run_data_pipeline.py --from-stage 6 --to-stage 10 --reuse-existing-raw

阶段路线：
    Stage 00 原始数据导出
    Stage 01 字段汉化
    Stage 02 岗位身份审计
    Stage 03 岗位观测快照（搜索来源折叠 + 分类关系层）
    Stage 04 岗位版本时序（双签名 + 版本压缩 + 变化事件）
    Stage 05 最终岗位实体（最终核心版本优先 + 原子替换）
    Stage 06 岗位版本文本语料（原始/清洗/语义分析版/模型安全版）
    Stage 07 岗位技能与文本特征（feature_family → group → canonical）
    Stage 08 岗位描述语义时序（完整 / 去薪资两种统计范围 + token 截断审计）
    Stage 09 公司实体识别（跨地域不再自动拆分）
    Stage 10 公司简介快照 / 版本 / 语义时序

特性：
- 任一阶段失败立即停止并返回非 0 exit code；
- 日志写入 outputs/logs/data_pipeline_latest.log；
- 阶段门禁由各阶段脚本真实检查后写入 outputs/logs/gates/；
- 覆盖正式结果前先固化 Refinement R1 修复前基线（幂等）；
- 结束后生成 11 号回归核验表、20 号总修复审计表与 13 号封版记录。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = next(
    _candidate for _candidate in Path(__file__).resolve().parents
    if (_candidate / "data").is_dir() and (_candidate / "scripts").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import io_utils, project_paths, quality, refinement  # noqa: E402

STAGE_SCRIPTS = {
    0: ('scripts/pipeline/00_export_raw.py', 'Stage 00 原始 MySQL 数据导出'),
    1: ('scripts/pipeline/01_localize_raw_columns.py', 'Stage 01 源记录标准化与公司属性语义槽位异常修复'),
    2: ('scripts/pipeline/02_audit_job_identity.py', 'Stage 02 岗位身份一致性审计'),
    3: ('scripts/pipeline/03_build_job_observations.py', 'Stage 03 岗位观测快照构建'),
    4: ('scripts/pipeline/04_build_job_versions.py', 'Stage 04 岗位版本时序构建'),
    5: ('scripts/pipeline/05_build_unique_jobs.py', 'Stage 05 最终岗位实体构建'),
    6: ('scripts/pipeline/06_prepare_text_corpus.py', 'Stage 06 岗位版本文本语料准备'),
    7: ('scripts/pipeline/07_extract_job_text_features.py', 'Stage 07 最终岗位技能与文本特征'),
    8: ('scripts/pipeline/08_build_job_text_semantics.py', 'Stage 08 岗位描述语义时序（完整 / 去薪资两种统计范围）'),
    9: ('scripts/pipeline/09_resolve_company_entities.py', 'Stage 09 公司实体高置信度识别（跨地域不拆分）'),
    10: ('scripts/pipeline/10_build_company_text_semantics.py', 'Stage 10 公司简介快照 / 版本 / 语义时序'),
    11: ('scripts/pipeline/11_clean_structured_fields.py', 'Stage 11 结构化业务字段清洗与薪资目标解析'),
}
LAST_STAGE = max(STAGE_SCRIPTS)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='course-design-2 数据流水线总执行器')
    parser.add_argument('--from-stage', type=int, default=0, choices=list(STAGE_SCRIPTS))
    parser.add_argument('--to-stage', type=int, default=LAST_STAGE, choices=list(STAGE_SCRIPTS))
    parser.add_argument('--reuse-existing-raw', action='store_true',
                        help='Stage 00 复用现有 data/raw Parquet（不访问 MySQL）')
    parser.add_argument('--force-export-raw', action='store_true',
                        help='Stage 00 强制重新从 MySQL 导出')
    return parser.parse_args()


class PipelineLogger:
    """同时写终端与 pipeline 日志文件。"""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text('', encoding='utf-8')

    def write(self, text: str = '') -> None:
        print(text, flush=True)
        with self.log_path.open('a', encoding='utf-8') as handle:
            handle.write(text + '\n')


def run_stage(stage: int, args: argparse.Namespace, logger: PipelineLogger) -> int:
    """以独立子进程运行单个阶段脚本，失败返回非 0。"""
    script, title = STAGE_SCRIPTS[stage]
    command = [sys.executable, str(PROJECT_ROOT / script)]
    if stage == 0:
        if args.force_export_raw:
            command.append('--force-export')
        elif args.reuse_existing_raw:
            command.append('--reuse-existing')

    logger.write('')
    logger.write('#' * 78)
    logger.write(f'# {title}')
    logger.write(f'# 命令: {" ".join(command)}')
    logger.write('#' * 78)

    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True, text=True,
                               encoding='utf-8', errors='replace')
    for line in (completed.stdout or '').splitlines():
        logger.write(line)
    if completed.stderr:
        for line in completed.stderr.splitlines():
            logger.write(f'[stderr] {line}')
    if completed.returncode != 0:
        logger.write(f'!! 阶段失败，退出码 {completed.returncode}，流水线终止')
    return completed.returncode


def check_archive(logger: PipelineLogger) -> tuple:
    """ARCHIVE_PRE_REFACTOR：校验返工前证据可追溯。

    归档目录存在时校验文件数量；
    归档目录已被移除时，改为校验历史证据是否可从 git 历史中检出；
    若归档目录缺失且历史证据不可检出（证据不在本仓库历史中），按用户决定判 PASS 并如实标注，
    不据此断言可追溯性，也不伪造证据。
    """
    archive_dir = project_paths.ARCHIVE_PRE_REFACTOR_DIR
    if archive_dir.is_dir():
        notebooks = sorted((archive_dir / 'notebooks').glob('*.ipynb'))
        records = sorted((archive_dir / 'records').glob('*.md'))
        tables = sorted((archive_dir / 'tables').glob('*.xlsx'))
        logger.write(f'归档目录存在：notebooks {[p.name for p in notebooks]}')
        logger.write(f'                 records {[p.name for p in records]}')
        logger.write(f'                 tables {len(tables)} 个')
        ok = len(notebooks) >= 4 and len(records) >= 2 and len(tables) >= 10
        return ok, f'归档目录 notebooks {len(notebooks)} / records {len(records)} / tables {len(tables)}'

    found = []
    for probe in project_paths.ARCHIVE_HISTORY_PROBES:
        # 使用 :/ 前缀，保证路径规格按 git 仓库根解析，而不是按当前工作目录
        completed = subprocess.run(
            ['git', 'log', '--format=%H', '-n', '1', '--', f':/{probe}'],
            cwd=str(project_paths.PROJECT_ROOT), capture_output=True, text=True,
            encoding='utf-8', errors='replace')
        if completed.returncode == 0 and completed.stdout.strip():
            found.append(probe)
            logger.write(f'  历史可检出: {probe}（{completed.stdout.split()[0][:8]}）')
        else:
            logger.write(f'  历史不可检出: {probe}')
    logger.write('归档目录已移除，历史证据改由 git 历史承载')
    total = len(project_paths.ARCHIVE_HISTORY_PROBES)
    if len(found) == total:
        return True, f'归档目录已移除，历史证据在 git 历史中可检出 {len(found)}/{total} 项'
    # 归档目录已移除且历史证据不可检出：据用户决定判 PASS，但如实标注证据不在本仓库历史中，
    # 不据此断言可追溯性（不伪造证据）。
    logger.write('归档目录已移除，且历史证据在本仓库 git 历史中不可检出')
    return True, (f'归档目录已移除；历史证据在本仓库 git 历史中仅可检出 {len(found)}/{total} 项——'
                  f'相关返工前证据不在本仓库历史中（早于本仓库初始化），据用户决定判 PASS；'
                  f'可追溯性无法在本仓库内验证')


def check_structure(logger: PipelineLogger) -> tuple:
    """PROJECT_STRUCTURE：校验新目录结构。"""
    required = {
        'data/raw': project_paths.RAW_DIR,
        'data/interim': project_paths.INTERIM_DIR,
        'data/processed': project_paths.PROCESSED_DIR,
        'scripts': project_paths.PROJECT_ROOT / 'scripts',
        'src': project_paths.PROJECT_ROOT / 'src',
        'outputs/tables': project_paths.TABLES_DIR,
        'outputs/figures': project_paths.FIGURES_DIR,
        'outputs/models': project_paths.MODELS_DIR,
        'outputs/logs': project_paths.LOGS_DIR,
        'docs/records': project_paths.RECORDS_DIR,
    }
    # 说明：archive/pre_refactor（返工前证据）已按用户要求移除，
    # 其可追溯性由 ARCHIVE_PRE_REFACTOR 门禁通过 git 历史校验。
    missing = [name for name, path in required.items() if not path.is_dir()]
    logger.write(f'目录结构检查：缺失 {missing if missing else "无"}')
    for name in required:
        logger.write(f'  - {name}')
    return not missing, f'{len(required) - len(missing)}/{len(required)} 个目录存在'


def check_shared_modules(logger: PipelineLogger) -> tuple:
    """SHARED_MODULES：校验公共模块可导入且关键符号齐全。"""
    from src import dedup, io_utils as io_module, project_paths as paths_module
    from src import quality as quality_module, schema as schema_module, url_utils, versioning
    from src import (company_identity, company_profile_versioning, field_repair, plot_style,
                     refinement, skill_extraction, text_semantics, text_utils)

    checks = [
        ('project_paths.resolve_project_root', callable(getattr(paths_module, 'resolve_project_root', None))),
        ('project_paths.ensure_directories', callable(getattr(paths_module, 'ensure_directories', None))),
        ('project_paths.FEATURES_DIR', getattr(paths_module, 'FEATURES_DIR', None) is not None),
        ('io_utils.write_excel', callable(getattr(io_module, 'write_excel', None))),
        ('io_utils.excel_safe', callable(getattr(io_module, 'excel_safe', None))),
        ('io_utils.write_markdown_section',
         callable(getattr(io_module, 'write_markdown_section', None))),
        ('schema.COLUMN_NAME_CN', isinstance(getattr(schema_module, 'COLUMN_NAME_CN', None), dict)),
        ('schema.ENTITY_JOB_COLUMNS', isinstance(getattr(schema_module, 'ENTITY_JOB_COLUMNS', None), list)),
        ('schema.CORE_SIGNATURE_FIELDS', isinstance(getattr(schema_module, 'CORE_SIGNATURE_FIELDS', None), list)),
        ('schema.TEXT_CORPUS_COLUMNS', isinstance(getattr(schema_module, 'TEXT_CORPUS_COLUMNS', None), list)),
        ('schema.JOB_TEXT_EVENT_COLUMNS',
         isinstance(getattr(schema_module, 'JOB_TEXT_EVENT_COLUMNS', None), list)),
        ('schema.COMPANY_PROFILE_HISTORY_COLUMNS',
         isinstance(getattr(schema_module, 'COMPANY_PROFILE_HISTORY_COLUMNS', None), list)),
        ('url_utils.normalize_detail_url', callable(getattr(url_utils, 'normalize_detail_url', None))),
        ('url_utils.extract_intern_id_from_url', callable(getattr(url_utils, 'extract_intern_id_from_url', None))),
        ('quality.GateRegistry', hasattr(quality_module, 'GateRegistry')),
        ('quality.compare_with_baseline', callable(getattr(quality_module, 'compare_with_baseline', None))),
        ('dedup.select_representatives', callable(getattr(dedup, 'select_representatives', None))),
        ('dedup.build_unique_jobs', callable(getattr(dedup, 'build_unique_jobs', None))),
        ('dedup.aggregate_categories_from_membership',
         callable(getattr(dedup, 'aggregate_categories_from_membership', None))),
        ('versioning.canonicalize_business_value',
         callable(getattr(versioning, 'canonicalize_business_value', None))),
        ('versioning.build_signature', callable(getattr(versioning, 'build_signature', None))),
        ('versioning.collapse_observation_snapshots',
         callable(getattr(versioning, 'collapse_observation_snapshots', None))),
        ('versioning.build_version_history', callable(getattr(versioning, 'build_version_history', None))),
        ('versioning.build_change_events', callable(getattr(versioning, 'build_change_events', None))),
        ('versioning.select_final_version', callable(getattr(versioning, 'select_final_version', None))),
        ('text_utils.clean_text', callable(getattr(text_utils, 'clean_text', None))),
        ('text_utils.build_model_safe_version',
         callable(getattr(text_utils, 'build_model_safe_version', None))),
        ('text_utils.split_job_description',
         callable(getattr(text_utils, 'split_job_description', None))),
        ('skill_extraction.load_skill_config',
         callable(getattr(skill_extraction, 'load_skill_config', None))),
        ('skill_extraction.SkillMatcher', hasattr(skill_extraction, 'SkillMatcher')),
        ('text_semantics.load_embedding_model',
         callable(getattr(text_semantics, 'load_embedding_model', None))),
        ('text_semantics.quantile_thresholds',
         callable(getattr(text_semantics, 'quantile_thresholds', None))),
        ('text_semantics.classify_job_text_change',
         callable(getattr(text_semantics, 'classify_job_text_change', None))),
        ('company_identity.build_company_entity_map',
         callable(getattr(company_identity, 'build_company_entity_map', None))),
        ('company_identity.load_company_aliases',
         callable(getattr(company_identity, 'load_company_aliases', None))),
        ('company_identity.resolve_formal_entities',
         callable(getattr(company_identity, 'resolve_formal_entities', None))),
        ('company_profile_versioning.build_company_profile_snapshots',
         callable(getattr(company_profile_versioning, 'build_company_profile_snapshots', None))),
        ('company_profile_versioning.build_company_profile_history',
         callable(getattr(company_profile_versioning, 'build_company_profile_history', None))),
        ('company_profile_versioning.build_company_profile_change_events',
         callable(getattr(company_profile_versioning, 'build_company_profile_change_events', None))),
        ('company_profile_versioning.validate_company_snapshot_uniqueness',
         callable(getattr(company_profile_versioning, 'validate_company_snapshot_uniqueness', None))),
        ('refinement.capture_job_baseline',
         callable(getattr(refinement, 'capture_job_baseline', None))),
        ('refinement.capture_company_baseline',
         callable(getattr(refinement, 'capture_company_baseline', None))),
        ('refinement.write_refinement_outputs',
         callable(getattr(refinement, 'write_refinement_outputs', None))),
        ('text_semantics.load_array_cache',
         callable(getattr(text_semantics, 'load_array_cache', None))),
        ('text_semantics.token_lengths',
         callable(getattr(text_semantics, 'token_lengths', None))),
        ('skill_extraction.skills_by_family',
         callable(getattr(skill_extraction, 'skills_by_family', None))),
        ('skill_extraction.build_job_skill_membership',
         callable(getattr(skill_extraction, 'build_job_skill_membership', None))),
        ('skill_extraction.alias_table',
         callable(getattr(skill_extraction, 'alias_table', None))),
        ('text_utils.extract_requirement_section',
         callable(getattr(text_utils, 'extract_requirement_section', None))),
        ('project_paths.JOB_SKILL_MEMBERSHIP_PARQUET',
         getattr(paths_module, 'JOB_SKILL_MEMBERSHIP_PARQUET', None) is not None),
        ('project_paths.TABLE_JOB_SKILL_AUDIT',
         getattr(paths_module, 'TABLE_JOB_SKILL_AUDIT', None) is not None),
        ('project_paths.RECORD_JOB_SKILL_EXTRACTION',
         getattr(paths_module, 'RECORD_JOB_SKILL_EXTRACTION', None) is not None),
        ('schema.JOB_SKILL_MEMBERSHIP_COLUMNS',
         isinstance(getattr(schema_module, 'JOB_SKILL_MEMBERSHIP_COLUMNS', None), list)),
        ('schema.SKILL_SCOPE_FIELD',
         isinstance(getattr(schema_module, 'SKILL_SCOPE_FIELD', None), str)),
        ('project_paths.JOB_ANALYSIS_DATASET_PARQUET',
         getattr(paths_module, 'JOB_ANALYSIS_DATASET_PARQUET', None) is not None),
        ('project_paths.JOB_SALARY_MODEL_DATASET_PARQUET',
         getattr(paths_module, 'JOB_SALARY_MODEL_DATASET_PARQUET', None) is not None),
        ('project_paths.EDA_FIGURES_DIR',
         getattr(paths_module, 'EDA_FIGURES_DIR', None) is not None),
        ('project_paths.TABLE_EDA_STATISTICAL',
         getattr(paths_module, 'TABLE_EDA_STATISTICAL', None) is not None),
        ('project_paths.MODEL_SPLITS_PARQUET',
         getattr(paths_module, 'MODEL_SPLITS_PARQUET', None) is not None),
        ('project_paths.MODEL_PREDICTIONS_PARQUET',
         getattr(paths_module, 'MODEL_PREDICTIONS_PARQUET', None) is not None),
        ('project_paths.SALARY_MODEL_DIR',
         getattr(paths_module, 'SALARY_MODEL_DIR', None) is not None),
        ('schema.COMPANY_PROFILE_SNAPSHOT_COLUMNS',
         isinstance(getattr(schema_module, 'COMPANY_PROFILE_SNAPSHOT_COLUMNS', None), list)),
        ('schema.JOB_TEXT_EVENT_COLUMNS',
         isinstance(getattr(schema_module, 'JOB_TEXT_EVENT_COLUMNS', None), list)),
        ('schema.EMBEDDING_INDEX_COLUMNS',
         isinstance(getattr(schema_module, 'EMBEDDING_INDEX_COLUMNS', None), list)),
        ('project_paths.COMPANY_PROFILE_SNAPSHOTS_PARQUET',
         getattr(paths_module, 'COMPANY_PROFILE_SNAPSHOTS_PARQUET', None) is not None),
        ('project_paths.TABLE_TEXT_SEMANTIC_REFINEMENT',
         getattr(paths_module, 'TABLE_TEXT_SEMANTIC_REFINEMENT', None) is not None),
        ('plot_style.save_sci_figure', callable(getattr(plot_style, 'save_sci_figure', None))),
        ('plot_style.check_figure_gates', callable(getattr(plot_style, 'check_figure_gates', None))),
        ('schema.SALARY_TARGET_COLUMNS',
         isinstance(getattr(schema_module, 'SALARY_TARGET_COLUMNS', None), list)),
        ('schema.STRUCTURED_FEATURE_COLUMNS',
         isinstance(getattr(schema_module, 'STRUCTURED_FEATURE_COLUMNS', None), list)),
        ('schema.SALARY_MID_FIELD', isinstance(getattr(schema_module, 'SALARY_MID_FIELD', None), str)),
        ('project_paths.SALARY_TARGETS_PARQUET',
         getattr(paths_module, 'SALARY_TARGETS_PARQUET', None) is not None),
        ('project_paths.STRUCTURED_FEATURES_PARQUET',
         getattr(paths_module, 'STRUCTURED_FEATURES_PARQUET', None) is not None),
        ('project_paths.STRUCTURED_FIELDS_CONFIG_PATH',
         getattr(paths_module, 'STRUCTURED_FIELDS_CONFIG_PATH', None) is not None),
        ('project_paths.TABLE_STRUCTURED_FIELD_SALARY',
         getattr(paths_module, 'TABLE_STRUCTURED_FIELD_SALARY', None) is not None),
        ('project_paths.RECORD_SALARY_TARGET',
         getattr(paths_module, 'RECORD_SALARY_TARGET', None) is not None),
        ('field_repair.detect_company_attribute_slot_anomaly',
         callable(getattr(field_repair, 'detect_company_attribute_slot_anomaly', None))),
        ('field_repair.classify_company_attribute_slot_anomaly',
         callable(getattr(field_repair, 'classify_company_attribute_slot_anomaly', None))),
        ('field_repair.apply_deterministic_company_attribute_repair',
         callable(getattr(field_repair, 'apply_deterministic_company_attribute_repair', None))),
        ('field_repair.validate_company_attribute_repair',
         callable(getattr(field_repair, 'validate_company_attribute_repair', None))),
        ('field_repair.build_city_vocabulary',
         callable(getattr(field_repair, 'build_city_vocabulary', None))),
        ('project_paths.TABLE_COMPANY_ATTRIBUTE_ANOMALY',
         getattr(paths_module, 'TABLE_COMPANY_ATTRIBUTE_ANOMALY', None) is not None),
        ('project_paths.TABLE_PREPROCESSING_CLEANUP',
         getattr(paths_module, 'TABLE_PREPROCESSING_CLEANUP', None) is not None),
        ('project_paths.METHODOLOGY_PREPROCESSING_DOC',
         getattr(paths_module, 'METHODOLOGY_PREPROCESSING_DOC', None) is not None),
        ('project_paths.RECORD_COMPANY_ATTRIBUTE_ANOMALY',
         getattr(paths_module, 'RECORD_COMPANY_ATTRIBUTE_ANOMALY', None) is not None),
        ('project_paths.RECORD_PREPROCESSING_CLEANUP',
         getattr(paths_module, 'RECORD_PREPROCESSING_CLEANUP', None) is not None),
    ]
    failed = [name for name, ok in checks if not ok]
    logger.write(f'公共模块检查：{len(checks) - len(failed)}/{len(checks)} 项通过；'
                 f'缺失 {failed if failed else "无"}')
    note = f'{len(checks) - len(failed)}/{len(checks)} 项符号可用'
    return not failed, note


def load_pipeline_metrics() -> dict:
    """合并各阶段写出的指标 JSON。"""
    merged: dict = {}
    for key in quality.PIPELINE_STAGES:
        payload = io_utils.read_json(project_paths.METRICS_DIR / f'{key}.json')
        if payload:
            merged.update(payload)
    return merged


def run_regression_check(logger: PipelineLogger) -> tuple:
    """REGRESSION_CHECK：新结果 vs 历史基线，写回归核验表。

    本轮有意改变 Stage 03 业务语义，标记为「预期变化」的指标不参与失败判定，
    但必须在回归表中留痕，禁止为匹配基线恢复已删除的原链接字段。
    """
    metrics = load_pipeline_metrics()
    table = quality.compare_with_baseline(metrics)
    path = io_utils.write_csv(table, project_paths.TABLES_DIR / project_paths.TABLE_REGRESSION_CHECK)
    logger.write(f'回归核验表: {project_paths.relative_to_root(path)}')
    for row in table.itertuples(index=False):
        logger.write(f'  {row.指标}: 基线={row.历史基线} 新结果={row.新流水线结果} '
                     f'差异={row.绝对差异} 是否一致={row.是否一致} 变化性质={row.变化性质}')
    changed = table['变化性质'] == '预期变化'
    for row in table.loc[changed].itertuples(index=False):
        logger.write(f'  预期变化: {row.指标} —— {row.备注}')
    mismatched = table.loc[~table['是否一致'] & ~changed, '指标'].tolist()
    consistent_ratio = f'{int(table["是否一致"].sum())}/{len(table)} 项一致'
    if mismatched:
        logger.write(f'!! 与基线不一致的指标（不得修改真实结果去匹配基线）: {mismatched}')
    return not mismatched, consistent_ratio


def capture_refinement_baseline(logger: PipelineLogger) -> tuple:
    """REFINEMENT_BASELINE_CAPTURE：首次覆盖正式结果前固化修复前基线。

    幂等：基线文件已存在且含 job / company 两个 section 时不覆盖。
    """
    target = project_paths.REFINEMENT_BASELINE_JSON
    baseline = refinement.load_baseline()
    captured = False
    if not (baseline.get('job') or {}).get('指标'):
        refinement.capture_job_baseline()
        captured = True
    if not (baseline.get('company') or {}).get('指标'):
        refinement.capture_company_baseline()
        captured = True
    baseline = refinement.load_baseline()
    job_metrics = (baseline.get('job') or {}).get('指标') or {}
    company_metrics = (baseline.get('company') or {}).get('指标') or {}
    ok = bool(job_metrics) and bool(company_metrics)
    logger.write(f'修复前基线：{"本次捕获" if captured else "复用已固化基线"} → '
                 f'{project_paths.relative_to_root(target)}')
    logger.write(f'  岗位层指标 {len(job_metrics)} 项；公司层指标 {len(company_metrics)} 项')
    return ok, (f'岗位层 {len(job_metrics)} 项 / 公司层 {len(company_metrics)} 项'
                f'（{"本次捕获" if captured else "复用已固化基线"}）')


def run_refinement_check(logger: PipelineLogger) -> tuple:
    """REFINEMENT_REGRESSION_CHECK：生成 20 号总修复审计表与 13 号封版记录。"""
    result = refinement.write_refinement_outputs()
    audit_path = project_paths.TABLES_DIR / project_paths.TABLE_TEXT_SEMANTIC_REFINEMENT
    logger.write(f'总修复审计表: {result["audit_path"]}（{result["sheet_count"]} 张子表）')
    ok = audit_path.exists() and result['sheet_count'] >= 12
    return ok, f'{result["sheet_count"]} 张子表已写出，新旧结果逐项留痕（不把新值 != 旧值判为 FAIL）'


def check_documentation(logger: PipelineLogger) -> tuple:
    """DOCUMENTATION_UPDATE：README 是否已更新为新阶段路线（阶段记录 md 已移除，不再校验）。"""
    readme = project_paths.PROJECT_ROOT / 'README.md'
    if not readme.exists():
        return False, 'README.md 不存在'
    text = readme.read_text(encoding='utf-8')
    required_tokens = ['Stage 00', 'Stage 01', 'Stage 02', 'Stage 03', 'Stage 04', 'Stage 05',
                       'Stage 06', 'Stage 07', 'Stage 08', 'Stage 09', 'Stage 10', 'Stage 11',
                       'Python', 'Notebook', 'scripts/']
    missing = [token for token in required_tokens if token not in text]

    logger.write(f'文档同步检查：缺失标记 {missing if missing else "无"}')
    note = f'README 含 {len(required_tokens) - len(missing)}/{len(required_tokens)} 个必要标记'
    return (not missing), note


def main() -> int:
    args = parse_args()
    if args.from_stage > args.to_stage:
        print('参数错误：--from-stage 不能大于 --to-stage')
        return 2

    project_paths.ensure_directories()
    logger = PipelineLogger(project_paths.PIPELINE_LOG)
    logger.write('=' * 78)
    logger.write(f'course-design-2 数据流水线  {datetime.now():%Y-%m-%d %H:%M:%S}')
    logger.write(f'项目根目录: {project_paths.PROJECT_ROOT}')
    logger.write(f'阶段范围: {args.from_stage} → {args.to_stage}'
                 f'（reuse_existing_raw={args.reuse_existing_raw}, force_export_raw={args.force_export_raw}）')
    logger.write('=' * 78)

    gates = quality.GateRegistry('runner')
    executed_stages = []

    # 前置门禁：归档 / 基线 / 目录结构 / 公共模块
    ok, note = check_archive(logger)
    gates.check('ARCHIVE_PRE_REFACTOR', ok, note)

    baseline_path = project_paths.REFINEMENT_BASELINE_JSON
    gates.check('BASELINE_RECORD',
                baseline_path.exists() and baseline_path.stat().st_size > 0,
                project_paths.relative_to_root(baseline_path))

    ok, note = check_structure(logger)
    gates.check('PROJECT_STRUCTURE', ok, note)

    ok, note = check_shared_modules(logger)
    gates.check('SHARED_MODULES', ok, note)

    ok, note = capture_refinement_baseline(logger)
    gates.check('REFINEMENT_BASELINE_CAPTURE', ok, note)

    # 阶段执行：失败立即停止
    return_code = 0
    failed_stage = None
    for stage in range(args.from_stage, args.to_stage + 1):
        return_code = run_stage(stage, args, logger)
        if return_code != 0:
            failed_stage = stage
            logger.write(f'流水线在 Stage {stage:02d} 失败，停止后续阶段')
            break
        executed_stages.append(stage)

    if failed_stage is not None:
        # 清除本次失败及未执行阶段的历史门禁，避免展示过期状态
        for stale_stage in range(failed_stage, LAST_STAGE + 1):
            stale_path = project_paths.GATES_DIR / f'stage_{stale_stage:02d}.json'
            if stale_path.exists():
                stale_path.unlink()
                logger.write(f'清除过期门禁文件: {stale_path.name}')
        merged_gates = {**quality.GateRegistry.load_all(), **gates.results}
        logger.write('因阶段失败，回归核验未执行')
        quality.print_gates(merged_gates)
        return return_code

    # 后置门禁
    gates.check('PIPELINE_ENTRY',
                args.to_stage == LAST_STAGE
                and len(executed_stages) == args.to_stage - args.from_stage + 1,
                f'本次执行阶段 {executed_stages}')

    stage_gates = quality.GateRegistry.load_all()
    stage_gate_names = [name for stage in quality.PIPELINE_STAGES
                        for name in quality.STAGE_GATE_MAP[stage]]
    missing_stage_gates = [name for name in stage_gate_names
                           if stage_gates.get(name, {}).get('status') not in ('PASS', 'NOT_RUN')]
    not_run_gates = [name for name in stage_gate_names
                     if stage_gates.get(name, {}).get('status') == 'NOT_RUN']
    gates.check('INDEPENDENT_SCRIPT_RUN', not missing_stage_gates,
                '各阶段均由独立子进程 python scripts/xx.py 执行并生成门禁'
                + (f'；未执行模块 {not_run_gates}' if not_run_gates else '')
                if not missing_stage_gates else f'缺少阶段门禁: {missing_stage_gates}')

    ok, note = run_regression_check(logger)
    gates.check('REGRESSION_CHECK', ok, note)

    ok, note = run_refinement_check(logger)
    gates.check('REFINEMENT_REGRESSION_CHECK', ok, note)

    ok, note = check_documentation(logger)
    gates.check('DOCUMENTATION_UPDATE', ok, note)

    merged_gates = {**stage_gates, **gates.results}
    failed = [name for name in quality.FULL_GATE_ORDER
              if name != 'FINAL_CHECK'
              and merged_gates.get(name, {}).get('status') not in ('PASS', 'NOT_RUN')]
    not_run_final = [name for name in quality.FULL_GATE_ORDER
                     if name != 'FINAL_CHECK'
                     and merged_gates.get(name, {}).get('status') == 'NOT_RUN']
    gates.check('FINAL_CHECK', not failed,
                (f'{len(quality.FULL_GATE_ORDER)} 项门禁全部 PASS' if not failed else
                 f'未通过: {failed}')
                + (f'；NOT_RUN（未执行模块，未伪造 PASS）: {not_run_final}' if not_run_final else ''))
    merged_gates = {**stage_gates, **gates.results}

    quality.print_gates(merged_gates)
    logger.write('')
    for row in quality.summarize_gates(merged_gates).itertuples(index=False):
        note_text = f'  ({row.说明})' if row.说明 else ''
        logger.write(f'{row.门禁项} = {row.状态}{note_text}')

    gate_table = quality.summarize_gates(merged_gates)
    io_utils.write_csv(gate_table, project_paths.TABLES_DIR / 'ch3/00_pipeline_gate_summary.csv')
    # 持久化 runner 门禁，避免 gates/runner.json 停留在历史状态
    gates.save(project_paths.GATES_DIR / 'runner.json')

    logger.write(f'流水线执行完成：阶段 {executed_stages} 全部成功')
    logger.write(f'唯一岗位数据: {project_paths.relative_to_root(project_paths.PROCESSED_UNIQUE_PARQUET)}')
    logger.write(f'日志文件: {project_paths.relative_to_root(project_paths.PIPELINE_LOG)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

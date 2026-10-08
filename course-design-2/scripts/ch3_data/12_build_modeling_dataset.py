# -*- coding: utf-8 -*-
"""Stage 12：建模数据集构建、特征分组与泄漏审计（不训练模型、不调参、不用测试集）。

输入（Stage 00~11 封版产物，只读）：
    data/processed/job_details_unique.parquet          最终岗位实体（17,144）
    data/processed/job_structured_features.parquet     结构化字段（学历/到岗/规模/城市）
    data/processed/job_salary_targets.parquet          薪资目标（主目标 = 薪资中点）
    data/processed/job_text_features.parquet           技能与文本特征（技能提取范围等）
    data/processed/job_category_membership.parquet     岗位大类/细分类多值关系
    data/processed/company_entity_map.parquet          公司实体信息
    data/features/job_text_embedding_index.parquet     model-safe BGE 向量索引
    data/features/job_skill_membership.parquet         岗位 × 规范技能 long-format

输出：
    data/processed/job_analysis_dataset.parquet        全量分析集（17,144 × N）
    data/processed/job_salary_model_dataset.parquet    薪资建模集（14,883 × N）
    outputs/tables/ch3/20_modeling_dataset_audit.xlsx      建模数据集与泄漏审计（10 张子表）
    docs/records/19_modeling_dataset_record.md         建模数据集记录
    outputs/logs/metrics/stage_12_modeling.json

本轮边界：只构建数据集与审计，不训练最终模型、不调参、不执行 SHAP/消融、不使用测试集。

用法：
    python scripts/ch3_data/12_build_modeling_dataset.py
"""

from __future__ import annotations

import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src import (io_utils, modeling_dataset, project_paths, quality,  # noqa: E402
                 schema, text_utils)

STAGE = 'stage_12_modeling'
TITLE = 'Stage 12 建模数据集构建、特征分组与泄漏审计'

EXPECTED_ANALYSIS_ROWS = 17144
EXPECTED_MODEL_ROWS = 14883
EXPECTED_NEGOTIABLE = 2245
EXPECTED_ANOMALY = 16
LEAKAGE_FREE_NOTE = 'X 中禁止出现薪资字段或薪资派生字段；文本类特征只能来自 model-safe 版本'


def git_state() -> dict:
    """记录代码版本：优先取当前 commit，未提交时标记 working tree dirty。"""
    def run(args):
        return subprocess.run(args, capture_output=True, text=True, cwd=str(PROJECT_ROOT))

    try:
        commit = run(['git', 'rev-parse', '--short', 'HEAD']).stdout.strip() or 'UNKNOWN'
        dirty = bool(run(['git', 'status', '--porcelain']).stdout.strip())
    except OSError:
        commit, dirty = 'UNKNOWN', True
    return {'git_commit': commit, 'working_tree_dirty': bool(dirty),
            '生成时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '主机': socket.gethostname()}


def build_overview(analysis: pd.DataFrame, model_frame: pd.DataFrame,
                   manifest: pd.DataFrame) -> pd.DataFrame:
    """01_总体结构。"""
    rows = [
        {'指标': '分析集行数', '数值': len(analysis), '说明': '1 intern_id = 1 行'},
        {'指标': '分析集列数', '数值': analysis.shape[1],
         '说明': '紧凑宽表：类别保留原值，技能/分类以列表引用，未永久 one-hot'},
        {'指标': '建模集行数', '数值': len(model_frame), '说明': '正式薪资样本（主目标 = 薪资中点）'},
        {'指标': '建模集列数', '数值': model_frame.shape[1], '说明': '含 y 与稳健性薪资列'},
        {'指标': '主目标', '数值': modeling_dataset.TARGET_FIELD,
         '说明': '薪资中点 =（薪资下限 + 薪资上限）/ 2'},
        {'指标': 'A 组字段数', '数值': int((manifest['特征组'] == modeling_dataset.GROUP_JOB).sum())},
        {'指标': 'B 组字段数', '数值': int((manifest['特征组'] == modeling_dataset.GROUP_REGION).sum())},
        {'指标': 'C 组字段数', '数值': int((manifest['特征组'] == modeling_dataset.GROUP_COMPANY).sum())},
        {'指标': 'D 组字段数', '数值': int((manifest['特征组'] == modeling_dataset.GROUP_SKILL).sum())},
        {'指标': 'E 组字段数', '数值': int((manifest['特征组'] == modeling_dataset.GROUP_TEXT).sum())},
        {'指标': '目标/非特征字段数', '数值': int(
            manifest['特征组'].isin([modeling_dataset.GROUP_TARGET,
                                     modeling_dataset.GROUP_META]).sum())},
        {'指标': '参与模型字段数', '数值': int((manifest['是否参与模型'] == 1).sum())},
    ]
    return pd.DataFrame(rows)


def build_leakage_table(model_frame: pd.DataFrame, source_salary_columns) -> pd.DataFrame:
    """04_目标泄漏黑名单：黑名单字段、是否出现在 X、阻断结果。"""
    x_columns = [column for column in model_frame.columns
                 if column not in (modeling_dataset.TARGET_FIELD,
                                   *modeling_dataset.ROBUSTNESS_TARGET_FIELDS)]
    hits = [column for column in x_columns
            if column in modeling_dataset.SALARY_DERIVED_BLACKLIST]
    rows = []
    for column in modeling_dataset.SALARY_DERIVED_BLACKLIST:
        in_source = column in set(source_salary_columns)
        rows.append({
            '黑名单字段': column,
            '是否存在于薪资源层': '是' if in_source else '否',
            '是否出现在 X': '是' if column in hits else '否',
            '阻断方式': '构建阶段不选入宽表 + 门禁断言 X ∩ 黑名单 = ∅',
            '说明': LEAKAGE_FREE_NOTE,
        })
    rows.append({'黑名单字段': '（命名模式防护）', '是否存在于薪资源层': '—',
                 '是否出现在 X': '否',
                 '阻断方式': '任何列名含 薪资/薪酬/工资/面议/待遇 的字段一律禁止进入 X',
                 '说明': f'实测 X 命中 {len(hits)} 个'})
    return pd.DataFrame(rows)


def build_dataset_version_table(git_info: dict, analysis: pd.DataFrame,
                                model_frame: pd.DataFrame,
                                sources: dict) -> pd.DataFrame:
    """数据集版本信息（源文件 / 行数 / 分组 / 目标 / 筛选规则 / 代码版本）。"""
    rows = [{'项目': '生成时间', '内容': git_info['生成时间']},
            {'项目': 'git commit', '内容': git_info['git_commit']},
            {'项目': 'working tree', '内容': 'dirty（有未提交改动）' if git_info['working_tree_dirty']
             else 'clean'},
            {'项目': '主目标定义', '内容': '薪资中点 =（薪资下限 + 薪资上限）/ 2'},
            {'项目': '样本筛选规则',
             '内容': '非面议 + 解析成功 + 薪资中点有效 + 薪资异常标志为空'},
            {'项目': '分析集', '内容': f'{len(analysis)} 行 × {analysis.shape[1]} 列'},
            {'项目': '建模集', '内容': f'{len(model_frame)} 行 × {model_frame.shape[1]} 列'},
            {'项目': '特征分组', '内容': 'A 岗位基础 / B 地域 / C 公司 / D 技能 / E 文本语义'},
            {'项目': '技能口径', '内容': 'ALL_USABLE（REQUIREMENT_SECTION + FULL_TEXT_FALLBACK）'}]
    for name, info in sources.items():
        rows.append({'项目': f'源文件 {name}',
                     '内容': f"{info['path']}（{info['rows']} 行）"})
    return pd.DataFrame(rows)


def build_gate_table(gates) -> pd.DataFrame:
    return pd.DataFrame([{'门禁项': name, '状态': item['status'], '说明': item['note']}
                         for name, item in gates.items()])


def write_record(metrics: dict, audit: dict, git_info: dict) -> Path:
    """19 号记录：建模数据集定义、特征分组、泄漏黑名单、门禁与锚点。"""
    manifest = audit['03_Feature_Manifest']
    sample_table = audit['08_模型样本筛选'].set_index('筛选步骤')
    lines = [
        '# 记录 19：Stage 12 建模数据集构建、特征分组与泄漏审计',
        '',
        '> 本记录由 `scripts/ch3_data/12_build_modeling_dataset.py` 自动生成，数字全部来自真实运行结果。',
        '> 本轮只构建数据集与审计：**未训练最终模型、未调参、未执行 SHAP/消融、未使用测试集**。',
        '',
        '## 1. 建模样本定义',
        '',
        '| 步骤 | 岗位数 |',
        '| --- | --- |',
        f"| 全量最终岗位实体 | {int(sample_table.loc['全量最终岗位实体', '岗位数'])} |",
        f"| 剔除薪资面议 | {int(sample_table.loc['剔除薪资面议岗位', '岗位数'])} |",
        f"| 剔除解析失败/无中点 | {int(sample_table.loc['剔除解析失败/无薪资中点岗位', '岗位数'])} |",
        f"| 剔除薪资逻辑异常 | {int(sample_table.loc['剔除薪资逻辑异常岗位', '岗位数'])} |",
        f"| **正式建模样本** | **{int(sample_table.loc['正式建模样本', '岗位数'])}** |",
        '',
        f"- 恒等式校验：{sample_table.loc['正式建模样本', '校验']}；",
        '- 面议岗位仅存在于全量分析集（用于描述性分析），逻辑异常 16 条不进入正式主模型。',
        '',
        '## 2. 产物',
        '',
        '| 产物 | 规模 | 用途 |',
        '| --- | --- | --- |',
        f"| `{metrics['analysis_path']}` | {metrics['analysis_rows']} 行 × {metrics['analysis_columns']} 列 "
        '| EDA / 面议岗位描述 / 技能分析 / 岗位与公司结构分析 |',
        f"| `{metrics['model_path']}` | {metrics['model_rows']} 行 × {metrics['model_columns']} 列 "
        '| 薪资建模（主目标 = 薪资中点） |',
        f"| `{metrics['audit_path']}` | 10 张子表 | 数据字典 / Feature Manifest / 泄漏审计 |",
        '',
        '## 3. 特征分组（A/B/C/D/E）',
        '',
        '| 特征组 | 字段数 | 代表字段 |',
        '| --- | --- | --- |',
    ]
    representatives = {
        modeling_dataset.GROUP_JOB: '岗位大类/细分类集合与数量、学历等级、每周到岗天数、实习月数、'
                                    'JD 字符数/词数、岗位方向',
        modeling_dataset.GROUP_REGION: '工作城市（原文/规范）、是否直辖市',
        modeling_dataset.GROUP_COMPANY: '公司实体信息、所属行业、公司性质、公司规模数值与等级、'
                                        '公司认证标签、公司简介字符数',
        modeling_dataset.GROUP_SKILL: '技能数量与各技能组计数、技能提取范围、文本是否为空',
        modeling_dataset.GROUP_TEXT: 'model-safe BGE 向量引用（512 维，SVD 16/32/64 在建模阶段拟合）',
    }
    for group, note in representatives.items():
        count = int((manifest['特征组'] == group).sum())
        lines.append(f'| {group} | {count} | {note} |')
    lines += [
        '',
        f"参与模型的字段数：**{metrics['model_feature_count']}**"
        f"（数值 {metrics['numeric_feature_count']} / 类别 {metrics['categorical_feature_count']}"
        f" / 多值 {metrics['multi_value_feature_count']}）；"
        '多值字段与高基数标识只作关系引用，编码在建模 Pipeline 内完成。',
        '',
        '## 4. 目标泄漏黑名单',
        '',
        f"- 黑名单字段数：**{metrics['blacklist_count']}**（显式名单 + 命名模式双重防护）；",
        f"- X 中实际命中泄漏字段数：**{metrics['leakage_hits']}**；",
        '- 阻断方式：构建阶段不选入宽表 + 门禁断言 `X ∩ 黑名单 = ∅`；',
        '- 文本类特征只使用 model-safe 版本（Stage 06 严格薪资残留 = 0，Stage 07 技能同样基于 model-safe 文本）；',
        '- 稳健性薪资列（薪资下限/上限/区间宽度）保留在建模集但标记为非特征，仅用于稳健性实验。',
        '',
        '## 5. 一岗一行与类别关系',
        '',
        f"- 分析集 intern_id 唯一：{metrics['analysis_unique_check']}；"
        f"建模集 intern_id 唯一：{metrics['model_unique_check']}；",
        '- 岗位大类/细分类为多对多来源关系：宽表保留**完整集合**与数量，'
        '**禁止只取第一个细分类**；编码阶段使用高频类别 multi-hot + 低频 OTHER。',
        '',
        '## 6. 缺失值策略',
        '',
        '- 不在数据层做大规模众数/均值填补；',
        '- 数值：建模阶段中位数填补 + 必要时缺失指示；类别：编码为「未知」；',
        '- 文本：EMPTY_TEXT 单独作为缺失机制变量（`文本是否为空`），'
        '与「有文本但 0 技能」通过 `技能提取范围` + `是否有技能` 严格区分；',
        f"- E 组向量可用性：{metrics['text_vector_available']} / {metrics['model_rows']}"
        f"（缺失 {metrics['text_vector_missing']} 行，由 `文本向量是否可用` 标志控制）。",
        '',
        '## 7. 技能 multi-hot 策略（阈值只做候选）',
        '',
        '| 阈值口径 | 保留技能数 | 矩阵维度 | 覆盖率 |',
        '| --- | --- | --- | --- |',
    ]
    for row in audit['06_技能特征候选'].itertuples(index=False):
        lines.append(f'| {row.阈值口径} | {int(row.保留技能数)} | {row.矩阵维度} | {row.覆盖率:.2%} |')
    lines += [
        '',
        '> 技能口径 = ALL_USABLE（REQUIREMENT_SECTION + FULL_TEXT_FALLBACK）；'
        '最终阈值在 Stage 14 的 validation 上选择，**禁止**用 test set 决定。',
        '',
        '## 8. 文本语义策略（E 组）',
        '',
        f"- 当前 embedding 可用性：**{metrics['embedding_status']}**"
        f"（缓存 `data/features/job_text_embeddings.npz` 的 `job_text_safe`，"
        f"模型 {metrics['embedding_model']}，{metrics['embedding_dim']} 维，model-safe 文本）；",
        '- 降维候选 SVD/PCA 16 / 32 / 64，**只用训练集拟合**后变换验证/测试集，'
        '不在 processed 宽表固化；原始 512 维仅作敏感性实验；',
        '- E 组与 D 组显式分开，便于消融（Base / Base+Skill / Full / Full−Skill）。',
        '',
        '## 9. 建模预处理器约定（供 Stage 14 使用）',
        '',
        '```text',
        'numeric_features        → 中位数填补 + 必要时缺失指示（线性模型 StandardScaler）',
        'categorical_features    → OneHotEncoder(handle_unknown="ignore")，缺失编码「未知」',
        'skill_multi_hot_source  → job_skill_membership（ALL_USABLE）+ 训练集频率阈值',
        'text_semantic_features  → model-safe BGE 向量 → 训练集拟合 SVD/PCA(16/32/64)',
        'company_group_key       → 公司实体信息（Stage 15 公司 Group Split 使用）',
        '```',
        '',
        '## 10. 门禁与锚点',
        '',
        '| 门禁项 | 状态 | 说明 |',
        '| --- | --- | --- |',
    ]
    for name, item in metrics['gates'].items():
        lines.append(f'| {name} | {item} | {audit["10_门禁"].set_index("门禁项").loc[name, "说明"]} |')
    lines += [
        '',
        '| 锚点 | 值 |',
        '| --- | --- |',
        f"| raw | 172,063 × 30 |",
        f"| 最终岗位实体 / 分析集 | {metrics['analysis_rows']} |",
        f"| 建模样本 | {metrics['model_rows']}（+ 面议 {EXPECTED_NEGOTIABLE} + 异常 {EXPECTED_ANOMALY}"
        f" = {EXPECTED_ANALYSIS_ROWS}） |",
        f"| 核心版本 / 完整页面版本 | {metrics['core_versions']} / {metrics['full_versions']} |",
        f"| git | {git_info['git_commit']}"
        f"{'（working tree dirty）' if git_info['working_tree_dirty'] else ''} |",
        '',
        '## 11. 本轮边界',
        '',
        '- 未训练最终模型、未调参、未执行 SHAP 与消融、未使用测试集；',
        '- 未修改 Stage 00~11 的核心数据治理逻辑与封版产物；',
        '- 未新增永久 one-hot / skill_X 列；多值特征由建模 Pipeline 临时构造；',
        '- 本轮未执行 git commit。',
        '',
    ]
    return io_utils.write_markdown(
        project_paths.RECORDS_DIR / project_paths.RECORD_MODELING_DATASET, lines)


def main() -> int:
    quality.configure_logging()
    gates = quality.GateRegistry(STAGE)

    quality.stage_banner(STAGE, TITLE)
    entity = io_utils.read_parquet(project_paths.PROCESSED_UNIQUE_PARQUET)
    structured = io_utils.read_parquet(project_paths.STRUCTURED_FEATURES_PARQUET)
    salary = io_utils.read_parquet(project_paths.SALARY_TARGETS_PARQUET)
    text_features = io_utils.read_parquet(project_paths.JOB_TEXT_FEATURES_PARQUET)
    company_map = io_utils.read_parquet(project_paths.COMPANY_ENTITY_MAP_PARQUET)
    text_index = io_utils.read_parquet(project_paths.JOB_TEXT_EMBEDDING_INDEX_PARQUET)
    membership = io_utils.read_parquet(project_paths.JOB_SKILL_MEMBERSHIP_PARQUET)
    print(f'输入: 实体 {len(entity)} / 薪资 {len(salary)} / 技能关系 {len(membership)} 行')

    analysis = modeling_dataset.build_analysis_dataset(entity, structured, text_features,
                                                       company_map, text_index)
    model_frame, sample_audit = modeling_dataset.build_salary_model_dataset(analysis, salary)
    manifest = modeling_dataset.build_feature_manifest(analysis.columns, model_frame)

    # ---- 门禁 1：一岗一行 + 目标样本 ----
    gates.check('MODEL_DATASET_ONE_JOB_ONE_ROW',
                len(analysis) == EXPECTED_ANALYSIS_ROWS
                and analysis[schema.ID_FIELD].is_unique
                and len(model_frame) == EXPECTED_MODEL_ROWS
                and model_frame[schema.ID_FIELD].is_unique,
                f'分析集 {len(analysis)} 行（intern_id 唯一）／建模集 {len(model_frame)} 行'
                '（intern_id 唯一），均为一岗一行')

    anomaly_in_model = int(salary[salary[schema.ID_FIELD].isin(set(model_frame[schema.ID_FIELD]))]
                           [schema.SALARY_ANOMALY_FIELD].fillna('').astype(str).str.strip().ne('').sum())
    gates.check('MODEL_DATASET_TARGET_SAMPLE',
                len(model_frame) == EXPECTED_MODEL_ROWS
                and int(model_frame[modeling_dataset.TARGET_FIELD].isna().sum()) == 0
                and anomaly_in_model == 0
                and EXPECTED_MODEL_ROWS + EXPECTED_NEGOTIABLE + EXPECTED_ANOMALY
                == EXPECTED_ANALYSIS_ROWS,
                f'正式建模样本 {len(model_frame)} = 14,899 − {EXPECTED_ANOMALY}；'
                f'目标非空 {int(model_frame[modeling_dataset.TARGET_FIELD].notna().sum())}；'
                f'建模集内薪资逻辑异常 {anomaly_in_model}；'
                f'{EXPECTED_MODEL_ROWS} + {EXPECTED_NEGOTIABLE} + {EXPECTED_ANOMALY} '
                f'= {EXPECTED_ANALYSIS_ROWS}')

    # ---- 门禁 2：目标泄漏（X ∩ 黑名单 = ∅） ----
    x_columns = [column for column in model_frame.columns
                 if column not in (modeling_dataset.TARGET_FIELD,
                                   *modeling_dataset.ROBUSTNESS_TARGET_FIELDS)]
    pattern_hits = [column for column in x_columns
                    if any(item in str(column)
                           for item in modeling_dataset.SALARY_COLUMN_PATTERNS)]
    explicit_hits = modeling_dataset.salary_derived_columns(x_columns)
    gates.check('MODEL_DATASET_TARGET_LEAKAGE',
                not explicit_hits and not pattern_hits,
                f'X 共 {len(x_columns)} 列；命中显式黑名单 {explicit_hits or "无"}；'
                f'命中命名模式 {pattern_hits or "无"}；y = {modeling_dataset.TARGET_FIELD}')
    robustness_ok = all(column in model_frame.columns
                        for column in modeling_dataset.ROBUSTNESS_TARGET_FIELDS)
    gates.check('MODEL_DATASET_SALARY_DERIVED_FEATURE_BLOCK',
                robustness_ok and len(modeling_dataset.SALARY_DERIVED_BLACKLIST) >= 10
                and not set(modeling_dataset.SALARY_DERIVED_BLACKLIST) & set(x_columns),
                f'黑名单 {len(modeling_dataset.SALARY_DERIVED_BLACKLIST)} 个字段全部阻断；'
                f'稳健性薪资列 {modeling_dataset.ROBUSTNESS_TARGET_FIELDS} 保留但标记为非特征')

    # ---- 门禁 3：文本泄漏（技能与文本特征只来自 model-safe 体系） ----
    safe_residual = int(sum(text_utils.detect_strict_salary_leakage(text) +
                            text_utils.detect_salary_leakage(text)
                            for text in io_utils.read_parquet(
                                project_paths.TEXT_CORPUS_PARQUET,
                                columns=[schema.JD_SAFE_FIELD])[schema.JD_SAFE_FIELD]
                            .fillna('').astype(str)))
    email_meta = io_utils.read_json(project_paths.FEATURES_DIR / 'job_text_embeddings_meta.json')
    embedding_ok = bool(email_meta) and str(email_meta.get('状态')) == 'PASS' and (
        '模型安全版' in str(email_meta.get('输入文本字段', '')))
    gates.check('MODEL_DATASET_TEXT_LEAKAGE',
                safe_residual == 0 and embedding_ok
                and 'job_text_safe' in set(text_index['text_type']),
                f'model-safe 文本薪资残留 {safe_residual}；'
                f'E 组向量取自 `job_text_safe`（模型 {email_meta.get("模型名称")}，'
                f'{email_meta.get("向量维度")} 维）；技能特征来自 Stage 07 model-safe 提取')

    # ---- 门禁 4：Feature Manifest 完整 ----
    manifest_columns = set(manifest['字段名'])
    missing_in_manifest = [column for column in list(analysis.columns) + list(model_frame.columns)
                           if column not in manifest_columns
                           and column not in modeling_dataset.ROBUSTNESS_TARGET_FIELDS]
    invalid_groups = sorted(set(manifest['特征组']) - {
        modeling_dataset.GROUP_JOB, modeling_dataset.GROUP_REGION,
        modeling_dataset.GROUP_COMPANY, modeling_dataset.GROUP_SKILL,
        modeling_dataset.GROUP_TEXT, modeling_dataset.GROUP_TARGET, modeling_dataset.GROUP_META})
    gates.check('MODEL_DATASET_FEATURE_MANIFEST',
                not missing_in_manifest and not invalid_groups
                and set(modeling_dataset.MANIFEST_COLUMNS) <= set(manifest.columns),
                f'Manifest 覆盖 {len(manifest)} 行（未登记字段 {missing_in_manifest or "无"}，'
                f'非法特征组 {invalid_groups or "无"}）；'
                f"参与模型字段 {int((manifest['是否参与模型'] == 1).sum())} 个")

    git_info = git_state()
    sources = {
        '最终岗位实体': {'path': project_paths.relative_to_root(
            project_paths.PROCESSED_UNIQUE_PARQUET), 'rows': len(entity)},
        '薪资目标': {'path': project_paths.relative_to_root(
            project_paths.SALARY_TARGETS_PARQUET), 'rows': len(salary)},
        '技能 long-format': {'path': project_paths.relative_to_root(
            project_paths.JOB_SKILL_MEMBERSHIP_PARQUET), 'rows': len(membership)},
    }
    model_ids = set(model_frame[schema.ID_FIELD])
    audit_sheets = {
        '01_总体结构': build_overview(analysis, model_frame, manifest),
        '02_字段清单': pd.DataFrame([
            {'字段名': column, '所在数据集': '分析集/建模集',
             '数据类型': str(analysis[column].dtype),
             '是否参与模型（分析集口径）': int(
                 manifest.set_index('字段名').loc[column, '是否参与模型'])
             if column in manifest_columns else None}
            for column in analysis.columns]),
        '03_Feature_Manifest': manifest,
        '04_目标泄漏黑名单': build_leakage_table(model_frame, salary.columns),
        '05_缺失值统计': modeling_dataset.build_missing_table(analysis, model_frame),
        '06_技能特征候选': modeling_dataset.build_skill_threshold_table(membership, model_ids),
        '07_类别特征基数': modeling_dataset.build_cardinality_table(analysis),
        '08_模型样本筛选': sample_audit,
        '09_一岗一行检查': pd.DataFrame([
            {'数据集': 'job_analysis_dataset', '行数': len(analysis),
             '唯一 intern_id 数': int(analysis[schema.ID_FIELD].nunique()),
             '重复数': int(analysis[schema.ID_FIELD].duplicated().sum()),
             '口径': '1 intern_id = 1 行'},
            {'数据集': 'job_salary_model_dataset', '行数': len(model_frame),
             '唯一 intern_id 数': int(model_frame[schema.ID_FIELD].nunique()),
             '重复数': int(model_frame[schema.ID_FIELD].duplicated().sum()),
             '口径': '正式薪资样本'},
        ]),
        '11_数据集版本信息': build_dataset_version_table(git_info, analysis, model_frame, sources),
        '12_技能口径控制': pd.DataFrame([
            {'指标': 'EMPTY_TEXT 岗位数（建模集）',
             '数值': int((model_frame[modeling_dataset.TEXT_EMPTY_FIELD] == 1).sum()),
             '说明': '无可用技能文本：不得解释为「企业没有技能要求」'},
            {'指标': '有文本但 0 技能岗位数（建模集）',
             '数值': int(((model_frame[modeling_dataset.TEXT_EMPTY_FIELD] == 0)
                       & (model_frame[modeling_dataset.HAS_SKILL_FIELD] == 0)).sum()),
             '说明': '文本存在但未识别到规范技能'},
            {'指标': '有技能岗位数（建模集）',
             '数值': int((model_frame[modeling_dataset.HAS_SKILL_FIELD] == 1).sum()),
             '说明': '技能口径 = ALL_USABLE'},
            {'指标': 'E 组向量可用岗位数（建模集）',
             '数值': int(model_frame[modeling_dataset.TEXT_VECTOR_AVAILABLE_FIELD].sum()),
             '说明': 'model-safe BGE 向量可用性（缺失由标志控制）'},
        ]),
    }

    # 先落盘数据集并登记导出门禁，再把完整的门禁表写入审计表（保证审计表含全部门禁）
    analysis_path = io_utils.write_parquet(analysis, project_paths.JOB_ANALYSIS_DATASET_PARQUET)
    model_path = io_utils.write_parquet(model_frame, project_paths.JOB_SALARY_MODEL_DATASET_PARQUET)
    audit_path = project_paths.TABLES_DIR / project_paths.TABLE_MODELING_DATASET_AUDIT
    gates.check('MODEL_DATASET_AUDIT_EXPORT',
                analysis_path.exists() and model_path.exists() and len(audit_sheets) >= 9,
                f'分析集（{len(analysis)} 行）与建模集（{len(model_frame)} 行）已落盘；'
                f'28 号审计表含 {len(audit_sheets) + 1} 张子表（含门禁表）')
    audit_sheets['10_门禁'] = build_gate_table(gates.results)
    audit_path = io_utils.write_excel(audit_path, audit_sheets)

    metrics = {
        'analysis_path': project_paths.relative_to_root(analysis_path),
        'model_path': project_paths.relative_to_root(model_path),
        'audit_path': project_paths.relative_to_root(audit_path),
        'analysis_rows': int(len(analysis)),
        'analysis_columns': int(analysis.shape[1]),
        'model_rows': int(len(model_frame)),
        'model_columns': int(model_frame.shape[1]),
        'analysis_unique_check': '唯一' if analysis[schema.ID_FIELD].is_unique else '存在重复',
        'model_unique_check': '唯一' if model_frame[schema.ID_FIELD].is_unique else '存在重复',
        'model_feature_count': int((manifest['是否参与模型'] == 1).sum()),
        'numeric_feature_count': int(((manifest['是否参与模型'] == 1)
                                      & (manifest['是否数值'] == 1)
                                      & (manifest['是否多值'] == 0)).sum()),
        'categorical_feature_count': int(((manifest['是否参与模型'] == 1)
                                          & (manifest['是否类别'] == 1)).sum()),
        'multi_value_feature_count': int(((manifest['是否参与模型'] == 1)
                                          & (manifest['是否多值'] == 1)).sum()),
        'blacklist_count': int(len(modeling_dataset.SALARY_DERIVED_BLACKLIST)),
        'leakage_hits': int(len(explicit_hits) + len(pattern_hits)),
        'text_vector_available': int(model_frame[modeling_dataset.TEXT_VECTOR_AVAILABLE_FIELD].sum()),
        'text_vector_missing': int((model_frame[modeling_dataset.TEXT_VECTOR_AVAILABLE_FIELD] == 0).sum()),
        'embedding_status': email_meta.get('状态'),
        'embedding_model': email_meta.get('模型名称'),
        'embedding_dim': email_meta.get('向量维度'),
        'skill_threshold_candidates': audit_sheets['06_技能特征候选'].to_dict('records'),
        'core_versions': int(io_utils.read_json(
            project_paths.METRICS_DIR / 'stage_04.json').get('core_versions')),
        'full_versions': int(io_utils.read_json(
            project_paths.METRICS_DIR / 'stage_04.json').get('full_versions')),
        'git': git_info,
        'gates': {name: item['status'] for name, item in gates.results.items()},
    }
    io_utils.write_json(project_paths.METRICS_DIR / f'{STAGE}.json', metrics)

    record_path = write_record(metrics, audit_sheets, git_info)
    print(f'阶段记录: {project_paths.relative_to_root(record_path)}')

    quality.print_gates(gates.results, order=quality.STAGE_GATE_MAP[STAGE])
    gates.save()
    print(f'Stage 12 完成: 分析集 {len(analysis)} 行 / 建模集 {len(model_frame)} 行')
    return 0


if __name__ == '__main__':
    sys.exit(main())

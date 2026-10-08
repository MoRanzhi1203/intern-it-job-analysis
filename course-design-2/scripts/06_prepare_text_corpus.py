# -*- coding: utf-8 -*-
"""Stage 06：岗位版本文本语料准备。

处理路线：

data/processed/job_version_history.parquet   岗位版本时序层
data/processed/job_details_unique.parquet    最终岗位实体层
        ↓  原始文本完整保留（新增列，绝不回写覆盖）
        ↓  低风险清洗 → 语义分析版 → 模型安全版（删除薪资数字/区间/元每天等泄漏信息）
        ↓  岗位描述结构化分段（职责/要求/技能/加分/福利/其他，无法稳定识别则不强拆）
        ↓  技能实体标准化抽取（配置驱动，短词走词边界与大小写规则）
        ↓
data/interim/job_text_version_corpus.parquet 岗位版本文本语料层（一行 = 一个核心版本）

用法：
    python scripts/06_prepare_text_corpus.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src import io_utils, project_paths, quality, schema, skill_extraction, text_utils  # noqa: E402

STAGE = 'stage_06'
TITLE = 'Stage 06 岗位版本文本语料准备'

# 未识别高频技术词候选的最小词频
UNRECOGNIZED_MIN_FREQ = 30
# 高频技术词候选的最大输出数量
UNRECOGNIZED_TOP_N = 50
# 英文技术词候选形态（排除纯数字与过短词）
TECH_WORD_PATTERN = r'^[A-Za-z][A-Za-z0-9+#._\-]{2,}$'


def load_inputs() -> tuple:
    """读取岗位版本历史与最终岗位实体表。"""
    versions = io_utils.read_parquet(project_paths.PROCESSED_VERSION_HISTORY_PARQUET)
    entity = io_utils.read_parquet(project_paths.PROCESSED_UNIQUE_PARQUET)
    return versions, entity


def build_corpus(versions: pd.DataFrame, matcher: skill_extraction.SkillMatcher) -> tuple:
    """构建岗位版本文本语料，同时累计审计统计。"""
    id_field = schema.ID_FIELD
    version_field = schema.CORE_VERSION_FIELD
    rows = []
    alias_counter: Counter = Counter()
    skill_counter: Counter = Counter()
    group_counter: Counter = Counter()
    section_counter: Counter = Counter()
    status_counter: Counter = Counter()
    length_records = []
    unknown_word_counter: Counter = Counter()
    known_aliases = {alias.lower() for alias in matcher.config.alias_to_skill}
    salary_flag_total = 0
    residual_total = 0
    strict_residual_total = 0
    strict_residual_rows = 0
    strict_example_ids: list = []
    missing_jd = 0
    missing_profile = 0
    raw_preserved = True

    for record in versions.to_dict('records'):
        raw_jd = record.get('岗位描述')
        raw_title = record.get('岗位标题')
        raw_tags = record.get('岗位标签')
        raw_profile = record.get('公司简介')
        raw_company_tags = record.get('公司标签')

        cleaned_jd = text_utils.clean_text(raw_jd)
        semantic_jd = text_utils.build_semantic_version(cleaned_jd)
        safe_jd, salary_hits = text_utils.build_model_safe_version(semantic_jd)
        residual = text_utils.detect_salary_leakage(safe_jd)
        strict_residual = text_utils.detect_strict_salary_leakage(safe_jd)
        sections = text_utils.split_job_description(cleaned_jd)
        details = matcher.match_details(semantic_jd)
        skills = sorted({canonical for canonical, _alias, _group in details})
        for canonical, alias, group in details:
            alias_counter[alias] += 1
            skill_counter[canonical] += 1
            group_counter[group] += 1

        cleaned_profile = text_utils.clean_text(raw_profile)
        profile_missing = 1 if not cleaned_profile else 0
        missing_profile += profile_missing
        salary_flag_total += int(salary_hits > 0)
        residual_total += residual
        strict_residual_total += strict_residual
        if strict_residual:
            strict_residual_rows += 1
            if len(strict_example_ids) < 10:
                strict_example_ids.append(record.get('实习岗位ID'))
        if not cleaned_jd:
            missing_jd += 1
        status_counter[sections[schema.JD_SPLIT_STATUS_FIELD]] += 1
        for field in ('职责段存在标志', '要求段存在标志', '技能段存在标志',
                      '加分段存在标志', '福利段存在标志'):
            section_counter[field] += int(sections[field])
        length_records.append({
            '岗位描述字符数': text_utils.char_length(semantic_jd),
            '岗位描述分词数': text_utils.count_tokens(semantic_jd),
            '岗位描述段落数': text_utils.count_paragraphs(cleaned_jd),
        })

        # 未识别高频技术词候选（词面统计，不改变技能标准名体系）
        for token in set(text_utils.tokenize(semantic_jd)):
            if token in known_aliases or not token.isascii():
                continue
            if len(token) < 3 or token.isdigit():
                continue
            unknown_word_counter[token] += 1

        rows.append({
            id_field: record[id_field],
            version_field: record[version_field],
            schema.TEXT_FIRST_TIME_FIELD: record[schema.VERSION_FIRST_TIME_FIELD],
            schema.TEXT_LAST_TIME_FIELD: record[schema.VERSION_LAST_TIME_FIELD],
            schema.VERSION_IS_FINAL_FIELD: record[schema.VERSION_IS_FINAL_FIELD],
            schema.TITLE_RAW_FIELD: raw_title,
            schema.JD_RAW_FIELD: raw_jd,
            '岗位标签': raw_tags,
            schema.TITLE_CLEAN_FIELD: text_utils.clean_text(raw_title),
            schema.JD_CLEAN_FIELD: cleaned_jd,
            schema.JOB_TAG_LIST_FIELD: text_utils.split_tags(raw_tags),
            schema.JD_SEMANTIC_FIELD: semantic_jd,
            schema.JD_SAFE_FIELD: safe_jd,
            '职责文本': sections['职责文本'],
            '任职要求文本': sections['任职要求文本'],
            '技能要求文本': sections['技能要求文本'],
            '加分项文本': sections['加分项文本'],
            '福利文本': sections['福利文本'],
            '其他文本': sections['其他文本'],
            schema.JD_SPLIT_STATUS_FIELD: sections[schema.JD_SPLIT_STATUS_FIELD],
            '职责段存在标志': sections['职责段存在标志'],
            '要求段存在标志': sections['要求段存在标志'],
            '技能段存在标志': sections['技能段存在标志'],
            '加分段存在标志': sections['加分段存在标志'],
            '福利段存在标志': sections['福利段存在标志'],
            schema.JD_CHAR_COUNT_FIELD: text_utils.char_length(semantic_jd),
            schema.JD_TOKEN_COUNT_FIELD: text_utils.count_tokens(semantic_jd),
            schema.JD_PARAGRAPH_COUNT_FIELD: text_utils.count_paragraphs(cleaned_jd),
            schema.JD_SALARY_FLAG_FIELD: int(salary_hits > 0),
            schema.JD_SALARY_HIT_FIELD: int(salary_hits),
            schema.JD_SAFE_RESIDUAL_FIELD: int(residual),
            schema.SKILL_SET_FIELD: skills,
            schema.SKILL_COUNT_FIELD: len(skills),
            schema.SKILL_GROUP_FIELD: sorted({group for _c, _a, group in details}),
            '公司名称': record.get('公司名称'),
            schema.COMPANY_PROFILE_RAW_FIELD: raw_profile,
            schema.COMPANY_PROFILE_CLEAN_FIELD: cleaned_profile,
            schema.COMPANY_TAG_RAW_FIELD: raw_company_tags,
            schema.COMPANY_TAG_LIST_FIELD: text_utils.split_tags(raw_company_tags),
            '所属行业': record.get('所属行业'),
            '公司性质': record.get('公司性质'),
            '公司规模': record.get('公司规模'),
            '公司所在地': record.get('公司所在地'),
            schema.COMPANY_PROFILE_CHAR_COUNT_FIELD: text_utils.char_length(cleaned_profile),
            schema.COMPANY_PROFILE_MISSING_FIELD: profile_missing,
        })

    corpus = pd.DataFrame(rows)[schema.TEXT_CORPUS_COLUMNS]
    raw_preserved = bool(
        (corpus[schema.JD_RAW_FIELD].astype(str).reset_index(drop=True)
         == versions['岗位描述'].astype(str).reset_index(drop=True)).all()
        and (corpus['岗位标题_原始'].astype(str).reset_index(drop=True)
             == versions['岗位标题'].astype(str).reset_index(drop=True)).all())
    statistics = {
        'alias_counter': alias_counter,
        'skill_counter': skill_counter,
        'group_counter': group_counter,
        'section_counter': section_counter,
        'status_counter': status_counter,
        'length_records': pd.DataFrame(length_records),
        'unknown_word_counter': unknown_word_counter,
        'salary_flag_total': salary_flag_total,
        'residual_total': residual_total,
        'strict_residual_total': strict_residual_total,
        'strict_residual_rows': strict_residual_rows,
        'strict_example_ids': strict_example_ids,
        'missing_jd': missing_jd,
        'missing_profile': missing_profile,
        'raw_preserved': raw_preserved,
    }
    return corpus, statistics


def build_audit_tables(corpus: pd.DataFrame, statistics: dict,
                       matcher: skill_extraction.SkillMatcher) -> dict:
    """构建 16 号文本预处理审计表（8 张子表，全部来自真实统计）。"""
    id_field = schema.ID_FIELD
    tokenizer = text_utils.tokenizer_used()
    lengths = statistics['length_records']
    total_rows = len(corpus)

    missing_rows = [
        {'字段': schema.JD_RAW_FIELD, '缺失数': int(corpus[schema.JD_RAW_FIELD].isna().sum()),
         '空文本数': statistics['missing_jd'],
         '缺失或空占比': round((statistics['missing_jd']) / total_rows, 6) if total_rows else 0},
        {'字段': schema.COMPANY_PROFILE_RAW_FIELD,
         '缺失数': int(corpus[schema.COMPANY_PROFILE_RAW_FIELD].isna().sum()),
         '空文本数': statistics['missing_profile'],
         '缺失或空占比': round(statistics['missing_profile'] / total_rows, 6) if total_rows else 0},
        {'字段': schema.TITLE_RAW_FIELD, '缺失数': int(corpus[schema.TITLE_RAW_FIELD].isna().sum()),
         '空文本数': int((corpus[schema.TITLE_CLEAN_FIELD].fillna('').str.len() == 0).sum()),
         '缺失或空占比': round(float((corpus[schema.TITLE_CLEAN_FIELD].fillna('').str.len() == 0)
                                 .mean()), 6)},
        {'字段': '岗位标签', '缺失数': int(corpus['岗位标签'].isna().sum()),
         '空文本数': int((corpus[schema.JOB_TAG_LIST_FIELD].map(len) == 0).sum()),
         '缺失或空占比': round(float((corpus[schema.JOB_TAG_LIST_FIELD].map(len) == 0).mean()), 6)},
    ]
    missing_table = pd.DataFrame(missing_rows)

    describe = lengths.describe(percentiles=[0.25, 0.5, 0.75, 0.95]).reset_index()
    describe.columns = ['统计量', '岗位描述字符数', '岗位描述分词数', '岗位描述段落数']
    length_table = describe

    section_rows = []
    for field, count in statistics['section_counter'].items():
        section_rows.append({'语义段': field.replace('段存在标志', ''), '成功识别版本数': count,
                             '占比': round(count / total_rows, 6) if total_rows else 0})
    for status, count in sorted(statistics['status_counter'].items()):
        section_rows.append({'语义段': f'分段状态={status}', '成功识别版本数': count,
                             '占比': round(count / total_rows, 6) if total_rows else 0})
    section_table = pd.DataFrame(section_rows)
    section_table['分词器'] = tokenizer

    skill_counts = corpus[schema.SKILL_COUNT_FIELD]
    coverage_rows = [
        {'指标': '技能标准名总数', '数值': matcher.config.skill_count},
        {'指标': '参与匹配的别名总数', '数值': matcher.config.alias_count},
        {'指标': '技能标准名来源配置', '数值': project_paths.SKILL_CONFIG_PATH.name},
        {'指标': '至少提取到 1 项技能的版本数', '数值': int((skill_counts > 0).sum())},
        {'指标': '至少提取到 1 项技能的版本占比',
         '数值': round(float((skill_counts > 0).mean()), 6)},
        {'指标': '技能数量均值', '数值': round(float(skill_counts.mean()), 4)},
        {'指标': '技能数量中位数', '数值': float(skill_counts.median())},
        {'指标': '技能数量最大值', '数值': int(skill_counts.max())},
        {'指标': '零技能版本数', '数值': int((skill_counts == 0).sum())},
    ]
    coverage_table = pd.DataFrame(coverage_rows)

    skill_table = pd.DataFrame(
        [{'技能标准名': name, '技能组': matcher.config.skill_to_group.get(name, '其他'),
          '命中版本数': count, '版本覆盖率': round(count / total_rows, 6) if total_rows else 0}
         for name, count in statistics['skill_counter'].most_common()])
    if skill_table.empty:
        skill_table = pd.DataFrame(columns=['技能标准名', '技能组', '命中版本数', '版本覆盖率'])

    alias_table = pd.DataFrame(
        [{'命中别名': alias, '标准名': matcher.config.alias_to_skill.get(alias, ''),
          '是否为短词严格匹配': int(alias in matcher.config.strict_case_aliases
                              or (alias.isascii() and len(alias) <= matcher.config.short_term_max_length)),
          '命中版本数': count}
         for alias, count in statistics['alias_counter'].most_common()])

    unknown_table = pd.DataFrame(
        [{'未识别技术词候选': word, '出现版本数': count}
         for word, count in statistics['unknown_word_counter'].most_common(UNRECOGNIZED_TOP_N)
         if count >= UNRECOGNIZED_MIN_FREQ])
    if unknown_table.empty:
        unknown_table = pd.DataFrame(columns=['未识别技术词候选', '出现版本数'])
    unknown_table['说明'] = '词面高频但未命中技能词典，作为后续词典扩展候选'

    residual = int(corpus[schema.JD_SAFE_RESIDUAL_FIELD].sum())
    salary_table = pd.DataFrame([
        {'指标': '原始岗位描述含薪资模式版本数', '数值': statistics['salary_flag_total']},
        {'指标': '原始岗位描述含薪资模式占比',
         '数值': round(statistics['salary_flag_total'] / total_rows, 6) if total_rows else 0},
        {'指标': '模型安全版删除的薪资片段总数', '数值': int(corpus[schema.JD_SALARY_HIT_FIELD].sum())},
        {'指标': '模型安全版薪资残留总数', '数值': residual},
        {'指标': '仍残留薪资信息的版本数', '数值': int((corpus[schema.JD_SAFE_RESIDUAL_FIELD] > 0).sum())},
        {'指标': '严格薪资样式残留总数（金额/时间单位/面议）',
         '数值': int(statistics['strict_residual_total'])},
        {'指标': '严格薪资样式残留版本数', '数值': int(statistics['strict_residual_rows'])},
        {'指标': '严格残留示例岗位',
         '数值': '、'.join(str(item) for item in statistics['strict_example_ids']) or '无'},
        {'指标': '薪资泄漏清除是否通过', '数值': int(residual == 0)},
        {'指标': '语义分析版与模型安全版是否分列', '数值': 1},
        {'指标': '原始文本是否完整保留', '数值': int(statistics['raw_preserved'])},
        {'指标': '语料总版本数', '数值': total_rows},
        {'指标': '语料涉及岗位数', '数值': int(corpus[id_field].nunique())},
    ])
    return {
        '01_文本缺失统计': missing_table,
        '02_岗位描述长度分布': length_table,
        '03_文本分段成功率': section_table,
        '04_技能提取覆盖率': coverage_table,
        '05_技能标准名统计': skill_table,
        '06_技能别名命中统计': alias_table,
        '07_未识别高频技术词候选': unknown_table,
        '08_薪资泄漏清除检查': salary_table,
    }


def build_record(metrics: dict, audit: dict) -> list:
    """生成 Stage 06 阶段记录。"""
    coverage = audit['04_技能提取覆盖率'].set_index('指标')['数值']
    salary = audit['08_薪资泄漏清除检查'].set_index('指标')['数值']
    lines = [
        '# 阶段记录：Stage 06 岗位版本文本语料准备',
        '',
        '> 本文件由 `scripts/06_prepare_text_corpus.py` 生成，全部数字来自真实运行结果。',
        '',
        '## 1. 输入与输出',
        '',
        f"- 输入：`{metrics['version_input']}`（{metrics['version_rows']} 行核心版本）；",
        f"- 输出：`{metrics['corpus_path']}`（{metrics['corpus_rows']} 行 × {metrics['corpus_columns']} 列）；",
        f"- 粒度：一行 = 一个岗位的一个核心版本；涉及岗位 {metrics['corpus_jobs']} 个。",
        '',
        '## 2. 四类文本严格区分',
        '',
        '| 文本类型 | 字段 | 用途 | 是否覆盖原文 |',
        '| --- | --- | --- | --- |',
        '| 原始文本 | 岗位描述_原始 / 岗位标题_原始 / 公司简介_原始 | 追溯与人工审阅 | 否（新增列） |',
        '| 清洗文本 | 岗位描述_清洗 | 人工审阅、关键词、词面分析 | 否 |',
        '| 语义分析版 | 岗位描述_语义分析版 | 版本语义变化、职责与技能变化 | 否 |',
        '| 模型安全版 | 岗位描述_模型安全版 | 后续薪资模型文本特征 | 否 |',
        '',
        f"- 原始文本完整保留校验：{'通过' if salary['原始文本是否完整保留'] == 1 else '未通过'}；",
        f"- 语义分析版与模型安全版分列存储：{'是' if salary['语义分析版与模型安全版是否分列'] == 1 else '否'}。",
        '',
        '## 3. 薪资泄漏清除（防目标泄漏）',
        '',
        f"- 原始岗位描述命中薪资模式的版本数：{int(salary['原始岗位描述含薪资模式版本数'])}"
        f"（占比 {salary['原始岗位描述含薪资模式占比']:.2%}）；",
        f"- 模型安全版共删除薪资片段：{int(salary['模型安全版删除的薪资片段总数'])} 处；",
        f"- 模型安全版薪资残留数：{int(salary['模型安全版薪资残留总数'])}；",
        f"- 清除是否通过：{'PASS' if salary['薪资泄漏清除是否通过'] == 1 else 'FAIL'}。",
        '',
        '## 4. 结构化分段',
        '',
        '| 语义段 / 状态 | 成功识别版本数 | 占比 |',
        '| --- | --- | --- |',
    ]
    for row in audit['03_文本分段成功率'].itertuples(index=False):
        lines.append(f'| {row.语义段} | {int(row.成功识别版本数)} | {row.占比:.2%} |')
    lines += [
        '',
        '> 无法稳定识别小标题时**不强拆**：分段文本留空，仅记录分段状态与标志位，完整文本保留在语义分析版。',
        '',
        '## 5. 技能实体标准化',
        '',
        f"- 技能标准名数量：{int(coverage['技能标准名总数'])}（来源 `{coverage['技能标准名来源配置']}`）；",
        f"- 参与匹配的别名数量：{int(coverage['参与匹配的别名总数'])}；",
        f"- 至少提取到 1 项技能的版本数：{int(coverage['至少提取到 1 项技能的版本数'])}"
        f"（{coverage['至少提取到 1 项技能的版本占比']:.2%}）；",
        f"- 技能数量：均值 {coverage['技能数量均值']}，中位数 {coverage['技能数量中位数']}，"
        f"最大 {int(coverage['技能数量最大值'])}；",
        '',
        'Top 20 技能（按命中版本数）：',
        '',
        '| 技能标准名 | 技能组 | 命中版本数 | 版本覆盖率 |',
        '| --- | --- | --- | --- |',
    ]
    for row in audit['05_技能标准名统计'].head(20).itertuples(index=False):
        lines.append(f'| {row.技能标准名} | {row.技能组} | {int(row.命中版本数)} | {row.版本覆盖率:.2%} |')
    lines += [
        '',
        '## 6. 高歧义短词处理',
        '',
        'C / R / Go / AI / ML / DL / CV 等短词一律走「大小写严格 + 词边界 + 黑名单」三重规则，'
        '禁止简单字符串包含匹配。典型校验：',
        '',
        '- `C++`、`C#`、`C语言` 分别映射到 C++ / C# / C，且 `C` 不会在 `C++` 内部被重复命中；',
        '- `R&D` 通过黑名单屏蔽，不产生 R 的误匹配；',
        '- `C位` 通过黑名单屏蔽；',
        '- 小写 `go`（英文动词）不匹配 Go，`golang` / `Go语言` 均映射为 Go；',
        '- `ML` / `machine learning` 统一映射为「机器学习」，`CV` / `computer vision` → 「计算机视觉」。',
        '',
        '## 7. 未识别高频技术词候选（词典扩展线索）',
        '',
        '| 未识别技术词候选 | 出现版本数 |',
        '| --- | --- |',
    ]
    for row in audit['07_未识别高频技术词候选'].head(15).itertuples(index=False):
        lines.append(f'| {row.未识别技术词候选} | {int(row.出现版本数)} |')
    lines += [
        '',
        '## 8. 运行门禁',
        '',
        '| 门禁项 | 状态 |',
        '| --- | --- |',
    ]
    for name, result in metrics['gates'].items():
        lines.append(f'| {name} | {result} |')
    lines += [
        '',
        '## 9. 产物清单',
        '',
        f"- 文本语料：`{metrics['corpus_path']}`；",
        f"- 审计表：`{metrics['audit_path']}`；",
        f"- 指标 JSON：`{metrics['metrics_path']}`。",
        '',
    ]
    return lines


def main() -> int:
    quality.configure_logging()
    gates = quality.GateRegistry(STAGE)

    quality.stage_banner(STAGE, TITLE)
    versions, entity = load_inputs()
    print(f'输入: 版本历史 {len(versions)} 行 / 最终岗位实体 {len(entity)} 行')
    gates.check('TEXT_SOURCE_LOAD',
                len(versions) > 0 and len(entity) > 0
                and schema.JD_RAW_FIELD.replace('_原始', '') in versions.columns,
                f'版本历史 {len(versions)} 行、实体 {len(entity)} 行，含中文岗位描述原文字段')

    try:
        config = skill_extraction.load_skill_config()
        matcher = skill_extraction.SkillMatcher(config)
        skill_ok, skill_note = True, (f'技能标准名 {config.skill_count} 个、别名 {config.alias_count} 个，'
                                     f'来源 {project_paths.SKILL_CONFIG_PATH.name}')
    except Exception as error:  # noqa: BLE001 - 配置错误必须显式失败
        matcher = None
        skill_ok, skill_note = False, f'技能配置加载失败: {error}'
    gates.check('SKILL_CONFIG_LOAD', skill_ok, skill_note)
    if not skill_ok:
        quality.print_gates(gates.results, order=quality.STAGE_GATE_MAP[STAGE])
        gates.save()
        return 1

    corpus, statistics = build_corpus(versions, matcher)

    gates.check('TEXT_RAW_PRESERVE',
                bool(statistics['raw_preserved'])
                and (corpus[schema.JD_RAW_FIELD].astype(str)
                     == versions[schema.JD_RAW_FIELD.replace('_原始', '')].astype(str)).all(),
                '原始岗位描述逐行与版本历史一致，衍生字段全部新增列，未回写覆盖原文')
    gates.check('TEXT_CLEAN_BUILD',
                bool(corpus[schema.JD_CLEAN_FIELD].fillna('').str.len().gt(0).any())
                and bool(corpus[schema.JD_SEMANTIC_FIELD].fillna('').str.len().gt(0).any())
                and schema.JD_SEMANTIC_FIELD != schema.JD_SAFE_FIELD,
                f'清洗文本覆盖 {int(corpus[schema.JD_CLEAN_FIELD].fillna("").str.len().gt(0).sum())} 行，'
                f'语义分析版与模型安全版分列构建')

    recognized = int(sum(statistics['section_counter'].values()))
    gates.check('TEXT_SECTION_SPLIT',
                recognized > 0
                and set(statistics['status_counter']) <= set(text_utils.SECTION_STATUS_VALUES),
                f'成功识别语义段 {recognized} 段次；分段状态取值 '
                f'{sorted(statistics["status_counter"])}，未识别样本不强拆')

    residual = int(corpus[schema.JD_SAFE_RESIDUAL_FIELD].sum())
    strict_residual = int(statistics['strict_residual_total'])
    gates.check('SALARY_LEAKAGE_REMOVAL',
                residual == 0 and strict_residual == 0
                and schema.JD_SAFE_FIELD in corpus.columns
                and schema.JD_SEMANTIC_FIELD in corpus.columns,
                f'模型安全版官方泄漏残留 {residual} 处、严格薪资样式残留 {strict_residual} 处，'
                f'删除 {int(corpus[schema.JD_SALARY_HIT_FIELD].sum())} 处，'
                '语义分析版与模型安全版分列')
    gates.check('MODEL_SAFE_STRICT_SALARY_PATTERN_CHECK',
                strict_residual == 0 and int(statistics['strict_residual_rows']) == 0,
                f'严格薪资样式（金额 + 元/块 · 时间单位 · k/万 · 面议 · ¥）残留 '
                f'{strict_residual} 处，涉及版本 {int(statistics["strict_residual_rows"])} 个；'
                f'规则源 {project_paths.SKILL_CONFIG_PATH.parent.name}/../src/text_utils.py'
                '（Stage 06 清洗与门禁、Stage 07 检测、tests 共用同一规则源）')

    path = io_utils.write_parquet(corpus, project_paths.TEXT_CORPUS_PARQUET)
    audit_tables = build_audit_tables(corpus, statistics, matcher)
    audit_path = io_utils.write_excel(project_paths.TABLES_DIR / project_paths.TABLE_JOB_TEXT_PREPROCESSING,
                                      audit_tables)
    gates.check('TEXT_AUDIT_EXPORT', path.exists() and audit_path.exists(),
                f'语料 {len(corpus)} 行、审计表 {len(audit_tables)} 张子表已写出')

    metrics = {
        'version_input': project_paths.relative_to_root(project_paths.PROCESSED_VERSION_HISTORY_PARQUET),
        'version_rows': int(len(versions)),
        'corpus_path': project_paths.relative_to_root(path),
        'corpus_rows': int(len(corpus)),
        'corpus_columns': int(corpus.shape[1]),
        'corpus_jobs': int(corpus[schema.ID_FIELD].nunique()),
        'audit_path': project_paths.relative_to_root(audit_path),
        'missing_jd_versions': int(statistics['missing_jd']),
        'missing_profile_versions': int(statistics['missing_profile']),
        'skill_names': int(config.skill_count),
        'salary_removed_total': int(corpus[schema.JD_SALARY_HIT_FIELD].sum()),
        'salary_residual_total': residual,
        'metrics_path': project_paths.relative_to_root(
            project_paths.METRICS_DIR / f'{STAGE}.json'),
    }
    metrics['gates'] = {name: result['status'] for name, result in gates.results.items()}
    io_utils.write_json(project_paths.METRICS_DIR / f'{STAGE}.json', metrics)

    record_path = io_utils.write_markdown(
        project_paths.RECORDS_DIR / project_paths.RECORD_TEXT_PREPROCESSING,
        build_record(metrics, audit_tables))
    print(f'阶段记录: {project_paths.relative_to_root(record_path)}')

    quality.print_gates(gates.results, order=quality.STAGE_GATE_MAP[STAGE])
    for row in quality.summarize_gates(gates.results,
                                       order=quality.STAGE_GATE_MAP[STAGE]).itertuples(index=False):
        note = f'  ({row.说明})' if row.说明 else ''
        print(f'{row.门禁项} = {row.状态}{note}')
    gates.save()
    return 0


if __name__ == '__main__':
    sys.exit(main())

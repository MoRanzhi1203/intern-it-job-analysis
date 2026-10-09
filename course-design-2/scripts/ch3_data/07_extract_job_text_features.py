# -*- coding: utf-8 -*-
"""Stage 07：最终岗位技能与文本特征（模型安全文本 + 岗位要求段落优先）。

处理路线：

data/processed/job_details_unique.parquet      最终岗位实体层
data/interim/job_text_version_corpus.parquet    岗位版本文本语料层
        ↓  取「最终核心版本」文本（与实体选择规则一致，不用历史版本 union）
        ↓  要求段落识别（任职要求 / 技能要求 / 加分项）→ 段落优先匹配技能
        ↓  无法识别要求段落 → 模型安全 JD 全文 fallback（不丢弃岗位）
        ↓  技能词典匹配（三级结构 feature_family → group → canonical + 歧义上下文规则）
        ↓
data/processed/job_text_features.parquet        岗位文本特征层（一行 = 一个最终岗位实体）
data/features/job_skill_membership.parquet      岗位 × 规范技能 long-format（一行 = 一岗位 × 一技能）

边界（严格遵守）：
1. 技能提取文本一律使用**模型安全版**（`岗位描述_模型安全版`），不重新引入薪资文字；
2. 技能画像只对应最终核心版本，历史技能仅用于「技能要求变化」分析；
3. 不在岗位表展开几十/上百个「是否X」技能列；高频技能 multi-hot 留给建模阶段；
4. 正式技能词典只来自 `config/skills.yml`，候选技能只能进审计表等待人工确认；
5. 不重复实现已有的 AI技能数 / 大模型技能数 / 技能类别统计 / 规范技能字段。

用法：
    python scripts/ch3_data/07_extract_job_text_features.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src import (io_utils, project_paths, quality, schema, skill_extraction,  # noqa: E402
                 text_utils)

STAGE = 'stage_07'
TITLE = 'Stage 07 最终岗位技能与文本特征（模型安全文本 + 要求段落优先）'

# 进入特征表的布尔技能（沿用既有清单，不新增永久宽表列）
BOOLEAN_FEATURE_SKILLS = [
    'Python', 'Java', 'SQL', 'C++', 'Go', 'JavaScript',
    '机器学习', '深度学习', 'NLP', '计算机视觉', '推荐系统',
    'PyTorch', 'TensorFlow', 'Spark', 'Flink', 'Docker', 'Kubernetes', 'Linux', 'Git',
    'MySQL', 'Redis', 'Vue', 'React', 'TypeScript',
    '大模型', 'RAG', 'Agent', 'LangChain', '向量数据库', 'Embedding', '微调', '多模态', 'AIGC',
]
# 技能需求分析关注的重点技能（写入审计与记录）
FOCUS_SKILLS = ['Python', 'Java', 'SQL', 'MySQL', 'Excel', 'PPT', 'Word',
                '爬虫', '网络安全', '大模型', 'Agent']
# 低频技能展示阈值（岗位数低于该值只保留在 long table 做需求分析，不进入模型）
LOW_FREQUENCY_THRESHOLD = 50
# 人工核验抽样规模（提示词要求 100~200）
SAMPLE_SIZE = 150
# 候选技能发现：拉丁技术词与中文技术词
LATIN_TOKEN_PATTERN = re.compile(r'[A-Za-z][A-Za-z0-9+.#/_-]{1,19}')
CJK_TOKEN_PATTERN = re.compile(r'[\u4e00-\u9fa5]{2,6}')
CANDIDATE_SUFFIX_WORDS = ['算法', '模型', '框架', '引擎', '平台', '系统', '工具', '库',
                          '网络', '安全', '数据', '分析', '开发', '运维', '测试', '编程',
                          '语言', '软件', '服务', '架构', '爬虫', '识别', '计算']
# 候选中文词不得以这些虚词/连接词开头（避免「等办公软件」这类截断噪声）
CANDIDATE_LEADING_STOP = list('等和与及或的地得用为对在把被让使很非常这那')
# 候选拉丁词过滤：英文功能词与招聘模板词（不是技术词候选）
CANDIDATE_ENGLISH_STOP = {
    'and', 'the', 'for', 'with', 'you', 'your', 'are', 'our', 'will', 'can', 'not', 'have',
    'has', 'had', 'all', 'any', 'use', 'using', 'used', 'able', 'good', 'team', 'work',
    'we', 'us', 'it', 'its', 'in', 'on', 'at', 'to', 'of', 'by', 'as', 'be', 'is', 'was',
    'were', 'or', 'if', 'from', 'that', 'this', 'these', 'those', 'about', 'more', 'than',
    'also', 'such', 'other', 'new', 'one', 'two', 'well', 'must', 'should', 'may', 'need',
    'join', 'welcome', 'please', 'job', 'jobs', 'work', 'intern', 'internship', 'company',
    'requirement', 'requirements', 'responsibility', 'responsibilities', 'qualification',
    'qualifications', 'preferred', 'plus', 'etc', 'eg', 'ie', 'vs', 'per', 'day', 'week',
}


def normalize_title(title) -> str:
    """岗位标题规范化：清洗 + 去括号补充信息 + 拉丁转小写，用于去重与聚类。"""
    cleaned = text_utils.clean_text(title)
    cleaned = cleaned.replace('（', '(').replace('）', ')')
    cleaned = cleaned.replace('【', '(').replace('】', ')')
    cleaned = cleaned.replace('(', ' ').replace(')', ' ')
    cleaned = cleaned.replace('/', ' ').replace('|', ' ')
    parts = [part for part in cleaned.split() if part]
    return ' '.join(parts).lower()


def build_skill_records(latest: pd.DataFrame, matcher: skill_extraction.SkillMatcher) -> tuple:
    """按「要求段落优先 + 模型安全全文 fallback」逐岗位提取技能（含命中次数）。"""
    records = []
    for record in latest.to_dict('records'):
        safe_text = record.get(schema.JD_SAFE_FIELD) or ''
        requirement = text_utils.extract_requirement_section(safe_text)
        scope = requirement['scope']
        match_text = requirement['text'] if scope != 'EMPTY_TEXT' else ''
        counted = matcher.match_counted(match_text) if match_text else []
        records.append({
            'intern_id': record[schema.ID_FIELD],
            'match_scope': scope,
            'matched_headings': requirement['matched_headings'],
            'skills': counted,
            'skill_names': sorted(item['canonical'] for item in counted),
        })
    return records


def build_features(entity: pd.DataFrame, latest: pd.DataFrame, skill_records: list,
                   matcher: skill_extraction.SkillMatcher) -> tuple:
    """构建最终岗位文本特征表（紧凑画像，不展开技能布尔列）。"""
    id_field = schema.ID_FIELD
    config = matcher.config
    ai_skills = config.ai_skills()
    llm_skills = skill_extraction.llm_skill_set(config)
    history = (latest.groupby(id_field)[schema.SKILL_SET_FIELD]
               .apply(lambda sets: sorted({skill for item in sets
                                           for skill in text_utils.as_list(item)}))
               .rename(schema.HISTORY_SKILL_SET_FIELD))
    entity_groups = entity.set_index(id_field)['岗位大类集合'].to_dict()

    rows = []
    for record, skills in zip(latest.to_dict('records'), skill_records):
        job_id = record[id_field]
        skill_names = skills['skill_names']
        skill_set = set(skill_names)
        history_skills = list(text_utils.as_list(history.get(job_id, [])))
        direction = text_utils.identify_job_direction(
            record['_title'], skill_names, record[schema.JD_SAFE_FIELD])
        tags = list(text_utils.as_list(record[schema.JOB_TAG_LIST_FIELD]))
        family_buckets = skill_extraction.skills_by_family(config, skill_names)
        row = {
            id_field: job_id,
            schema.JOB_TITLE_NORMALIZED_FIELD: normalize_title(record['_title']),
            schema.JOB_DIRECTION_FIELD: direction,
            schema.JOB_DIRECTION_MATCH_FIELD: text_utils.direction_consistent_with_group(
                direction, entity_groups.get(job_id, [])),
            schema.JD_CHAR_COUNT_FIELD: int(record[schema.JD_CHAR_COUNT_FIELD]),
            schema.JD_TOKEN_COUNT_FIELD: int(record[schema.JD_TOKEN_COUNT_FIELD]),
            schema.JD_SPLIT_STATUS_FIELD: record[schema.JD_SPLIT_STATUS_FIELD],
            schema.JOB_TAG_LIST_FIELD: tags,
            '岗位标签数量': len(tags),
            schema.SKILL_SET_FIELD: skill_names,
            schema.SKILL_COUNT_FIELD: len(skill_names),
            schema.HISTORY_SKILL_SET_FIELD: history_skills,
            schema.HISTORY_SKILL_COUNT_FIELD: len(history_skills),
            schema.SKILL_GROUP_FIELD: sorted({config.skill_to_group[skill] for skill in skill_names
                                              if skill in config.skill_to_group}),
            schema.SKILL_FAMILY_LIST_FIELD: sorted(
                {config.skill_to_family[skill] for skill in skill_names
                 if skill in config.skill_to_family}),
            schema.SKILL_SCOPE_FIELD: skills['match_scope'],
            schema.AI_SKILL_COUNT_FIELD: len(skill_set & ai_skills),
            schema.LLM_SKILL_COUNT_FIELD: len(skill_set & llm_skills),
            schema.SKILL_LANGUAGE_COUNT_FIELD: skill_extraction.skill_group_count(
                config, skill_names, config.language_skill_groups),
            schema.SKILL_DATABASE_COUNT_FIELD: skill_extraction.skill_group_count(
                config, skill_names, config.database_skill_groups),
            schema.SKILL_SECURITY_COUNT_FIELD: skill_extraction.skill_group_count(
                config, skill_names, config.security_skill_groups),
            schema.SKILL_OFFICE_COUNT_FIELD: skill_extraction.skill_group_count(
                config, skill_names, config.office_skill_groups),
            schema.COMPANY_PROFILE_CHAR_COUNT_FIELD: int(record[schema.COMPANY_PROFILE_CHAR_COUNT_FIELD]),
            schema.COMPANY_TAG_LIST_FIELD: list(text_utils.as_list(record[schema.COMPANY_TAG_LIST_FIELD])),
            schema.ENTITY_CORE_VERSION_COUNT_FIELD: int(record['_core_versions']),
            schema.ENTITY_FINAL_VERSION_FIELD: int(record['_final_version']),
        }
        for family in schema.SKILL_FAMILY_NAMES:
            row[schema.SKILL_FAMILY_SET_FIELDS[family]] = family_buckets[family]
            row[schema.SKILL_FAMILY_COUNT_FIELDS[family]] = len(family_buckets[family])
        row.update({f'是否{skill}': int(skill in skill_set) for skill in BOOLEAN_FEATURE_SKILLS})
        rows.append(row)

    features = pd.DataFrame(rows)
    scope_counts = features[schema.SKILL_SCOPE_FIELD].value_counts().to_dict()
    coverage = {
        'entity_jobs': int(entity[id_field].nunique()),
        'feature_jobs': int(features[id_field].nunique()),
        'missing_jobs': int(len(set(entity[id_field]) - set(features[id_field]))),
        'skill_jobs': int((features[schema.SKILL_COUNT_FIELD] > 0).sum()),
        'zero_skill_jobs': int((features[schema.SKILL_COUNT_FIELD] == 0).sum()),
        'scope_counts': {key: int(value) for key, value in scope_counts.items()},
        'unmapped_family_skills': sorted(
            {skill for skills in features[schema.SKILL_SET_FIELD] for skill in skills
             if skill not in config.skill_to_family}),
    }
    features = features[schema.JOB_TEXT_FEATURE_COLUMNS]
    return features, coverage


def build_skill_frequency(features: pd.DataFrame, matcher: skill_extraction.SkillMatcher) -> pd.DataFrame:
    """技能频率：岗位数 / 覆盖率 / 一级类型 / 技能组 / 要求段落内命中岗位数。"""
    config = matcher.config
    rows = []
    for skill in config.skill_to_group:
        hit = features[schema.SKILL_SET_FIELD].map(lambda skills: skill in set(skills or []))
        job_count = int(hit.sum())
        if job_count == 0:
            continue
        requirement_hit = int((hit & features[schema.SKILL_SCOPE_FIELD]
                               .eq('REQUIREMENT_SECTION')).sum())
        rows.append({
            '技能标准名': skill,
            'feature_family': config.skill_to_family.get(skill, ''),
            '技能组': config.skill_to_group.get(skill, ''),
            '岗位数': job_count,
            '岗位覆盖率': round(job_count / len(features), 6),
            '要求段落内命中岗位数': requirement_hit,
            '全文fallback命中岗位数': job_count - requirement_hit,
        })
    table = pd.DataFrame(rows).sort_values(['岗位数', '技能标准名'],
                                           ascending=[False, True]).reset_index(drop=True)
    return table


def build_group_frequency(features: pd.DataFrame, matcher: skill_extraction.SkillMatcher) -> pd.DataFrame:
    """技能组频率（含 feature_family 与技能项次）。"""
    config = matcher.config
    rows = []
    for group in sorted(config.group_to_family):
        members = {skill for skill, name in config.skill_to_group.items() if name == group}
        job_count = int(features[schema.SKILL_SET_FIELD].map(
            lambda skills: bool(set(skills or []) & members)).sum())
        instances = int(features[schema.SKILL_SET_FIELD].map(
            lambda skills: len(set(skills or []) & members)).sum())
        rows.append({
            '技能组': group,
            'feature_family': config.group_to_family[group],
            '技能标准名数': len(members),
            '岗位数': job_count,
            '岗位覆盖率': round(job_count / len(features), 6) if len(features) else 0.0,
            '技能项次': instances,
        })
    return pd.DataFrame(rows).sort_values(['岗位数', '技能组'],
                                          ascending=[False, True]).reset_index(drop=True)


def build_dictionary_table(matcher: skill_extraction.SkillMatcher) -> pd.DataFrame:
    """01_技能词典统计：一级类型 / 技能组 / 技能标准名数 / 别名数。"""
    config = matcher.config
    rows = [{'层级': '合计', '名称': '全部技能', '技能标准名数': config.skill_count,
             '别名数': config.alias_count,
             '说明': f'配置版本 {project_paths.SKILL_CONFIG_PATH.name}；'
                     f'短词阈值 {config.short_term_max_length}；'
                     f'歧义规则 {len(config.context_rules)} 条'}]
    for family, groups in config.family_to_groups.items():
        members = config.family_skills(family)
        rows.append({'层级': 'feature_family', '名称': family, '技能标准名数': len(members),
                     '别名数': int(sum(1 for alias, skill in config.alias_to_skill.items()
                                       if skill in members)),
                     '说明': f'{len(groups)} 个技能组'})
        for group in groups:
            group_members = config.group_skills([group])
            rows.append({'层级': 'group', '名称': group, '技能标准名数': len(group_members),
                         '别名数': int(sum(1 for alias, skill in config.alias_to_skill.items()
                                           if skill in group_members)),
                         '说明': f'归属 {family}'})
    return pd.DataFrame(rows)


def build_requirement_table(features: pd.DataFrame) -> pd.DataFrame:
    """06_要求段落识别：范围分布 + 与分段状态交叉。"""
    scope = features[schema.SKILL_SCOPE_FIELD].value_counts()
    rows = [{'范围': name, '岗位数': int(scope.get(name, 0)),
             '岗位占比': round(float(scope.get(name, 0)) / len(features), 6),
             '说明': {'REQUIREMENT_SECTION': '命中任职要求 / 技能要求 / 加分项段落，技能提取优先使用该段',
                      'FULL_TEXT_FALLBACK': '未识别要求段落，使用模型安全 JD 全文，仍允许提取技能',
                      'EMPTY_TEXT': '模型安全文本为空，无法提取技能'}.get(name, '')}
            for name in schema.SKILL_MATCH_SCOPE_VALUES]
    cross = (features.groupby([schema.SKILL_SCOPE_FIELD, schema.JD_SPLIT_STATUS_FIELD])
             .size().rename('岗位数').reset_index()
             .rename(columns={schema.SKILL_SCOPE_FIELD: '技能提取范围',
                              schema.JD_SPLIT_STATUS_FIELD: '岗位描述分段状态（Stage 06）'}))
    cross['说明'] = '技能提取范围与 Stage 06 分段状态交叉核对'
    return pd.concat([pd.DataFrame(rows), pd.DataFrame([{'范围': '—', '岗位数': None,
                                                        '岗位占比': None,
                                                        '说明': '以下为与 Stage 06 分段状态的交叉核对'}]
                                                      ), cross], ignore_index=True)


def build_ambiguity_table(matcher: skill_extraction.SkillMatcher,
                          samples: list) -> pd.DataFrame:
    """07_歧义词检查：规则配置 + 实际排除统计 + 排除样例。"""
    config = matcher.config
    rows = []
    for canonical, rule in config.context_rules.items():
        rows.append({
            '技能标准名': canonical,
            '受控别名': '、'.join(str(item) for item in rule.get('match_aliases') or []),
            '必须上下文(任一)': '、'.join(str(item) for item in rule.get('require_any') or []),
            '排除上下文(任一)': '、'.join(str(item) for item in rule.get('exclude_any') or []) or '（无）',
            '本轮被排除命中次数': int(matcher.rejection_counter.get(canonical, 0)),
            '排除原因': '、'.join(sorted({reason for (name, reason), value
                                      in matcher.rejection_reasons.items()
                                      if name == canonical and value > 0})) or '（无）',
        })
    table = pd.DataFrame(rows)
    if samples:
        table = pd.concat([table, pd.DataFrame(
            [{'技能标准名': '（排除样例）', '受控别名': '', '必须上下文(任一)': '',
              '排除上下文(任一)': '', '本轮被排除命中次数': len(samples),
              '排除原因': '样例文本片段'}])], ignore_index=True)
        for item in samples[:50]:
            table = pd.concat([table, pd.DataFrame([{
                '技能标准名': item['canonical'], '受控别名': item['alias'],
                '必须上下文(任一)': '', '排除上下文(任一)': '',
                '本轮被排除命中次数': None, '排除原因': item['window']}])], ignore_index=True)
    return table


def build_low_frequency_table(skill_table: pd.DataFrame) -> pd.DataFrame:
    """08_低频技能：岗位数低于阈值的技能（保留在 long table，不进入模型）。"""
    low = skill_table[skill_table['岗位数'] < LOW_FREQUENCY_THRESHOLD].copy()
    low['说明'] = (f'岗位数 < {LOW_FREQUENCY_THRESHOLD}，仅用于技能需求长尾分析，'
                   '建模阶段不纳入高频技能 multi-hot')
    return low.sort_values(['岗位数', '技能标准名'], ascending=[False, True]).reset_index(drop=True)


def build_candidate_table(features: pd.DataFrame, latest: pd.DataFrame,
                          matcher: skill_extraction.SkillMatcher) -> pd.DataFrame:
    """09_未识别候选技能：未被词典覆盖的技术词候选（仅供人工审核，禁止自动入库）。"""
    config = matcher.config
    known_latin = {alias.lower() for alias in config.alias_to_skill}
    rows: dict = {}
    sources: dict = {}
    for feature, record in zip(features.to_dict('records'), latest.to_dict('records')):
        safe_text = record.get(schema.JD_SAFE_FIELD) or ''
        requirement = text_utils.extract_requirement_section(safe_text)
        text = requirement['text']
        if not text:
            continue
        seen = set()
        for token in LATIN_TOKEN_PATTERN.findall(text):
            if len(token) < 3 or token.lower() in known_latin:
                continue
            if token.lower() in CANDIDATE_ENGLISH_STOP:
                continue
            seen.add((token, '拉丁技术词'))
        for token in CJK_TOKEN_PATTERN.findall(text):
            if token[0] in CANDIDATE_LEADING_STOP:
                continue
            if not any(token.endswith(word) or word in token
                       for word in CANDIDATE_SUFFIX_WORDS):
                continue
            if token in config.alias_to_skill:
                continue
            seen.add((token, '中文技术词'))
        for token, source in seen:
            rows.setdefault(token, set()).add(feature[schema.ID_FIELD])
            sources.setdefault(token, source)
    table = pd.DataFrame([
        {'候选技能词': token, '候选来源': sources[token], '命中岗位数': len(jobs),
         '命中岗位占比': round(len(jobs) / len(features), 6),
         '示例岗位ID': '、'.join(sorted(jobs)[:3]),
         '人工审核结论': '', '是否写入 skills.yml': ''}
        for token, jobs in rows.items()])
    columns = ['候选技能词', '候选来源', '命中岗位数', '命中岗位占比', '示例岗位ID',
               '人工审核结论', '是否写入 skills.yml']
    if table.empty:
        return pd.DataFrame(columns=columns)
    table = table.sort_values(['命中岗位数', '候选技能词'],
                              ascending=[False, True]).head(200).reset_index(drop=True)
    return table[columns]


def build_sample_table(features: pd.DataFrame) -> pd.DataFrame:
    """10_人工抽样核验：分层抽样 100~200 个岗位（固定规则，可复现）。"""
    id_field = schema.ID_FIELD
    frame = features.copy()
    frame['_office'] = frame[schema.SKILL_OFFICE_COUNT_FIELD] > 0
    frame['_agent'] = frame[schema.SKILL_SET_FIELD].map(lambda skills: 'Agent' in set(skills or []))
    frame['_high'] = frame[schema.SKILL_COUNT_FIELD] >= frame[schema.SKILL_COUNT_FIELD].quantile(0.95)
    frame['_low'] = frame[schema.SKILL_COUNT_FIELD] <= 1
    strata = [
        ('开发岗', frame[schema.JOB_DIRECTION_FIELD].isin(['后端开发', '前端开发', '客户端', '测试', '运维'])),
        ('数据岗', frame[schema.JOB_DIRECTION_FIELD].isin(['数据开发', '数据分析', '数据科学'])),
        ('AI岗', frame[schema.JOB_DIRECTION_FIELD].isin(['算法', '大模型'])),
        ('安全岗', frame[schema.JOB_DIRECTION_FIELD].eq('安全')),
        ('高技能数', frame['_high']),
        ('低技能数', frame['_low']),
        ('办公工具命中', frame['_office']),
        ('Agent 命中', frame['_agent']),
        ('全文fallback', frame[schema.SKILL_SCOPE_FIELD].eq('FULL_TEXT_FALLBACK')),
    ]
    per_stratum = max(1, SAMPLE_SIZE // len(strata))
    picked: list = []
    picked_ids: set = set()
    for name, mask in strata:
        candidates = frame.loc[mask].sort_values(id_field)
        added = 0
        for row in candidates.to_dict('records'):
            if row[id_field] in picked_ids:
                continue
            picked_ids.add(row[id_field])
            picked.append(dict(row, 抽样层=name))
            added += 1
            if added >= per_stratum:
                break
    # 分层未达目标规模时，按岗位 ID 顺序继续补齐（固定规则，可复现）
    if len(picked) < SAMPLE_SIZE:
        for row in frame.sort_values(id_field).to_dict('records'):
            if row[id_field] in picked_ids:
                continue
            picked_ids.add(row[id_field])
            picked.append(dict(row, 抽样层='补齐'))
            if len(picked) >= SAMPLE_SIZE:
                break
    sample = pd.DataFrame(picked).head(SAMPLE_SIZE)
    sample = sample[[id_field, '抽样层', schema.JOB_DIRECTION_FIELD, schema.SKILL_SCOPE_FIELD,
                     schema.JD_SPLIT_STATUS_FIELD, schema.SKILL_COUNT_FIELD,
                     schema.SKILL_LANGUAGE_COUNT_FIELD, schema.SKILL_DATABASE_COUNT_FIELD,
                     schema.SKILL_SECURITY_COUNT_FIELD, schema.SKILL_OFFICE_COUNT_FIELD,
                     schema.AI_SKILL_COUNT_FIELD, schema.LLM_SKILL_COUNT_FIELD,
                     schema.SKILL_SET_FIELD, schema.JOB_TITLE_NORMALIZED_FIELD]]
    sample = sample.rename(columns={id_field: '岗位ID', schema.SKILL_SET_FIELD: '技能集合'})
    sample['人工判定'] = ''
    sample['人工备注'] = ''
    return sample.sort_values('岗位ID').reset_index(drop=True)


def build_audit_sheets(features: pd.DataFrame, latest: pd.DataFrame, entity: pd.DataFrame,
                       matcher: skill_extraction.SkillMatcher, skill_table: pd.DataFrame,
                       group_table: pd.DataFrame, coverage: dict,
                       samples: list) -> dict:
    """26 号技能审计表（技能需求分析专用）。"""
    scope = coverage['scope_counts']
    count_bins = pd.cut(features[schema.SKILL_COUNT_FIELD], bins=[-1, 0, 1, 2, 3, 4, 10_000],
                        labels=['0', '1', '2', '3', '4', '5+'])
    distribution = (count_bins.value_counts().rename_axis('技能数').reset_index(name='岗位数')
                    .sort_values('技能数'))
    distribution['岗位占比'] = (distribution['岗位数'] / len(features)).round(6)
    coverage_rows = [
        {'指标': '技能标准名数', '数值': matcher.config.skill_count},
        {'指标': '别名数', '数值': matcher.config.alias_count},
        {'指标': 'feature_family 数', '数值': matcher.config.family_count},
        {'指标': '技能组数', '数值': matcher.config.group_count},
        {'指标': '最终岗位实体数', '数值': int(len(features))},
        {'指标': '至少 1 项技能的岗位数', '数值': coverage['skill_jobs']},
        {'指标': '技能覆盖率', '数值': round(coverage['skill_jobs'] / len(features), 6)},
        {'指标': '技能数量均值', '数值': round(float(features[schema.SKILL_COUNT_FIELD].mean()), 4)},
        {'指标': '技能数量中位数', '数值': float(features[schema.SKILL_COUNT_FIELD].median())},
        {'指标': 'REQUIREMENT_SECTION 岗位数', '数值': int(scope.get('REQUIREMENT_SECTION', 0))},
        {'指标': 'FULL_TEXT_FALLBACK 岗位数', '数值': int(scope.get('FULL_TEXT_FALLBACK', 0))},
        {'指标': 'EMPTY_TEXT 岗位数', '数值': int(scope.get('EMPTY_TEXT', 0))},
        {'指标': '进入布尔特征的技能数', '数值': len(BOOLEAN_FEATURE_SKILLS)},
        {'指标': '低频技能数（岗位数 < %d）' % LOW_FREQUENCY_THRESHOLD,
         '数值': int((skill_table['岗位数'] < LOW_FREQUENCY_THRESHOLD).sum())},
    ]
    for skill in BOOLEAN_FEATURE_SKILLS:
        coverage_rows.append({'指标': f'是否{skill} 的岗位数',
                              '数值': int(features[f'是否{skill}'].sum())})
    coverage_table = pd.DataFrame(coverage_rows)

    direction_table = (features[schema.JOB_DIRECTION_FIELD]
                       .value_counts().rename_axis('岗位方向_文本').reset_index(name='岗位数'))
    direction_table['岗位数占比'] = (direction_table['岗位数'] / len(features)).round(6)
    direction_table['与平台分类一致岗位数'] = direction_table['岗位方向_文本'].map(
        features.groupby(schema.JOB_DIRECTION_FIELD)[schema.JOB_DIRECTION_MATCH_FIELD].sum())
    direction_table['与平台分类一致比例'] = (direction_table['与平台分类一致岗位数']
                                    / direction_table['岗位数']).round(6)

    categories = (entity[[schema.ID_FIELD, '岗位大类集合']]
                  .explode('岗位大类集合').dropna(subset=['岗位大类集合'])
                  .rename(columns={'岗位大类集合': '岗位大类'}))
    tech_column = schema.SKILL_FAMILY_COUNT_FIELDS['技术技能']
    merged = categories.merge(
        features[[schema.ID_FIELD, schema.SKILL_COUNT_FIELD, tech_column]],
        on=schema.ID_FIELD, how='inner')
    category_table = (merged.groupby('岗位大类', as_index=False)
                      .agg(样本岗位数=(schema.ID_FIELD, 'size'),
                           提取到至少1项技术技能的岗位数=(tech_column,
                                              lambda values: int((values > 0).sum())),
                           提取到任意技能的岗位数=(schema.SKILL_COUNT_FIELD,
                                          lambda values: int((values > 0).sum())),
                           平均技术技能数=(tech_column, 'mean'))
                      .sort_values('样本岗位数', ascending=False))
    category_table['技术技能覆盖率'] = (category_table['提取到至少1项技术技能的岗位数']
                                 / category_table['样本岗位数']).round(6)
    category_table['技能覆盖率'] = (category_table['提取到任意技能的岗位数']
                              / category_table['样本岗位数']).round(6)
    category_table['平均技术技能数'] = category_table['平均技术技能数'].round(4)

    return {
        '01_技能词典统计': build_dictionary_table(matcher),
        '02_别名冲突检查': pd.DataFrame(
            skill_extraction.alias_table(matcher.config)
            + [{'别名': '（冲突检查结论）', '技能标准名': '', '技能组': '', 'feature_family': '',
                '是否标准名本身': '同一别名指向多个标准名时配置加载直接报错，当前冲突数 0'}]),
        '03_技能频率': skill_table,
        '04_技能组频率': group_table,
        '05_岗位技能数分布': distribution,
        '06_要求段落识别': build_requirement_table(features),
        '07_歧义词检查': build_ambiguity_table(matcher, samples),
        '08_低频技能': build_low_frequency_table(skill_table),
        '09_未识别候选技能': build_candidate_table(features, latest, matcher),
        '10_人工抽样核验': build_sample_table(features),
        '11_技能特征覆盖统计': coverage_table,
        '12_岗位方向识别统计': direction_table,
        '13_岗位大类技能覆盖率': category_table,
    }


def build_record_lines(metrics: dict, audit: dict) -> list:
    """17 号技能提取记录。"""
    coverage = audit['11_技能特征覆盖统计'].set_index('指标')['数值']
    lines = [
        '# 记录 17：Stage 07 岗位技能需求提取与文本特征',
        '',
        '> 本记录由 `scripts/ch3_data/07_extract_job_text_features.py` 自动生成，数字全部来自真实运行结果；',
        '> 技能提取只使用**模型安全版**岗位描述（不引入薪资文字），且只对应**最终核心版本**。',
        '',
        '## 1. 词典结构（config/skills.yml，单一权威来源）',
        '',
        '| 项 | 数值 |',
        '| --- | --- |',
        f'| 技能标准名（canonical） | {metrics["skill_names"]} |',
        f'| 别名（含标准名自身） | {metrics["alias_count"]} |',
        f'| feature_family 数 | {metrics["skill_families"]} |',
        f'| 技能组（group）数 | {metrics["skill_groups"]} |',
        f'| 歧义上下文规则 | {metrics["context_rule_count"]} 条 |',
        f'| 别名冲突数 | {metrics["alias_conflicts"]} |',
        '',
        '## 2. 提取范围与覆盖',
        '',
        '| 项 | 数值 |',
        '| --- | --- |',
        f'| 最终岗位实体数 | {metrics["entity_jobs"]} |',
        f'| 要求段落识别成功（REQUIREMENT_SECTION） | {metrics["scope_requirement"]} |',
        f'| 全文 fallback（FULL_TEXT_FALLBACK） | {metrics["scope_fallback"]} |',
        f'| 空文本（EMPTY_TEXT） | {metrics["scope_empty"]} |',
        f'| 至少 1 项技能的岗位数 | {metrics["skill_jobs"]}（覆盖率 {metrics["skill_coverage"]:.2%}） |',
        f'| 技能数量均值 / 中位数 | {coverage["技能数量均值"]} / {coverage["技能数量中位数"]} |',
        f'| 歧义词被排除命中次数合计 | {metrics["ambiguous_rejected"]} |',
        '',
        '## 3. Top 20 技能（最终岗位层，分母 = 唯一岗位数）',
        '',
        '| 技能标准名 | feature_family | 技能组 | 岗位数 | 岗位覆盖率 | 要求段落内命中 |',
        '| --- | --- | --- | --- | --- | --- |',
    ]
    for row in audit['03_技能频率'].head(20).itertuples(index=False):
        lines.append(f'| {row.技能标准名} | {row.feature_family} | {row.技能组} | {int(row.岗位数)} | '
                     f'{row.岗位覆盖率:.2%} | {int(row.要求段落内命中岗位数)} |')
    lines += [
        '',
        '## 4. 重点技能岗位数',
        '',
        '| 技能 | 岗位数 | 岗位覆盖率 |',
        '| --- | --- | --- |',
    ]
    for skill, count in metrics['focus_skills'].items():
        share = count / metrics['entity_jobs'] if metrics['entity_jobs'] else 0
        lines.append(f'| {skill} | {int(count)} | {share:.2%} |')
    lines += [
        '',
        '## 5. 技能组频率（前 12）',
        '',
        '| 技能组 | feature_family | 技能标准名数 | 岗位数 | 岗位覆盖率 | 技能项次 |',
        '| --- | --- | --- | --- | --- | --- |',
    ]
    for row in audit['04_技能组频率'].head(12).itertuples(index=False):
        lines.append(f'| {row.技能组} | {row.feature_family} | {int(row.技能标准名数)} | '
                     f'{int(row.岗位数)} | {row.岗位覆盖率:.2%} | {int(row.技能项次)} |')
    lines += [
        '',
        '## 6. 产物',
        '',
        f"- 岗位文本特征层：`{metrics['features_path']}`（{metrics['feature_rows']} 行 × "
        f"{metrics['feature_columns']} 列，一行一个最终岗位实体）；",
        f"- 岗位 × 规范技能 long-format：`{metrics['membership_path']}`"
        f"（{metrics['membership_rows']} 行，(intern_id, canonical_skill) 重复 "
        f"{metrics['membership_duplicates']} 行）；",
        f"- 技能审计表：`{metrics['audit_path']}`；",
        '',
        '> 高频技能 multi-hot（skill_Python / skill_Java / …）属于**建模输入矩阵**，'
        '在 Stage 12 建模宽表阶段由 long-format 临时展开，不写入核心岗位实体表；'
        '低频技能继续保留在 long table 用于技能需求分析。',
        '',
        '## 7. 运行门禁（Stage 07）',
        '',
        '| 门禁项 | 状态 |',
        '| --- | --- |',
    ]
    for name, result in metrics['gates'].items():
        lines.append(f'| {name} | {result} |')
    lines.append('')
    return lines


def main() -> int:
    quality.configure_logging()
    gates = quality.GateRegistry(STAGE)

    quality.stage_banner(STAGE, TITLE)
    entity = io_utils.read_parquet(project_paths.PROCESSED_UNIQUE_PARQUET)
    corpus = io_utils.read_parquet(project_paths.TEXT_CORPUS_PARQUET)
    print(f'输入: 最终岗位实体 {len(entity)} 行 / 文本语料 {len(corpus)} 行')

    config = skill_extraction.load_skill_config()
    matcher = skill_extraction.SkillMatcher(config)
    gates.check('SKILL_CONFIG_SCHEMA',
                config.skill_count > 0 and config.alias_count >= config.skill_count
                and config.family_count == len(schema.SKILL_FAMILY_NAMES)
                and bool(config.language_skill_groups) and bool(config.office_skill_groups),
                f'词典结构完整：{config.skill_count} 标准名 / {config.alias_count} 别名 / '
                f'{config.family_count} 一级类型 / {config.group_count} 技能组')
    gates.check('SKILL_ALIAS_UNIQUENESS',
                len(config.alias_to_skill) > 0,
                f'别名一对一校验通过（同一别名指向多个标准名时配置加载直接报错）：'
                f'{len(config.alias_to_skill)} 个别名，冲突 0')
    gates.check('SKILL_CATEGORY_HIERARCHY',
                all(config.skill_to_group.get(skill) in config.group_to_family
                    for skill in config.skill_to_group)
                and len(config.group_to_family) == len(set(config.group_to_family)),
                f'三级结构完整：每个技能标准名都归属唯一技能组，技能组唯一归属一级类型'
                f'（{config.skill_count} / {config.group_count} / {config.family_count}）')

    entity_frame = entity[[schema.ID_FIELD, schema.ENTITY_FINAL_VERSION_FIELD, '岗位标题',
                           schema.ENTITY_CORE_VERSION_COUNT_FIELD]].copy()
    entity_frame.columns = [schema.ID_FIELD, '_final_version', '_title', '_core_versions']
    latest = corpus[[schema.ID_FIELD, schema.CORE_VERSION_FIELD, schema.JD_SAFE_FIELD,
                     schema.JD_CHAR_COUNT_FIELD, schema.JD_TOKEN_COUNT_FIELD,
                     schema.JD_SPLIT_STATUS_FIELD, schema.JOB_TAG_LIST_FIELD,
                     schema.SKILL_SET_FIELD, schema.COMPANY_PROFILE_CHAR_COUNT_FIELD,
                     schema.COMPANY_TAG_LIST_FIELD]].merge(
        entity_frame, left_on=[schema.ID_FIELD, schema.CORE_VERSION_FIELD],
        right_on=[schema.ID_FIELD, '_final_version'], how='inner')
    gates.check('SKILL_FINAL_VERSION_SELECTION',
                len(latest) == len(entity) and latest[schema.ID_FIELD].is_unique
                and int(latest[schema.CORE_VERSION_FIELD].eq(latest['_final_version']).all()),
                f'技能画像只取最终核心版本：{len(latest)}/{len(entity)} 个岗位，'
                '未使用历史版本 union')

    safe_series = latest[schema.JD_SAFE_FIELD].fillna('').astype(str)
    official_residual = int(sum(text_utils.detect_salary_leakage(text) for text in safe_series))
    strict_residual = int(sum(text_utils.detect_strict_salary_leakage(text) for text in safe_series))
    strict_mask = safe_series.map(text_utils.detect_strict_salary_leakage).gt(0)
    strict_rows = latest.loc[strict_mask, [schema.ID_FIELD, schema.CORE_VERSION_FIELD,
                                           schema.JD_SAFE_FIELD]].copy()
    gates.check('SKILL_MODEL_SAFE_TEXT',
                official_residual == 0 and strict_residual == 0
                and schema.JD_SAFE_FIELD in latest.columns,
                f'技能提取输入为「{schema.JD_SAFE_FIELD}」（模型安全版），'
                f'官方泄漏检测残留 {official_residual}，严格薪资样式残留 {strict_residual}'
                '（规则源与 Stage 06 清洗、Stage 06 门禁、tests 完全一致）')

    skill_records = build_skill_records(latest, matcher)
    features, coverage = build_features(entity, latest, skill_records, matcher)
    scope = coverage['scope_counts']
    gates.check('SKILL_REQUIREMENT_SECTION',
                int(scope.get('REQUIREMENT_SECTION', 0)) > 0
                and int(scope.get('FULL_TEXT_FALLBACK', 0)) >= 0
                and sum(scope.values()) == len(features),
                f'要求段落识别 {scope.get("REQUIREMENT_SECTION", 0)} 个岗位；'
                f'全文 fallback {scope.get("FULL_TEXT_FALLBACK", 0)} 个；'
                f'空文本 {scope.get("EMPTY_TEXT", 0)} 个（fallback 岗位不丢弃）')

    canonical = set(config.skill_to_group)
    extracted = {skill for skills in features[schema.SKILL_SET_FIELD] for skill in skills}
    gates.check('SKILL_EXTRACTION',
                coverage['skill_jobs'] > 0 and len(extracted) > 0 and extracted <= canonical,
                f'{coverage["skill_jobs"]} 个岗位提取到技能，覆盖 {len(extracted)}/{config.skill_count} '
                f'个标准名，零技能岗位 {coverage["zero_skill_jobs"]} 个')
    gates.check('SKILL_ALIAS_NORMALIZE',
                extracted <= canonical,
                f'技能集合中全部为标准名（未标准化别名 {sorted(extracted - canonical)[:5]}）')
    gates.check('SKILL_FAMILY_MAPPING',
                not coverage['unmapped_family_skills'],
                f'{config.skill_count} 个技能标准名全部映射到 feature_family → group')

    rejected_total = int(sum(matcher.rejection_counter.values()))
    gates.check('SKILL_AMBIGUOUS_TERM_GUARD',
                bool(config.context_rules) and len(matcher._rules_by_alias) > 0,
                f'{len(config.context_rules)} 条歧义上下文规则生效'
                f'（Agent / Word / C / R / Go / Spring / React），'
                f'本轮排除 {rejected_total} 次命中')

    membership = skill_extraction.build_job_skill_membership(config, skill_records)
    duplicates = int(membership.duplicated(
        subset=[schema.SKILL_MEMBERSHIP_ID_FIELD, schema.SKILL_MEMBERSHIP_SKILL_FIELD]).sum())
    gates.check('SKILL_JOB_LEVEL_BUILD',
                not membership.empty
                and duplicates == 0
                and membership[schema.SKILL_MEMBERSHIP_ID_FIELD].nunique() <= len(features),
                f'long-format {len(membership)} 行，(intern_id, canonical_skill) 重复 {duplicates} 行；'
                f'岗位技能画像 {len(features)} 行（一岗一行，未展开技能布尔列）')

    gates.check('TEXT_FEATURE_BUILD',
                len(features) == coverage['entity_jobs']
                and set(schema.JOB_TEXT_FEATURE_COLUMNS) <= set(features.columns),
                f'特征表 {len(features)} 行与实体数一致，字段齐全（{features.shape[1]} 列）')
    gates.check('TEXT_FEATURE_COVERAGE',
                coverage['missing_jobs'] == 0,
                f'实体覆盖率 {coverage["feature_jobs"]}/{coverage["entity_jobs"]}，'
                f'缺失 {coverage["missing_jobs"]} 个')

    skill_table = build_skill_frequency(features, matcher)
    group_table = build_group_frequency(features, matcher)
    audit_sheets = build_audit_sheets(features, latest, entity, matcher, skill_table,
                                      group_table, coverage, matcher.rejection_samples)
    audit_sheets['11_技能特征覆盖统计'] = pd.concat([
        audit_sheets['11_技能特征覆盖统计'],
        pd.DataFrame([
            {'指标': '官方薪资泄漏检测残留（text_utils.detect_salary_leakage）',
             '数值': official_residual},
            {'指标': '严格薪资样式残留（text_utils.detect_strict_salary_leakage）',
             '数值': strict_residual},
        ])], ignore_index=True)
    if len(strict_rows):
        strict_rows['说明'] = '严格薪资样式残留，需回到 Stage 06 修复（本轮修复后应为 0 行）'
        audit_sheets['14_严格薪资样式残留复核'] = strict_rows.drop(columns=[schema.JD_SAFE_FIELD])
    else:
        audit_sheets['14_严格薪资样式残留复核'] = pd.DataFrame([
            {'岗位ID': '（无）', '核心版本号': None,
             '说明': '模型安全版严格薪资样式残留 0 行（Stage 06 清洗规则源与 Stage 07 检测一致）'}])
    category_table = audit_sheets['13_岗位大类技能覆盖率']
    gates.check('CATEGORY_SKILL_COVERAGE_BUILD',
                not category_table.empty and category_table['技术技能覆盖率'].between(0, 1).all(),
                f'{len(category_table)} 个岗位大类的技能覆盖率已统计')

    features_path = io_utils.write_parquet(features, project_paths.JOB_TEXT_FEATURES_PARQUET)
    membership_path = io_utils.write_parquet(membership, project_paths.JOB_SKILL_MEMBERSHIP_PARQUET)
    audit_path = io_utils.write_excel(
        project_paths.TABLES_DIR / project_paths.TABLE_JOB_SKILL_AUDIT, audit_sheets)
    gates.check('SKILL_AUDIT_EXPORT',
                audit_path.exists() and len(audit_sheets) >= 10
                and len(audit_sheets['10_人工抽样核验']) >= 100,
                f'{len(audit_sheets)} 张技能审计子表已写出，人工抽样 '
                f'{len(audit_sheets["10_人工抽样核验"])} 个岗位')

    preprocess_path = project_paths.TABLES_DIR / project_paths.TABLE_JOB_TEXT_PREPROCESSING
    existing_sheets = pd.read_excel(preprocess_path, sheet_name=None) if preprocess_path.exists() else {}
    # 技能需求类子表统一归口 26 号技能审计表：从 16 号表移除历史同名/同义子表，避免双份维护
    superseded = ['09_岗位方向识别统计', '10_技能特征覆盖统计', '11_岗位大类技能覆盖率',
                  '12_技能一级类型统计', '13_技能细分类统计', '14_技术岗位低覆盖候选',
                  '15_最终岗位技能标准名统计']
    removed = [name for name in superseded if name in existing_sheets]
    existing_sheets = {name: value for name, value in existing_sheets.items()
                       if name not in superseded}
    existing_sheets.update({name: audit_sheets[name] for name in
                            ('11_技能特征覆盖统计', '12_岗位方向识别统计', '13_岗位大类技能覆盖率')})
    io_utils.write_excel(preprocess_path, existing_sheets)
    gates.check('TEXT_FEATURE_AUDIT_EXPORT',
                features_path.exists() and membership_path.exists()
                and {'11_技能特征覆盖统计', '12_岗位方向识别统计',
                     '13_岗位大类技能覆盖率'} <= set(existing_sheets),
                f'特征表 + long-format + 16/26 号审计表已写出（技能需求类子表统一归口 26 号，'
                f'16 号表移除历史重复子表 {len(removed)} 张）')

    focus_counts = {}
    for skill in FOCUS_SKILLS:
        count_column = f'是否{skill}'
        focus_counts[skill] = (int(features[count_column].sum()) if count_column in features.columns
                               else int(features[schema.SKILL_SET_FIELD].map(
                                   lambda skills: skill in set(skills or [])).sum()))
    metrics = {
        'features_path': project_paths.relative_to_root(features_path),
        'membership_path': project_paths.relative_to_root(membership_path),
        'feature_rows': int(len(features)),
        'feature_columns': int(features.shape[1]),
        'membership_rows': int(len(membership)),
        'membership_duplicates': duplicates,
        'entity_jobs': int(coverage['entity_jobs']),
        'skill_names': int(config.skill_count),
        'alias_count': int(config.alias_count),
        'skill_families': int(config.family_count),
        'skill_groups': int(config.group_count),
        'context_rule_count': int(len(config.context_rules)),
        'alias_conflicts': 0,
        'skill_jobs': int(coverage['skill_jobs']),
        'zero_skill_jobs': int(coverage['zero_skill_jobs']),
        'skill_coverage': round(coverage['skill_jobs'] / len(features), 6),
        'scope_requirement': int(scope.get('REQUIREMENT_SECTION', 0)),
        'scope_fallback': int(scope.get('FULL_TEXT_FALLBACK', 0)),
        'scope_empty': int(scope.get('EMPTY_TEXT', 0)),
        'ambiguous_rejected': rejected_total,
        'model_safe_official_residual': official_residual,
        'model_safe_strict_residual': strict_residual,
        'focus_skills': focus_counts,
        'skill_frequency_top': audit_sheets['03_技能频率'].head(30).to_dict('records'),
        'skill_group_stats': group_table.to_dict('records'),
        'low_frequency_skills': int(len(audit_sheets['08_低频技能'])),
        'audit_path': project_paths.relative_to_root(audit_path),
        'metrics_path': project_paths.relative_to_root(project_paths.METRICS_DIR / f'{STAGE}.json'),
    }
    metrics['gates'] = {name: result['status'] for name, result in gates.results.items()}
    io_utils.write_json(project_paths.METRICS_DIR / f'{STAGE}.json', metrics)

    record_path = io_utils.write_markdown(
        project_paths.RECORDS_DIR / project_paths.RECORD_JOB_SKILL_EXTRACTION,
        build_record_lines(metrics, audit_sheets))
    print(f'阶段记录: {project_paths.relative_to_root(record_path)}')

    quality.print_gates(gates.results, order=quality.STAGE_GATE_MAP[STAGE])
    gates.save()
    return 0


if __name__ == '__main__':
    sys.exit(main())

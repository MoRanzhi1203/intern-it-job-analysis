# -*- coding: utf-8 -*-
"""IT 技能实体标准化模块：技能词典加载、别名标准化与上下文安全匹配。

设计要点（对应提示词第 8~10 节）：

1. 技能词典与别名**全部**来自 ``config/skills.yml``，本模块不硬编码任何技能；
2. 别名统一映射到标准名（cpp → C++、golang → Go、ML → 机器学习、大模型 → 大模型 等）；
3. 对 C / R / Go / AI / ML / DL / CV 等高歧义短词，强制「大小写严格 + 词边界 + 黑名单」
   三重约束，禁止使用 ``if skill.lower() in text.lower()`` 这类包含匹配；
4. 匹配按别名长度降序优先，保证 ``C++`` / ``C#`` 不会被 ``C`` 抢先命中。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import project_paths, schema

SKILL_CONFIG_PATH = project_paths.PROJECT_ROOT / 'config' / 'skills.yml'

LATIN_PATTERN = re.compile(r'[A-Za-z]')
# 拉丁词边界字符：字母/数字/下划线/+# 紧邻时不算边界
BOUNDARY_CHARS = set('abcdefghijklmnopqrstuvwxyz'
                     'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                     '0123456789_+#')


@dataclass
class SkillConfig:
    """技能配置的内存视图（三级结构 feature_family → group → canonical）。"""

    skill_to_group: dict
    skill_to_family: dict
    family_to_groups: dict
    group_to_family: dict
    alias_to_skill: dict
    alias_group: dict
    strict_case_aliases: set
    blocklist: list
    llm_skills: list
    ai_skill_groups: list
    ai_keywords: list
    company_keywords: list
    short_term_max_length: int
    raw_groups: dict
    language_skill_groups: list = field(default_factory=list)
    database_skill_groups: list = field(default_factory=list)
    security_skill_groups: list = field(default_factory=list)
    office_skill_groups: list = field(default_factory=list)
    context_rules: dict = field(default_factory=dict)
    context_window: int = 30
    rank_layers: dict = field(default_factory=dict)

    @property
    def skill_count(self) -> int:
        """技能标准名数量。"""
        return len(self.skill_to_group)

    @property
    def alias_count(self) -> int:
        """参与匹配的别名数量（含标准名自身）。"""
        return len(self.alias_to_skill)

    @property
    def group_count(self) -> int:
        """二级类别（group）数量。"""
        return len(self.group_to_family)

    @property
    def family_count(self) -> int:
        """一级类型（feature_family）数量。"""
        return len(self.family_to_groups)

    def family_of(self, skill: str) -> str:
        """技能标准名 → 一级类型；未定义返回空串（由门禁拦截）。"""
        return self.skill_to_family.get(skill, '')

    def family_skills(self, family: str) -> set:
        """某一级类型下的全部技能标准名。"""
        return {skill for skill, name in self.skill_to_family.items() if name == family}

    def group_skills(self, groups) -> set:
        """若干技能组下的全部技能标准名。"""
        wanted = set(groups or [])
        return {skill for skill, group in self.skill_to_group.items() if group in wanted}

    def ai_skills(self) -> set:
        """AI 技能统计范围：配置中显式声明的 AI 相关组内的全部标准实体。"""
        return self.group_skills(self.ai_skill_groups)

    def language_skills(self) -> set:
        """编程语言技能统计范围（编程语言技能数）。"""
        return self.group_skills(self.language_skill_groups)

    def database_skills(self) -> set:
        """数据库技能统计范围（数据库技能数）。"""
        return self.group_skills(self.database_skill_groups)

    def security_skills(self) -> set:
        """安全技能统计范围（安全技能数）。"""
        return self.group_skills(self.security_skill_groups)

    def office_skills(self) -> set:
        """办公工具技能统计范围（办公工具技能数）。"""
        return self.group_skills(self.office_skill_groups)

    def rule_canonicals(self) -> list:
        """配置了歧义上下文规则的标准名（按配置顺序）。"""
        return list(self.context_rules)



def load_skill_config(path: Path | None = None) -> SkillConfig:
    """加载三级技能配置；缺失、格式错误或归属不全直接抛错（不允许静默降级）。"""
    target = Path(path or SKILL_CONFIG_PATH)
    if not target.exists():
        raise FileNotFoundError(f'技能配置不存在: {target}')
    payload = yaml.safe_load(target.read_text(encoding='utf-8')) or {}
    groups = payload.get('groups') or {}
    families = payload.get('feature_families') or {}

    # ---- 1) group → feature_family（显式配置，禁止运行时猜类别） ----
    group_to_family: dict = {}
    family_to_groups: dict = {}
    for family_name, group_names in families.items():
        family_name = str(family_name).strip()
        if family_name not in schema.SKILL_FAMILY_NAMES:
            raise ValueError(f'feature_families 中出现未登记的一级类型: {family_name}')
        members = [str(item).strip() for item in group_names or []]
        if not members:
            raise ValueError(f'一级类型 {family_name} 未配置任何二级类别')
        family_to_groups[family_name] = members
        for group_name in members:
            if group_name in group_to_family:
                raise ValueError(f'二级类别 {group_name} 被重复归属到多个一级类型')
            group_to_family[group_name] = family_name
    missing_family = sorted(set(groups) - set(group_to_family))
    if missing_family:
        raise ValueError(f'以下技能组未配置 feature_family: {missing_family}')
    unknown_group = sorted(set(group_to_family) - set(groups))
    if unknown_group:
        raise ValueError(f'feature_families 中声明了不存在的技能组: {unknown_group}')

    # ---- 2) group + canonical → 标准实体（别名必须一对一，冲突直接报错） ----
    skill_to_group: dict = {}
    skill_to_family: dict = {}
    alias_to_skill: dict = {}
    alias_group: dict = {}
    alias_conflicts: list = []
    for group_name, entries in groups.items():
        for entry in entries or []:
            canonical = str(entry['canonical']).strip()
            if canonical in skill_to_group:
                raise ValueError(f'技能标准名重复定义: {canonical}')
            skill_to_group[canonical] = group_name
            skill_to_family[canonical] = group_to_family[group_name]
            aliases = [canonical, *(entry.get('aliases') or [])]
            for alias in aliases:
                alias_text = str(alias).strip()
                if not alias_text:
                    continue
                owner = alias_to_skill.get(alias_text)
                if owner is not None and owner != canonical:
                    alias_conflicts.append((alias_text, owner, canonical))
                    continue
                alias_to_skill[alias_text] = canonical
                alias_group[alias_text] = group_name
    if alias_conflicts:
        detail = '；'.join(f'{alias} → {owner} / {other}'
                         for alias, owner, other in alias_conflicts[:10])
        raise ValueError(f'别名冲突（同一别名指向多个标准名，且无显式上下文规则）: {detail}')

    ai_skill_groups = [str(item).strip() for item in payload.get('ai_skill_groups') or []]
    unknown_ai_group = sorted(set(ai_skill_groups) - set(groups))
    if unknown_ai_group:
        raise ValueError(f'ai_skill_groups 中出现未定义的技能组: {unknown_ai_group}')

    def _group_list(key: str) -> list:
        items = [str(item).strip() for item in payload.get(key) or []]
        unknown = sorted(set(items) - set(groups))
        if unknown:
            raise ValueError(f'{key} 中出现未定义的技能组: {unknown}')
        return items

    context_rules = payload.get('ambiguous_context_rules') or {}
    for canonical, rule in context_rules.items():
        if canonical not in skill_to_group:
            raise ValueError(f'ambiguous_context_rules 中出现未定义技能: {canonical}')
        rule_aliases = [str(item).strip() for item in (rule or {}).get('match_aliases') or []]
        if not rule_aliases:
            raise ValueError(f'{canonical} 的上下文规则未配置 match_aliases')
        wrong_owner = [alias for alias in rule_aliases
                       if alias_to_skill.get(alias) != canonical]
        if wrong_owner:
            raise ValueError(f'{canonical} 的上下文规则引用了不属于该技能的别名: {wrong_owner}')
        for key in ('require_any', 'exclude_any'):
            values = (rule or {}).get(key)
            if values is not None and not isinstance(values, list):
                raise ValueError(f'{canonical} 的 {key} 必须是列表')

    rank_layers = payload.get('rank_layers') or {}
    for layer, rule in rank_layers.items():
        for family in (rule or {}).get('families') or []:
            if family not in family_to_groups:
                raise ValueError(f'rank_layers.{layer} 引用了未定义的 feature_family: {family}')
        for group in (rule or {}).get('groups') or []:
            if group not in groups:
                raise ValueError(f'rank_layers.{layer} 引用了未定义的技能组: {group}')
        for skill in (rule or {}).get('skills') or []:
            if skill not in skill_to_group:
                raise ValueError(f'rank_layers.{layer} 引用了未定义的技能标准名: {skill}')

    config = SkillConfig(
        skill_to_group=skill_to_group,
        skill_to_family=skill_to_family,
        family_to_groups=family_to_groups,
        group_to_family=group_to_family,
        alias_to_skill=alias_to_skill,
        alias_group=alias_group,
        strict_case_aliases={str(item).strip() for item in payload.get('strict_case_aliases') or []},
        blocklist=[str(item) for item in payload.get('blocklist') or []],
        llm_skills=[str(item) for item in payload.get('llm_related_skills') or []],
        ai_skill_groups=ai_skill_groups,
        ai_keywords=[str(item) for item in payload.get('ai_keywords') or []],
        company_keywords=[str(item) for item in payload.get('company_keywords') or []],
        short_term_max_length=int(payload.get('short_term_max_length', 4)),
        raw_groups=groups,
        language_skill_groups=_group_list('language_skill_groups'),
        database_skill_groups=_group_list('database_skill_groups'),
        security_skill_groups=_group_list('security_skill_groups'),
        office_skill_groups=_group_list('office_skill_groups'),
        context_rules={str(key): value or {} for key, value in context_rules.items()},
        context_window=int(payload.get('ambiguous_context_window', 30)),
        rank_layers={str(key): (value or {}) for key, value in rank_layers.items()},
    )
    missing_skills = [name for name in config.llm_skills if name not in config.skill_to_group]
    if missing_skills:
        raise ValueError(f'llm_related_skills 中存在未定义技能: {missing_skills}')
    missing_aliases = sorted(name for name in config.strict_case_aliases
                             if name not in config.alias_to_skill)
    if missing_aliases:
        raise ValueError(f'strict_case_aliases 中存在未定义别名: {missing_aliases}')
    empty_family = sorted(skill for skill, family in config.skill_to_family.items() if not family)
    if empty_family:
        raise ValueError(f'以下技能标准名缺少 feature_family: {empty_family}')
    return config


def skills_by_family(config: SkillConfig, skills) -> dict:
    """把技能集合按一级类型分桶（返回 {feature_family: 排序后的标准名列表}）。"""
    from . import text_utils  # noqa: PLC0415 - 避免模块级循环依赖

    buckets = {family: [] for family in schema.SKILL_FAMILY_NAMES}
    for skill in text_utils.as_list(skills):
        family = config.skill_to_family.get(skill)
        if family:
            buckets[family].append(skill)
    return {family: sorted(items) for family, items in buckets.items()}


def _is_latin_alias(alias: str) -> bool:
    """别名是否包含拉丁字母（决定是否启用词边界规则）。"""
    return bool(LATIN_PATTERN.search(alias))


def _boundary_ok(text: str, start: int, end: int, alias: str) -> bool:
    """检查拉丁别名左右词边界是否成立（中文相邻不受限制）。"""
    if start > 0:
        previous = text[start - 1]
        if alias[:1] in ('.', '+', '#'):
            if previous.isalnum() or previous == '_':
                return False
        elif previous in BOUNDARY_CHARS:
            return False
    if end < len(text):
        following = text[end]
        if alias[-1:] in ('+', '#'):
            if following.isalnum() or following == '_':
                return False
        elif following in BOUNDARY_CHARS:
            return False
    return True


class SkillMatcher:
    """技能匹配器：按首字符分桶逐位置扫描，最长匹配优先（最大匹配原则）。

    相比「整词典拼成超长正则」的做法，分桶扫描避免了 Python 正则 100 个命名分组上限，
    同时把复杂度降到 O(文本长度 × 桶内候选数)，并保证 ``C++`` 不会被 ``C`` 抢先命中。
    """

    def __init__(self, config: SkillConfig, text_limit: int = 200_000):
        self.config = config
        self.text_limit = text_limit
        buckets: dict = {}
        single_char: dict = {}
        for alias, canonical in config.alias_to_skill.items():
            is_latin = _is_latin_alias(alias)
            if not is_latin and len(alias) < 2:
                continue  # 单字中文别名歧义过大，不参与匹配
            if is_latin and len(alias) <= config.short_term_max_length:
                strict = True
            else:
                strict = alias in config.strict_case_aliases
            entry = {
                'alias': alias,
                'lower': alias.lower(),
                'canonical': canonical,
                'latin': is_latin,
                'strict': strict,
                'length': len(alias),
            }
            if len(alias) == 1:
                single_char.setdefault(alias.lower(), []).append(entry)
                continue
            buckets.setdefault(alias[:2].lower(), []).append(entry)
        for entries in buckets.values():
            entries.sort(key=lambda item: item['length'], reverse=True)
        for entries in single_char.values():
            entries.sort(key=lambda item: item['length'], reverse=True)
        self._buckets = buckets
        self._single_char = single_char
        self._blocklist = [re.compile(re.escape(item)) for item in config.blocklist]

        # 歧义上下文规则：仅作用于配置中显式声明的别名
        self._rules_by_alias = {}
        for canonical, rule in config.context_rules.items():
            compiled = {
                'canonical': canonical,
                'require': [str(item) for item in (rule or {}).get('require_any') or []],
                'exclude': [str(item) for item in (rule or {}).get('exclude_any') or []],
                'window': config.context_window,
            }
            for alias in (rule or {}).get('match_aliases') or []:
                self._rules_by_alias[str(alias).lower()] = compiled
        self.rejection_counter: Counter = Counter()
        self.rejection_reasons: Counter = Counter()
        self.rejection_samples: list = []
        self.rejection_sample_limit = 200

    def _blocked_spans(self, text: str) -> list:
        spans = []
        for pattern in self._blocklist:
            spans.extend((match.start(), match.end()) for match in pattern.finditer(text))
        return spans

    def _context_ok(self, text: str, start: int, end: int, entry: dict) -> bool:
        """歧义上下文判定：窗口内必须满足 require_any 且不得命中 exclude_any。"""
        rule = self._rules_by_alias.get(entry['alias'].lower())
        if rule is None:
            return True
        window = text[max(0, start - rule['window']):end + rule['window']].lower()
        for item in rule['exclude']:
            if item.lower() in window:
                self._record_rejection(rule, entry, text, start, end, 'EXCLUDE_CONTEXT', item)
                return False
        require = rule['require']
        if require and not any(item.lower() in window for item in require):
            self._record_rejection(rule, entry, text, start, end, 'NO_REQUIRED_CONTEXT', '')
            return False
        return True

    def _record_rejection(self, rule: dict, entry: dict, text: str, start: int, end: int,
                          reason: str, pattern: str) -> None:
        """登记歧义排除（计数 + 少量样例，供审计表人工核对）。"""
        self.rejection_counter[rule['canonical']] += 1
        self.rejection_reasons[(rule['canonical'], reason)] += 1
        if len(self.rejection_samples) < self.rejection_sample_limit:
            window = text[max(0, start - rule['window']):end + rule['window']]
            self.rejection_samples.append({
                'canonical': rule['canonical'],
                'alias': entry['alias'],
                'reason': reason,
                'pattern': pattern,
                'window': window.replace('\n', ' ')[:80],
            })

    def _try_candidates(self, sample: str, index: int, candidates, blocked) -> dict | None:
        total = len(sample)
        for entry in candidates or ():
            length = entry['length']
            if index + length > total:
                continue
            fragment = sample[index:index + length]
            if entry['strict'] or not entry['latin']:
                if fragment != entry['alias']:
                    continue
            elif fragment.lower() != entry['lower']:
                continue
            end = index + length
            if entry['latin'] and not _boundary_ok(sample, index, end, entry['alias']):
                continue
            if any(index < block_end and end > block_start
                   for block_start, block_end in blocked):
                continue
            if not self._context_ok(sample, index, end, entry):
                continue
            return entry
        return None

    def _scan(self, text) -> list:
        """逐位置扫描，返回 [(标准名, 命中别名, 技能组)]，保留重复命中。"""
        if not isinstance(text, str) or not text.strip():
            return []
        sample = text[:self.text_limit]
        lowered = sample.lower()
        blocked = self._blocked_spans(sample)
        hits = []
        total = len(sample)
        index = 0
        while index < total:
            key = lowered[index:index + 2]
            entry = self._try_candidates(sample, index, self._buckets.get(key), blocked)
            if entry is None:
                entry = self._try_candidates(sample, index,
                                             self._single_char.get(lowered[index]), blocked)
            if entry is None:
                index += 1
                continue
            canonical = entry['canonical']
            hits.append((canonical, entry['alias'],
                         self.config.alias_group.get(
                             entry['alias'],
                             self.config.skill_to_group.get(canonical, '其他'))))
            index += entry['length']
        return hits

    def match_details(self, text) -> list:
        """返回 [(标准名, 命中别名, 技能组)]，同一标准名只保留一次。

        逐位置扫描：先尝试两字符前缀桶（长别名优先），再尝试单字符别名（C / R），
        命中后直接跳过整个匹配片段，因此 ``C`` 不会在 ``C++`` 内部被重复命中。
        """
        details = []
        seen = set()
        for canonical, alias, group in self._scan(text):
            if canonical in seen:
                continue
            seen.add(canonical)
            details.append((canonical, alias, group))
        return details

    def match_counted(self, text) -> list:
        """返回 [{canonical, alias, group, hit_count}]：同一标准名聚合命中次数。"""
        counts: dict = {}
        order: list = []
        for canonical, alias, group in self._scan(text):
            if canonical not in counts:
                counts[canonical] = {'canonical': canonical, 'alias': alias,
                                     'group': group, 'hit_count': 0}
                order.append(canonical)
            counts[canonical]['hit_count'] += 1
        return [counts[canonical] for canonical in order]

    def rejection_table(self) -> list:
        """歧义词排除统计（写入技能审计表）。"""
        rows = []
        for canonical in self.config.rule_canonicals():
            if not self.rejection_counter[canonical]:
                continue
            rows.append({
                '技能标准名': canonical,
                '被排除命中次数': int(self.rejection_counter[canonical]),
                '排除原因': '、'.join(sorted({
                    reason for (name, reason), value in self.rejection_reasons.items()
                    if name == canonical and value > 0})),
            })
        return rows

    def extract(self, text) -> list:
        """返回排序后的技能标准名列表。"""
        return sorted({canonical for canonical, _alias, _group in self.match_details(text)})


def llm_skill_set(config: SkillConfig) -> set:
    """大模型相关技能集合（配置驱动）。"""
    return set(config.llm_skills)


def skill_group_count(config: SkillConfig, skills, groups) -> int:
    """岗位技能集合中落在指定技能组内的标准实体数量。"""
    return len(set(skills or []) & config.group_skills(groups))


def alias_table(config: SkillConfig) -> list:
    """别名 → 标准名映射表（写入技能审计表，用于人工核对别名唯一性）。"""
    rows = []
    for alias, canonical in config.alias_to_skill.items():
        rows.append({
            '别名': alias,
            '技能标准名': canonical,
            '技能组': config.skill_to_group.get(canonical, ''),
            'feature_family': config.skill_to_family.get(canonical, ''),
            '是否标准名本身': '是' if alias == canonical else '否',
        })
    return sorted(rows, key=lambda row: (row['技能标准名'], row['别名']))


def resolve_rank_layers(config: SkillConfig) -> dict:
    """把技能标准名归属到榜单层级（按配置顺序，每个技能只属于第一个命中的层级）。

    返回 {层级: {技能标准名}}；层级划分只做视角映射，不改变 feature_family / group 归属。
    """
    layers: dict = {}
    assigned: set = set()
    for layer, rule in config.rank_layers.items():
        members = set()
        for family in (rule or {}).get('families') or []:
            members |= config.family_skills(str(family))
        for group in (rule or {}).get('groups') or []:
            members |= config.group_skills([str(group)])
        for skill in (rule or {}).get('skills') or []:
            members.add(str(skill))
        layers[layer] = {skill for skill in members if skill not in assigned}
        assigned |= layers[layer]
    return layers


def skill_layer_map(config: SkillConfig) -> dict:
    """技能标准名 → 榜单层级（未归属任何层级时返回空串，由门禁拦截）。"""
    mapping = {}
    for layer, skills in resolve_rank_layers(config).items():
        for skill in skills:
            mapping[skill] = layer
    return mapping


def build_job_skill_membership(config: SkillConfig, records) -> 'object':
    """构建岗位 × 规范技能 long-format 关系表（(intern_id, canonical_skill) 唯一）。

    records 元素需包含：intern_id、skills（[{canonical, group, hit_count}]）、match_scope。
    """
    import pandas as pd  # noqa: PLC0415 - 局部导入，避免模块级依赖

    rows = []
    for record in records:
        job_id = record['intern_id']
        for item in record.get('skills') or []:
            canonical = item['canonical']
            rows.append({
                schema.SKILL_MEMBERSHIP_ID_FIELD: job_id,
                schema.SKILL_MEMBERSHIP_SKILL_FIELD: canonical,
                'feature_family': config.skill_to_family.get(canonical, ''),
                'group': config.skill_to_group.get(canonical, ''),
                'match_scope': record.get('match_scope', ''),
                'hit_count': int(item.get('hit_count', 1)),
            })
    frame = pd.DataFrame(rows, columns=schema.JOB_SKILL_MEMBERSHIP_COLUMNS)
    return frame


def keyword_match_plain(text, keywords) -> list:
    """普通业务/AI 关键词命中（公司简介使用；词表来自配置，非技能标准名）。

    拉丁关键词同样要求词边界，避免 RAG 误命中 fragment 之类的子串。
    """
    if not isinstance(text, str) or not text:
        return []
    lowered = text.lower()
    hits = []
    for keyword in keywords:
        if not keyword:
            continue
        if _is_latin_alias(keyword):
            target = str(keyword).lower()
            start = 0
            while True:
                index = lowered.find(target, start)
                if index < 0:
                    break
                if _boundary_ok(text, index, index + len(target), str(keyword)):
                    hits.append(keyword)
                    break
                start = index + 1
        elif keyword in text:
            hits.append(keyword)
    return sorted(set(hits))

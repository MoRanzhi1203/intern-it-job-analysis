# -*- coding: utf-8 -*-
"""Schema 模块：统一维护列名映射、字段说明与各类字段清单。

禁止在其他脚本中重复硬编码 30 个字段的中英文映射。
"""

from __future__ import annotations

# ---- 英文 → 中文列名映射（与 MySQL 实际列名严格一一对应） ----
COLUMN_NAME_CN = {
    'id': '记录ID',
    'group_name': '岗位大类',
    'item_text': '岗位细分类',
    'item_url': '岗位搜索链接',
    'search_page': '搜索页码',
    'intern_id': '实习岗位ID',
    'detail_url': '岗位详情链接',
    'job_title_detail': '岗位标题',
    'publish_time_detail': '发布时间',
    'salary_detail': '薪资信息',
    'city_detail': '工作城市',
    'degree_detail': '学历要求',
    'days_per_week_detail': '每周到岗要求',
    'intern_months_detail': '实习时长要求',
    'job_tags_detail': '岗位标签',
    'job_header_img_links': '岗位头图链接',
    'job_description': '岗位描述',
    'resume_language': '简历语言要求',
    'deadline_detail': '投递截止日期',
    'job_address': '工作地址',
    'company_name_detail': '公司名称',
    'company_desc_detail': '公司简介',
    'company_tags_detail': '公司标签',
    'industry_detail': '所属行业',
    'company_nature_detail': '公司性质',
    'company_size_detail': '公司规模',
    'company_location_detail': '公司所在地',
    'company_intro_img_links': '公司介绍图片链接',
    'created_at': '数据创建时间',
    'updated_at': '数据更新时间',
}

# ---- Stage 03 前置语义映射：公司介绍图片链接 → 公司认证标签 ----
# 源字段仅存在于 interim 数据，语义映射成功后即从 Stage 03 工作副本与
# 最终唯一岗位表删除，不得出现在任何 processed 输出中。
CERT_SOURCE_FIELD = '公司介绍图片链接'
CERT_TAG_FIELD = '公司认证标签'

# ---- 字段说明 ----
COLUMN_DESCRIPTION = {
    'id': '数据库自增主键，采集记录的入库序号，仅用于数据追溯',
    'group_name': '采集时的一级岗位大类（如运营、产品、人工智能等）',
    'item_text': '采集时的搜索关键词，即岗位细分类，同一岗位可能命中多个细分类',
    'item_url': '搜索列表页链接，仅具搜索分类意义，不是岗位实体属性',
    'search_page': '该条记录来自搜索结果第几页，属抓取来源字段',
    'intern_id': '实习僧岗位 ID（inn_ 开头），岗位实体主键候选',
    'detail_url': '岗位详情页链接，岗位身份辅助校验字段',
    'job_title_detail': '岗位标题，招聘方发布的职位名称',
    'publish_time_detail': '岗位发布时间，原始为字符串时间戳',
    'salary_detail': '薪资信息原文，可能为区间（如 150-300/天）或薪资面议',
    'city_detail': '工作城市，取自详情页岗位信息',
    'degree_detail': '学历要求（本科 / 硕士 / 大专 / 博士 / 不限）',
    'days_per_week_detail': '每周到岗要求，原文形如 4天／周',
    'intern_months_detail': '实习时长要求，原文形如 实习3个月',
    'job_tags_detail': '岗位福利标签原文，顿号分隔',
    'job_header_img_links': '岗位头图链接，其中急招图标可派生急招标志',
    'job_description': '岗位描述全文，含岗位职责与任职资格',
    'resume_language': '简历语言要求（中文 / 英文等）',
    'deadline_detail': '岗位投递截止日期',
    'job_address': '工作地点详细地址',
    'company_name_detail': '公司名称',
    'company_desc_detail': '公司简介原文',
    'company_tags_detail': '公司标签原文',
    'industry_detail': '公司所属行业',
    'company_nature_detail': '公司性质（民营 / 外资 / 国有等）',
    'company_size_detail': '公司规模（人数区间）',
    'company_location_detail': '公司所在地，粒度可能与工作城市不一致',
    'company_intro_img_links': '公司介绍图片链接，Stage 03 前置语义映射源字段，映射成功后删除',
    'created_at': '记录写入数据库的时间',
    'updated_at': '记录最近一次更新的时间',
}

# ---- 身份字段 ----
ID_FIELD = '实习岗位ID'
URL_FIELD = '岗位详情链接'
URL_NORM_FIELD = '岗位详情链接_规范化'
RECORD_ID_FIELD = '记录ID'
SEARCH_URL_FIELD = '岗位搜索链接'
PAGE_FIELD = '搜索页码'
IDENTITY_FIELDS = [ID_FIELD, URL_FIELD]

# ---- 唯一岗位表禁止出现的字段（去重后必须删除） ----
FORBIDDEN_UNIQUE_FIELDS = [RECORD_ID_FIELD, SEARCH_URL_FIELD, PAGE_FIELD, CERT_SOURCE_FIELD]

# ---- 岗位分类字段（同一岗位可能被多个搜索类别命中，需聚合保留） ----
CATEGORY_GROUP_FIELD = '岗位大类'
CATEGORY_ITEM_FIELD = '岗位细分类'
CATEGORY_SET_FIELDS = ['岗位大类集合', '岗位细分类集合']
CATEGORY_HIT_FIELDS = ['岗位大类命中数', '岗位细分类命中数']
CATEGORY_COUNT_FIELDS = ['岗位大类计数', '岗位细分类计数']
CATEGORY_AGG_FIELDS = CATEGORY_SET_FIELDS + CATEGORY_HIT_FIELDS + CATEGORY_COUNT_FIELDS
ORIGINAL_RECORD_FIELD = '原始记录数'

# ---- 岗位—分类关系表（岗位分类关系的权威数据源） ----
CATEGORY_MEMBERSHIP_COLUMNS = [ID_FIELD, CATEGORY_GROUP_FIELD, CATEGORY_ITEM_FIELD, '原始命中次数']

# ---- 时间字段（采集/版本时间，不作为业务冲突） ----
TIME_FIELDS = ['数据创建时间', '数据更新时间']

# ---- 业务字段（参与重复组一致性判定） ----
BUSINESS_FIELDS = [
    '岗位标题', '发布时间', '薪资信息', '工作城市', '学历要求', '每周到岗要求',
    '实习时长要求', '岗位标签', '岗位头图链接', '岗位描述', '简历语言要求',
    '投递截止日期', '工作地址', '公司名称', '公司简介', '公司标签', '所属行业',
    '公司性质', '公司规模', '公司所在地', CERT_TAG_FIELD,
]

# ---- 关键业务字段（冲突必须单独留档） ----
KEY_CONFLICT_FIELDS = ['岗位标题', '薪资信息', '公司名称', '工作城市', '岗位描述']

# ---- 冲突汇总字段 ----
CONFLICT_COUNT_FIELD = '业务字段冲突数'
KEY_CONFLICT_FLAG_FIELD = '关键业务字段冲突标志'

# ---- 人工复核队列关注字段（同ID下多值，需人工判定是否为页面版本漂移） ----
REVIEW_FIELDS = ['薪资信息', '公司名称', '工作城市', '工作地址']

# ---- Stage 03 派生字段说明（不属于 MySQL 原始 30 列） ----
DERIVED_FIELD_DESCRIPTION = {
    CERT_TAG_FIELD: '该岗位对应公司在实习僧页面展示的认证/荣誉图标语义集合（list[str]）',
    '岗位大类集合': '该岗位命中的全部岗位大类（去重排序，list[str]）',
    '岗位细分类集合': '该岗位命中的全部岗位细分类（去重排序，list[str]）',
    '岗位大类命中数': '该岗位命中的不同岗位大类数量（int）',
    '岗位细分类命中数': '该岗位命中的不同岗位细分类数量（int）',
    '岗位大类计数': '该岗位各岗位大类出现次数的稳定 JSON 字符串',
    '岗位细分类计数': '该岗位各岗位细分类出现次数的稳定 JSON 字符串',
    ORIGINAL_RECORD_FIELD: '该岗位在原始采集数据中的记录条数',
    CONFLICT_COUNT_FIELD: '该岗位下存在多值的业务字段数量',
    KEY_CONFLICT_FLAG_FIELD: '关键业务字段是否存在组内多值（1 是 / 0 否）',
    '代表记录选择依据': '代表记录按四级规则选中的依据',
}

# ---- 一致性审计字段 = 业务字段 + 规范化链接 ----
AUDIT_FIELDS = BUSINESS_FIELDS + [URL_NORM_FIELD]

# ---- 唯一岗位表：来源/分类聚合字段 ----
SOURCE_AGG_FIELDS = CATEGORY_AGG_FIELDS + [ORIGINAL_RECORD_FIELD]

# ---- 唯一岗位表：正式保留字段（顺序即输出顺序） ----
UNIQUE_JOB_COLUMNS = [
    ID_FIELD, URL_FIELD,
    '岗位标题', '发布时间', '薪资信息', '工作城市', '学历要求', '每周到岗要求',
    '实习时长要求', '岗位标签', '岗位头图链接', '岗位描述', '简历语言要求',
    '投递截止日期', '工作地址', '公司名称', '公司简介', '公司标签', '所属行业',
    '公司性质', '公司规模', '公司所在地', CERT_TAG_FIELD,
    '数据创建时间', '数据更新时间',
    *CATEGORY_SET_FIELDS, *CATEGORY_HIT_FIELDS, *CATEGORY_COUNT_FIELDS,
    ORIGINAL_RECORD_FIELD,
    CONFLICT_COUNT_FIELD, KEY_CONFLICT_FLAG_FIELD,
    '代表记录选择依据',
]

# ---- 关键字段（Stage 00 校验用） ----
REQUIRED_RAW_FIELDS = ['id', 'group_name', 'item_text', 'intern_id', 'item_url',
                       'detail_url', 'salary_detail', 'job_description']

# ---- 代表记录选择依据取值 ----
SELECTION_BASIS_VALUES = ['唯一记录', '完整度最高', '完整度相同_更新时间最新',
                          '完整度相同_创建时间最新', '完全一致_原始首条']

# ============================================================================
# 岗位重复观测时序重构：观测快照层 / 版本时序层 / 变化事件层 / 最终实体层
# ============================================================================

# ---- 观测快照层 ----
OBSERVATION_TIME_FIELD = '观测时间'
SNAPSHOT_RECORD_COUNT_FIELD = '原始观测记录数'
SNAPSHOT_GROUP_COUNT_FIELD = '来源岗位大类数量'
SNAPSHOT_ITEM_COUNT_FIELD = '来源岗位细分类数量'
SNAPSHOT_COLUMNS = [
    ID_FIELD, URL_FIELD, OBSERVATION_TIME_FIELD,
    SNAPSHOT_RECORD_COUNT_FIELD, SNAPSHOT_GROUP_COUNT_FIELD, SNAPSHOT_ITEM_COUNT_FIELD,
    *BUSINESS_FIELDS, '数据创建时间', '数据更新时间',
]

# ---- 两套版本签名 ----
CORE_SIGNATURE_FIELD = '核心业务签名'
FULL_SIGNATURE_FIELD = '完整页面签名'
CORE_VERSION_FIELD = '核心版本号'
FULL_VERSION_FIELD = '完整页面版本号'

# 明确排除在核心业务签名之外：发布时间 / 投递截止日期 / 岗位头图链接 / 搜索来源 / 采集时间
CORE_SIGNATURE_EXCLUDED_FIELDS = [
    '发布时间', '投递截止日期', '岗位头图链接',
    CATEGORY_GROUP_FIELD, CATEGORY_ITEM_FIELD, PAGE_FIELD,
    '数据创建时间', '数据更新时间',
]
CORE_SIGNATURE_FIELDS = [field for field in BUSINESS_FIELDS
                         if field not in CORE_SIGNATURE_EXCLUDED_FIELDS]
FULL_SIGNATURE_EXTRA_FIELDS = ['发布时间', '投递截止日期', '岗位头图链接']
FULL_SIGNATURE_FIELDS = CORE_SIGNATURE_FIELDS + FULL_SIGNATURE_EXTRA_FIELDS

# ---- 版本时序层字段（顺序即输出顺序） ----
VERSION_FIRST_TIME_FIELD = '版本首次观测时间'
VERSION_LAST_TIME_FIELD = '版本末次观测时间'
VERSION_NEXT_TIME_FIELD = '下一版本开始时间'
VERSION_DURATION_FIELD = '已观测持续时间_天'
VERSION_GAP_FIELD = '到下一版本间隔_天'
VERSION_IS_FINAL_FIELD = '是否最终核心版本'
VERSION_OBS_COUNT_FIELD = '版本观测次数'
VERSION_FULL_COUNT_FIELD = '该核心版本内完整页面版本数'
VERSION_CHANGED_COUNT_FIELD = '相对上一核心版本变化字段数'
VERSION_CHANGED_LIST_FIELD = '相对上一核心版本变化字段列表'
VERSION_CHANGE_FLAG_FIELDS = ['是否标题变化', '是否薪资变化', '是否城市变化',
                             '是否公司变化', '是否岗位描述变化']
VERSION_HISTORY_COLUMNS = [
    ID_FIELD, URL_FIELD,
    CORE_VERSION_FIELD, FULL_VERSION_FIELD, CORE_SIGNATURE_FIELD, FULL_SIGNATURE_FIELD,
    VERSION_FIRST_TIME_FIELD, VERSION_LAST_TIME_FIELD, VERSION_NEXT_TIME_FIELD,
    VERSION_DURATION_FIELD, VERSION_GAP_FIELD,
    VERSION_IS_FINAL_FIELD, VERSION_OBS_COUNT_FIELD, VERSION_FULL_COUNT_FIELD,
    *FULL_SIGNATURE_FIELDS,
    VERSION_CHANGED_COUNT_FIELD, VERSION_CHANGED_LIST_FIELD,
    *VERSION_CHANGE_FLAG_FIELDS,
]

# ---- 变化事件层字段 ----
CHANGE_EVENT_COLUMNS = [ID_FIELD, '旧核心版本号', '新核心版本号', '变化时间', '变化字段',
                        '旧值', '新值', '岗位标题', '公司名称', '工作城市']

# ---- 最终实体层：版本摘要字段 ----
ENTITY_CORE_VERSION_COUNT_FIELD = '核心版本数'
ENTITY_FULL_VERSION_COUNT_FIELD = '完整页面版本数'
ENTITY_MULTI_VERSION_FIELD = '是否多版本岗位'
ENTITY_FINAL_VERSION_FIELD = '最终核心版本号'
ENTITY_FINAL_FIRST_TIME_FIELD = '最终版本首次观测时间'
ENTITY_FINAL_LAST_TIME_FIELD = '最终版本末次观测时间'
ENTITY_VERSION_FIELDS = [
    ENTITY_CORE_VERSION_COUNT_FIELD, ENTITY_FULL_VERSION_COUNT_FIELD,
    ENTITY_MULTI_VERSION_FIELD, ENTITY_FINAL_VERSION_FIELD,
    ENTITY_FINAL_FIRST_TIME_FIELD, ENTITY_FINAL_LAST_TIME_FIELD,
    '历史是否发生薪资变化', '历史是否发生城市变化', '历史是否发生公司变化',
    '历史是否发生标题变化', '历史是否发生岗位描述变化',
]
ENTITY_JOB_COLUMNS = [*UNIQUE_JOB_COLUMNS, *ENTITY_VERSION_FIELDS]

# ---- 版本变化重点标志 → 实体层历史变化字段 ----
VERSION_FLAG_TO_ENTITY_FIELD = {
    '是否薪资变化': '历史是否发生薪资变化',
    '是否城市变化': '历史是否发生城市变化',
    '是否公司变化': '历史是否发生公司变化',
    '是否标题变化': '历史是否发生标题变化',
    '是否岗位描述变化': '历史是否发生岗位描述变化',
}

# ---- 版本变化字段 → 观察字段的映射（用于变化事件与一致性口径） ----
CHANGE_FLAG_FIELDS = {
    '是否标题变化': ['岗位标题'],
    '是否薪资变化': ['薪资信息'],
    '是否城市变化': ['工作城市'],
    '是否公司变化': ['公司名称'],
    '是否岗位描述变化': ['岗位描述'],
}

# ---- 派生字段说明补充（观测快照层 / 版本时序层 / 变化事件层 / 最终实体层） ----
DERIVED_FIELD_DESCRIPTION.update({
    OBSERVATION_TIME_FIELD: '观测时点，等于该快照的数据更新时间',
    SNAPSHOT_RECORD_COUNT_FIELD: '折叠到该观测快照的原始采集记录条数',
    SNAPSHOT_GROUP_COUNT_FIELD: '该观测时点命中的不同岗位大类数量（搜索来源，不进入版本签名）',
    SNAPSHOT_ITEM_COUNT_FIELD: '该观测时点命中的不同岗位细分类数量（搜索来源，不进入版本签名）',
    CORE_SIGNATURE_FIELD: '核心业务字段的稳定 SHA256 签名（不含发布时间/投递截止日期/岗位头图链接）',
    FULL_SIGNATURE_FIELD: '完整页面字段的稳定 SHA256 签名（核心字段 + 发布时间/投递截止日期/岗位头图链接）',
    CORE_VERSION_FIELD: '连续相同核心业务签名压缩后的版本序号（同一岗位内严格递增）',
    FULL_VERSION_FIELD: '完整页面版本序号（非递减，>= 核心版本号）',
    VERSION_FIRST_TIME_FIELD: '该版本第一次被观测到的数据更新时间',
    VERSION_LAST_TIME_FIELD: '该版本最后一次被观测到的数据更新时间',
    VERSION_NEXT_TIME_FIELD: '下一核心版本的首次观测时间；最终版本为空',
    VERSION_DURATION_FIELD: '版本末次观测时间 − 版本首次观测时间（天）',
    VERSION_GAP_FIELD: '下一版本开始时间 − 本版本末次观测时间（天）；最终版本为空',
    VERSION_IS_FINAL_FIELD: '是否为该岗位的最终核心版本',
    VERSION_OBS_COUNT_FIELD: '该版本被多少个观测快照观测到（非原始搜索记录条数）',
    VERSION_FULL_COUNT_FIELD: '该核心版本内包含的完整页面版本数量',
    VERSION_CHANGED_COUNT_FIELD: '相对上一核心版本发生变化的业务字段数量（首个版本为 0）',
    VERSION_CHANGED_LIST_FIELD: '相对上一核心版本发生变化的业务字段列表（JSON 数组）',
    '是否标题变化': '本核心版本相对上一版本是否发生岗位标题变化',
    '是否薪资变化': '本核心版本相对上一版本是否发生薪资变化',
    '是否城市变化': '本核心版本相对上一版本是否发生工作城市变化',
    '是否公司变化': '本核心版本相对上一版本是否发生公司名称变化',
    '是否岗位描述变化': '本核心版本相对上一版本是否发生岗位描述变化',
    '旧核心版本号': '变化前的核心版本号',
    '新核心版本号': '变化后的核心版本号',
    '变化时间': '新核心版本的首次观测时间',
    '变化字段': '本次版本切换发生变化的业务字段',
    '旧值': '变化前的取值',
    '新值': '变化后的取值',
    ENTITY_CORE_VERSION_COUNT_FIELD: '该岗位历史核心业务版本数量',
    ENTITY_FULL_VERSION_COUNT_FIELD: '该岗位历史完整页面版本数量（>= 核心版本数）',
    ENTITY_MULTI_VERSION_FIELD: '该岗位是否发生过核心业务版本变化',
    ENTITY_FINAL_VERSION_FIELD: '该岗位最终核心版本号',
    ENTITY_FINAL_FIRST_TIME_FIELD: '最终核心版本的首次观测时间',
    ENTITY_FINAL_LAST_TIME_FIELD: '最终核心版本的末次观测时间',
    '历史是否发生薪资变化': '该岗位历史核心版本中是否出现过薪资变化（1 是 / 0 否）',
    '历史是否发生城市变化': '该岗位历史核心版本中是否出现过工作城市变化（1 是 / 0 否）',
    '历史是否发生公司变化': '该岗位历史核心版本中是否出现过公司名称变化（1 是 / 0 否）',
    '历史是否发生标题变化': '该岗位历史核心版本中是否出现过岗位标题变化（1 是 / 0 否）',
    '历史是否发生岗位描述变化': '该岗位历史核心版本中是否出现过岗位描述变化（1 是 / 0 否）',
})


# ============================================================================
# 文本语义层（Stage 06~10：岗位版本文本语料 / 技能特征 / 语义事件 / 公司实体）
# ============================================================================

# ---- 文本语料层（一行 = 一个岗位的一个核心版本） ----
TEXT_FIRST_TIME_FIELD = '版本首次观测时间'
TEXT_LAST_TIME_FIELD = '版本末次观测时间'
TITLE_RAW_FIELD = '岗位标题_原始'
TITLE_CLEAN_FIELD = '岗位标题_清洗'
JD_RAW_FIELD = '岗位描述_原始'
JD_CLEAN_FIELD = '岗位描述_清洗'
JD_SEMANTIC_FIELD = '岗位描述_语义分析版'
JD_SAFE_FIELD = '岗位描述_模型安全版'
JD_CHAR_COUNT_FIELD = '岗位描述字符数'
JD_TOKEN_COUNT_FIELD = '岗位描述分词数'
JD_PARAGRAPH_COUNT_FIELD = '岗位描述段落数'
JD_SPLIT_STATUS_FIELD = '岗位描述分段状态'
JD_SALARY_HIT_FIELD = '岗位描述薪资模式命中数'
JD_SALARY_FLAG_FIELD = '岗位描述是否含薪资信息'
JD_SAFE_RESIDUAL_FIELD = '模型安全版薪资残留数'
SKILL_SET_FIELD = '技能集合'
SKILL_COUNT_FIELD = '技能数量'
SKILL_GROUP_FIELD = '技能组列表'
# ---- 技能提取范围与分组计数（Stage 07 岗位级紧凑画像） ----
SKILL_SCOPE_FIELD = '技能提取范围'
SKILL_MATCH_SCOPE_VALUES = ['REQUIREMENT_SECTION', 'FULL_TEXT_FALLBACK', 'EMPTY_TEXT']
SKILL_LANGUAGE_COUNT_FIELD = '编程语言技能数'
SKILL_DATABASE_COUNT_FIELD = '数据库技能数'
SKILL_SECURITY_COUNT_FIELD = '安全技能数'
SKILL_OFFICE_COUNT_FIELD = '办公工具技能数'
# ---- 岗位 × 规范技能 long-format 关系表（一行 = 一个岗位 × 一个规范技能） ----
SKILL_MEMBERSHIP_ID_FIELD = 'intern_id'
SKILL_MEMBERSHIP_SKILL_FIELD = 'canonical_skill'
JOB_SKILL_MEMBERSHIP_COLUMNS = [
    SKILL_MEMBERSHIP_ID_FIELD, SKILL_MEMBERSHIP_SKILL_FIELD,
    'feature_family', 'group', 'match_scope', 'hit_count',
]

# ---- 技能三级结构（feature_family → group → canonical，Refinement R1） ----
SKILL_FAMILY_FIELD = '技能一级类型'
SKILL_FAMILY_LIST_FIELD = '技能一级类型列表'
SKILL_GROUP_LIST_FIELD = '技能组列表'
SKILL_FAMILY_NAMES = ['技术技能', '工程工具', '数据工具', '业务能力', '技术领域']
SKILL_FAMILY_SET_FIELDS = {family: f'{family}集合' for family in SKILL_FAMILY_NAMES}
SKILL_FAMILY_COUNT_FIELDS = {family: f'{family}数' for family in SKILL_FAMILY_NAMES}
AI_SKILL_COUNT_FIELD = 'AI技能数'
LLM_SKILL_COUNT_FIELD = '大模型技能数'
COMPANY_PROFILE_RAW_FIELD = '公司简介_原始'
COMPANY_PROFILE_CLEAN_FIELD = '公司简介_清洗'
COMPANY_PROFILE_CHAR_COUNT_FIELD = '公司简介字符数'
COMPANY_PROFILE_MISSING_FIELD = '公司简介是否缺失标志'
JOB_TAG_LIST_FIELD = '岗位标签列表'
COMPANY_TAG_RAW_FIELD = '公司标签_原始'
COMPANY_TAG_LIST_FIELD = '公司标签列表'

TEXT_CORPUS_COLUMNS = [
    ID_FIELD, CORE_VERSION_FIELD, TEXT_FIRST_TIME_FIELD, TEXT_LAST_TIME_FIELD,
    VERSION_IS_FINAL_FIELD,
    TITLE_RAW_FIELD, JD_RAW_FIELD, '岗位标签',
    TITLE_CLEAN_FIELD, JD_CLEAN_FIELD, JOB_TAG_LIST_FIELD,
    JD_SEMANTIC_FIELD, JD_SAFE_FIELD,
    '职责文本', '任职要求文本', '技能要求文本', '加分项文本', '福利文本', '其他文本',
    JD_SPLIT_STATUS_FIELD,
    '职责段存在标志', '要求段存在标志', '技能段存在标志', '加分段存在标志', '福利段存在标志',
    JD_CHAR_COUNT_FIELD, JD_TOKEN_COUNT_FIELD, JD_PARAGRAPH_COUNT_FIELD,
    JD_SALARY_FLAG_FIELD, JD_SALARY_HIT_FIELD, JD_SAFE_RESIDUAL_FIELD,
    SKILL_SET_FIELD, SKILL_COUNT_FIELD, SKILL_GROUP_FIELD,
    '公司名称', COMPANY_PROFILE_RAW_FIELD, COMPANY_PROFILE_CLEAN_FIELD,
    COMPANY_TAG_RAW_FIELD, COMPANY_TAG_LIST_FIELD,
    '所属行业', '公司性质', '公司规模', '公司所在地',
    COMPANY_PROFILE_CHAR_COUNT_FIELD, COMPANY_PROFILE_MISSING_FIELD,
]

# ---- 岗位文本特征层（一行 = 一个最终岗位实体） ----
JOB_DIRECTION_FIELD = '岗位方向_文本'
JOB_DIRECTION_MATCH_FIELD = '岗位方向_与平台分类一致标志'
JOB_TITLE_NORMALIZED_FIELD = '岗位标题_规范'
HISTORY_SKILL_SET_FIELD = '历史技能集合'
HISTORY_SKILL_COUNT_FIELD = '历史技能数量'
JOB_TEXT_FEATURE_COLUMNS = [
    ID_FIELD, JOB_TITLE_NORMALIZED_FIELD, JOB_DIRECTION_FIELD, JOB_DIRECTION_MATCH_FIELD,
    JD_CHAR_COUNT_FIELD, JD_TOKEN_COUNT_FIELD, JD_SPLIT_STATUS_FIELD,
    JOB_TAG_LIST_FIELD, '岗位标签数量',
    SKILL_SET_FIELD, SKILL_COUNT_FIELD, HISTORY_SKILL_SET_FIELD, HISTORY_SKILL_COUNT_FIELD,
    SKILL_GROUP_FIELD, SKILL_FAMILY_LIST_FIELD, SKILL_SCOPE_FIELD,
    *SKILL_FAMILY_SET_FIELDS.values(), *SKILL_FAMILY_COUNT_FIELDS.values(),
    AI_SKILL_COUNT_FIELD, LLM_SKILL_COUNT_FIELD,
    SKILL_LANGUAGE_COUNT_FIELD, SKILL_DATABASE_COUNT_FIELD,
    SKILL_SECURITY_COUNT_FIELD, SKILL_OFFICE_COUNT_FIELD,
    '是否Python', '是否Java', '是否SQL', '是否C++', '是否Go', '是否JavaScript',
    '是否机器学习', '是否深度学习', '是否NLP', '是否计算机视觉', '是否推荐系统',
    '是否PyTorch', '是否TensorFlow', '是否Spark', '是否Flink', '是否Docker',
    '是否Kubernetes', '是否Linux', '是否Git', '是否MySQL', '是否Redis',
    '是否Vue', '是否React', '是否TypeScript',
    '是否大模型', '是否RAG', '是否Agent', '是否LangChain', '是否向量数据库',
    '是否Embedding', '是否微调', '是否多模态', '是否AIGC',
    COMPANY_PROFILE_CHAR_COUNT_FIELD, COMPANY_TAG_LIST_FIELD,
    ENTITY_CORE_VERSION_COUNT_FIELD, ENTITY_FINAL_VERSION_FIELD,
]

# ---- 岗位文本语义双口径字段（Refinement R1：正式主口径 = 去薪资） ----
JD_SIMILARITY_FULL_FIELD = '岗位描述语义相似度_完整'
JD_DISTANCE_FULL_FIELD = '岗位描述语义距离_完整'
TFIDF_FULL_FIELD = 'TFIDF余弦相似度_完整'
JD_SIMILARITY_SAFE_FIELD = '岗位描述语义相似度_去薪资'
JD_DISTANCE_SAFE_FIELD = '岗位描述语义距离_去薪资'
TFIDF_SAFE_FIELD = 'TFIDF余弦相似度_去薪资'
# 仅作诊断：Embedding 距离不是可加性分解，禁止解释为「薪资文本贡献率」
DISTANCE_GAP_FIELD = '完整减去薪资安全语义距离差'
SIGNIFICANT_FIELD = '是否显著语义变化候选'
EXTREME_FIELD = '是否极端语义变化候选'
SIGNIFICANT_FULL_CALIBER_FIELD = '是否显著语义变化候选_完整口径'
EXTREME_FULL_CALIBER_FIELD = '是否极端语义变化候选_完整口径'
CANDIDATE_TYPE_FIELD = '语义变化候选类型'

# ---- BGE token 截断审计字段 ----
TOKEN_RAW_COUNT_FIELD = '原始token数'
TOKEN_MAX_COUNT_FIELD = '模型最大token数'
TOKEN_TRUNCATED_FIELD = '是否发生截断'
TOKEN_EXCEED_FIELD = '超出token数'
TOKEN_RETENTION_FIELD = '理论保留比例'
EMBEDDING_INDEX_COLUMNS = [
    ID_FIELD, CORE_VERSION_FIELD, 'text_type', 'embedding_row', 'model_name',
    TOKEN_RAW_COUNT_FIELD, TOKEN_MAX_COUNT_FIELD, TOKEN_TRUNCATED_FIELD,
    TOKEN_EXCEED_FIELD, TOKEN_RETENTION_FIELD,
]

# ---- 岗位文本语义事件层（一行 = 一个岗位一次相邻核心版本文本变化） ----
JOB_TEXT_EVENT_COLUMNS = [
    ID_FIELD, '旧核心版本号', '新核心版本号', '变化时间', '岗位标题', '公司名称',
    JD_RAW_FIELD, '新岗位描述',
    '旧文本字符数', '新文本字符数', '字符数变化', '字符数变化率',
    '旧文本词数', '新文本词数', '词数变化', '文本完全一致标志',
    'Token_Jaccard相似度', '编辑相似度',
    TFIDF_FULL_FIELD, JD_SIMILARITY_FULL_FIELD, JD_DISTANCE_FULL_FIELD,
    TFIDF_SAFE_FIELD, JD_SIMILARITY_SAFE_FIELD, JD_DISTANCE_SAFE_FIELD,
    DISTANCE_GAP_FIELD,
    '职责语义相似度', '职责语义距离', '任职要求语义相似度', '任职要求语义距离',
    '技能段语义相似度', '技能段语义距离',
    '职责段字符数变化率', '任职要求段字符数变化率', '技能段字符数变化率',
    '旧技能集合', '新技能集合', '新增技能集合', '删除技能集合', '保留技能集合',
    '新增技能数', '删除技能数', '技能变化数', '技能Jaccard相似度',
    '是否新增LLM', '是否删除LLM', '是否新增RAG', '是否删除RAG',
    '是否新增Agent', '是否删除Agent', '是否新增大模型相关技能', '是否删除大模型相关技能',
    '是否同时薪资变化', '是否同时城市变化', '是否同时公司变化',
    '是否同时岗位标题变化', '是否同时学历要求变化',
    '旧文本是否截断_完整', '新文本是否截断_完整',
    '旧文本是否截断_去薪资', '新文本是否截断_去薪资', '版本对是否存在截断',
    CANDIDATE_TYPE_FIELD, SIGNIFICANT_FIELD, EXTREME_FIELD,
    SIGNIFICANT_FULL_CALIBER_FIELD, EXTREME_FULL_CALIBER_FIELD,
]
# 岗位版本历史表允许追加的文本摘要字段（详细明细由文本事件表承载）
VERSION_TEXT_SUMMARY_FIELDS = [
    '相对上一版本JD语义距离', '相对上一版本JD语义距离_完整', '相对上一版本JD语义距离_去薪资',
    '相对上一版本新增技能数', '相对上一版本删除技能数',
    '是否显著JD语义变化候选',
]

# ---- 公司实体层 ----
COMPANY_ENTITY_ID_FIELD = 'company_entity_id'
COMPANY_NAME_ORIGINAL_FIELD = 'company_name_original'
COMPANY_NAME_NORMALIZED_FIELD = 'company_name_normalized'
COMPANY_MAPPING_METHOD_FIELD = '映射方式'
COMPANY_MAPPING_CONFIDENCE_FIELD = '映射置信度'
COMPANY_NEED_REVIEW_FIELD = '是否需人工复核'
COMPANY_FORMAL_FIELD = '是否进入正式公司时序'
# Refinement R1 新增：跨地域不再作为自动拆分正式实体的充分条件
COMPANY_LOCATION_COUNT_FIELD = '所在地数量'
COMPANY_LOCATION_SET_FIELD = '所在地集合'
COMPANY_CROSS_LOCATION_FIELD = '是否跨地域'
COMPANY_PROFILE_FP_COUNT_FIELD = '公司简介指纹数量'
COMPANY_SAME_NAME_AMBIGUOUS_FIELD = '是否存在同名跨地域歧义'
COMPANY_FORMAL_EXCLUDE_REASON_FIELD = '不可进入正式公司时序原因'
COMPANY_LEGACY_LOCATION_METHOD_FIELD = '旧映射方式_EXACT_NAME_LOCATION适用性'
COMPANY_ENTITY_MAP_COLUMNS = [
    COMPANY_ENTITY_ID_FIELD, COMPANY_NAME_ORIGINAL_FIELD, COMPANY_NAME_NORMALIZED_FIELD,
    '公司所在地', COMPANY_MAPPING_METHOD_FIELD, COMPANY_MAPPING_CONFIDENCE_FIELD,
    COMPANY_NEED_REVIEW_FIELD, COMPANY_FORMAL_FIELD, '岗位数', '观测记录数',
    COMPANY_LOCATION_COUNT_FIELD, COMPANY_LOCATION_SET_FIELD, COMPANY_CROSS_LOCATION_FIELD,
    COMPANY_PROFILE_FP_COUNT_FIELD, COMPANY_SAME_NAME_AMBIGUOUS_FIELD,
    COMPANY_LEGACY_LOCATION_METHOD_FIELD, COMPANY_FORMAL_EXCLUDE_REASON_FIELD,
]

# ---- 公司简介快照层（一行 = 一个正式公司实体 × 一个观测时间） ----
COMPANY_SNAPSHOT_STATE_FIELD = '公司简介快照状态'
COMPANY_SNAPSHOT_STATES = ['CONSISTENT', 'AMBIGUOUS', 'MISSING']
COMPANY_SNAPSHOT_CANDIDATE_COUNT_FIELD = '非空简介候选数'
COMPANY_SNAPSHOT_CANDIDATE_LIST_FIELD = '简介候选列表'
COMPANY_SNAPSHOT_CANDIDATE_SUPPORT_FIELD = '简介候选支持岗位数列表'
COMPANY_SNAPSHOT_MAIN_PROFILE_FIELD = '主候选简介'
COMPANY_SNAPSHOT_MAIN_SUPPORT_FIELD = '主候选简介支持岗位数'
COMPANY_SNAPSHOT_MAIN_SUPPORT_RATE_FIELD = '主候选简介支持率'
COMPANY_SNAPSHOT_JOB_COUNT_FIELD = '总来源岗位数'
COMPANY_SNAPSHOT_OBS_COUNT_FIELD = '总原始观测数'
COMPANY_SNAPSHOT_FORMAL_FIELD = '是否进入正式公司版本'
COMPANY_SNAPSHOT_JOB_SET_FIELD = '来源岗位集合'
COMPANY_PROFILE_SNAPSHOT_COLUMNS = [
    COMPANY_ENTITY_ID_FIELD, '公司规范名称', OBSERVATION_TIME_FIELD,
    COMPANY_SNAPSHOT_STATE_FIELD,
    COMPANY_SNAPSHOT_CANDIDATE_COUNT_FIELD, COMPANY_SNAPSHOT_CANDIDATE_LIST_FIELD,
    COMPANY_SNAPSHOT_CANDIDATE_SUPPORT_FIELD,
    COMPANY_SNAPSHOT_MAIN_PROFILE_FIELD, COMPANY_SNAPSHOT_MAIN_SUPPORT_FIELD,
    COMPANY_SNAPSHOT_MAIN_SUPPORT_RATE_FIELD,
    COMPANY_SNAPSHOT_JOB_COUNT_FIELD, COMPANY_SNAPSHOT_OBS_COUNT_FIELD,
    COMPANY_SNAPSHOT_FORMAL_FIELD,
    COMPANY_TAG_LIST_FIELD, '所属行业', '公司性质', '公司规模', COMPANY_LOCATION_SET_FIELD,
    COMPANY_SNAPSHOT_JOB_SET_FIELD,
]

# ---- 公司简介版本层 ----
COMPANY_PROFILE_VERSION_FIELD = '公司简介版本号'
COMPANY_PROFILE_SIGNATURE_FIELD = '公司简介语义签名'
COMPANY_PROFILE_OBS_COUNT_FIELD = '版本观测次数'
COMPANY_PROFILE_HISTORY_COLUMNS = [
    COMPANY_ENTITY_ID_FIELD, '公司规范名称', COMPANY_PROFILE_VERSION_FIELD,
    TEXT_FIRST_TIME_FIELD, TEXT_LAST_TIME_FIELD, COMPANY_PROFILE_OBS_COUNT_FIELD,
    COMPANY_PROFILE_CLEAN_FIELD, COMPANY_TAG_LIST_FIELD,
    '所属行业', '公司性质', '公司规模', '公司所在地',
    COMPANY_PROFILE_SIGNATURE_FIELD, '来源岗位数',
]

# ---- 公司文本语义事件层 ----
# Refinement R1 新增：跨越歧义快照的说明字段，避免把变化时间误读为精确发生时刻
COMPANY_EVENT_AMBIGUOUS_FLAG_FIELD = '中间存在歧义快照'
COMPANY_EVENT_AMBIGUOUS_COUNT_FIELD = '跨越歧义快照数量'
COMPANY_EVENT_OLD_SNAPSHOT_COUNT_FIELD = '旧版本有效快照数'
COMPANY_EVENT_NEW_SNAPSHOT_COUNT_FIELD = '新版本有效快照数'
COMPANY_EVENT_OLD_JOB_COUNT_FIELD = '旧版本来源岗位数'
COMPANY_EVENT_NEW_JOB_COUNT_FIELD = '新版本来源岗位数'
COMPANY_TEXT_EVENT_COLUMNS = [
    COMPANY_ENTITY_ID_FIELD, '公司规范名称',
    '旧公司简介版本号', '新公司简介版本号', '变化时间',
    '旧公司简介', '新公司简介',
    '旧文本字符数', '新文本字符数', '文本长度变化', '文本长度变化率',
    'TFIDF余弦相似度', 'Embedding语义相似度', 'Embedding语义距离',
    '新增业务关键词', '删除业务关键词', '新增关键词数', '删除关键词数',
    '是否新增AI关键词', '是否新增大模型关键词',
    COMPANY_EVENT_AMBIGUOUS_FLAG_FIELD, COMPANY_EVENT_AMBIGUOUS_COUNT_FIELD,
    COMPANY_EVENT_OLD_SNAPSHOT_COUNT_FIELD, COMPANY_EVENT_NEW_SNAPSHOT_COUNT_FIELD,
    COMPANY_EVENT_OLD_JOB_COUNT_FIELD, COMPANY_EVENT_NEW_JOB_COUNT_FIELD,
    '公司简介变化候选类型',
]


# ============================================================================
# 结构化业务字段清洗与薪资目标（Stage 11）
# ============================================================================

# ---- 薪资目标层（一行 = 一个最终岗位实体） ----
SALARY_RAW_FIELD = '薪资信息'
SALARY_MIN_FIELD = '薪资下限'
SALARY_MAX_FIELD = '薪资上限'
SALARY_MID_FIELD = '薪资中点'
SALARY_SPAN_FIELD = '薪资区间宽度'
SALARY_UNIT_FIELD = '薪资单位'
SALARY_NEGOTIABLE_FIELD = '是否面议'
SALARY_PARSE_STATUS_FIELD = '薪资解析状态'
SALARY_ANOMALY_FIELD = '薪资异常标志'
SALARY_FIRST_MIN_FIELD = '首版本薪资下限'
SALARY_FIRST_MID_FIELD = '首版本薪资中点'
SALARY_MID_DRIFT_FIELD = '薪资中点变化'
SALARY_UNIT_VALUE = '元/天'
SALARY_PARSE_STATUS_VALUES = ['已解析', '面议', '解析失败']
# 只标记「逻辑无效」值，极端值交由分布审计与缩尾敏感性处理（禁止拍脑袋阈值）
SALARY_ANOMALY_FLAGS = ['上下限倒置', '下限非正数', '上限非正数']
SALARY_TARGET_COLUMNS = [
    ID_FIELD, SALARY_RAW_FIELD, SALARY_MIN_FIELD, SALARY_MAX_FIELD, SALARY_MID_FIELD,
    SALARY_SPAN_FIELD, SALARY_UNIT_FIELD, SALARY_NEGOTIABLE_FIELD,
    SALARY_PARSE_STATUS_FIELD, SALARY_ANOMALY_FIELD,
    '历史是否发生薪资变化', SALARY_FIRST_MIN_FIELD, SALARY_FIRST_MID_FIELD,
    SALARY_MID_DRIFT_FIELD,
]

# ---- 结构化特征层（一行 = 一个最终岗位实体） ----
EDUCATION_RAW_FIELD = '学历要求'
EDUCATION_LEVEL_FIELD = '学历等级'
WEEKLY_RAW_FIELD = '每周到岗要求'
WEEKLY_DAYS_FIELD = '每周到岗天数'
DURATION_RAW_FIELD = '实习时长要求'
INTERN_MONTHS_FIELD = '实习月数'
SCALE_RAW_FIELD = '公司规模'
SCALE_MIN_FIELD = '公司规模下限'
SCALE_MAX_FIELD = '公司规模上限'
SCALE_MID_FIELD = '公司规模中点'
SCALE_LEVEL_FIELD = '公司规模等级'
SCALE_KNOWN_FIELD = '公司规模是否已知'
SCALE_SHIFT_FIELD = '公司规模槽位异常标志'
NATURE_RAW_FIELD = '公司性质'
NATURE_SHIFT_FIELD = '公司性质槽位异常标志'
LOCATION_RAW_FIELD = '公司所在地'
CITY_RAW_FIELD = '工作城市'
CITY_NORMALIZED_FIELD = '工作城市_规范'
CITY_MUNICIPALITY_FIELD = '是否直辖市'
STRUCTURED_FEATURE_COLUMNS = [
    ID_FIELD,
    EDUCATION_RAW_FIELD, EDUCATION_LEVEL_FIELD,
    WEEKLY_RAW_FIELD, WEEKLY_DAYS_FIELD,
    DURATION_RAW_FIELD, INTERN_MONTHS_FIELD,
    SCALE_RAW_FIELD, SCALE_MIN_FIELD, SCALE_MAX_FIELD, SCALE_MID_FIELD,
    SCALE_LEVEL_FIELD, SCALE_KNOWN_FIELD, SCALE_SHIFT_FIELD,
    NATURE_RAW_FIELD, NATURE_SHIFT_FIELD,
    LOCATION_RAW_FIELD, CITY_RAW_FIELD, CITY_NORMALIZED_FIELD, CITY_MUNICIPALITY_FIELD,
]


def english_to_chinese(columns) -> list:
    """按给定顺序返回对应的中文列名，缺失映射直接抛错。"""
    missing = [c for c in columns if c not in COLUMN_NAME_CN]
    if missing:
        raise KeyError(f'以下英文列名缺少中文映射: {missing}')
    return [COLUMN_NAME_CN[c] for c in columns]


def validate_mapping_coverage(columns) -> tuple:
    """校验映射字典与实际列是否一一对应，返回 (未映射列, 多余映射列)。"""
    columns = list(columns)
    unmapped = [c for c in columns if c not in COLUMN_NAME_CN]
    unused = [c for c in COLUMN_NAME_CN if c not in columns]
    return unmapped, unused

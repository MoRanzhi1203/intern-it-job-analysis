# -*- coding: utf-8 -*-
"""Stage27.0A 数据集截图 —— 对应的 Jupyter 查询单元（只读）。

用途：在 Jupyter / VS Code Notebook 中逐「单元」运行，得到论文中
`outputs/figures/evidence_native/03_data/*.png` 对应的行与列，
再由人工对输出区域截图。

约定：
1. 全部为只读查询，不写入任何文件、不修改冻结数据；
2. 输出为**真实数据**：公司名称、实习岗位 ID 均按冻结数据原值展示，不做任何代号替换；
3. 每个单元的输出即为对应图的**全部内容**（无标题栏、无 (a)/(b)），
   截图后图题由 Word 承担。

VS Code 中可直接按 `# %%` 分单元运行；也可把每段粘贴为 notebook 的一个 cell。
"""
# %% [单元 0] 环境与路径
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:                       # 纯 Python 运行时兜底
    display = print

ROOT = Path(r'e:\Jupter_Notebook\课程设计\course-design-2')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 240)
pd.set_option('display.max_colwidth', 60)
pd.set_option('display.unicode.east_asian_width', True)

RAW = ROOT / 'data' / 'raw' / 'shixiseng_job_details.parquet'
SALARY = ROOT / 'data' / 'processed' / 'job_salary_targets.parquet'
MODEL = ROOT / 'data' / 'processed' / 'job_salary_model_dataset.parquet'
UNIQUE = ROOT / 'data' / 'processed' / 'job_details_unique.parquet'
CORPUS = ROOT / 'data' / 'interim' / 'job_text_version_corpus.parquet'
MEMBERSHIP = ROOT / 'data' / 'features' / 'job_skill_membership.parquet'
ANOMALY = ROOT / 'outputs' / 'tables' / '22_company_attribute_semantic_anomaly_audit.xlsx'


# %% [单元 1] 公共工具
def hhmm(value) -> str:
    """业务时间截断到分钟（与图中一致）。"""
    text = str(value)
    return text[:16] if len(text) >= 16 else text


def flat_cell(value) -> str:
    """list / ndarray 单元格 → 顿号连接文本；空值 → '—'。"""
    if isinstance(value, (list, tuple, set)):
        items = [str(v) for v in value if str(v).strip()]
    elif isinstance(value, np.ndarray):
        items = [str(v) for v in value.tolist() if str(v).strip()]
    elif value is None:
        items = []
    else:
        text = str(value).strip()
        items = [] if text in ('', 'nan', 'None', '[]', "['']") else [text]
    return '、'.join(items) if items else '—'


print('环境就绪：', ROOT.name)


# %% [单元 2] 图 3-7 原始岗位观测数据的岗位与薪资字段片段
raw = pd.read_parquet(RAW)

frame = raw[['intern_id', 'job_title_detail', 'item_text', 'city_detail', 'salary_detail',
             'degree_detail', 'intern_months_detail', 'publish_time_detail',
             'deadline_detail', 'company_name_detail']].copy()
frame = frame[frame['job_title_detail'].str.len() <= 16]
frame = frame[frame['city_detail'].str.len() <= 4]
frame = frame[frame['intern_months_detail'].str.len() <= 8]
frame = frame.sort_values('intern_id', kind='stable').reset_index(drop=True)
step = max(len(frame) // 7, 1)
e04 = frame.iloc[[step * i for i in range(1, 7)]].reset_index(drop=True)

fig_e04a = pd.DataFrame({
    '岗位标识': e04['intern_id'].values,
    '岗位名称': e04['job_title_detail'].values,
    '搜索分类': e04['item_text'].values,
    '工作城市': e04['city_detail'].values,
    '薪资原文': e04['salary_detail'].values,
})
print('图 3-7 原始岗位观测数据的岗位与薪资字段片段  (6 行 × 5 列)')
display(fig_e04a)


# %% [单元 3] 图 3-8 原始岗位观测数据的业务时间与公司字段片段
# 依赖单元 2 的 e04（同一 kernel 内变量复用，保持 6 行记录顺序一致）
fig_e04b = pd.DataFrame({
    '岗位标识': e04['intern_id'].values,
    '学历要求': e04['degree_detail'].values,
    '实习时长': e04['intern_months_detail'].values,
    '发布时间': [hhmm(v) for v in e04['publish_time_detail']],
    '投递截止日期': [str(v) for v in e04['deadline_detail']],
    '公司名称': e04['company_name_detail'].values,
})
print('图 3-8 原始岗位观测数据的业务时间与公司字段片段  (6 行 × 6 列，与上图记录顺序一致)')
display(fig_e04b)


# %% [单元 4] 图 3-10 公司属性语义槽位异常修复前后数据片段
cand = pd.read_excel(ANOMALY, sheet_name='03_全部候选行')

def _cell(v):
    return '空' if pd.isna(v) else ('缺失' if v == '' else str(v))

picks = []
for kind in ['FULL_SHIFT', 'NATURE_ONLY', 'SIZE_ONLY']:
    row = cand[cand['异常类型'] == kind].iloc[0]
    picks.append([row['公司名称'], kind,
                  _cell(row['公司性质_原值']), _cell(row['公司规模_原值']),
                  _cell(row['公司所在地_原值']), '缺失' if pd.isna(row['公司性质_修复值'])
                  else str(row['公司性质_修复值']),
                  '缺失' if pd.isna(row['公司规模_修复值']) else str(row['公司规模_修复值']),
                  '缺失' if pd.isna(row['公司所在地_修复值']) else str(row['公司所在地_修复值'])])

fig_e07b = pd.DataFrame(picks, columns=['公司', '异常类型', '性质(前)', '规模(前)', '所在地(前)',
                                        '性质(后)', '规模(后)', '所在地(后)'])
print('图 3-10 公司属性语义槽位异常修复前后数据片段  (3 行 × 8 列)')
display(fig_e07b)


# %% [单元 5] 图 3-12 薪资字段解析前后数据片段
salary = pd.read_parquet(SALARY)
fmt = lambda v: '缺失' if pd.isna(v) else ('%g' % v)

picks = []
for cond, label in [
        (salary['薪资信息'].eq('120-150/天'), '区间'),
        (salary['薪资信息'].eq('100/天'), '单值'),
        (salary['薪资信息'].eq('薪资面议'), '面议'),
        (salary['薪资异常标志'].fillna('').str.contains('上下限倒置'), '异常区间')]:
    r = salary[cond].iloc[0]
    picks.append([label, str(r['薪资信息']), fmt(r['薪资下限']), fmt(r['薪资上限']),
                  fmt(r['薪资中点']), str(r['薪资解析状态']),
                  str(r['薪资异常标志']) or '—'])

fig_e08b = pd.DataFrame(picks, columns=['形态', '薪资原文', '下限', '上限', '中点',
                                        '解析状态', '异常标志'])
print('图 3-12 薪资字段解析前后数据片段  (4 行 × 7 列)')
display(fig_e08b)


# %% [单元 6] 图 3-14 岗位描述要求段提取示例
E09_JOB = 'inn_4fezolxa9tsh'
corpus = pd.read_parquet(CORPUS)
row = corpus[(corpus['实习岗位ID'] == E09_JOB)
             & (corpus['是否最终核心版本'] == 1)].iloc[0]

print('图 3-14 岗位描述要求段提取示例（公司名称 %s）' % row['公司名称'])
print('岗位标识 %s\n' % E09_JOB)
for label, text in [
        ('Q1 岗位描述_原始（节选）', str(row['岗位描述_原始'])[-160:]),
        ('Q2 岗位描述_模型安全版（节选）', str(row['岗位描述_模型安全版'])[-160:]),
        ('Q3 REQUIREMENT_SECTION（任职要求文本节选）', str(row['任职要求文本'])[:200])]:
    print(label)
    print(text)
    print('-' * 96)
print('Q1→Q2 为薪资表达清理，Q3 为要求段落口径的提取结果。')


# %% [单元 7] 图 3-15 要求段技能规范化结果片段
membership = pd.read_parquet(MEMBERSHIP)
sub = membership[(membership['intern_id'] == E09_JOB)
                 & (membership['match_scope'] == 'REQUIREMENT_SECTION')]
sub = sub.sort_values(['feature_family', 'canonical_skill'])

fig_e09c = pd.DataFrame({
    '规范技能': sub['canonical_skill'].values,
    '技能族': sub['feature_family'].values,
    '技能组': sub['group'].values,
    '命中次数': [int(v) for v in sub['hit_count']],
})
print('图 3-15 要求段技能规范化结果片段  (%d 行 × 4 列；匹配口径 = REQUIREMENT_SECTION)'
      % len(fig_e09c))
display(fig_e09c)


# %% [单元 8] 图 3-16 最终建模数据集岗位与地域字段片段
model = pd.read_parquet(MODEL)
unique = pd.read_parquet(UNIQUE)
frame = model.merge(unique[['实习岗位ID', '公司认证标签']], on='实习岗位ID', how='left')

usable = membership[membership['match_scope'].isin(['REQUIREMENT_SECTION',
                                                    'FULL_TEXT_FALLBACK'])]
skill_sets = usable.groupby('intern_id')['canonical_skill'].apply(set)

picks = frame.iloc[[7, 1500, 3600, 6200, 9500, 13200]].reset_index(drop=True)

fig_e10a = pd.DataFrame({
    '岗位标识': picks['实习岗位ID'].values,
    '薪资中点': ['%g' % v for v in picks['薪资中点']],
    '岗位大类': [flat_cell(v) for v in picks['岗位大类集合']],
    '工作城市': picks['工作城市_规范'].values,
    '学历要求': picks['学历要求'].values,
    '实习时长': picks['实习时长要求'].values,
    '每周到岗': picks['每周到岗要求'].values,
})
print('图 3-16 最终建模数据集岗位与地域字段片段  (6 行 × 7 列；正式建模样本 14,883 个岗位)')
display(fig_e10a)


# %% [单元 9] 图 3-17 最终建模数据集公司属性字段片段
# 依赖单元 8 的 picks（同一 kernel 内变量复用，保持 6 行顺序一致）

fig_e10b1 = pd.DataFrame({
    '岗位标识': picks['实习岗位ID'].values,
    '公司名称': picks['公司名称'].values,
    '公司规模': picks['公司规模'].values,
    '公司性质': picks['公司性质'].values,
    '所属行业': picks['所属行业'].values,
    '公司认证': [flat_cell(v) for v in picks['公司认证标签']],
})
print('图 3-17 最终建模数据集公司属性字段片段  (6 行 × 6 列，与上图记录顺序一致)')
display(fig_e10b1)


# %% [单元 10] 图 3-18 最终建模数据集技能与文本字段片段
# 依赖单元 8 的 picks / membership / skill_sets
fig_e10b2 = pd.DataFrame({
    '岗位标识': picks['实习岗位ID'].values,
    'skill_Python': ['1' if 'Python' in skill_sets.get(v, set()) else '0'
                     for v in picks['实习岗位ID']],
    'skill_SQL': ['1' if 'SQL' in skill_sets.get(v, set()) else '0'
                  for v in picks['实习岗位ID']],
    '技能计数': [str(int(v)) for v in picks['技能数量']],
    '描述字符数': [str(int(v)) for v in picks['岗位描述字符数']],
})
print('图 3-18 最终建模数据集技能与文本字段片段  (6 行 × 5 列，与上图记录顺序一致)')
display(fig_e10b2)
print('\n全部 9 张数据集截图对应的查询已完成。')

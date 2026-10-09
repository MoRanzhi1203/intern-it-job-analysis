# -*- coding: utf-8 -*-
"""E6：技能抽取人工核验——分层抽样表（未标注前不生成 Precision / Recall / F1）。

输出：
    data/interim/skill_annotation_sample.xlsx   供人工填写的标注表（含抽样说明与字段字典）

原则：
    1. 只从「有可解析岗位描述」的岗位中按岗位大类分层随机抽取；
    2. 规则抽取结果原样给出，供人工逐条确认，不由脚本改写；
    3. 人工确认字段留空，未完成标注前不计算任何准确率指标；
    4. 标注完成后以人工确认技能为金标准，分别评估要求段主统计范围、扩展统计范围、
       具体技术技能层级与较粗层级，并报告微平均与宏平均。

用法：
    python scripts\\04_E6_skill_annotation_sample.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = next(
    _candidate for _candidate in Path(__file__).resolve().parents
    if (_candidate / "data").is_dir() and (_candidate / "scripts").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import project_paths, schema  # noqa: E402

CORPUS = project_paths.TEXT_CORPUS_PARQUET
MEMBERSHIP = project_paths.JOB_SKILL_MEMBERSHIP_PARQUET
CATEGORY = project_paths.PROCESSED_CATEGORY_MEMBERSHIP_PARQUET
SAMPLE_OUT = project_paths.SKILL_ANNOTATION_SAMPLE_XLSX
SAMPLE_SIZE = 240
MIN_PER_STRATUM = 5
SEED = 42
TRUNCATE = 600


def main() -> int:
    corpus = pd.read_parquet(CORPUS)
    membership = pd.read_parquet(MEMBERSHIP)
    category = pd.read_parquet(CATEGORY)

    core = corpus[corpus['是否最终核心版本'] == 1].copy()
    core['描述可用'] = core['岗位描述_原始'].fillna('').astype(str).str.len() > 0
    pool = core[core['描述可用']].copy()
    print('可解析岗位描述的最终核心版本岗位:', len(pool))

    weight = (category.groupby([schema.ID_FIELD, '岗位大类'])['原始命中次数']
              .sum().reset_index())
    primary = (weight.sort_values(['实习岗位ID', '原始命中次数'], ascending=[True, False])
               .drop_duplicates('实习岗位ID').set_index(schema.ID_FIELD))
    pool['岗位大类'] = pool['实习岗位ID'].map(primary['岗位大类']).fillna('未分类')
    pool['岗位细分类'] = [', '.join(value[:6]) if isinstance(value, (list, np.ndarray)) else ''
                    for value in pool['实习岗位ID'].map(
                        category.groupby(schema.ID_FIELD)['岗位细分类']
                        .apply(lambda block: sorted(set(block))).to_dict())]

    skills = {}
    for job_id, block in membership.groupby('intern_id'):
        for scope, group in block.groupby('match_scope'):
            skills.setdefault(job_id, {})[scope] = sorted(set(group['canonical_skill']))
    pool['_要求段'] = pool['实习岗位ID'].map(
        lambda value: '、'.join(skills.get(value, {}).get('REQUIREMENT_SECTION', [])))
    pool['_扩展'] = pool['实习岗位ID'].map(
        lambda value: '、'.join(sorted(set(skills.get(value, {}).get('REQUIREMENT_SECTION', []))
                                       | set(skills.get(value, {}).get('FULL_TEXT_FALLBACK', [])))))

    rng = np.random.default_rng(SEED)
    sizes = pool['岗位大类'].value_counts()
    quota = np.maximum(MIN_PER_STRATUM, np.floor(sizes / sizes.sum() * SAMPLE_SIZE).astype(int))
    quota = quota.clip(upper=sizes)
    picked = []
    for label, count in quota.items():
        block = pool[pool['岗位大类'] == label]
        take = int(min(count, len(block)))
        index = rng.choice(block.index.to_numpy(), size=take, replace=False)
        picked.extend(index)
    sample = pool.loc[sorted(picked)].copy()
    sample = sample.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print('分层抽样条数:', len(sample), '| 层数:', sample['岗位大类'].nunique())

    table = pd.DataFrame({
        'intern_id': sample['实习岗位ID'],
        '公司名称': sample['公司名称'],
        '岗位大类': sample['岗位大类'],
        '岗位细分类': sample['岗位细分类'],
        '要求段原文（节选）': [str(value)[:TRUNCATE] for value in sample['任职要求文本']],
        '全文回退原文（节选）': [str(value)[:TRUNCATE] for value in sample['岗位描述_原始']],
        '规则抽取_要求段统计范围': sample['_要求段'],
        '规则抽取_扩展统计范围': sample['_扩展'],
        '人工确认技能': '',
        '是否可判定': '',
        '备注': '',
    })

    stratum = sample['岗位大类'].value_counts().rename_axis('岗位大类').reset_index(
        name='抽样条数')
    stratum['占抽样比'] = (stratum['抽样条数'] / len(sample)).round(4)
    stratum['抽样框岗位数'] = stratum['岗位大类'].map(sizes).fillna(0).astype(int)
    notes = pd.DataFrame([
        {'项目': '抽样总体说明', '取值': '有可解析岗位描述且已识别最终核心版本的岗位'},
        {'项目': '抽样框规模', '取值': int(len(pool))},
        {'项目': '分层变量', '取值': '岗位大类（平台分类，取命中次数最多的一类）'},
        {'项目': '抽样条数', '取值': int(len(sample))},
        {'项目': '随机种子', '取值': SEED},
        {'项目': '每层最少条数', '取值': MIN_PER_STRATUM},
        {'项目': '原文节选长度上限', '取值': TRUNCATE},
        {'项目': '规则抽取来源', '取值': 'data/features/job_skill_membership.parquet'},
        {'项目': '当前状态', '取值': '仅完成抽样，人工确认列待填写；未生成任何准确率指标'},
    ])
    with pd.ExcelWriter(SAMPLE_OUT) as writer:
        table.to_excel(writer, sheet_name='01_标注表', index=False)
        stratum.to_excel(writer, sheet_name='02_分层构成', index=False)
        notes.to_excel(writer, sheet_name='03_抽样说明', index=False)
    print('标注表写出:', SAMPLE_OUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())

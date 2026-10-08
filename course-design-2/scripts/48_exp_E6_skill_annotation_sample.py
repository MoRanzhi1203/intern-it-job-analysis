# -*- coding: utf-8 -*-
"""E6：技能抽取人工核验——分层抽样表与标注规范（未标注前不生成 Precision / Recall / F1）。

输出：
    data/interim/skill_annotation_sample.xlsx   供人工填写的标注表（含抽样说明与字段字典）
    docs/skill_annotation_guideline.md  标注规范

原则：
    1. 只从「有可解析岗位描述」的岗位中按岗位大类分层随机抽取；
    2. 规则抽取结果原样给出，供人工逐条确认，不由脚本改写；
    3. 人工确认字段留空，未完成标注前不计算任何准确率指标；
    4. 标注完成后以人工确认技能为金标准，分别评估要求段主口径、扩展口径、
       具体技术技能层级与较粗层级，并报告微平均与宏平均。

用法：
    python scripts\\48_exp_E6_skill_annotation_sample.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src import schema  # noqa: E402

CORPUS = PROJECT / 'data' / 'interim' / 'job_text_version_corpus.parquet'
MEMBERSHIP = PROJECT / 'data' / 'features' / 'job_skill_membership.parquet'
CATEGORY = PROJECT / 'data' / 'processed' / 'job_category_membership.parquet'
SAMPLE_OUT = PROJECT / 'data' / 'interim' / 'skill_annotation_sample.xlsx'
GUIDE_OUT = PROJECT / 'docs' / 'skill_annotation_guideline.md'
SAMPLE_SIZE = 240
MIN_PER_STRATUM = 5
SEED = 42
TRUNCATE = 600

GUIDE = """# 技能抽取人工核验标注规范

对应论文 6.4 节，用于为技能抽取口径提供直接准确性证据。

## 1. 目的与现状

技能需求统计、技能薪资关联、技能消融与技能 SHAP 均建立在「技能词典 + 匹配规则」之上，
因此需要人工核验作为金标准。**当前进度：已完成抽样方案与标注表，尚未形成金标准评估。**
在人工标注完成之前，本文不报告 Precision / Recall / F1，也不以任何估计值代替。

## 2. 抽样方案

| 项目 | 设定 |
| --- | --- |
| 抽样总体 | 有可解析岗位描述文本、且已识别最终核心版本的岗位 |
| 分层变量 | 岗位大类（平台分类，取该岗位命中次数最多的一类） |
| 抽样方式 | 分层随机抽样，固定随机种子 {seed} |
| 目标样本量 | {size} 条（每层至少 {min_per} 条） |
| 抽样框规模 | 见标注表「抽样说明」工作表 |

## 3. 标注表字段

| 字段 | 含义 | 填写方式 |
| --- | --- | --- |
| `intern_id` | 岗位标识，用于回查原始页面 | 只读 |
| `岗位大类` / `岗位细分类` | 平台岗位分类 | 只读 |
| `要求段原文（节选）` | 任职要求 / 岗位要求 / 任职资格段落的实际文本 | 只读 |
| `全文兜底原文（节选）` | 主口径未命中时使用的完整岗位描述文本 | 只读 |
| `规则抽取_要求段口径` | 主口径（REQUIREMENT_SECTION）抽取到的规范技能，顿号分隔 | 只读 |
| `规则抽取_扩展口径` | 扩展口径（要求段 ∪ 全文兜底）抽取到的规范技能，顿号分隔 | 只读 |
| `人工确认技能` | **金标准**：阅读原文后确认企业明确要求的技能，顿号分隔 | 人工填写 |
| `是否可判定` | 文本可读且能判断技能要求填「是」，否则填「否」 | 人工填写 |
| `备注` | 歧义、漏检、误检等说明 | 人工填写 |

## 4. 判定规则

1. **以文本为准**：只根据标注表提供的原文判断，不查阅岗位页面、不参考外部信息；
2. **「明确要求」才算**：出现在任职要求类段落、或全文明确要求掌握的技能计入；
   仅在岗位职责、公司介绍、产品描述中被提及的技术名词不计入；
3. **同义归一**：把同义表述归并到规范技能名（如 `py`、`Python3` 统一为 `Python`）；
4. **不可判定**：文本被截断、以图片为主、或语义含混无法判断时，`是否可判定` 填「否」，
   该行不进入金标准评估，但保留在表中，并在结果中报告不可判定行数；
5. **不得修改规则抽取列**：人工只填自己确认的结果，便于后续直接对比。

## 5. 标注完成后的评估口径

以 `人工确认技能` 为金标准，逐行计算 TP / FP / FN：

```text
Precision = TP / (TP + FP)     Recall = TP / (TP + FN)     F1 = 2PR / (P + R)
```

需要分别报告的四个口径：

| 口径 | 说明 |
| --- | --- |
| 要求段优先主口径 | 只统计 `规则抽取_要求段口径`，对应 REQUIREMENT_SECTION |
| 要求段加全文兜底扩展口径 | 统计 `规则抽取_扩展口径`，对应 REQUIREMENT_SECTION ∪ FULL_TEXT_FALLBACK |
| 具体技术技能层级 | 只统计技术技能层的条目 |
| 较粗层级（技术领域 / 业务能力 / 办公工具） | 只统计粗层级条目 |

同时报告**微平均**（先汇总全部 TP / FP / FN 再算）与**宏平均**（先按技能逐项算再平均），
并给出典型漏检、误检案例各不少于 5 条。

## 6. 论文中的表述

- 标注完成前（当前状态）：在 6.4 节与 9.2 节写明「当前仅完成抽样方案，尚未形成金标准评估」；
- 标注完成后：在 6.4 节报告上表四个口径的 P / R / F1 与典型案例，
  并说明该结果只反映词典与规则在抽样范围内的表现，不外推为对全部岗位的准确率。
"""


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
        '全文兜底原文（节选）': [str(value)[:TRUNCATE] for value in sample['岗位描述_原始']],
        '规则抽取_要求段口径': sample['_要求段'],
        '规则抽取_扩展口径': sample['_扩展'],
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
        {'项目': '当前状态', '取值': '仅完成抽样与规范，人工确认列待填写；未生成任何准确率指标'},
    ])
    with pd.ExcelWriter(SAMPLE_OUT) as writer:
        table.to_excel(writer, sheet_name='01_标注表', index=False)
        stratum.to_excel(writer, sheet_name='02_分层构成', index=False)
        notes.to_excel(writer, sheet_name='03_抽样说明', index=False)
    print('标注表写出:', SAMPLE_OUT)

    GUIDE_OUT.parent.mkdir(parents=True, exist_ok=True)
    GUIDE_OUT.write_text(GUIDE.format(seed=SEED, size=SAMPLE_SIZE, min_per=MIN_PER_STRATUM),
                         encoding='utf-8')
    print('标注规范写出:', GUIDE_OUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())

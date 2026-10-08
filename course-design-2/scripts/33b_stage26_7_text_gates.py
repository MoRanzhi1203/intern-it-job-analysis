# -*- coding: utf-8 -*-
"""Stage26.7：文本内容终审稿合并 + G01~G34 门禁核查。

- 合并 docs/paper/stage23/*.md 与参考文献 → docs/paper/课程设计论文_文本内容终审稿_Stage26.7.md；
- 对 G01~G14（文本终审）、G15~G25（业务时间专题）、G26~G34（技术时间与正式结果保护）
  逐项做机器判定，写入 outputs/logs/metrics/stage_26_7_text_and_time.json；
- 不重训模型、不改动 data/raw 与 outputs/models（与 Stage26.6 保护快照比对 SHA-256）。

用法：python scripts/33b_stage26_7_text_gates.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / 'docs' / 'paper' / 'stage23'
REFS = ROOT / 'docs' / 'paper' / '参考文献_Stage26.2.md'
TARGET = ROOT / 'docs' / 'paper' / '课程设计论文_文本内容终审稿_Stage26.7.md'
METRICS_DIR = ROOT / 'outputs' / 'logs' / 'metrics'
TIME_METRICS = METRICS_DIR / 'stage_26_7_time_and_audit.json'
BASELINE = METRICS_DIR / 'stage_26_6_integrity_baseline.json'

TITLE = '基于实习僧平台的互联网 IT 实习岗位薪资影响因素分析与预测'
CHAPTERS = ['00_摘要与Abstract.md', '01_绪论.md', '02_相关理论与分析方法.md',
            '03_数据获取与预处理.md', '04_互联网IT实习岗位特征分析.md',
            '05_实习岗位薪资影响因素分析.md', '06_实习岗位技能需求分析.md',
            '07_薪资预测模型构建与结果分析.md', '08_模型稳健性与解释.md',
            '09_总结与展望.md']
HEADER = [
    '<!-- 文本内容终审稿（内部构建信息，Word 转换时剥离） -->',
    '<!-- FINAL_WORD_APPENDIX_POLICY = NONE（最终 Word 不含附录） -->',
    '<!-- Stage26.7：文本内容终审（14 项）+ 新增 4.6 业务时间维度专题 + 技术时间字段排除 -->',
    '<!-- 分章来源：docs/paper/stage23/00_摘要与Abstract.md ~ 09_总结与展望.md -->',
    '<!-- 参考文献来源：docs/paper/参考文献_Stage26.2.md「一、论文用文献表」（30 条） -->',
]

ANCHORS = ['172,063', '17,144', '14,899', '14,883', '2,245', '175', '192.87', '90',
           '0.1975', '4,328', '183', '114', '731', '784', '8,822', '16,378', '766',
           '469', '46', '144', '288', '290', '56',
           '35.48', '64.81', '0.582', '51.98', '0.136']
FORBIDDEN_OLD = ['35.335506', '64.623479', '0.584606']
TIME_FORBIDDEN = ['ARIMA', 'SARIMA', 'Prophet', 'LSTM', 'GRU']
CAUSAL_FORBIDDEN = ['时间导致薪资上涨', '导致薪资上涨', '具有薪资提升效应',
                    '某月份具有薪资提升效应']


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def promote(line: str) -> str:
    match = re.match(r'^(#{2,5})\s+(.*)$', line.strip())
    if match:
        return '#' * (len(match.group(1)) - 1) + ' ' + match.group(2)
    return line.rstrip()


def strip_comments(text: str) -> str:
    return re.sub(r'<!--.*?-->', '', text, flags=re.S)


def build_manuscript() -> tuple:
    text = REFS.read_text(encoding='utf-8')
    part = text.split('## 一、论文用文献表')[1].split('## 二、')[0]
    references = [line.strip() for line in part.splitlines()
                  if re.match(r'^\[\d+\]', line.strip())]
    blocks = ['\n'.join(HEADER), '', '# ' + TITLE, '']
    for name in CHAPTERS:
        raw = (SOURCE_DIR / name).read_text(encoding='utf-8')
        promoted = '\n'.join(promote(line) for line in raw.splitlines()).strip('\n')
        blocks.append(promoted if promoted.startswith('# ') else
                      '# ' + name.rstrip('.md') + '\n\n' + promoted)
    blocks.append('# 参考文献')
    blocks.append('\n\n'.join(references))
    TARGET.write_text('\n\n'.join(blocks) + '\n', encoding='utf-8')
    return TARGET.read_text(encoding='utf-8'), len(references)


def main() -> int:
    full, reference_count = build_manuscript()
    visible = strip_comments(full)
    body = '\n'.join(line for line in visible.splitlines()
                     if not line.strip().startswith(('>', '|', '```')))

    time_metrics = json.loads(TIME_METRICS.read_text(encoding='utf-8'))
    tech = time_metrics['技术时间审计']
    time_topic = time_metrics['业务时间专题']

    baseline = json.loads(BASELINE.read_text(encoding='utf-8'))
    baseline_map = {item['文件']: item['sha256'] for item in baseline['明细']}

    protected_drift = []
    for folder in ('data/raw', 'outputs/models'):
        for path in sorted((ROOT / folder).rglob('*')):
            if not path.is_file():
                continue
            rel = str(path.relative_to(ROOT)).replace('\\', '/')
            if rel in baseline_map and sha256_of(path) != baseline_map[rel]:
                protected_drift.append(rel)

    abc_bad = [m.group(0) for m in re.finditer(r'A\+B\+\s*(?!C)', visible)] if False else \
        [1 for m in re.finditer(r'A\+B\+', visible)
         if not re.match(r'A\+B\+[CDE]', visible[m.start():m.start() + 12])]

    gates = {
        'G01_RESEARCH_QUESTION_MATCHES_MODEL_FEATURES':
            '给定岗位基础属性、地域、企业属性、技能要求与岗位文本信息' in visible
            and '文本与招聘生命周期信息的情况下' not in visible,
        'G02_EMPTY_TEXT_766_SEMANTICS_CORRECT':
            'EMPTY_TEXT' in visible and '无可用岗位描述文本' in visible
            and '在可解析文本中未识别到技能' not in visible,
        'G03_GROUP_SPLIT_WORDING_CORRECT':
            '不利用测试集结果进行模型选择' in visible
            and '划分过程也未使用薪资以外的任何信息' not in visible,
        'G04_ABSTRACT_SKILL_SCOPE_CORRECT':
            '在具体技术技能中，Python 与 SQL 的岗位覆盖最广' in visible
            and 'among concrete technical skills, Python and SQL' in visible,
        'G05_EFFECT_SIZE_CROSS_METRIC_OVERCLAIM_ZERO':
            '岗位职能方向与薪资分布的关联最清晰' not in visible
            and '公司侧以公司认证的关联强度最高' not in visible,
        'G06_TABLE5_5_SKILL_RANGE_CORRECT':
            '代表性高效应技能的中位数差可达 100～150 元/天' in visible
            and '有技能与无技能的中位数差 100～150 元/天' not in visible,
        'G07_INDUSTRY_EVIDENCE_NOT_BORROWED_FROM_COMPANY_SHAP':
            '公司侧特征占 SHAP 主要贡献特征多数' not in visible
            and '中位数回归中行业变量已纳入联合控制' in visible,
        'G08_SHAP_EXPECTED_VALUE_DEFINITION_CORRECT':
            'TreeSHAP 的基准值（expected value）' in visible
            and '即全部解释样本上的平均预测值' not in visible,
        'G09_SHAP_SEED_REFERENCE_CLEAR': '以随机种子 42 的主模型为基准' in visible,
        'G10_T2_OVERCLAIM_ZERO':
            '与真实招聘行为不符' not in visible
            and '极可能是平台发布时间或截止日期被重置造成的伪影' not in visible,
        'G11_WELFARE_PROXY_WORDING_CORRECT':
            '代理信号' in visible
            and '模型只需其中一个标签即可识别该群体' not in visible,
        'G12_INTRO_OVERCLAIM_ZERO':
            '主要通道' not in visible and '判断岗位技术含量与培养价值的直接线索' not in visible,
        'G13_ABC_DIMENSION_TEXT_VALID': len(abc_bad) == 0,
        'G14_ALT_SPLIT_DIMENSION_EXPLANATION_PRESENT':
            '各替代划分均在对应训练子集内重新拟合预处理流程' in visible,

        'G15_BUSINESS_TIME_SECTION_PRESENT':
            '4.6 业务时间维度下的岗位与薪资变化特征' in visible
            and '4.7 本章小结' in visible and '### 4.6 本章小结' not in visible,
        'G16_CRAWL_TIME_NOT_USED_AS_TIME_AXIS':
            '主时间轴是平台的岗位发布时间' in visible,
        'G17_PUBLISH_TIME_IS_PRIMARY_TIME_AXIS':
            '主时间轴是平台的岗位发布时间' in visible,
        'G18_COHORT_COUNT_NOT_CALLED_MARKET_TOTAL':
            '不代表对应月份的全市场实际新增岗位量' in visible,
        'G19_SALARY_TIME_MEDIAN_AND_IQR_PRESENT':
            'P25、P75 与四分位距' in visible or 'P25' in visible and '四分位距' in visible,
        'G20_TIME_WINDOW_SAMPLE_SIZE_REPORTED':
            'n = 10,413' in visible and 'n < 30 的窗口仅展示，不用于趋势解释' in visible,
        'G21_SMALL_N_WINDOWS_FLAGGED':
            'n < 30 的窗口仅展示，不用于趋势解释' in visible
            and tech['TECH_TIME_DERIVED_FEATURE_COUNT'] == 0,
        'G22_TIME_ANALYSIS_UNADJUSTED_BOUNDARY_PRESENT':
            '不控制公司、城市、岗位类别等结构差异' in visible,
        'G23_NO_CAUSAL_TIME_LANGUAGE':
            all(word not in visible for word in CAUSAL_FORBIDDEN)
            and '不能解释为发布时间本身对薪资的独立作用' in visible,
        'G24_NO_ARIMA_PROPHET_LSTM':
            all(word not in visible for word in TIME_FORBIDDEN),
        'G25_STAGE8_TIME_FEATURE_LINK_PRESENT':
            '第 4 章的发布时间队列分析刻画的是未经结构调整的样本时间分布' in visible,

        'G26_RAW_SOURCE_UNCHANGED': not any(p.startswith('data/raw') for p in protected_drift),
        'G27_CRAWL_TIME_ANALYSIS_ZERO':
            tech['CRAWL_TIME_IN_ANALYSIS_DATASET'] == 0,
        'G28_CRAWL_TIME_MODEL_ZERO': tech['CRAWL_TIME_IN_MODEL_DATASET'] == 0,
        'G29_CREATE_TIME_ANALYSIS_ZERO':
            tech['CREATE_TIME_IN_ANALYSIS_DATASET'] == 0,
        'G30_UPDATE_TIME_ANALYSIS_ZERO':
            tech['UPDATE_TIME_IN_ANALYSIS_DATASET'] == 0,
        'G31_TECH_TIME_DERIVED_FEATURE_ZERO':
            tech['TECH_TIME_DERIVED_FEATURE_COUNT'] == 0,
        'G32_BUSINESS_TIME_FIELDS_AUDITED':
            (ROOT / 'outputs' / 'tables' / '33_business_time_dimension_analysis.xlsx').is_file()
            and len(time_topic['图件']) >= 2,
        'G33_NO_MODEL_RETRAIN':
            not any(p.startswith('outputs/models') for p in protected_drift),
        'G34_NO_FORMAL_RESULT_DRIFT':
            all(anchor in visible for anchor in ANCHORS)
            and all(old not in visible for old in FORBIDDEN_OLD),
    }

    failed = [key for key, value in gates.items() if not value]
    payload = {
        '生成时间': __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
        '文本终审稿': str(TARGET.relative_to(ROOT)).replace('\\', '/'),
        '参考文献条数': reference_count,
        '可见正文汉字数': len(re.findall(r'[\u4e00-\u9fff]', body)),
        '门禁': gates,
        '未通过门禁': failed,
        '受保护文件漂移': protected_drift,
        'A+B+ 缺 C 命中数（可见正文）': len(abc_bad),
        '正式结果锚点缺失': [a for a in ANCHORS if a not in visible],
        'STAGE26_7_TEXT_CONTENT_READY': not [k for k in failed if k.startswith('G0') or k.startswith('G1')],
        'BUSINESS_TIME_ANALYSIS_READY': all(gates[k] for k in gates if k.startswith('G1') or k.startswith('G2')),
        'TECHNICAL_TIME_FIELDS_EXCLUDED': all(
            gates[k] for k in ('G26_RAW_SOURCE_UNCHANGED', 'G27_CRAWL_TIME_ANALYSIS_ZERO',
                               'G28_CRAWL_TIME_MODEL_ZERO', 'G29_CREATE_TIME_ANALYSIS_ZERO',
                               'G30_UPDATE_TIME_ANALYSIS_ZERO',
                               'G31_TECH_TIME_DERIVED_FEATURE_ZERO', 'G33_NO_MODEL_RETRAIN')),
    }
    (METRICS_DIR / 'stage_26_7_text_and_time.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    print('=' * 88)
    print('Stage26.7 文本终审与门禁')
    print('文本终审稿：', payload['文本终审稿'], '（汉字 %d）' % payload['可见正文汉字数'])
    print('参考文献：', reference_count, '条')
    print('门禁通过：%d / %d' % (len(gates) - len(failed), len(gates)))
    if failed:
        print('未通过：', failed)
    print('受保护文件漂移：', protected_drift or '无')
    print('A+B+ 缺 C（可见正文）：', len(abc_bad))
    print('锚点缺失：', payload['正式结果锚点缺失'] or '无')
    print('STAGE26_7_TEXT_CONTENT_READY =', payload['STAGE26_7_TEXT_CONTENT_READY'])
    print('BUSINESS_TIME_ANALYSIS_READY =', payload['BUSINESS_TIME_ANALYSIS_READY'])
    print('TECHNICAL_TIME_FIELDS_EXCLUDED =', payload['TECHNICAL_TIME_FIELDS_EXCLUDED'])
    print('=' * 88)
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())

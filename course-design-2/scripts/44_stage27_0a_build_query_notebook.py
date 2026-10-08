# -*- coding: utf-8 -*-
"""由 scripts/43_*.py 生成并执行 Jupyter 查询 Notebook（含 pandas 输出）。

产物：notebooks/27_0a_data_screenshot_queries.ipynb
特点：查询逻辑与 scripts/43_stage27_0a_data_screenshot_cells.py 同源（解析 `# %%` 单元），
     执行后把 DataFrame 的 pandas 渲染结果固化进 notebook，便于直接截图。

用法：
    E:\\anaconda3\\envs\\reptile\\python.exe scripts\\44_stage27_0a_build_query_notebook.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'scripts' / '43_stage27_0a_data_screenshot_cells.py'
NB_OUT = ROOT / 'notebooks' / '27_0a_data_screenshot_queries.ipynb'

# 每个查询单元对应的图号、图题、数据源、筛选规则、输出规模
UNIT_META = {
    2: ('图 3-7', '原始岗位观测数据的岗位与薪资字段片段',
        'data/raw/shixiseng_job_details.parquet',
        '岗位名 ≤16 字 且 城市 ≤4 字 且 实习时长 ≤8 字 → 按 intern_id 稳定排序 → 等距取 6 行',
        '6 行 × 5 列'),
    3: ('图 3-8', '原始岗位观测数据的业务时间与公司字段片段',
        'data/raw/shixiseng_job_details.parquet',
        '复用单元 2 的 e04（同一批 6 行，保证两图记录顺序一致）',
        '6 行 × 6 列'),
    4: ('图 3-10', '公司属性语义槽位异常修复前后数据片段',
        'outputs/tables/22_company_attribute_semantic_anomaly_audit.xlsx（Sheet 03_全部候选行）',
        'FULL_SHIFT / NATURE_ONLY / SIZE_ONLY 三类各取首行',
        '3 行 × 8 列'),
    5: ('图 3-12', '薪资字段解析前后数据片段',
        'data/processed/job_salary_targets.parquet',
        '取 120-150/天、100/天、薪资面议、含"上下限倒置"各一行',
        '4 行 × 7 列'),
    6: ('图 3-14', '岗位描述要求段提取示例',
        'data/interim/job_text_version_corpus.parquet',
        '实习岗位ID == inn_4fezolxa9tsh 且 是否最终核心版本 == 1',
        '3 个文本块'),
    7: ('图 3-15', '要求段技能规范化结果片段',
        'data/features/job_skill_membership.parquet',
        '同一岗位 且 match_scope == REQUIREMENT_SECTION，按技能族 / 技能名排序',
        '10 行 × 4 列'),
    8: ('图 3-16', '最终建模数据集岗位与地域字段片段',
        'data/processed/job_salary_model_dataset.parquet',
        '固定索引 7 / 1500 / 3600 / 6200 / 9500 / 13200',
        '6 行 × 7 列'),
    9: ('图 3-17', '最终建模数据集公司属性字段片段',
        'job_salary_model_dataset + job_details_unique（公司认证）',
        '复用单元 8 的 6 行（保证与图 3-16 记录顺序一致）',
        '6 行 × 6 列'),
    10: ('图 3-18', '最终建模数据集技能与文本字段片段',
        'job_salary_model_dataset + job_skill_membership（技能指示）',
        '复用单元 8 的 6 行（保证与图 3-16 记录顺序一致）',
        '6 行 × 5 列'),
}

INTRO = """# Stage27.0A 数据集截图 —— Jupyter 查询与 pandas 输出

本 Notebook 直接产出论文中 9 张数据集截图（`outputs/figures/evidence_native/03_data/*.png`）对应的查询结果。
每个查询块上方为图号、图题、数据源与筛选规则；**截图时只截下方的输出区**。

约定：

1. 全部为只读查询，不写入任何文件、不修改冻结数据；
2. 输出为**真实数据**：公司名称、实习岗位 ID 均按冻结数据原值展示，不做任何代号替换或掩码；
3. 单元 3 依赖单元 2、单元 10 依赖单元 8，用于保证两图记录顺序一致；
4. 输出由本环境实际执行生成，未做任何人工修饰。

| 单元 | 图号 | 图题 | 输出规模 |
| --- | --- | --- | --- |
| 2 | 图 3-7 | 原始岗位观测数据的岗位与薪资字段片段 | 6 行 × 5 列 |
| 3 | 图 3-8 | 原始岗位观测数据的业务时间与公司字段片段 | 6 行 × 6 列 |
| 4 | 图 3-10 | 公司属性语义槽位异常修复前后数据片段 | 3 行 × 8 列 |
| 5 | 图 3-12 | 薪资字段解析前后数据片段 | 4 行 × 7 列 |
| 6 | 图 3-14 | 岗位描述要求段提取示例 | 3 个文本块 |
| 7 | 图 3-15 | 要求段技能规范化结果片段 | 10 行 × 4 列 |
| 8 | 图 3-16 | 最终建模数据集岗位与地域字段片段 | 6 行 × 7 列 |
| 9 | 图 3-17 | 最终建模数据集公司属性字段片段 | 6 行 × 6 列 |
| 10 | 图 3-18 | 最终建模数据集技能与文本字段片段 | 6 行 × 5 列 |

> 截图提醒：不要截 Notebook 工具栏、文件名栏与左侧文件树——其中包含本地绝对路径。
"""


def parse_cells() -> list:
    """解析 43 号脚本的 `# %%` 单元，返回 [(unit_label, code)]。"""
    text = SOURCE.read_text(encoding='utf-8')
    chunks = re.split(r'^# %% ', text, flags=re.M)[1:]
    cells = []
    for chunk in chunks:
        lines = chunk.splitlines()
        header = lines[0].strip()
        body = '\n'.join(lines[1:]).strip('\n')
        cells.append((header, body))
    return cells


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata.update({
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
        'language_info': {'name': 'python', 'pygments_lexer': 'ipython3'},
    })
    nb.cells.append(nbf.v4.new_markdown_cell(INTRO))

    for header, code in parse_cells():
        m = re.match(r'\[单元 (\d+)\]\s*(.*)$', header)
        unit = int(m.group(1)) if m else None
        title = m.group(2) if m else header
        if unit in UNIT_META:
            fig, fig_title, source, rule, size = UNIT_META[unit]
            nb.cells.append(nbf.v4.new_markdown_cell(
                '### %s %s\n\n'
                '- **数据源**：`%s`\n'
                '- **筛选 / 抽样**：%s\n'
                '- **输出规模**：%s\n'
                '- **截图**：只截下方输出区，不要截本说明块与 Notebook 工具栏\n'
                % (fig, fig_title, source, rule, size)))
        else:
            nb.cells.append(nbf.v4.new_markdown_cell('### %s' % title))
        nb.cells.append(nbf.v4.new_code_cell(code, metadata={'tags': []}))
    return nb


def execute(nb: nbf.NotebookNode):
    from jupyter_client import KernelManager
    from nbclient import NotebookClient

    class ScopedKernelManager(KernelManager):
        """强制用当前解释器启动 kernel，避免命中系统默认 python3。"""

        def kernel_cmd(self):
            return [sys.executable, '-m', 'ipykernel_launcher',
                    '-f', '{connection_file}']

    client = NotebookClient(
        nb, timeout=900, kernel_name='python3',
        kernel_manager_class=ScopedKernelManager,
        resources={'metadata': {'path': str(ROOT)}})
    client.execute()
    return nb


def main() -> int:
    nb = build_notebook()
    print('单元数:', len(nb.cells))
    execute(nb)
    NB_OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(NB_OUT))
    print('Notebook 写出:', NB_OUT, NB_OUT.stat().st_size, 'bytes')

    print('=' * 78)
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        head = cell.source.splitlines()[0]
        kinds = [out.get('output_type') for out in cell.get('outputs', [])]
        print('  %-58s %s' % (head[:58], kinds))
    errors = [c for c in nb.cells if c.cell_type == 'code'
              and any(o.get('output_type') == 'error' for o in c.get('outputs', []))]
    print('执行报错单元数:', len(errors))
    return 0


if __name__ == '__main__':
    sys.exit(main())

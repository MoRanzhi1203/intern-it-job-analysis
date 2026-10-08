# -*- coding: utf-8 -*-
"""IO 工具模块：Parquet / Excel / JSON / Markdown 的安全读写。

约定：
- Parquet 写入后必须回读校验（行数、列名）；
- Excel 写入统一做单元格长度保护，超长文本截断并标记；
- Markdown / JSON 记录统一 UTF-8 写出。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

EXCEL_CELL_LIMIT = 32000


def read_parquet(path: Path, columns=None) -> pd.DataFrame:
    """安全读取 Parquet：文件必须存在且非空。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Parquet 文件不存在: {path}')
    frame = pd.read_parquet(path, columns=columns)
    if frame.empty:
        raise ValueError(f'Parquet 文件为空: {path}')
    LOGGER.info('读取 %s: %s 行 × %s 列', path.name, len(frame), frame.shape[1])
    return frame


def write_parquet(frame: pd.DataFrame, path: Path, verify: bool = True) -> Path:
    """写入 Parquet 并（默认）回读校验行数与列名。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    if verify:
        reloaded = pd.read_parquet(path)
        if len(reloaded) != len(frame):
            raise ValueError(f'回读行数不一致: {len(reloaded)} != {len(frame)}（{path.name}）')
        if list(reloaded.columns) != list(frame.columns):
            raise ValueError(f'回读列名不一致（{path.name}）')
    LOGGER.info('写入 %s: %s 行 × %s 列, %.2f MB',
                path.name, len(frame), frame.shape[1], path.stat().st_size / 1024 / 1024)
    return path


def excel_safe(frame: pd.DataFrame, limit: int = EXCEL_CELL_LIMIT) -> pd.DataFrame:
    """Excel 单元格长度保护：list 转文本、超长文本截断、datetime 转字符串。"""
    safe = frame.copy()
    for column in safe.columns:
        dtype_text = str(safe[column].dtype)
        if safe[column].dtype == object:
            safe[column] = safe[column].map(lambda value: _excel_cell(value, limit))
        elif dtype_text.startswith('datetime'):
            safe[column] = safe[column].astype(str)
    return safe


def _sheet_sort_key(name: str):
    """Sheet 排序键：优先按名称前的数字编号排序，其余保持稳定顺序。"""
    prefix = str(name).split('_', 1)[0]
    return (0, int(prefix), str(name)) if prefix.isdigit() else (1, 0, str(name))


def _excel_cell(value, limit: int):
    """单个单元格值归一化：list/tuple/set 渲染为顿号分隔文本，超长文本截断。"""
    if isinstance(value, (list, tuple, set, frozenset)):
        return '、'.join(str(item) for item in value)
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + '...[已截断]'
    return value


def write_excel(path: Path, sheets: dict, limit: int = EXCEL_CELL_LIMIT,
                carry_over_sheets=()) -> Path:
    """写入多 Sheet Excel；每个 Sheet 值必须是 DataFrame。

    carry_over_sheets：本次不重算、但必须保留在同一工作簿中的既有 Sheet
    （用于把一次性取证结果与流水线每次重算的结果合并到同一张审计表）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = dict(sheets)
    if carry_over_sheets and path.exists():
        existing = pd.ExcelFile(path).sheet_names
        for name in carry_over_sheets:
            if name in merged or name not in existing:
                continue
            merged[name] = pd.read_excel(path, sheet_name=name)
        # 合并后按 Sheet 名前的编号排序，保证人工查看顺序稳定（仅在有结转 Sheet 时启用）
        merged = dict(sorted(merged.items(), key=lambda item: _sheet_sort_key(item[0])))
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for sheet_name, frame in merged.items():
            if not isinstance(frame, pd.DataFrame):
                frame = pd.DataFrame(frame)
            excel_safe(frame, limit=limit).to_excel(
                writer, sheet_name=str(sheet_name)[:31], index=False)
    LOGGER.info('写入统计表 %s（%s 个 Sheet, %.1f KB）',
                path.name, len(merged), path.stat().st_size / 1024)
    return path


def write_json(path: Path, payload) -> Path:
    """写出 JSON（UTF-8，保留中文）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                    encoding='utf-8')
    return path


def read_json(path: Path):
    """读取 JSON，文件不存在返回 None。"""
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def write_markdown(path: Path, lines) -> Path:
    """写出 Markdown（UTF-8）；lines 可以是字符串列表或单个字符串。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = lines if isinstance(lines, str) else '\n'.join(str(line) for line in lines)
    path.write_text(text + ('' if text.endswith('\n') else '\n'), encoding='utf-8')
    LOGGER.info('写出记录 %s（%.1f KB）', path.name, path.stat().st_size / 1024)
    return path


def write_markdown_section(path: Path, marker: str, lines) -> Path:
    """幂等追加 Markdown 章节：截断到 marker 之前再追加，重复运行不会累积。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding='utf-8') if path.exists() else ''
    if marker in existing:
        existing = existing.split(marker)[0]
    prefix = existing.rstrip('\n')
    body = lines if isinstance(lines, str) else '\n'.join(str(line) for line in lines)
    text = f'{prefix}\n\n{marker}\n{body}\n' if prefix else f'{marker}\n{body}\n'
    path.write_text(text, encoding='utf-8')
    LOGGER.info('写出记录 %s（%.1f KB）', path.name, path.stat().st_size / 1024)
    return path

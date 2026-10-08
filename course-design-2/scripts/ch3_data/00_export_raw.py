# -*- coding: utf-8 -*-
"""Stage 00：原始 MySQL 数据导出（只读，不做任何语义加工）。

MySQL shixiseng_db.shixiseng_job_details
        ↓
data/raw/shixiseng_job_details.parquet（172063 行 × 30 列，永久不可修改）

职责严格限定：MySQL 只读导出 / 结构验证 / 行数验证 / 字段集合验证 / raw 文件写出。
禁止在本阶段做中文化、字段修复、空值填充、业务标准化、去重或实体合并。

用法：
    python scripts/ch3_data/00_export_raw.py --reuse-existing   # 复用已存在 raw（默认行为）
    python scripts/ch3_data/00_export_raw.py --force-export     # 强制重新从 MySQL 导出
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from src import db, io_utils, project_paths, quality, schema  # noqa: E402

STAGE = 'stage_00'
TITLE = 'Stage 00 原始 MySQL 数据导出'
LOGGER = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Stage 00：从 MySQL 导出或复用原始数据')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--reuse-existing', action='store_true',
                       help='复用已存在的 data/raw Parquet（不访问 MySQL）')
    group.add_argument('--force-export', action='store_true',
                       help='强制重新从 MySQL 导出')
    parser.add_argument('--chunk-size', type=int, default=db.DEFAULT_CHUNK_SIZE,
                        help='分块读取行数，默认 20000')
    return parser.parse_args()


def validate_raw_frame(frame: pd.DataFrame) -> None:
    """校验原始数据结构：非空、关键字段存在、映射可覆盖。"""
    if frame.empty:
        raise ValueError('原始数据为空')
    unmapped, unused = schema.validate_mapping_coverage(frame.columns)
    if unmapped:
        raise KeyError(f'存在未映射的原始字段: {unmapped}')
    if unused:
        raise KeyError(f'映射字典包含数据中不存在的字段: {unused}')
    missing_required = [c for c in schema.REQUIRED_RAW_FIELDS if c not in frame.columns]
    if missing_required:
        raise KeyError(f'缺少关键原始字段: {missing_required}')


def export_from_mysql(chunk_size: int) -> tuple:
    """从 MySQL 分块导出到 data/raw Parquet，返回（行数, 列名）。"""
    writer = None
    total_rows = 0
    columns = None
    print(f'从 MySQL 导出: {db.SCHEMA_NAME}.{db.TABLE_NAME}（{db.SELECT_ALL_SQL}）')
    for chunk in db.iter_raw_table_chunks(chunk_size=chunk_size):
        if columns is None:
            columns = list(chunk.columns)
            print(f'源表列数: {len(columns)}')
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(project_paths.RAW_PARQUET), table.schema,
                                      compression='snappy')
        writer.write_table(table)
        total_rows += len(chunk)
        print(f'  已写入 {total_rows} 行', flush=True)
    if writer is not None:
        writer.close()
    if total_rows == 0:
        raise ValueError('MySQL 导出结果为空')
    return total_rows, columns


def load_or_export(reuse_existing: bool, force_export: bool, chunk_size: int) -> tuple:
    """按模式取得原始数据，返回 (df_raw, source_mode)。"""
    raw_exists = project_paths.RAW_PARQUET.exists()
    if force_export or (not reuse_existing and not raw_exists):
        rows, columns = export_from_mysql(chunk_size)
        print(f'MySQL 导出完成: {rows} 行 × {len(columns)} 列 → {project_paths.relative_to_root(project_paths.RAW_PARQUET)}')
        source_mode = 'mysql_export'
    else:
        if not raw_exists:
            raise FileNotFoundError(f'原始数据不存在且未指定导出: {project_paths.RAW_PARQUET}')
        print(f'复用已存在原始数据: {project_paths.relative_to_root(project_paths.RAW_PARQUET)}（未访问 MySQL）')
        source_mode = 'reuse_existing'

    frame = io_utils.read_parquet(project_paths.RAW_PARQUET)
    with pd.option_context('display.width', 200):
        print(f'原始数据规模: {len(frame)} 行 × {frame.shape[1]} 列')
        print(f'内存占用: {round(frame.memory_usage(deep=True).sum() / 1024 / 1024, 1)} MB')
    validate_raw_frame(frame)
    return frame, source_mode


def write_record(frame: pd.DataFrame, source_mode: str) -> Path:
    """写出 Stage 00 阶段记录。"""
    quality_table = quality.field_quality_table(frame)
    mode_text = {
        'mysql_export': f'重新从 MySQL 导出（{db.SCHEMA_NAME}.{db.TABLE_NAME}）',
        'reuse_existing': '复用已存在的 data/raw Parquet（未访问 MySQL）',
    }[source_mode]
    missing_fields = quality_table.loc[quality_table['缺失数量'] > 0, '字段名'].tolist()

    lines = [
        '# 阶段 00 记录：原始数据导出与 Schema 审计',
        '',
        '> 本文件由 `scripts/ch3_data/00_export_raw.py` 自动生成，数字均来自真实运行结果。',
        '',
        '## 1. 数据来源',
        '',
        '| 项 | 内容 |',
        '| --- | --- |',
        f'| 数据源 | MySQL `{db.SCHEMA_NAME}.{db.TABLE_NAME}` |',
        f'| 读取语句 | `{db.SELECT_ALL_SQL}` |',
        f'| 本次模式 | {mode_text} |',
        f'| 输出文件 | `{project_paths.relative_to_root(project_paths.RAW_PARQUET)}` |',
        '',
        '## 2. 数据规模',
        '',
        '| 指标 | 数值 |',
        '| --- | --- |',
        f'| 行数 | {len(frame)} |',
        f'| 列数 | {frame.shape[1]} |',
        f'| 内存占用 | {round(frame.memory_usage(deep=True).sum() / 1024 / 1024, 1)} MB |',
        f'| 文件大小 | {round(project_paths.RAW_PARQUET.stat().st_size / 1024 / 1024, 2)} MB |',
        '',
        '## 3. 列名与 dtype',
        '',
        '| 序号 | 列名 | dtype | 非空数量 | 缺失数量 | 缺失率 | 唯一值数量 |',
        '| --- | --- | --- | --- | --- | --- | --- |',
        *[f'| {row.序号} | {row.字段名} | {row.数据类型} | {row.非空数量} | '
          f'{row.缺失数量} | {row.缺失率} | {row.唯一值数量} |'
          for row in quality_table.itertuples(index=False)],
        '',
        '## 4. 缺失情况',
        '',
        f'- 存在缺失的字段数：{len(missing_fields)}',
        f'- 存在缺失的字段：{"、".join(missing_fields) if missing_fields else "无"}',
        '',
        '## 5. 本阶段边界',
        '',
        '- 使用 `SELECT *`，未删除任何列、未修改字段名；',
        '- 未去重、未清洗、未做特征构造；',
        '- 未修改 MySQL 原始表、未运行爬虫。',
        '',
    ]
    return io_utils.write_markdown(project_paths.RECORDS_DIR / project_paths.RECORD_RAW_EXPORT, lines)


def main() -> int:
    global LOGGER
    args = parse_args()
    quality.configure_logging()
    quality.configure_logging()
    gates = quality.GateRegistry(STAGE)

    quality.stage_banner(STAGE, TITLE)
    frame, source_mode = load_or_export(args.reuse_existing, args.force_export, args.chunk_size)

    # 统计表
    quality_table = quality.field_quality_table(frame)
    overview = pd.DataFrame([
        {'指标': '行数', '数值': len(frame)},
        {'指标': '列数', '数值': frame.shape[1]},
        {'指标': '内存占用(MB)', '数值': round(frame.memory_usage(deep=True).sum() / 1024 / 1024, 1)},
        {'指标': '文件大小(MB)', '数值': round(project_paths.RAW_PARQUET.stat().st_size / 1024 / 1024, 2)},
        {'指标': '本次模式', '数值': source_mode},
    ])
    io_utils.write_excel(project_paths.TABLES_DIR / project_paths.TABLE_RAW_SCHEMA_AUDIT,
                         {'数据规模概览': overview, '字段Schema审计': quality_table})

    # 阶段记录
    record_path = write_record(frame, source_mode)
    print(f'阶段记录: {project_paths.relative_to_root(record_path)}')

    # 门禁：真实校验（列结构 / 非空 / 关键字段 / 映射覆盖）
    validate_raw_frame(frame)
    gates.check(
        'RAW_DATA_CHECK',
        condition=(len(frame) > 0 and frame.shape[1] == len(schema.COLUMN_NAME_CN)),
        note=f'{len(frame)} 行 × {frame.shape[1]} 列，关键字段齐全，模式={source_mode}',
    )
    gates.save()
    print(f'Stage 00 完成: {len(frame)} 行 × {frame.shape[1]} 列')
    return 0


if __name__ == '__main__':
    sys.exit(main())

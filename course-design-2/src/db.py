# -*- coding: utf-8 -*-
"""MySQL 连接模块：仅提供引擎与原始表分块读取能力。

流水线中的数据读写、路径、门禁、字段映射等公共逻辑分别位于
src/project_paths.py、src/io_utils.py、src/quality.py、src/schema.py。
"""

from __future__ import annotations

from sqlalchemy import create_engine

DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': '123456',
    'database': 'shixiseng_db',
    'charset': 'utf8',
}

SCHEMA_NAME = DB_CONFIG['database']
TABLE_NAME = 'shixiseng_job_details'
SELECT_ALL_SQL = f'SELECT * FROM {TABLE_NAME}'

DEFAULT_CHUNK_SIZE = 20000

# 分块读取时统一 dtype，保证跨分片 schema 一致
INT_COLUMNS = ['id']
FLOAT_COLUMNS = ['search_page']
DATETIME_COLUMNS = ['created_at', 'updated_at']


def create_db_engine():
    """创建 SQLAlchemy 引擎（连接 shixiseng_db）。"""
    return create_engine(
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?charset=utf8"
    )


def normalize_chunk(chunk):
    """统一原始表各列 dtype，保证分块写入 Parquet 时 schema 稳定。"""
    frame = chunk.copy()
    for column in frame.columns:
        if column in INT_COLUMNS:
            frame[column] = frame[column].astype('int64')
        elif column in FLOAT_COLUMNS:
            frame[column] = frame[column].astype('float64')
        elif column in DATETIME_COLUMNS:
            import pandas as pd
            frame[column] = pd.to_datetime(frame[column])
        else:
            frame[column] = frame[column].astype('object')
    return frame


def iter_raw_table_chunks(chunk_size: int = DEFAULT_CHUNK_SIZE):
    """分块迭代 shixiseng_job_details 原始表（SELECT *，不做任何清洗）。"""
    import pandas as pd

    engine = create_db_engine()
    for chunk in pd.read_sql(SELECT_ALL_SQL, engine, chunksize=chunk_size):
        yield normalize_chunk(chunk)

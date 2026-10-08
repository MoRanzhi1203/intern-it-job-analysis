# -*- coding: utf-8 -*-
"""图 6-2「岗位细分类与技能命中率」配图优化：移除图内文本「主口径」。

背景：正式配图源 ``outputs/figures/supplementary/图S63_岗位细分类技能命中率热力图.png``
的 x 轴标题已为「具体技术技能（命中率前 20）」，但多个 Word 产物中嵌入的是该图的旧版副本，
x 轴标题仍为「具体技术技能（主口径命中率前 20）」，即图内残留「主口径」字样。

处理：把各 Word 中嵌入的旧版热力图（按字节长度精确识别）替换为当前正式配图，
只替换图片字节，不改动文档结构、图题、正文与显示尺寸（两图像素尺寸与宽高比完全一致：
3950 × 3430，宽高比 1.1516，替换后不会被拉伸）。

用法：E:\\anaconda3\\envs\\reptile\\python.exe scripts\\64_replace_stale_heatmap_in_word.py
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / 'outputs' / 'paper'
CLEAN_SOURCE = (ROOT / 'outputs' / 'figures' / 'supplementary' /
                '图S63_岗位细分类技能命中率热力图.png')
STALE_SIZE = 874318
TARGETS = [
    '基于实习僧平台的互联网 IT 实习岗位薪资影响因素分析与预测.docx',
    '课程设计论文_摘要图表编号终版.docx',
    '课程设计论文_图表收口与结果分析强化版.docx',
    '课程设计论文_图4-6移除与三线表修复版.docx',
    '课程设计论文_图表精简修订版.docx',
]


def replace_in_docx(path: Path, payload: bytes) -> list:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        blobs = {info.filename: archive.read(info.filename) for info in infos}
    targets = [name for name in blobs
               if name.startswith('word/media') and len(blobs[name]) == STALE_SIZE]
    if not targets:
        return []
    for name in targets:
        blobs[name] = payload
    tmp = path.with_suffix('.docx.tmp')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as out:
        for info in infos:
            out.writestr(info.filename, blobs[info.filename])
    shutil.move(str(tmp), str(path))
    return targets


def main() -> int:
    payload = CLEAN_SOURCE.read_bytes()
    print('正式配图：%s（%d 字节）' % (CLEAN_SOURCE.name, len(payload)))
    total = 0
    for name in TARGETS:
        path = PAPER_DIR / name
        if not path.exists():
            print('跳过（不存在）：%s' % name)
            continue
        replaced = replace_in_docx(path, payload)
        total += len(replaced)
        print('%s：替换 %s' % (name, replaced or '无（未嵌入旧版热力图）'))
    print('完成，共替换 %d 处嵌入图片。' % total)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

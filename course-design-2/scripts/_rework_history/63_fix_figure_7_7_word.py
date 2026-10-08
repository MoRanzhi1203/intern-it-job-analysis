# -*- coding: utf-8 -*-
"""修复论文 Word 中的图 7-7 编号冲突（任务 6）。

仅做一处最小文本改写：把 7.4 节中指向主模型锁定代码截图的「如图 7-7 所示」句子
替换为纯正文（不含图号），使「图 7-7」唯一地指向 7.6 节的「薪资分组预测误差」图。
不删除图片、不改动任何其它内容、不触碰 SEQ 域。

用法：E:\\anaconda3\\envs\\reptile\\python.exe scripts\\63_fix_figure_7_7_word.py
"""
from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / 'outputs' / 'paper'

OLD_SENTENCE = '验证集模型选择与主模型锁定的实现'
NEW_TEXT = ('各候选配置在训练集上拟合，并依据验证集 MAE 确定主模型及超参数；'
            '配置锁定后，在独立测试集上完成正式评估。')

TARGETS = [
    '基于实习僧平台的互联网 IT 实习岗位薪资影响因素分析与预测.docx',
    '课程设计论文_摘要图表编号终版.docx',
]


def rewrite_paragraph(xml: str, text: str) -> tuple[str, bool]:
    paras = re.findall(r'<w:p[ >].*?</w:p>', xml, flags=re.S)
    for para in paras:
        plain = re.sub(r'<[^>]+>', '', para)
        if OLD_SENTENCE in plain and '图 7-7' in plain:
            head = re.match(r'(<w:p[ >].*?<w:pPr>.*?</w:pPr>|[^<]*<w:p[^>]*>)', para, flags=re.S)
            prefix = head.group(1) if head else para.split('>', 1)[0] + '>'
            new_para = (prefix + '<w:r><w:rPr><w:lang w:eastAsia="zh-CN"/></w:rPr>'
                        '<w:t xml:space="preserve">%s</w:t></w:r></w:p>' % text)
            return xml.replace(para, new_para, 1), True
    return xml, False


def main() -> int:
    changed = []
    for name in TARGETS:
        path = PAPER_DIR / name
        if not path.exists():
            print('跳过（不存在）：%s' % name)
            continue
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            payload = {info.filename: archive.read(info.filename) for info in infos}
            order = [info.filename for info in infos]
        xml = payload['word/document.xml'].decode('utf-8')
        if OLD_SENTENCE not in re.sub(r'<[^>]+>', '', xml):
            print('无需修改：%s' % name)
            continue
        new_xml, done = rewrite_paragraph(xml, NEW_TEXT)
        if not done:
            print('未命中目标段落：%s' % name)
            continue
        payload['word/document.xml'] = new_xml.encode('utf-8')
        tmp = path.with_suffix('.docx.tmp')
        with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as out:
            for filename in order:
                out.writestr(filename, payload[filename])
        shutil.move(str(tmp), str(path))
        changed.append((name, done))
        plain_after = re.sub(r'<[^>]+>', '', new_xml)
        print('已修改：%s；剩余命中「%s」= %d 处'
              % (name, OLD_SENTENCE, plain_after.count(OLD_SENTENCE)))
        print('  新句子在文中：', NEW_TEXT in plain_after)
    print('完成，修改文件数：%d' % len(changed))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

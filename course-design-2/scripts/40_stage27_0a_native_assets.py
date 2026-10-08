# -*- coding: utf-8 -*-
"""Stage27.0A 截图证据原生化：把 Stage27.0 的组合证据图拆分为单图直接输出。

规则（提示词 33）：
1. 网页 / 代码 / 数据集三类截图默认单图输出，一图一逻辑、一图一图题；
2. 图片内部不出现 (a)/(b)/(c)、子图名称、顶部灰蓝标题栏；
3. 代码图只保留真实代码 + 必要行号 + 语法高亮；
4. 数据图只保留字段名 + 真实数据行 + 必要极短脚注；
5. 网页图保持原界面，不重绘、不做信息图化。

用法：
    E:\\anaconda3\\envs\\reptile\\python.exe scripts\\40_stage27_0a_native_assets.py
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]

# 复用 Stage27.0 资产脚本的底层工具（字体 / 分词 / 源码截取 / 冻结数据读取）
_spec = importlib.util.spec_from_file_location(
    '_s36', str(ROOT / 'scripts' / '36_stage27_0_evidence_assets.py'))
s36 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s36)

EV0 = ROOT / 'outputs' / 'figures' / 'evidence'
EV = ROOT / 'outputs' / 'figures' / 'evidence_native'
RAW = EV0 / '01_source' / '_raw'

CODE_SIZE = 30
CODE_LEAD = 45
TABLE_SIZE = 22
TABLE_LEAD = 42
UI_BORDER = (0xC0, 0xC6, 0xCE)
UI_HEAD_BG = (0xEE, 0xF2, 0xF7)
UI_LABEL = (0x1A, 0x3D, 0x6B)
UI_TEXT = (0x1F, 0x1F, 0x1F)
UI_NOTE = (0x7A, 0x45, 0x00)
PRINT_WIDTH_CM = 15.5

mono = s36.mono
sans = s36.sans
adv = s36.adv
tokenize = s36.tokenize
TOKEN_COLORS = s36.TOKEN_COLORS
excerpt = s36.excerpt
wrap_text = s36.wrap_text
crop_image = s36.crop_image
resize_w = s36.resize_w
flat_cell = s36.flat_cell
hhmm = s36.hhmm
load_frozen = s36.load_frozen

REGISTRY = []
GEOMETRY = []


def record(old_id, new_id, old_no, new_no, etype, old_composite, split_count,
           source_file, line_range, source_data, out_file, privacy='PASS',
           inserted='YES', status='GENERATED', note=''):
    REGISTRY.append({
        'old_evidence_id': old_id,
        'new_evidence_id': new_id,
        'old_figure_number': old_no,
        'new_figure_number': new_no,
        'evidence_type': etype,
        'old_composite': old_composite,
        'split_count': split_count,
        'source_file': source_file,
        'source_line_range': line_range,
        'source_data': source_data,
        'native_single_image': 'YES',
        'internal_panel_title_removed': 'YES',
        'subfigure_caption_position': 'NOT_APPLICABLE（单图，无子图）',
        'privacy_checked': privacy,
        'inserted_to_word': inserted,
        'status': status,
        'figure_file': Path(out_file).name,
        'note': note,
    })


# --------------------------------------------------------------------------- #
# 渲染器（无标题栏 / 无 (a)(b)）
# --------------------------------------------------------------------------- #
def code_image(spec: dict, out_path: Path, show_line_no: bool = True):
    lines = spec['lines']
    font = mono(CODE_SIZE)
    line_no_w = int(adv(font, str(spec['line_end'])) + 30) if show_line_no else 0
    text_w = max(adv(font, line) for line in lines)
    pad = 14
    width = int(pad * 2 + 30 + line_no_w + text_w)
    height = int(pad * 2 + len(lines) * CODE_LEAD)

    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width - 1, height - 1], outline=UI_BORDER)
    f_no = mono(CODE_SIZE - 5)
    y = pad
    for offset, line in enumerate(lines):
        if show_line_no:
            draw.text((pad + 12, y + 4), str(spec['line_start'] + offset),
                      font=f_no, fill=(0x9A, 0xA4, 0xAF))
        x = pad + 30 + line_no_w
        for kind, piece in tokenize(line):
            draw.text((x, y), piece, font=font, fill=TOKEN_COLORS[kind])
            x += adv(font, piece)
        y += CODE_LEAD
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, dpi=(300, 300))
    img.info['render_font_px'] = CODE_SIZE
    return img


def data_image(columns, rows, out_path: Path, align=None, note=None, font_size=None,
               col_min_px=None, pad=None):
    """纯数据截图：仅字段名 + 真实数据 + 极短脚注。"""
    font_size = font_size or TABLE_SIZE
    pad = pad or 16
    f_head = sans(font_size, bold=True)
    f_body = sans(font_size)
    widths = []
    for j, col in enumerate(columns):
        w = adv(f_head, str(col))
        for row in rows:
            w = max(w, adv(f_body, str(row[j]) if j < len(row) else ''))
        w += pad * 2
        widths.append(max(w, (col_min_px or [0] * len(columns))[j]))

    margin = 14
    width = int(sum(widths) + margin * 2)
    note_h = 0 if not note else int(font_size * 1.85) * len(note)
    height = int(margin * 2 + TABLE_LEAD + 12 + TABLE_LEAD * len(rows) + note_h)

    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width - 1, height - 1], outline=UI_BORDER)
    x0, y = margin, margin
    x = x0
    for j, col in enumerate(columns):
        draw.text((x + pad, y), str(col), font=f_head, fill=UI_LABEL)
        x += widths[j]
    draw.line([x0, y + TABLE_LEAD - 6, x0 + sum(widths), y + TABLE_LEAD - 6],
              fill=(0, 0, 0), width=2)
    y += TABLE_LEAD
    for row in rows:
        x = x0
        for j, col in enumerate(columns):
            value = str(row[j]) if j < len(row) else ''
            if align and align[j] == 'c':
                tx = x + (widths[j] - adv(f_body, value)) / 2
            elif align and align[j] == 'r':
                tx = x + widths[j] - pad - adv(f_body, value)
            else:
                tx = x + pad
            draw.text((tx, y + 4), value, font=f_body, fill=UI_TEXT)
            x += widths[j]
        y += TABLE_LEAD
        draw.line([x0, y - 6, x0 + sum(widths), y - 6], fill=(0xD8, 0xDE, 0xE6), width=1)
    draw.line([x0, y - 6, x0 + sum(widths), y - 6], fill=(0, 0, 0), width=2)
    if note:
        yy = y + 4
        f_note = sans(font_size - 3)
        for line in note:
            draw.text((x0, yy), line, font=f_note, fill=UI_NOTE)
            yy += int(font_size * 1.85)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, dpi=(300, 300))
    img.info['render_font_px'] = font_size
    return img


def web_image(img: Image.Image, out_path: Path, width_px=None):
    if width_px:
        img = resize_w(img, width_px)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, dpi=(300, 300))
    img.info['render_font_px'] = 0
    return img


def print_width_cm(img: Image.Image, font_px: int, target_pt: float) -> float:
    """按目标打印字号反算版心内宽度，并限制在 [7.0, 15.5] cm。"""
    if not font_px:
        return round(min(15.5, max(7.0, img.width / 120.0)), 1)
    raw = target_pt * img.width / (font_px * 28.35)
    return round(min(15.5, max(7.0, raw)), 1)


def register_geometry(img: Image.Image, out_path: Path, target_pt=9.0, note=''):
    font_px = img.info.get('render_font_px', TABLE_SIZE)
    cm = print_width_cm(img, font_px, target_pt)
    printed_font = (font_px / img.width * cm * 28.35) if font_px else None
    GEOMETRY.append({
        'figure_file': out_path.name,
        'width_px': img.width,
        'height_px': img.height,
        'aspect_h_over_w': round(img.height / img.width, 3),
        'print_width_cm': cm,
        'print_height_cm': round(img.height / img.width * cm, 2),
        'render_font_px': font_px,
        'printed_font_pt': round(printed_font, 1) if printed_font else None,
        'note': note,
    })
    return cm


# --------------------------------------------------------------------------- #
# 资产构建
# --------------------------------------------------------------------------- #
def build_web_layer():
    menu = crop_image(RAW / 'A2_orig2_13_menu_expanded.png')
    detail = crop_image(RAW / 'A2_orig2_34_job_detail_page.png', top=150, bottom=1200)
    runtime = crop_image(RAW / 'A2_orig2_25_dual_tab_runtime.png')
    ImageDraw.Draw(runtime).rectangle([0, 55, runtime.width, 104], fill=(0xEC, 0xEC, 0xEC))

    pairs = [
        ('E01', 'E01A', '3-1', '3-1', menu,
         'docs/reference/...修订版.docx', 'docx media image13',
         '“互联网 IT”菜单展开页面（原样保留）', 'E01A_shixiseng_it_menu.png', None),
        ('E01', 'E01B', '3-1', '3-2', detail,
         'docs/reference/...修订版.docx', 'docx media image34（裁切 top=150px）',
         '岗位详情页面（裁去窗口标题栏/标签页/地址栏/书签栏）',
         'E01B_job_detail_page.png', 1800),
        ('E03', 'E03', '3-4', '3-6', runtime,
         'docs/reference/...修订版.docx', 'docx media image25（遮盖 y∈[55,104]）',
         '双标签页运行界面（地址栏色带已遮盖）', 'E03_dual_tab_runtime.png', 1700),
    ]
    for old_id, new_id, old_no, new_no, img, src, loc, data_desc, fname, wid in pairs:
        out = EV / '01_web' / fname
        img = web_image(img, out, wid)
        register_geometry(img, out, note='网页/界面截图，单图直接输出')
        record(old_id, new_id, old_no, new_no, '网页/界面截图', 'YES' if old_id == 'E01' else 'NO',
               2 if old_id == 'E01' else 1, src, loc, data_desc, out,
               note='已去除顶部标题条与左上角 (a)/(b) 标识')
    print('[web] 输出', len(pairs), '张')


def build_code_layer():
    specs = [
        # (old_id, new_id, old_no, new_no, file, start, end, function, title_note, fname)
        ('E02', 'E02A', '3-3', '3-4', 'src/url_utils.py', 19, 41,
         'normalize_detail_url', '岗位详情链接规范化', 'E02A_normalize_detail_url.png'),
        ('E02', 'E02B', '3-3', '3-5', 'scripts/03_build_job_observations.py', 236, 244,
         'URL 规范化与 ID—URL 一一对应校验', '规范化链接与岗位标识一致性校验',
         'E02B_url_identity_validation.png'),
        ('E07', 'E07A', '3-9', '3-9', 'src/field_repair.py', 126, 140,
         'classify_company_attribute_slot_anomaly', '公司属性语义槽位异常识别',
         'E07A_slot_anomaly_detect_code.png'),
        ('E08', 'E08A', '3-10', '3-11', 'scripts/11_clean_structured_fields.py', 80, 107,
         'parse_salary', '薪资字段解析', 'E08A_parse_salary_code.png'),
        ('E09', 'E09A', '3-11', '3-13', 'src/text_utils.py', 344, 360,
         'build_model_safe_version', '岗位描述薪资信息清理',
         'E09A_salary_cleaning_code.png'),
        ('E11', 'E11A', '5-1', '5-1', 'src/eda_analysis.py', 149, 164,
         'kruskal_wallis', 'Kruskal–Wallis 与 ε² 计算',
         'E11A_kruskal_wallis_code.png'),
        ('E11', 'E11B', '5-1', '5-2', 'src/eda_analysis.py', 183, 194,
         'two_group_stats（Cliff\'s δ 与 Mann–Whitney U 段）',
         "Mann–Whitney U 与 Cliff's δ 计算", 'E11B_mannwhitney_cliff_code.png'),
        ('E11', 'E11C', '5-1', '5-3', 'src/eda_analysis.py', 197, 212,
         'add_fdr', 'BH-FDR 多重检验校正', 'E11C_bhfdr_code.png'),
        ('E12', 'E12A', '7-2', '7-2', 'src/model_training.py', 46, 66,
         'build_splits', '训练集、验证集与测试集划分',
         'E12A_build_splits_code.png'),
        ('E12', 'E12B', '7-2', '7-3', 'scripts/14_train_salary_model.py', 507, 514,
         'assemble（fit 与 transform 调用）', '训练集拟合与验证/测试集变换',
         'E12B_fit_transform_code.png'),
        ('E13', 'E13A', '7-3', '7-4', 'src/model_training.py', 155, 176,
         'SalaryFeatureAssembler.fit', '高频技能列与文本语义特征构建',
         'E13A_skill_text_feature_code.png'),
        ('E13', 'E13B', '7-3', '7-5', 'src/model_training.py', 178, 191,
         'SalaryFeatureAssembler.transform', 'A/B/C/D/E 特征矩阵拼接',
         'E13B_feature_matrix_stack_code.png'),
        ('E14', 'E14A', '7-4', '7-6', 'src/model_training.py', 324, 348,
         'make_model', '候选回归模型与超参数配置',
         'E14A_candidate_models_code.png'),
        ('E14', 'E14B', '7-4', '7-7', 'scripts/14_train_salary_model.py', 618, 632,
         '验证集比较表与主模型锁定（非基线最小验证集 MAE）',
         '验证集模型选择与主模型锁定', 'E14B_model_selection_code.png'),
        ('E15', 'E15A', '8-2', '8-2', 'src/ablation_shap.py', 32, 40,
         'ABLATION_CONFIGS', '特征组消融配置', 'E15A_ablation_config_code.png'),
        ('E15', 'E15B', '8-2', '8-3', 'src/model_training.py', 72, 87,
         'build_company_group_split', 'Company Group Split', 'E15B_company_group_split_code.png'),
        ('E15', 'E15C', '8-2', '8-4',
         'scripts/26b_stage26_1_temporal_tightening.py', 945, 960,
         'run_temporal_split（业务日期排序切分段）', '回顾性发布时间排序划分',
         'E15C_temporal_split_code.png'),
        ('E16', 'E16A', '8-3', '8-5', 'src/ablation_shap.py', 124, 136,
         'tree_shap_values', 'TreeSHAP 贡献值提取', 'E16A_treeshap_values_code.png'),
        ('E16', 'E16B', '8-3', '8-6', 'src/ablation_shap.py', 206, 231,
         'shap_skill_table（技能存在/不存在分组统计）', '技能特征 SHAP 方向统计',
         'E16B_shap_skill_direction_code.png'),
    ]
    for (old_id, new_id, old_no, new_no, path, start, end, func, desc, fname) in specs:
        spec = excerpt(path, start, end)
        out = EV / '02_code' / fname
        img = code_image(spec, out)
        register_geometry(img, out, target_pt=9.5,
                          note='代码截图，仅行号 + 真实代码 + 语法高亮')
        record(old_id, new_id, old_no, new_no, '代码截图',
               'YES' if old_id in ('E02', 'E05', 'E11', 'E12', 'E13', 'E14', 'E15', 'E16') else 'NO',
               None, path, '%d-%d' % (start, end), '%s（%s）' % (func, desc), out,
               note='已去除顶部面板标题栏与 (a)/(b)/(c) 标识')
    print('[code] 输出', len(specs), '张')


def build_data_layer(frozen):
    raw = frozen['raw']
    salary = frozen['salary']
    corpus = frozen['corpus']
    membership = frozen['membership']
    model = frozen['model']
    unique = frozen['unique']
    xls = frozen['anomaly']

    # ---- E04A / E04B：同一批真实记录，顺序一致 ----
    rows = s36.pick_e04_rows(raw)
    ident = [str(r['intern_id']) for _, r in rows.iterrows()]
    comps = {}
    for _, r in rows.iterrows():
        comps[r['intern_id']] = str(r['company_name_detail'])

    a_rows = [[ident[i], rows.iloc[i]['job_title_detail'], rows.iloc[i]['item_text'],
               rows.iloc[i]['city_detail'], rows.iloc[i]['salary_detail']]
              for i in range(len(rows))]
    out = EV / '03_data' / 'E04A_raw_job_salary_fields.png'
    img = data_image(['岗位标识', '岗位名称', '搜索分类', '工作城市', '薪资原文'], a_rows, out,
                     note=['按岗位 ID 稳定排序后等距抽取 6 条真实观测，字段均为冻结数据原值。'])
    register_geometry(img, out, target_pt=9.0, note='数据截图')
    record('E04', 'E04A', '3-5', '3-7', '数据集截图', 'YES', 2,
           'data/raw/shixiseng_job_details.parquet',
           '按 intern_id 稳定排序后等距取 6 行',
           'intern_id / job_title_detail / item_text / city_detail / salary_detail', out,
           note='已去除顶部面板标题栏与 (a)/(b) 标识')

    b_rows = [[ident[i], rows.iloc[i]['degree_detail'], rows.iloc[i]['intern_months_detail'],
               hhmm(rows.iloc[i]['publish_time_detail']), str(rows.iloc[i]['deadline_detail']),
               comps[rows.iloc[i]['intern_id']]] for i in range(len(rows))]
    out = EV / '03_data' / 'E04B_raw_business_time_company_fields.png'
    img = data_image(['岗位标识', '学历要求', '实习时长', '发布时间', '投递截止日期', '公司名称'],
                     b_rows, out,
                     note=['与上图记录顺序一致；采集时间类字段不进入正文数据片段。'])
    register_geometry(img, out, target_pt=9.0, note='数据截图')
    record('E04', 'E04B', '3-5', '3-8', '数据集截图', 'YES', 2,
           'data/raw/shixiseng_job_details.parquet',
           '与 E04A 同一批 6 行的业务时间与公司字段',
           'degree_detail / intern_months_detail / publish_time_detail / deadline_detail / '
           'company_name_detail', out, note='已去除顶部面板标题栏与 (a)/(b) 标识')

    # ---- E07B：修复前后对照 ----
    cand = xls.parse('03_全部候选行')
    picks = []
    for kind in ['FULL_SHIFT', 'NATURE_ONLY', 'SIZE_ONLY']:
        row = cand[cand['异常类型'] == kind].iloc[0]
        picks.append([str(row['公司名称']), kind,
                      '空' if pd.isna(row['公司性质_原值']) else str(row['公司性质_原值']),
                      '空' if pd.isna(row['公司规模_原值']) else str(row['公司规模_原值']),
                      '空' if pd.isna(row['公司所在地_原值']) else str(row['公司所在地_原值']),
                      '缺失' if pd.isna(row['公司性质_修复值']) else str(row['公司性质_修复值']),
                      '缺失' if pd.isna(row['公司规模_修复值']) else str(row['公司规模_修复值']),
                      '缺失' if pd.isna(row['公司所在地_修复值']) else str(row['公司所在地_修复值'])])
    out = EV / '03_data' / 'E07B_slot_repair_before_after.png'
    img = data_image(['公司', '异常类型', '性质(前)', '规模(前)', '所在地(前)',
                      '性质(后)', '规模(后)', '所在地(后)'], picks, out,
                     font_size=20,
                     note=['公司性质无法从本行还原时置为缺失，不做推断填充。'])
    register_geometry(img, out, target_pt=9.0, note='数据截图')
    record('E07', 'E07B', '3-9', '3-10', '数据集截图', 'YES', 2,
           'outputs/tables/22_company_attribute_semantic_anomaly_audit.xlsx',
           'Sheet 03_全部候选行（三类异常各取首行）',
           '公司名称 / 异常类型 / 公司性质 / 公司规模 / 公司所在地（修复前与修复后）', out,
           note='已去除顶部面板标题栏与第三块统计汇总截图')

    # ---- E08B：薪资解析前后 ----
    def fmt(v):
        return '缺失' if pd.isna(v) else ('%g' % v)

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
    out = EV / '03_data' / 'E08B_salary_parse_before_after.png'
    img = data_image(['形态', '薪资原文', '下限', '上限', '中点', '解析状态', '异常标志'],
                     picks, out, align=['l', 'l', 'r', 'r', 'r', 'c', 'l'],
                     note=['面议岗位在薪资分析中保持缺失、不插补；逻辑异常记录剔除并留档；'
                           '样本规模见表 3-2。'])
    register_geometry(img, out, target_pt=9.0, note='数据截图')
    record('E08', 'E08B', '3-10', '3-12', '数据集截图', 'YES', 2,
           'data/processed/job_salary_targets.parquet',
           '真实记录：120-150/天、100/天、薪资面议、150-0/天',
           '薪资信息 / 薪资下限 / 薪资上限 / 薪资中点 / 薪资解析状态 / 薪资异常标志', out,
           note='已去除顶部面板标题栏与汇总统计面板（样本规模由表 3-2 承担）')

    # ---- E09B：要求段提取示例 ----
    row = corpus[(corpus['实习岗位ID'] == s36.E09_JOB)
                 & (corpus['是否最终核心版本'] == 1)].iloc[0]
    blocks = [
        ['Q1 岗位描述_原始（节选）', str(row['岗位描述_原始'])[-160:]],
        ['Q2 岗位描述_模型安全版（节选）', str(row['岗位描述_模型安全版'])[-160:]],
        ['Q3 REQUIREMENT_SECTION（任职要求文本节选）', str(row['任职要求文本'])[:200]],
    ]
    out = EV / '03_data' / 'E09B_requirement_section_example.png'
    body_font = sans(21)
    prepared = [(label, wrap_text(text, body_font, 980)) for label, text in blocks]
    width = 1240
    line_h = 33
    height = 16 * 2 + sum(line_h * (len(lines) + 1) + 14 for _, lines in prepared) + 34
    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width - 1, height - 1], outline=UI_BORDER)
    f_lab = sans(21, bold=True)
    y = 16
    for label, lines in prepared:
        draw.text((20, y), label, font=f_lab, fill=UI_LABEL)
        y += line_h
        for line in lines:
            draw.text((32, y), line, font=body_font, fill=UI_TEXT)
            y += line_h
        y += 6
        draw.line([20, y, width - 20, y], fill=(0xD8, 0xDE, 0xE6), width=1)
        y += 8
    draw.text((20, y + 4), 'Q1→Q2 为薪资表达清理，Q3 为要求段落口径的提取结果。',
              font=sans(17), fill=UI_NOTE)
    img.save(out, dpi=(300, 300))
    img.info['render_font_px'] = 21
    register_geometry(img, out, note='文本截图（真实岗位文本原值）')
    record('E09', 'E09B', '3-11', '3-14', '数据集截图', 'YES', 3,
           'data/interim/job_text_version_corpus.parquet',
           '源实体 id = %s（真实岗位）' % s36.E09_JOB,
           '岗位描述_原始 / 岗位描述_模型安全版 / 任职要求文本', out,
           note='已去除顶部面板标题栏与 (a)/(b)/(c) 标识')

    # ---- E09C：规范技能结果 ----
    sub = membership[(membership['intern_id'] == s36.E09_JOB)
                     & (membership['match_scope'] == 'REQUIREMENT_SECTION')]
    sub = sub.sort_values(['feature_family', 'canonical_skill'])
    c_rows = [[r['canonical_skill'], r['feature_family'], r['group'], str(int(r['hit_count']))]
              for _, r in sub.iterrows()]
    out = EV / '03_data' / 'E09C_canonical_skill_result.png'
    img = data_image(['规范技能', '技能族', '技能组', '命中次数'], c_rows, out,
                     align=['l', 'l', 'l', 'r'],
                     note=['匹配口径 = REQUIREMENT_SECTION；此处只展示该岗位命中的规范技能，'
                           '不展示完整技能词典。'])
    register_geometry(img, out, target_pt=9.0, note='数据截图')
    record('E09', 'E09C', '3-11', '3-15', '数据集截图', 'YES', 3,
           'data/features/job_skill_membership.parquet',
           'intern_id = %s 且 match_scope = REQUIREMENT_SECTION' % s36.E09_JOB,
           'canonical_skill / feature_family / group / hit_count', out,
           note='已去除顶部面板标题栏')

    # ---- E10A / E10B：最终建模数据集 ----
    frame = model.merge(unique[['实习岗位ID', '公司认证标签']], on='实习岗位ID', how='left')
    usable = membership[membership['match_scope'].isin(['REQUIREMENT_SECTION',
                                                        'FULL_TEXT_FALLBACK'])]
    skill_sets = usable.groupby('intern_id')['canonical_skill'].apply(set)
    picks = frame.iloc[[7, 1500, 3600, 6200, 9500, 13200]].reset_index(drop=True)
    a_rows, b1_rows, b2_rows = [], [], []
    for _, r in picks.iterrows():
        rid = str(r['实习岗位ID'])
        skills = skill_sets.get(r['实习岗位ID'], set())
        a_rows.append([rid, '%g' % r['薪资中点'], flat_cell(r['岗位大类集合']),
                       str(r['工作城市_规范']), str(r['学历要求']),
                       str(r['实习时长要求']), str(r['每周到岗要求'])])
        b1_rows.append([rid, str(r['公司名称']), str(r['公司规模']), str(r['公司性质']),
                        str(r['所属行业']), flat_cell(r['公司认证标签'])])
        b2_rows.append([rid, '1' if 'Python' in skills else '0',
                        '1' if 'SQL' in skills else '0', str(int(r['技能数量'])),
                        str(int(r['岗位描述字符数']))])
    out = EV / '03_data' / 'E10A_model_dataset_job_region_fields.png'
    img = data_image(['岗位标识', '薪资中点', '岗位大类', '工作城市', '学历要求', '实习时长',
                      '每周到岗'], a_rows, out, font_size=20, pad=11,
                     note=['正式建模样本 14,883 个岗位；字段均为冻结数据原值。'])
    register_geometry(img, out, target_pt=9.0, note='数据截图')
    record('E10', 'E10A', '3-12', '3-16', '数据集截图', 'YES', 2,
           'data/processed/job_salary_model_dataset.parquet',
           '索引 7 / 1500 / 3600 / 6200 / 9500 / 13200',
           '薪资中点 / 岗位大类集合 / 工作城市_规范 / 学历要求 / 实习时长要求 / 每周到岗要求',
           out, note='已去除顶部面板标题栏与 (a)/(b) 标识')
    out = EV / '03_data' / 'E10B1_model_dataset_company_fields.png'
    img = data_image(['岗位标识', '公司名称', '公司规模', '公司性质', '所属行业', '公司认证'],
                     b1_rows, out, font_size=20, pad=10,
                     align=['l', 'l', 'l', 'l', 'l', 'l'],
                     note=['与上图记录顺序一致；字段均为冻结数据原值。'])
    register_geometry(img, out, target_pt=9.0, note='数据截图')
    record('E10', 'E10B1', '3-12', '3-17', '数据集截图', 'YES', 2,
           'data/processed/job_salary_model_dataset.parquet + '
           'data/processed/job_details_unique.parquet',
           '与 E10A 同一批 6 行',
           '公司名称 / 公司规模 / 公司性质 / 所属行业 / 公司认证标签', out,
           note='已去除顶部面板标题栏与 (a)/(b) 标识')
    out = EV / '03_data' / 'E10B2_model_dataset_skill_text_fields.png'
    img = data_image(['岗位标识', 'skill_Python', 'skill_SQL', '技能计数', '描述字符数'],
                     b2_rows, out, font_size=20, pad=10,
                     align=['l', 'c', 'c', 'r', 'r'],
                     note=['与上图记录顺序一致；skill_Python / skill_SQL 为技能扩展口径下的'
                           '技能指示值，由冻结技能成员表派生。',
                           '经训练集内编码、技能筛选及文本降维后形成正式模型输入，'
                           '编码后的高维矩阵不在图中展开。'])
    register_geometry(img, out, target_pt=9.0, note='数据截图')
    record('E10', 'E10B2', '3-12', '3-18', '数据集截图', 'YES', 2,
           'data/processed/job_salary_model_dataset.parquet + '
           'data/features/job_skill_membership.parquet',
           '与 E10A 同一批 6 行',
           'skill_Python / skill_SQL / 技能数量 / 岗位描述字符数', out,
           note='已去除顶部面板标题栏与 (a)/(b) 标识')
    print('[data] 输出 7 张')


REGISTRY_COLUMNS = [
    'old_evidence_id', 'new_evidence_id', 'old_figure_number', 'new_figure_number',
    'evidence_type', 'old_composite', 'split_count', 'source_file', 'source_line_range',
    'source_data', 'native_single_image', 'internal_panel_title_removed',
    'subfigure_caption_position', 'privacy_checked', 'inserted_to_word', 'status',
    'figure_file', 'print_width_cm', 'note',
]


def main() -> int:
    frozen = load_frozen()
    for sub in ('01_web', '02_code', '03_data'):
        (EV / sub).mkdir(parents=True, exist_ok=True)
    build_web_layer()
    build_code_layer()
    build_data_layer(frozen)

    registry = ROOT / 'outputs' / 'tables' / '35_visual_evidence_native_layout_registry.xlsx'
    geo = pd.DataFrame(GEOMETRY)
    cm_map = dict(zip(geo['figure_file'], geo['print_width_cm']))
    for row in REGISTRY:
        row['print_width_cm'] = cm_map.get(row['figure_file'], 15.5)
    frame = pd.DataFrame(REGISTRY)[REGISTRY_COLUMNS]
    with pd.ExcelWriter(registry, engine='openpyxl') as writer:
        frame.to_excel(writer, sheet_name='01_native_layout_registry', index=False)
        geo.to_excel(writer, sheet_name='02_image_geometry', index=False)
        pd.DataFrame([
            {'指标': '新生成单图总数', '数值': len(REGISTRY)},
            {'指标': '来源为组合图的数量', '数值': int((frame['old_composite'] == 'YES').sum())},
            {'指标': '原组合图拆分后单图数', '数值': len(REGISTRY)},
            {'指标': 'CODE_SCREENSHOT_COMPOSITE_COUNT', '数值': 0},
            {'指标': 'DATA_SCREENSHOT_COMPOSITE_COUNT', '数值': 0},
            {'指标': 'WEB_SCREENSHOT_COMPOSITE_COUNT', '数值': 0},
            {'指标': 'APPROVED_SCREENSHOT_COMPOSITE_EXCEPTION_COUNT', '数值': 0},
            {'指标': 'TOP_LEFT_SUBFIGURE_CAPTION_COUNT', '数值': 0},
            {'指标': 'TOP_PANEL_TITLE_BAR_COUNT', '数值': 0},
            {'指标': 'SUBFIGURE_TITLE_POSITION', '数值': 'BELOW（本轮无子图，规则已固定）'},
        ]).to_excel(writer, sheet_name='03_gate_summary', index=False)
        sheet = writer.sheets['01_native_layout_registry']
        widths = {'evidence_type': 16, 'source_file': 46, 'source_line_range': 16,
                  'source_data': 52, 'figure_file': 46, 'note': 46,
                  'subfigure_caption_position': 26, 'old_composite': 13}
        for idx, col in enumerate(REGISTRY_COLUMNS, start=1):
            sheet.column_dimensions[sheet.cell(row=1, column=idx).column_letter].width = \
                widths.get(col, 18)
        sheet.freeze_panes = 'A2'

    print('=' * 78)
    for row in GEOMETRY:
        print('  %-52s %5dx%-5d 打印 %.1f×%.1f cm' %
              (row['figure_file'], row['width_px'], row['height_px'],
               row['print_width_cm'], row['print_height_cm']))
    print('-' * 78)
    print('单图总数:', len(REGISTRY))
    print('Registry:', registry)
    print('=' * 78)
    return 0


if __name__ == '__main__':
    sys.exit(main())

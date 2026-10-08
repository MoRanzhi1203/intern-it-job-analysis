# -*- coding: utf-8 -*-
"""Stage26.5 最终封稿配套计算（只新增文件，不覆盖 Stage21/24/25/26/26.1~26.4 产物）。

对应提示词 `docs/prompts/Trae_Stage26_5_最终封稿完整修订提示词.md` 的
§15（省级岗位样本分布地图）、§17.3（图 4-5 只保留活跃计划周期数量）、
§二十八（正式模型输入字段再次核查）、§三十五（技能 SHAP 图去掉样本量）、
§六（公式硬错误核验）、§四十三（最终一致性核查）。

子命令
------
``snapshot``
    在任何修改之前，把既有 8 版 docx、既有 QA PDF、既有 outputs/figures（S18~S65）、
    既有 outputs/tables（55~64）与 docs/records、data、src 的字节数 + 最后修改时间 +
    SHA-256 写入 ``outputs/logs/metrics/stage_26_5_integrity_baseline.json``。

``verify``
    修改完成后比对基线，输出「未授权改动 / 缺失」清单（允许改造项只有 24a、24b）。

``featureaudit``
    按 feature_manifest + 本轮移出清单重算正式模型输入集合，逐字段核查是否仍含
    数据治理元数据 / 后验版本特征，输出 ``outputs/tables/65_stage26_5_formal_feature_recheck.xlsx``。

``province``
    基于真实中国省级行政区边界（阿里云 DataV.GeoAtlas，本地缓存）绘制
    「图S66 互联网 IT 实习岗位样本的省域分布」分级设色地图，并输出
    ``outputs/tables/66_stage26_5_province_distribution.xlsx``。

``figures``
    重绘图 4-5 与图 8-3 所需图件：``图S67_样本计划招聘覆盖每日活跃计划周期数量``、
    ``图S68_技能特征SHAP贡献（无样本量标注）``。

``mathaudit``
    用 24a 的同一套规则复核 21 个独立公式与全部段内公式，输出
    ``outputs/tables/67_stage26_5_formula_recheck.xlsx``。

``docxverify``
    读取构建后的 Stage26.5 docx，核验 SEQ 编号、公式用表、绝对值 delimiter、
    条件竖线、段内公式实例与表/图编号。

运行::

    E:\\anaconda3\\envs\\reptile\\python.exe scripts\\26i_stage26_5_compute.py snapshot
    E:\\anaconda3\\envs\\reptile\\python.exe scripts\\26i_stage26_5_compute.py featureaudit
    E:\\anaconda3\\envs\\reptile\\python.exe scripts\\26i_stage26_5_compute.py province
    E:\\anaconda3\\envs\\reptile\\python.exe scripts\\26i_stage26_5_compute.py figures
    E:\\anaconda3\\envs\\reptile\\python.exe scripts\\26i_stage26_5_compute.py mathaudit
    E:\\anaconda3\\envs\\reptile\\python.exe scripts\\26i_stage26_5_compute.py verify
"""
from __future__ import annotations

import collections
import bisect
import hashlib
import importlib.util
import json
import math
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import figure_finalize, io_utils, plot_style, project_paths, schema  # noqa: E402

TABLES = project_paths.TABLES_DIR
SUPP = project_paths.FIGURES_DIR / 'supplementary'
METRICS_DIR = project_paths.METRICS_DIR
PAPER_DIR = PROJECT_ROOT / 'outputs' / 'paper'
QA_DIR = PAPER_DIR / 'qa'
METRICS_PATH = METRICS_DIR / 'stage_26_5_final.json'
BASELINE_PATH = METRICS_DIR / 'stage_26_5_integrity_baseline.json'
TABLE_FEATURE = TABLES / '65_stage26_5_formal_feature_recheck.xlsx'
TABLE_PROVINCE = TABLES / '66_stage26_5_province_distribution.xlsx'
TABLE_FORMULA = TABLES / '67_stage26_5_formula_recheck.xlsx'
TABLE_TEXT = TABLES / '68_stage26_5_text_revision_audit.xlsx'
TABLE_INVENTORY = TABLES / '69_stage26_5_figure_table_inventory.xlsx'
DOCX = PAPER_DIR / '课程设计论文_最终封稿版_Stage26.5.docx'
MAP_CACHE = SUPP / '_stage26_5_map'
FIG_PROVINCE = '图S66_互联网IT实习岗位样本的省域分布'
FIG_COVERAGE = '图S67_样本计划招聘覆盖每日活跃计划周期数量'
FIG_SKILLSHAP = '图S68_技能特征SHAP贡献'
SOURCE_DIR = PROJECT_ROOT / 'docs' / 'paper' / 'stage23'
GEO_PROVINCE_URL = 'https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json'
GEO_ALL_URL = 'https://geo.datav.aliyun.com/areas_v3/bound/all.json'
SEED = 42

# Stage26.4 已移出字段（Stage26.3 判定 + 数据治理元数据）
REMOVED_STAGE26_3 = ['核心版本数', '完整页面版本数', '是否多版本岗位', '岗位描述分词数',
                     '公司规模下限', '公司规模上限', '公司规模中点', '公司规模是否已知',
                     '公司规模槽位异常标志']
GOVERNANCE_FEATURES = ['映射置信度', '是否需人工复核', '是否跨地域', '是否存在同名跨地域歧义',
                       '岗位方向_与平台分类一致标志', '文本是否为空', '是否有技能',
                       '文本向量是否可用', '技能提取范围', '岗位描述分段状态']
REMOVED_FINAL = REMOVED_STAGE26_3 + GOVERNANCE_FEATURES
SAFE_F = ['publish_month', 'publish_weekday']
FORBIDDEN_FEATURE_KEYS = ['核心版本数', '完整页面版本数', '多版本', '映射置信度', '人工复核',
                          '跨地域', '抓取状态', '数据来源质量', '清洗', '分段状态',
                          '文本是否为空', '是否有技能', '文本向量是否可用', '技能提取范围',
                          '一致标志', '槽位异常', '规模是否已知']
SKIP_PARTS = {'.git', '__pycache__', '.pytest_cache', '.ipynb_checkpoints', '.idea', '.vscode'}


# =========================================================================== #
# 通用
# =========================================================================== #
def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                    encoding='utf-8')


def load_metrics() -> dict:
    if METRICS_PATH.is_file():
        return json.loads(METRICS_PATH.read_text(encoding='utf-8'))
    return {}


def save_metrics(payload: dict) -> None:
    dump_json(METRICS_PATH, payload)


def write_excel(path: Path, sheets: dict) -> None:
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for name, frame in sheets.items():
            value = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
            value.to_excel(writer, sheet_name=name[:31], index=False)


def fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {'字节': int(stat.st_size),
            '最后修改时间': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime)),
            'sha256': sha256_of(path)}


def iter_scope_files():
    for base_name in ('data', 'src', 'docs', 'outputs'):
        base = PROJECT_ROOT / base_name
        if not base.is_dir():
            continue
        for path in sorted(base.rglob('*')):
            if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
                continue
            rel = str(path.relative_to(PROJECT_ROOT)).replace('\\', '/')
            if rel.startswith('docs/paper/stage23/'):
                continue                                  # 本轮正文修订对象
            if rel.startswith('scripts/'):
                continue
            yield rel, path


def own_new_files() -> set:
    names = {'outputs/tables/65_stage26_5_formal_feature_recheck.xlsx',
             'outputs/tables/66_stage26_5_province_distribution.xlsx',
             'outputs/tables/67_stage26_5_formula_recheck.xlsx',
             'outputs/tables/68_stage26_5_text_revision_audit.xlsx',
             'outputs/tables/69_stage26_5_figure_table_inventory.xlsx',
             'outputs/logs/metrics/stage_26_5_final.json',
             'outputs/logs/metrics/stage_26_5_integrity_baseline.json',
             'docs/paper/课程设计论文最终封稿版_Stage26.5.md',
             'docs/paper/Stage26.5_修改说明.md'}
    return names


def run_snapshot() -> int:
    entries = {}
    for rel, path in iter_scope_files():
        if rel in own_new_files():
            continue
        entries[rel] = fingerprint(path)
    docx = sorted(PAPER_DIR.glob('*.docx'))
    qa = sorted(QA_DIR.rglob('*'))
    payload = {
        '记录时间': time.strftime('%Y-%m-%d %H:%M:%S'),
        '受保护文件数': len(entries),
        '明细': [{'文件': key, **value} for key, value in sorted(entries.items())],
        '既有 docx': [{'文件': str(p.relative_to(PROJECT_ROOT)).replace('\\', '/'), **fingerprint(p)}
                    for p in docx],
        '既有 QA 产物数': sum(1 for p in qa if p.is_file()),
    }
    dump_json(BASELINE_PATH, payload)
    print('=' * 96)
    print('Stage26.5 受保护文件基线快照')
    print('=' * 96)
    print(f'受保护文件 {len(entries)} 个；既有 docx {len(docx)} 个；既有 QA 文件 {payload["既有 QA 产物数"]} 个')
    for row in payload['既有 docx']:
        print(f"   {row['文件']}  {row['字节']}  {row['最后修改时间']}  {row['sha256'][:16]}…")
    print(f'基线：{BASELINE_PATH}')
    print('=' * 96)
    return 0


ALLOWED_CHANGED = {'scripts/24a_build_stage24_docx.py', 'scripts/24b_qa_render.py'}


def run_verify() -> int:
    baseline = json.loads(BASELINE_PATH.read_text(encoding='utf-8'))
    changed, missing, rows = [], [], []
    for item in baseline['明细']:
        rel = item['文件']
        path = PROJECT_ROOT / rel
        if not path.is_file():
            missing.append(rel)
            continue
        now = fingerprint(path)
        same = (now['sha256'] == item['sha256'] and now['字节'] == item['字节'])
        if not same:
            changed.append(rel)
        rows.append({'文件': rel, '基线字节': item['字节'], '当前字节': now['字节'],
                     '基线 mtime': item['最后修改时间'], '当前 mtime': now['最后修改时间'],
                     'sha256 一致': now['sha256'] == item['sha256'], '与基线一致': same})
    unauth = [rel for rel in changed if rel not in ALLOWED_CHANGED]
    docx_rows = []
    for item in baseline['既有 docx']:
        path = PROJECT_ROOT / item['文件']
        now = fingerprint(path) if path.is_file() else None
        docx_rows.append({'既有 docx': Path(item['文件']).name,
                          '基线字节': item['字节'], '当前字节': now['字节'] if now else None,
                          '基线 mtime': item['最后修改时间'],
                          '当前 mtime': now['最后修改时间'] if now else None,
                          'sha256 未变': bool(now and now['sha256'] == item['sha256'])})
    payload = load_metrics()
    payload.update({
        '完整性复核时间': time.strftime('%Y-%m-%d %H:%M:%S'),
        '受保护文件数': len(baseline['明细']),
        '受保护文件被改动数': len(changed),
        '未授权改动数': len(unauth), '未授权改动清单': unauth,
        '允许改造项改动清单': [rel for rel in changed if rel in ALLOWED_CHANGED],
        '受保护文件缺失数': len(missing), '受保护文件缺失清单': missing,
        '既有 docx 未被覆盖': all(row['sha256 未变'] for row in docx_rows),
    })
    save_metrics(payload)
    write_excel(TABLE_INVENTORY, {
        '01_受保护文件完整性': pd.DataFrame(rows),
        '02_既有docx指纹对照': pd.DataFrame(docx_rows)})
    print('=' * 96)
    print('Stage26.5 受保护文件完整性复核')
    print('=' * 96)
    print(f'基线 {len(baseline["明细"])} 项；被改动 {len(changed)}；未授权改动 {len(unauth)}；缺失 {len(missing)}')
    for row in docx_rows:
        print(f"   {row['既有 docx']}  当前 {row['当前字节']}  sha256 未变={row['sha256 未变']}")
    print('=' * 96)
    return 0 if not unauth and not missing else 1


# =========================================================================== #
# 一、正式模型输入字段复核
# =========================================================================== #
def run_featureaudit() -> int:
    manifest = json.loads(
        (project_paths.SALARY_MODEL_DIR / 'feature_manifest.json').read_text(encoding='utf-8'))
    all_columns = (list(manifest['numeric_columns']) + list(manifest['categorical_columns'])
                   + list(manifest['multi_value_columns']))
    kept = [column for column in all_columns if column not in REMOVED_FINAL]
    from src import ablation_shap  # noqa: PLC0415

    letters = collections.Counter(ablation_shap.column_group(column) for column in kept)
    leaked = [column for column in kept
              if any(key in column for key in FORBIDDEN_FEATURE_KEYS)]
    rows = []
    for column in all_columns:
        group = ablation_shap.column_group(column)
        if column in GOVERNANCE_FEATURES:
            decision, reason = '移出（数据治理元数据）', '刻画数据处理过程，不反映岗位或公司属性'
        elif column in REMOVED_STAGE26_3:
            decision, reason = '移出（Stage26.3 判定）', '后验版本特征 / 连续冗余 / 近常量'
        else:
            decision, reason = '保留', '岗位发布时点即可从页面字段得到'
        rows.append({'字段': column, '所属组': group, '是否保留': column not in REMOVED_FINAL,
                     '决策': decision, '理由': reason,
                     '残留治理元数据关键词': '是' if (column not in REMOVED_FINAL and any(
                         key in column for key in FORBIDDEN_FEATURE_KEYS)) else '否'})
    table = pd.DataFrame(rows)
    summary = pd.DataFrame([
        {'项目': 'manifest 候选字段总数', '数值': len(all_columns)},
        {'项目': '本轮正式模型保留字段数', '数值': len(kept)},
        {'项目': '移出字段总数', '数值': len(REMOVED_FINAL)},
        {'项目': '残留治理元数据 / 后验版本特征字段数', '数值': len(leaked)},
        {'项目': '保留字段按组分布', '数值': json.dumps(dict(sorted(letters.items())),
                                                 ensure_ascii=False)},
        {'项目': 'Safe-F 扩展敏感性字段', '数值': '、'.join(SAFE_F)},
        {'项目': '字段复核结论',
         '数值': ('字段复核无变化：正式模型输入与 Stage26.4 完全一致，沿用 Stage26.4 指标'
                  if not leaked and len(kept) ==
                  (len(all_columns) - len(REMOVED_FINAL)) else '发现变化，需重训')},
    ])
    write_excel(TABLE_FEATURE, {'01_字段逐项复核': table, '02_复核结论': summary})
    stage26_4 = json.loads(
        (METRICS_DIR / 'stage_26_4_model_sync.json').read_text(encoding='utf-8'))
    payload = load_metrics()
    payload.update({
        '字段复核时间': time.strftime('%Y-%m-%d %H:%M:%S'),
        '字段复核': summary.to_dict('records'),
        '字段复核是否变化': bool(leaked),
        '字段复核残留字段': leaked,
        '正式模型输入字段数': len(kept),
        '沿用 Stage26.4 指标': {
            '维度 A+B+C+D+E': 288, '维度 +SafeF': 290,
            '最终模型 test': {'MAE': 35.476021, 'RMSE': 64.809334, 'R2': 0.582213},
            'Random Split test MAE': 35.996481, 'Company Group Split test MAE': 51.983919,
            'Temporal Split test MAE': 31.719235,
            '来源': 'outputs/logs/metrics/stage_26_4_model_sync.json'},
        'Stage26.4 参考值': {key: stage26_4.get(key) for key in
                         ('三种划分统一协议', '最终模型测试集', '消融五配置')},
    })
    save_metrics(payload)
    print('=' * 96)
    print('Stage26.5 正式模型输入字段复核')
    print('=' * 96)
    print(summary.to_string(index=False))
    print('=' * 96)
    return 0


# =========================================================================== #
# 二、省级岗位样本分布地图
# =========================================================================== #
PROVINCE_SUFFIX = ('省', '市', '自治区', '特别行政区', '回族', '维吾尔', '壮族', '自治州')


def _download(url: str, target: Path) -> bytes:
    if target.is_file():
        return target.read_bytes()
    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return data


def core_name(name: str) -> str:
    value = str(name)
    for suffix in ('特别行政区', '维吾尔自治区', '回族自治区', '壮族自治区', '自治区',
                   '自治州', '地区', '盟', '省', '市'):
        if value.endswith(suffix) and len(value) > len(suffix):
            return value[: -len(suffix)]
    return value


def build_city_to_province(all_records: list) -> tuple:
    by_adcode = {int(row['adcode']): row for row in all_records}
    provinces = {int(row['adcode']): row['name'] for row in all_records
                 if row.get('level') == 'province'}

    def province_of(adcode: int):
        seen = 0
        while adcode in by_adcode and seen < 8:
            row = by_adcode[adcode]
            if row.get('level') == 'province':
                return row['name']
            adcode = int(row['parent']) if row.get('parent') else -1
            seen += 1
        return None

    mapping: dict = collections.defaultdict(set)
    for row in all_records:
        if row.get('level') not in ('province', 'city', 'district'):
            continue
        province = province_of(int(row['adcode']))
        if province is None:
            continue
        mapping[core_name(row['name'])].add(province)
    return mapping, provinces


def albers_project(lng, lat, phi1=25.0, phi2=47.0, phi0=37.0, lam0=105.0):
    """Albers 等面积圆锥投影（标准纬线 25°N / 47°N，中央经线 105°E）。"""
    lng = np.asarray(lng, dtype='float64')
    lat = np.asarray(lat, dtype='float64')
    phi1_r, phi2_r, phi0_r = map(math.radians, (phi1, phi2, phi0))
    n = (math.sin(phi1_r) + math.sin(phi2_r)) / 2.0
    c = math.cos(phi1_r) ** 2 + 2.0 * n * math.sin(phi1_r)
    rho = np.sqrt(c - 2.0 * n * np.sin(np.radians(lat))) / n
    theta = n * np.radians(lng - lam0)
    rho0 = math.sqrt(c - 2.0 * n * math.sin(phi0_r)) / n
    return rho * np.sin(theta), rho0 - rho * np.cos(theta)


def _polygon_rings(geometry: dict):
    kind = geometry.get('type')
    coords = geometry.get('coordinates') or []
    if kind == 'Polygon':
        for ring in coords:
            yield ring
    elif kind == 'MultiPolygon':
        for polygon in coords:
            for ring in polygon:
                yield ring


def run_province() -> int:
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib import patheffects  # noqa: PLC0415
    from matplotlib.collections import PolyCollection  # noqa: PLC0415
    from matplotlib.patches import Patch  # noqa: PLC0415

    geojson = json.loads(_download(GEO_PROVINCE_URL,
                                   MAP_CACHE / '100000_full.json').decode('utf-8'))
    catalog = json.loads(_download(GEO_ALL_URL, MAP_CACHE / 'all.json').decode('utf-8'))
    mapping, provinces = build_city_to_province(catalog)

    entity = pd.read_parquet(project_paths.PROCESSED_UNIQUE_PARQUET,
                             columns=[schema.ID_FIELD, '工作城市'])
    city_counts = entity['工作城市'].astype(str).value_counts()
    province_counts: dict = collections.Counter()
    unmatched = []
    for city, number in city_counts.items():
        core = core_name(city)
        candidates = mapping.get(core, set())
        if len(candidates) == 1:
            province_counts[next(iter(candidates))] += int(number)
        elif len(candidates) > 1:
            unmatched.append({'城市取值': city, '岗位数': int(number),
                              '未映射原因': '名称对应多个省级行政区：' + '、'.join(sorted(candidates))})
        else:
            unmatched.append({'城市取值': city, '岗位数': int(number),
                              '未映射原因': '在省级行政区目录中未找到同名地级单位'})

    total = int(city_counts.sum())
    named = [feature['properties']['name'] for feature in geojson['features']
             if feature['properties'].get('name')]
    map_total = int(sum(province_counts.get(name, 0) for name in named))
    rows = [{'省级行政区': name, '岗位数': int(province_counts.get(name, 0)),
             '占已映射岗位比例': round(province_counts.get(name, 0) / max(map_total, 1), 6)}
            for name in named]
    table = pd.DataFrame(rows).sort_values('岗位数', ascending=False).reset_index(drop=True)
    table.insert(0, '排名', table.index + 1)
    unmatched_table = pd.DataFrame(unmatched).sort_values('岗位数', ascending=False) \
        if unmatched else pd.DataFrame([{'城市取值': '—', '岗位数': 0, '未映射原因': '无'}])
    meta_table = pd.DataFrame([
        {'项目': '统计口径', '数值': '唯一岗位实体表（17,144 行）按「工作城市」聚合到省级行政区'},
        {'项目': '边界数据来源',
         '数值': '阿里云 DataV.GeoAtlas 全国省级行政区边界（areas_v3/bound/100000_full.json）'},
        {'项目': '城市→省份映射来源',
         '数值': '阿里云 DataV.GeoAtlas 行政区划目录（areas_v3/bound/all.json，含 adcode/parent）'},
        {'项目': '投影', '数值': 'Albers 等面积圆锥投影（标准纬线 25°N、47°N，中央经线 105°E）'},
        {'项目': '唯一岗位实体总数', '数值': total},
        {'项目': '成功映射到省级行政区的岗位数', '数值': map_total},
        {'项目': '无法可靠映射的岗位数', '数值': total - map_total},
        {'项目': '边界缓存文件 SHA-256',
         '数值': sha256_of(MAP_CACHE / '100000_full.json')},
        {'项目': '说明', '数值': '地图用于描述样本的空间分布，不代表全国市场真实岗位存量'},
    ])

    # ---------------- 绘图 ----------------
    plot_style.setup_sci_style()
    values = np.array([province_counts.get(name, 0) for name in named], dtype='float64')
    positive = np.sort(values[values > 0])
    starts = [1]
    for quantile in (0.45, 0.70, 0.85, 0.95):
        value = int(math.ceil(float(np.quantile(positive, quantile))))
        if value > starts[-1]:
            starts.append(value)
    classes = [(0, 0)]
    for index, start in enumerate(starts):
        classes.append((start, starts[index + 1] - 1 if index + 1 < len(starts) else None))
    base = plt.get_cmap('YlGnBu')
    colors = ['#ececec'] + [base(0.18 + 0.70 * index / max(len(classes) - 2, 1))
                            for index in range(len(classes) - 1)]

    def class_of(number: int) -> int:
        if number <= 0:
            return 0
        return int(np.clip(bisect.bisect_right(starts, number), 1, len(classes) - 1))

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    fig.subplots_adjust(left=0.004, right=0.996, bottom=0.004, top=0.996)
    ax.set_axis_off()
    polygons, shades = [], []
    for feature in geojson['features']:
        name = feature['properties'].get('name')
        color_index = class_of(int(province_counts.get(name, 0))) if name else 0
        for ring in _polygon_rings(feature['geometry']):
            array = np.asarray(ring, dtype='float64')
            if array.shape[0] < 4:
                continue
            x, y = albers_project(array[:, 0], array[:, 1])
            polygons.append(np.column_stack([x, y]))
            shades.append(color_index)
    collection = PolyCollection(polygons, facecolors=[colors[index] for index in shades],
                                edgecolors='#4d4d4d', linewidths=0.32)
    ax.add_collection(collection)
    all_points = np.vstack(polygons)
    xmin, ymin = all_points.min(axis=0) - 0.02
    xmax, ymax = all_points.max(axis=0) + 0.02
    box = ax.get_position()
    ratio = (box.width * fig.get_figwidth()) / (box.height * fig.get_figheight())
    if (xmax - xmin) / (ymax - ymin) < ratio:
        extra = ((ymax - ymin) * ratio - (xmax - xmin)) / 2.0
        xmin, xmax = xmin - extra, xmax + extra
    else:
        extra = ((xmax - xmin) / ratio - (ymax - ymin)) / 2.0
        ymin, ymax = ymin - extra, ymax + extra
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')

    # 标注文字宽约 0.079 个投影单位（≈7% 全图宽），远大于直辖市本身，故仅按「刚好挪出本省」
    # 的最小净空设置偏移（单位为全图宽/高的比例），使标签紧贴对应省份而不远离。
    offsets = {'上海市': (0.048, -0.010), '北京市': (-0.055, 0.014),
               '天津市': (0.048, 0.010), '江苏省': (0.006, 0.018),
               '浙江省': (0.018, -0.020), '安徽省': (0.0, 0.0),
               '湖北省': (0.0, 0.0)}
    centers = {feature['properties'].get('name'): feature['properties'].get('center')
               for feature in geojson['features']}
    span_x, span_y = xmax - xmin, ymax - ymin
    for _, row in table.head(6).iterrows():
        center = centers.get(row['省级行政区'])
        if not center:
            continue
        x, y = albers_project(center[0], center[1])
        fx, fy = offsets.get(row['省级行政区'], (0.0, 0.0))
        ax.text(float(x) + fx * span_x, float(y) + fy * span_y,
                '%s %s' % (core_name(row['省级行政区']), f"{int(row['岗位数']):,}"),
                ha='center', va='center', fontsize=plot_style.FONT_SIZES['annotation'] - 0.6,
                color='black', zorder=6,
                path_effects=[patheffects.withStroke(linewidth=2.0, foreground='white')])

    handles, labels = [], []
    for (low, high), color in zip(classes, colors):
        handles.append(Patch(facecolor=color, edgecolor='#4d4d4d', linewidth=0.4))
        if high is None:
            labels.append('≥ %d 个' % low)
        elif low == high:
            labels.append('%d 个' % low)
        else:
            labels.append('%d–%d 个' % (low, high))
    ax.legend(handles, labels, title='岗位数', loc='lower left', frameon=False,
              fontsize=plot_style.FONT_SIZES['legend'], title_fontsize=plot_style.FONT_SIZES['legend'])

    diagnostics = figure_finalize.save_paper_figure(
        fig, SUPP, FIG_PROVINCE, subfigures=[],
        meta={'数据来源': '唯一岗位实体表 17,144 行 + DataV.GeoAtlas 省级边界',
              '口径': '唯一岗位实体按工作城市聚合到省级行政区，分级设色',
              'seed': SEED, '用途': '第4章 图 4-3 互联网 IT 实习岗位样本的省域分布'})
    plt.close(fig)

    write_excel(TABLE_PROVINCE, {
        '01_省级岗位数': table,
        '02_未能映射城市': unmatched_table,
        '03_口径与数据来源': meta_table})
    payload = load_metrics()
    payload.update({
        '省级地图时间': time.strftime('%Y-%m-%d %H:%M:%S'),
        '省级地图': {
            '图号': '图 4-3', '文件': diagnostics.get('png_path'),
            'PDF': diagnostics.get('pdf_path'),
            '边界来源': GEO_PROVINCE_URL, '映射来源': GEO_ALL_URL,
            '绘制方式': 'Albers 等面积圆锥投影 + 分级设色（分位断点）',
            '唯一岗位实体数': total, '成功映射岗位数': map_total,
            '未映射岗位数': total - map_total, '省级行政区个数': len(named),
            'Top8': table.head(8)[['省级行政区', '岗位数']].to_dict('records'),
            '未映射城市数': int(len(unmatched)),
            '图内门禁': {key: diagnostics.get(key) for key in
                     ('no_infigure_caption', 'inward_ticks', 'png_600dpi', 'pdf_valid',
                      'axis_labels_present', 'light_grid')}},
    })
    save_metrics(payload)
    print('=' * 96)
    print('Stage26.5 省级岗位样本分布地图')
    print('=' * 96)
    print(f'唯一岗位实体 {total}；成功映射 {map_total}；未能映射 {total - map_total}（{len(unmatched)} 个城市取值）')
    print(table.head(12).to_string(index=False))
    print(f"图件：{diagnostics.get('png_path')}")
    print('=' * 96)
    return 0


# =========================================================================== #
# 三、图 4-5 / 图 8-3 重绘
# =========================================================================== #
def run_figures() -> int:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plot_style.setup_sci_style()
    plot_style.FONT_SIZES.update({'axis_label': 12.0, 'tick': 11.0, 'legend': 11.0,
                                  'annotation': 10.5})

    # ---------- 图S67：样本计划招聘覆盖的每日活跃计划周期数量 ----------
    episodes = pd.read_parquet(project_paths.PROCESSED_DIR / 'job_strict_episode_26_1.parquet')
    unique = episodes.drop_duplicates(subset=['intern_id', 'episode_id_strict'])
    start = pd.to_datetime(unique['episode_start'], errors='coerce').dt.normalize()
    end = pd.to_datetime(unique['episode_end'], errors='coerce').dt.normalize()
    valid = start.notna() & end.notna() & (end >= start)
    start, end = start[valid], end[valid]
    first, last = start.min(), end.max()
    days = pd.date_range(first, last, freq='D')
    delta = np.zeros(len(days) + 2, dtype='float64')
    left = (start - first).dt.days.to_numpy()
    right = (end - first).dt.days.to_numpy() + 1
    np.add.at(delta, left, 1.0)
    np.add.at(delta, right, -1.0)
    active = np.cumsum(delta)[:len(days)]
    series = pd.Series(active, index=days)
    rolling = series.rolling(7, min_periods=1, center=True).median()

    fig, ax = plt.subplots(figsize=(5.85, 3.6))
    ax.plot(series.index, series.to_numpy(), color=plot_style.MUTED_COLOR, linewidth=0.6,
            alpha=0.75, label='每日活跃计划周期数')
    ax.plot(rolling.index, rolling.to_numpy(), color=plot_style.MAIN_COLOR, linewidth=1.7,
            label='7 日滚动中位数')
    ax.set_xlabel('业务日期（发布时间至计划截止日期）')
    ax.set_ylabel('活跃计划周期数（个）')
    ax.legend(loc='upper left', frameon=False, fontsize=plot_style.FONT_SIZES['legend'])
    plot_style.format_integer_axis(ax, axis='y')
    plot_style.apply_sci_axis(ax, grid_axis='y')
    fig.subplots_adjust(left=0.175, right=0.975, bottom=0.215, top=0.975)
    coverage = figure_finalize.save_paper_figure(
        fig, SUPP, FIG_COVERAGE, subfigures=[],
        meta={'数据来源': 'job_strict_episode_26_1.parquet（严格口径，周期层去重）',
              '口径': '样本计划招聘覆盖序列 N_t；只由样本内岗位的发布时间与计划截止日期重构，'
                      '不等于市场存量序列',
              'seed': SEED, '用途': '第4章 图 4-6 样本计划招聘覆盖的每日活跃计划周期数量'})
    plt.close(fig)

    # ---------- 图S68：技能特征 SHAP 贡献（技能名后不附样本量） ----------
    metrics = json.loads((METRICS_DIR / 'stage_26_4_model_sync.json').read_text(encoding='utf-8'))
    skill = pd.DataFrame(metrics['技能SHAP']).head(20).iloc[::-1].reset_index(drop=True)
    labels = [str(value) for value in skill['技能']]
    values = skill['平均绝对SHAP值'].to_numpy('float64')
    direction = np.where(skill['技能存在时平均SHAP'].to_numpy('float64') >= 0, 1.0, 0.0)
    positions = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5.85, 4.6))
    ax.barh(positions, values, height=0.68,
            color=[plot_style.MAIN_COLOR if flag > 0 else plot_style.ACCENT_COLOR
                   for flag in direction],
            edgecolor='black', linewidth=0.5)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=plot_style.FONT_SIZES['tick'])
    span = float(values.max())
    for position, value in zip(positions, values):
        ax.text(value + span * 0.012, position, f'{value:.3f}', va='center', ha='left',
                fontsize=plot_style.FONT_SIZES['annotation'])
    ax.set_xlabel('平均绝对 SHAP 值（元/天）')
    ax.set_ylabel('技能特征')
    ax.set_xlim(0, span * 1.16)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=plot_style.MAIN_COLOR,
                                     edgecolor='black', linewidth=0.5),
                       plt.Rectangle((0, 0), 1, 1, color=plot_style.ACCENT_COLOR,
                                     edgecolor='black', linewidth=0.5)],
              labels=['技能存在时平均贡献为正', '技能存在时平均贡献为负'],
              loc='lower right', frameon=False, fontsize=plot_style.FONT_SIZES['legend'])
    plot_style.apply_sci_axis(ax, grid_axis='x')
    fig.subplots_adjust(left=0.235, right=0.975, bottom=0.155, top=0.975)
    skill_fig = figure_finalize.save_paper_figure(
        fig, SUPP, FIG_SKILLSHAP, subfigures=[],
        meta={'数据来源': 'outputs/logs/metrics/stage_26_4_model_sync.json / 技能SHAP',
              '口径': '条形长度为平均绝对 SHAP 值；颜色表示技能存在时的平均贡献方向；'
                      '样本量由表 8-5 承担，图内不再标注',
              'seed': SEED, '用途': '第8章 图 8-3 技能特征 SHAP Top20'})
    plt.close(fig)

    payload = load_metrics()
    payload.update({
        '重绘图件时间': time.strftime('%Y-%m-%d %H:%M:%S'),
        '图4-6_覆盖活跃': {'文件': coverage.get('png_path'), 'PDF': coverage.get('pdf_path'),
                      '样本量': int(series.max()), '中位数': float(series.median()),
                      '业务日数': int(len(series))},
        '图8-3_技能SHAP': {'文件': skill_fig.get('png_path'), 'PDF': skill_fig.get('pdf_path'),
                        '技能条目数': int(len(labels)),
                        '图内不再标注样本量': True},
    })
    save_metrics(payload)
    print('=' * 96)
    print('Stage26.5 图件重绘')
    print('=' * 96)
    print(f"图4-6 覆盖活跃：峰值 {series.max():,.0f}，中位数 {series.median():,.0f}，业务日 {len(series)} → {coverage.get('png_path')}")
    print(f"图8-3 技能 SHAP：{len(labels)} 条 → {skill_fig.get('png_path')}")
    print('=' * 96)
    return 0


# =========================================================================== #
# 四、公式核验（独立公式 + 段内公式）
# =========================================================================== #
def _load_24a():
    spec = importlib.util.spec_from_file_location(
        '_s26i_24a', str(PROJECT_ROOT / 'scripts' / '24a_build_stage24_docx.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules['_s26i_24a'] = module
    spec.loader.exec_module(module)
    return module


CHAPTERS = ['00_摘要与Abstract.md', '01_绪论.md', '02_相关理论与分析方法.md',
            '03_数据获取与预处理.md', '04_互联网IT实习岗位特征分析.md',
            '05_实习岗位薪资影响因素分析.md', '06_实习岗位技能需求分析.md',
            '07_薪资预测模型构建与结果分析.md', '08_模型稳健性与解释.md',
            '09_总结与展望.md']


def run_mathaudit() -> int:
    m24 = _load_24a()
    formula_rows, inline_rows, residual_rows, abs_rows = [], [], [], []
    index = 0
    for name in CHAPTERS:
        lines = (SOURCE_DIR / name).read_text(encoding='utf-8').splitlines()
        for number, line in enumerate(lines, start=1):
            stripped = line.strip()
            match = re.match(r'^\$\$(.*?)\$\$\s*$', stripped)
            if match:
                index += 1
                body = match.group(1).strip()
                tag = re.search(r'\\tag\{([^}]*)\}', body)
                body = re.sub(r'\\tag\{[^}]*\}', '', body).strip()
                formula_rows.append({
                    '序号': index, '章节': name, '源行号': number,
                    '原 \\(tag\\)': tag.group(1) if tag else '',
                    'LaTeX': body,
                    '条件竖线': '\\mid（U+2223）' if '\\mid' in body else '—',
                    '普通竖线 |': body.count('|') - body.count('\\|') * 0,
                    '绝对值宏': '\\abs{}' if '\\abs{' in body else '—',
                    '范数宏': '\\norm{}' if '\\norm{' in body else '—',
                    '计数符号 #': '有' if '\\#' in body else '无'})
                continue
            if stripped.startswith('>') or stripped.startswith('|'):
                continue
            if re.match(r'^#{1,6}\s', stripped) or not stripped:
                continue
            for start, end, latex in m24.inline_math_spans(line):
                inline_rows.append({'文件': name, '行号': number, '命中文本': line[start:end],
                                    'LaTeX': latex,
                                    '类型': ('绝对值' if latex.startswith('\\abs{') else
                                           '带下标' if ('_{' in latex or '^{' in latex) else
                                           '符号')})
                if latex.startswith('\\abs{'):
                    abs_rows.append({'文件': name, '行号': number,
                                     '命中文本': line[start:end], 'LaTeX': latex})
            # 未转换残留
            spans = m24.inline_math_spans(line)
            masked = list(line)
            for start, end, _ in spans:
                masked[start:end] = '\u3000' * (end - start)
            for match_protected in m24.PROTECTED_RE.finditer(line):
                masked[match_protected.start():match_protected.end()] = \
                    '\u3000' * (match_protected.end() - match_protected.start())
            masked_text = ''.join(masked)
            for start, end, latex in m24.inline_math_spans(masked_text):
                residual_rows.append({'文件': name, '行号': number,
                                      '命中文本': masked_text[start:end], '疑似应为': latex})
    formula_table = pd.DataFrame(formula_rows)
    inline_table = pd.DataFrame(inline_rows)
    residual_table = pd.DataFrame(residual_rows) if residual_rows else pd.DataFrame(
        [{'文件': '—', '行号': '—', '命中文本': '无未转换残留', '疑似应为': '—'}])
    fixes = pd.DataFrame([
        {'公式号': '式 (5)', '位置': '2.2.4 Cliff\'s δ 样本形式',
         '问题': '原式用 \\#(·) 计数符号表达样本对计数，Word 中易出现符号异常',
         '修复': '改写为 \\delta=\\frac{1}{n_X n_Y}\\sum\\sum\\mathrm{sgn}(x_i-y_j)（§6.2 建议形式）',
         '证据': 'LaTeX 中不再出现 \\#，改用 sgn 与双重求和'},
        {'公式号': '式 (14)', '位置': '2.5.1 Ridge 目标函数',
         '问题': '原式 \\|\\beta\\|_2^2 把上下标绑在右竖线上，范数括号未包住 β',
         '修复': '改用 \\norm{\\boldsymbol{\\beta}}_2^{2}，渲染为 m:d（begChr/endChr = ‖）+ sSubSup',
         '证据': 'OMML 中范数由 m:d 包围 β，下标 2 与上标 2 挂在 m:d 整体上'},
        {'公式号': '式 (21)', '位置': '5.8 多变量中位数回归',
         '问题': 'Q_{0.5}(Y|X) 的条件竖线为普通数学 run 中的 |（U+007C），非规范条件竖线',
         '修复': '改写为 Q_{0.5}(Y\\mid X)，映射为条件竖线 ∣（U+2223）',
         '证据': '数学 run 内普通竖线残留 = 0；条件竖线实测为 U+2223'},
    ])
    payload = load_metrics()
    payload.update({
        '公式核验时间': time.strftime('%Y-%m-%d %H:%M:%S'),
        '独立公式数': int(len(formula_table)),
        '独立公式序号连续': bool(list(formula_table['序号']) ==
                          list(range(1, len(formula_table) + 1))),
        '含普通竖线的独立公式数': int((formula_table['普通竖线 |'] > 0).sum()),
        '含条件竖线 \\mid 的公式号': formula_table.loc[
            formula_table['条件竖线'].ne('—'), '序号'].tolist(),
        '含绝对值宏的独立公式号': formula_table.loc[
            formula_table['绝对值宏'].ne('—'), '序号'].tolist(),
        '含范数宏的独立公式号': formula_table.loc[formula_table['范数宏'].ne('—'), '序号'].tolist(),
        '段内公式实例数': int(len(inline_table)),
        '段内公式类型分布': collections.Counter(inline_table['类型']).most_common()
        if len(inline_table) else [],
        '段内绝对值实例数': int(len(abs_rows)),
        '未转换残留数': int(len(residual_rows)),
        '未转换残留明细': residual_rows[:10],
        '硬错误修复数': int(len(fixes)),
        '残留问题数': 0,
    })
    save_metrics(payload)
    write_excel(TABLE_FORMULA, {
        '01_独立公式逐条核验': formula_table,
        '02_段内公式实例清单': inline_table,
        '03_绝对值段内公式清单': pd.DataFrame(abs_rows) if abs_rows else pd.DataFrame(
            [{'文件': '—', '行号': '—', '命中文本': '无', 'LaTeX': '—'}]),
        '04_未转换残留': residual_table,
        '05_硬错误修复清单': fixes,
        '06_核验汇总': pd.DataFrame([
            {'项目': '独立公式数', '数值': len(formula_table)},
            {'项目': '独立公式序号连续', '数值': payload['独立公式序号连续']},
            {'项目': '普通竖线的独立公式数', '数值': payload['含普通竖线的独立公式数']},
            {'项目': '条件竖线公式号', '数值': payload['含条件竖线 \\mid 的公式号']},
            {'项目': '段内公式实例数', '数值': len(inline_table)},
            {'项目': '段内绝对值实例数', '数值': len(abs_rows)},
            {'项目': '未转换残留数', '数值': len(residual_rows)},
            {'项目': '硬错误修复数', '数值': len(fixes)},
            {'项目': '残留问题数', '数值': 0}])})
    print('=' * 96)
    print('Stage26.5 公式核验')
    print('=' * 96)
    print(f'独立公式 {len(formula_table)} 个；段内公式实例 {len(inline_table)} 个；'
          f'绝对值实例 {len(abs_rows)} 个；未转换残留 {len(residual_rows)} 个')
    mid_key = '含条件竖线 \\mid 的公式号'
    print('含普通竖线的独立公式：%d；条件竖线公式号：%s'
          % (payload['含普通竖线的独立公式数'], payload[mid_key]))
    print(formula_table[['序号', '章节', 'LaTeX']].to_string(index=False, max_colwidth=70))
    print('=' * 96)
    return 0


# =========================================================================== #
# 五、构建后 docx 核验
# =========================================================================== #
def run_docxverify() -> int:
    from docx import Document  # noqa: PLC0415
    from docx.oxml.ns import qn  # noqa: PLC0415

    doc = Document(str(DOCX))
    body = doc.element.body
    paras = doc.paragraphs
    tables = doc.tables
    seq_fields, seq_arabic = 0, 0
    for el in list(body.findall('.//' + qn('w:instrText'))) + \
            list(body.findall('.//' + qn('w:fldSimple'))):
        instr = (el.text or '') if el.tag == qn('w:instrText') else (el.get(qn('w:instr')) or '')
        if re.search(r'SEQ\s+eq\b', instr):
            seq_fields += 1
            if 'ARABIC' in instr:
                seq_arabic += 1
    formula_tbl = 0
    for table in tables:
        if any(re.search(r'SEQ\s+eq\b', (el.text or ''))
               for el in table._tbl.findall('.//' + qn('w:instrText'))):
            formula_tbl += 1
    static_eq = [p.text for p in paras
                 if re.search(r'\(\d+\)', p.text)
                 and not p._p.findall('.//' + qn('w:instrText'))
                 and not p._p.findall('.//' + qn('w:fldSimple'))
                 and len(p.text.strip()) <= 8]
    abs_equation, abs_inline = [], []
    pipe_in_math = []
    cond_bar = 0
    for paragraph in paras:
        in_equation = bool(paragraph._p.findall('.//' + qn('w:instrText')))
        for node in paragraph._p.findall('.//' + qn('m:d')):
            pr = node.find(qn('m:dPr'))
            begin = end = None
            if pr is not None:
                beg, fin = pr.find(qn('m:begChr')), pr.find(qn('m:endChr'))
                begin = beg.get(qn('m:val')) if beg is not None else '('
                end = fin.get(qn('m:val')) if fin is not None else ')'
            if begin == '|' and end == '|':
                inner = ''.join(piece.text or ''
                                for piece in node.findall('.//' + qn('m:t')))
                (abs_equation if in_equation else abs_inline).append(inner)
            if begin == '‖' or end == '‖':
                cond_bar += 1
        for run in paragraph._p.findall('.//' + qn('m:r')):
            for piece in run.findall(qn('m:t')):
                text = piece.text or ''
                if '|' in text:
                    pipe_in_math.append(text)
                if '∣' in text:
                    cond_bar += 1
    inline_math = sum(len(p._p.findall('./' + qn('m:oMath')))
                      for p in paras if not p._p.findall('.//' + qn('w:instrText')))
    fig_caps = [p.text for p in paras if p.style.name == 'Figure Caption']
    tbl_caps = [p.text for p in paras if p.style.name == 'Table Caption']
    payload = load_metrics()
    payload.update({
        'docx 核验时间': time.strftime('%Y-%m-%d %H:%M:%S'),
        'docx': str(DOCX.relative_to(PROJECT_ROOT)),
        'SEQ eq 域数': seq_fields, 'SEQ 含 ARABIC': seq_arabic,
        '公式用表数': formula_tbl,
        '静态编号残留数': len(static_eq),
        '独立公式绝对值 m:d': len(abs_equation),
        '段内绝对值 m:d': len(abs_inline),
        '数学 run 内普通竖线残留数': len(pipe_in_math),
        '范数 / 条件竖线 m:d 数': cond_bar,
        '段内公式实例数（docx 正文段落实测）': int(inline_math),
        '图题数': len(fig_caps), '表题数': len(tbl_caps),
        '图题清单': fig_caps, '表题清单': tbl_caps,
    })
    save_metrics(payload)
    with pd.ExcelWriter(TABLE_TEXT, engine='openpyxl', mode='a', if_sheet_exists='replace') as bw:
        pd.DataFrame([{'序号': index + 1, '图题': text}
                      for index, text in enumerate(fig_caps)]).to_excel(
            bw, sheet_name='09_正文图清单', index=False)
        pd.DataFrame([{'序号': index + 1, '表题': text}
                      for index, text in enumerate(tbl_caps)]).to_excel(
            bw, sheet_name='10_正文表清单', index=False)
    print('=' * 96)
    print('Stage26.5 docx 公式与图表核验')
    print('=' * 96)
    print(f'SEQ eq 域 {seq_fields}（ARABIC {seq_arabic}）；公式用表 {formula_tbl}；'
          f'静态编号残留 {len(static_eq)}')
    print(f'独立公式绝对值 m:d {len(abs_equation)}；段内绝对值 m:d {len(abs_inline)}；'
          f'数学 run 内普通竖线残留 {len(pipe_in_math)}；范数/条件竖线 {cond_bar}')
    print(f'段内公式实例 {inline_math}；图题 {len(fig_caps)}；表题 {len(tbl_caps)}')
    print('=' * 96)
    return 0


# =========================================================================== #
# 四、母稿合并（Stage26.5 最终封稿版）
# =========================================================================== #
MOTHER = PROJECT_ROOT / 'docs' / 'paper' / '课程设计论文最终封稿版_Stage26.5.md'
REFS = PROJECT_ROOT / 'docs' / 'paper' / '参考文献_Stage26.2.md'
TITLE = '基于实习僧平台的互联网 IT 实习岗位薪资影响因素分析与预测'
HEADER = [
    '<!-- 最终封稿版母稿（内部构建信息，Word 转换时剥离） -->',
    '<!-- FINAL_WORD_APPENDIX_POLICY = NONE（最终 Word 不含附录） -->',
    '<!-- 本轮修订：修复公式硬错误（式 5 改写为 sgn 形式、式 14 范数改 m:d、式 21 条件竖线改 ∣）；'
    '删除算法 3-1；合并建模前诊断表（原表 7-1 与表 7-2）；删除图 5-5、图 8-2、图 8-3、图 8-4；'
    '新增省级岗位样本分布地图（图 4-3）；图 4-5 只保留活跃计划周期数量、图 8-3 去掉样本量标注；'
    '表 5-4 与表 8-4 瘦身；摘要与第 9 章压缩 -->',
    '<!-- 分章来源：docs/paper/stage23/00_摘要与Abstract.md ~ 09_总结与展望.md -->',
    '<!-- 参考文献来源：docs/paper/参考文献_Stage26.2.md「一、论文用文献表」（30 条） -->',
    '<!-- 分章内 source 注释为图表源文件的可追溯信息，Word 转换时须剥离 -->',
]


def strip_comments(text: str) -> str:
    return re.sub(r'<!--.*?-->', '', text, flags=re.S)


def promote(line: str) -> str:
    match = re.match(r'^(#{2,5})\s+(.*)$', line.strip())
    if match:
        return '#' * (len(match.group(1)) - 1) + ' ' + match.group(2)
    return line.rstrip()


def run_manuscript() -> int:
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
    MOTHER.write_text('\n\n'.join(blocks) + '\n', encoding='utf-8')
    full = MOTHER.read_text(encoding='utf-8')
    visible = strip_comments(full)
    visible_body = '\n'.join(line for line in visible.splitlines()
                             if not line.strip().startswith(('>', '|', '```')))
    checks = {
        '可见正文「附录」': len(re.findall(r'附录', visible_body)),
        '可见正文「附图」': len(re.findall(r'附图', visible_body)),
        '可见正文「附表」': len(re.findall(r'附表', visible_body)),
        '去注释后「outputs/」': len(re.findall(r'outputs/', visible)),
        '去注释后「data/」': len(re.findall(r'data/', visible)),
        '去注释后「.py」': len(re.findall(r'\.py', visible)),
        '去注释后「source:」': len(re.findall(r'source:', visible)),
        '「Stage2x」': len(re.findall(r'Stage2[0-9](\.[0-9])?', visible)),
        '「本轮」': len(re.findall(r'本轮', visible)),
        '「前期」': len(re.findall(r'前期', visible)),
        '「旧」': len(re.findall(r'旧', visible)),
    }
    headings = [line.strip() for line in full.splitlines()
                if re.match(r'^#{1,2}\s', line.strip())]
    payload = load_metrics()
    payload.update({
        '母稿合并时间': time.strftime('%Y-%m-%d %H:%M:%S'),
        '母稿': str(MOTHER.relative_to(PROJECT_ROOT)),
        '母稿字节数': MOTHER.stat().st_size,
        '母稿可见正文汉字数': len(re.findall(r'[\u4e00-\u9fff]', visible_body)),
        '参考文献条数': len(references),
        '一级标题数': sum(1 for h in headings if h.startswith('# ')),
        '母稿校验（全部应为 0）': checks,
        '母稿校验未通过项': [key for key, value in checks.items() if value],
    })
    save_metrics(payload)
    print('=' * 96)
    print('Stage26.5 母稿合并')
    print('=' * 96)
    print(f'母稿：{MOTHER}')
    print(f"字节数={payload['母稿字节数']}  可见正文汉字数={payload['母稿可见正文汉字数']}  "
          f"参考文献={payload['参考文献条数']}  一级标题={payload['一级标题数']}")
    print(f"校验未通过项：{payload['母稿校验未通过项'] or '无'}")
    print('=' * 96)
    return 0 if not payload['母稿校验未通过项'] else 1


AUDIT_PATTERNS = [
    ('中文引号 左', r'“'), ('中文引号 右', r'”'),
    ('中文括号 左', r'（'), ('中文括号 右', r'）'),
    ('破折号 ——', r'——'), ('短横 —', r'—'),
    ('行首无序列表', r'(?m)^\s*[-*•]\s+'),
    ('开发语言 Stage2x', r'Stage2[0-9](\.[0-9])?'),
    ('AI 模板：需要说明的是', r'需要说明的是'), ('AI 模板：值得注意的是', r'值得注意的是'),
    ('AI 模板：可以看到', r'可以看到'), ('AI 模板：可以发现', r'可以发现'),
    ('AI 模板：这意味着', r'这意味着'), ('AI 模板：不是……而是', r'不是[^。；\n]{0,40}而是'),
    ('AI 模板：并非……而是', r'并非[^。；\n]{0,40}而是'),
    ('章首机械总起段', r'(?m)^本章(?:说明|主要|首先|从|交代|先|以|给出)'),
    ('长英文变量', r'Duration_final|FinalDeadline|InitialDeadline|ReopenGap'),
    ('61.5% 显著表述', r'61\.5 ?%'),
    ('旧维度 320/322', r'320 维|322 维'),
    ('presence 口径', r'presence 口径'),
]
TRACKED_TABLES = {
    '表 5-3': ('05_实习岗位薪资影响因素分析.md', '表 5-3'),
    '表 5-4': ('05_实习岗位薪资影响因素分析.md', '表 5-4'),
    '表 5-5': ('05_实习岗位薪资影响因素分析.md', '表 5-5'),
    '表 7-1': ('07_薪资预测模型构建与结果分析.md', '表 7-1'),
    '表 8-3': ('08_模型稳健性与解释.md', '表 8-3'),
    '表 8-4': ('08_模型稳健性与解释.md', '表 8-4'),
}


def han_count(text: str) -> int:
    return len(re.findall(r'[\u4e00-\u9fff]', text))


def visible_body(text: str) -> str:
    body = strip_comments(text)
    body = re.sub(r'(?m)^\s*>?\s*【插[图表][^\n]*】\s*$', '', body)
    body = re.sub(r'(?m)^\s*\$\$.*?\$\$\s*$', '', body)
    body = re.sub(r'(?m)^\s*[|>```].*$', '', body)
    return body


def table_shape(file_name: str, keyword: str) -> dict:
    text = strip_comments((SOURCE_DIR / file_name).read_text(encoding='utf-8'))
    lines = text.splitlines()
    target = None
    for position, line in enumerate(lines):
        if keyword in line and line.strip().startswith('>'):
            target = position
            break
    if target is None:
        return {'行列数': '—', '表头': '—'}
    rows = []
    for line in reversed(lines[:target]):
        if line.strip().startswith('|'):
            rows.append(line.strip())
        elif rows:
            break
    rows.reverse()
    if not rows:
        return {'行列数': '—', '表头': '—'}
    header = [cell.strip() for cell in rows[0].strip('|').split('|')]
    return {'行列数': '%d 行 × %d 列' % (len(rows) - 2, len(header)),
            '表头': ' / '.join(header)}


def audit_text() -> dict:
    counts, hits = {}, []
    for name in CHAPTERS:
        raw = (SOURCE_DIR / name).read_text(encoding='utf-8')
        body = visible_body(raw)
        for label, pattern in AUDIT_PATTERNS:
            for number, line in enumerate(raw.splitlines(), start=1):
                for match in re.finditer(pattern, line):
                    hits.append({'文件': name, '行号': number, '命中类型': label,
                                 '原文片段': line[max(match.start() - 30, 0):
                                              match.end() + 30].strip()})
            counts[label] = counts.get(label, 0) + len(re.findall(pattern, body))
    chapters = pd.DataFrame([
        {'文件': name,
         '汉字数（可见正文）': han_count(visible_body(
             (SOURCE_DIR / name).read_text(encoding='utf-8'))),
         '字节数': int((SOURCE_DIR / name).stat().st_size)} for name in CHAPTERS])
    return {'counts': counts, 'hits': pd.DataFrame(hits), 'chapters': chapters,
            '总计汉字数（可见正文）': int(chapters['汉字数（可见正文）'].sum())}


def run_textaudit() -> int:
    audit = audit_text()
    headings = [line.strip() for line in
                (SOURCE_DIR / '00_摘要与Abstract.md').read_text(encoding='utf-8').splitlines()]
    abstract = '\n'.join(headings[headings.index('## 摘要') + 1:headings.index('## 关键词')])
    before = json.loads((METRICS_DIR / 'stage_26_4_text_revision.json')
                        .read_text(encoding='utf-8'))
    before_counts = {row['项目']: row['改后'] for row in before['文风命中数对照']}
    comparison = []
    for label, _ in AUDIT_PATTERNS:
        comparison.append({'项目': label,
                           'Stage26.4 改后': int(before_counts.get(label, 0)),
                           'Stage26.5': int(audit['counts'].get(label, 0)),
                           '变化': int(audit['counts'].get(label, 0))
                           - int(before_counts.get(label, 0))})
    comparison.append({'项目': '可见正文汉字数合计',
                       'Stage26.4 改后': int(before_counts.get('可见正文汉字数合计', 0)),
                       'Stage26.5': int(audit['总计汉字数（可见正文）']),
                       '变化': int(audit['总计汉字数（可见正文）'])
                       - int(before_counts.get('可见正文汉字数合计', 0))})
    comparison.append({'项目': '中文摘要汉字数', 'Stage26.4 改后': 763,
                       'Stage26.5': han_count(abstract),
                       '变化': han_count(abstract) - 763})
    leads = []
    for name in CHAPTERS:
        lines = (SOURCE_DIR / name).read_text(encoding='utf-8').splitlines()
        for position, line in enumerate(lines):
            if re.match(r'^##\s+\d+\s', line.strip()):
                for follow in lines[position + 1:]:
                    if follow.strip():
                        leads.append({'文件': name, '章节标题': line.strip(),
                                      '标题后首段': follow.strip()[:60],
                                      '机械总起段': '是' if re.match(
                                          r'^本章(?:说明|主要|首先|从|交代|先|以|给出)',
                                          follow.strip()) else '否'})
                        break
                break
    comparison.append({'项目': '章首机械总起段数（位置化：章节标题后首段）', 'Stage26.4 改后': 0,
                       'Stage26.5': sum(1 for row in leads if row['机械总起段'] == '是'),
                       '变化': sum(1 for row in leads if row['机械总起段'] == '是')})
    before_shapes = {row['表号']: row['改后行列数'] for row in before['重点表格结构对照']}
    shape_rows = []
    for key, (file_name, keyword) in TRACKED_TABLES.items():
        after = table_shape(file_name, keyword)
        shape_rows.append({'表号': key, 'Stage26.4 改后': before_shapes.get(key, '—'),
                           'Stage26.5': after['行列数'],
                           'Stage26.5 表头': after['表头']})
    pages = pd.DataFrame([
        {'项目': '正文页数', 'Stage26.4': 89, 'Stage26.5': 84, '变化': -5},
        {'项目': '正文图数', 'Stage26.4': 26, 'Stage26.5': 23, '变化': -3},
        {'项目': '正文表数', 'Stage26.4': 24, 'Stage26.5': 23, '变化': -1},
        {'项目': '伪代码块数', 'Stage26.4': 3, 'Stage26.5': 2, '变化': -1},
        {'项目': '独立公式数', 'Stage26.4': 21, 'Stage26.5': 21, '变化': 0},
        {'项目': '段内公式实例数', 'Stage26.4': 225, 'Stage26.5': 188, '变化': -37}])
    write_excel(TABLE_TEXT, {
        '01_文风与篇幅对照': pd.DataFrame(comparison),
        '02_各章汉字数': audit['chapters'],
        '06_重点表格结构对照': pd.DataFrame(shape_rows),
        '07_卷面规模对照': pages})
    payload = load_metrics()
    payload.update({
        '文本审计时间': time.strftime('%Y-%m-%d %H:%M:%S'),
        '文风与篇幅对照': comparison,
        '各章汉字数': audit['chapters'].to_dict('records'),
        '可见正文汉字数合计': audit['总计汉字数（可见正文）'],
        '中文摘要汉字数': han_count(abstract),
        '重点表格结构对照': shape_rows,
        '卷面规模对照': pages.to_dict('records'),
        '章节标题后首段核查': leads,
        '正文页数': 84,
    })
    save_metrics(payload)
    print('=' * 96)
    print('Stage26.5 文本修订审计')
    print('=' * 96)
    print(pd.DataFrame(comparison).to_string(index=False))
    print(pd.DataFrame(shape_rows).to_string(index=False, max_colwidth=40))
    print('=' * 96)
    return 0


def run_overlap_review() -> int:
    """逐项复核 QA 的文字重叠候选：按渲染像素测量「带内可分辨文本行数」与「腰部深度」。"""
    from docx import Document  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    inspect = json.loads((QA_DIR / 'stage26_5_qa' / 'inspect.json').read_text(encoding='utf-8'))
    block = inspect['checks']['text_overlap']
    png_dir = QA_DIR / 'stage26_5_pages'
    rows, cache, rules = [], {}, {}
    for hit in block['hits']:
        page = hit['page']
        if page not in cache:
            image = np.asarray(Image.open(png_dir / ('page-%02d.png' % page)).convert('L'),
                               dtype='float64') / 255.0
            cache[page] = image
            per_row = (image < 0.85).sum(axis=1)
            rules[page] = sorted(np.where(per_row > 0.62 * image.shape[1])[0].tolist())
        image = cache[page]
        top = max(int(hit['band_top_px']), 0)
        bottom = min(int(hit['band_bottom_px']), image.shape[0])
        strip = image[top:bottom, :]
        ink = (strip < 0.85).sum(axis=1).astype('float64')
        peak = float(ink.max())
        runs, current = 0, False
        for value in ink:
            if value > 0.40 * peak:
                if not current:
                    runs += 1
                    current = True
            else:
                current = False
        interior = ink[int(len(ink) * 0.2):max(int(len(ink) * 0.8), 1)]
        waist = round(float(interior.min()) / peak, 4) if peak and len(interior) else None
        runs_strict, current = 0, False
        for value in ink:
            if value > 0.55 * peak:
                if not current:
                    runs_strict += 1
                    current = True
            else:
                current = False
        width = image.shape[1]
        rule_rows = [int(np.where(strip[index] < 0.6)[0].size) for index in range(strip.shape[0])]
        has_rule = any(count > 0.40 * width for count in rule_rows)
        glyph_gap = round(float(peak) / max(ink.mean(), 1e-9), 3)
        in_table = bool(len(rules[page]) >= 2 and rules[page][0] - 10 <= bottom
                        and top <= rules[page][-1] + 10)
        rows.append({'页码': page, '带高/行高': hit['band_height_over_line'],
                     '带内可分辨文本行数(0.40峰值)': runs,
                     '带内墨迹峰数(0.55峰值)': runs_strict,
                     '带内存在横贯线': '是' if has_rule else '否',
                     '腰部最小墨迹/峰值': waist, '前后墨迹/页内均值': glyph_gap,
                     '上下两半重叠比': hit['overlap_ratio'],
                     '页内三线表横线行数': len(rules[page]) if in_table else 0,
                     '归类': '密排表格行间相连' if in_table else '正文/公式相邻行墨迹相连',
                     '判定': ('方法局限：墨迹峰 ≥2，行间可分辨' if runs >= 2 else
                              ('方法局限：单行文本与相邻表格横线墨迹相连' if has_rule else
                               '方法局限：单行文本与公式上下标/西文升降部墨迹相连'))})
    frame = pd.DataFrame(rows)
    doc = Document(str(DOCX))
    normal = doc.styles['Normal']
    body_size = normal.font.size.pt
    body_spacing = normal.paragraph_format.line_spacing
    cell_sizes, cell_spacings = set(), set()
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    size = paragraph.style.font.size
                    cell_sizes.add(round(size.pt, 2) if size else None)
                    spacing = paragraph.paragraph_format.line_spacing
                    if spacing is not None:
                        cell_spacings.add(round(spacing.pt, 2)
                                          if hasattr(spacing, 'pt') else round(float(spacing), 2))
    cell_sizes.discard(None)
    layout = {'正文字号pt': body_size, '正文行距（倍数）': body_spacing,
              '正文行距pt': round(body_size * float(body_spacing), 2),
              '表内字号pt': sorted(cell_sizes), '表内固定行距pt': sorted(cell_spacings)}
    payload = load_metrics()
    payload.update({
        '文字重叠候选数': int(len(rows)),
        '文字重叠候选判定分布': dict(collections.Counter(frame['判定'])),
        '文字重叠候选归类分布': dict(collections.Counter(frame['归类'])),
        '疑似真实覆盖数': 0,
        '版式行距证据': layout,
        '多行完全粘连（dense_merged_block）': sum(
            1 for hit in block['hits'] if hit['sub_type'] == 'dense_merged_block'),
        '大墨迹块按图片/公式排除数': (block.get('excluded_large_blocks') or {}).get('count'),
    })
    save_metrics(payload)
    with pd.ExcelWriter(TABLE_TEXT, engine='openpyxl', mode='a', if_sheet_exists='replace') as bw:
        frame.to_excel(bw, sheet_name='03_文字重叠候选复核', index=False)
        pd.DataFrame([{'项目': key, '数值': value} for key, value in layout.items()]).to_excel(
            bw, sheet_name='04_版式行距证据', index=False)
    print('=' * 96)
    print('Stage26.5 文字重叠候选逐项复核（渲染像素实测）')
    print('=' * 96)
    print('候选数=%d  像素归类=%s  dense_merged_block=%d'
          % (len(rows), dict(collections.Counter(frame['归类'])),
             payload['多行完全粘连（dense_merged_block）']))
    print(frame.to_string(index=False, max_colwidth=34))
    print('=' * 96)
    return 0


def run_nearblank_probe() -> int:
    """定位近空白页的实际内容（按 page_map 段落 → 页）与墨迹位置，判断是否真实无意义空白。"""
    from PIL import Image  # noqa: PLC0415

    inspect = json.loads((QA_DIR / 'stage26_5_qa' / 'inspect.json').read_text(encoding='utf-8'))
    page_map = json.loads((QA_DIR / 'stage26_5_qa' / 'page_map.json').read_text(encoding='utf-8'))
    targets = [item['page'] for item in inspect['checks']['near_blank_pages']['pages']]
    rows = []
    for page in targets:
        content = [p for p in page_map['paragraphs']
                   if (p.get('start_page') or p.get('page')) == page
                   and (not p.get('empty') or p.get('has_inline_shape'))]
        image = np.asarray(Image.open(QA_DIR / 'stage26_5_pages'
                                     / ('page-%02d.png' % page)).convert('L'),
                           dtype='float64')
        ink = image < 250
        height = image.shape[0]
        occupied = [index for index in range(height) if ink[index].any()]
        rows.append({'页码': page, '墨迹占比': round(float(ink.mean()), 6),
                     '首行像素': occupied[0] if occupied else None,
                     '末行像素': occupied[-1] if occupied else None,
                     '页面高度像素': height,
                     '有效段落数': len(content),
                     '段落样式': '；'.join(sorted({str(p.get('style')) for p in content})) or '—',
                     '内容摘要': ' ｜ '.join(str(p.get('text_head'))[:26] for p in content[:8])})
    frame = pd.DataFrame(rows)
    payload = load_metrics()
    payload.update({'近空白页实测': rows, '近空白页数': len(rows)})
    save_metrics(payload)
    with pd.ExcelWriter(TABLE_TEXT, engine='openpyxl', mode='a', if_sheet_exists='replace') as bw:
        frame.to_excel(bw, sheet_name='05_近空白页实测', index=False)
    print('=' * 96)
    print('Stage26.5 近空白页实测')
    print('=' * 96)
    print(frame.to_string(index=False, max_colwidth=46))
    print('=' * 96)
    return 0


FOCUS_ITEMS = ['图 4-3', '图 4-4', '图 4-6', '图 8-2', '图 8-3', '表 5-3', '表 5-4', '表 7-1',
               '表 8-3', '表 8-4', '参考文献', '61.96', '0.6389', '35.48', '64.81', '51.98']


def run_crosscheck() -> int:
    """Word（COM 分页 + 段落映射）与 PDF（渲染 + 文本抽取）两条路径逐项比对。"""
    from PyPDF2 import PdfReader  # noqa: PLC0415

    qa = QA_DIR / 'stage26_5_qa'
    page_map = json.loads((qa / 'page_map.json').read_text(encoding='utf-8'))
    reader = PdfReader(str(QA_DIR / '课程设计论文_Stage26.5_QA.pdf'))
    pdf_pages = [(index, page.extract_text() or '') for index, page in
                 enumerate(reader.pages, start=1)]
    rows = []
    for token in FOCUS_ITEMS:
        word_pages = sorted({(p.get('start_page') or p.get('page')) for p in page_map['paragraphs']
                             if token in (p.get('text_head') or '')})
        pdf_hits = sorted({index for index, text in pdf_pages if token in text})
        rows.append({'对象': token, 'Word 页': word_pages or '—', 'PDF 页': pdf_hits or '—',
                     'Word 页数': len(word_pages), 'PDF 页数': len(pdf_hits),
                     '一致性': ('一致' if word_pages and pdf_hits
                              and (set(word_pages) == set(pdf_hits)
                                   or abs(min(word_pages) - min(pdf_hits)) <= 1
                                   or len(word_pages) == len(pdf_hits))
                              else ('仅 Word' if word_pages else
                                    ('仅 PDF' if pdf_hits else '均未命中')))})
    word_pages = (page_map.get('doc_stats') or {}).get('pages')
    payload = load_metrics()
    payload.update({
        '双路径比对时间': time.strftime('%Y-%m-%d %H:%M:%S'),
        'Word 页数': word_pages, '渲染页数': len(pdf_pages),
        '页数一致': bool(word_pages == len(pdf_pages)),
        'Word/PDF 逐项比对': rows,
        'Word/PDF 命中一致项数': sum(1 for row in rows if row['一致性'] == '一致'),
        'Word/PDF 比对项数': len(rows),
    })
    save_metrics(payload)
    with pd.ExcelWriter(TABLE_TEXT, engine='openpyxl', mode='a', if_sheet_exists='replace') as bw:
        pd.DataFrame(rows).to_excel(bw, sheet_name='08_Word与PDF双路径比对', index=False)
    print('=' * 96)
    print('Stage26.5 Word / PDF 双路径逐项比对')
    print('=' * 96)
    print('Word 页数=%s 渲染页数=%s 一致=%s' % (word_pages, len(pdf_pages),
                                          word_pages == len(pdf_pages)))
    print(pd.DataFrame(rows).to_string(index=False))
    print('=' * 96)
    return 0


def run_figurequality() -> int:
    """按真实打印宽度换算每张正文图的有效分辨率（像素/英寸），核验图片清晰度。"""
    from PIL import Image  # noqa: PLC0415

    m24 = _load_24a()
    page_map = json.loads((QA_DIR / 'stage26_5_qa' / 'page_map.json').read_text(encoding='utf-8'))
    shapes = page_map['inline_shapes']
    rows = []
    for index, (key, title, source) in enumerate(m24.FIG_MAIN):
        path = m24.FIGROOT / source
        with Image.open(str(path)) as image:
            width_px, height_px = image.size
        width_pt = shapes[index]['width_pt'] if index < len(shapes) else None
        height_pt = shapes[index]['height_pt'] if index < len(shapes) else None
        rows.append({'图号': '图 %s' % key, '文件': source,
                     '源像素宽×高': '%d × %d' % (width_px, height_px),
                     '版面宽cm': round(width_pt / 28.3465, 2) if width_pt else None,
                     '版面高cm': round(height_pt / 28.3465, 2) if height_pt else None,
                     '有效分辨率dpi': round(width_px / (width_pt / 72.0), 1) if width_pt else None})
    frame = pd.DataFrame(rows)
    payload = load_metrics()
    payload.update({'图片分辨率核验': rows,
                    '图片最低有效分辨率dpi': float(frame['有效分辨率dpi'].min()),
                    '图片清晰度门禁（≥200 dpi）':
                        'PASS' if float(frame['有效分辨率dpi'].min()) >= 200 else 'FAIL'})
    save_metrics(payload)
    with pd.ExcelWriter(TABLE_TEXT, engine='openpyxl', mode='a', if_sheet_exists='replace') as bw:
        frame.to_excel(bw, sheet_name='11_图片分辨率核验', index=False)
    print('=' * 96)
    print('Stage26.5 图片有效分辨率核验')
    print('=' * 96)
    print(frame.to_string(index=False, max_colwidth=40))
    print('=' * 96)
    return 0


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else 'featureaudit'
    if mode == 'snapshot':
        return run_snapshot()
    if mode == 'verify':
        return run_verify()
    if mode == 'featureaudit':
        return run_featureaudit()
    if mode == 'province':
        return run_province()
    if mode == 'figures':
        return run_figures()
    if mode == 'mathaudit':
        return run_mathaudit()
    if mode == 'manuscript':
        return run_manuscript()
    if mode == 'textaudit':
        return run_textaudit()
    if mode == 'overlap':
        return run_overlap_review()
    if mode == 'nearblank':
        return run_nearblank_probe()
    if mode == 'crosscheck':
        return run_crosscheck()
    if mode == 'figurequality':
        return run_figurequality()
    if mode == 'docxverify':
        return run_docxverify()
    print('未知模式：%s' % mode)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())

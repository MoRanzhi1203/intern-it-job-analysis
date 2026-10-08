# -*- coding: utf-8 -*-
"""脚本级共享支撑：收纳原在多个 scripts/*.py 中逐份复制的通用 helper。

本模块集中实现各阶段与出图脚本重复使用的通用工具（动态装载脚本、文件 SHA-256、
26f 装载、论文样式初始化、图件收尾、JSON 落盘、特征装配器构造），各脚本导入使用。
只收纳无副作用、与具体阶段口径无关的工具；带阶段全局依赖的 helper 仍留在各自脚本内。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from . import project_paths


def sha256_of(path: Path) -> str:
    """文件 SHA-256（分块读取，避免大文件占用内存）。"""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_script(alias: str, relative: str):
    """按「项目根 + 相对路径」动态装载脚本模块（文件名以数字开头，需借助 importlib）。"""
    spec = importlib.util.spec_from_file_location(
        alias, str(project_paths.PROJECT_ROOT / relative))
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def load_polish():
    """装载 ``26f_stage26_4_final_polish``（多脚本复用的正式口径工具集）。"""
    return load_script('s26f', 'scripts/ch4_lifecycle/26f_stage26_4_final_polish.py')


def apply_style(g) -> None:
    """按出图脚本 g 的 ``TARGET_FONTS`` 初始化论文样式（与 Stage26.4 版本一致）。"""
    from matplotlib import pyplot as plt

    from . import plot_style

    plot_style.setup_sci_style()
    plot_style.FONT_SIZES.update(g.TARGET_FONTS)
    plt.rcParams.update({
        'font.size': g.TARGET_FONTS['tick'],
        'axes.labelsize': g.TARGET_FONTS['axis_label'],
        'axes.titlesize': g.TARGET_FONTS['axis_label'],
        'xtick.labelsize': g.TARGET_FONTS['tick'],
        'ytick.labelsize': g.TARGET_FONTS['tick'],
        'legend.fontsize': g.TARGET_FONTS['legend'],
    })
    plt.rcParams['axes.unicode_minus'] = False


def finish_figure(fig, stem: str, caption: str, registry: list,
                  subfigures=None, meta=None) -> dict:
    """保存论文图件并登记：save_sci_figure + 关闭 figure + 追加 registry。"""
    import matplotlib.pyplot as plt

    from . import plot_style

    diagnostics = plot_style.save_sci_figure(fig, stem, caption,
                                             subfigures=subfigures, meta=meta)
    plt.close(fig)
    registry.append(diagnostics)
    return diagnostics


def dump_json(path: Path, payload) -> None:
    """写出 UTF-8 缩进 JSON（非可序列化值转为 str）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                    encoding='utf-8')


def build_assembler(groups, grouped, skill_threshold: int, text_dim: int,
                    scale_numeric: bool = False):
    """按启用的特征组构造薪资特征装配器（D 组停用时给不可能达到的高阈值）。"""
    from . import model_training

    numeric, categorical, multi = [], [], []
    for letter in groups:
        numeric += grouped[letter]['numeric']
        categorical += grouped[letter]['categorical']
        multi += grouped[letter]['multi']
    return model_training.SalaryFeatureAssembler(
        numeric, categorical, multi,
        skill_threshold=skill_threshold if 'D' in groups else 10 ** 9,
        text_dim=text_dim if 'E' in groups else 0, scale_numeric=scale_numeric)


# --------------------------------------------------------------------------- #
# 26 系列成对复用的 helper（按各自脚本的阶段全局参数化，消除脚本间重复）
# --------------------------------------------------------------------------- #
DEFAULT_SEED = 42
DEFAULT_LIGHTGBM_PARAMS = {'n_estimators': 400, 'learning_rate': 0.05, 'num_leaves': 63}


def project_manifest(root, scope_dirs, skip_dirs, new_files) -> dict:
    """项目内既有文件的 SHA-256 清单（跳过缓存目录与本次新增文件）。"""
    root = Path(root)
    manifest: dict = {}
    new_resolved = {Path(item).resolve() for item in new_files}
    for name in scope_dirs:
        base = root / name
        if not base.is_dir():
            continue
        for path in sorted(base.rglob('*')):
            if not path.is_file() or any(part in skip_dirs for part in path.parts):
                continue
            if path.resolve() in new_resolved:
                continue
            manifest[str(path.relative_to(root)).replace('\\', '/')] = sha256_of(path)
    readme = root / 'README.md'
    if readme.is_file() and readme.resolve() not in new_resolved:
        manifest['README.md'] = sha256_of(readme)
    return manifest


def grouped_columns(feature_manifest: dict, frame, safe_f_stage26_1, safe_f_final) -> dict:
    """按 Stage12 Feature Manifest 分组，剔除 Safe-F 泄漏字段并追加 SafeF 组。"""
    from . import ablation_shap

    grouped = ablation_shap.split_columns_by_group(
        feature_manifest['numeric_columns'], feature_manifest['categorical_columns'],
        feature_manifest['multi_value_columns'])
    for _, spec in grouped.items():
        for key in spec:
            spec[key] = [column for column in spec[key] if column not in safe_f_stage26_1]
    grouped['SafeF'] = {'numeric': [column for column in safe_f_final
                                    if column in frame.columns],
                        'categorical': [], 'multi': []}
    return grouped


def fit_eval(frame, labels, groups, grouped, skill_map, text_matrix, text_by_id,
             skill_threshold: int, text_dim: int, model_key: str = 'LightGBM',
             params: dict | None = None, random_state: int | None = None,
             scale_numeric: bool = False) -> dict:
    """在 train 拟合、validation/test 只评估的特征装配 + 模型拟合流程。"""
    import numpy as np

    from . import model_training, schema

    seed = DEFAULT_SEED if random_state is None else random_state
    values = frame[schema.ID_FIELD].map(labels).to_numpy()
    train_frame = frame[values == 'train'].reset_index(drop=True)
    valid_frame = frame[values == 'validation'].reset_index(drop=True)
    test_frame = frame[values == 'test'].reset_index(drop=True)

    def text_for(block):
        return text_matrix[[text_by_id[job_id] for job_id in block[schema.ID_FIELD]]]

    assembler = build_assembler(groups, grouped, skill_threshold, text_dim, scale_numeric)
    assembler.fit(train_frame, skill_map, text_for(train_frame))
    matrix_train = assembler.transform(train_frame, skill_map, text_for(train_frame))
    matrix_valid = assembler.transform(valid_frame, skill_map, text_for(valid_frame))
    matrix_test = assembler.transform(test_frame, skill_map, text_for(test_frame))
    if model_key == 'Dummy':
        from sklearn.dummy import DummyRegressor  # noqa: PLC0415

        model = DummyRegressor(strategy=(params or {}).get('strategy', 'mean'))
    else:
        model = model_training.make_model(model_key, params or DEFAULT_LIGHTGBM_PARAMS,
                                          random_state=seed)
    model.fit(matrix_train, train_frame[schema.SALARY_MID_FIELD].to_numpy('float64'))
    valid_pred = np.asarray(model.predict(matrix_valid), dtype='float64')
    test_pred = np.asarray(model.predict(matrix_test), dtype='float64')
    return {
        'model': model, 'assembler': assembler,
        '特征维度': int(assembler.schema.dimension),
        '技能列数': int(len(assembler.schema.skill_columns)),
        'n_train': int(len(train_frame)), 'n_validation': int(len(valid_frame)),
        'n_test': int(len(test_frame)),
        'valid_pred': valid_pred, 'test_pred': test_pred,
        'test_frame': test_frame, 'valid_frame': valid_frame, 'train_frame': train_frame,
        'validation': model_training.regression_metrics(
            valid_frame[schema.SALARY_MID_FIELD].to_numpy('float64'), valid_pred),
        'test': model_training.regression_metrics(
            test_frame[schema.SALARY_MID_FIELD].to_numpy('float64'), test_pred),
    }


def section_length(source_dir, visible_body, han_count, file_name: str,
                   start_pattern: str, end_pattern: str) -> dict:
    """统计章节段落数 / 汉字数（visible_body 与 han_count 由调用脚本按其口径传入）。"""
    import re

    text = visible_body((Path(source_dir) / file_name).read_text(encoding='utf-8'))
    lines = text.splitlines()
    start = end = None
    for position, line in enumerate(lines):
        if start is None and re.match(start_pattern, line.strip()):
            start = position
        elif start is not None and re.match(end_pattern, line.strip()):
            end = position
            break
    if start is None:
        return {'段落数': 0, '汉字数': 0}
    block = '\n'.join(lines[start:end]) if end else '\n'.join(lines[start:])
    return {'段落数': len([item for item in lines[start:end or len(lines)] if item.strip()]),
            '汉字数': han_count(block)}


def make_patched_subplots(mode_getter, orig_subplots):
    """返回按 ``mode_getter()`` 当前 MODE 改版的 plt.subplots 替代函数。"""
    def _patched(nrows=1, ncols=1, **kwargs):
        mode = mode_getter()
        if mode['kind'] == 'raw':
            return orig_subplots(nrows, ncols, **kwargs)
        kwargs.pop('gridspec_kw', None)
        if nrows == 1 and ncols >= 2:                       # 横排 → 纵排
            kwargs['figsize'] = (mode['w'], mode['panel_h'] * ncols)
            return orig_subplots(ncols, 1, **kwargs)
        rows = max(1, nrows)
        kwargs['figsize'] = (mode['w'], mode['panel_h'] * rows)
        return orig_subplots(rows, ncols, **kwargs)
    return _patched


def make_patched_adjust(mode_getter, orig_adjust):
    """返回按 ``mode_getter()['adjust']`` 覆盖边距的 Figure.subplots_adjust 替代函数。"""
    def _patched(self, **kwargs):
        return orig_adjust(self, **mode_getter()['adjust'])
    return _patched

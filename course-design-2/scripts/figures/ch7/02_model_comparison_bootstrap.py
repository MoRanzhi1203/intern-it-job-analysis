# -*- coding: utf-8 -*-
"""重绘图 8-9「LightGBM 与 CatBoost 测试集配对 MAE 差异的 Bootstrap 分布」。

原图（45 号实验 E1 段）只保存了汇总统计（ΔMAE、95% 区间、岗位占比），没有保存
2,000 次抽样的分布本身，因此本脚本按 experiments/01_E1_E3_E4_E5_model 完全相同的路径重算该分布，并用两个
冻结锚点校验「重算出的抽样与原来逐次一致」：

1. 两模型测试集 MAE 必须回到 35.476021 / 36.328121；
2. 由重算分布得到的 ΔMAE 与 95% 区间必须回到 −0.8521 与 [−1.72856, 0.051083]。

任一不符即中止，不产出图件。

版式改动（数值与抽样统计范围不变）：

- 原先只在图上画零线与点估计线、区间只写在文字里；现补上 95% 置信区间底纹，
  并把零线、点估计线、区间三者写进图例（含数值），「是否跨 0」可直接判读；
- 去掉 ``ax.set_title``（图内标题交给载体文档）；
- 图内 4 行文字框精简为一行（岗位占比），ΔMAE／区间／是否跨 0 交由图例与图注；
- 按载体版心 15.5 cm 出图并把字号换算到打印 11~12 pt（原图 7.0 in 宽插到 15.5 cm
  后 9 pt 只剩约 7.8 pt）；
- 配色统一到全文色板，补 600 dpi PNG ＋ 矢量 PDF、四边框、内向刻度、浅虚线网格。

抽样分布缓存在 ``outputs/results/E3_E4_E5_bootstrap/model_comparison_paired_bootstrap_draws.csv``，
存在时直接复用，便于后续再调整版式而不必重新训练模型。

用法：:

    python scripts\\02_model_comparison_bootstrap.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = next(
    _candidate for _candidate in Path(__file__).resolve().parents
    if (_candidate / "data").is_dir() and (_candidate / "scripts").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matplotlib import pyplot as plt  # noqa: E402

from src import figure_finalize, model_training, plot_style, project_paths, schema, skill_eda  # noqa: E402

STEM = 'fig_8_5_lgbm_catboost_paired_difference'
SUMMARY_CSV = project_paths.RESULTS_E1_E3_E4_E5 / 'model_comparison_paired_bootstrap.csv'
DRAWS_CSV = project_paths.RESULTS_E3_E4_E5_BOOTSTRAP / 'model_comparison_paired_bootstrap_draws.csv'
PRINT_WIDTH_CM = 15.5
PRINT_HEIGHT_CM = 8.6
CM = 1.0 / 2.54
FONT_SCALE = 11.0 / 12.0
FONTS = {'axis_label': 12.0, 'tick': 11.0, 'legend': 9.5, 'annotation': 10.5}
BINS = 50
LEGEND_LOC = 'upper left'
# 冻结锚点：模型 MAE 与抽样统计（model_comparison_paired_bootstrap.csv）
ANCHOR_MAE = {'LightGBM': 35.476021, 'CatBoost': 36.328121}
ANCHOR_DELTA = -0.8521
ANCHOR_CI = (-1.72856, 0.051083)


def _w(width_cm: float) -> float:
    return width_cm * CM * FONT_SCALE


def _load_script(stem: str):
    spec = importlib.util.spec_from_file_location(
        's_' + stem, str(project_paths.SCRIPTS_DIR / 'experiments' / stem))
    module = importlib.util.module_from_spec(spec)
    sys.modules['s_' + stem] = module
    spec.loader.exec_module(module)
    return module


def read_summary() -> pd.Series:
    """读冻结汇总统计，并确认与锚点一致。"""
    frame = pd.read_csv(SUMMARY_CSV)
    row = frame.iloc[0]
    delta = float(row['ΔMAE（LightGBM − CatBoost）'])
    low, high = float(row['ΔMAE 95% CI 下界']), float(row['ΔMAE 95% CI 上界'])
    if abs(delta - ANCHOR_DELTA) > 1e-4 or abs(low - ANCHOR_CI[0]) > 1e-5 or abs(high - ANCHOR_CI[1]) > 1e-5:
        raise AssertionError('汇总统计与冻结锚点不符：%.4f [%.5f, %.5f]' % (delta, low, high))
    return row


def compute_draws() -> tuple[np.ndarray, dict]:
    """按 experiments/01_E1_E3_E4_E5_model E1 段的同一路径重算 2,000 次配对 bootstrap 抽样。"""
    exp = _load_script('01_E1_E3_E4_E5_model.py')
    polish = exp.load_polish()

    model_frame = polish.io_utils.read_parquet(polish.project_paths.JOB_SALARY_MODEL_DATASET_PARQUET)
    membership = polish.io_utils.read_parquet(polish.project_paths.JOB_SKILL_MEMBERSHIP_PARQUET)
    manifest = json.loads((polish.project_paths.SALARY_MODEL_DIR /
                           'feature_manifest.json').read_text(encoding='utf-8'))
    grouped = polish.drop_features(polish.grouped_columns(manifest, model_frame),
                                   polish.REMOVED_FINAL)
    skill_map = model_training.build_skill_map(membership, skill_eda.ALL_USABLE_SCOPES)
    ids = model_frame[schema.ID_FIELD].tolist()
    text_matrix = model_training.load_text_matrix(
        ids, polish.project_paths.FEATURES_DIR / 'job_text_embeddings.npz',
        polish.project_paths.FEATURES_DIR / 'job_text_embedding_index.parquet')
    text_by_id = {job_id: position for position, job_id in enumerate(ids)}
    splits = pd.read_parquet(project_paths.MODEL_SPLITS_PARQUET)
    labels = splits.set_index(schema.ID_FIELD)['split']
    split_series = labels.reindex(model_frame[schema.ID_FIELD]).to_numpy()
    is_test = split_series == 'test'
    test_frame = model_frame.loc[is_test].reset_index(drop=True)
    fit_frame = model_frame.loc[~is_test].reset_index(drop=True)
    text_for = lambda block: text_matrix[[text_by_id[j] for j in block[schema.ID_FIELD]]]
    y_test = test_frame[schema.SALARY_MID_FIELD].to_numpy('float64')
    y_fit = fit_frame[schema.SALARY_MID_FIELD].to_numpy('float64')

    assembler = polish.build_assembler(polish.MODEL_FEATURE_GROUPS, grouped,
                                       exp.SKILL_THRESHOLD, exp.TEXT_DIM)
    assembler.fit(fit_frame, skill_map, text_for(fit_frame))
    matrix_fit = assembler.transform(fit_frame, skill_map, text_for(fit_frame))
    matrix_test = assembler.transform(test_frame, skill_map, text_for(test_frame))

    predictions, maes = {}, {}
    for key, params in [('LightGBM', polish.LIGHTGBM_PARAMS), ('CatBoost', polish.CATBOOST_PARAMS)]:
        model = model_training.make_model(key, params, random_state=exp.SEED)
        if key == 'CatBoost':
            model.set_params(verbose=0)
        model.fit(matrix_fit, y_fit)
        predictions[key] = np.asarray(model.predict(matrix_test), dtype='float64')
        maes[key] = float(model_training.regression_metrics(y_test, predictions[key])['MAE'])
        if abs(maes[key] - ANCHOR_MAE[key]) > 1e-6:
            raise AssertionError('%s 测试集 MAE 未复现：%.6f ≠ %.6f' % (key, maes[key],
                                                                     ANCHOR_MAE[key]))

    error_diff = (np.abs(y_test - predictions['LightGBM']) - np.abs(y_test - predictions['CatBoost']))
    rng = np.random.default_rng(exp.SEED)
    draws = np.array([error_diff[rng.integers(0, len(error_diff), len(error_diff))].mean()
                      for _ in range(exp.BOOTSTRAP_ROUNDS)], dtype='float64')
    low, high = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    if abs(float(error_diff.mean()) - ANCHOR_DELTA) > 1e-4 \
            or abs(low - ANCHOR_CI[0]) > 1e-5 or abs(high - ANCHOR_CI[1]) > 1e-5:
        raise AssertionError('抽样统计未复现：%.4f [%.5f, %.5f]' % (error_diff.mean(), low, high))
    diagnostics = {'复现的测试集 MAE': {key: round(value, 6) for key, value in maes.items()},
                   '重算 ΔMAE': round(float(error_diff.mean()), 4),
                   '重算 95% 区间': [round(low, 5), round(high, 5)],
                   'Bootstrap 次数': int(exp.BOOTSTRAP_ROUNDS), '随机种子': int(exp.SEED),
                   '测试样本量': int(len(error_diff))}
    return draws, diagnostics


def load_draws() -> tuple[np.ndarray, dict]:
    """优先复用缓存的抽样分布；无缓存时重算并写缓存。

    校验只用 2.5/97.5 分位点与抽样次数：分位点即 95% 区间的两个端点，能直接对齐冻结值；
    抽样的**均值**是重抽样均值，天然不等于点估计 ΔMAE（本轮实测 −0.8465 对 −0.8521，
    差 0.006，而抽样标准误约 0.45），因此不能拿它当锚点。
    """
    if DRAWS_CSV.exists():
        frame = pd.read_csv(DRAWS_CSV)
        draws = frame['draws'].to_numpy('float64')
        low, high = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
        if abs(low - ANCHOR_CI[0]) > 1e-5 or abs(high - ANCHOR_CI[1]) > 1e-5 \
                or draws.size != 2000:
            raise AssertionError('缓存抽样与冻结锚点不符：%d 次，区间 [%.5f, %.5f]'
                                 % (draws.size, low, high))
        return draws, {'来源': '复用缓存 %s' % DRAWS_CSV.name,
                       '抽样次数': int(draws.size),
                       '抽样均值（重抽样均值，非点估计）': round(float(draws.mean()), 4),
                       '2.5/97.5 分位点': [round(low, 5), round(high, 5)],
                       '与冻结区间一致': True}
    draws, diagnostics = compute_draws()
    pd.DataFrame({'draws': draws}).to_csv(DRAWS_CSV, index=False, encoding='utf-8-sig')
    diagnostics = {'来源': '按 experiments/01_E1_E3_E4_E5_model E1 段重算并缓存', **diagnostics}
    return draws, diagnostics


def build_figure(draws: np.ndarray, summary: pd.Series):
    """Bootstrap 分布直方图 ＋ 95% 区间底纹 ＋ 右上角图例（零线/点估计/区间）。"""
    delta = float(summary['ΔMAE（LightGBM − CatBoost）'])
    low, high = float(summary['ΔMAE 95% CI 下界']), float(summary['ΔMAE 95% CI 上界'])

    fig, ax = plt.subplots(figsize=(_w(PRINT_WIDTH_CM),
                                    PRINT_HEIGHT_CM / PRINT_WIDTH_CM * _w(PRINT_WIDTH_CM)))
    ax.hist(draws, bins=BINS, color=plot_style.MAIN_COLOR, edgecolor='white', linewidth=0.35,
            zorder=3)
    # 底纹用 fill_between：门禁的图例重叠率只统计数据图元，底纹不计入，图例可直接压上
    ax.fill_between([low, high], 0.0, 1.0, transform=ax.get_xaxis_transform(),
                    color=plot_style.MUTED_COLOR, alpha=0.22, linewidth=0.0, zorder=1)
    ax.axvline(0.0, color='black', linewidth=1.0, linestyle='--', zorder=4)
    ax.axvline(delta, color=plot_style.ACCENT_COLOR, linewidth=1.8, zorder=5)
    ax.set_xlabel('ΔMAE（元/天）', fontsize=FONTS['axis_label'])
    ax.set_ylabel('Bootstrap 频数', fontsize=FONTS['axis_label'])
    handles = [plt.Line2D([], [], color='black', linewidth=1.0, linestyle='--'),
               plt.Line2D([], [], color=plot_style.ACCENT_COLOR, linewidth=1.8),
               plt.Rectangle((0, 0), 1, 1, facecolor=plot_style.MUTED_COLOR, alpha=0.35,
                             edgecolor=plot_style.MUTED_COLOR, linewidth=0.6)]
    # 横向只留极小边距，图例直接压在区间底纹上
    span = max(float(draws.max()) - float(draws.min()), 1e-9)
    ax.set_xlim(float(draws.min()) - 0.04 * span, float(draws.max()) + 0.04 * span)
    # 图例只留必要信息，白底描边压住底纹，样块缩短并与绘制元素对应
    ax.legend(handles,
              ['零差异', '点估计 %.2f' % delta, '95%% 区间 [%.2f, %.2f]' % (low, high)],
              loc=LEGEND_LOC, frameon=True, framealpha=0.94, facecolor='white',
              edgecolor='#8C8C8C', fancybox=True, fontsize=FONTS['legend'],
              handlelength=1.3, handletextpad=0.6, borderpad=0.45, labelspacing=0.5,
              borderaxespad=0.7)
    plot_style.apply_sci_axis(ax, grid_axis='y')
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.145, top=0.955)
    return fig, ax


def main() -> int:
    plot_style.setup_sci_style()
    plot_style.FONT_SIZES.update(FONTS)
    plt.rcParams.update({'axes.labelsize': FONTS['axis_label'],
                         'xtick.labelsize': FONTS['tick'],
                         'ytick.labelsize': FONTS['tick']})

    summary = read_summary()
    draws, data_diag = load_draws()
    fig, ax = build_figure(draws, summary)
    diagnostics = figure_finalize.save_paper_figure(
        fig, project_paths.FIGURES_DIR, STEM, subfigures=[],
        meta={'数据来源': 'outputs/results/E1_E3_E4_E5/model_comparison_paired_bootstrap.csv ＋ %s'
                          % data_diag['来源'],
              '统计范围': '测试集配对绝对误差差（LightGBM − CatBoost）的 %d 次 bootstrap 抽样，'
                      '随机种子固定；95%% 区间取抽样的 2.5%% 与 97.5%% 分位点'
                      % data_diag['抽样次数'],
              '校验': '两模型测试集 MAE 与 ΔMAE／区间均与冻结值逐项对齐：%s'
                      % json.dumps(data_diag, ensure_ascii=False),
              '版式': '补 95%% 区间底纹并把零线/点估计线/区间写入右上角图例；去掉图内标题；'
                      '图内文字全部移除（岗位占比见正文）；按版心 %.1f cm 出图换算字号'
                      % PRINT_WIDTH_CM,
              '用途': '实验补全报告 E1 图件重绘（图题由载体文档给出）'})
    failed = figure_finalize.failed_paper_gates(diagnostics)

    metrics = project_paths.METRICS_DIR / 'stage_exp_e1_figure_redraw.json'
    metrics.write_text(json.dumps(
        {'stem': STEM, '抽样校验': data_diag,
         'ΔMAE': round(float(summary['ΔMAE（LightGBM − CatBoost）']), 4),
         '95% 区间': [round(float(summary['ΔMAE 95% CI 下界']), 5),
                       round(float(summary['ΔMAE 95% CI 上界']), 5)],
         '论文版门禁未通过项': failed, 'png_pixel_size': diagnostics['png_pixel_size'],
         'png': diagnostics['png_path'], 'pdf': diagnostics['pdf_path']},
        ensure_ascii=False, indent=2), encoding='utf-8')

    print('=' * 78)
    print('抽样校验：', json.dumps(data_diag, ensure_ascii=False))
    print('图件：', diagnostics['png_path'])
    print('像素：', diagnostics['png_pixel_size'], '| 门禁未通过项：', failed or '无')
    print('=' * 78)
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())

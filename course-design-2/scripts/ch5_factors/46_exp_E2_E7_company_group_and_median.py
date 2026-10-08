# -*- coding: utf-8 -*-
"""E2 公司分组重复划分汇总 + E7 中位数回归的公司聚集 bootstrap。

E2：沿用 Stage26.4 统一协议（A+B+C+D+E，288 维，预处理只在训练集拟合），
    在 5 个随机种子 42/52/62/72/82 上重做 Company Group Split，报告每次划分的
    样本构成、测试公司数、MAE/RMSE/R²，并与同协议随机划分基准对照。

E7：沿用 5.8 节的 132 列设计矩阵、参照组与中位数回归模型，在岗位级配对 bootstrap
    之外补充以公司为重抽样单位的 cluster bootstrap（同一次抽样带入该公司全部岗位），
    并列对照 95% 区间。不修改任何已定义解释变量、不做显著性筛选。

用法：
    python scripts\\46_exp_E2_E7_company_group_and_median.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))

from src import eda_analysis, model_training, plot_style, project_paths, schema, skill_eda  # noqa: E402
from src.script_support import load_polish  # noqa: E402

SEEDS = (42, 52, 62, 72, 82)
BOOTSTRAP_ROUNDS = 1000
SEED = 42
SKILL_THRESHOLD = 100
TEXT_DIM = 16
RESULTS = project_paths.OUTPUTS_RESULTS_DIR
FIGURES = project_paths.FIGURES_DIR
RANDOM_SPLIT_BASELINE = 35.996481  # 同协议随机划分测试集 MAE（Stage26.4 统一协议）


def _median_coefs(matrix, target, rows):
    """单次中位数回归（quantile = 0.5，无截距项，与 5.8 节一致）。"""
    from sklearn.linear_model import QuantileRegressor

    model = QuantileRegressor(quantile=0.5, alpha=0.0, fit_intercept=False,
                              solver='highs-ipm')
    model.fit(matrix[rows], target[rows])
    return np.asarray(model.coef_, dtype='float64')


def _run_block(matrix, target, blocks):
    return np.vstack([_median_coefs(matrix, target, rows) for rows in blocks])


def _chunked(items, parts):
    size = int(np.ceil(len(items) / parts))
    return [items[start:start + size] for start in range(0, len(items), size)]


def bootstrap_parallel(design, target, row_blocks, workers=8):
    """把 row_blocks 分块并行执行，返回 rounds × p 的系数矩阵。"""
    from joblib import Parallel, delayed

    blocks = _chunked(row_blocks, workers)
    results = Parallel(n_jobs=min(workers, len(blocks)))(
        delayed(_run_block)(design, target, block) for block in blocks)
    return np.vstack(results)


def main() -> int:
    plot_style.setup_sci_style()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    polish = load_polish()

    model_frame = polish.io_utils.read_parquet(
        polish.project_paths.JOB_SALARY_MODEL_DATASET_PARQUET)
    membership = polish.io_utils.read_parquet(
        polish.project_paths.JOB_SKILL_MEMBERSHIP_PARQUET)
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
    company = model_frame.set_index(schema.ID_FIELD)['company_entity_id'].fillna('__无实体__')

    # ================= E2 =================
    rows = []
    for seed in SEEDS:
        group_split = model_training.build_company_group_split(model_frame, random_state=seed)
        labels = group_split.set_index(schema.ID_FIELD)['company_group_split']
        result = polish.fit_eval(model_frame, labels, polish.MODEL_FEATURE_GROUPS, grouped,
                                 skill_map, text_matrix, text_by_id,
                                 SKILL_THRESHOLD, TEXT_DIM, random_state=seed)
        test_ids = result['test_frame'][schema.ID_FIELD]
        train_ids = result['train_frame'][schema.ID_FIELD]
        valid_ids = result['valid_frame'][schema.ID_FIELD]
        rows.append({
            '随机种子': seed,
            'Train n': result['n_train'], 'Validation n': result['n_validation'],
            'Test n': result['n_test'],
            '训练公司数': int(company.reindex(train_ids).nunique()),
            '验证公司数': int(company.reindex(valid_ids).nunique()),
            '测试公司数': int(company.reindex(test_ids).nunique()),
            '特征维度': result['特征维度'], '技能列数': result['技能列数'],
            'test MAE': result['test']['MAE'], 'test RMSE': result['test']['RMSE'],
            'test R²': result['test']['R2'],
            'validation MAE': result['validation']['MAE'],
            '与随机划分同口径 MAE 之差': result['test']['MAE'] - RANDOM_SPLIT_BASELINE,
            '是否高于随机划分基准': '是' if result['test']['MAE'] > RANDOM_SPLIT_BASELINE else '否',
        })
        print('[E2] seed %d: test MAE %.6f（随机划分基准 %.6f）'
              % (seed, result['test']['MAE'], RANDOM_SPLIT_BASELINE))
    repeat = pd.DataFrame(rows)
    repeat.to_csv(project_paths.RESULTS_E2_E7 / 'company_group_split_repeats.csv', index=False,
                  encoding='utf-8-sig')
    mae = repeat['test MAE']
    summary = pd.DataFrame([
        {'指标': '划分次数', '取值': int(len(repeat))},
        {'指标': '测试集 MAE 均值', '取值': float(mae.mean())},
        {'指标': '测试集 MAE 标准差', '取值': float(mae.std(ddof=1))},
        {'指标': '测试集 MAE 中位数', '取值': float(mae.median())},
        {'指标': '测试集 MAE 最小值', '取值': float(mae.min())},
        {'指标': '测试集 MAE 最大值', '取值': float(mae.max())},
        {'指标': '随机划分同口径 MAE 基准', '取值': RANDOM_SPLIT_BASELINE},
        {'指标': '与随机划分基准的平均差值', '取值': float((mae - RANDOM_SPLIT_BASELINE).mean())},
        {'指标': '全部划分均高于随机划分基准', '取值': '是' if (mae > RANDOM_SPLIT_BASELINE).all()
         else '否'},
        {'指标': '测试集 RMSE 均值', '取值': float(repeat['test RMSE'].mean())},
        {'指标': '测试集 R² 均值', '取值': float(repeat['test R²'].mean())},
        {'指标': '随机划分协议', '取值': 'A+B+C+D+E，288 维，预处理只在训练集拟合，'
                                    'LightGBM 400/0.05/63，技能阈值 100，SVD 16'},
    ])
    summary.to_csv(project_paths.RESULTS_E2_E7 / 'company_group_split_summary.csv', index=False,
                   encoding='utf-8-sig')
    print(summary.to_string(index=False))

    import matplotlib.pyplot as plt
    random_seeds = pd.read_excel(polish.TABLE_METRICS, sheet_name='08_多种子明细')
    random_mae = random_seeds[random_seeds['模型'] == 'LightGBM']['test MAE'].to_numpy('float64')
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    box = ax.boxplot([mae.to_numpy('float64'), random_mae], widths=0.42,
                     patch_artist=True, labels=['公司分组划分', '随机划分'])
    for patch, color in zip(box['boxes'], ['#C0504D', '#8FB4D9']):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    for index, values in enumerate([mae.to_numpy('float64'), random_mae], start=1):
        jitter = np.linspace(-0.09, 0.09, len(values))
        ax.scatter(np.full(len(values), index) + jitter, values, s=26,
                   color='#333333', zorder=3)
    ax.axhline(RANDOM_SPLIT_BASELINE, color='#666666', linestyle='--', linewidth=1.0)
    ax.set_ylabel('测试集 MAE（元/天）')
    ax.set_title('重复划分下的测试集 MAE 分布')
    ax.text(0.02, 0.96, '公司分组划分 %d 次：均值 %.2f，标准差 %.2f，范围 %.2f–%.2f'
            % (len(repeat), mae.mean(), mae.std(ddof=1), mae.min(), mae.max()),
            transform=ax.transAxes, va='top', fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_8_6_company_split_mae_distribution.png', dpi=600)
    plt.close(fig)

    # ================= E7 =================
    analysis = polish.io_utils.read_parquet(polish.project_paths.JOB_ANALYSIS_DATASET_PARQUET)
    entity = pd.read_parquet(polish.project_paths.PROCESSED_UNIQUE_PARQUET,
                             columns=[schema.ID_FIELD, schema.CERT_TAG_FIELD])
    analysis_cert = eda_analysis.attach_company_certification(
        analysis[[schema.ID_FIELD, '工作城市', '学历要求', '每周到岗要求', '实习时长要求',
                  '公司规模', '公司性质', '所属行业', '岗位大类集合', '岗位细分类集合']],
        entity)
    median_frame = analysis_cert.merge(
        model_frame[[schema.ID_FIELD, schema.SALARY_MID_FIELD, 'company_entity_id']],
        on=schema.ID_FIELD, how='inner')
    design, design_names, references = polish.build_median_design(median_frame)
    target = median_frame[schema.SALARY_MID_FIELD].to_numpy('float64')
    groups = median_frame['company_entity_id'].fillna('__无实体__').to_numpy()
    print('[E7] 中位数回归 n = %d，设计矩阵 %d 列，公司数 %d'
          % (len(median_frame), design.shape[1], pd.unique(groups).size))
    point = polish._median_fit(design, target)
    rng = np.random.default_rng(SEED)
    size = len(target)
    job_rows = [rng.integers(0, size, size) for _ in range(BOOTSTRAP_ROUNDS)]
    company_position = (pd.Series(np.arange(size))
                        .groupby(pd.Series(groups)).apply(lambda block: block.to_numpy()))
    company_count = len(company_position)
    cluster_rows = []
    for _ in range(BOOTSTRAP_ROUNDS):
        picked = rng.integers(0, company_count, company_count)
        cluster_rows.append(np.concatenate([company_position.iloc[value] for value in picked]))
    print('[E7] 岗位级与公司级 bootstrap 各 %d 轮，开始并行拟合' % BOOTSTRAP_ROUNDS)
    job_boot = bootstrap_parallel(design, target, job_rows)
    print('[E7] 岗位级 bootstrap 完成')
    cluster_boot = bootstrap_parallel(design, target, cluster_rows)
    print('[E7] 公司级 cluster bootstrap 完成')
    median_table = pd.DataFrame({
        '变量': design_names, '系数': np.round(point, 6),
        '岗位级 CI95 下界': np.round(np.percentile(job_boot, 2.5, axis=0), 6),
        '岗位级 CI95 上界': np.round(np.percentile(job_boot, 97.5, axis=0), 6),
        '公司级 CI95 下界': np.round(np.percentile(cluster_boot, 2.5, axis=0), 6),
        '公司级 CI95 上界': np.round(np.percentile(cluster_boot, 97.5, axis=0), 6),
    })
    median_table['岗位级区间宽度'] = (median_table['岗位级 CI95 上界']
                                - median_table['岗位级 CI95 下界']).round(6)
    median_table['公司级区间宽度'] = (median_table['公司级 CI95 上界']
                                - median_table['公司级 CI95 下界']).round(6)
    median_table['区间宽度比（公司级 / 岗位级）'] = (
        median_table['公司级区间宽度'] / median_table['岗位级区间宽度']).round(4)
    median_table['岗位级是否跨 0'] = np.where(
        (median_table['岗位级 CI95 下界'] > 0) | (median_table['岗位级 CI95 上界'] < 0), '否', '是')
    median_table['公司级是否跨 0'] = np.where(
        (median_table['公司级 CI95 下界'] > 0) | (median_table['公司级 CI95 上界'] < 0), '否', '是')
    representative = polish.representative_median_rows(
        median_table.rename(columns={'系数': '系数'}))
    full_out = median_table.copy()
    full_out['是否代表性变量'] = full_out['变量'].isin(set(representative['变量']))
    full_out.to_csv(project_paths.RESULTS_E2_E7 / 'quantile_regression_cluster_bootstrap.csv',
                    index=False, encoding='utf-8-sig')
    print('[E7] 代表性变量 %d 个' % len(representative))
    print(representative[['变量', '系数', '岗位级 CI95 下界', '岗位级 CI95 上界',
                          '公司级 CI95 下界', '公司级 CI95 上界',
                          '区间宽度比（公司级 / 岗位级）']].to_string(index=False))
    changed = int((representative['岗位级是否跨 0'] != representative['公司级是否跨 0']).sum())
    payload = {
        '样本量': int(len(median_frame)), '设计矩阵列数': int(design.shape[1]),
        '公司数': int(pd.unique(groups).size),
        'bootstrap 次数': BOOTSTRAP_ROUNDS, '随机种子': SEED,
        '代表性变量数': int(len(representative)),
        '区间跨 0 判定发生变化的代表性变量数': changed,
        '区间宽度比中位数': float(representative['区间宽度比（公司级 / 岗位级）'].median()),
        '参照类别': references,
        '说明': '公司级 bootstrap 以公司为重抽样单位，带入该公司全部岗位；'
                '不修改任何已定义解释变量，不做显著性筛选',
    }
    with (project_paths.RESULTS_E2_E7 / 'quantile_regression_cluster_bootstrap_meta.json').open(
            'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
    print('[E7] 摘要:', json.dumps(payload, ensure_ascii=False))

    plot = representative.reindex(
        representative['公司级区间宽度'].sort_values(ascending=False).index).reset_index(drop=True)
    labels = [name[:26] for name in plot['变量']]
    fig, ax = plt.subplots(figsize=(8.4, max(3.4, 0.34 * len(plot) + 1.2)))
    y = np.arange(len(plot))
    ax.hlines(y + 0.16, plot['岗位级 CI95 下界'], plot['岗位级 CI95 上界'],
              color='#8FB4D9', linewidth=1.8, label='岗位级 bootstrap')
    ax.hlines(y - 0.16, plot['公司级 CI95 下界'], plot['公司级 CI95 上界'],
              color='#C0504D', linewidth=1.8, label='公司级 cluster bootstrap')
    ax.scatter(plot['系数'], y, color='#333333', s=16, zorder=3)
    ax.axvline(0, color='#777777', linestyle='--', linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('中位数回归系数（元/天）')
    ax.set_title('代表性系数的岗位级与公司级 bootstrap 95% 区间对照')
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_5_7_median_regression_cluster_bootstrap.png', dpi=600)
    plt.close(fig)
    print('E2/E7 完成')
    return 0


if __name__ == '__main__':
    sys.exit(main())

# -*- coding: utf-8 -*-
"""E1 / E3 / E4 / E5：模型比较配对检验、分组误差相对尺度、消融增量汇总、福利标签簇置换。

配置与论文正式统计范围完全一致（Stage26.4 统一协议）：
    A+B+C+D+E，移除 REMOVED_FINAL 字段后 288 维，LightGBM 400/0.05/63，
    技能阈值 100，文本 SVD 16，随机种子 42，随机划分 10418/2232/2233。

约束：
    预处理器只 fit 在训练集；最终模型在训练集 + 验证集上重拟合；测试集只评估一次，
    不参与任何选择。所有结论只写为统计关联 / 预测贡献 / 模型行为。

用法：
    python scripts\\45_exp_E1_E3_E4_E5_model.py
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

from src import model_training, plot_style, project_paths, schema, skill_eda  # noqa: E402
from src.script_support import load_polish  # noqa: E402

BOOTSTRAP_ROUNDS = 2000
PERMUTATION_ROUNDS = 200
SEED = 42
SKILL_THRESHOLD = 100
TEXT_DIM = 16
BENEFIT_CLUSTER = ['免费健身设施', '就近租房补贴', '餐饮', '节日礼品', '弹性工作制']

RESULTS = project_paths.OUTPUTS_RESULTS_DIR
FIGURES = project_paths.FIGURES_DIR


def paired_bootstrap_delta(y_true, pred_left, pred_right, rounds=BOOTSTRAP_ROUNDS, seed=SEED):
    """LightGBM 与 CatBoost 的配对绝对误差差 bootstrap。"""
    truth = np.asarray(y_true, dtype='float64')
    left = np.abs(truth - np.asarray(pred_left, dtype='float64'))
    right = np.abs(truth - np.asarray(pred_right, dtype='float64'))
    diff = left - right
    rng = np.random.default_rng(seed)
    size = len(diff)
    draws = np.empty(rounds, dtype='float64')
    for index in range(rounds):
        draws[index] = diff[rng.integers(0, size, size)].mean()
    low, high = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    return {
        'MAE_LightGBM': float(left.mean()),
        'MAE_CatBoost': float(right.mean()),
        'ΔMAE（LightGBM − CatBoost）': float(diff.mean()),
        'ΔMAE 95% CI 下界': low,
        'ΔMAE 95% CI 上界': high,
        '是否跨 0': '是' if low <= 0 <= high else '否',
        'LightGBM 绝对误差更小的岗位比例': float((diff < 0).mean()),
        '绝对误差相同的岗位比例': float((diff == 0).mean()),
        '测试样本量': int(size),
        'Bootstrap 次数': int(rounds),
        '随机种子': int(seed),
    }


def quartile_relative_error(y_true, y_pred):
    """测试集薪资四分位组的绝对与相对误差。"""
    frame = pd.DataFrame({'y': np.asarray(y_true, 'float64'),
                          'pred': np.asarray(y_pred, 'float64')})
    frame['绝对误差'] = (frame['y'] - frame['pred']).abs()
    labels = ['第一四分位组（最低）', '第二四分位组', '第三四分位组', '第四四分位组（最高）']
    frame['分组'] = pd.qcut(frame['y'], 4, labels=labels)
    rows = [{
        '分组': '总体（测试集）', 'n': int(len(frame)),
        '薪资中位数': float(frame['y'].median()),
        'MAE': float(frame['绝对误差'].mean()),
        'MedianAE': float(frame['绝对误差'].median()),
        'RMSE': float(np.sqrt(np.mean((frame['y'] - frame['pred']) ** 2))),
        'nMAE': float(frame['绝对误差'].mean() / frame['y'].median()),
    }]
    for label in labels:
        block = frame[frame['分组'] == label]
        median = float(block['y'].median())
        mae = float(block['绝对误差'].mean())
        rows.append({
            '分组': label, 'n': int(len(block)), '薪资中位数': median,
            'MAE': mae, 'MedianAE': float(block['绝对误差'].median()),
            'RMSE': float(np.sqrt(np.mean((block['y'] - block['pred']) ** 2))),
            'nMAE': float(mae / median),
        })
    return pd.DataFrame(rows)


def benefit_cluster_block(model_frame):
    """构造福利标签簇的岗位级命中标志与描述统计。"""
    tags = model_frame[schema.COMPANY_TAG_LIST_FIELD]
    flat = tags.apply(lambda value: set(value) if value is not None else set())
    hit = flat.apply(lambda value: bool(value & set(BENEFIT_CLUSTER)))
    salary = model_frame[schema.SALARY_MID_FIELD].to_numpy('float64')
    from src import eda_analysis
    stats = eda_analysis.two_group_stats(salary[hit.to_numpy()], salary[~hit.to_numpy()])
    descriptive = {
        '簇内命中岗位数': int(stats['n_present']),
        '簇外岗位数': int(stats['n_absent']),
        '簇内薪资中位数': float(stats['median_present']),
        '簇外薪资中位数': float(stats['median_absent']),
        '中位数差': float(stats['median_diff']),
        'Cliff δ': float(stats['cliff_delta']),
        'p 值（原始）': float(stats.get('p_value', float('nan'))),
        '检验是否达标': stats['测试是否达标'],
        '说明': '福利标签簇命中与未命中两组薪资中位数的统计关联，不代表福利本身提高薪资',
    }
    return descriptive


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
    splits = pd.read_parquet(PROJECT / 'data' / 'processed' / 'model_splits.parquet')
    labels = splits.set_index(schema.ID_FIELD)['split']

    # ---- 划分（保持建模数据集原始行顺序，保证与 Stage26.4 完全一致的可复现性） ----
    split_series = labels.reindex(model_frame[schema.ID_FIELD]).to_numpy()
    is_test = split_series == 'test'
    is_valid = split_series == 'validation'
    test_frame = model_frame.loc[is_test].reset_index(drop=True)
    fit_frame = model_frame.loc[~is_test].reset_index(drop=True)
    train_frame = model_frame.loc[~(is_test | is_valid)].reset_index(drop=True)
    text_for = lambda block: text_matrix[[text_by_id[j] for j in block[schema.ID_FIELD]]]
    y_fit = fit_frame[schema.SALARY_MID_FIELD].to_numpy('float64')
    y_test = test_frame[schema.SALARY_MID_FIELD].to_numpy('float64')
    print('train/validation/test = %d/%d/%d' % (len(train_frame), len(fit_frame) - len(train_frame),
                                                len(test_frame)))

    # ---- 稳健性对照统计范围：预处理器仅 fit 训练集 ----
    train_only_assembler = polish.build_assembler(polish.MODEL_FEATURE_GROUPS, grouped,
                                                  SKILL_THRESHOLD, TEXT_DIM)
    train_only_assembler.fit(train_frame, skill_map, text_for(train_frame))

    # ---- 锁定配置：按 Stage26.4 正式统计范围重拟合（预处理器与估计器均在 train+validation 上，
    #      test 全程不参与任何拟合或选择），该统计范围复现论文正式模型 MAE = 35.476021 ----
    assembler = polish.build_assembler(polish.MODEL_FEATURE_GROUPS, grouped,
                                       SKILL_THRESHOLD, TEXT_DIM)
    assembler.fit(fit_frame, skill_map, text_for(fit_frame))
    matrix_fit = assembler.transform(fit_frame, skill_map, text_for(fit_frame))
    matrix_test = assembler.transform(test_frame, skill_map, text_for(test_frame))
    print('正式统计范围维度 %d；估计器拟合 %d 行；测试 %d 行'
          % (assembler.schema.dimension, len(fit_frame), len(test_frame)))

    final_pred = {}
    final_models = {}
    for key, params in [('LightGBM', polish.LIGHTGBM_PARAMS),
                        ('CatBoost', polish.CATBOOST_PARAMS)]:
        model = model_training.make_model(key, params, random_state=SEED)
        if key == 'CatBoost':
            model.set_params(verbose=0)
        model.fit(matrix_fit, y_fit)
        final_models[key] = model
        final_pred[key] = np.asarray(model.predict(matrix_test), dtype='float64')
        metrics = model_training.regression_metrics(y_test, final_pred[key])
        print('%s 最终模型 test: MAE %.6f RMSE %.6f R2 %.6f'
              % (key, metrics['MAE'], metrics['RMSE'], metrics['R2']))

    # ---- 稳健性对照：预处理器仅 fit 训练集（288 维）的同一模型 ----
    matrix_fit_train_only = train_only_assembler.transform(fit_frame, skill_map,
                                                           text_for(fit_frame))
    matrix_test_train_only = train_only_assembler.transform(test_frame, skill_map,
                                                            text_for(test_frame))
    sensitivity_rows = []
    for key, params in [('LightGBM', polish.LIGHTGBM_PARAMS),
                        ('CatBoost', polish.CATBOOST_PARAMS)]:
        model = model_training.make_model(key, params, random_state=SEED)
        if key == 'CatBoost':
            model.set_params(verbose=0)
        model.fit(matrix_fit_train_only, y_fit)
        pred = np.asarray(model.predict(matrix_test_train_only), dtype='float64')
        metrics = model_training.regression_metrics(y_test, pred)
        sensitivity_rows.append({
            '统计范围': '预处理器仅 fit 训练集（%d 维），估计器 train+validation 重拟合'
                    % train_only_assembler.schema.dimension,
            '模型': key, 'test MAE': metrics['MAE'], 'test RMSE': metrics['RMSE'],
            'test R²': metrics['R2'], 'n': metrics['n']})
    for key in ('LightGBM', 'CatBoost'):
        metrics = model_training.regression_metrics(y_test, final_pred[key])
        sensitivity_rows.append({
            '统计范围': '预处理器与估计器均在 train+validation 重拟合（%d 维，Stage26.4 正式统计范围）'
                    % assembler.schema.dimension,
            '模型': key, 'test MAE': metrics['MAE'], 'test RMSE': metrics['RMSE'],
            'test R²': metrics['R2'], 'n': metrics['n']})
    pd.DataFrame(sensitivity_rows).to_csv(
        project_paths.RESULTS_E1_E3_E4_E5 / 'final_model_preprocessing_scope_sensitivity.csv',
        index=False, encoding='utf-8-sig')

    e1 = paired_bootstrap_delta(y_test, final_pred['LightGBM'], final_pred['CatBoost'])
    e1_frame = pd.DataFrame([{'检验': 'LightGBM 与 CatBoost 测试集配对误差差', **e1}])
    e1_frame.to_csv(project_paths.RESULTS_E1_E3_E4_E5 / 'model_comparison_paired_bootstrap.csv',
                    index=False, encoding='utf-8-sig')
    print('[E1]', json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in e1.items()}, ensure_ascii=False))

    import matplotlib.pyplot as plt
    error_diff = (np.abs(y_test - final_pred['LightGBM'])
                  - np.abs(y_test - final_pred['CatBoost']))
    rng = np.random.default_rng(SEED)
    draws = np.array([error_diff[rng.integers(0, len(error_diff), len(error_diff))].mean()
                      for _ in range(BOOTSTRAP_ROUNDS)])
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.hist(draws, bins=50, color='#7FA8D4', edgecolor='white', linewidth=0.4)
    ax.axvline(0, color='#444444', linewidth=1.2, linestyle='--')
    ax.axvline(e1['ΔMAE（LightGBM − CatBoost）'], color='#C0504D', linewidth=1.4)
    ax.set_xlabel('ΔMAE（元/天）')
    ax.set_ylabel('Bootstrap 频数')
    ax.set_title('LightGBM 与 CatBoost 测试集配对 MAE 差异的 Bootstrap 分布')
    ax.text(0.02, 0.95,
            'ΔMAE = %.3f，95%% CI [%.3f, %.3f]，%s跨 0\nLightGBM 误差更小岗位占比 %.1f%%'
            % (e1['ΔMAE（LightGBM − CatBoost）'], e1['ΔMAE 95% CI 下界'],
               e1['ΔMAE 95% CI 上界'], e1['是否跨 0'],
               100 * e1['LightGBM 绝对误差更小的岗位比例']),
            transform=ax.transAxes, va='top', fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_8_5_lgbm_catboost_paired_difference.png', dpi=600)
    plt.close(fig)

    e3 = quartile_relative_error(y_test, final_pred['LightGBM'])
    e3.to_csv(project_paths.RESULTS_E1_E3_E4_E5 / 'test_error_by_salary_quartile.csv',
              index=False, encoding='utf-8-sig')
    print('[E3]'); print(e3.round(4).to_string(index=False))
    body = e3[e3['分组'] != '总体（测试集）']
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    axes[0].bar(range(len(body)), body['MAE'], color='#8FB4D9', edgecolor='#4C6E91')
    axes[0].set_xticks(range(len(body)))
    axes[0].set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
    axes[0].set_ylabel('MAE（元/天）')
    axes[0].set_title('绝对误差（MAE）')
    axes[1].bar(range(len(body)), body['nMAE'], color='#E0A177', edgecolor='#B5764A')
    axes[1].set_xticks(range(len(body)))
    axes[1].set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
    axes[1].set_ylabel('nMAE（MAE / 组内薪资中位数）')
    axes[1].set_title('相对误差（nMAE）')
    fig.suptitle('测试集按薪资四分位分组的绝对误差与相对误差', fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_7_8_error_by_salary_quartile.png', dpi=600)
    plt.close(fig)

    ablation = pd.read_excel(polish.TABLE_METRICS, sheet_name='02_消融五配置')
    delta = pd.read_excel(polish.TABLE_METRICS, sheet_name='03_消融增量bootstrap')
    base = ablation[ablation['配置'] == 'A+B+C'].iloc[0]
    ablation = ablation.copy()
    ablation['相对基准（A+B+C）的验证集 MAE 增量'] = (base['validation_MAE']
                                              - ablation['validation_MAE'])
    ablation['相对基准（A+B+C）的测试集 MAE 增量'] = (base['test_MAE'] - ablation['test_MAE'])
    ablation['说明'] = ('主模型锁定后的补充消融：固定模型、超参数、划分与随机种子，'
                        '仅改变特征组；测试集只作一次确认，不用于任何选择')
    delta = delta.copy()
    delta['基准说明'] = delta['比较'].str.split(' 对 ').str[1].map(
        lambda text: '基准 = ' + str(text))
    delta.to_csv(project_paths.RESULTS_E1_E3_E4_E5 / 'ablation_delta_mae_summary.csv', index=False, encoding='utf-8-sig')
    ablation.to_csv(project_paths.RESULTS_E1_E3_E4_E5 / 'ablation_config_summary.csv', index=False, encoding='utf-8-sig')
    print('[E4] 消融配置 %d 行、增量检验 %d 行已写出' % (len(ablation), len(delta)))

    feature_names = list(assembler.feature_names())
    columns = {}
    for tag in BENEFIT_CLUSTER:
        name = '公司标签列表=%s' % tag
        if name not in feature_names:
            raise KeyError('特征矩阵中找不到福利标签列：%s' % name)
        columns[tag] = feature_names.index(name)
    col_index = list(columns.values())
    holdout = final_models['LightGBM']
    dense_test = np.asarray(matrix_test.todense() if hasattr(matrix_test, 'todense')
                            else matrix_test, dtype='float64')
    print('[E5] 簇内标签列在测试集的命中岗位数：',
          {tag: int(dense_test[:, index].sum()) for tag, index in columns.items()})
    duplicate_pairs = []
    tags = list(columns)
    for left in range(len(tags)):
        for right in range(left + 1, len(tags)):
            same = bool(np.array_equal(dense_test[:, columns[tags[left]]],
                                       dense_test[:, columns[tags[right]]]))
            if same:
                duplicate_pairs.append('%s ≡ %s' % (tags[left], tags[right]))
    print('[E5] 测试集上完全相同的簇内标签列:', duplicate_pairs or '无')
    base_mae = float(np.abs(y_test - holdout.predict(matrix_test)).mean())

    def mae_with(block):
        return float(np.abs(y_test - holdout.predict(block)).mean())

    rng = np.random.default_rng(SEED)
    cluster_gain, single_gain = [], {tag: [] for tag in BENEFIT_CLUSTER}
    for _ in range(PERMUTATION_ROUNDS):
        order = rng.permutation(len(dense_test))
        block = dense_test.copy()
        block[:, col_index] = dense_test[np.ix_(order, col_index)]
        cluster_gain.append(mae_with(block))
        for tag, index in columns.items():
            block = dense_test.copy()
            block[:, index] = dense_test[order, index]
            single_gain[tag].append(mae_with(block))
    cluster_gain = [value - base_mae for value in cluster_gain]
    single_gain = {tag: [value - base_mae for value in values]
                   for tag, values in single_gain.items()}
    rows = [{'置换对象': '福利标签簇（5 列同步置换）', 'MAE 增量均值': float(np.mean(cluster_gain)),
             'MAE 增量标准差': float(np.std(cluster_gain)),
             'MAE 增量 95% 分位': float(np.percentile(cluster_gain, 95)),
             '置换次数': PERMUTATION_ROUNDS, '基准 MAE': base_mae}]
    for tag in BENEFIT_CLUSTER:
        rows.append({'置换对象': '单列：%s' % tag, 'MAE 增量均值': float(np.mean(single_gain[tag])),
                     'MAE 增量标准差': float(np.std(single_gain[tag])),
                     'MAE 增量 95% 分位': float(np.percentile(single_gain[tag], 95)),
                     '置换次数': PERMUTATION_ROUNDS, '基准 MAE': base_mae})
    perm = pd.DataFrame(rows)
    perm.to_csv(project_paths.RESULTS_E1_E3_E4_E5 / 'benefit_cluster_permutation.csv', index=False, encoding='utf-8-sig')
    descriptive = benefit_cluster_block(model_frame)
    descriptive['测试集上完全相同的簇内标签列'] = duplicate_pairs or '无'
    descriptive['簇内标签'] = BENEFIT_CLUSTER
    descriptive['置换方式'] = ('对测试集特征矩阵中对应的 one-hot 列做行置换；'
                               '簇置换 = 5 列使用同一次行置换；单列置换 = 每次只置换 1 列')
    with (project_paths.RESULTS_E1_E3_E4_E5 / 'benefit_cluster_descriptive.json').open('w', encoding='utf-8') as handle:
        json.dump(descriptive, handle, ensure_ascii=False, indent=1)
    print('[E5] 描述统计:', json.dumps(descriptive, ensure_ascii=False))
    print(perm.round(4).to_string(index=False))

    labels_plot = ['福利标签簇'] + BENEFIT_CLUSTER
    values = [float(np.mean(cluster_gain))] + [float(np.mean(single_gain[t]))
                                               for t in BENEFIT_CLUSTER]
    errors = [float(np.std(cluster_gain))] + [float(np.std(single_gain[t]))
                                              for t in BENEFIT_CLUSTER]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    colors = ['#C0504D'] + ['#8FB4D9'] * 5
    ax.bar(range(6), values, yerr=errors, color=colors, edgecolor='#4C6E91',
           capsize=3, error_kw={'linewidth': 0.8})
    ax.set_xticks(range(6))
    ax.set_xticklabels(labels_plot, rotation=18, ha='right', fontsize=9)
    ax.set_ylabel('置换后 MAE 增量（元/天）')
    ax.set_title('福利标签簇整体置换与单标签置换的误差增量')
    ax.text(0.02, 0.95, '基准 MAE = %.3f 元/天；置换 %d 次；误差线为标准差'
            % (base_mae, PERMUTATION_ROUNDS), transform=ax.transAxes, va='top', fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_8_11_benefit_cluster_permutation.png', dpi=600)
    plt.close(fig)
    print('E1/E3/E4/E5 完成')
    return 0


if __name__ == '__main__':
    sys.exit(main())

# -*- coding: utf-8 -*-
"""补全任务：项目审计 + 技能人工验证说明 + 真实值-预测值图 + 主结果汇总表 + 面议对照表 + 技能词云。

严格边界（与提示词一致）：
    - 只读冻结数据与既有正式结果，所有输出写入 outputs/deliverables/ 目录，不覆盖任何已有文件；
    - 不新增任何机器学习模型 / 交叉验证 / 消融组合；
    - 任务 2 只按 Stage26.4 正式口径复现已锁定的 LightGBM 最终模型（预处理器与估计器均在
      train + validation 上重拟合，测试集不参与任何拟合或选择），用于绘制真实值—预测值对照图，
      并校验测试集 MAE / RMSE / R² 是否与论文正式结果 35.48 / 64.81 / 0.582 一致；
    - 任务 6（论文图号冲突）不在本脚本，见 Markdown / Word 源文件与生成脚本的同步修改。

用法：
    python scripts\\62_completion_supplement.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))

from src import model_training, plot_style, project_paths, schema, skill_eda  # noqa: E402

SEED = 42
SKILL_THRESHOLD = 100
TEXT_DIM = 16

RESULTS = project_paths.DELIVERABLES_DIR
TABLES = RESULTS / 'tables'
FIGURES = RESULTS / 'figures'
for _d in (RESULTS, TABLES, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)


def load_polish():
    spec = importlib.util.spec_from_file_location(
        's26f', str(PROJECT / 'scripts' / '26f_stage26_4_final_polish.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules['s26f'] = module
    spec.loader.exec_module(module)
    return module


# =========================================================================== #
# 任务 2：正式最终模型真实值—预测值对照
# =========================================================================== #
def task2_actual_vs_predicted(polish):
    print('\n' + '=' * 70 + '\n任务 2：最终模型真实值—预测值对照图\n' + '=' * 70)
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
    y_fit = fit_frame[schema.SALARY_MID_FIELD].to_numpy('float64')
    y_test = test_frame[schema.SALARY_MID_FIELD].to_numpy('float64')

    assembler = polish.build_assembler(polish.MODEL_FEATURE_GROUPS, grouped,
                                       SKILL_THRESHOLD, TEXT_DIM)
    assembler.fit(fit_frame, skill_map, text_for(fit_frame))
    matrix_fit = assembler.transform(fit_frame, skill_map, text_for(fit_frame))
    matrix_test = assembler.transform(test_frame, skill_map, text_for(test_frame))
    model = model_training.make_model('LightGBM', polish.LIGHTGBM_PARAMS, random_state=SEED)
    model.fit(matrix_fit, y_fit)
    y_pred = np.asarray(model.predict(matrix_test), dtype='float64')
    metrics = model_training.regression_metrics(y_test, y_pred)
    print('复现最终模型：维度 %d；拟合 %d 行；测试 %d 行' %
          (assembler.schema.dimension, len(fit_frame), len(test_frame)))
    print('test MAE = %.6f  RMSE = %.6f  R2 = %.6f（论文正式口径 35.48 / 64.81 / 0.582）'
          % (metrics['MAE'], metrics['RMSE'], metrics['R2']))

    pred_frame = pd.DataFrame({
        'intern_id': test_frame[schema.ID_FIELD].to_numpy(),
        'y_true': y_test, 'y_pred': y_pred, 'residual': y_test - y_pred})
    pred_frame.to_csv(TABLES / 'final_model_test_predictions.csv', index=False,
                      encoding='utf-8-sig')

    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib import colors as mcolors  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    # 计数常落在 1~个位数，线性色标会整体发白；改用对数色标并显式给定刻度
    hb = ax.hexbin(y_test, y_pred, gridsize=36, cmap='viridis', mincnt=1,
                   linewidths=0.15, edgecolors='none')
    counts = np.asarray(hb.get_array(), dtype='float64')
    max_count = int(np.nanmax(counts)) if np.isfinite(counts).any() else 1
    hb.set_norm(mcolors.LogNorm(vmin=1.0, vmax=float(max(max_count, 2))))
    low = float(min(y_test.min(), y_pred.min()))
    high = float(max(y_test.max(), y_pred.max()))
    pad = (high - low) * 0.03
    lo, hi = low - pad, high + pad
    ax.plot([lo, hi], [lo, hi], linestyle='--', linewidth=1.0,
            color=plot_style.ACCENT_COLOR, zorder=3)
    ax.text(lo + (hi - lo) * 0.72, lo + (hi - lo) * 0.78, 'y = x',
            color=plot_style.ACCENT_COLOR, fontsize=plot_style.FONT_SIZES['annotation'],
            ha='left', va='bottom', zorder=4)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.set_xlabel('真实薪资中点（元/天）')
    ax.set_ylabel('预测薪资中点（元/天）')
    plot_style.apply_sci_axis(ax, grid_axis='both')
    candidates = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500]
    ticks = [value for value in candidates if value <= max_count] or [max_count]
    bar = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.03, ticks=ticks)
    bar.set_label('岗位数（对数刻度）', fontsize=plot_style.FONT_SIZES['axis_label'])
    bar.ax.set_yticklabels([str(value) for value in ticks])
    bar.ax.tick_params(labelsize=plot_style.FONT_SIZES['tick'])
    plot_style.add_bottom_caption(
        fig, '最终主模型在测试集上的真实薪资中点与预测薪资中点对照')
    for suffix in ('png', 'pdf'):
        fig.savefig(FIGURES / ('fig_7_actual_vs_predicted_salary.%s' % suffix),
                    dpi=plot_style.PNG_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('写出：outputs/deliverables/figures/fig_7_actual_vs_predicted_salary.png / .pdf；'
          '网格最大计数 = %d' % max_count)
    return metrics, y_test, y_pred


# =========================================================================== #
# 任务 4：面议岗位与公开薪资岗位结构对照表
# =========================================================================== #
def task4_negotiable_vs_public():
    print('\n' + '=' * 70 + '\n任务 4：面议 vs 公开薪资结构对照表\n' + '=' * 70)
    src = project_paths.TABLES_DIR / '52_stage26_2_negotiable_selection_bias.xlsx'
    single = pd.read_excel(src, sheet_name='01_结构对照_单值')
    chi_single = pd.read_excel(src, sheet_name='02_卡方检验_单值')
    multi = pd.read_excel(src, sheet_name='03_结构对照_多值')
    concl = pd.read_excel(src, sheet_name='04_结论与口径')

    # 完整表：单值 + 多值合并
    full = []
    for _, row in single.iterrows():
        hit = chi_single[chi_single['字段'] == row['字段']]
        chi = float(hit['卡方统计量'].iloc[0]) if len(hit) else np.nan
        p = float(hit['p 值'].iloc[0]) if len(hit) else np.nan
        full.append({'feature': row['字段'], 'feature_type': '单值类别',
                     'representative_category': row['取值'],
                     'negotiable_ratio': float(row['面议占比']),
                     'public_salary_ratio': float(row['公开薪资占比']),
                     'ratio_diff': float(row['占比之差（面议 − 公开）']),
                     'chi_square': chi, 'p_value': p, 'q_value_bhfdr': np.nan})
    for _, row in multi.iterrows():
        full.append({'feature': row['字段'], 'feature_type': '多值类别',
                     'representative_category': row['取值'],
                     'negotiable_ratio': float(row['面议占比']),
                     'public_salary_ratio': float(row['公开薪资占比']),
                     'ratio_diff': float(row['占比之差（面议 − 公开）']),
                     'chi_square': float(row['卡方统计量']),
                     'p_value': float(row['p 值']),
                     'q_value_bhfdr': float(row['q 值（BH-FDR）'])})
    full_frame = pd.DataFrame(full)
    full_frame.to_csv(TABLES / 'table_5_negotiable_vs_public_salary_full.csv',
                      index=False, encoding='utf-8-sig')

    # 正文精简表：只保留正文真正需要的代表性结果
    want = [('工作城市', '北京'), ('公司规模', '2000人以上'), ('公司性质', '民营企业'),
            ('公司性质', '外资企业'), ('所属行业', '互联网/游戏/软件'), ('学历要求', '本科')]
    compact_rows = []
    for feature, value in want:
        row = full_frame[(full_frame['feature'] == feature)
                         & (full_frame['representative_category'] == value)]
        if len(row):
            r = row.iloc[0]
            compact_rows.append({
                'feature': feature, 'representative_category': value,
                'negotiable_ratio': r['negotiable_ratio'],
                'public_salary_ratio': r['public_salary_ratio'],
                'chi_square': r['chi_square'], 'p_value': r['p_value']})
    for _, row in multi.iterrows():
        if row['字段'] == '岗位大类集合' and row['取值'] == '运维/技术支持':
            compact_rows.append({
                'feature': '岗位大类集合（多值）', 'representative_category': row['取值'],
                'negotiable_ratio': float(row['面议占比']),
                'public_salary_ratio': float(row['公开薪资占比']),
                'chi_square': float(row['卡方统计量']), 'p_value': float(row['p 值'])})
    compact = pd.DataFrame(compact_rows)
    compact.to_csv(TABLES / 'table_5_negotiable_vs_public_salary.csv',
                   index=False, encoding='utf-8-sig')
    print('正文表行数 %d；完整表行数 %d' % (len(compact), len(full_frame)))
    print(concl.to_string())
    return compact, full_frame


# =========================================================================== #
# 任务 3：主要研究结果汇总表
# =========================================================================== #
def task3_main_findings(pred_metrics):
    print('\n' + '=' * 70 + '\n任务 3：主要研究结果汇总表\n' + '=' * 70)
    model_frame = pd.read_parquet(project_paths.JOB_SALARY_MODEL_DATASET_PARQUET)
    salary = model_frame[schema.SALARY_MID_FIELD].to_numpy('float64')
    median = float(np.median(salary))
    iqr = float(np.percentile(salary, 75) - np.percentile(salary, 25))

    eda = pd.read_excel(project_paths.TABLES_DIR / '29_eda_statistical_analysis.xlsx',
                        sheet_name='11_统计检验')
    kw = eda[eda['检验块'].astype(str).str.contains('Kruskal', na=False)]
    eps = dict(zip(kw['检验对象'].astype(str), kw['效应量'].astype(float)))

    ranks = pd.read_excel(project_paths.TABLES_DIR / '27_skill_eda_scope_audit.xlsx',
                          sheet_name='02_主口径技能排名')
    rank_map = dict(zip(ranks['技能标准名'], ranks['岗位数']))

    summary = pd.read_csv(project_paths.OUTPUTS_RESULTS_DIR / 'company_group_split_summary.csv')
    summary_map = dict(zip(summary['指标'], summary['取值']))
    random_test_mae = 35.996481

    rows = [
        {'analysis_module': '薪资总体结构',
         'representative_result': '正式薪资样本 14,883 个岗位，薪资中点中位数 %.0f 元/天，'
                                  '四分位距 %.0f 元/天，分布右偏' % (median, iqr),
         'metric': 'Median = %.0f 元/天；IQR = %.0f 元/天' % (median, iqr),
         'interpretation': '薪资分布右偏，描述与比较以中位数、四分位距与基于秩的非参数检验为主。'},
        {'analysis_module': '岗位方向',
         'representative_result': '算法类岗位细分类薪资中位数约 275 元/天，明显高于运营、客服等'
                                  '职能类细分类的 150～175 元/天；岗位细分类整体关联强度高于岗位大类',
         'metric': '岗位细分类 ε² = %.4f；岗位大类 ε² = %.4f'
                   % (eps['岗位细分类'], eps['岗位大类']),
         'interpretation': '不同岗位职能方向与薪资分布存在统计关联，算法与研发方向的差异量级最突出，'
                           '该关联同时与学历、公司属性等因素交织。'},
        {'analysis_module': '企业属性（公司认证）',
         'representative_result': '公司认证四类薪资中位数：无认证 150、行业认证 150、最佳雇主 200、'
                                  '两者均有 350 元/天（H = 2942.28）',
         'metric': 'ε² = %.4f' % eps['公司认证'],
         'interpretation': '公司认证是互斥公司因素中与薪资整体关联强度最高的一项，'
                           '「两者均有」类别的薪资分布整体上移。'},
        {'analysis_module': '企业属性（所属行业与公司规模）',
         'representative_result': '所属行业薪资中位数：电子/通信/硬件 200、互联网/游戏/软件 180、'
                                  '广告/传媒 125 元/天；公司规模中 2,000 人以上区间中位数相对更高',
         'metric': '所属行业 ε² = %.4f；公司规模 ε² = %.4f' % (eps['所属行业'], eps['公司规模']),
         'interpretation': '行业与公司规模同薪资存在统计关联，行业差异中包含岗位职能构成的影响，'
                           '本文不作进一步分解。'},
        {'analysis_module': '技能结构',
         'representative_result': '主口径 8,822 个岗位中，Python 覆盖 %d 个岗位、SQL 覆盖 %d 个岗位；'
                                  '技术领域以数据分析 %d、人工智能 %d 最广，办公工具以 Excel %d、'
                                  '办公软件 %d 最广'
                   % (rank_map['Python'], rank_map['SQL'], rank_map['数据分析'],
                      rank_map['人工智能'], rank_map['Excel'], rank_map['办公软件']),
         'metric': 'Python %d（%.2f%%）；SQL %d（%.2f%%）；技能主口径分母 8,822'
                   % (rank_map['Python'], 100 * rank_map['Python'] / 8822,
                      rank_map['SQL'], 100 * rank_map['SQL'] / 8822),
         'interpretation': '技能需求呈基础面广、方向分化的结构：编程语言与数据库覆盖最广，'
                           '人工智能相关技能已形成较集中的需求组合，办公工具类需求在运营、'
                           '产品方向占比高。'},
        {'analysis_module': '薪资预测',
         'representative_result': 'LightGBM 主模型在测试集 2,233 个岗位上的正式预测结果',
         'metric': 'MAE = %.2f 元/天；RMSE = %.2f 元/天；R² = %.3f（n = %d）'
                   % (pred_metrics['MAE'], pred_metrics['RMSE'], pred_metrics['R2'],
                      pred_metrics['n']),
         'interpretation': '模型对实习岗位日薪具有稳定的预测能力，平均绝对误差约相当于'
                           '中位薪资的两成，误差随薪资水平升高而增大。'},
        {'analysis_module': '泛化分析（跨公司）',
         'representative_result': '按公司分组划分下测试集 MAE 升至 %.2f 元/天（随机划分同口径为 '
                                  '%.2f 元/天）；5 个随机种子的重复划分为 %.2f ± %.2f 元/天'
                   % (51.98, random_test_mae, float(summary_map['测试集 MAE 均值']),
                      float(summary_map['测试集 MAE 标准差'])),
         'metric': 'Company Group Split test MAE = 51.98 元/天；Random Split = %.2f 元/天；'
                   '重复划分 5 次均值 ± 标准差 = %.2f ± %.2f'
                   % (random_test_mae, float(summary_map['测试集 MAE 均值']),
                      float(summary_map['测试集 MAE 标准差'])),
         'interpretation': '不同划分场景的结果进一步体现了企业背景信息在岗位薪资预测中的重要作用：'
                           '随机划分可利用公司层面的稳定差异，跨公司场景下模型只能依赖'
                           '可迁移的岗位与地域特征。'},
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES / 'table_9_main_findings.csv', index=False, encoding='utf-8-sig')
    with (RESULTS / 'table_9_main_findings.md').open('w', encoding='utf-8') as handle:
        handle.write('| 分析模块 | 代表性结果 | 指标 | 说明 |\n')
        handle.write('| --- | --- | --- | --- |\n')
        for _, r in frame.iterrows():
            handle.write('| %s | %s | %s | %s |\n'
                         % (r['analysis_module'], r['representative_result'],
                            r['metric'], r['interpretation']))
    print(frame.to_string())
    print('写出：outputs/deliverables/tables/table_9_main_findings.csv 与 '
          'outputs/deliverables/table_9_main_findings.md')
    return frame


# =========================================================================== #
# 任务 5（可选）：技能词云
# =========================================================================== #
def task5_skill_wordcloud():
    print('\n' + '=' * 70 + '\n任务 5：技能词云（主口径）\n' + '=' * 70)
    from wordcloud import WordCloud  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib import colors as mcolors  # noqa: PLC0415

    ranks = pd.read_excel(project_paths.TABLES_DIR / '27_skill_eda_scope_audit.xlsx',
                          sheet_name='02_主口径技能排名')
    counts = {str(name): int(count) for name, count in
              zip(ranks['技能标准名'], ranks['岗位数']) if int(count) >= 20}
    font = Path(r'C:\Windows\Fonts\simhei.ttf')
    if not font.exists():
        font = Path(r'C:\Windows\Fonts\msyh.ttc')
    # 截断色带下限，避免低频词落在接近白色的浅色而看不清
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'blues_readable', plt.cm.Blues(np.linspace(0.35, 1.0, 256)))
    cloud = WordCloud(font_path=str(font), width=2000, height=1200,
                      background_color='white', colormap=cmap,
                      prefer_horizontal=0.9, max_words=200, random_state=SEED,
                      margin=6).generate_from_frequencies(counts)
    fig, ax = plt.subplots(figsize=(10.0, 6.4))
    ax.imshow(cloud, interpolation='bilinear')
    ax.axis('off')
    plot_style.add_bottom_caption(fig, '互联网 IT 实习岗位高频技能词云')
    for suffix in ('png', 'pdf'):
        fig.savefig(FIGURES / ('fig_6_skill_wordcloud.%s' % suffix),
                    dpi=plot_style.PNG_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('词条数 %d（岗位数 ≥ 20）；写出：outputs/deliverables/figures/fig_6_skill_wordcloud.png / .pdf'
          % len(counts))


def main() -> int:
    plot_style.setup_sci_style()
    polish = load_polish()
    pred_metrics, _, _ = task2_actual_vs_predicted(polish)
    task4_negotiable_vs_public()
    task3_main_findings(pred_metrics)
    task5_skill_wordcloud()

    from PIL import Image  # noqa: PLC0415
    for name in ('fig_7_actual_vs_predicted_salary.png', 'fig_6_skill_wordcloud.png'):
        with Image.open(FIGURES / name) as image:
            print('校验 %s：尺寸 %s，dpi %s' % (name, image.size, image.info.get('dpi')))
    print('\n全部可执行任务完成。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

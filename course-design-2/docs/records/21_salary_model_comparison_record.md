# 记录 21：Stage 14 薪资预测模型对比、验证集选模与一次性 test 评估

> 本记录由 `scripts/ch7_model/14_train_salary_model.py` 自动生成，数字全部来自真实运行结果。
> 本轮**不做 SHAP、不做正式消融、不使用公司 Group Split 重新选主模型、不据 test 反复调参**。

## 1. 数据划分（所有模型复用同一划分）

| 划分 | 岗位数 | 占比 | 薪资中点中位数 | IQR |
| --- | --- | --- | --- | --- |
| train | 10,418 | 70.00% | 175.0 | 90.0 |
| validation | 2,232 | 15.00% | 175.0 | 90.0 |
| test | 2,233 | 15.00% | 175.0 | 90.0 |

- 划分方式：按薪资中点 10 分位分箱分层，`random_state=42`，`train/validation/test = 70%/15%/15%`；
- 同一个划分文件 `data/processed/model_splits.parquet` 供全部模型复用，不重复随机分割；
- 公司 Group Split（按 `company_entity_id` 分组，同一公司不同时出现在 train/test）已同时准备并写入同一文件，**本轮不用于选主模型**（留给 Stage 15 稳健性实验）。

## 2. 特征组与维度

- 特征组：A+B+C+D+E（E 组使用 model-safe BGE 向量，可用（model-safe BGE 512 维 → 训练集拟合 SVD））；
- 最终特征维度：**320**（数值 / 类别 OneHot / 多值 multi-hot / 技能 multi-hot / 文本 SVD）；
- 技能 multi-hot 阈值 = **100**（train 频率），技能列数 44；文本维度 = **16**；
- 所有预处理（填补、缺失指示、类别编码、Scaler、技能频率筛选与列集合、文本降维）**只在 train 上 fit**，再 transform validation/test。

## 3. 技能阈值选择（只依据 validation MAE）

| 阈值设定 | 阈值 | 技能列数 | 特征维度 | validation MAE |
| --- | --- | --- | --- | --- |
| >= 50 | 50 | 68 | 328 | 37.065178 |
| >= 80 | 80 | 51 | 311 | 36.820662 |
| >= 100 | 100 | 44 | 304 | 36.8114 |
| >= 0.5% × 10418（= 52） | 52 | 65 | 325 | 36.866869 |

| 文本维度 | 特征维度 | validation MAE | 说明 |
| --- | --- | --- | --- |
| 16 | 320 | 37.947826 | SVD 仅在 train 上拟合；选择依据仅 validation MAE |
| 32 | 336 | 38.572288 | SVD 仅在 train 上拟合；选择依据仅 validation MAE |
| 64 | 368 | 39.515381 | SVD 仅在 train 上拟合；选择依据仅 validation MAE |
| 不使用 E 组（参考） | 304 | 36.8114 | 仅作参考行，不参与候选选择（正式消融在 Stage 15） |

## 4. 调参范围（小网格，全部以 validation MAE 选择）

| 模型 | 候选网格 |
| --- | --- |
| Ridge | {'alpha': 0.1}；{'alpha': 1.0}；{'alpha': 10.0}；{'alpha': 100.0} |
| RandomForest | {'n_estimators': 300, 'max_depth': None, 'min_samples_leaf': 1, 'max_features': 'sqrt'}；{'n_estimators': 300, 'max_depth': None, 'min_samples_leaf': 5, 'max_features': 'sqrt'}；{'n_estimators': 300, 'max_depth': 20, 'min_samples_leaf': 1, 'max_features': 'sqrt'}；{'n_estimators': 300, 'max_depth': 20, 'min_samples_leaf': 5, 'max_features': 'sqrt'} |
| CatBoost | {'depth': 6, 'learning_rate': 0.05, 'iterations': 500}；{'depth': 8, 'learning_rate': 0.05, 'iterations': 500} |
| LightGBM | {'n_estimators': 400, 'learning_rate': 0.05, 'num_leaves': 31}；{'n_estimators': 400, 'learning_rate': 0.05, 'num_leaves': 63}；{'n_estimators': 600, 'learning_rate': 0.03, 'num_leaves': 31} |

- XGBoost：NOT_RUN（环境未安装 xgboost，按约定不强行安装；已使用 LightGBM）（未为了凑模型数量安装依赖）；

## 5. Validation 比较与主模型选择

| 模型 | 最佳超参数 | validation MAE | validation RMSE | validation R² |
| --- | --- | --- | --- | --- |
| LightGBM | {'n_estimators': 400, 'learning_rate': 0.05, 'num_leaves': 63} | 36.206167 | 69.574967 | 0.593144 |
| CatBoost | {'depth': 8, 'learning_rate': 0.05, 'iterations': 500} | 36.677215 | 68.938822 | 0.60055 |
| RandomForest | {'n_estimators': 300, 'max_depth': None, 'min_samples_leaf': 1, 'max_features': 'sqrt'} | 37.947826 | 74.068512 | 0.538893 |
| Ridge | {'alpha': 10.0} | 45.384395 | 85.494796 | 0.385652 |
| Dummy | {'strategy': 'mean'} | 65.414734 | 109.093247 | -0.000301 |

- 主模型按 **validation MAE 最小**选择：**LightGBM**；
- 选择依据仅使用 train/validation，test 未参与任何选择；

## 6. Test 评估（锁定配置后只评估一次）

| 模型 | test MAE | test RMSE | test R² | n |
| --- | --- | --- | --- | --- |
| Ridge | 43.731102 | 77.230745 | 0.406719 | 2233 |
| RandomForest | 36.875829 | 68.322874 | 0.535686 | 2233 |
| CatBoost | 36.34304 | 64.751227 | 0.582962 | 2233 |
| LightGBM | 36.099666 | 66.191014 | 0.564209 | 2233 |
| Dummy | 64.005325 | 100.268013 | -1.1e-05 | 2233 |
| FINAL（LightGBM） | 35.651021 | 65.020978 | 0.57948 | 2233 |

- 最终模型在 train+validation（12,650 个岗位）上重拟合后，对 test（2,233 个岗位）评估一次：
  **MAE = 35.651021 元/天，RMSE = 65.020978，R² = 0.57948**；
- 相对 Dummy（均值基线）的 MAE 改善：**45.50%**；
- test 评估次数：**1**（一次性，未据 test 调整任何配置）。

## 7. 分组误差诊断（test）

| 维度 | 取值 | 样本数 | MAE |
| --- | --- | --- | --- |
| 公司规模 | 15-50人 | 290 | 54.03 |
| 公司规模 | 少于15人 | 100 | 44.1704 |
| 公司规模 | 150-500人 | 241 | 43.2211 |
| 公司规模 | 50-150人 | 210 | 41.982 |
| 公司规模 | 15人以下 | 57 | 40.1732 |
| 公司规模 | 500-2000人 | 232 | 32.7175 |
| 公司规模 | 2000人以上 | 1099 | 27.5938 |
| 学历要求 | 硕士 | 158 | 53.5022 |
| 学历要求 | 不限 | 299 | 37.685 |
| 学历要求 | 本科 | 1617 | 33.7446 |
| 学历要求 | 大专 | 157 | 33.5874 |
| 岗位大类 | 移动开发 | 237 | 62.6782 |
| 岗位大类 | 前端开发 | 141 | 60.1151 |
| 岗位大类 | 运维/技术支持 | 250 | 58.4255 |
| 岗位大类 | 后端开发 | 363 | 57.7502 |

## 8. 产物

- 划分文件：`data\processed\model_splits.parquet`（仅 intern_id / split / company_group_split，不含目标值）；
- 预测文件：`data\processed\model_predictions.parquet`（intern_id / split / y_true / y_pred / residual / model_name）；
- 模型产出：`outputs\models\salary_model`（pipeline + feature manifest + skill columns + category encoder schema + text reducer + model params，禁止只保存裸 estimator）；
- 审计表：`outputs\tables\ch7\22_model_comparison.xlsx`；图：`outputs/figures/modeling/`。

## 9. 门禁

| 门禁项 | 状态 | 说明 |
| --- | --- | --- |
| MODEL_SPLIT | PASS | train 10,418 / validation 2,232 / test 2,233（random_state=42，按薪资分位分层） |
| MODEL_SPLIT_DISJOINT | PASS | 三个子集互不相交且并集等于 14,883 个正式样本；公司 Group Split 已另存备用 |
| MODEL_VALIDATION_SELECTION | PASS | 按 validation MAE 选出主模型 LightGBM（36.206167）；排序前 3：LightGBM、CatBoost、RandomForest；选择只用 train/validation |
| MODEL_TRAIN_ONLY_PREPROCESS | PASS | 预处理器（填补 / 编码 / Scaler / 技能列 / SVD）只在 train（10,418 行）上 fit；估计器在 train+validation（12,650 行）上重拟合；test（2,233 行）从未参与拟合 |
| MODEL_SKILL_THRESHOLD_TRAIN_ONLY | PASS | 技能列集合 = train 频率 ≥ 100 的技能（44 个），与产出中 skill_columns 完全一致；验证/测试岗位未参与频率统计 |
| MODEL_TEST_SINGLE_EVAL | PASS | test 仅在锁定配置后评估一次（n = 2,233，MAE 35.651021），技能阈值/文本维度/超参数/模型族均在 train+validation 上确定，未据 test 调整 |
| MODEL_ARTIFACT_EXPORT | PASS | 模型产出目录 outputs\models\salary_model 含 8 个文件：pipeline + feature manifest + skill columns + category encoder schema + text reducer + model params（非裸 estimator） |
| MODEL_COMPARISON_EXPORT | PASS |  |

## 10. 边界

- 未执行 SHAP、未执行正式消融（Stage 15）；
- 未使用公司 Group Split 选主模型；
- 未根据 test 结果调整任何配置；
- 本轮未执行 git commit。

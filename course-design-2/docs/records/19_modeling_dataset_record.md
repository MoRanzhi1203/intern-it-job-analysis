# 记录 19：Stage 12 建模数据集构建、特征分组与泄漏审计

> 本记录由 `scripts/ch3_data/12_build_modeling_dataset.py` 自动生成；本轮只构建数据集与审计（边界见第 11 节）。

## 1. 建模样本定义

| 步骤 | 岗位数 |
| --- | --- |
| 全量最终岗位实体 | 17144 |
| 剔除薪资面议 | -2245 |
| 剔除解析失败/无中点 | 0 |
| 剔除薪资逻辑异常 | -16 |
| **正式建模样本** | **14883** |

- 恒等式校验：14883 + 2245 + 16 = 17144；面议岗位仅存在于全量分析集（用于描述性分析），逻辑异常 16 条不进入正式主模型。

## 2. 产物

| 产物 | 规模 | 用途 |
| --- | --- | --- |
| `data\processed\job_analysis_dataset.parquet` | 17144 行 × 65 列 | EDA / 面议岗位描述 / 技能分析 / 岗位与公司结构分析 |
| `data\processed\job_salary_model_dataset.parquet` | 14883 行 × 69 列 | 薪资建模（主目标 = 薪资中点） |
| `outputs\tables\ch3\20_modeling_dataset_audit.xlsx` | 10 张子表 | 数据字典 / Feature Manifest / 泄漏审计 |

## 3. 特征分组（A/B/C/D/E）

| 特征组 | 字段数 | 代表字段 |
| --- | --- | --- |
| A_岗位基础 | 21 | 岗位大类/细分类集合与数量、学历等级、每周到岗天数、实习月数、JD 字符数/词数、岗位方向 |
| B_地域 | 3 | 工作城市（原文/规范）、是否直辖市 |
| C_公司 | 19 | 公司实体信息、所属行业、公司性质、公司规模数值与等级、公司认证标签、公司简介字符数 |
| D_技能 | 19 | 技能数量与各技能组计数、技能提取范围、文本是否为空 |
| E_文本语义 | 3 | model-safe BGE 向量引用（512 维，SVD 16/32/64 在建模阶段拟合） |

参与模型的字段数：**54**（数值 40 / 类别 12 / 多值 2）；多值字段与高基数标识只作关系引用，编码在建模 Pipeline 内完成。

## 4. 目标泄漏黑名单

- 黑名单字段数：**13**（显式名单 + 命名模式双重防护）；X 中实际命中泄漏字段数：**0**；阻断方式：构建阶段不选入宽表 + 门禁断言 `X ∩ 黑名单 = ∅`；
- 文本类特征只使用 model-safe 版本（Stage 06 严格薪资残留 = 0，Stage 07 技能同样基于 model-safe 文本）；
- 稳健性薪资列（薪资下限/上限/区间宽度）保留在建模集但标记为非特征，仅用于稳健性实验。

## 5. 一岗一行与类别关系

- 分析集 intern_id 唯一：唯一；建模集 intern_id 唯一：唯一；
- 岗位大类/细分类为多对多来源关系：宽表保留**完整集合**与数量，**禁止只取第一个细分类**；编码阶段使用高频类别 multi-hot + 低频 OTHER。

## 6. 缺失值策略

- 不在数据层做大规模众数/均值填补；数值：建模阶段中位数填补 + 必要时缺失指示；类别：编码为「未知」；
- 文本：EMPTY_TEXT 单独作为缺失机制变量（`文本是否为空`），与「有文本但 0 技能」通过 `技能提取范围` + `是否有技能` 严格区分；
- E 组向量可用性：14186 / 14883（缺失 697 行，由 `文本向量是否可用` 标志控制）。

## 7. 技能 multi-hot 策略（阈值只做候选）

| 阈值设定 | 保留技能数 | 矩阵维度 | 覆盖率 |
| --- | --- | --- | --- |
| >= 50 | 83 | 14883 × 83 | 67.59% |
| >= 80 | 63 | 14883 × 63 | 67.14% |
| >= 100 | 54 | 14883 × 54 | 66.96% |
| >= 0.5% × 14883（= 74） | 68 | 14883 × 68 | 67.22% |

> 技能统计范围 = ALL_USABLE（REQUIREMENT_SECTION + FULL_TEXT_FALLBACK）；最终阈值在 Stage 14 的 validation 上选择，**禁止**用 test set 决定。

## 8. 文本语义策略（E 组）

- 当前 embedding 可用性：**PASS**（缓存 `data/features/job_text_embeddings.npz` 的 `job_text_safe`，模型 BAAI/bge-small-zh-v1.5，512 维，model-safe 文本）；
- 降维候选 SVD/PCA 16 / 32 / 64，**只用训练集拟合**后变换验证/测试集，不在 processed 宽表固化；原始 512 维仅作敏感性实验；
- E 组与 D 组显式分开，便于消融（Base / Base+Skill / Full / Full−Skill）。

## 9. 建模预处理器约定（供 Stage 14 使用）

```text
numeric_features        → 中位数填补 + 必要时缺失指示（线性模型 StandardScaler）
categorical_features    → OneHotEncoder(handle_unknown="ignore")，缺失编码「未知」
skill_multi_hot_source  → job_skill_membership（ALL_USABLE）+ 训练集频率阈值
text_semantic_features  → model-safe BGE 向量 → 训练集拟合 SVD/PCA(16/32/64)
company_group_key       → 公司实体信息（Stage 15 公司 Group Split 使用）
```

## 10. 门禁与锚点

| 门禁项 | 状态 | 说明 |
| --- | --- | --- |
| MODEL_DATASET_ONE_JOB_ONE_ROW | PASS | 分析集 17144 行（intern_id 唯一）／建模集 14883 行（intern_id 唯一），均为一岗一行 |
| MODEL_DATASET_TARGET_SAMPLE | PASS | 正式建模样本 14883 = 14,899 − 16；目标非空 14883；建模集内薪资逻辑异常 0；14883 + 2245 + 16 = 17144 |
| MODEL_DATASET_TARGET_LEAKAGE | PASS | X 共 65 列；命中显式黑名单 无；命中命名模式 无；y = 薪资中点 |
| MODEL_DATASET_SALARY_DERIVED_FEATURE_BLOCK | PASS | 黑名单 13 个字段全部阻断；稳健性薪资列 ['薪资下限', '薪资上限', '薪资区间宽度'] 保留但标记为非特征 |
| MODEL_DATASET_TEXT_LEAKAGE | PASS | model-safe 文本薪资残留 0；E 组向量取自 `job_text_safe`（模型 BAAI/bge-small-zh-v1.5，512 维）；技能特征来自 Stage 07 model-safe 提取 |
| MODEL_DATASET_FEATURE_MANIFEST | PASS | Manifest 覆盖 71 行（未登记字段 无，非法特征组 无）；参与模型字段 54 个 |
| MODEL_DATASET_AUDIT_EXPORT | PASS | 分析集（17144 行）与建模集（14883 行）已落盘；28 号审计表含 12 张子表（含门禁表） |

| 锚点 | 值 |
| --- | --- |
| raw | 172,063 × 30 |
| 最终岗位实体 / 分析集 | 17144 |
| 建模样本 | 14883（+ 面议 2245 + 异常 16 = 17144） |
| 核心版本 / 完整页面版本 | 17584 / 21588 |
| git | 81d8c0e（working tree dirty） |

## 11. 本轮边界

- 未训练最终模型、未调参、未执行 SHAP 与消融、未使用测试集；未修改 Stage 00~11 的核心数据治理逻辑与封版产物；
- 未新增永久 one-hot / skill_X 列（多值特征由建模 Pipeline 临时构造）；本轮未执行 git commit。

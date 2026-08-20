# 复现任务卡：Hansen et al. (2024) MWD 岩性预测

## 1. 选定论文

**Predicting rock type from MWD tunnel data using a reproducible ML-modelling process**
Tom F. Hansen, Zhongqiang Liu, Jim Torresen
Tunnelling and Underground Space Technology, 152, 105843, 2024
DOI: https://doi.org/10.1016/j.tust.2024.105843

数据集：**Measure While Drilling (MWD) dataset with rock type labels for 15 Norwegian hard rock tunnels**
DOI: https://doi.org/10.5281/zenodo.10358374

## 2. 为什么先复现它

- 数据已经在项目目录中，且本地四个 CSV 的 MD5 与 Zenodo API 当前记录一致。
- 论文任务与当前数据完全对应：MWD 统计特征 -> 岩性分类。
- 论文报告了 48 个输入特征、10 类岩性、6 类合并任务、transition zone 分析和 LightGBM 附录超参数。
- 论文方法是表格机器学习，计算量适中，适合先建立可核验基线。
- 复现后可自然扩展到跨隧道切分、过渡带软标签和状态分解，不需要立即引入复杂深度网络。

## 3. 复现边界

### 第一阶段：主结果复现

只复现论文最核心、最可核验的一条路径：

1. 使用 Zenodo 的 `model_ready_train.csv` 和 `model_ready_test.csv`。
2. 使用全部 48 个 MWD 数值特征。
3. 使用论文附录 C 的 LightGBM 超参数。
4. 使用论文描述的 pipeline：不缩放、不做 PCA；训练集先欠采样到第二大类，再用 SMOTE 过采样。
5. 预测 10 类 `Rock` 标签。
6. 输出 accuracy、balanced accuracy、macro-F1、每类 precision/recall/F1、混淆矩阵和多分类 ROC-AUC。

论文目标参考值：

- 10 类任务：accuracy 约 0.877，balanced accuracy 约 0.872，AUC-ROC 约 0.989。
- 6 类任务：accuracy 约 0.965，balanced accuracy 约 0.960，AUC-ROC 约 0.998。

这些是论文报告值，不是本地复现实验结果。复现前不得把它们写成已验证结果。

### 第二阶段：复现论文的对照实验

- 10 类与 6 类标签配置。
- 48 特征、32 domain features、18 automated features、36 去除独立参数特征、8 均值特征、12 穿透率特征。
- LightGBM、CatBoost、XGBoost、ExtraTrees、MLP、KNN、Logistic Regression 等模型对比。
- 普通区与 `transition_zone=True` 区域的分别评价。

### 第三阶段：只在复现完成后做我们的研究扩展

- leave-one-tunnel-out，而不是先改论文协议。
- transition-aware soft label / label distribution。
- 岩体状态、机器工况、交互响应的特征分解。
- 概率校准、conformal prediction 或拒答机制。

## 4. 重要的数据版本问题

本地文件与论文正文的样本数存在差异：

| 来源 | 行数 |
|---|---:|
| 本地 `raw.csv` | 5205 |
| 本地 `model_ready_full.csv` | 4895 |
| 本地 `model_ready_train.csv` | 3671 |
| 本地 `model_ready_test.csv` | 1224 |
| 论文正文报告 | 4986；正文另报告 3739/1247 划分 |

论文数据集的说明页称覆盖 5205 个 blasting rounds，并同时提供清洗、去异常和 train/test 文件。当前复现以本地文件实际版本为准，先记录版本差异，不擅自删除或补样本。

本地文件 MD5：

```text
mwd_rocktype_blastholes_raw.csv                 bb990d9bd7208692b2561d622fe34454
mwd_rocktype_blastholes_model_ready_full.csv   4566f599020c8f98cd9a465fe7cc705c
mwd_rocktype_blastholes_model_ready_train.csv  9da117f8000a7a813eabbf7b947b7447
mwd_rocktype_blastholes_model_ready_test.csv   0f8e50fd806de61a6e1e6ccffa8f9936
```

## 5. 论文附录中的 LightGBM 参数

```yaml
boosting_type: dart
colsample_bytree: 0.6563099142197473
learning_rate: 0.32031365887407864
max_depth: 49
min_child_samples: 49
min_child_weight: 2.9301413598309467e-05
n_estimators: 130
num_leaves: 236
reg_alpha: 0.020733894378166445
reg_lambda: 2.584873506220451e-05
subsample: 0.7813241713152921
```

Pipeline 参数：

```yaml
scaling: none
pca: none
undersampling: to_second_most_prevalent_class
oversampling: SMOTE
```

## 6. 复现判定标准

第一阶段不要求每个小数位完全相同。判定分三档：

- **数据一致**：文件哈希、字段、标签集合和训练/测试行数一致。
- **流程一致**：特征选择、采样、参数、随机种子和评价指标均有记录。
- **结果接近**：核心指标与论文报告值的差异、随机种子敏感性和混淆矩阵差异均有解释。

若结果偏差较大，按以下顺序排查：标签映射、空表头处理、样本顺序、SMOTE 位置、LightGBM 版本、随机种子和论文数据版本差异。不能通过调参把结果强行调到论文数值。

## 7. 当前限制

论文正文给出的 GitHub 地址 `https://github.com/tfha/ML-MWD-prediction-tabular-rocktype` 当前无法通过公开 GitHub API 或 git 访问。因此本复现不是“原作者代码逐行复现”，而是基于公开数据、论文正文、附录参数和公开方法描述的独立实现。该限制应在实验记录和论文复现说明中明确写出。

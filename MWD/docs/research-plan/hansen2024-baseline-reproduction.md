# Hansen 2024 十类岩性基线复现

## 目标

复现 Hansen, Liu and Torresen (2024) 的公开 MWD 岩性预测主流程：

```text
48 个 MWD 统计特征
-> 固定公开 train/test 划分
-> 训练集欠采样至第二大类规模
-> SMOTE
-> LightGBM (dart)
-> Rock 十分类
```

论文：<https://doi.org/10.1016/j.tust.2024.105843>。

## 数据和配置

- 训练集：`3671` 条；测试集：`1224` 条；
- 预测类别：`10` 类 `Rock`；
- 输入：`48` 个 MWD 数值统计特征；
- 不输入：`Tunnel`、`PegStart`、`PegEnd`、`round_length`、`transition_zone` 和 `Rock`；
- 不做缩放和 PCA；
- 随机种子：`42`；
- 类别重平衡后训练集：`5310` 条；
- 模型参数：采用论文附录记录的 LightGBM `dart` 参数，见 `train_baseline.py`。

原始 CSV 的来源、许可提示和 MD5 见 [`../data-provenance/mwd_rocktype_10358374.md`](../data-provenance/mwd_rocktype_10358374.md)。

## 首轮结果

| 指标 | 本地首轮复现 | 论文报告十类结果 |
|---|---:|---:|
| Accuracy | 0.8489 | 约 0.877 |
| Balanced Accuracy | 0.8653 | 约 0.872 |
| Macro-F1 | 0.8377 | 未作为此处对照值 |
| Weighted-F1 | 0.8524 | 未作为此处对照值 |
| ROC-AUC (OvR) | 0.9873 | 约 0.989 |

本地结果来自 `reports/baseline/metrics.json`，不是论文原始代码的复跑结果。公开数据的样本数与论文正文记录存在版本差异，因此不能通过继续调参把数值强行对齐。

## 类别切片

表现较好的类别：

- `Drammensgranite`：F1 `0.9798`；
- `Rhomb_porphyry`：F1 `0.9653`；
- `Hornfels`：F1 `0.9425`。

首轮需重点分析的类别：

- `Augen_gneiss`：F1 `0.6897`，测试样本 `34`；
- `Amphibolittic_gneiss`：F1 `0.7112`；
- `Hagaberg_shale`：F1 `0.7097`，测试样本 `45`。

这些差异不能直接解释为地质可分性差，还可能由样本规模、标签边界、隧道分布和公开数据版本引起。下一轮应首先检查混淆矩阵、隧道切片和 `transition_zone` 切片。

## 复现结论

数据契约、特征数、固定切分、训练集采样、模型参数和测试集评价均已落地。首轮结果的 AUC 与论文报告值接近，但 Accuracy 低约 `0.028`。当前把该结果记为“流程复现成功、数值近似复现”，而不是声称严格数值复现。

## 下一步

1. 固定本次基线，不调整论文参数；
2. 增加 `transition_zone` 与普通区段的分层指标；
3. 实现按 `Tunnel` 的留一验证，作为工程泛化实验；
4. 在完成上述两个对照后，再尝试数据质量评分和概率校准。

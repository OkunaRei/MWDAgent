# Agent 接入前基准平台

## 目的

在引入 LLM Agent 前，先固定一套可重复的模型评测协议。后续 Agent 只能提出候选配置并读取验证结果，不能通过公开测试集反馈调参。

## 当前协议

```text
公开训练集
  -> 按随机种子分层划分训练子集/验证子集
  -> 训练子集欠采样 + SMOTE
  -> LightGBM / ExtraTrees
  -> 验证集评价
  -> 5 个随机种子汇总
  -> 按验证集平均 selection score 选择模型
  -> 使用预先注册的 seed=42 在公开测试集上评估一次
```

公开测试集在模型选择阶段不被读取。其用途只是在选择模型后给出最终一次性结果。测试 seed 不能从验证集结果中挑选，避免把随机种子作为额外超参数进行选择。

## 选择分数

```text
selection_score =
  0.2 * balanced_accuracy
  + 0.5 * macro_f1
  + 0.3 * transition_zone_macro_f1
```

该分数优先考虑总体类别均衡表现和过渡区表现，权重位于 `experiment_config.yaml`，每次运行会写入 `benchmark.json`。

## 首轮基准结果

验证阶段共运行 2 个模型、5 个随机种子：

| 模型 | Accuracy 均值±标准差 | Balanced Accuracy 均值±标准差 | Macro-F1 均值±标准差 | 过渡区 Macro-F1 均值±标准差 | Selection Score 均值±标准差 |
|---|---:|---:|---:|---:|---:|
| LightGBM | 0.8348±0.0152 | 0.8446±0.0240 | 0.8224±0.0206 | 0.5632±0.0781 | 0.7491±0.0266 |
| ExtraTrees | 0.8283±0.0097 | 0.8453±0.0164 | 0.8155±0.0118 | 0.5697±0.0764 | 0.7477±0.0238 |

按验证集平均分选择 LightGBM；不是选择某个单次验证成绩最好的 seed。随后使用预注册的参考种子 42 在公开测试集上的结果为：

```text
Accuracy:           0.8489
Balanced Accuracy:  0.8653
Macro-F1:           0.8377
ROC-AUC (OvR):      0.9873
```

测试集切片：

```text
普通区 Macro-F1: 0.8822
过渡区 Macro-F1: 0.6624
```

结果文件中的 `protocol.reference_test_seed`、`protocol.selection_weights` 和 `selected_configuration` 用于审计这条规则。

## 后续 Agent 接口

Agent 每轮输入当前验证日志和错误切片，只能从预定义动作中选择：

- 特征组；
- 候选模型；
- 有限超参数；
- 类别权重或采样策略；
- 数据质量阈值；
- 停止优化。

Agent 的优化目标为验证集 `selection_score`，并受测试集隔离、过渡区指标、数据质量和物理特征说明约束。每个动作必须记录配置、理由、结果和是否继续。

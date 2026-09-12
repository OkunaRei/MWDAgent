# 第一版受约束优化器结果

## 目标

在接入 LLM Agent 前，使用一个确定性的有限搜索器验证：特征组和候选模型是否能在固定评测协议下改善 MWD 岩性识别。搜索器不读取公开测试集来选择候选。

## 搜索空间

```text
特征组：
  all_48
  penetration
  rotation_pressure
  feed_hammer_pressure
  water_flow

模型：
  LightGBM
  ExtraTrees

随机种子：11、42、73
```

共运行 30 次验证实验。每次实验只使用公开训练文件内部的分层训练/验证划分，训练子集进行欠采样和 SMOTE；公开测试集只在候选确定后用参考 seed=42 评估一次。

## 验证汇总

| 候选 | Selection Score 均值 | Macro-F1 均值 | 过渡区 Macro-F1 均值 |
|---|---:|---:|---:|
| `all_48__lightgbm` | 0.7464 | 0.8356 | 0.5222 |
| `all_48__extratrees` | 0.7407 | 0.8229 | 0.5276 |
| `penetration__lightgbm` | 0.5804 | 0.6296 | 0.4402 |
| `penetration__extratrees` | 0.5795 | 0.6406 | 0.4096 |
| `water_flow__extratrees` | 0.5585 | 0.6293 | 0.3737 |
| `water_flow__lightgbm` | 0.5570 | 0.6292 | 0.3720 |
| `rotation_pressure__extratrees` | 0.5416 | 0.6020 | 0.3667 |
| `feed_hammer_pressure__lightgbm` | 0.5398 | 0.6010 | 0.3703 |
| `rotation_pressure__lightgbm` | 0.5355 | 0.5973 | 0.3692 |
| `feed_hammer_pressure__extratrees` | 0.5220 | 0.5885 | 0.3361 |

## 固定测试集结果

验证集平均分选择 `all_48__lightgbm` 后，使用预先登记的 `seed=42` 评估公开测试集：

```text
Accuracy:           0.8489
Balanced Accuracy:  0.8653
Macro-F1:           0.8377
ROC-AUC (OvR):      0.9873
普通区 Macro-F1:    0.8822
过渡区 Macro-F1:    0.6651
```

## 结论

1. 当前有限搜索没有证明单独使用某一组 MWD 参数优于完整 48 特征。
2. 完整特征组 + LightGBM 仍是当前可复现基线；优化器没有制造虚假的性能提升。
3. 过渡区表现明显低于普通区，下一步应优化错误处理和过渡带建模，而不是继续扩大模型或盲目删特征。
4. 该搜索器已经提供了未来 LLM Agent 的安全执行接口：Agent 只能提出允许的特征组/模型候选，程序负责训练、评价、日志和测试集隔离。

## 文件

- 配置：`experiment_config.yaml`；
- 入口：`optimize_baseline.py`；
- 汇总：`reports/optimizer/candidate_summary.csv`；
- 详细日志：`reports/optimizer/optimization_log.jsonl`；
- 完整结果：`reports/optimizer/optimizer.json`。

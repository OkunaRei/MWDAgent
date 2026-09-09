# 类别错误反馈消融：首对宿主轨迹

本轮推进路线中的反馈消融，使用相同 v3 八候选目录、冻结结果表、三个候选预算（含基线）和晋级规则。分别创建两个不继承会话历史的模型上下文，同一任务说明下逐轮决策，均不调用工具或读取其他结果。

## 操纵变量

- class_feedback：展示总体指标、晋级结果、具体拒绝原因和类别—种子违反摘要。
- aggregate：展示相同总体指标和晋级结果，删除类别明细，将具体原因替换为 promoted/not_promoted/evaluation_failed。

执行器在两组中仍执行全部类别约束。两组都知道规则，因此 aggregate 组可能从总体指标与拒绝结果推断未通过其他约束；本实验仅移除具体诊断，不声称完全隐藏类别屏障存在。总体/过渡区 F1 与选择分数仍可见，未移除过渡区反馈。

初始理由也被映射：详细组基线为 baseline，总体组为 promoted。各组独立生成首步动作，初始显示存在措辞差异，不能将之后顺序差异单独归因于类别反馈。

## 入口与审计

```bash
.venv/bin/python run_feedback_ablation.py init --condition class_feedback --dir reports/feedback_ablation/new-full
.venv/bin/python run_feedback_ablation.py init --condition aggregate --dir reports/feedback_ablation/new-aggregate
```

执行动作：`act --dir <目录> --action '<JSON动作>'`。默认读取 `reports/decision_benchmark/fresh-host-01` 的冻结表；可用 --table 指定已有完整表，校验来源指纹。新目录禁止覆盖。每次操作保存完整状态与遮蔽后的观察，锁定条件及表/遮蔽代码哈希；宿主仅收到其组内已选候选的观察，不获得另一组反馈。

产物位于 `reports/feedback_ablation/{aggregate,class_feedback}/`，顶层 comparison.json 汇总结果。delivered-NNNN.json 文件保存的是接口生成的可发送视图；本轮仅发送 revision 0/1，终态 revision 2 未再次发给宿主。实际发送时省略双方相同的冗余来源标识/已知规则段，保留全部可见候选指标和诊断；不能声称所有文件字节都原样作为宿主消息。两组原始提案保存在状态 actions 内，观察哈希均与前序状态匹配。

## 实际结果

| 条件 | 第一步 | 第二步 | 最终分数 | 使用候选数 |
|---|---|---|---:|---:|
| 详细类别反馈 | class_weight | midpoint_smote | 0.749084 | 3 |
| 仅总体反馈 | midpoint_smote | class_weight | 0.749084 | 3 |

两组均使用 LightGBM、完整特征，全部新候选未满足类别约束，最终保留 paper_smote。详细组在第二步理由中指出“四类召回跌破约束”并提及具体岩性风险；总体组只指出“汇总反馈未说明具体失败项”。这种文字差异可观察，但不是性能改善证据。

首步选择在任何候选诊断出现之前已经不同，第二步看到的历史也不同，因此不能把候选顺序差异因果归于反馈。只有一条轨迹/条件，无显著性分析、无模型方差估计；相同候选集在当前屏障下没有改善机会，终点得分缺少区分力。

新增训练0次，复用之前40次拟合生成的表；新增两个模型上下文、四次实际决策。未计量宿主token/推理耗时，不宣称计算收益。该工作是可执行反馈消融的先导，不是已完成充分的论文消融证据。

下一步若继续评估反馈作用，应固定同一中间候选状态，再向新上下文分别提供两种视图，重复多次并对首步选择进行配对或控制；同时预先定义不依赖最终得分的诊断准确性或约束识别指标。不得仅挑选有利的两条轨迹，或继续在已知目录上调阈值制造优势。

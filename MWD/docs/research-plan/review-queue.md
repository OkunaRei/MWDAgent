# 拒答证据复核队列

## 2026-09-08 快照

执行 `python export_review_queue.py --output-dir reports/review_queue/2026-09-08`，从既有质量消融报告内嵌的逐样本预测生成队列，没有重新训练或读取公开测试集。

3,675 次验证出现对应 2,450 个唯一训练文件行；其中 1,228 行在至少一个种子下预测错误或未通过双门控，因此进入复核队列。186 行出现过高置信度误判，241 行出现过正确预测被统计代理额外送审。其余队列项还包括低置信度预测；这些集合一般允许重叠，本快照前两组交集为零。

导出位置：`reports/review_queue/2026-09-08/queue.csv` 和 `queue.json`。JSON 保存结构化特征频数和种子列表；CSV 中这些列使用 JSON 字符串。每行一个唯一样本，保留所有验证出现次数作为分母。按高置信度误判次数、额外送审正确预测次数、误判次数降序排列，最后按行号稳定排序。排序只是复核工作顺序，次数不代表独立重复证据或风险概率。

## 可追溯性

导出时校验训练源文件 SHA-256、每个种子的分区索引哈希、分区互斥且完整、验证预测覆盖率、种子、类别、布尔判断和阈值的一致性。行索引是零起始 CSV 数据行位置，不是 CSV 首列原始编号。源文件 Rock 与 transition_zone 必须与预测记录匹配；位置字段 Tunnel、PegStart、PegEnd 从经过哈希校验的源文件读取。

该验证保障文件关联的一致性，不能替代数据来源的地质真实性验证。队列快照记录源文件和实验报告哈希。人工复核文件不能由重新运行覆盖：输出目录必须不存在，后续导出需用新目录。

## 复核字段

初始 review_status 为 pending；quality_label、evidence_reference、reviewer、reviewed_at、notes 全部留空。不得从模型正确率、置信度或 IQR 越界直接填入质量标签。

质量结论词表：measurement_issue（测量或对齐问题）、geological_variation（真实地质差异）、mixed（两者并存）、insufficient_evidence（证据不足）。review_status 可在 pending、in_review、completed 之间变化；完成项须保留复核者、YYYY-MM-DD 日期与说明。具体质量结论必须有证据引用；证据不足项允许引用为空，但必须说明缺失信息。

测量问题应引用原始传感器日志、校验记录、停钻/接杆记录、时间与深度同步证据。真实地质异常应引用地质编录、取芯、孔内观测或独立现场记录。仅有公开统计特征不能确认测量故障；没有独立证据时填写证据不足，并说明缺失资料。预测错误不等价于标签错误，预测正确也不证明输入质量合格。

## 后续研究入口

人工填写后运行：

```bash
.venv/bin/python validate_review_queue.py \
  --snapshot reports/review_queue/2026-09-08/queue.json \
  --annotations reports/review_queue/2026-09-08/queue.csv \
  --output-dir reports/review_validation/next-review
```

快照 JSON 应保留原样，仅编辑 CSV 的六个复核字段。必须提交完整队列，可以重新排序；不得删除未完成项。校验器拒绝缺行、重复行、只读字段变化、非法状态、未知标签和不完整完成项。输出目录必须不存在，避免覆盖前次审计。

`validation.json` 记录源文件、实验、快照和填写文件哈希，以及逐行错误与状态计数。任何一行校验失败时退出码为 1，并且不导出任何完成结论；有效报告中的 completed_reviews 只包含完成项。insufficient_evidence 可以是完成的复核，但不计入 evidence_backed_count。这个计数只表示填写了规定字段，不意味着外部证据已被独立核验；本流程不生成训练标签、不改写数据集、不触发训练。

2026-09-08 对现有未填写队列执行校验：1,228 行全部 pending，completed 为 0，evidence_backed_count 为 0，结构校验通过。结果保存在 `reports/review_validation/2026-09-08/validation.json`。这是接口验证，不是已完成人工质量标注。

优先复核高置信度误判，再检查质量代理额外拒答的正确样本，兼顾岩性和过渡区覆盖。当前是事后诊断队列，具有选择偏差；不能用其复核结果直接估计全量故障率。未来质量模型的评估需要另行定义抽样方案及冻结验证集，不能将本队列重复用于调参后宣称独立验证提升。

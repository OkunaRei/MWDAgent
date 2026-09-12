# Agent 优化随钻地层识别模型：论文思路

> 定位更新（2026-09-06）：本文保留为模型研发 Agent 的候选方案，不作为项目唯一主线。执行阶段与验收以[总体路线](../../../docs/project-roadmap.md)为准：先验证空间区块评估、校准与拒识，再以同预算对照检验 Agent 的独立增益。下文跨隧道题目与实验为条件性设想，须先解决类别覆盖与未知类别问题；现场 Agent 和注浆闭环另依赖自有数据。

## 1. 建议题目

中文：**面向过渡区识别的知识约束随钻地层识别模型优化 Agent**

英文：**A Knowledge-Constrained Agent for Optimizing MWD-Based Geological Prediction with Transition-Aware Evaluation**

## 2. 核心观点

Agent 不直接替代地质预测模型，也不直接生成注浆设备控制指令。它负责审计数据、选择特征、选择模型、诊断错误、检查地质约束和组织重新训练；LightGBM、ExtraTrees、TCN 等专业模型负责数值预测。

研究重点不是证明 Agent 能自动调参，而是验证：

> Agent 能否根据过渡区等地质错误切片识别模型失败原因，并在知识约束下选择更有利于稳定识别的下一步优化动作。

## 3. 研究链路

```text
MWD 数据
  -> 数据审计与质量评分
  -> 候选特征/模型/采样策略
  -> 专业模型训练
  -> 普通区与过渡区评价
  -> 错误诊断
  -> Agent 选择下一优化动作
  -> 重新训练与记录证据
```

在总项目中，该模块属于“随钻感知 Agent + 地质认知 Agent”的模型研发支撑层，后续再连接注浆决策 Agent 和注浆反馈 Agent。

## 4. 当前可验证任务

使用 Hansen 等人的公开 MWD 数据：

- 数据：<https://doi.org/10.5281/zenodo.10358374>
- 论文：<https://doi.org/10.1016/j.tust.2024.105843>
- 主任务：48 个 MWD 特征预测 `Rock`；
- 辅助切片：`transition_zone=True` 与普通区；
- 评价切片：公开固定 train/test 下的普通区与过渡区。

当前公开数据不包含富水破碎带、Lugeon、注浆量或注浆过程曲线。因此第一篇不能声称完成现场注浆预测，只能验证 MWD 地层识别和面向预注浆的接口能力。

## 5. Agent 的允许动作

Agent 每轮只能从预定义动作集合中选择：

1. 选择特征组；
2. 选择候选模型；
3. 调整有限范围内的超参数；
4. 调整类别权重或训练集采样；
5. 调整数据质量阈值；
6. 请求错误分析或停止优化。

每轮保存动作、配置、随机种子、数据版本、指标、错误切片和下一步理由。禁止无限试错和使用测试集反馈调参。

## 6. 知识约束

- 不使用 `Tunnel`、里程和未来标签作为预测特征；
- 不将相邻空间窗口随机拆到训练和测试两侧；
- 论文固定测试集只用于最终复现，不用于 Agent 每轮优化；
- 不因总体 Accuracy 提高而牺牲过渡区或异常召回率；
- 低质量数据不能输出不相称的高置信度；
- 删除特征前必须说明物理含义和潜在影响；
- 只允许在规定的模型、参数和计算资源范围内搜索。

## 7. 对照实验

至少比较：

1. 人工固定流程 + LightGBM；
2. 普通随机搜索或 Optuna；
3. 无知识约束的 LLM Agent；
4. 知识约束 Agent。

主要消融：

- 无错误切片反馈；
- 无地质错误切片；
- 无数据质量门控；
- 无概率校准；
- 无规则约束。

## 8. 评价指标

模型指标：

- Accuracy、Balanced Accuracy、Macro-F1；
- 各类别 Recall 和混淆矩阵；
- 普通区与过渡区分别评价；
- 按隧道的性能均值和最差值；
- 概率校准误差和低置信度拒识率。

Agent 指标：

- 达到目标性能所需迭代次数；
- 跨随机种子的性能方差；
- 无效动作比例；
- 测试集泄漏次数；
- 违反知识约束次数；
- 运行时间和 LLM 调用次数；
- 错误诊断与后续性能改善的一致性。

## 9. 论文贡献的成立条件

只有满足以下证据，才能声称 Agent 方法有效：

1. 在相同数据、候选模型和计算预算下，知识约束 Agent 的过渡区结果和整体稳定性优于固定流程和普通搜索；
2. 提升主要来自对错误切片和数据质量的正确诊断，而不是测试集调参；
3. Agent 产生的每个优化动作都有可追溯理由；
4. 当数据质量下降或模型不确定性升高时，系统能选择保守路径或停止输出；
5. 在不同随机种子和普通区/过渡区切片上结果稳定。

## 10. 与注浆研究的连接

取得现场配对数据后，将目标从 `Rock` 扩展为：

```text
MWD + 水流/失水 + 地质编录
  -> 完整围岩 / 破碎围岩 / 富水破碎带
```

再增加：

```text
MWD + WPT/Lugeon + 注浆工艺
  -> 注浆状态 / 注浆量 / 注浆时间 / 注后效果
```

Agent 的作用仍然是模型研发、证据组织和规则检查；注浆压力、流量和水灰比必须由专业响应模型与确定性安全规则共同约束，并由工程师审批。

## 11. 关键参考依据

- Luo & Xiao (2026), *Large language model-based AI agent for well logging data processing and interpretation*, Petroleum Science, DOI: <https://doi.org/10.1016/j.petsci.2026.05.031>。LogACF 提供了角色—记忆—规划—执行架构、预定义工程工作流、错误诊断—修正循环和物理质量门控的参考，但其对象是井测储层参数，不是 MWD 或注浆。
- Hansen, Liu & Torresen (2024), *Predicting rock type from MWD tunnel data using a reproducible ML-modelling process*, DOI: <https://doi.org/10.1016/j.tust.2024.105843>。提供当前公开 MWD 岩性基准和可复现监督学习任务。
- Hansen & Aarset (2024/2025), *Unsupervised Machine Learning for Data-Driven Rock Mass Classification: Addressing Limitations in Existing Systems Using Drilling Data*, DOI: <https://doi.org/10.1007/s00603-024-04280-z>。支持用 MWD 形成岩体状态表征，并强调其对地下工程决策的潜力。
- van Eldert et al. (2021), *Drill Monitoring for Rock Mass Grouting: Case Study at the Stockholm Bypass*, DOI: <https://doi.org/10.1007/s00603-020-02279-w>。提供 MWD Fracturing Index 与注浆消耗、注浆需求判别的直接工程背景，但原始配对数据未公开。
- Rongved, Hansen & Erharter (2024), *Toward machine learning based decision support for pre-grouting in hard rock*, DOI: <https://doi.org/10.1002/cend.202400012>。说明常规 MWD、地质和施工字段对总注浆量/注浆时间的预测能力有限，支持引入更细粒度水文与过程反馈信息。

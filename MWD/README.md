# MWDAgent

面向地下工程的随钻感知（MWD）地层识别与注浆决策研究模块。

本模块服从总项目《地下空间工程钻注一体化地质认知与注浆决策智能体研究方案》。完整的研究目标不是单独做一个 MWD 分类器，而是构建“随钻感知—地质认知—注浆方案生成—施工反馈修正”的人在回路闭环。总方案对齐说明见 [`docs/research-plan/project-plan-alignment.md`](docs/research-plan/project-plan-alignment.md)。

当前阶段聚焦于总方案第一阶段的可公开复现实验：

```text
MWD 钻进响应 -> 数据质量与地层状态识别 -> 不确定性输出 -> 注浆决策接口
```

## 当前核心入口：实验 Agent

`run_mwd_agent.py` 提供语言模型宿主可调用的实验闭环：读取验证观察、提交结构化候选动作、执行五种子评估、检查过渡区约束、更新或保留配置、停止。宿主提出动作，执行器只运行白名单内的专业模型；不内置远端 LLM API。

```bash
.venv/bin/python run_mwd_agent.py init --session reports/agent_loop/new-session --budget 4
.venv/bin/python run_mwd_agent.py observe --session reports/agent_loop/new-session
.venv/bin/python run_mwd_agent.py act --session reports/agent_loop/new-session --action proposal.json
```

动作必须引用最新观察哈希，不能指定测试文件、任意代码或未经允许的参数。四个候选的预算包含基线；只有综合分数改善且整体/过渡区 Macro-F1 不低于基线，才能替换当前配置。接口、动作格式和限制见 [`docs/research-plan/agent-loop.md`](docs/research-plan/agent-loop.md)。`init` 不会独立生成后续动作，需要模型宿主继续调用 `observe` 与 `act`。

首轮宿主闭环已完成三个模型提出的候选动作、四组配置共 20 次训练，预算耗尽后保留完整特征 LightGBM。提案、观察版本与评估证据保存在 `reports/agent_loop/2026-09-08/`；这证明执行闭环已贯通，尚未证明 Agent 优于普通搜索。

同预算顺序回放已接入 `compare_agent_search.py --output-dir <新目录>`。它比较固定顺序、实际宿主顺序和全部六种随机顺序，复用既有结果、不重新训练。当前各预算点全部持平，完整目录中无可晋级候选；协议及结果见 [`docs/research-plan/search-comparison.md`](docs/research-plan/search-comparison.md)。

## 当前复现对象

首个公开数据复现实验采用 Hansen & Aarset (2024)：

> Unsupervised Machine Learning for Data-Driven Rock Mass Classification: Addressing Limitations in Existing Systems Using Drilling Data

- DOI: <https://doi.org/10.1007/s00603-024-04280-z>
- 公开数据：<https://doi.org/10.5281/zenodo.10358374>
- 数据内容：15 条挪威硬岩隧道、MWD 特征、岩性标签、Q 值/Q-class 等

公开数据阶段复现：

1. 使用 MWD 特征复现岩性和过渡区识别；
2. 使用 LightGBM、ExtraTrees 等模型建立有监督基线；
3. 增加普通区与过渡区的分层评价；
4. 增加数据质量评分和概率校准接口（当前已提供 ECE 校准误差）；
5. 为后续富水破碎带识别和注浆决策提供数据契约。

## 注浆相关边界

公开 MWD 数据不包含现场注浆量、注浆时间、Lugeon 或压力-流量-累计量曲线。因此当前仓库不能声称已经完成现场注浆量预测。

与注浆直接相关的对照论文是：

van Eldert et al. (2021), *Drill Monitoring for Rock Mass Grouting: Case Study at the Stockholm Bypass*，DOI <https://doi.org/10.1007/s00603-020-02279-w>。

该论文的 MWD-注浆配对原始数据未公开，当前作为工程规则和评价协议参考。待获得项目现场数据后，再将其注浆类别、注浆量和水压试验结果接入本项目。

## 目录

```text
data/
  mwd_rocktype_10358374/       公开 MWD 数据
docs/
  literature/                  论文与复现任务卡
  research-plan/               文献检索与研究方案
  data-provenance/             数据来源、许可和版本记录
```

## 数据使用

公开数据文件位于 `data/mwd_rocktype_10358374/`。数据集页面标注为 CC BY 4.0，同时注明仅限科研、不可商业使用。使用时必须保留原作者、数据来源和论文引用；本项目不重新声明第三方数据的所有权。

本地验证文件哈希和数据版本见 `docs/data-provenance/mwd_rocktype_10358374.md`。

## 论文基线复现

当前实现复现 Hansen et al. (2024) 的十类 `Rock` 预测主流程：48 个 MWD 统计特征、公开固定训练/测试划分、训练集欠采样到第二大类后使用 SMOTE、LightGBM `dart` 参数。

```bash
python3 -m venv .venv
.venv/bin/pip install --only-binary=:all: -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python train_baseline.py
.venv/bin/python evaluate_slices.py
.venv/bin/python run_benchmark.py
.venv/bin/python optimize_baseline.py
```

输出位置：

```text
reports/baseline/metrics.json
reports/baseline/classification_report.csv
reports/baseline/confusion_matrix.png
reports/slices/metrics.json
reports/slices/slice_summary.csv
reports/benchmark/benchmark.json
reports/benchmark/validation_summary.csv
reports/benchmark/optimization_log.jsonl
reports/optimizer/optimizer.json
reports/optimizer/candidate_summary.csv
reports/optimizer/optimization_log.jsonl
```

首轮运行结果与解释见 [`docs/research-plan/hansen2024-baseline-reproduction.md`](docs/research-plan/hansen2024-baseline-reproduction.md)。

## Agent 接入前基准平台

在接入 LLM Agent 前，使用 `run_benchmark.py` 固定模型选择和评估协议：公开训练集内部按种子划分训练/验证集，使用验证集平均分选择模型，公开测试集只在最后用预注册的 `seed=42` 评估一次。

```bash
.venv/bin/python run_benchmark.py
```

输出位置：

```text
reports/benchmark/benchmark.json
reports/benchmark/validation_summary.csv
reports/benchmark/optimization_log.jsonl
```

综合选择分数为：

```text
0.2 × Balanced Accuracy
+ 0.5 × Macro-F1
+ 0.3 × transition_zone Macro-F1
```

所有整体和切片评估同时记录 `expected_calibration_error`（ECE，固定等宽置信度分箱）；ECE 越低表示概率置信度越接近实际正确率。

下游研究接口提供 `prediction_gate(quality=..., confidence=...)`：只有数据质量和预测置信度同时达到阈值才返回 `accept`；单项不足返回 `review`，两项都很低返回 `reject`。默认阈值是研究占位参数，接入现场质量评分后需在验证集预注册。这里的 `accept` 只表示保留模型预测，不代表批准注浆施工；ECE 是批量统计指标，不能代替单样本置信度或独立质量评分。

## 温度校准实验

```bash
.venv/bin/python run_calibration.py
```

只读取公开训练集，分层划分约 60% 模型训练、20% 温度拟合、20% 留出验证。欠采样和 SMOTE 仅用于模型训练部分；温度仅用校准部分的对数损失拟合。对五个固定种子报告校准前后 ECE、对数损失及置信度门控的覆盖率和错误率，包含普通区和过渡区切片。公开测试集不参与本实验。

结果写入 `reports/calibration/`，实验说明见 [`docs/research-plan/calibration-results.md`](docs/research-plan/calibration-results.md)。置信度阈值曲线属于探索性验证；目前尚无独立现场质量评分，因此不声称完成双门控实证验证。

### 统计异常代理门控对照

```bash
.venv/bin/python run_calibration.py --quality-ablation
.venv/bin/python summarize_rejections.py
.venv/bin/python export_review_queue.py --output-dir reports/review_queue/2026-09-08
```

使用原始训练分区的逐特征 Tukey 边界（Q1 - 1.5 IQR 至 Q3 + 1.5 IQR），计算验证样本落在边界内的特征比例。固定代理阈值 0.9、校准后置信度阈值 0.8，对照无门控、仅统计代理、仅置信度和双门控。输出独立写入 `reports/quality_ablation/`，包含每个种子、地质切片和类别的保留统计。

这个统计评分不等价于传感器质量：真实地质异常也可能落在边界外。门控拒绝的样本应进入人工复核，不能据此认定数据错误或删除地质证据。结果与局限见 [`docs/research-plan/quality-ablation-results.md`](docs/research-plan/quality-ablation-results.md)。

去重复核队列的字段和证据要求见 [`docs/research-plan/review-queue.md`](docs/research-plan/review-queue.md)。导出命令要求输出目录尚不存在，避免覆盖人工批注；再次导出应使用新目录名。当前快照没有人工确认的质量标签。

人工填写后使用 `validate_review_queue.py --snapshot <原始queue.json> --annotations <填写后的queue.csv> --output-dir <新目录>` 校验。它检查只读字段和完成项证据要求，生成独立审计报告，不修改原队列或训练数据。

详细协议和首轮结果见 [`docs/research-plan/pre-agent-benchmark.md`](docs/research-plan/pre-agent-benchmark.md)。后续 Agent 只能提出候选配置并读取验证结果，不能使用公开测试集反馈调参。

有限搜索结果见 [`docs/research-plan/baseline-optimizer-results.md`](docs/research-plan/baseline-optimizer-results.md)。当前结果显示，5 个特征组与 2 个模型的搜索仍选择完整 48 特征 + LightGBM；下一轮应优先研究过渡区错误、质量门控和概率校准，而不是继续盲目删减特征。

## 论文方向

本模块拟研究“知识约束 Agent 优化 MWD 地层识别模型”，而不是让大语言模型直接预测地层或注浆参数。Agent 负责数据审计、特征和模型选择、错误诊断、约束检查和实验编排；专业模型负责数值预测。完整论文思路见 [`docs/research-plan/agent-optimized-mwd-paper-idea.md`](docs/research-plan/agent-optimized-mwd-paper-idea.md)。

## 对齐总方案的后续阶段

- `随钻感知 Agent`：时间同步、深度对齐、工况识别、区段划分和质量门控；
- `地质认知 Agent`：可靠地质属性、富水破碎异常和不确定性；
- `注浆决策 Agent`：案例检索、工程规则、安全屏障和人在回路建议；
- `注浆反馈 Agent`：压力—流量—累计注入量状态识别与历史过程回放。

获得真实钻孔—注浆段配对数据后，再训练注浆量、注浆时间和治理效果模型。

## 研究资料说明

论文 PDF、Office 文档、临时分析文件和生成结果保留在本地研究目录，不进入公开仓库。仓库只维护可公开共享的结构化数据、文档和后续代码。

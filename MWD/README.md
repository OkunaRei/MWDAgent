# MWDAgent

面向地下工程的随钻感知（MWD）地层识别与注浆决策研究模块。

项目执行顺序与阶段验收见[总体路线](../docs/project-roadmap.md)，自有数据准备见[现场数据契约](../docs/field-data-contract.md)。公开数据用于先行验证方法，后续接入自有数据完成工程验证；当前分类模型的权重、标签和阈值不默认直接迁移。

本模块服从总项目《地下空间工程钻注一体化地质认知与注浆决策智能体研究方案》。完整的研究目标不是单独做一个 MWD 分类器，而是构建“随钻感知—地质认知—注浆方案生成—施工反馈修正”的人在回路闭环。总方案对齐说明见 [`docs/research-plan/project-plan-alignment.md`](docs/research-plan/project-plan-alignment.md)。

当前阶段聚焦于总方案第一阶段的可公开复现实验：

```text
MWD 钻进响应 -> 数据质量与地层状态识别 -> 不确定性输出 -> 注浆决策接口
```

## 当前复现对象

当前已实现的监督分类基线采用 Hansen, Liu & Torresen (2024)：

> Predicting rock type from MWD tunnel data using a reproducible ML-modelling process

- DOI: <https://doi.org/10.1016/j.tust.2024.105843>
- 公开数据：<https://doi.org/10.5281/zenodo.10358374>
- 当前分类 CSV：15 条挪威硬岩隧道、48 个 MWD 汇总特征、岩性、位置和过渡区等字段；不是单孔原始连续时序。

Hansen & Aarset 的[无监督岩体分类研究](https://doi.org/10.1007/s00603-024-04280-z)作为相关研究保留，其研究目标与当前已实现的岩性监督分类基线区分。

公开数据阶段复现：

1. 使用 MWD 特征复现岩性预测，并按过渡区做离线评价；
2. 使用 LightGBM、ExtraTrees 等模型建立有监督基线；
3. 保留论文固定划分，新增空间区块隔离诊断；按隧道留出前先处理缺类与未知类别问题；
4. 增加数据质量评分和概率校准接口；
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
```

输出位置：

```text
reports/baseline/metrics.json
reports/baseline/classification_report.csv
reports/baseline/confusion_matrix.png
reports/slices/metrics.json
reports/slices/slice_summary.csv
```

首轮运行结果与解释见 [`docs/research-plan/hansen2024-baseline-reproduction.md`](docs/research-plan/hansen2024-baseline-reproduction.md)。

## 论文方向

近期研究主线是空间相关条件下的地质识别可靠性，以及质量、校准和拒识对后续决策的价值。Agent 负责数据审计、错误诊断、约束检查和条件编排，其相对固定工作流或普通搜索的收益需要独立验证。既有[模型优化 Agent 论文思路](docs/research-plan/agent-optimized-mwd-paper-idea.md)保留为候选分支，具体优先级和贡献成立条件以[总体路线](../docs/project-roadmap.md)为准。

## 对齐总方案的后续阶段

- `随钻感知 Agent`：时间同步、深度对齐、工况识别、区段划分和质量门控；
- `地质认知 Agent`：可靠地质属性、富水破碎异常和不确定性；
- `注浆决策 Agent`：案例检索、工程规则、安全屏障和人在回路建议；
- `注浆反馈 Agent`：压力—流量—累计注入量状态识别与历史过程回放。

获得真实钻孔—注浆段配对数据后，再训练注浆量、注浆时间和治理效果模型。

## 研究资料说明

论文 PDF、Office 文档、临时分析文件和生成结果保留在本地研究目录，不进入公开仓库。仓库只维护可公开共享的结构化数据、文档和后续代码。

# MWDAgent

面向地下工程的随钻感知（MWD）地层识别与注浆决策研究项目。

当前阶段聚焦于一个可公开复现、可逐步扩展的研究链条：

```text
MWD 钻进响应 -> 岩体/地层状态识别 -> 注浆需求风险分层 -> 注浆过程预测
```

## 当前复现对象

首个公开数据复现实验采用 Hansen & Aarset (2024)：

> Unsupervised Machine Learning for Data-Driven Rock Mass Classification: Addressing Limitations in Existing Systems Using Drilling Data

- DOI: <https://doi.org/10.1007/s00603-024-04280-z>
- 公开数据：<https://doi.org/10.5281/zenodo.10358374>
- 数据内容：15 条挪威硬岩隧道、MWD 特征、岩性标签、Q 值/Q-class 等

第一阶段复现：

1. 使用 MWD 特征识别岩体状态簇；
2. 使用岩性、Q 值和 Q-class 解释状态簇；
3. 使用 LightGBM、ExtraTrees 等模型建立有监督预测对照；
4. 增加按隧道留出的泛化验证；
5. 将岩体状态映射为注浆需求风险等级代理。

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

## 下一步

- 复现无监督聚类主线；
- 复现岩性/Q-class 有监督基线；
- 加入跨隧道留出、过渡带和概率校准；
- 获取真实钻孔水文、注浆和水压试验数据后，训练注浆需求与注浆过程模型。

## 研究资料说明

论文 PDF、Office 文档、临时分析文件和生成结果保留在本地研究目录，不进入公开仓库。仓库只维护可公开共享的结构化数据、文档和后续代码。

# 复现任务卡：Hansen & Aarset (2024) MWD 岩体分类与注浆决策基础

## 1. 论文与数据

- 论文：**Unsupervised Machine Learning for Data-Driven Rock Mass Classification: Addressing Limitations in Existing Systems Using Drilling Data**
- 作者：Tom F. Hansen, Arnstein Aarset
- 期刊：Rock Mechanics and Rock Engineering, 2024
- DOI：<https://doi.org/10.1007/s00603-024-04280-z>
- 数据集：**Measure While Drilling (MWD) dataset with rock type labels for 15 Norwegian hard rock tunnels**
- Zenodo DOI：<https://doi.org/10.5281/zenodo.10358374>

## 2. 为什么选它

这篇论文用 MWD 数据形成岩体状态簇，并将簇与岩性、岩体质量和地下工程决策联系起来。论文明确讨论了将这种数据驱动岩体分类用于支护、开挖设计和注浆工作量决策的潜力。

它不声称已经预测了现场注浆量，因此复现边界必须写清楚：第一阶段复现的是“MWD -> 岩体状态/地层类别”，第二阶段才接入注浆标签或工程规则。

## 3. 本地数据

数据目录：

`MWD/data/mwd_rocktype_10358374/`

主要文件：

- `mwd_rocktype_blastholes_model_ready_full.csv`
- `mwd_rocktype_blastholes_raw.csv`
- `mwd_rocktype_blastholes_model_ready_train.csv`
- `mwd_rocktype_blastholes_model_ready_test.csv`

数据包含按 1 m 隧道区段聚合的 MWD 特征、隧道标识、岩性标签、Q 值/Q-class 等字段。当前本地文件哈希和样本版本已在 Hansen 2024 任务卡中记录。

## 4. 先复现论文主线

### RQ1：无监督地层状态识别

1. 读取 48 个 MWD 统计特征。
2. 进行 PCA/UMAP 表征降维。
3. 复现 HDBSCAN、层次聚类和 K-means。
4. 通过聚类稳定性、轮廓系数和空间区段一致性选择状态簇。
5. 用 `Rock`、`Q-class`、`Q` 对聚类结果进行外部解释，而不是把标签直接用于聚类训练。

### RQ2：有监督地层预测对照

将 `Rock` 或 `Q-class` 作为预测目标，比较 LightGBM、ExtraTrees 和 MLP。使用公开固定训练/测试划分，并对普通区与过渡区分别评价。

## 5. 注浆相关扩展

由于公开数据没有现场总注浆量、注浆时间或水压试验结果，不能训练真实的注浆量监督模型。可以先建立两种透明的注浆决策代理：

### 方案 A：工程规则代理

根据岩体状态簇的 Q-class、裂隙相关 MWD 特征和水流特征，构造低/中/高注浆需求等级。该等级必须标注为工程规则标签，不能称为真实注浆标签。

### 方案 B：外部注浆论文规则迁移

参考 van Eldert et al. (2021) 的 MWD Fracturing Index 和 Stockholm Bypass 注浆类别，将公开 MWD 数据中的地层状态解释为“完整岩体、裂隙岩体、局部破碎区”等预注浆风险类别。由于没有原论文配对数据，只能做规则方法复现或概念迁移，不能复现其 85%/93% 工程准确率。

## 6. 最终实验结构

```text
MWD特征
   |
   +--> 无监督岩体状态簇
   |       |
   |       +--> 岩性/Q-class解释
   |       +--> 注浆需求等级代理
   |
   +--> 有监督 Rock/Q-class预测
           |
           +--> 分层评价和不确定性
```

## 7. 必须避免的表述

- 不能把公开 MWD 数据称为“注浆数据集”。
- 不能把聚类得到的注浆需求等级称为真实注浆量预测。
- 不能把 Stockholm Bypass 论文的工程准确率迁移到本数据集。
- 在没有真实水压试验、注浆量或注浆曲线前，论文题目应写成“面向注浆决策的 MWD 岩体分类”，而不是“注浆量预测”。

## 8. 直接注浆论文对照

van Eldert et al. (2021), **Drill Monitoring for Rock Mass Grouting: Case Study at the Stockholm Bypass**，DOI：<https://doi.org/10.1007/s00603-020-02279-w>。

该论文与注浆最贴切，但其 97 个注浆伞、2646 个注浆孔和 MWD 配对数据没有公开下载入口。因此将其作为工程规则和评价设计参考，不作为第一篇严格数据复现论文。

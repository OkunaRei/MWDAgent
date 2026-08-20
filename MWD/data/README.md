# Data

本目录只放可公开共享、且与当前复现任务直接相关的数据。

## MWD rock-type dataset

目录：`mwd_rocktype_10358374/`

来源：

- Zenodo record: <https://doi.org/10.5281/zenodo.10358374>
- 数据题目：Measure While Drilling (MWD) dataset with rock type labels for 15 Norwegian hard rock tunnels
- 论文：Hansen, Liu & Torresen, *Predicting rock type from MWD tunnel data using a reproducible ML-modelling process*, DOI <https://doi.org/10.1016/j.tust.2024.105843>
- 研究方法论文：Hansen & Aarset (2024), DOI <https://doi.org/10.1007/s00603-024-04280-z>

文件说明：

- `*_raw.csv`：清洗、处理和异常值删除前的数据；
- `*_model_ready_full.csv`：清洗后的完整数据；
- `*_model_ready_train.csv`：公开训练划分；
- `*_model_ready_test.csv`：公开测试划分。

许可证和限制：Zenodo 页面标注 CC BY 4.0，并注明仅限科研、不可商业使用。发布或使用这些文件时应保留原作者和数据来源信息。

# cell_STFlow 所需数据与权重

本目录包含细胞级 STFlow 流匹配去噪器源码。**模型权重与训练输出未包含。**

## 需要的外部数据

适配器从共享 benchmark 接口读取预计算的 UNI 特征：

```
/home/sb202604/cell-benchmark/processed_cell/<dataset>/
├── adata_S1.h5ad      # obsm['he'] 为 UNI 特征，obsm['spatial'] 为空间坐标
└── adata_S2.h5ad
```

`config.yaml` 中涉及的数据集：
- `hSkin_Melanoma`
- `hColon_Non_diseased`
- `mouse_Colon`
- `Human_Breast_Cancer`

## 需要的基础模型权重

运行时不需要额外基础模型权重——代码直接使用 `obsm['he']` 中的预计算 UNI 特征。

## 需要的外部源码包

`model.py` 从独立的 STFlow 仓库导入去噪器/插值器：

```python
sys.path.insert(0, "/home/sb202604/STFlow")
from stflow.model.denoiser import Denoiser
from stflow.flow.interpolant import Interpolant
from stflow.model.transformer import SpatialTransformer
```

你需要有 `/home/sb202604/STFlow`（或等价的 pip 安装版 `stflow` 包），并相应修改该路径。

## 共享 benchmark 辅助脚本

`train.py` 通过仓库内相对路径（`Path(__file__)` 定位 `cell-benchmark/scripts`）导入 `cell_data`，无需修改路径。

## 可选依赖

若使用 `prior_sampler: zinb`，需要安装 `scvi-tools`。

## 运行时生成的目录

- `runs/<dataset>/` — checkpoint、预测 (`pred_*.npy`)、`result.json` / `summary.json`。

以上已加入 `.gitignore`。

## 原始方法链接

- STFlow 官方仓库：[Graph-and-Geometric-Learning/STFlow](https://github.com/Graph-and-Geometric-Learning/STFlow)

## 需要修改路径的配置文件

- `config.yaml`：`output_root`

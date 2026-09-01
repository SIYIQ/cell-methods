# cell_STFlow

STFlow 流匹配去噪器的细胞级适配版本，用于共享 `cell-benchmark` 的 Xenium 数据集。

## 依赖安装

- 本目录**未提供 `requirements.txt`**。依赖请参照原仓库 [Graph-and-Geometric-Learning/STFlow](https://github.com/Graph-and-Geometric-Learning/STFlow) 或 HEST 框架 [mahmoodlab/HEST](https://github.com/mahmoodlab/HEST) 的安装说明。
- 核心依赖大致包括：`torch`、`torch_geometric`、`scanpy`、`anndata`、`numpy`、`pandas`、`scipy`、`scikit-learn`、`pyyaml`。
- **必须额外 clone STFlow 原仓库**：`model.py` 从 STFlow 仓库导入 `Denoiser`、`Interpolant`、`SpatialTransformer`：
  ```bash
  git clone https://github.com/Graph-and-Geometric-Learning/STFlow.git /path/to/STFlow
  ```
  然后通过环境变量指定路径（**无需改代码**）：
  ```bash
  export STFLOW_ROOT=/path/to/STFlow
  ```
  代码默认路径为 `/home/sb202604/STFlow`，所以如果不设置环境变量，需要把 STFlow clone 到该路径。
- 若使用默认的 `prior_sampler: zinb`，需要安装 `scvi-tools`：
  ```bash
  pip install scvi-tools
  ```
  如果环境里没有 `scvi`，可在 `config.yaml` 中把 `prior_sampler` 改为 `gaussian`。

## 数据与权重

详见 `DATA_WEIGHTS.md`。输入使用共享 benchmark 的 `processed_cell/` 接口：

```
cell-benchmark/processed_cell/<dataset>/
├── adata_S1.h5ad      # obsm['he'] 为 UNI 特征，obsm['spatial'] 为空间坐标
└── adata_S2.h5ad
```

运行时不需要额外基础模型权重——代码直接使用 `obsm['he']` 中的预计算 UNI 特征。

## 路径修改

- （可选）通过 `export STFLOW_ROOT=...` 设置 STFlow 原仓库路径（默认 `/home/sb202604/STFlow`），避免改代码。

共享脚本的 sys.path 已改为仓库内相对定位，`config.yaml` 的 `output_root` 默认为 `./runs`，均无需修改。

## 运行

```bash
python train.py --config config.yaml
python train.py --config config.yaml --dataset hSkin_Melanoma
python train.py --config config.yaml --dataset hSkin_Melanoma --seeds 42 --device cpu
```

## 输出

```
runs/<dataset>/
├── seed_42/
│   ├── pred_S1_from_S2.npy
│   ├── pred_S2_from_S1.npy
│   ├── per_gene_pcc_S1.npy
│   ├── per_gene_pcc_S2.npy
│   └── result.json
├── seed_43/...
├── seed_44/...
└── summary.json
```

## 训练细节

- 固定 `epochs` 训练，无验证集早停，最终模型直接用于测试。
- `config.yaml` 中的 `loss` 是死配置键：STFlow 使用其原生 MSE 流匹配损失，该键仅为与其他方法的接口对齐保留，仅在 `train.py` 写 `summary.json` 时被原样记录（硬取 `cfg[k]`），**不要从 config 中删除**，否则训练结束写 summary 时会 KeyError。
- 默认 `prior_sampler: zinb` 使用 `scvi-tools` 拟合 ZINB 先验；若不想安装 `scvi-tools`，可改为 `gaussian`。

## 文件结构

- `model.py` — `CellSTFlow`：UNI 特征 + STFlow `Denoiser`/`Interpolant`
- `train.py` — 跨切片训练（S1↔S2），3 个 seed
- `config.yaml` — 超参数
- `_legacy_HST/` — 原始 spot-level HST_STFlow 源码快照（参考用，不含 `runs/`）

## 原始方法链接

- STFlow 官方仓库：[Graph-and-Geometric-Learning/STFlow](https://github.com/Graph-and-Geometric-Learning/STFlow)

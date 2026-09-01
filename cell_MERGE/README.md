# cell_MERGE

MERGE 的细胞级适配版本。原版的 CNN 图像编码器被替换为在预计算 UNI 特征上训练的 MLP；GATNet 图精化阶段保留。

## 依赖安装

- 本目录**未提供 `requirements.txt`**。依赖请参照原仓库 [ags3927/MERGE](https://github.com/ags3927/MERGE) 或 HEST 框架 [mahmoodlab/HEST](https://github.com/mahmoodlab/HEST) 的安装说明。
- 核心依赖大致包括：`torch`、`torch_geometric`、`torch-sparse`（或 `pyg-lib`）、`scanpy`、`anndata`、`numpy`、`pandas`、`scipy`、`scikit-learn`、`pyyaml`。

## 数据与权重

详见 `DATA_WEIGHTS.md`。输入使用共享 benchmark 的 `processed_cell/` 接口：

```
cell-benchmark/processed_cell/<dataset>/
├── adata_S1.h5ad      # obsm['he'] 为 UNI 特征
└── adata_S2.h5ad
```

运行时不需要额外基础模型权重——代码直接使用 `obsm['he']` 中的预计算 UNI 特征。

## 路径修改

共享脚本的 sys.path 已改为仓库内相对定位，`config.yaml` 的 `output_root` 默认为 `./runs`，均无需修改。数据根目录由 `cell-benchmark/scripts/cell_data.py` 的 `PROCESSED_ROOT` 决定。

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

- 两阶段训练：先训练 `CellMLP`，再训练 `GATNet` 做图精化。
- GNN 阶段使用 PyG 的 `NeighborLoader` 做节点级 mini-batch。配置中：
  - `gnn_batch_size`：每批 seed 节点数；
  - `gnn_num_neighbors`：每个 GAT 层每跳采样的邻居数（列表长度必须等于 `GATNet` 层数，默认 4 层）。
- GNN 阶段固定 `gnn_epochs` 训练，无验证集与早停，最终权重直接用于预测。
- `config.yaml` 中的 `epochs` / `lr` 是死配置键：训练实际使用 `mlp_*` / `gnn_*` 系列键。`epochs` / `lr` 仅在 `train.py` 写 `summary.json` 时被原样记录（硬取 `cfg[k]`），**不要从 config 中删除**，否则训练结束写 summary 时会 KeyError。

## 文件结构

- `model.py` — `CellMLP` + `GATNet`
- `utils.py` — 基于 SpatialEx 超图构建细胞级图，可选层次聚类边
- `train.py` — 两阶段跨切片训练
- `config.yaml` — 超参数
- `_legacy_HST/` — 原始 HST_MERGE 源码快照（参考用，不含 `runs/`）

## 原始方法链接

- MERGE 官方仓库：[ags3927/MERGE](https://github.com/ags3927/MERGE)

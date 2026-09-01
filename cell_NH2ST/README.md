# cell_NH2ST

NH2ST（NGHist2ST）的细胞级适配版本。原版的 H0-mini 图像编码器被替换为在预计算 UNI 特征上的投影；邻域超图 + 对比学习 + 重建损失的流程保留。

## 依赖安装

- 本目录**未提供 `requirements.txt`**。依赖请参照原仓库 [MCPathology/NH2ST](https://github.com/MCPathology/NH2ST) 或 HEST 框架 [mahmoodlab/HEST](https://github.com/mahmoodlab/HEST) 的安装说明。
- 核心依赖大致包括：`torch`、`torch_geometric`、`scanpy`、`anndata`、`numpy`、`pandas`、`scipy`、`scikit-learn`、`pyyaml`。

## 数据与权重

详见 `DATA_WEIGHTS.md`。输入使用共享 benchmark 的 `processed_cell/` 接口：

```
cell-benchmark/processed_cell/<dataset>/
├── adata_S1.h5ad      # obsm['he'] 为 UNI 特征，obsm['spatial'] 为空间坐标
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

- 固定 `epochs` 训练，无验证集与早停，最终权重直接用于目标切片推理。

## 文件结构

- `model.py` — `CellNGHist2ST`
- `dataset.py` — 基于 SpatialEx 超图采样邻域的 `CellNH2STDataset`
- `train.py` — 跨切片训练
- `config.yaml` — 超参数
- `_legacy_HST/` — 原始 HST_NH2ST 源码快照（参考用，不含 `runs/`）

## 原始方法链接

- NH2ST 官方仓库：[MCPathology/NH2ST](https://github.com/MCPathology/NH2ST)

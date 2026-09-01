# cell_HST_h0mini_lp

将 `HST_h0mini_lp`（冻结 H0-mini + PCA + Ridge 线性探针）适配到单细胞 Xenium 数据上的对比方法。

## 依赖安装

- 本目录**未提供 `requirements.txt`**。依赖请参照原仓库 [mahmoodlab/HEST](https://github.com/mahmoodlab/HEST) 的安装说明。
- 核心依赖大致包括：`torch`、`torchvision`、`timm`、`scanpy`、`anndata`、`numpy`、`pandas`、`scipy`、`scikit-learn`、`Pillow`、`h5py`、`pyyaml`。
- H0-mini 权重需单独下载，见 `DATA_WEIGHTS.md`。

## 数据与权重

详见 `DATA_WEIGHTS.md`。输入为 `/home/sb202604/cell-benchmark/processed/<dataset>/` 下的：

- `cells.h5ad`：原始计数矩阵，含 `var['is_gene']`
- `patches.npy`：每个细胞对应的 224×224 H&E patch，形状 `(N, 3, 224, 224)`
- `splits.json`：`spatial_ood.S1/S2` 索引

`Human_Breast_Cancer_Rep1/Rep2` 作为 pair 数据集处理（`mode='pair'`）。

## 路径修改

代码中写死了开发服务器的绝对路径。运行前请修改：

- `config.yaml` / `config_processed_cell.yaml` 中的 `processed_root`、`processed_cell_root`、`h0mini_local_path`。

（共享脚本的 sys.path 已改为仓库内相对定位，`output_root` 默认为 `./runs`，均无需修改。）

## 运行

```bash
cd cell_level_methods/cell_HST_h0mini_lp
python predict.py --config config.yaml

# 单个数据集或单个 fold
python predict.py --config config.yaml --dataset hSkin_Melanoma
python predict.py --config config.yaml --dataset hSkin_Melanoma --fold 0
```

## 输出

```
runs/
├── summary.json
└── hSkin_Melanoma/
    └── v1/
        ├── config.yaml
        ├── summary.json
        ├── hSkin_Melanoma_fold0_seed42/
        │   ├── config.json
        │   ├── config.yaml
        │   ├── result.json
        │   ├── pred.npy
        │   └── pcc_per_gene.npy
        └── hSkin_Melanoma_fold1_seed42/
            ...
```

- fold 0：train=S1, test=S2
- fold 1：train=S2, test=S1

## 缓存

编码后的 H0-mini 特征缓存到 `./cache/<dataset>/<half>_h0mini_cls.npy`，并附带 `.manifest.json` 校验源 `patches.npy`，避免重复编码。

## 原始方法链接

- HEST 框架：[mahmoodlab/HEST](https://github.com/mahmoodlab/HEST)

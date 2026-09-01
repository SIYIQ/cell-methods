# cell_Stem

将 Stem（UNI + CONCH 条件扩散模型）适配到单细胞 Xenium 数据上的对比方法。

## 依赖安装

- 本目录**未提供 `requirements.txt`**。依赖请参照原仓库 [SichenZhu/Stem](https://github.com/SichenZhu/Stem) 的安装说明。
- 核心依赖大致包括：`torch`、`torchvision`、`timm`、`transformers`、`scanpy`、`anndata`、`numpy`、`pandas`、`scipy`、`scikit-learn`、`Pillow`、`pyyaml`。
- **必须额外 clone Stem 原仓库**：代码中从 `/home/sb202604/Stem` 导入模型与扩散模块：
  ```bash
  git clone https://github.com/SichenZhu/Stem.git /path/to/Stem
  ```
  然后把 `predict.py` 中的 `sys.path.insert(0, '/home/sb202604/Stem')` 改为你的 Stem 路径。
- CONCH 包可通过以下命令安装：
  ```bash
  pip install git+https://github.com/mahmoodlab/CONCH.git
  ```

## 数据与权重

详见 `DATA_WEIGHTS.md`。支持两种输入模式，由 `use_processed_cell` 控制：

### 模式 1：原始 patch（默认）

输入为 `cell-benchmark/processed/<dataset>/` 下的：

- `cells.h5ad`：原始计数矩阵，含 `var['is_gene']`
- `patches.npy`：每个细胞对应的 224×224 H&E patch，形状 `(N, 3, 224, 224)`
- `splits.json`：`spatial_ood.S1/S2` 索引

此模式会在线调用 UNI 和 CONCH 编码 patch，并缓存为 `./cache/<dataset>/<half>_uni_conch_cls.npy`。

### 模式 2：预计算细胞特征

输入为 `cell-benchmark/processed_cell/<dataset>/` 下的：

- `adata_S1.h5ad`：其中 `obsm['he']` 为 UNI CLS 特征
- `adata_S2.h5ad`

此模式不调用 CONCH，仅使用 UNI 特征。

`Human_Breast_Cancer_Rep1/Rep2` 作为 pair 数据集处理（`mode='pair'`）。

## 路径修改

代码中写死了开发服务器的绝对路径。运行前请修改：

- `predict.py` 中的 `sys.path.insert(0, '/home/sb202604/Stem')`，改为你的 Stem 仓库路径。
- `config.yaml`、`config_processed_cell.yaml`、`config_gpu7.yaml`、`config_processed_cell_fast.yaml` 中的 `processed_root`、`processed_cell_root`、`uni_local_path`、`conch_local_path`。

（`output_root` 默认为 `./runs`，无需修改。）

## 运行

```bash
cd /home/sb202604/cell_Stem
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
        └── hSkin_Melanoma_fold0_seed42/
            ├── config.json
            ├── config.yaml
            ├── checkpoints/
            │   └── final.pt
            ├── pred.npy
            ├── pcc_per_gene.npy
            └── result.json
```

- fold 0：train=S1, test=S2
- fold 1：train=S2, test=S1

## 训练配置

默认 `config.yaml` 是轻量快速验证配置：

- `train_steps: 500`
- `num_sampling_steps: 100`
- `sample_num_per_cond: 2`

如需更高质量结果，可参照 Stem 原仓库或 HEST-Bench 配置增加 `train_steps` 和 `sample_num_per_cond`。

## 原始方法链接

- Stem 官方仓库：[SichenZhu/Stem](https://github.com/SichenZhu/Stem)

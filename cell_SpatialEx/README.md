# cell_SpatialEx

SpatialEx 的细胞级 benchmark 适配版本。本目录保留了 SpatialEx 原始包（`SpatialEx/`）以及针对共享 `cell-benchmark` 写的包装脚本。

## 依赖安装

- 本目录已提供 `requirements.txt` 和 `setup.py`，可参照 SpatialEx 原仓库安装：
  ```bash
  pip install -r requirements.txt
  ```
- 关键依赖包括：`torch`、`torchvision`、`timm`、`transformers`、`scanpy`、`anndata`、`cellpose`、`scikit-image`、`scikit-learn`、`numpy`、`pandas`。
- 若使用 H0-mini 分支，需下载 H0-mini 权重；若使用 UNI/CONCH 分支，需下载对应权重。详见 `DATA_WEIGHTS.md`。

## 数据与权重

详见 `DATA_WEIGHTS.md`。本目录提供两条细胞级入口：

### 路径 A：从 `processed/` 生成 SpatialEx 兼容数据

```bash
python scripts/build_adata.py --dataset hSkin_Melanoma --device cuda:0
```

- 输入：`cell-benchmark/processed/<dataset>/cells.h5ad` + `patches.npy`
- 使用 H0-mini 编码 H&E patch，输出 `spatialex/S1.h5ad`、`spatialex/S2.h5ad`

### 路径 B：直接从原始 Xenium 数据运行（native）

```bash
# 单 section 数据集（hSkin_Melanoma, hColon_Non_diseased, mouse_Colon）
python scripts/run_native.py --dataset hSkin_Melanoma --device cuda:0

# 多 section / Rep1-Rep2 数据集（Human_Breast_Cancer）
python scripts/run_pair_native.py --dataset Human_Breast_Cancer --device cuda:0
```

- 输入：`cell-benchmark/raw/<dataset>/` 下的 Xenium bundle + H&E OME-TIFF
- 会在 `runs_native/<dataset>/` 下缓存 UNI/H0-mini 特征并输出结果。

## 路径修改

代码中写死了开发服务器的绝对路径。运行前请修改：

- `scripts/build_adata.py` 中的 `PROCESSED_ROOT`、`H0_MINI_DIR`。
- `scripts/run_native.py` 与 `scripts/run_pair_native.py` 中的 `RAW_ROOT`。
- `test.py` 中的 `PROCESSED_ROOT`。
- `SpatialEx/utils.py` 中的 UNI 权重路径 `local_dir`。

（`RUNS_ROOT` 与共享脚本的 sys.path 已改为仓库内相对定位，无需修改。）

## 共享 benchmark 依赖

其他方法（如 cell_HST_middle、cell_MERGE、cell_NH2ST、cell_STFlow）通过 `cell-benchmark/scripts/cell_data.py` 复用 SpatialEx 的超图工具。`cell_data.py` 已通过相对路径引用本仓库的 `cell_SpatialEx` 目录，无需额外 clone 或修改路径。

## 输出

```
runs/<dataset>/              # run_pair.py / 包装脚本 pair 模式
runs_native/<dataset>/     # run_native.py / run_pair_native.py
├── cache/
│   ├── full_he.npz
│   └── full_adata.h5ad
├── seed_42/
│   ├── pred_S1_from_S2.npy
│   ├── pred_S2_from_S1.npy
│   └── result.json
└── summary.json
```

## 文件结构

- `SpatialEx/` — 原始 SpatialEx 包
- `scripts/run_native.py` — 单 section 数据集（S1/S2 划分）
- `scripts/run_pair_native.py` — 多 section / Rep1-Rep2 数据集
- `scripts/build_adata.py` — 从 `processed/` 生成 SpatialEx 输入
- `test.py` — 结果评估与汇总
- `_legacy_HST/` — 原始 spot-level HST 快照（参考用）

## 原始方法链接

- SpatialEx 官方仓库：[KEAML-JLU/SpatialEx](https://github.com/KEAML-JLU/SpatialEx)

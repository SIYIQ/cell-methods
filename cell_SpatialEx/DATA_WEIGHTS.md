# cell_SpatialEx 所需数据与权重

本目录包含 SpatialEx 源码及细胞级 benchmark 包装脚本。**模型权重、缓存特征与训练输出未包含。**

## 需要的外部数据

SpatialEx 有两条入口路径，数据需求不同。

### 路径 A：`scripts/build_adata.py`

读取 benchmark `processed/` 目录，并用 H0-mini 编码 H&E patch：

```
/home/sb202604/cell-benchmark/processed/<dataset>/
├── cells.h5ad
└── patches.npy
```

脚本会生成 SpatialEx 兼容的 `S1.h5ad` / `S2.h5ad` 到 `spatialex/` 子目录。

### 路径 B：`scripts/run_native.py` / `scripts/run_pair_native.py`

读取 10x Xenium 原始数据及预提取的 UNI 特征：

```
/home/sb202604/cell-benchmark/raw/<dataset>/
├── *.ome.tif
└── outs/
    └── ...

/home/sb202604/cell-benchmark/processed/<dataset>/
├── cells.h5ad
└── patches.npy
```

这些脚本会在 `runs_native/` 下生成 `*_he.npz` / `*_adata.h5ad` 缓存。

## 需要的基础模型权重

- **H0-mini**：`scripts/build_adata.py` 使用。
  - HuggingFace：[bioptimus/H0-mini](https://huggingface.co/bioptimus/H0-mini)
- **UNI**：`SpatialEx/utils.py` 在包装脚本请求 UNI 编码时使用。
  - GitHub：[mahmoodlab/UNI](https://github.com/mahmoodlab/UNI)
  - HuggingFace：[MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI)

`SpatialEx/utils.py` 中原本硬编码了 UNI 加载路径；若权重在其他位置，请修改该文件。

## 共享 benchmark 辅助脚本

`test.py` 通过仓库内相对路径（`Path(__file__)` 定位）引用共享脚本目录，无需修改路径。

## 运行时生成的目录

- `runs_native/<dataset>/` — 缓存 `*_adata.h5ad`、`*_he.npz`、预测结果。
- `runs/<dataset>/` — pair 模式预测与结果文件。

以上已加入 `.gitignore`。

## 原始方法链接

- SpatialEx 官方仓库：[KEAML-JLU/SpatialEx](https://github.com/KEAML-JLU/SpatialEx)

## 需要修改路径的文件

- `scripts/build_adata.py`：`PROCESSED_ROOT`、`H0_MINI_DIR`
- `scripts/run_native.py`、`scripts/run_pair_native.py`：`RAW_ROOT`、`PROC_ROOT`
- `SpatialEx/utils.py`：UNI 权重路径

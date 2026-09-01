# cell_HST_h0mini_lp 所需数据与权重

本目录包含冻结 H0-mini + PCA + Ridge 线性探针源码。**模型权重、缓存与训练输出未包含。**

## 需要的外部数据

脚本通过 `use_processed_cell` 控制两种输入模式。

### 模式 1：原始 patch（默认）

将 `processed_root` 指向 benchmark processed 目录：

```
/home/sb202604/cell-benchmark/processed/<dataset>/
├── cells.h5ad
├── patches.npy
└── splits.json
```

### 模式 2：预计算细胞特征

将 `processed_cell_root` 指向：

```
/home/sb202604/cell-benchmark/processed_cell/<dataset>/
├── adata_S1.h5ad      # obsm['he'] 中为 H0-mini CLS 特征
└── adata_S2.h5ad
```

## 需要的基础模型权重

- **H0-mini** 权重：`h0mini_local_path`（默认 `/home/sb202604/H0-mini`）
  - 需要文件：`config.json`，以及 `pytorch_model.bin` 或 `model.safetensors` 之一。
  - HuggingFace 下载：[bioptimus/H0-mini](https://huggingface.co/bioptimus/H0-mini)

## 共享 benchmark 辅助脚本

`test.py` 通过仓库内相对路径（`Path(__file__)` 定位 `cell-benchmark/scripts`）导入 `cell_data`，无需修改路径。

## 运行时生成的目录

- `cache/<dataset>/` — 编码后的 H0-mini CLS 特征 (`*_h0mini_cls.npy`、`*_he.npy`) 及 manifest 文件。
- `runs/<dataset>/` — 预测结果与结果汇总。

以上已加入 `.gitignore`。

## 需要修改路径的配置文件

- `config.yaml`
- `config_processed_cell.yaml`

# cell_Stem 所需数据与权重

本目录包含细胞级 Stem（UNI+CONCH 条件扩散模型）源码。**模型权重、缓存与训练输出未包含。**

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
├── adata_S1.h5ad      # obsm['he'] 中为 UNI CLS 特征
└── adata_S2.h5ad
```

## 需要的基础模型权重

- **UNI** 权重：`uni_local_path`（默认 `/home/sb202604/UNI`）
  - GitHub：[mahmoodlab/UNI](https://github.com/mahmoodlab/UNI)
  - HuggingFace：[MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI)
- **CONCH** 权重：`conch_local_path`（默认 `/home/sb202604/CONCH`）
  - GitHub：[mahmoodlab/CONCH](https://github.com/mahmoodlab/CONCH)
  - HuggingFace：[MahmoodLab/CONCH](https://huggingface.co/MahmoodLab/CONCH)

## 需要的外部源码包

`predict.py` 从独立的 Stem 仓库导入扩散模型：

```python
sys.path.insert(0, "/home/sb202604/Stem")
from Stem.models import Stem_models
from Stem.diffusion import create_diffusion
```

你需要有 `/home/sb202604/Stem`（或等价的 pip 安装版 `Stem` 包），并相应修改该路径。

## 运行时生成的目录

- `cache/<dataset>/` — 拼接后的 UNI+CONCH CLS 特征 (`*_uni_conch_cls.npy`) 及 manifest 文件。
- `runs/<dataset>/` — checkpoint (`final.pt`)、预测结果、结果汇总。

以上已加入 `.gitignore`。

## 原始方法链接

- Stem 官方仓库：[SichenZhu/Stem](https://github.com/SichenZhu/Stem)

## 需要修改路径的配置文件

- `config.yaml`、`config_processed_cell.yaml`、`config_gpu7.yaml`

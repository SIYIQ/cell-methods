# Cell-Level Methods（纯代码集合）

本仓库以独立子目录的形式，收集了一组**细胞级空间转录组预测方法**的源码，仅包含代码，不包含权重、运行结果或数据。

## 包含的方法

| 子目录 | 方法 | 简介 | 原始代码 / 论文 |
|---|---|---|---|
| `cell_HST_h0mini_lp` | H0-mini 线性探针 | 冻结的 H0-mini 编码器 + PCA + Ridge 回归（HEST-bench 风格基线）。 | 基于 [mahmoodlab/HEST](https://github.com/mahmoodlab/HEST) |
| `cell_HST_middle` | 我们的方法 | 在 image + gene 节点上构建异构图 Transformer（HGT）。 | - |
| `cell_HST_uni_lp` | UNI 线性探针 | 冻结的 UNI 编码器 + PCA + Ridge 回归（HEST-bench 风格基线）。 | 基于 [mahmoodlab/HEST](https://github.com/mahmoodlab/HEST) |
| `cell_MERGE` | MERGE 细胞级适配 | 在 UNI 特征上训练 MLP + GATNet 图精化。 | [ags3927/MERGE](https://github.com/ags3927/MERGE) |
| `cell_NH2ST` | NH2ST / NGHist2ST | 邻域超图编码器，带对比学习与重建损失。 | [MCPathology/NH2ST](https://github.com/MCPathology/NH2ST) |
| `cell_SpatialEx` | SpatialEx | Nature Methods 组织学锚定的多组学整合方法。 | [KEAML-JLU/SpatialEx](https://github.com/KEAML-JLU/SpatialEx) |
| `cell_Stem` | Stem | UNI + CONCH 条件扩散模型。 | [SichenZhu/Stem](https://github.com/SichenZhu/Stem) |
| `cell_STFlow` | STFlow | 基于流匹配（flow-matching）的基因表达去噪预测模型。 | [Graph-and-Geometric-Learning/STFlow](https://github.com/Graph-and-Geometric-Learning/STFlow) |

**本集合未包含：** `GHIST`。

## 基础模型权重

以下基础模型在配置文件中被引用，但**不包含在本仓库中**。请单独下载，并更新各方法 config 中的路径。

| 模型 | 类型 | 代码 | 权重 |
|---|---|---|---|
| UNI | 病理学 ViT | [mahmoodlab/UNI](https://github.com/mahmoodlab/UNI) | [HuggingFace MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI) |
| CONCH | 病理学视觉-语言模型 | [mahmoodlab/CONCH](https://github.com/mahmoodlab/CONCH) | [HuggingFace MahmoodLab/CONCH](https://huggingface.co/MahmoodLab/CONCH) |
| H0-mini | 蒸馏病理学编码器 | — | [HuggingFace bioptimus/H0-mini](https://huggingface.co/bioptimus/H0-mini) |

## 数据与权重

每个方法需要以下一项或两项：

1. **共享 benchmark 数据**，位于 `/home/sb202604/cell-benchmark`：
   - `raw/` — 10x Xenium 原始数据包（H&E OME-TIFF、细胞/细胞核分割等）
   - `processed/` — 每个数据集的 `cells.h5ad`、`patches.npy`、`splits.json`
   - `processed_cell/` — `adata_S1.h5ad` / `adata_S2.h5ad`，其中 `obsm['he']` 为预计算 UNI 特征
   - `scripts/cell_data.py` — 多个方法共用的数据集注册与辅助脚本

2. **基础模型权重**，需单独下载（见上表）。

各子目录下的 `DATA_WEIGHTS.md` 详细说明了该方法所需的具体文件。

## S1/S2 划分口径（重要，勿混淆）

`processed_cell/` 与 `processed/` 中的数据都来自 SpatialEx 使用的相同原始数据，但 **S1/S2 存在两套同名不同义的划分**，混用会得到"看起来正常但不可比"的数字：

- **`processed_cell/adata_S1.h5ad` / `adata_S2.h5ad`（本 benchmark 的正式口径）**：**200µm 棋盘格（chessboard）划分**，由 `cell-benchmark/scripts/build_cell_dataset.py` 用与 SpatialEx 相同的数据、相同的代码产出。SpatialEx 论文只提到将切片划分为 S1/S2 两半，**并未说明具体划分方式**（对半切还是棋盘格）；实测只有棋盘格划分才能复现出论文报告的结果，因此本 benchmark 统一采用棋盘格。
- **`processed/<dataset>/splits.json` 中的 `"spatial_ood"`**：按 **x 坐标中位数左右对半切**（`cell-benchmark/scripts/preprocess.py` 的 `make_spatial_ood_split`），同样使用 S1/S2 命名，但这是另一种划分，与棋盘格的 S1/S2 不是同一个任务。

注意：`cell_SpatialEx` 的原生运行脚本（`run_native.py`）默认棋盘格，但支持 `--split-mode x_median` 切换；用非默认划分跑出的结果与 `processed_cell/` 口径不一致。汇总结果时（如 `summarize_cell_benchmark.py`）请确认每一行数字来自哪种划分，不要把两种划分的结果放在同一列直接比较。

## `processed_cell/` 数据格式

`processed_cell/` 里就是**普通的 10x Xenium 细胞级数据**——每个切片一个标准 `AnnData`（h5ad），`obs` 列就是 Xenium 官方输出的逐细胞元数据，并未做任何特殊封装。相比原始 Xenium 输出只多了两件事：① 按 S1/S2（或 Rep1/Rep2）拆成两个文件；② 把预计算的 UNI 特征缓存在 `obsm['he']` 里。

目录结构：

```
processed_cell/
├── hSkin_Melanoma/
│   ├── adata_S1.h5ad                     # S1 半区（棋盘格划分）
│   ├── adata_S2.h5ad                     # S2 半区
│   ├── clusters_hSkin_Melanoma.csv       # 细胞聚类标签（Barcode,Cluster 两列）
│   └── meta.json                         # 划分方式、细胞数、基因数等元信息
├── hColon_Non_diseased/                  # 同上（S1/S2）
├── mouse_Colon/                          # 同上（S1/S2）
└── Human_Breast_Cancer/                  # 两个生物学重复，用 Rep1/Rep2 代替 S1/S2
    ├── adata_Rep1.h5ad
    ├── adata_Rep2.h5ad
    ├── clusters_Human_Breast_Cancer__Rep1.csv
    ├── clusters_Human_Breast_Cancer__Rep2.csv
    └── meta.json
```

h5ad 内部结构（以 `hSkin_Melanoma/adata_S1.h5ad` 为例，`43957 cells × 382 genes`）：

| 位置 | 内容 |
|---|---|
| `X` | `float32 (n_cells, n_genes)`，log1p 归一化表达（`normalize_total` + `log1p`） |
| `layers['raw']` | 原始 UMI 计数 |
| `obs` | Xenium 原生逐细胞元数据：`x_centroid` / `y_centroid`（µm 坐标）、`transcript_counts`、`total_counts`、`cell_area`、`nucleus_area`、`control_probe_counts` 等 |
| `var` | `gene_ids`、`feature_types`、`genome`；`var_names` 为基因符号 |
| `obsm['spatial']` | `(n_cells, 2)` 空间坐标（µm） |
| `obsm['he']` | `(n_cells, 1024)` 预计算 UNI 特征（各方法可直接读取，无需 UNI 权重） |
| `uns` | `dataset`、`parent_dataset`、`split`（`S1`/`S2`/`Rep1`/`Rep2`）、`log1p` |

`meta.json` 字段：`dataset`、`mode`（`chessboard` 棋盘格 / `pair` 重复对）、`block_um`（棋盘格边长，200µm）、`n_cells`、`n_genes`、`he_dim`、`outputs`（split → 文件名映射）。

## 运行前准备

1. **依赖安装**：
   - 由于不同模型依赖版本相差很大，各子目录**未统一提供 `requirements.txt`**。
   - 请按各子目录 `README.md` 中注明的原仓库安装依赖，或参照 HEST 框架 [mahmoodlab/HEST](https://github.com/mahmoodlab/HEST) 的环境。
   - 通用依赖大致包括：`torch`、`torchvision`、`torch_geometric`、`scanpy`、`anndata`、`numpy`、`pandas`、`scipy`、`scikit-learn`、`Pillow`、`h5py`、`pyyaml`、`transformers`、`timm`。

2. **需要额外 clone 原仓库的方法**：
   - `cell_Stem`：需要 [SichenZhu/Stem](https://github.com/SichenZhu/Stem)，并修改 `predict.py` 中的 `sys.path.insert` 路径。
   - `cell_STFlow`：需要 [Graph-and-Geometric-Learning/STFlow](https://github.com/Graph-and-Geometric-Learning/STFlow)，可通过环境变量 `STFLOW_ROOT` 指定路径。
   - 所有使用 `cell-benchmark/scripts/cell_data.py` 的方法：`cell_data.py` 已通过相对路径引用本仓库内的 `cell_SpatialEx` 包，无需额外 clone。

3. **路径修改**：
   - 代码中有一些硬编码了本机 `/home/sb202604/...` 路径（数据根目录、UNI/CONCH/H0-mini 权重目录、外部 Stem/STFlow 仓库路径等）。各方法共享脚本的 sys.path 与各 `output_root` 已改为仓库内相对定位，无需修改。
   - 运行前请根据各子目录 `README.md` 的说明修改对应文件。

4. **数据与权重**：
   - 本仓库不包含任何患者数据、模型权重或训练输出。
   - 需要自行准备 `cell-benchmark/processed_cell/` 等数据，并下载 UNI / CONCH / H0-mini 权重。

## 使用方法

1. 安装依赖（各子目录应提供 `requirements.txt` 或 `setup.py`；部分方法可能还需补充）。
2. 更新 `config.yaml`（或等价文件）中的绝对路径，使其指向你本地的数据和权重目录。原始代码沿用了开发服务器上的 `/home/sb202604/...` 路径。
3. 按照各子目录 README 中的说明运行训练/评估脚本。

## 注意事项

- 若干子目录包含 `_legacy_HST/` 文件夹，保存了原始 spot-level 低分辨率 HST-benchmark 源码作为参考快照，但是在高分辨率数据上训练并不实际使用他们。

## 已知的上游实现问题（与原版保持一致，未修复）

以下问题均来自上游原仓库（已逐项核实原版代码相同），为保持与已发表方法的可比性而**原样保留**。

### cell_NH2ST：预测路径不经过超图融合

- `model.py` 的 `forward` 中，用于评估的预测 `pred_exp` 在 `cross_encoder` 交叉融合**之前**就计算完毕，输入仅为目标细胞自身的 UNI 特征；邻居超图 / HGNN / 对比学习分支不参与推理输出（仅在训练时通过对比损失间接塑造表征）。
- `decoded_exp` 与 `pred_exp` 是同一表达式计算两次（逐元素相等），因此"重建损失"与预测损失实为同一个量。
- 这与上游 [MCPathology/NH2ST](https://github.com/MCPathology/NH2ST) 官方代码行为完全一致（其 `models/NGHist2ST.py` 的训练损失与 validation/test 均使用该路径），即原方法的官方数字同样产自这条路径。因此本仓库的 NH2ST 结果实质上是一个"target-only MLP + 对比学习辅助训练"的口径，请勿将其解读为完整的邻域超图模型。

### cell_SpatialEx：上游包内存在"死而坏"的代码路径

- 本仓库的 `cell_SpatialEx/SpatialEx/` 包与上游 [KEAML-JLU/SpatialEx](https://github.com/KEAML-JLU/SpatialEx) 原仓库**逐字节一致**。benchmark 仅使用 `SpatialEx` 基类（内部为 `model.Model`），该路径是健康的。
- 但包内以下上游路径存在问题，**请勿使用**：
  - `HyperSAGE.weight_list` 用普通 Python list 装 `nn.Parameter`，未注册到模块——参数不会被优化器更新、不进 `state_dict`（影响 `Model_Big` / `SpatialExP_Big`，其图网络主干全程不学习）；
  - `HyperSAGE.predict()` 引用不存在的 `self.weight1` / `self.weight2`，调用即 `AttributeError`；
  - `HGNN` 设置 `num_layers > 2` 时不创建 `W1` / `W2` 但 `forward` 会使用，必崩；
  - `preprocess.normalize_graph` 的 `row` / `col` / `both` 分支引用未定义变量 `adj`，必 `NameError`；
  - `SpatialExP.train()` 调用 `Model_Plus.forward` 时缺少 `agg_mtx`，非 Visium 平台（本 benchmark 全部为 Xenium）必 `TypeError`。


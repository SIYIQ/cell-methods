# cell_HST_middle：细胞级 HeteroST（HST-middle 适配）

本目录是 HST-middle 的细胞级适配版本。原版的 HeteroST/HGT 模型以 spot-level 数据上运行；这里的实现改为读取细胞级 benchmark 接口 `cell-benchmark/processed_cell/`，每个细胞以预计算的 UNI 特征作为 image 节点输入。

## 模型简介

- **图结构**：每个细胞同时作为 `image` 节点（UNI 特征）和 `gene` 节点（基因表达）。
- **边类型**：
  - `spatial` — 基于细胞空间邻域超图构建的边。
  - `morphologically_similar` — 基于 UNI 特征余弦相似度的 top-K 形态学边。
- **骨干网络**：`CellHST`（HeteroST 简化版），使用 `HGTConv` 在异构图上传递信息。
- **可选 MGM**：支持 Masked Gene Modeling，训练时随机 mask 部分细胞的基因输入，强制模型通过 image 分支恢复表达。默认关闭，便于与只用 image 分支的方法公平对比。

## 文件结构

```
cell_HST_middle/
├── config.yaml          # 训练配置
├── model.py             # CellHST 模型定义
├── train.py             # 主训练脚本（S1↔S2 跨切片训练/评估）
├── test.py              # 从 runs/ 加载预测并汇总指标
├── utils.py             # 异构图构建、基因名读取、MGM mask 等辅助函数
├── DATA_WEIGHTS.md      # 数据与权重说明
├── README.md            # 本文件
└── _legacy_HST/         # 原始 spot-level HST-benchmark 源码快照（参考用）
```

## 依赖安装

- 本目录已提供 `requirements.txt`，可直接安装：
  ```bash
  pip install -r requirements.txt
  ```
  该文件记录了本地 baseline 环境的版本，CUDA 构建可能与你的机器不同，必要时请从 PyTorch 官网选择对应 CUDA/CPU 版本。
- 也可参照 HEST 框架 [mahmoodlab/HEST](https://github.com/mahmoodlab/HEST) 的安装说明自行配置。
- 运行时不需要额外基础模型权重——代码直接使用 `obsm['he']` 中的预计算 UNI 特征。

## 路径修改

代码中写死了开发服务器的绝对路径。运行前请修改：

- `config.yaml` 中的 `processed_root`。

（共享脚本的 sys.path 已改为仓库内相对定位，`output_root` 默认为 `./runs`，均无需修改。）

## 输入数据

需要共享 benchmark 的 `processed_cell` 目录：

```
/home/sb202604/cell-benchmark/processed_cell/<dataset>/
├── adata_S1.h5ad      # obsm['he'] 为 UNI 特征，obsm['spatial'] 为空间坐标
└── adata_S2.h5ad
```

涉及的数据集（在 `config.yaml` 中配置）：
- `hSkin_Melanoma`
- `hColon_Non_diseased`
- `mouse_Colon`
- `Human_Breast_Cancer`

## 使用方法

### 训练所有数据集

```bash
python train.py --config config.yaml
```

### 只训练单个数据集

```bash
python train.py --config config.yaml --dataset hSkin_Melanoma
```

### 只跑指定 seed

```bash
python train.py --config config.yaml --dataset hSkin_Melanoma --seeds 42
```

## 输出结构

每个数据集的结果保存在 `config.yaml` 中 `output_root` 指定的目录下：

```
runs/<dataset>/
├── seed_42/
│   ├── pred_S1_from_S2.npy
│   ├── pred_S2_from_S1.npy
│   ├── per_gene_pcc_S1.npy
│   ├── per_gene_pcc_S2.npy
│   └── result.json
├── seed_43/
│   └── ...
├── seed_44/
│   └── ...
└── summary.json
```

`summary.json` 汇总了多个 seed 以及两个 cross-section 方向（S1→S2、S2→S1）的 mean/median PCC。

## 训练细节

- **无在线编码器**：模型直接使用 `obsm['he']` 中的预计算 UNI 特征，不加载或 finetune UNI。
- **Loss**：默认 `mse`，可选 `pearson`（1 - mean per-gene PCC）。
- **Epochs**：默认 100，固定轮数，无验证集早停。
- **Cross-section**：对每一对 `(S1, S2)`，分别用一半训练、另一半测试，再反过来。

## 原始方法链接

- HEST 框架：[mahmoodlab/HEST](https://github.com/mahmoodlab/HEST)

# Low-Light Image Enhancement via Knowledge Distillation — 复现与修订材料

本目录是论文《基于知识蒸馏的轻量低光照图像增强网络》配套的**代码、数据、论文 Markdown 与修订文档**，已按可复现、可评审的标准重新整理。

---

## 一、目录结构

```
lowlight_paper_revision/
├── README.md                      # 本文件
├── paper.md                       # 论文全文 Markdown（已嵌入 Fig.1–3）
├── paper_revision.md              # 修改意见 + 原文→改后对照
├── sci_revision_suggestions.md    # 对标 SCI 期刊的修改意见（P0/P1/P2）
├── figs/                          # 论文插图
│   ├── fig1.png                   # Fig.1 DDNet 教师模型架构
│   ├── fig2.png                   # Fig.2 轻量学生模型架构
│   └── fig3.png                   # Fig.3 各方法增强效果视觉对比
├── data/                          # LOW-LIGHT (LOL) 数据集
│   ├── train/                     # our485 训练集：low/ (485) 与 high/ (485)
│   └── test/                      # eval15 测试集：low/ (15) 与 high/ (15)
├── code/
│   ├── our_model/                 # 论文自研学生模型（完整代码 + 全部权重）
│   │   ├── config.py / models.py / losses.py / dataset.py
│   │   ├── trainer.py / train.py / test.py / evaluator.py
│   │   ├── ablation_study.py / warmup_analysis.py / utils.py
│   │   ├── models/                # 最终权重（ablation_full.pth 等）
│   │   ├── saved_models/          # 训练过程中间检查点
│   │   └── results/               # 各版本指标 JSON
│   ├── baselines/                 # 对比方法源码（第三方）
│   │   ├── DDNet/                 # 教师模型（评估脚本，权重需另行获取）
│   │   ├── SCI/                   # SCI-main（TPAMI/CVPR）
│   │   ├── RUAS/                  # RUAS-main
│   │   ├── CLAHE/                 # CLAHE.py（直方图均衡）
│   │   ├── Dong/                  # Dong 等人的低光照视频增强算法
│   │   └── our_method_v1/         # 学生模型的早期版本代码
│   └── evaluation/
│       ├── unified_evaluate.py    # 统一评估协议（重算 PSNR/SSIM/NIQE/PIQE/亮度）
│       ├── generate_tables.py     # 由 JSON 生成修正后的 Table V/VI 与 FPS 汇总
│       └── student_eval/          # 原始评估脚本（img_accessment_py3.9.py 等）
├── enhanced_outputs/              # 各对比方法在 eval15 上的预计算增强结果
│   └── *_with_brightness_results.json
└── results/
    └── unified_results/
        ├── unified_metrics.json   # 统一重算指标（Full + 4 个消融）
        └── enhanced/              # 统一协议下各模型的增强输出图
```

---

## 二、环境依赖

- Python 3.9（实验在 `py3_9` conda 环境）
- PyTorch 1.9.0，CUDA 11.1
- 关键包：`torch`, `torchvision`, `numpy`, `pillow`, `scikit-image`, `pyiqa`, `pypiqe`, `opencv-python`

```bash
conda activate py3_9
```

说明：统一评估依赖 `pyiqa`（PSNR/SSIM/NIQE）与 `pypiqe`（PIQE），如缺失可用 `pip install pyiqa pypiqe` 补装。

---

## 三、数据准备

LOL 数据集已整理为 `data/train`（our485，485 对）与 `data/test`（eval15，15 对），路径符合 `code/our_model/dataset.py` 的 `Dataset_train/Dataset_test` 约定。

- 训练集：`data/train/low` → `data/train/high`
- 测试集：`data/test/low` → `data/test/high`

---

## 四、训练学生模型

```bash
cd code/our_model
python train.py          # 训练优化后的学生模型（默认读取 config.py 的超参）
python ablation_study.py # 运行消融实验（无 skip/残差/注意力/减通道等变体）
```

主要超参见 `code/our_model/config.py`（学习率、batch、epoch、损失权重等）。

---

## 五、统一评估（推荐，全文一致口径）

```bash
cd code/evaluation
python unified_evaluate.py          # 对 Full + 消融变体重算统一指标，输出 results/unified_results/
python generate_tables.py --json ../results/unified_results/unified_metrics.json
```

统一协议：**原始分辨率 + pyiqa 计算 PSNR/SSIM/NIQE + pypiqe 计算 PIQE + 亮度 0–255 + 离群值剔除 + 参数量/推理延迟统计**。该协议消除了旧版 `img_accessment_py3.9.py` 与 `evaluator.py` 两套口径造成的数字矛盾。

关键结果（Full model）：

| 指标 | 值 |
| --- | --- |
| PSNR | 18.72 ± 3.69 dB |
| SSIM | 0.817 |
| 亮度（0–255） | 118.4 |
| 参数量 | 0.144 M |
| 推理延迟 | 2.88 ms ≈ 347 FPS |

---

## 六、对比方法

- 各对比方法源码位于 `code/baselines/`。
- 各方法在 eval15 上的**预计算增强输出**与亮度统计在 `enhanced_outputs/` 下（用于复现 Table II 的对比数字）。
- **注意**：对比方法（DDNet/SCI 等）的已训练权重体积较大、未随包内置；如需完整重训/复现，请按各方法原始仓库获取权重，或在本数据集上按其训练脚本重训。`Zero-DCE` 在服务器上无源码，仅有 `enhanced_outputs/Zero-DCE/` 的预计算输出。

---

## 七、修订说明速览

- 论文 Markdown 见 [paper.md](paper.md)（已嵌入图片）。
- 逐条修改意见与"原文→改后"对照见 [paper_revision.md](paper_revision.md)。
- 对标 SCI 期刊的送审/录用修改意见（含 P0/P1/P2 优先级清单）见 [sci_revision_suggestions.md](sci_revision_suggestions.md)。

---

## 八、重要说明

1. `code/our_model/saved_models/`（约 183 MB）为训练过程中间检查点，如只需最终权重可仅保留 `code/our_model/models/` 下的 `ablation_full.pth`、`optimized_model_c32.pth` 等。
2. 论文原 PDF 仅作者上传；本目录未含 LaTeX 源，故"同步修改"以 Markdown + 数字重算落地。如需直接改 LaTeX 表格，可用 `generate_tables.py` 输出 LaTeX 片断。
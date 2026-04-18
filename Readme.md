# Brain Encoder — NSDGeneral func1pt8mm

## 概述

本仓库提供基于视觉刺激预测大脑 fMRI 响应的**编码器（Encoder）**推理代码与预训练权重。

- **训练代码来源**：[Algonauts 2023 Challenge — huze 方案](https://github.com/huzeyann/MemoryEncodingModel)（MemoryEncodingModel）。
- **数据格式迁移**：在原始 `nsdgeneral` 空间的基础上，迁移训练至 **`func1pt8mm`** 数据格式（1.8 mm 各向同性功能像空间），以兼容更高分辨率的 fMRI 数据。
- **预训练权重**：模型权重托管于 [Hugging Face Hub — wzcfantasy/huze_func1pt8mm](https://huggingface.co/wzcfantasy/huze_func1pt8mm)，可直接下载使用。

---

## 目录结构

```
Encoder/                          ← GitHub 仓库（仅代码）
├── infer.py                      # 单张图像推理入口脚本
├── run_encoder_infer.sh          # 封装好的推理启动脚本
├── requirements.txt              # Python 依赖
├── architecture/                 # 模型架构定义
│   ├── encoder_model.py          # 推理封装（BrainEncoder）
│   ├── models.py                 # 核心模型（MemVoxelWiseEncodingModel 等）
│   ├── backbone.py / blocks.py / ...
│   └── third_party/
│       └── facebookresearch_dinov2_main/   # DINOv2 视觉骨干
└── model_zoo/                    ← 不在 GitHub 中，需从 HuggingFace 下载
    └── subjects/
        ├── subj01/
        │   ├── config.yaml       # 模型配置
        │   ├── coords.npy        # 体素空间坐标
        │   └── model.pth         # 权重（约 1.1 GB）
        ├── subj02/
        ├── subj05/
        └── subj07/
```

> **注**：`model_zoo/` 目录已加入 `.gitignore`，GitHub 仓库中不包含权重文件。权重统一托管于 HuggingFace，见下方下载说明。

---

## 环境安装

```bash
pip install -r requirements.txt
```

主要依赖：

| 包 | 版本要求 |
|---|---|
| torch | >=2.1 |
| torchvision | >=0.16 |
| numpy | >=1.24, <2.0 |
| timm | >=1.0 |
| open-clip-torch | >=2.24 |
| einops | >=0.7 |
| yacs | >=0.1.8 |

---

## 快速推理

### 方式一：使用 Shell 脚本（推荐）

```bash
bash run_encoder_infer.sh \
    --image /path/to/image.jpg \
    --subject subj01 \
    --output /path/to/output.npy
```

### 方式二：直接调用 Python 脚本

```bash
python infer.py \
    --image /path/to/image.jpg \
    --subject subj01 \
    --output /path/to/output.npy \
    --device cuda
```

### 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `--image` | 是 | — | 输入图像路径 |
| `--subject` | 否 | `subj01` | 被试 ID（`subj01/02/05/07`） |
| `--subject_root` | 否 | `model_zoo/subjects` | 被试权重根目录 |
| `--config` | 否 | 自动推断 | 覆盖配置文件路径 |
| `--checkpoint` | 否 | 自动推断 | 覆盖模型权重路径 |
| `--coords` | 否 | 自动推断 | 覆盖体素坐标文件路径 |
| `--output` | 是 | — | 输出 `.npy` 文件路径（预测的 fMRI 信号） |
| `--device` | 否 | `cuda`（若可用） | 运行设备 |
| `--preserve_aspect_ratio` | 否 | False | 使用 resize+center-crop 而非直接缩放 |

输出为 `.npy` 文件，包含预测的 `nsdgeneral_func1pt8mm` 空间下各体素的响应幅值。

---

## 预训练权重下载

权重托管于 [wzcfantasy/huze_func1pt8mm](https://huggingface.co/wzcfantasy/huze_func1pt8mm)，包含 subj01、subj02、subj05、subj07 四名被试的权重（每个约 1.1 GB）。

```bash
# 方式一：使用 huggingface_hub Python API
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='wzcfantasy/huze_func1pt8mm',
    local_dir='.',
    allow_patterns='model_zoo/**',
)
"
```

```bash
# 方式二：使用 huggingface-cli（需安装 huggingface_hub[cli]）
huggingface-cli download wzcfantasy/huze_func1pt8mm \
    --include 'model_zoo/**' \
    --local-dir .
```

下载完成后，`model_zoo/` 目录应位于本仓库根目录下，结构与上方目录树一致。

---



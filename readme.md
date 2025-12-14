# AI-Based Digital Assessment of Gross Motor Function in Metachromatic Leukodystrophy

## Overview

This repository provides code for AI-based digital assessment of gross motor function in **Metachromatic Leukodystrophy (MLD)**. It supports two training paradigms:

- **Clip-supervised** methods 
- **Multiple Instance Learning (MIL)** based methods 

---

## Table of Contents

- [Installation](#installation)
- [Data layout and description](#data-layout-and-description)
- [Pre-trained models and results](#pre-trained-models-and-results)
- [Run inference (evaluation)](#run-inference-evaluation)
- [Train from scratch](#train-from-scratch)
- [Project structure](#project-structure)
- [Advanced usage](#advanced-usage)
- [System requirements](#system-requirements)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)
- [Changelog](#changelog)

---

## Installation

Two installation options are provided: **Conda (recommended)** and **pip**.

### Option 1 — Conda (recommended)

```bash
# Clone repository
git clone https://github.com/Toytiny/Takeda-MIT-MLD.git
cd Takeda-MIT-MLD

# Create conda environment from provided file
conda env create -f environment.yml

# Activate environment
conda activate takeda
```

### Option 2 — pip

```bash
# Clone repository
git clone https://github.com/Toytiny/Takeda-MIT-MLD.git
cd Takeda-MIT-MLD

# Install pip requirements
pip install -r requirements.txt
```

Notes:
- `environment.yml` is recommended for reproducibility and to avoid CUDA/PyTorch mismatch issues.
- If you install via pip, ensure `torch/torchvision/torchaudio` match your CUDA driver/runtime.

---

## Data layout and description

We place **skeleton sequences** and **split lists** under the `data/` folder with the following structure:

```text
data/
├── new_takeda_processed_coco_merged_ls_flt/          # clip-supervised (per-visit clips)
│   ├── <visit_id>/                                  # e.g., 001_1001=02-DEC-2019
│   │   ├── clip_000.npy
│   │   ├── clip_001.npy
│   │   └── ...
│   └── ...
├── new_takeda_processed_coco_merged_video/           # MIL (per-visit full sequence)
│   ├── <visit_id>.npy                               # e.g., 001_1001=02-DEC-2019.npy
│   ├── <visit_id>.npy
│   └── ...
└── split_info_new/
    └── original_setup/
        ├── train_majority.txt
        ├── val_majority.txt
        ├── test_majority.txt
        └── test_challenge.txt
```

### Directory descriptions

- `new_takeda_processed_coco_merged_ls_flt/`  
  Clip-supervised inputs. Each **visit** is a folder named by `<visit_id>` (e.g., `001_1001=02-DEC-2019/`) containing multiple **short clips** saved as `clip_###.npy`.

- `new_takeda_processed_coco_merged_video/`  
  MIL inputs. Each **visit** is stored as a single numpy file `<visit_id>.npy` representing the **full sequence** for that visit.

- `split_info_new/original_setup/`  
  Split lists used by training/evaluation scripts. We provide primary splits (`*_majority.txt`) and a challenge split (`test_challenge.txt`).

> Naming convention: `visit_id` follows `site_subject=DATE` (e.g., `001_1001=02-DEC-2019`).

---

## Pre-trained models and results

Experiments (checkpoints/logs/metrics) are saved under an experiment directory, typically one of:

- `results/<EXP_NAME>/` (common for repo-root runs), or
- `train/results/<EXP_NAME>/` (if your workflow uses `train/` as a workspace)

Each experiment folder typically contains:
- model weights (e.g., `best_model.pth`, `final_model.pth`)
- checkpoints
- logs
- saved args/config
- evaluation outputs (predictions/metrics)

Example experiment name:
- `exp_stgcn_cp_noo_1_full`

> If your local setup stores results elsewhere (e.g., an absolute path on a cluster), keep it consistent across training and evaluation.

---

## Run inference (evaluation)

### MIL evaluation

Evaluate a trained MIL experiment on a split:

```bash
python eval_mil.py --exp-name <EXP_NAME> --splits test
```

Example:

```bash
python eval_mil.py --exp-name exp_stgcn_cp_noo_1_full --splits test
```

### Clip-supervised evaluation

Evaluate a trained clip-supervised experiment (if applicable in your setup):

```bash
python eval_clip.py --exp-name <EXP_NAME> --splits test
```

> Use `-h` to see all options:
```bash
python eval_mil.py -h
python eval_clip.py -h
```

---

## Train from scratch

### Train MIL

```bash
python train_mil.py \
  --setup-list original_setup \
  --use-cp True \
  --epochs 200 \
  --pool-mode noisy_or_ord \
  --tau 1 \
  --exp-name <EXP_NAME>
```

Example:

```bash
python train_mil.py \
  --setup-list original_setup \
  --use-cp True \
  --epochs 200 \
  --pool-mode noisy_or_ord \
  --tau 1 \
  --exp-name exp_stgcn_cp_noo_1_full_new
```

### Train clip-supervised (optional)

```bash
python train_clip.py --setup-list original_setup --epochs 200 --exp-name <EXP_NAME>
```

> Use `-h` for the authoritative list of supported flags:
```bash
python train_mil.py -h
python train_clip.py -h
```

---

## Project structure

```text
Takeda-MIT-MLD/
├── data/
│   ├── new_takeda_processed_coco_merged_ls_flt/
│   ├── new_takeda_processed_coco_merged_video/
│   └── split_info_new/
├── models/                       # model architectures (e.g., ST-GCN, heads)
├── utils/                        # shared utilities
├── train/                        # optional workspace (may contain scripts/artifacts depending on setup)
├── results/                      # generated after training/evaluation (if configured)
├── weights/                      # optional pretrained weights
├── augmentation.py               # augmentation utilities
├── dataset.py                    # frame/clip dataset loader (if used)
├── dataset_mil.py                # MIL dataset loader
├── mil_utils.py                  # pooling/metrics helpers
├── train_mil.py                  # MIL training entry
├── eval_mil.py                   # MIL evaluation entry
├── train_clip.py                 # clip-supervised training entry
├── eval_clip.py                  # clip-supervised evaluation entry
├── environment.yml
├── requirements.txt
└── README.md
```

---

## Advanced usage

### Resume training from a checkpoint

If your training script supports resuming, a common pattern is:

```bash
python train_mil.py \
  --setup-list original_setup \
  --use-cp True \
  --epochs 400 \
  --pool-mode noisy_or_ord \
  --tau 1 \
  --exp-name <EXP_NAME> \
  --resume-from-checkpoint <PATH_TO_CHECKPOINT>
```

Example:

```bash
python train_mil.py \
  --setup-list original_setup \
  --use-cp True \
  --epochs 400 \
  --pool-mode noisy_or_ord \
  --tau 1 \
  --exp-name exp_stgcn_cp_noo_1_full_new \
  --resume-from-checkpoint results/exp_stgcn_cp_noo_1_full_new/best_model.pth
```

> Please verify the exact flag names with `python train_mil.py -h`.

### Batch evaluate multiple experiments

```bash
#!/bin/bash
set -e

for exp in exp_stgcn_cp_noo_1_full exp_other_example; do
  python eval_mil.py --exp-name "$exp" --splits test
done
```

---

## System requirements

Recommended:
- Linux
- NVIDIA GPU (CUDA-capable)
- Python version consistent with `environment.yml`
- 16+ GB RAM
- Sufficient disk space for data + experiment outputs

CPU-only may work for limited evaluation, but training will be slow.

---

## Troubleshooting

### Data not found

Verify the folder layout and split files:

```bash
ls -la data/new_takeda_processed_coco_merged_video/train/
ls -la data/split_info_new/original_setup/
```

### CUDA / PyTorch mismatch

Prefer Conda installation and verify CUDA works:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.__version__)"
```

### Out-of-memory (OOM)

Try reducing:
- batch size
- clip length / number of frames
- MIL bag size / number of instances per bag

---

## Contributing

We welcome contributions.

1. Fork the repository and create a feature branch:
   ```bash
   git checkout -b feature/description
   ```
2. Commit changes with clear messages.
3. Push and open a Pull Request with a concise description (and tests if applicable).

Please open GitHub issues for bug reports and feature requests.

---

## Citation

If you use this project in research, please cite:

```bibtex
@article{takeda_mit_mld_2024,
  title={AI-Based Digital Assessment of Gross Motor Function in Metachromatic Leukodystrophy},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

---

## License

- Code and documentation: **CC BY-NC 4.0** (unless otherwise noted)
- Data: not distributed; subject to clinical data governance and any Takeda/institutional agreements

> Consider adding a `LICENSE` file to the repository root to make the license explicit.

---

## Contact

- Project lead: [Your Name]
- Email: [Your Email]
- Affiliation: MIT & Takeda

For questions or issues, please open a GitHub issue.

---

## Acknowledgments

Thanks to MIT and Takeda for support, and to all contributors.

---

## Changelog

### v1.0 (2025-12-13)
- Initial public release
- Support for MIL training and evaluation
- Support for clip-supervised training/evaluation

---

*Last updated: 2025-12-13*

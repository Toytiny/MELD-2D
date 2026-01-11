# AI-Based Digital Assessment of Gross Motor Function in Metachromatic Leukodystrophy

## Overview

This repository provides code for AI-based digital assessment of gross motor function in **Metachromatic Leukodystrophy (MLD)**. It supports two training paradigms:

- **Clip-supervised** methods (confidence-based soft voting and max voting)
- **Multiple Instance Learning (MIL)** based methods (ordinal Noisy-OR pooling and log-exp-sum pooling) 

We also provide the first video-based MLD dataset with expert-annotated GMFC-MLD scores and extracted 2D skeleton sequences.

---

## Installation

We recommend **Conda** for reproducibility and to avoid PyTorch/CUDA version mismatches.

### Option — Conda (recommended)

```bash
# Clone repository
git clone https://github.com/Toytiny/MELD-2D.git
cd MELD-2D

# Create environment (recommended Python version)
conda create -n takeda python=3.10 -y
conda activate takeda

# Install PyTorch + CUDA 11.7 (GPU)
conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install remaining dependencies (torch is handled by conda above)
pip install -r requirements.txt

```

###  Qucik sanity check

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())"
```

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

## Using our trained weights (inference)

All experiment artifacts (checkpoints/args) are stored under:

```text
train/results/<EXP_NAME>/
```

Each experiment folder typically contains:
- `args.json`: the exact configuration/arguments used for the run
- checkpoints / model weights (e.g., `bestEMA_acc_original_setup_rep*.pth`)
- training logs (if enabled)
- evaluation outputs (predictions/metrics, if generated)

Our experiment names:
- `clip_stgcn_cp_max`
- `clip_stgcn_cp_soft`
- `exp_stgcn_cp_noo_1_full`
- `exp_stgcn_cp_lse_10_full`

> Tip: keep the `train/results/<EXP_NAME>/` directory intact. Evaluation scripts load both the checkpoint(s) and `args.json` from this folder.

### Run inference (evaluation)

**MIL (video-based)**
```bash
python eval_mil.py --exp-name <EXP_NAME> --splits test
```

Example:
```bash
python eval_mil.py --exp-name exp_stgcn_cp_noo_1_full --splits test
```

**Clip-supervised (clip-based)**
```bash
python eval_clip.py --exp-name <EXP_NAME> --splits test
```

Example:
```bash
python eval_clip.py --exp-name clip_stgcn_cp_max --splits test
```

See all options:
```bash
python eval_mil.py -h
python eval_clip.py -h
```

---

## Train your own models

### Train MIL (video-based)

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

### Train clip-supervised (clip-based)

```bash
python train_clip.py --setup-list original_setup --epochs 200 --exp-name <EXP_NAME>
```

Example:
```bash
python train_clip.py --setup-list original_setup --epochs 200 --exp-name clip_stgcn_cp_max_new
```

> Use `-h` for the full list of flags:
```bash
python train_mil.py -h
python train_clip.py -h
```


---

## Citation

If you use this project in research, please cite:

```bibtex
@article{meld_2d_2026,
  title={AI-Based Digital Assessment of Gross Motor Function in Metachromatic Leukodystrophy},
  author={[Authors]},
  journal={[Journal]},
  year={2025}
}
```

---

## License

- **Code and documentation:** **CC BY-NC 4.0** (unless otherwise noted)
- **Skeleton data (in this repository):** released for **research and non-commercial use only**, subject to any underlying clinical data governance and institutional/Takeda agreements.

> Please make sure your use and any redistribution comply with the applicable data use agreement. Consider adding a `LICENSE` (and/or a dedicated `DATA_LICENSE`) file at the repository root to make these terms explicit.

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

*Last updated: 2025-12-13*

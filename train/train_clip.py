#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, io, atexit, math, platform, shlex, glob, unicodedata
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
import json
import argparse
import random

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

# Backbone & utils from your repo
from models.stgcn import STGCN
from models.heads import Classifier
from utils.data_processing import Preprocess_Module

# Reuse helpers to align with the MIL script
from mil_utils import (
    str2bool, seed_worker, log_args, print_env_brief, print_repro_cmd,
    unwrap_state_dict, load_unwrapped_state, read_lines,
    compute_metrics, compute_macro_from_conf
)

# ----------------- CLI -----------------
parser = argparse.ArgumentParser()

# --- experiment/meta ---
parser.add_argument("--exp-name", type=str, default=None,
                    help="Experiment name; default = exp_YYYYmmdd_HHMMSS")
parser.add_argument("--seed", type=int, default=3407, help="Random seed for torch/numpy workers")

# --- device ---
parser.add_argument("--gpus", type=str, default="0,1", help='CUDA_VISIBLE_DEVICES, e.g. "0" or "0,1"')
parser.add_argument("--primary", type=int, default=0, help="Primary GPU index after CUDA remap")

# --- backbone / weights (interface aligned with MIL) ---
parser.add_argument("--backbone", type=str, default="stgcn", choices=["stgcn"])
parser.add_argument("--pretrain", type=str2bool, default=True)
parser.add_argument("--freeze-backbone", type=str2bool, default=False,
                    help="Freeze backbone; train classifier head only")
parser.add_argument("--use-cp", type=str2bool, default=False,
                    help="Use NTU checkpoint weights cp.pth when backbone=stgcn")
parser.add_argument("--backbone-state-path", type=str, default=None,
                    help="Override checkpoint path; if None, auto-select by backbone/use-cp")

# --- training hyperparams (names aligned with MIL) ---
parser.add_argument("--batch-size-videos", type=int, default=256,
                    help="Here it means *clip* batch size for training")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--weight-decay", type=float, default=5e-5)
parser.add_argument("--exp-times", type=int, default=5, help="Repetitions per case")
parser.add_argument("--loss", type=str, default="CE", choices=["CE"])  # clip-based uses CE

# --- EMA selection (aligned with MIL) ---
parser.add_argument("--ema-alpha", type=float, default=0.80,
                    help="EMA smoothing factor for val metric (0-1)")
parser.add_argument("--select-metric", type=str, default="acc",
                    choices=["acc", "qwk", "within1", "mae"],
                    help="Validation metric for EMA-based selection")

# --- data paths / splits (names aligned with MIL) ---
parser.add_argument("--root-dir-clips", type=str, default="../data/new_takeda_processed_coco_merged_ls_flt",
                    help="Root dir with <visit>/clip_*.npy")
parser.add_argument("--split-root", type=str, default="../data/split_info_new",
                    help="Root containing per-case split folders")
parser.add_argument("--setup-list", type=str, default="original_setup",
                    help='Comma-separated cases, e.g. "original_setup,balance_test"')

# --- visit-level eval ---
parser.add_argument("--soft-voting", type=str2bool, default=True,
                    help="Visit-level aggregation via soft voting over clips")
parser.add_argument("--test-batch", type=int, default=256,
                    help="Clip inference batch size during visit-level eval")

args, _ = parser.parse_known_args()

# Default exp_name
if not args.exp_name or args.exp_name.strip() == "":
    args.exp_name = datetime.now().strftime("exp_%Y%m%d_%H%M%S")

# -------------------------
# Results / logging
# -------------------------
results_dir = os.path.join(SCRIPT_DIR, "results", args.exp_name)
log_dir = os.path.join(results_dir, "logs")
os.makedirs(log_dir, exist_ok=True)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(log_dir, f"log_train_{ts}.txt")
_log_f = open(log_path, "a", buffering=1, encoding="utf-8")
_log_f.write("\n" + "=" * 80 + f"\nRun started: {datetime.now()}\n" + "=" * 80 + "\n")

class _Tee(io.TextIOBase):
    def __init__(self, *streams): self._streams = streams
    def write(self, s):
        for st in self._streams:
            try: st.write(s); st.flush()
            except Exception: pass
        return len(s)
    def flush(self):
        for st in self._streams:
            try: st.flush()
            except Exception: pass

# Mirror stdout/stderr to file
sys.stdout = _Tee(sys.stdout, _log_f)
sys.stderr = _Tee(sys.stderr, _log_f)

@atexit.register
def _close_logfile():
    try: _log_f.flush(); _log_f.close()
    except Exception: pass

# -------------------------
# Config (driven by args)
# -------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
primary = args.primary
device = "cuda" if torch.cuda.is_available() else "cpu"

g = torch.Generator()
g.manual_seed(args.seed)

backbone_cfg = {
    "type": "STGCN",
    "gcn_adaptive": "init",
    "gcn_with_res": True,
    "tcn_type": "mstcn",
    "graph_cfg": {"layout": "coco", "mode": "spatial"},
    "pretrained": None,
}

pretrain = args.pretrain
backbone = args.backbone
freeze_backbone = args.freeze_backbone
use_cp = args.use_cp

# Auto-select backbone_state_path (same logic as MIL)
if args.backbone_state_path:
    backbone_state_path = args.backbone_state_path
else:
    if backbone == "stgcn":
        # Prefer cp.pth, otherwise j.pth
        backbone_state_path = os.path.join(SCRIPT_DIR, "weights", "cp.pth") if use_cp \
                              else os.path.join(SCRIPT_DIR, "j.pth")
    else:
        backbone_state_path = None

# Training hparams
batch_size_clips = args.batch_size_videos   # name aligned with MIL; here it means clip batch size
num_epochs       = args.epochs
init_lr          = args.lr
weight_decay     = args.weight_decay
exp_times        = args.exp_times

# Data paths / setups
root_dir_clips = args.root_dir_clips
split_root     = args.split_root
setup_list     = [s.strip() for s in args.setup_list.split(",") if s.strip()]

# Print config once (to stdout & log)
log_args(args, save_json_path=os.path.join(results_dir, "args.json"))
print_env_brief()
print_repro_cmd()

print(f"[EXP] results dir: {results_dir}")
print(f"[EXP] logs dir:    {log_dir}")

# -------------------------
# Dataset (clip-supervised; visit-level aggregation at eval)
# -------------------------
class DataSet_Classification(Dataset):
    """
    Train/val supervised per clip. Each sample is one clip .npy (T,17,3).
    """
    def __init__(self, data_list, root_dir, transform=None, data_augmentation=False):
        self.npy_data = []
        self.annotations = []
        self.data_list = data_list
        self.root_dir = root_dir
        for video_data in self.data_list:
            visit_rel = video_data.split('+++')[0].strip()
            label_txt = video_data.split('+++')[1].strip()
            visit_path = os.path.join(self.root_dir, visit_rel)
            npy_ls = sorted(glob.glob(os.path.join(visit_path, "*.npy")))
            self.npy_data.extend(npy_ls)
            self.annotations.extend([label_txt] * len(npy_ls))
        self.transform = Preprocess_Module(data_augmentation)

    def __len__(self):
        return len(self.npy_data)

    def __getitem__(self, idx):
        data_file_path = self.npy_data[idx]
        label_gmfcs = int(self.annotations[idx]) - 1  # to 0-based
        data_file_path = unicodedata.normalize('NFC', data_file_path)

        data = np.load(data_file_path)         # (T, 17, 3)
        data[np.isnan(data)] = 0

        tmp = dict(
            img_shape=(800, 1422),
            label=-1,
            start_index=0,
            modality='Pose',
            total_frames=data.shape[0],
        )
        # Two-person tiling to match your Preprocess pipeline
        tmp['keypoint'] = np.tile(data[np.newaxis, :, :, :2], (2, 1, 1, 1))
        tmp['keypoint_score'] = np.tile(data[np.newaxis, :, :, 2], (2, 1, 1))

        out = self.transform(tmp)
        x = out['keypoint'][0]  # Tensor [1,T,V,2] expected by ST-GCN pipeline
        y = label_gmfcs
        return x, y

def clip_conf_mean(path: str) -> float:
    """Mean keypoint confidence per clip (used as soft-voting weight)."""
    arr = np.load(path)  # (T,17,3)
    if arr.size == 0: return 0.0
    a = np.nan_to_num(arr[..., 2], nan=0.0)
    return max(float(a.mean()), 0.0)

# -------------------------
# Model wrappers
# -------------------------
class STGCN_Classifier(nn.Module):
    """Clip-level classifier: ST-GCN backbone + linear head."""
    def __init__(self, backbone_cfg, num_classes: int):
        super().__init__()
        args_local = dict(backbone_cfg); args_local.pop("type", None)
        self.backbone = STGCN(**args_local)
        self.cls_head = Classifier(num_classes=num_classes, dropout=0.5, latent_dim=512)
    def forward(self, keypoint):   # keypoint: [B,1,T,V,2]
        feat = self.backbone(keypoint)   # [B,C,T',V] (pooling inside head)
        logits = self.cls_head(feat)     # [B,num_classes]
        return logits

def freeze_backbone_only(model, freeze: bool = True):
    """Freeze all backbone params; keep classification head trainable."""
    m = model.module if isinstance(model, nn.DataParallel) else model
    if hasattr(m, "backbone") and hasattr(m, "cls_head"):
        for p in m.backbone.parameters():
            p.requires_grad = not freeze and p.requires_grad
        if freeze:
            m.backbone.eval()

# -------------------------
# Visit-level evaluator (soft/hard voting)
# -------------------------
@torch.no_grad()
def evaluate_visit_split(model, split_lines, root_dir, num_classes, device,
                         soft_voting=True, test_bs=256):
    """
    Evaluate at visit-level by aggregating clip predictions.
    - soft_voting: average of class probabilities weighted by clip confidence.
    - hard voting: majority of per-clip argmax.
    """
    model.eval()
    # Expand visit -> clips
    pairs = []
    for s in split_lines:
        key, lab = s.split('+++', 1)
        visit = key.strip().split('/')[-1]
        pairs.append((visit, int(lab.strip()) - 1))

    all_true, all_pred = [], []
    transform = Preprocess_Module(data_augmentation=False)

    for visit, gt in pairs:
        visit_dir = os.path.join(root_dir, visit)
        if not os.path.isdir(visit_dir):
            print(f"[WARN] missing visit_dir: {visit_dir}")
            continue
        clip_files = sorted([os.path.join(visit_dir, f) for f in os.listdir(visit_dir) if f.endswith(".npy")])
        if not clip_files:
            print(f"[WARN] no clips in: {visit_dir}")
            continue

        agg_vec = None
        hard_preds = []

        # Batched clip inference
        for i in range(0, len(clip_files), test_bs):
            batch_paths = clip_files[i:i+test_bs]

            xs = []
            for p in batch_paths:
                data = np.load(p); data[np.isnan(data)] = 0
                tmp = {
                    'img_shape': (800, 1422),
                    'label': -1,
                    'start_index': 0,
                    'modality': 'Pose',
                    'total_frames': data.shape[0],
                    'keypoint': np.tile(data[np.newaxis, :, :, :2], (2, 1, 1, 1)),
                    'keypoint_score': np.tile(data[np.newaxis, :, :, 2], (2, 1, 1)),
                }
                out = transform(tmp)['keypoint'][0].numpy()
                xs.append(out)
            batch = torch.from_numpy(np.stack(xs, 0)).float().to(device)  # [B,1,T,V,2]

            outputs = model(batch)  # [B, C]

            if soft_voting:
                probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()  # (B, C)
                w = np.array([clip_conf_mean(p) for p in batch_paths], dtype=np.float64)  # (B,)
                s = w.sum()
                if s > 0: w = w / s
                weighted = (probs * w[:, None]).sum(axis=0)  # (C,)
                agg_vec = weighted if agg_vec is None else (agg_vec + weighted)
            else:
                hard_preds.extend(outputs.argmax(dim=1).detach().cpu().tolist())

        pred = int(np.argmax(agg_vec)) if soft_voting else int(np.bincount(hard_preds, minlength=num_classes).argmax())
        all_true.append(int(gt))
        all_pred.append(int(pred))

    # Return metrics + confusion (then derive BA/macroF1)
    metrics, conf = compute_metrics(np.array(all_true), np.array(all_pred), num_classes)
    return metrics, conf

# -------------------------
# Main
# -------------------------
def main():
    print("root_dir(clips):", root_dir_clips)
    print("setup_list:", setup_list)

    val_case_means, test_case_means = [], []
    all_case_metrics = []

    for case in setup_list:
        num_class_total = 4 if case == "scores_1to4_only" else 5

        test_metrics_list = []
        test_conf_sum = None
        val_best_list = []

        split_dir = os.path.join(split_root, case)
        train_list = read_lines(os.path.join(split_dir, "train_majority.txt"))
        val_list   = read_lines(os.path.join(split_dir, "val_majority.txt"))
        test_list  = read_lines(os.path.join(split_dir, "test_majority.txt"))

        # -------- TRAIN loader (clip-level supervision) ----------
        train_ds = DataSet_Classification(train_list, root_dir_clips, data_augmentation=True)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size_clips, shuffle=True, num_workers=8,
            pin_memory=True, drop_last=True, worker_init_fn=seed_worker, generator=g
        )

        for rep in range(exp_times):
            print(f"\n===== Case: {case} | Rep {rep+1}/{exp_times} =====")

            model = STGCN_Classifier(backbone_cfg, num_classes=num_class_total)

            # Load backbone weights (mask head params; same as MIL)
            if pretrain and (backbone_state_path is not None) and os.path.isfile(backbone_state_path):
                sd = torch.load(backbone_state_path, map_location="cpu")
                sd = sd.get("state_dict", sd)
                sd = {k.replace("module.", ""): v for k, v in sd.items()}
                sd = {k: v for k, v in sd.items() if not k.startswith("cls_head.") and "fc_cls" not in k}
                try:
                    model.load_state_dict(sd, strict=False)
                except Exception as e:
                    print("[WARN] loose load backbone failed:", e)

            model = nn.DataParallel(model).to(device)

            if freeze_backbone:
                freeze_backbone_only(model, freeze=True)

            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=init_lr, weight_decay=weight_decay
            )
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-8)

            best_ema = -1.0
            ema_val = None
            best_state = None

            # ---------------- Train/Eval Loop ----------------
            for epoch in range(num_epochs):
                model.train()
                epoch_loss, total, correct = 0.0, 0, 0

                for x, y in tqdm(train_loader, desc=f"[{case}] train", file=sys.__stderr__):
                    x = x.to(device)              # [B,1,T,V,2]
                    y = y.long().to(device)

                    optimizer.zero_grad(set_to_none=True)
                    out = model(x)                # [B,C]
                    loss = F.cross_entropy(out, y)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += float(loss.item()) * x.size(0)
                    pred = out.argmax(1)
                    correct += int((pred == y).sum().item())
                    total   += int(x.size(0))

                epoch_loss /= max(total, 1)
                train_acc  = correct / max(total, 1)
                print(f"[{case}] Epoch {epoch+1}/{num_epochs}  loss={epoch_loss:.4f}  clip-acc={train_acc:.4f}")
                scheduler.step()

                # ----- VAL: visit-level aggregation (soft/hard voting) -----
                val_metrics, val_conf = evaluate_visit_split(
                    model=model, split_lines=val_list, root_dir=root_dir_clips,
                    num_classes=num_class_total, device=device,
                    soft_voting=args.soft_voting, test_bs=args.test_batch
                )

                val_macro = compute_macro_from_conf(val_conf)
                print("[VAL] "
                    f"acc={val_metrics['acc']:.4f}  qwk={val_metrics['qwk']:.4f}  "
                    f"within1={val_metrics['within1']:.4f}  mae={val_metrics['mae']:.4f}\n"
                    f"[VAL] BA(macro recall)={val_macro['balanced_acc']:.4f}  macroF1={val_macro['macro_f1']:.4f}\n"
                    f"[VAL] confusion(row-norm):\n{np.array(val_macro['conf_row_norm'])}")

                # EMA on select_metric (acc/qwk/within1: higher is better; mae: lower is better)
                sel = args.select_metric
                val_score = (val_metrics[sel] if sel in ("acc","qwk","within1") else (-val_metrics["mae"]))
                if ema_val is None: ema_val = val_score
                else:               ema_val = args.ema_alpha * ema_val + (1.0 - args.ema_alpha) * val_score
                print(f"[VAL][EMA] ema_{sel}={ema_val:.6f} (alpha={args.ema_alpha})")

                if ema_val > best_ema:
                    best_ema = ema_val
                    best_state = unwrap_state_dict(model.state_dict())
                    ckpt_path = os.path.join(results_dir, f"bestEMA_{sel}_{case}_rep{rep}.pth")
                    torch.save(best_state, ckpt_path)
                    shown_metric = (val_metrics[sel] if sel != "mae" else val_metrics["mae"])
                    print(f"[CKPT][EMA] saved -> {ckpt_path}  (epoch={epoch+1}, ema_{sel}={best_ema:.6f}, "
                          f"val_{sel}={shown_metric:.4f})")

            # TEST using the best EMA checkpoint
            if best_state is not None:
                load_unwrapped_state(model, best_state, strict=True)

            test_metrics, test_conf = evaluate_visit_split(
                model=model, split_lines=test_list, root_dir=root_dir_clips,
                num_classes=num_class_total, device=device,
                soft_voting=args.soft_voting, test_bs=args.test_batch
            )
            test_macro = compute_macro_from_conf(test_conf)
            test_metrics = dict(test_metrics,
                                BA=test_macro["balanced_acc"],
                                macroF1=test_macro["macro_f1"],
                                per_class_recall=test_macro["per_class_recall"],
                                per_class_precision=test_macro["per_class_precision"],
                                per_class_f1=test_macro["per_class_f1"])

            print("[TEST] "
                f"acc={test_metrics['acc']:.4f}  qwk={test_metrics['qwk']:.4f}  "
                f"within1={test_metrics['within1']:.4f}  mae={test_metrics['mae']:.4f}\n"
                f"[TEST] BA(macro recall)={test_metrics['BA']:.4f}  macroF1={test_metrics['macroF1']:.4f}")

            val_best_list.append(best_ema)
            test_metrics_list.append(test_metrics)
            test_conf_sum = test_conf.astype(np.float64) if test_conf_sum is None else (test_conf_sum + test_conf)

        print(f"[CASE {case}] mean best-VAL EMA({args.select_metric}) across reps: {np.mean(val_best_list):.6f}  values={val_best_list}")
        val_case_means.append(np.mean(val_best_list))

        if len(test_metrics_list) > 0:
            keys = ['acc', 'qwk', 'within1', 'mae', 'BA', 'macroF1']
            mean_metrics = {k: float(np.mean([m[k] for m in test_metrics_list])) for k in keys}
            std_metrics  = {k: float(np.std([m[k] for m in test_metrics_list], ddof=0)) for k in keys}
            avg_conf = (test_conf_sum / len(test_metrics_list)) if test_conf_sum is not None else None

            print(f"[CASE {case}] mean TEST metrics across {len(test_metrics_list)} reps:")
            print("           " + "  ".join([f"{k}={mean_metrics[k]:.4f}±{std_metrics[k]:.4f}" for k in keys]))
            if avg_conf is not None:
                print(f"[CASE {case}] mean TEST confusion:\n{avg_conf}")

            # Persist JSON summary (schema aligned with MIL)
            case_out = {
                "case": case,
                "per_rep": test_metrics_list,
                "mean": mean_metrics,
                "std": std_metrics,
                "N": len(test_list)
            }
            os.makedirs(results_dir, exist_ok=True)
            with open(os.path.join(results_dir, f"summary_{case}.json"), "w") as f:
                json.dump(case_out, f, indent=2)

            test_case_means.append((case, mean_metrics['acc']))

    print("\n========== FINAL SUMMARY ==========")
    if val_case_means:
        print("[VAL] per-case best-ACC means (EMA proxy):", val_case_means)
        print(f"[VAL] overall mean of best-ACC: {np.mean(val_case_means):.4f}")
    print("===================================\n")

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, io, atexit, math, platform, shlex
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

from dataset_mil import BagDatasetSingleNpy, collate_bags
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from models.heads import Classifier
from models.stgcn import STGCN
from models.skateformer import SkateFormer_
from utils.data_processing import Preprocess_Module

from mil_utils import (
    str2bool, seed_worker, log_args, print_env_brief, print_repro_cmd,
    unwrap_state_dict, load_unwrapped_state, read_lines, score_from_metrics,
    compute_metrics, build_window_weights_from_X, choose_pooling_for_epoch,
    apply_mil_pool, emd2_loss, apply_human_prior, argmin_cost_decoding, compute_macro_from_conf
)

# ----------------- CLI -----------------
parser = argparse.ArgumentParser()

# --- experiment/meta ---
parser.add_argument("--exp-name", type=str, default=None,
                    help="Experiment name; default = timestamp like exp_YYYYmmdd_HHMMSS")
parser.add_argument("--seed", type=int, default=3407, help="Random seed for torch/numpy workers")

# --- device ---
parser.add_argument("--gpus", type=str, default="0,1", help='CUDA_VISIBLE_DEVICES, e.g. "0" or "0,1"')
parser.add_argument("--primary", type=int, default=0, help="primary GPU index after CUDA remap")

# --- backbone / weights ---
parser.add_argument("--backbone", type=str, default="stgcn", choices=["stgcn","skateformer"])
parser.add_argument("--pretrain", type=str2bool, default=True)
parser.add_argument("--freeze-backbone", type=str2bool, default=False,
                    help="Freeze backbone, train only classifier head")
parser.add_argument("--use-cp", type=str2bool, default=False,
                    help="Use NTU checkpoint j_ntu_cp.pth when backbone=stgcn")
parser.add_argument("--backbone-state-path", type=str, default=None,
                    help="Override ckpt path; if None, auto-select by backbone/use-cp")

# --- training hyperparams ---
parser.add_argument("--batch-size-videos", type=int, default=8)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--weight-decay", type=float, default=5e-5)
parser.add_argument("--exp-times", type=int, default=5, help="number of repetitions per case")
parser.add_argument("--loss", type=str, default="CE", choices=["CE", "EMD"])
parser.add_argument("--data-aug", type=str2bool, default=True)

# --- windowing / MIL ---
parser.add_argument("--win-len", type=int, default=64)
parser.add_argument("--stride", type=int, default=32)
# >>> changed: split train/eval caps
parser.add_argument("--cap-windows-train", type=int, default=64,
                    help="Max windows per video for TRAIN. Use small value to cap.")
parser.add_argument("--cap-windows-eval", type=int, default=-1,
                    help="Max windows per video for EVAL/TEST. -1 means 'use all by stride'.")
parser.add_argument("--sample-mode", type=str, default="uniform_k",
                    choices=["uniform_k","stride"],
                    help="Sampling mode for TRAIN only; EVAL/TEST always use 'stride'.")
parser.add_argument("--frame-step", type=int, default=1)
parser.add_argument("--pad-short", type=str2bool, default=True)
parser.add_argument("--weight-pool", type=str2bool, default=False,
                    help="Use confidence-weighted MIL pooling")
parser.add_argument("--max-repeat", type=int, default=4)
parser.add_argument("--tau", type=float, default=5)
parser.add_argument("--pool-mode", type=str, default="topk_lse",
                    choices=["lse", "topk_lse", "topk", "noisy_or_ord", "staged"])
parser.add_argument("--k-topk", type=int, default=1, help="k for topk_lse/topk when not staged")

parser.add_argument("--prior-mode", type=str, default="none",
    choices=["none","linear","bayes","costdec"], help="Apply human prior at bag level")
parser.add_argument("--prior-lam", type=float, default=0.3)
parser.add_argument("--prior-alpha", type=float, default=0.5)

# --- staged pooling schedule ---
parser.add_argument("--stage1-epochs", type=int, default=5,
                    help="warmup epochs with LSE")
parser.add_argument("--stage1-tau", type=float, default=2.0)
parser.add_argument("--stage2-epochs", type=int, default=5,
                    help="middle epochs with topk_lse(k=3)")
parser.add_argument("--stage2-k", type=int, default=10)
parser.add_argument("--stage2-tau", type=float, default=1.0)
parser.add_argument("--stage3-k", type=int, default=3,
                    help="final stage topk_lse k")
parser.add_argument("--stage3-tau", type=float, default=0.5)

# --- EMA selection ---
parser.add_argument("--ema-alpha", type=float, default=0.80,
                    help="EMA smoothing factor for val acc selection (0-1)")
parser.add_argument("--select-metric", type=str, default="acc",
                    choices=["acc", "qwk", "within1", "mae"],
                    help="Validation metric used for EMA-based model selection")

# --- data paths / splits ---
parser.add_argument("--root-dir-single-npy", type=str, default="../data/new_takeda_processed_coco_merged_video",
                    help="Directory that contains <video>.npy with shape [T,17,3]")
parser.add_argument("--split-root", type=str, default="../data/split_info_new",
                    help="Root that contains per-case split folders")
parser.add_argument("--setup-list", type=str, default="original_setup",
                    help='Comma-separated cases; e.g. "original_setup,balance_test"')

args, _ = parser.parse_known_args()

# default exp_name
if not args.exp_name or args.exp_name.strip() == "":
    args.exp_name = datetime.now().strftime("exp_%Y%m%d_%H%M%S")

# -------------------------
# Results / logging
# -------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

results_dir = os.path.join(script_dir, "results", args.exp_name)
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

# auto-select backbone_state_path if not provided
if args.backbone_state_path:
    backbone_state_path = args.backbone_state_path
else:
    if backbone == "skateformer":
        backbone_state_path = os.path.join(script_dir, "SkateFormer_j.pt")
    elif backbone == "stgcn":
        backbone_state_path = os.path.join(script_dir, "weights", "cp.pth") if use_cp \
                              else os.path.join(script_dir, "j.pth")
    else:
        backbone_state_path = None

# training hparams
batch_size_videos = args.batch_size_videos
num_epochs        = args.epochs
init_lr           = args.lr
weight_decay      = args.weight_decay
exp_times         = args.exp_times

# windowing / MIL
WIN_LEN     = args.win_len
STRIDE      = args.stride
CAP_TRAIN   = args.cap_windows_train
CAP_EVAL    = None if (args.cap_windows_eval is None or args.cap_windows_eval < 0) else int(args.cap_windows_eval)
SAMPLE_MODE = args.sample_mode          # TRAIN only
FRAME_STEP  = args.frame_step
PAD_SHORT   = args.pad_short
WEIGHT_POOL = args.weight_pool
TAU = args.tau
POOL_MODE = args.pool_mode

# EMA
EMA_ALPHA   = args.ema_alpha
SELECT_METRIC = args.select_metric

# data paths / setups
root_dir_single_npy = args.root_dir_single_npy
split_root          = args.split_root
setup_list          = [s.strip() for s in args.setup_list.split(",") if s.strip()]

# --- Print all configs once (to stdout & log) ---
log_args(args, save_json_path=os.path.join(results_dir, "args.json"))
print_env_brief()
print_repro_cmd()

print(f"[EXP] results dir: {results_dir}")
print(f"[EXP] logs dir:    {log_dir}")

# -------------------------
# Model wrappers
# -------------------------
class SkateFormerAdapter(nn.Module):
    """Adapts X:[B*W,1,T,V,3] -> SkateFormer input [B*W,3,T,V,1], returns logits [B*W,C]."""
    def __init__(self, num_classes: int, num_frames: int, num_joints: int = 17):
        super().__init__()
        self.net = SkateFormer_(
            in_channels=3,
            num_classes=num_classes,
            num_frames=num_frames,
            num_points=num_joints,
            num_people=1,
            index_t=True,
            global_pool="avg"
        )
    def forward(self, x_bw):  # x_bw: [B*W, 1, T, V, 3]
        BW, _, T, V, C = x_bw.shape
        x = x_bw.squeeze(1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)  # [B*W,3,T,V,1]
        idx_t = torch.arange(T, device=x.device).unsqueeze(0).expand(BW, T) # [B*W,T]
        return self.net(x, index_t=idx_t)

class STGCN_Classifier(nn.Module):
    """Per-window classifier: ST-GCN backbone + linear head -> logits per window."""
    def __init__(self, backbone_cfg, num_classes: int):
        super().__init__()
        args_local = dict(backbone_cfg); args_local.pop("type", None)
        self.backbone = STGCN(**args_local)
        self.cls_head = Classifier(num_classes=num_classes, dropout=0.5, latent_dim=512)
    def forward(self, x):  # x: [B*W, 1, T, V, 3]
        feat = self.backbone(x)       # [B*W, C, T/4, V]
        logits = self.cls_head(feat)  # [B*W, C]
        return logits

def build_perwindow_model(backbone_name: str, num_classes: int):
    if backbone_name.lower() == "skateformer":
        return SkateFormerAdapter(num_classes=num_classes, num_frames=WIN_LEN)
    else:
        return STGCN_Classifier(backbone_cfg, num_classes=num_classes)

def freeze_backbone_only(model, freeze: bool = True):
    """Freeze all backbone params, keep classification head trainable."""
    m = model.module if isinstance(model, nn.DataParallel) else model
    # ST-GCN
    if hasattr(m, "backbone") and hasattr(m, "cls_head"):
        for p in m.backbone.parameters():
            p.requires_grad = not freeze and p.requires_grad
        if freeze:
            m.backbone.eval()
    # SkateFormer
    if hasattr(m, "net") and hasattr(m.net, "head"):
        for _, p in m.net.named_parameters():
            p.requires_grad = not freeze
        for p in m.net.head.parameters():
            p.requires_grad = True
        if freeze:
            m.net.train()
            for mod_name, mod in m.net.named_modules():
                if mod_name.startswith("head"):
                    continue
                if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
                    mod.eval()
                if isinstance(mod, nn.Dropout):
                    mod.p = getattr(mod, "p", 0.0)

# -------------------------
# Train / Eval (MIL)
# -------------------------
def run_epoch(model, loader, optimizer=None, epoch: int = 0, pool_cfg: Optional[dict] = None):
    train = optimizer is not None
    model.train(mode=train)
    total_loss, total_correct, total_count = 0.0, 0, 0
    for X, M, Y in tqdm(loader, desc="train" if train else "eval", file=sys.__stderr__):
        B, W = X.shape[:2]
        X = X.to(device); M = M.to(device); Y = Y.to(device)
        if WEIGHT_POOL:
            Q = build_window_weights_from_X(X, M)
        Xfw = X.flatten(0,1)
        logits_fw = model(Xfw)     # [B*W,C]
        C = logits_fw.shape[-1]
        logits_bw = logits_fw.view(B, W, C)
        
        cfg = pool_cfg or {"mode": POOL_MODE, "tau": TAU, "k": args.k_topk}
        video_logits = apply_mil_pool(
            logits_bw, M, mode=cfg["mode"], tau=cfg["tau"], k=cfg.get("k"),
            weights_bw=(Q if WEIGHT_POOL else None)
        )
        if args.loss == "CE":
            loss = F.cross_entropy(video_logits, Y)
        elif args.loss == "EMD":
            loss = emd2_loss(video_logits, Y)
        else:
            loss = None
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * B

        # ---- after computing video_logits ----
        if args.prior_mode in ("linear","bayes"):
            priors = None
            if args.prior_mode == "bayes" and hasattr(args, "prior_priors") and args.prior_priors:
                priors = [float(x) for x in args.prior_priors.split(",")]
            video_logits = apply_human_prior(
                video_logits, mode=args.prior_mode,
                lam=args.prior_lam, alpha=args.prior_alpha, priors=priors
            )
            pred = video_logits.argmax(dim=1)
        elif args.prior_mode == "costdec":
            pred = argmin_cost_decoding(video_logits, over_penalty=2.0)
        else:
            pred = video_logits.argmax(dim=1)
        total_correct += int((pred == Y).sum().item())
        total_count  += B
    avg_loss = total_loss / max(total_count, 1)
    avg_acc  = total_correct / max(total_count, 1)
    return avg_loss, avg_acc

def evaluate_split(model, split_lines, num_classes, mode_str, pool_cfg: Optional[dict] = None):
    """
    For EVAL/TEST we always want deterministic, full-coverage windows:
    - sample_mode='stride'
    - cap_windows=None (or user-provided --cap-windows-eval >= 0)
    """
    ds = BagDatasetSingleNpy(
        data_list=split_lines,
        root_dir=root_dir_single_npy,
        win_len=WIN_LEN, stride=STRIDE,
        mode=mode_str,
        cap_windows=(CAP_EVAL if CAP_EVAL is not None else None),
        sample_mode="stride",                 # <<< full coverage by stride
        frame_step=FRAME_STEP,
        pad_short=PAD_SHORT,
        transform=Preprocess_Module(data_augmentation=False),
    )
    loader = DataLoader(
        ds, batch_size=batch_size_videos, shuffle=False, num_workers=8,
        collate_fn=collate_bags, pin_memory=True
    )
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for X, M, Y in loader:
            B, W = X.shape[:2]
            X, M, Y = X.to(device), M.to(device), Y.to(device)
            if WEIGHT_POOL:
                Q = build_window_weights_from_X(X, M)
            logits_fw = model(X.flatten(0,1))
            C = logits_fw.shape[-1]
            logits_bw = logits_fw.view(B, W, C)
            cfg = pool_cfg or {"mode": POOL_MODE, "tau": TAU, "k": args.k_topk}
            video_logits = apply_mil_pool(
                logits_bw, M, mode=cfg["mode"], tau=cfg["tau"], k=cfg.get("k"),
                weights_bw=(Q if WEIGHT_POOL else None)
            )

            # ---- after computing video_logits ----
            if args.prior_mode in ("linear","bayes"):
                priors = None
                if args.prior_mode == "bayes" and hasattr(args, "prior_priors") and args.prior_priors:
                    priors = [float(x) for x in args.prior_priors.split(",")]
                video_logits = apply_human_prior(
                    video_logits, mode=args.prior_mode,
                    lam=args.prior_lam, alpha=args.prior_alpha, priors=priors
                )
                pred = video_logits.argmax(dim=1)
            elif args.prior_mode == "costdec":
                pred = argmin_cost_decoding(video_logits, over_penalty=2.0)
            else:
                pred = video_logits.argmax(dim=1)

            all_true.extend(Y.cpu().tolist())
            all_pred.extend(pred.cpu().tolist())
    metrics, conf = compute_metrics(np.array(all_true), np.array(all_pred), num_classes)
    return metrics, conf

# -------------------------
# Main
# -------------------------
def main():
    print("root_dir(single .npy videos):", root_dir_single_npy)
    print("setup_list:", setup_list)
    print(f"SAMPLE_MODE(train only): {SAMPLE_MODE} | CAP_TRAIN={CAP_TRAIN} | CAP_EVAL={'all' if CAP_EVAL is None else CAP_EVAL}")

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

        # -------- TRAIN loader: uses cap + chosen sample mode ----------
        train_ds = BagDatasetSingleNpy(
            data_list=train_list,
            root_dir=root_dir_single_npy,
            win_len=WIN_LEN, stride=STRIDE,
            mode="train",
            cap_windows=CAP_TRAIN,                 # <<< cap only here
            sample_mode=SAMPLE_MODE,               # <<< could be 'uniform_k'
            frame_step=FRAME_STEP,
            pad_short=PAD_SHORT,
            transform=Preprocess_Module(data_augmentation=args.data_aug),
            balance_videos=True, target_mul=1.0, max_repeat=args.max_repeat,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size_videos, shuffle=True, num_workers=8,
            collate_fn=collate_bags, pin_memory=True, drop_last=True,
            worker_init_fn=seed_worker, generator=g,
        )

        for rep in range(exp_times):
            print(f"\n===== Case: {case} | Rep {rep+1}/{exp_times} =====")

            perwin = build_perwindow_model(backbone, num_classes=num_class_total)

            # preload weights if available
            if pretrain and (backbone_state_path is not None) and os.path.isfile(backbone_state_path):
                sd = torch.load(backbone_state_path, map_location="cpu")
                sd = sd.get("state_dict", sd)
                sd = {k.replace("module.", ""): v for k, v in sd.items()}
                sd = {k: v for k, v in sd.items() if not k.startswith("cls_head.") and not k.startswith("head.")}
                try:
                    perwin.load_state_dict(sd, strict=False)
                except Exception as e:
                    print("[WARN] loose load backbone failed:", e)
                
                # model_state = perwin.state_dict()
                # missing_keys = []
                # mismatched = []
                # for k, v in sd.items():
                #     if k not in model_state:
                #         missing_keys.append(k)
                #     elif model_state[k].shape != v.shape:
                #         mismatched.append((k, model_state[k].shape, v.shape))
                # print("[SkateFormer] pretrain missing keys:", len(missing_keys))
                # print("[SkateFormer] pretrain mismatched shapes:", mismatched[:10])


            perwin = nn.DataParallel(perwin).to(device)

            if freeze_backbone:
                freeze_backbone_only(perwin, freeze=True)

            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, perwin.parameters()),
                lr=init_lr, weight_decay=weight_decay
            )
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-8)

            best_ema = -1.0
            ema_val = None
            best_state = None

            for epoch in range(num_epochs):
                if hasattr(train_ds, "set_epoch"):
                    train_ds.set_epoch(epoch)

                pool_cfg = choose_pooling_for_epoch(epoch, POOL_MODE, args)

                tr_loss, tr_acc = run_epoch(perwin, train_loader, optimizer=optimizer, epoch=epoch, pool_cfg=pool_cfg)
                print(f"[{case}] Epoch {epoch+1}/{num_epochs}  "
                f"loss={tr_loss:.4f}  acc={tr_acc:.4f}  "
                f"pool={{mode:{pool_cfg['mode']}, tau:{pool_cfg['tau']}, k:{pool_cfg.get('k')}}}")
                scheduler.step()

                # ----- EVAL (no cap, stride coverage) -----
                val_pool_cfg = pool_cfg
                val_metrics, val_conf = evaluate_split(perwin, val_list, num_class_total, mode_str="val", pool_cfg=val_pool_cfg)
                val_score = score_from_metrics(val_metrics, SELECT_METRIC)

                val_macro = compute_macro_from_conf(val_conf)
                print("[VAL] "
                    f"acc={val_metrics['acc']:.4f}  qwk={val_metrics['qwk']:.4f}  "
                    f"within1={val_metrics['within1']:.4f}  mae={val_metrics['mae']:.4f}\n"
                    f"[VAL] BA(macro recall)={val_macro['balanced_acc']:.4f}  macroF1={val_macro['macro_f1']:.4f}\n"
                    f"[VAL] select_metric={SELECT_METRIC} | raw_val_score={val_score:.6f}\n"
                    f"[VAL] confusion(row-norm):\n{np.array(val_macro['conf_row_norm'])}")

                # EMA update on selected metric
                if ema_val is None:
                    ema_val = val_score
                else:
                    ema_val = EMA_ALPHA * ema_val + (1.0 - EMA_ALPHA) * val_score

                print(f"[VAL][EMA] ema_{SELECT_METRIC}={ema_val:.6f} (alpha={EMA_ALPHA})")

                # Save when EMA improves (maximize ema_val; for mae we already flipped sign)
                if ema_val > best_ema:
                    best_ema = ema_val
                    best_state = unwrap_state_dict(perwin.state_dict())
                    ckpt_path = os.path.join(results_dir, f"bestEMA_{SELECT_METRIC}_{case}_rep{rep}.pth")
                    torch.save(best_state, ckpt_path)
                    shown_metric = val_metrics[SELECT_METRIC]
                    print(f"[CKPT][EMA] saved -> {ckpt_path}  (epoch={epoch+1}, ema_{SELECT_METRIC}={best_ema:.6f}, "
                        f"val_{SELECT_METRIC}={shown_metric:.4f})")

            # TEST with best
            if best_state is not None:
                load_unwrapped_state(perwin, best_state, strict=True)

            test_pool_cfg = (choose_pooling_for_epoch(args.epochs-1, "staged", args)
                 if POOL_MODE == "staged"
                 else {"mode": POOL_MODE, "tau": TAU, "k": args.k_topk})
            
            test_metrics, test_conf = evaluate_split(perwin, test_list, num_class_total, mode_str="test", pool_cfg=test_pool_cfg)
            test_macro = compute_macro_from_conf(test_conf)
            # merge for averaging + JSON
            test_metrics = dict(test_metrics,
                                BA=test_macro["balanced_acc"],
                                macroF1=test_macro["macro_f1"],
                                per_class_recall=test_macro["per_class_recall"],
                                per_class_precision=test_macro["per_class_precision"],
                                per_class_f1=test_macro["per_class_f1"])

            print("[TEST] "
                f"acc={test_metrics['acc']:.4f}  qwk={test_metrics['qwk']:.4f}  "
                f"within1={test_metrics['within1']:.4f}  mae={test_metrics['mae']:.4f}\n"
                f"[TEST] BA(macro recall)={test_metrics['BA']:.4f}  macroF1={test_metrics['macroF1']:.4f}\n"
                f"[TEST] confusion(row-norm):\n{np.array(test_macro['conf_row_norm'])}")

            val_best_list.append(best_ema)
            test_metrics_list.append(test_metrics)
            test_conf_sum = test_conf.astype(np.float64) if test_conf_sum is None else (test_conf_sum + test_conf)

        print(f"[CASE {case}] mean best-VAL EMA({SELECT_METRIC}) across reps: {np.mean(val_best_list):.6f}  values={val_best_list}")
        val_case_means.append(np.mean(val_best_list))

        if len(test_metrics_list) > 0:
            keys = ['acc', 'qwk', 'within1', 'mae',  'BA', 'macroF1']
            mean_metrics = {k: float(np.mean([m[k] for m in test_metrics_list])) for k in keys}
            std_metrics  = {k: float(np.std([m[k] for m in test_metrics_list], ddof=0)) for k in keys}
            avg_conf = (test_conf_sum / len(test_metrics_list)) if test_conf_sum is not None else None

            print(f"[CASE {case}] mean TEST metrics across {len(test_metrics_list)} reps:")
            print("           " + "  ".join([f"{k}={mean_metrics[k]:.4f}±{std_metrics[k]:.4f}" for k in keys]))
            if avg_conf is not None:
                print(f"[CASE {case}] mean TEST confusion:\n{avg_conf}")

            test_case_means.append((case, mean_metrics['acc']))

            test_size = len(test_list)
            case_out = {
                "case": case,
                "per_rep": test_metrics_list,
                "mean": mean_metrics,
                "std": std_metrics,
                "N": test_size
            }
            os.makedirs(results_dir, exist_ok=True)
            with open(os.path.join(results_dir, f"summary_{case}.json"), "w") as f:
                json.dump(case_out, f, indent=2)

    print("\n========== FINAL SUMMARY ==========")
    if val_case_means:
        print("[VAL] per-case best-ACC means:", val_case_means)
        print(f"[VAL] overall mean of best-ACC: {np.mean(val_case_means):.4f}")
    print("===================================\n")

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    main()

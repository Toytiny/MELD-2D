#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust MIL Eval:
- Deterministic VAL/TEST coverage: sample_mode='stride', CAP_EVAL=-1 => all windows
- Exact train-time pooling & priors; can override via --override-*
- Safe checkpoint loading: load on single-GPU model first, then wrap DataParallel (if any)
- Saves per-ckpt metrics + confusion + macro stats; per-video CSV (probs) + windows.csv
"""

import os, sys, json, argparse, platform, hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# NEW: for confusion matrix plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==== project paths ====
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

# ==== project modules ====
from dataset_mil import BagDatasetSingleNpy, collate_bags
from models.heads import Classifier
from models.stgcn import STGCN
from models.skateformer import SkateFormer_
from utils.data_processing import Preprocess_Module

from mil_utils import (
    str2bool, build_window_weights_from_X, choose_pooling_for_epoch,
    apply_mil_pool, apply_human_prior, argmin_cost_decoding,
    compute_metrics, read_lines, read_challenge_file
)

# -------------------------
# argparse
# -------------------------
parser = argparse.ArgumentParser("Evaluate saved MIL checkpoints (aligned with training logic)")
parser.add_argument("--exp-name", type=str, required=True, help="Experiment name under results/")
parser.add_argument("--gpus", type=str, default="0,1", help='CUDA_VISIBLE_DEVICES, e.g. "0" or "0,1"')
parser.add_argument("--primary", type=int, default=0)
parser.add_argument(
    "--splits", type=str, default="test",
    help='Comma-separated: "val,test", "test", "val", "challenge", or combinations'
)
parser.add_argument("--save-details", type=str2bool, default=True, help="Save per-video predictions to CSV")
parser.add_argument("--print-details", type=str2bool, default=False, help="Also print per-video predictions to stdout")
parser.add_argument("--batch-size-videos", type=int, default=None, help="Override batch size (default: from args.json)")
parser.add_argument("--ckpt-pattern", type=str, default="bestEMA_*.pth", help="Glob for checkpoints in results/<exp>/")

# Optional: override pooling/prior (if not provided, strictly replicate train args.json)
parser.add_argument("--override-pool-mode", type=str, default=None,
                    choices=[None,"lse","topk_lse","topk","noisy_or_ord","staged"])
parser.add_argument("--override-tau", type=float, default=None)
parser.add_argument("--override-k", type=int, default=None)
parser.add_argument("--override-weight-pool", type=str2bool, default=None)
parser.add_argument("--override-prior-mode", type=str, default=None,
                    choices=[None,"none","linear","bayes","costdec"])
parser.add_argument("--override-prior-lam", type=float, default=None)
parser.add_argument("--override-prior-alpha", type=float, default=None)
args = parser.parse_args()

# -------------------------
# env & load training config
# -------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = "cuda" if torch.cuda.is_available() else "cpu"
results_dir = os.path.join(SCRIPT_DIR, "results", args.exp_name)
args_json = os.path.join(results_dir, "args.json")
assert os.path.isfile(args_json), f"args.json not found: {args_json}"

with open(args_json, "r", encoding="utf-8") as f:
    train_cfg: Dict[str, Any] = json.load(f)

def _get(k, default=None): return train_cfg.get(k, default)

# ---- paths / split ----
root_dir_single_npy = _get("root_dir_single_npy")
split_root          = _get("split_root")
setup_list_str      = _get("setup_list", "original_setup")
setup_list          = [s.strip() for s in setup_list_str.split(",") if s.strip()]

# ---- windowing / MIL (training-time args) ----
WIN_LEN  = int(_get("win_len", 64))
STRIDE   = int(_get("stride", 32))
CAP_TRAIN = int(_get("cap_windows_train", 64))
_raw_cap_eval = _get("cap_windows_eval", -1)
CAP_EVAL = None if (_raw_cap_eval is None or int(_raw_cap_eval) < 0) else int(_raw_cap_eval)

SAMPLE_MODE_EVAL = "stride"
FRAME_STEP  = int(_get("frame_step", 1))
PAD_SHORT   = bool(_get("pad_short", True))

# weight-pooling flag (confidence-weighted MIL)
WEIGHT_POOL = bool(_get("weight_pool", False)) if args.override_weight_pool is None else bool(args.override_weight_pool)

# ---- pooling params (default: same as training; can be overridden) ----
POOL_MODE = _get("pool_mode", _get("POOL_MODE", "topk_lse"))
TAU       = float(_get("tau", 5.0))
K_TOPK    = int(_get("k_topk", 1))

# staged schedule (training-time)
stage1_epochs = int(_get("stage1_epochs", 5))
stage1_tau    = float(_get("stage1_tau", 2.0))
stage2_epochs = int(_get("stage2_epochs", 5))
stage2_k      = int(_get("stage2_k", 10))
stage2_tau    = float(_get("stage2_tau", 1.0))
stage3_k      = int(_get("stage3_k", 3))
stage3_tau    = float(_get("stage3_tau", 0.5))
num_epochs    = int(_get("epochs", 200))  # used to pick the final stage

# allow overrides
if args.override_pool_mode is not None: POOL_MODE = args.override_pool_mode
if args.override_tau is not None:        TAU = args.override_tau
if args.override_k is not None:          K_TOPK = args.override_k

# ---- human priors (training-time; can be overridden) ----
PRIOR_MODE   = _get("prior_mode", "none")   # none | linear | bayes | costdec
PRIOR_LAM    = float(_get("prior_lam", 0.3))
PRIOR_ALPHA  = float(_get("prior_alpha", 0.5))
PRIOR_PRIORS = _get("prior_priors", None)   # optional string like "p1,p2,..."

if args.override_prior_mode is not None:  PRIOR_MODE  = args.override_prior_mode
if args.override_prior_lam  is not None:  PRIOR_LAM   = args.override_prior_lam
if args.override_prior_alpha is not None: PRIOR_ALPHA = args.override_prior_alpha

# ---- backbone ----
backbone = _get("backbone", "stgcn")

# ---- batch size ----
batch_size_videos = args.batch_size_videos if args.batch_size_videos is not None else int(_get("batch_size_videos", 1))

# -------------------------
# backbone wrappers (same as training)
# -------------------------
backbone_cfg = {
    "type": "STGCN",
    "gcn_adaptive": "init",
    "gcn_with_res": True,
    "tcn_type": "mstcn",
    "graph_cfg": {"layout": "coco", "mode": "spatial"},
    "pretrained": None,
}

class SkateFormerAdapter(nn.Module):
    """Adapts per-window skeleton tensors to SkateFormer and returns logits: [B*W, C]."""
    def __init__(self, num_classes: int, num_frames: int, num_joints: int = 17):
        super().__init__()
        self.net = SkateFormer_(
            in_channels=3, num_classes=num_classes,
            num_frames=num_frames, num_points=num_joints,
            num_people=1, index_t=True, global_pool="avg"
        )
    def forward(self, x_bw):  # x_bw: [B*W,1,T,V,3]
        BW, _, T, V, C = x_bw.shape
        x = x_bw.squeeze(1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)  # [B*W,3,T,V,1]
        idx_t = torch.arange(T, device=x_bw.device).unsqueeze(0).expand(BW, T)
        return self.net(x, index_t=idx_t)

class STGCN_Classifier(nn.Module):
    """ST-GCN backbone + linear head -> logits per window."""
    def __init__(self, backbone_cfg, num_classes: int):
        super().__init__()
        args_local = dict(backbone_cfg); args_local.pop("type", None)
        self.backbone = STGCN(**args_local)
        self.cls_head = Classifier(num_classes=num_classes, dropout=0.5, latent_dim=512)
    def forward(self, x):  # [B*W,1,T,V,3]
        feat = self.backbone(x)       # [B*W, C, T/4, V]
        return self.cls_head(feat)    # [B*W, C]

def build_perwindow_model(backbone_name: str, num_classes: int):
    if backbone_name.lower() == "skateformer":
        return SkateFormerAdapter(num_classes=num_classes, num_frames=WIN_LEN)
    else:
        return STGCN_Classifier(backbone_cfg, num_classes=num_classes)

# -------------------------
# utils
# -------------------------
def compute_macro_from_conf(conf: np.ndarray) -> Dict[str, Any]:
    conf = conf.astype(np.float64); eps = 1e-12
    support = conf.sum(axis=1); pred_sum = conf.sum(axis=0); tp = np.diag(conf)
    recall = tp / np.maximum(support, eps)
    prec   = tp / np.maximum(pred_sum, eps)
    f1     = 2 * prec * recall / np.maximum(prec + recall, eps)
    return {
        "per_class_recall": recall.tolist(),
        "per_class_precision": prec.tolist(),
        "per_class_f1": f1.tolist(),
        "balanced_acc": float(np.nanmean(recall)),
        "macro_f1": float(np.nanmean(f1)),
        "conf_row_norm": (conf / np.maximum(support[:, None], eps)).tolist(),
    }

def make_loader(list_lines: List[str], mode: str) -> DataLoader:
    ds = BagDatasetSingleNpy(
        data_list=list_lines,
        root_dir=root_dir_single_npy,
        win_len=WIN_LEN, stride=STRIDE, mode=mode,
        cap_windows=(CAP_EVAL if CAP_EVAL is not None else None),
        sample_mode=SAMPLE_MODE_EVAL,
        frame_step=FRAME_STEP, pad_short=PAD_SHORT,
        transform=Preprocess_Module(data_augmentation=False),
    )
    return DataLoader(
        ds, batch_size=batch_size_videos, shuffle=False, num_workers=16,
        collate_fn=collate_bags, pin_memory=True
    )

def safe_load_ckpt_to_model(ckpt_path: str, model: nn.Module):
    """Load unwrapped state_dict to a single-GPU model, then wrap DP if needed."""
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing or unexpected:
        print(f"[LOAD][WARN] missing={len(missing)} unexpected={len(unexpected)}")
        if missing:   print("  missing (sample):", missing[:8])
        if unexpected:print("  unexpected (sample):", unexpected[:8])

def eval_collect(model: nn.Module,
                 list_lines: List[str],
                 num_classes: int,
                 split_name: str,
                 save_csv_path: Optional[str],
                 print_details: bool,
                 final_stage_cfg: Dict[str, Any]):
    loader = make_loader(list_lines, mode="val" if split_name=="val" else "test")
    model.eval()
    all_true, all_pred = [], []
    details = []
    win_rows = []  # 记录每个视频有效窗口数
    with torch.no_grad():
        idx_ptr = 0
        for X, M, Y in loader:
            B, W = X.shape[:2]
            X, M, Y = X.to(device), M.to(device), Y.to(device)
            Q = build_window_weights_from_X(X, M) if WEIGHT_POOL else None

            logits_fw = model(X.flatten(0,1))            # [B*W, C]
            C = logits_fw.shape[-1]
            logits_bw = logits_fw.view(B, W, C)

            video_logits = apply_mil_pool(
                logits_bw, M,
                mode=final_stage_cfg["mode"],
                tau=final_stage_cfg.get("tau"),
                k=final_stage_cfg.get("k"),
                weights_bw=(Q if WEIGHT_POOL else None)
            )

            # Priors / decoding
            if PRIOR_MODE in ("linear","bayes"):
                priors = [float(x) for x in PRIOR_PRIORS.split(",")] if (PRIOR_PRIORS and isinstance(PRIOR_PRIORS, str)) else None
                video_logits = apply_human_prior(video_logits, mode=PRIOR_MODE, lam=PRIOR_LAM, alpha=PRIOR_ALPHA, priors=priors)
                pred = video_logits.argmax(dim=1)
            elif PRIOR_MODE == "costdec":
                pred = argmin_cost_decoding(video_logits, over_penalty=2.0)
            else:
                pred = video_logits.argmax(dim=1)

            probs = torch.softmax(video_logits, dim=1)   # [B, C]

            for i in range(B):
                line  = list_lines[idx_ptr + i]
                gt_i  = int(Y[i].item())
                pd_i  = int(pred[i].item())
                ok_i  = int(pd_i == gt_i)
                prob_i = probs[i].detach().cpu().numpy().tolist()
                w_i    = int(M[i].sum().item())

                if print_details:
                    print(f"[{split_name}] {line} | gt={gt_i} pred={pd_i} ok={ok_i} probs={np.round(prob_i,4)} W={w_i}")

                details.append({"split": split_name, "video_id": line, "gt": gt_i, "pred": pd_i, "ok": ok_i, "probs": prob_i})
                win_rows.append({"video_id": line, "valid_windows": w_i})
                all_true.append(gt_i); all_pred.append(pd_i)

            idx_ptr += B

    metrics, conf = compute_metrics(np.array(all_true), np.array(all_pred), num_classes)

    # Save CSV + windows
    if save_csv_path:
        import csv
        p = Path(save_csv_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["split","video_id","gt","pred","ok","probs"])
            w.writeheader()
            for d in details: w.writerow(d)
        with open(p.with_suffix(".windows.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["video_id","valid_windows"])
            w.writeheader()
            for r in win_rows: w.writerow(r)

    return metrics, conf, details

# NEW: simple confusion matrix plotter (average over ckpts or per-ckpt)
def plot_confusion_matrix(conf: np.ndarray,
                          out_path: str,
                          title: str = "",
                          normalize: bool = False):
    """
    conf: 2D numpy array, shape [C, C]
    out_path: path to save png
    normalize: if True, row-normalize for visualization
    """
    cm = conf.astype(np.float64)
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        cm = cm / row_sum

    num_classes = cm.shape[0]
    # label as 1..C (你如果内部用0..4，更习惯可以改成 range(num_classes))
    classes = np.arange(1, num_classes + 1)

    fig, ax = plt.subplots(figsize=(4 + num_classes * 0.6, 4 + num_classes * 0.6))
    im = ax.imshow(cm, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Normalized" if normalize else "Count", rotation=270, labelpad=15)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if title:
        ax.set_title(title)

    # 在格子里标注数值
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm[i, j]
            if normalize:
                text = format(val, ".2f")
            else:
                text = str(int(round(val)))
            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)

# -------------------------
# Main
# -------------------------
def main():
    print("========== ENV ==========")
    devname = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"python: {platform.python_version()} | cuda: {torch.cuda.is_available()} | device: {devname}")
    print("=========================\n")

    # Find checkpoints
    ckpts = sorted([str(p) for p in Path(results_dir).glob(args.ckpt_pattern)])
    assert len(ckpts) > 0, f"No checkpoints matched in {results_dir} with pattern {args.ckpt_pattern}"
    print(f"Found {len(ckpts)} ckpts:")
    for p in ckpts: print(" -", os.path.basename(p))

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    assert all(s in ("val", "test", "challenge") for s in splits)


    # Output directory
    out_root = os.path.join(results_dir, "eval_outputs")
    Path(out_root).mkdir(exist_ok=True, parents=True)

    # Determine final pooling config for eval
    if POOL_MODE == "staged":
        # 与训练最后一个 epoch 一致
        A = argparse.Namespace(
            pool_mode=POOL_MODE, tau=TAU, k_topk=K_TOPK,
            stage1_epochs=stage1_epochs, stage1_tau=stage1_tau,
            stage2_epochs=stage2_epochs, stage2_k=stage2_k, stage2_tau=stage2_tau,
            stage3_k=stage3_k, stage3_tau=stage3_tau,
            epochs=num_epochs
        )
        final_stage_cfg = choose_pooling_for_epoch(A.epochs - 1, "staged", A)
    else:
        final_stage_cfg = {"mode": POOL_MODE, "tau": TAU, "k": K_TOPK}

    print(f"[EVAL] final pooling cfg -> {final_stage_cfg} | WEIGHT_POOL={WEIGHT_POOL} | PRIOR={PRIOR_MODE} | CAP_EVAL={'all' if CAP_EVAL is None else CAP_EVAL}")
    print(f"[EVAL] WIN_LEN={WIN_LEN} STRIDE={STRIDE} FRAME_STEP={FRAME_STEP} PAD_SHORT={PAD_SHORT}")
    print(f"[EVAL] root_dir={root_dir_single_npy} | split_root={split_root} | setups={setup_list}")

    # Evaluate each case independently (so class count matches split definition)
    for case in setup_list:
        num_class_total = 4 if case == "scores_1to4_only" else 5
        split_dir = os.path.join(split_root, case)
        val_list  = read_lines(os.path.join(split_dir, "val_majority.txt"))
        test_list = read_lines(os.path.join(split_dir, "test_majority.txt"))
        challenge_txt = os.path.join(split_dir, "test_challenge.txt")
        challenge_list = read_challenge_file(challenge_txt) if os.path.isfile(challenge_txt) else []

        print(f"\n===== CASE: {case} | C={num_class_total} =====")

        per_ckpt_metrics: Dict[str, Dict[str, Any]] = {}

        for ck in ckpts:
            # ---- Build single-GPU model, load, then optionally wrap DP ----
            model_single = build_perwindow_model(backbone, num_classes=num_class_total).to(device)
            safe_load_ckpt_to_model(ck, model_single)
            # （可选）打印一个权重范数做 sanity check
            try:
                print("Wnorm(after load):", (model_single.cls_head.fc.weight.data.norm().item()))
            except Exception:
                pass

            model = model_single
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)

            per_ckpt_metrics[ck] = {}

            if "val" in splits:
                csv_path = os.path.join(out_root, f"{case}__{Path(ck).stem}__val.csv") if args.save_details else None
                m_val, conf_val, _ = eval_collect(
                    model, val_list, num_class_total, "val", csv_path, args.print_details, final_stage_cfg
                )
                val_macro = compute_macro_from_conf(conf_val)
                per_ckpt_metrics[ck]["val"] = {"metrics": m_val, "conf": conf_val, "macro": val_macro}

                print(f"[VAL][{Path(ck).stem}] "
                      f"acc={m_val['acc']:.4f}  qwk={m_val['qwk']:.4f}  within1={m_val['within1']:.4f}  mae={m_val['mae']:.4f}\n"
                      f"    BA(macro recall)={val_macro['balanced_acc']:.4f}  macroF1={val_macro['macro_f1']:.4f}")

            if "test" in splits:
                csv_path = os.path.join(out_root, f"{case}__{Path(ck).stem}__test.csv") if args.save_details else None
                m_test, conf_test, _ = eval_collect(
                    model, test_list, num_class_total, "test", csv_path, args.print_details, final_stage_cfg
                )
                test_macro = compute_macro_from_conf(conf_test)
                per_ckpt_metrics[ck]["test"] = {"metrics": m_test, "conf": conf_test, "macro": test_macro}

                print(f"[TEST][{Path(ck).stem}] "
                      f"acc={m_test['acc']:.4f}  qwk={m_test['qwk']:.4f}  within1={m_test['within1']:.4f}  mae={m_test['mae']:.4f}\n"
                      f"     BA(macro recall)={test_macro['balanced_acc']:.4f}  macroF1={test_macro['macro_f1']:.4f}")
                
            # NEW: challenge split
            if "challenge" in splits and len(challenge_list) > 0:
                csv_path = os.path.join(out_root, f"{case}__{Path(ck).stem}__challenge.csv") if args.save_details else None
                m_chal, conf_chal, _ = eval_collect(
                    model, challenge_list, num_class_total, "challenge",
                    csv_path, args.print_details, final_stage_cfg
                )
                chal_macro = compute_macro_from_conf(conf_chal)
                per_ckpt_metrics[ck]["challenge"] = {"metrics": m_chal, "conf": conf_chal, "macro": chal_macro}

                print(f"[CHALLENGE][{Path(ck).stem}] "
                      f"acc={m_chal['acc']:.4f}  qwk={m_chal['qwk']:.4f}  within1={m_chal['within1']:.4f}  mae={m_chal['mae']:.4f}\n"
                      f"           BA(macro recall)={chal_macro['balanced_acc']:.4f}  macroF1={chal_macro['macro_f1']:.4f}")

        # Aggregate across ckpts for this case/split
        def _agg(split: str):
            keys = ["acc","qwk","within1","mae","BA","macroF1"]
            vals = {k: [] for k in keys}
            conf_sum = None; n = 0
            for ck, rec in per_ckpt_metrics.items():
                if split not in rec: continue
                m = rec[split]["metrics"]; macro = rec[split]["macro"]
                vals["acc"].append(float(m["acc"]))
                vals["qwk"].append(float(m["qwk"]))
                vals["within1"].append(float(m["within1"]))
                vals["mae"].append(float(m["mae"]))
                vals["BA"].append(float(macro["balanced_acc"]))
                vals["macroF1"].append(float(macro["macro_f1"]))
                conf = rec[split]["conf"]
                conf_sum = conf.astype(np.float64) if conf_sum is None else (conf_sum + conf)
                n += 1
            if n == 0: return None
            mean = {k: float(np.mean(vals[k])) for k in keys}
            std  = {k: float(np.std(vals[k], ddof=0)) for k in keys}
            avg_conf = conf_sum / n if conf_sum is not None else None
            return {"mean": mean, "std": std, "avg_conf": avg_conf, "n": n}

        summary = {}
        for sp in splits:
            ag = _agg(sp)
            if ag is None: continue
            print(f"[{sp.upper()}][{case}] mean over {ag['n']} ckpts: " +
                  "  ".join([f"{k}={ag['mean'][k]:.4f}±{ag['std'][k]:.4f}" for k in ["acc","qwk","within1","mae","BA","macroF1"]]))
            summary[sp] = {
                "n_ckpts": ag["n"],
                "mean": ag["mean"],
                "std": ag["std"],
                "avg_conf": ag["avg_conf"].tolist() if ag["avg_conf"] is not None else None
            }

            # NEW: plot confusion matrix using avg_conf
            if ag["avg_conf"] is not None:
                cm_png = os.path.join(out_root, f"{case}__{sp}_avg_confusion.png")
                title = f"{case} | {sp} (avg over {ag['n']} ckpts)"
                # 你可以把 normalize=False 改成 True，看 row-normalized 矩阵
                plot_confusion_matrix(ag["avg_conf"], cm_png, title=title, normalize=True)
                print(f"[{sp.upper()}][{case}] saved confusion matrix to: {cm_png}")

        # Save case-level summary JSON
        case_sum_path = os.path.join(out_root, f"{case}__summary.json")
        with open(case_sum_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Save per-ckpt raw metrics (including macro)
        raw_path = os.path.join(out_root, f"{case}__per_ckpt_metrics.json")
        serializable = {}
        for ck, rec in per_ckpt_metrics.items():
            serializable[Path(ck).name] = {}
            for sp, rr in rec.items():
                serializable[Path(ck).name][sp] = {
                    "metrics": rr["metrics"],
                    "macro": rr["macro"],
                    "conf": rr["conf"].tolist()
                }
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

    print("\nDone.\n")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    main()

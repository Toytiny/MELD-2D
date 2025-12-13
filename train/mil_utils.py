#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, io, platform, shlex
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import json
import numpy as np
import torch
import torch.nn.functional as F
import re

# -------------------------
# CLI / misc utils
# -------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise ValueError("Boolean value expected.")

def seed_worker(worker_id):
    """Set per-worker RNG seeds for dataloader workers (reproducibility)."""
    import random
    import numpy as np
    import torch
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def log_args(args, save_json_path: str | None = None, title="ARGS"):
    """Pretty-print args to stdout and optionally dump to JSON."""
    d = vars(args)
    print("\n" + "="*20 + f" {title} " + "="*20)
    for k in sorted(d):
        print(f"{k}: {d[k]}")
    print("=" * (42 + len(title)) + "\n")
    if save_json_path:
        Path(save_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)

def print_env_brief():
    """Print a compact environment summary."""
    cuda = torch.cuda.is_available()
    num = torch.cuda.device_count() if cuda else 0
    dev = torch.cuda.get_device_name(0) if (cuda and num > 0) else "CPU"
    print("\n========== ENV ==========")
    print(f"python: {platform.python_version()}")
    print(f"platform: {platform.platform()}")
    print(f"cuda_available: {cuda} | num_gpus: {num} | device0: {dev}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES','')}")
    print("=========================\n")

def print_repro_cmd():
    """Echo the exact CLI command to reproduce this run."""
    cmd = " ".join(shlex.quote(x) for x in sys.argv)
    print("\n" + "="*20 + " REPRO CMD " + "="*20)
    print(cmd)
    print("="*42 + "\n")

def unwrap_state_dict(state_dict):  # strip "module."
    return {k.replace("module.", ""): v for k, v in state_dict.items()}

def load_unwrapped_state(model, state_dict, strict=True):
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(state_dict, strict=strict)

def read_lines(fp: str) -> list[str]:
    return open(fp, "r", encoding="utf-8").read().splitlines()

def score_from_metrics(metrics: dict, metric_name: str) -> float:
    """Return scalar to maximize; for 'mae' we minimize (flip sign)."""
    if metric_name == "mae":
        return -float(metrics["mae"])
    return float(metrics[metric_name])

# -------------------------
# Metrics
# -------------------------
def _qwk(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    C = num_classes
    O = np.zeros((C, C), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < C and 0 <= p < C:
            O[t, p] += 1.0
    t_hist = O.sum(axis=1); p_hist = O.sum(axis=0)
    N = O.sum()
    if N == 0:
        return 0.0
    E = np.outer(t_hist, p_hist) / max(N, 1.0)
    W = np.zeros((C, C), dtype=np.float64)
    for i in range(C):
        for j in range(C):
            W[i, j] = ((i - j) ** 2) / ((C - 1) ** 2)
    num = (W * O).sum(); den = (W * E).sum()
    return 1.0 - (num / den) if den > 0 else 0.0

def compute_macro_from_conf(conf: np.ndarray):
    """
    Input: conf [C,C], rows=GT, cols=Pred
    Returns per-class precision/recall/F1 + Balanced Acc (macro recall) + Macro-F1.
    """
    conf = conf.astype(np.float64)
    eps = 1e-12
    support = conf.sum(axis=1)          # GT per class
    pred_sum = conf.sum(axis=0)         # Pred per class
    tp = np.diag(conf)

    per_class_recall = tp / np.maximum(support, eps)
    per_class_precision = tp / np.maximum(pred_sum, eps)
    per_class_f1 = 2 * per_class_precision * per_class_recall / np.maximum(per_class_precision + per_class_recall, eps)

    balanced_acc = float(np.nanmean(per_class_recall))
    macro_f1 = float(np.nanmean(per_class_f1))

    row_norm = conf / np.maximum(support[:, None], eps)  # row-normalized confusion

    return {
        "per_class_recall": per_class_recall.tolist(),
        "per_class_precision": per_class_precision.tolist(),
        "per_class_f1": per_class_f1.tolist(),
        "balanced_acc": balanced_acc,
        "macro_f1": macro_f1,
        "conf_row_norm": row_norm.tolist(),
    }

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    """Return dict of metrics and a confusion matrix."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    C = num_classes
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    qwk = float(_qwk(y_true, y_pred, C))
    within1 = float((np.abs(y_pred - y_true) <= 1).mean()) if len(y_true) else 0.0
    mae = float(np.abs(y_pred - y_true).mean()) if len(y_true) else 0.0
    conf = np.zeros((C, C), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < C and 0 <= p < C:
            conf[t, p] += 1
    return {"acc": acc, "qwk": qwk, "within1": within1, "mae": mae}, conf

# -------------------------
# MIL: weighting from inputs (e.g., keypoint confidence)
# -------------------------
def build_window_weights_from_X(X, M, conf_thr: float = 0.0, sharpen_alpha: float = 1.0, eps: float = 1e-8):
    """
    Build per-window weights Q from input tensor X using the 3rd channel as confidence if present.
    X: [B,W,T,V,C]; M: [B,W] (bool); returns Q: [B,W] that sums to 1 per bag (masked)
    """
    X = X.detach()
    has_conf = X.size(-1) >= 3
    if has_conf:
        Cconf = X[..., 2]
        if conf_thr > 0.0:
            mask_conf = (Cconf >= conf_thr).to(X.dtype)
            denom = torch.clamp(mask_conf.sum(dim=(2, 3, 4)), min=1.0)
            q = (Cconf * mask_conf).sum(dim=(2, 3, 4)) / denom
        else:
            q = Cconf.mean(dim=(2, 3, 4))
    else:
        q = torch.ones(X.size(0), X.size(1), device=X.device, dtype=X.dtype)
    if sharpen_alpha != 1.0:
        q = torch.pow(q.clamp(min=0.0), sharpen_alpha)
    q = q * M.to(q.dtype)
    denom = torch.clamp(q.sum(dim=1, keepdim=True), min=eps)
    Q = q / denom
    return Q

# -------------------------
# MIL pooling ops
# -------------------------
def mil_pool_lse_weighted(logits_bw: torch.Tensor, mask_bw: torch.Tensor, weights_bw: torch.Tensor,
                          tau: float = 5.0, eps: float = 1e-8) -> torch.Tensor:
    """Weighted LogSumExp pooling with per-window weights."""
    neg_inf = torch.finfo(logits_bw.dtype).min
    masked = logits_bw.masked_fill(~mask_bw[..., None], neg_inf)
    w = torch.clamp(weights_bw, min=eps)[..., None]
    z = (masked / tau) + torch.log(w)
    m, _ = torch.max(z, dim=1, keepdim=True)
    lse = m.squeeze(1) + torch.log(torch.clamp_min(torch.sum(torch.exp(z - m), dim=1), eps))
    return lse

def mil_pool_lse(logits_bw: torch.Tensor, mask_bw: torch.Tensor, tau: float = 5.0):
    """Plain LogSumExp across instances (optionally averaged upstream if desired)."""
    neg_inf = torch.finfo(logits_bw.dtype).min
    masked = logits_bw.masked_fill(~mask_bw[..., None], neg_inf)
    z = (masked / tau)
    m, _ = torch.max(z, dim=1, keepdim=True)
    lse = m.squeeze(1) + torch.log(torch.clamp_min(torch.sum(torch.exp(z - m), dim=1), 1e-12))
    # NOTE: intentionally not dividing by valid_counts to avoid diluting rare decisive clips.
    return lse

def mil_pool_topk_lse(logits_bw: torch.Tensor, mask_bw: torch.Tensor, k: int = 1, tau: float = 5.0):
    """Per-class top-k LogSumExp over instance dimension."""
    B, T, C = logits_bw.shape
    neg_inf = torch.finfo(logits_bw.dtype).min
    z = logits_bw.masked_fill(~mask_bw[..., None], neg_inf) / tau
    topk, _ = z.topk(k=min(k, T), dim=1)                        # [B,k,C]
    m, _ = topk.max(dim=1, keepdim=True)                        # [B,1,C]
    bag_logits = m.squeeze(1) + torch.log(torch.clamp_min((topk - m).exp().sum(1), 1e-12))
    return bag_logits

def mil_pool_topk(logits_bw: torch.Tensor, mask_bw: torch.Tensor, k_frac: float = 0.1):
    """Per-class top-k MEAN pooling (linear); use only if you really need mean behavior."""
    B, W, C = logits_bw.shape
    k = torch.clamp((mask_bw.sum(1).float() * k_frac).ceil().long(), min=1)
    video_logits = []
    for b in range(B):
        valid = logits_bw[b][mask_bw[b]]
        if valid.numel() == 0:
            video_logits.append(torch.zeros(C, device=logits_bw.device))
            continue
        topk_vals, _ = torch.topk(valid, k[b].item(), dim=0)
        video_logits.append(topk_vals.mean(0))
    return torch.stack(video_logits, dim=0)

def emd2_loss(logits, y):
    """Squared EMD on class CDFs (ordered labels)."""
    p = torch.softmax(logits, dim=1)           # [B,C]
    cdf_p = torch.cumsum(p, dim=1)             # [B,C]
    onehot = F.one_hot(y, num_classes=p.size(1)).float()
    cdf_t = torch.cumsum(onehot, dim=1)
    return F.mse_loss(cdf_p, cdf_t)

# -------------------------
# Pooling schedule & dispatcher
# -------------------------
def choose_pooling_for_epoch(epoch: int, pool_mode: str, args):
    """
    Return a dict {'mode','tau','k'} for the current epoch.
    If not staged: use --pool-mode / --tau / --k-topk directly.
    If staged:
        [0, stage1)            -> LSE
        [stage1, stage1+stage2)-> topk_lse(k=stage2_k)
        [else]                 -> topk_lse(k=stage3_k)
    """
    if pool_mode != "staged":
        cfg = {"mode": pool_mode, "tau": args.tau, "k": None}
        if pool_mode in ("topk_lse", "topk"):
            cfg["k"] = max(1, int(getattr(args, "k_topk", 1)))
        return cfg

    s1 = max(0, int(args.stage1_epochs))
    s2 = max(0, int(args.stage2_epochs))
    if epoch < s1:
        return {"mode": "lse", "tau": float(args.stage1_tau), "k": None}
    elif epoch < s1 + s2:
        return {"mode": "topk_lse", "tau": float(args.stage2_tau), "k": max(1, int(args.stage2_k))}
    else:
        return {"mode": "topk_lse", "tau": float(args.stage3_tau), "k": max(1, int(args.stage3_k))}

def apply_mil_pool(logits_bw, mask_bw, *, mode: str, tau: float, k: Optional[int], weights_bw=None):
    """Unified MIL pooling entry."""
    if weights_bw is not None:
        return mil_pool_lse_weighted(logits_bw, mask_bw, weights_bw, tau=tau)
    if mode == "lse":
        return mil_pool_lse(logits_bw, mask_bw, tau=tau)
    elif mode == "topk_lse":
        k_eff = 1 if (k is None) else max(1, int(k))
        return mil_pool_topk_lse(logits_bw, mask_bw, k=k_eff, tau=tau)
    elif mode == "topk":
        k_frac = 0.1 if k is None else (k / logits_bw.size(1))
        return mil_pool_topk(logits_bw, mask_bw, k_frac=k_frac)
    elif mode == "noisy_or_ord":
        # temperature == tau for a unified interface
        return mil_pool_noisyor_ordinal_from_logits(logits_bw, mask_bw, temperature=max(tau, 1e-6))
    else:
        raise ValueError(f"Unknown pool mode: {mode}")

# mil_utils.py
def apply_prior_bias_linear(logits, lam=0.3):
    """
    logits: [B,C], class index 0=lowest level
    Return: biased logits
    """
    B, C = logits.shape
    idx = torch.arange(C, device=logits.device, dtype=logits.dtype)  # [C]
    bias = -lam * idx                                               # [C]
    return logits + bias

# mil_utils.py
def apply_human_prior(video_logits, mode="none", **kw):
    if mode == "none":
        return video_logits
    if mode == "linear":
        return apply_prior_bias_linear(video_logits, lam=kw.get("lam", 0.3))
    if mode == "bayes":
        return apply_prior_bias_bayes(video_logits, priors=kw.get("priors"), alpha=kw.get("alpha", 0.5))
    raise ValueError(f"Unknown prior mode {mode}")

# mil_utils.py
def apply_prior_bias_bayes(logits, priors=None, alpha=0.5):
    """
    priors: tensor/list len=C (概率), 若为 None 用 exp(-alpha*c) 生成
    """
    B, C = logits.shape
    if priors is None:
        c = torch.arange(C, device=logits.device, dtype=logits.dtype)
        pi = torch.softmax(-alpha * c, dim=0)        # 低级概率更高
    else:
        pi_np = np.asarray(priors, dtype=np.float64)
        pi = torch.tensor(pi_np / np.clip(pi_np.sum(), 1e-12, None),
                          device=logits.device, dtype=logits.dtype)
    log_pi = torch.log(torch.clamp(pi, 1e-8, 1.0))
    return logits + log_pi

# mil_utils.py
def argmin_cost_decoding(logits, over_penalty=2.0):
    """
    logits: [B,C] -> probs -> choose class minimizing expected cost.
    over_penalty>1 让“高估”成本更高（偏向低级）。
    """
    B, C = logits.shape
    p = torch.softmax(logits, dim=1)        # [B,C]
    # 构建代价矩阵 C[k,c]
    i = torch.arange(C, device=logits.device).unsqueeze(1)
    j = torch.arange(C, device=logits.device).unsqueeze(0)
    dist = (i - j).abs().float()            # |k-c|
    over = (i > j).float()                  # 预测>真实
    cost = dist * (1.0 + (over_penalty-1.0)*over)
    # 期望代价: [B,C]
    exp_cost = p @ cost.T
    return exp_cost.argmin(dim=1)

# ---------- Noisy-OR (Ordinal) MIL Pooling ----------
# Rationale:
# - We want the bag-level score to go LOWER if ANY window provides strong evidence for a lower class.
# - Convert per-window class probabilities to an ordinal CDF P(Y <= k), then fuse across windows via Noisy-OR:
#     P_bag(Y <= k) = 1 - Π_w (1 - P_w(Y <= k))
# - Recover PMF from the bag-level CDF and return log-probabilities as bag "logits".
# - Fully differentiable; works with CE/EMD and your human priors.


import re
import os

def read_challenge_file(path: str):
    """
    Read test_challenge.txt which has 4 clinician scores per line.
    Format example:
        001_1001=02-DEC-2019+++1 2 2 2

    Steps:
      1) Parse all integers after '+++'
      2) Use the LAST one as the final teleconference/resolution score
      3) Merge Level 5 and 6 => map {5,6} -> 5
      4) Return a list of 'vid+++label' strings for standard MIL loader
    """
    lines = []
    if not os.path.isfile(path):
        print(f"[WARN] challenge file not found: {path}")
        return lines

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or "+++" not in line:
                continue

            vid, tail = line.split("+++", 1)
            vid = vid.strip()
            if not vid:
                continue

            # 提取所有整数分数
            nums = re.findall(r"-?\d+", tail)
            if not nums:
                print(f"[WARN] no scores found in challenge line: {raw.strip()}")
                continue

            try:
                final_label = int(nums[-1])      # 用最后一个（Resolution / Teleconference）
            except ValueError:
                print(f"[WARN] cannot parse final score in: {raw.strip()}")
                continue

            # === 合并 Level 5 和 6 ===
            if final_label >= 5:
                final_label = 5

            lines.append(f"{vid}+++{final_label}")

    return lines

def _mask_windows_probs(p_bw: torch.Tensor, mask_bw: torch.Tensor) -> torch.Tensor:
    """
    Zero out invalid windows so they act as neutral elements in Noisy-OR.
    p_bw:   [B, T, C] per-window class probabilities
    mask_bw:[B, T]    valid-window mask (True for valid)
    """
    return p_bw * mask_bw.unsqueeze(-1).to(p_bw.dtype)

def mil_pool_noisyor_ordinal_from_logits(
    logits_bw: torch.Tensor, mask_bw: torch.Tensor, temperature: float = 1.0, eps: float = 1e-8
) -> torch.Tensor:
    """
    Ordinal Noisy-OR pooling for MIL.

    Args:
        logits_bw:   [B, T, C] per-window logits
        mask_bw:     [B, T]    valid-window mask
        temperature: temperature scaling before softmax (smaller -> sharper evidence)
        eps:         numerical stability

    Returns:
        bag_logprobs: [B, C] bag-level log-probabilities (sum to 1 in prob space)
    """
    # 1) Softmax with temperature -> per-window class probabilities
    p_bw = F.softmax(logits_bw / max(temperature, eps), dim=-1)      # [B,T,C]
    p_bw = _mask_windows_probs(p_bw, mask_bw)                        # [B,T,C]

    # 2) Per-window ordinal CDF: P(Y <= k)
    cdf_bw = torch.cumsum(p_bw, dim=-1).clamp_(0.0, 1.0)             # [B,T,C]

    # 3) Noisy-OR across windows: 1 - Π_w (1 - CDF_w)
    one_minus = (1.0 - cdf_bw).clamp_min(0.0)                        # [B,T,C]
    prod_term = torch.prod(one_minus + eps, dim=1)                   # [B,C]
    cdf_bag = (1.0 - prod_term).clamp_(0.0, 1.0)                     # [B,C]

    # 4) Recover PMF from CDF
    pmf_bag = torch.empty_like(cdf_bag)
    pmf_bag[:, 0] = cdf_bag[:, 0]
    pmf_bag[:, 1:] = (cdf_bag[:, 1:] - cdf_bag[:, :-1]).clamp_min(0.0)

    # 5) Normalize and convert to log-probabilities ("logits" for CE/EMD)
    pmf_bag = (pmf_bag + eps) / (pmf_bag.sum(dim=1, keepdim=True) + eps)
    return torch.log(pmf_bag)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clip-supervised Eval (visit-level):
- For clip-supervised ST-GCN + CE training (no MIL).
- Loads training config from results/<exp>/args.json to match:
    - root_dir_clips, split_root, setup_list
    - soft_voting, test_batch, backbone config
- For each setup (case) and each checkpoint (e.g. bestEMA_*),
  runs visit-level evaluation on VAL / TEST / CHALLENGE splits:
    - Visit-level aggregation by soft-voting (default) or hard voting.
    - Metrics: acc, qwk, within1, mae, BA (macro recall), macroF1.
- Aggregates metrics across ckpts (mean ± std) and saves:
    - <case>__summary.json   (per-split aggregated stats)
    - <case>__per_ckpt_metrics.json (per-ckpt raw metrics & confusion)
- Optionally saves per-visit CSVs with probs & #clips.
"""

import os, sys, json, argparse, platform
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==== project paths ====
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

# ==== project modules ====
from models.stgcn import STGCN
from models.heads import Classifier
from utils.data_processing import Preprocess_Module

from mil_utils import (
    str2bool, read_lines, read_challenge_file,
    compute_metrics, compute_macro_from_conf
)

# -------------------------
# argparse
# -------------------------
parser = argparse.ArgumentParser("Evaluate clip-supervised checkpoints (visit-level)")

parser.add_argument("--exp-name", type=str, required=True,
                    help="Experiment name under results/ (same as training)")
parser.add_argument("--gpus", type=str, default="0,1",
                    help='CUDA_VISIBLE_DEVICES, e.g. "0" or "0,1"')
parser.add_argument("--primary", type=int, default=0)

parser.add_argument(
    "--splits", type=str, default="test",
    help='Comma-separated: "val,test", "test", "val", "challenge", or combinations'
)

parser.add_argument("--ckpt-pattern", type=str, default="bestEMA_*.pth",
                    help="Glob for checkpoints in results/<exp>/")

parser.add_argument("--save-details", type=str2bool, default=True,
                    help="Save per-visit predictions to CSV")
parser.add_argument("--print-details", type=str2bool, default=False,
                    help="Print per-visit predictions to stdout")

# Optional overrides (otherwise from args.json)
parser.add_argument("--override-soft-voting", type=str2bool, default=None,
                    help="Override soft_voting (default: use training args.json)")
parser.add_argument("--override-test-batch", type=int, default=None,
                    help="Override test_batch (default: use training args.json)")

args = parser.parse_args()

# -------------------------
# env & load training config
# -------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = "cuda" if torch.cuda.is_available() else "cpu"

if os.path.isabs(args.exp_name):
    results_dir = args.exp_name
else:
    results_dir = os.path.join(SCRIPT_DIR, "results", args.exp_name)

args_json = os.path.join(results_dir, "args.json")
assert os.path.isfile(args_json), f"args.json not found: {args_json}"

with open(args_json, "r", encoding="utf-8") as f:
    train_cfg: Dict[str, Any] = json.load(f)

def _get(k, default=None): return train_cfg.get(k, default)

# ---- paths / setups ----
root_dir_clips = _get("root_dir_clips")
split_root     = _get("split_root")
setup_list_str = _get("setup_list", "original_setup")
setup_list     = [s.strip() for s in setup_list_str.split(",") if s.strip()]

assert root_dir_clips is not None, "root_dir_clips must be in args.json"
assert split_root     is not None, "split_root must be in args.json"

# ---- voting / batch size ----
soft_voting_cfg = bool(_get("soft_voting", True))
test_batch_cfg  = int(_get("test_batch", 256))

SOFT_VOTING = soft_voting_cfg if args.override_soft_voting is None else bool(args.override_soft_voting)
TEST_BS     = test_batch_cfg  if args.override_test_batch  is None else int(args.override_test_batch)

# ---- backbone config (same as training) ----
backbone = _get("backbone", "stgcn")
assert backbone == "stgcn", "Current eval only supports backbone=stgcn"

backbone_cfg = {
    "type": "STGCN",
    "gcn_adaptive": "init",
    "gcn_with_res": True,
    "tcn_type": "mstcn",
    "graph_cfg": {"layout": "coco", "mode": "spatial"},
    "pretrained": None,
}

# -------------------------
# Model wrapper
# -------------------------
class STGCN_Classifier(nn.Module):
    """Clip-level classifier: ST-GCN backbone + linear head."""
    def __init__(self, backbone_cfg, num_classes: int):
        super().__init__()
        args_local = dict(backbone_cfg); args_local.pop("type", None)
        self.backbone = STGCN(**args_local)
        self.cls_head = Classifier(num_classes=num_classes, dropout=0.5, latent_dim=512)
    def forward(self, keypoint):   # keypoint: [B,1,T,V,2]
        feat = self.backbone(keypoint)   # [B,C,T',V]
        logits = self.cls_head(feat)     # [B,num_classes]
        return logits

def safe_load_ckpt_to_model(ckpt_path: str, model: nn.Module):
    """Load unwrapped state_dict (saved from training) to a single-GPU model."""
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing or unexpected:
        print(f"[LOAD][WARN] missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("  missing(sample):", missing[:8])
        if unexpected:
            print("  unexpected(sample):", unexpected[:8])

# -------------------------
# Visit-level evaluation utils
# -------------------------
def clip_conf_mean(path: str) -> float:
    """Mean keypoint confidence per clip (used as soft-voting weight)."""
    arr = np.load(path)  # (T,17,3)
    if arr.size == 0:
        return 0.0
    a = np.nan_to_num(arr[..., 2], nan=0.0)
    return max(float(a.mean()), 0.0)

@torch.no_grad()
def eval_visit_split(model: nn.Module,
                     split_lines: List[str],
                     split_name: str,
                     root_dir: str,
                     num_classes: int,
                     soft_voting: bool,
                     test_bs: int,
                     save_csv_path: Optional[str],
                     print_details: bool):
    """
    Evaluate visit-level metrics for a given split.
    Each line in split_lines: "<visit_rel>+++<label_int (1..C)>"
    """
    model.eval()
    transform = Preprocess_Module(data_augmentation=False)

    all_true, all_pred = [], []
    details = []

    for line in split_lines:
        key, lab = line.split("+++", 1)
        visit_rel = key.strip()
        visit = visit_rel.split("/")[-1]
        gt = int(lab.strip()) - 1  # to 0-based

        visit_dir = os.path.join(root_dir, visit)
        if not os.path.isdir(visit_dir):
            print(f"[{split_name}][WARN] missing visit_dir: {visit_dir}")
            continue

        clip_files = sorted(
            [os.path.join(visit_dir, f) for f in os.listdir(visit_dir) if f.endswith(".npy")]
        )
        if not clip_files:
            print(f"[{split_name}][WARN] no clips in: {visit_dir}")
            continue

        n_clips = len(clip_files)
        hard_preds: List[int] = []

        if soft_voting:
            weighted_sum = np.zeros((num_classes,), dtype=np.float64)
            w_total = 0.0

        # Batched clip inference
        for i in range(0, n_clips, test_bs):
            batch_paths = clip_files[i:i+test_bs]

            xs = []
            for p in batch_paths:
                data = np.load(p)           # (T,17,3)
                data[np.isnan(data)] = 0
                tmp = {
                    'img_shape': (800, 1422),
                    'label': -1,
                    'start_index': 0,
                    'modality': 'Pose',
                    'total_frames': data.shape[0],
                    'keypoint': np.tile(data[np.newaxis, :, :, :2], (2, 1, 1, 1)),
                    'keypoint_score': np.tile(data[np.newaxis, :, :, 2], (2, 1, 1)),
                }
                out = transform(tmp)['keypoint'][0].numpy()  # [1,T,V,2]
                xs.append(out)

            batch = torch.from_numpy(np.stack(xs, 0)).float().to(device)  # [B,1,T,V,2]
            outputs = model(batch)  # [B, C]

            if soft_voting:
                probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()  # (B, C)
                w = np.array([clip_conf_mean(p) for p in batch_paths], dtype=np.float64)  # (B,)
                w = np.maximum(w, 0.0)
                weighted_sum += (probs * w[:, None]).sum(axis=0)
                w_total += float(w.sum())
            else:
                hard_preds.extend(outputs.argmax(dim=1).detach().cpu().tolist())

        if soft_voting:
            if w_total > 0:
                agg_vec = weighted_sum / w_total
            else:
                # fallback: uniform average if all conf=0
                agg_vec = weighted_sum
            probs_visit = agg_vec
            pred = int(np.argmax(agg_vec))
        else:
            if len(hard_preds) == 0:
                print(f"[{split_name}][WARN] no predictions for visit: {visit_dir}")
                continue
            counts = np.bincount(hard_preds, minlength=num_classes).astype(np.float64)
            probs_visit = counts / max(counts.sum(), 1.0)
            pred = int(counts.argmax())

        ok = int(pred == gt)

        if print_details:
            print(f"[{split_name}] {visit} | gt={gt} pred={pred} ok={ok} "
                  f"probs={np.round(probs_visit, 4)} #clips={n_clips}")

        details.append({
            "split": split_name,
            "visit_id": visit,
            "gt": gt,
            "pred": pred,
            "ok": ok,
            "probs": probs_visit.tolist(),
            "n_clips": n_clips,
        })
        all_true.append(gt)
        all_pred.append(pred)

    y_true = np.array(all_true, dtype=np.int64)
    y_pred = np.array(all_pred, dtype=np.int64)

    if y_true.size == 0:
        # nothing evaluated
        metrics = {"acc": 0.0, "qwk": 0.0, "within1": 0.0, "mae": 0.0}
        conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    else:
        metrics, conf = compute_metrics(y_true, y_pred, num_classes)

    # Save CSV
    if save_csv_path:
        import csv
        p = Path(save_csv_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["split", "visit_id", "gt", "pred", "ok", "probs", "n_clips"]
            )
            w.writeheader()
            for d in details:
                w.writerow(d)

    return metrics, conf, details

# -------------------------
# Main
# -------------------------
def main():
    print("========== ENV ==========")
    devname = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"python: {platform.python_version()} | cuda: {torch.cuda.is_available()} | device: {devname}")
    print("=========================\n")

    print(f"[EVAL] results_dir={results_dir}")
    print(f"[EVAL] root_dir_clips={root_dir_clips}")
    print(f"[EVAL] split_root={split_root}")
    print(f"[EVAL] setups={setup_list}")
    print(f"[EVAL] SOFT_VOTING={SOFT_VOTING} | TEST_BS={TEST_BS}")

    # Find checkpoints
    ckpts = sorted([str(p) for p in Path(results_dir).glob(args.ckpt_pattern)])
    assert len(ckpts) > 0, f"No checkpoints matched in {results_dir} with pattern {args.ckpt_pattern}"
    print(f"Found {len(ckpts)} ckpts:")
    for p in ckpts:
        print(" -", os.path.basename(p))

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    assert all(s in ("val", "test", "challenge") for s in splits), "splits must be val/test/challenge"

    out_root = os.path.join(results_dir, "eval_clip_outputs")
    Path(out_root).mkdir(exist_ok=True, parents=True)

    # Evaluate each setup independently
    for case in setup_list:
        num_class_total = 4 if case == "scores_1to4_only" else 5
        split_dir = os.path.join(split_root, case)

        val_list  = read_lines(os.path.join(split_dir, "val_majority.txt"))
        test_list = read_lines(os.path.join(split_dir, "test_majority.txt"))

        challenge_txt  = os.path.join(split_dir, "test_challenge.txt")
        challenge_list = read_challenge_file(challenge_txt) if os.path.isfile(challenge_txt) else []

        print(f"\n===== CASE: {case} | C={num_class_total} =====")

        per_ckpt_metrics: Dict[str, Dict[str, Any]] = {}

        for ck in ckpts:
            print(f"\n[LOAD] ckpt: {os.path.basename(ck)}")

            model_single = STGCN_Classifier(backbone_cfg, num_classes=num_class_total).to(device)
            safe_load_ckpt_to_model(ck, model_single)

            # Optional: simple sanity check on head weights
            try:
                print("  Wnorm(cls_head.fc):",
                      float(model_single.cls_head.fc.weight.data.norm().item()))
            except Exception:
                pass

            model = model_single
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)

            per_ckpt_metrics[ck] = {}

            # VAL
            if "val" in splits and len(val_list) > 0:
                csv_path = (os.path.join(out_root, f"{case}__{Path(ck).stem}__val.csv")
                            if args.save_details else None)
                m_val, conf_val, _ = eval_visit_split(
                    model=model,
                    split_lines=val_list,
                    split_name="val",
                    root_dir=root_dir_clips,
                    num_classes=num_class_total,
                    soft_voting=SOFT_VOTING,
                    test_bs=TEST_BS,
                    save_csv_path=csv_path,
                    print_details=args.print_details,
                )
                val_macro = compute_macro_from_conf(conf_val)
                per_ckpt_metrics[ck]["val"] = {
                    "metrics": m_val,
                    "conf": conf_val,
                    "macro": val_macro,
                }

                print(f"[VAL][{Path(ck).stem}] "
                      f"acc={m_val['acc']:.4f}  qwk={m_val['qwk']:.4f}  "
                      f"within1={m_val['within1']:.4f}  mae={m_val['mae']:.4f}\n"
                      f"    BA(macro recall)={val_macro['balanced_acc']:.4f}  "
                      f"macroF1={val_macro['macro_f1']:.4f}")

            # TEST
            if "test" in splits and len(test_list) > 0:
                csv_path = (os.path.join(out_root, f"{case}__{Path(ck).stem}__test.csv")
                            if args.save_details else None)
                m_test, conf_test, _ = eval_visit_split(
                    model=model,
                    split_lines=test_list,
                    split_name="test",
                    root_dir=root_dir_clips,
                    num_classes=num_class_total,
                    soft_voting=SOFT_VOTING,
                    test_bs=TEST_BS,
                    save_csv_path=csv_path,
                    print_details=args.print_details,
                )
                test_macro = compute_macro_from_conf(conf_test)
                per_ckpt_metrics[ck]["test"] = {
                    "metrics": m_test,
                    "conf": conf_test,
                    "macro": test_macro,
                }

                print(f"[TEST][{Path(ck).stem}] "
                      f"acc={m_test['acc']:.4f}  qwk={m_test['qwk']:.4f}  "
                      f"within1={m_test['within1']:.4f}  mae={m_test['mae']:.4f}\n"
                      f"     BA(macro recall)={test_macro['balanced_acc']:.4f}  "
                      f"macroF1={test_macro['macro_f1']:.4f}")

            # CHALLENGE
            if "challenge" in splits and len(challenge_list) > 0:
                csv_path = (os.path.join(out_root, f"{case}__{Path(ck).stem}__challenge.csv")
                            if args.save_details else None)
                m_chal, conf_chal, _ = eval_visit_split(
                    model=model,
                    split_lines=challenge_list,
                    split_name="challenge",
                    root_dir=root_dir_clips,
                    num_classes=num_class_total,
                    soft_voting=SOFT_VOTING,
                    test_bs=TEST_BS,
                    save_csv_path=csv_path,
                    print_details=args.print_details,
                )
                chal_macro = compute_macro_from_conf(conf_chal)
                per_ckpt_metrics[ck]["challenge"] = {
                    "metrics": m_chal,
                    "conf": conf_chal,
                    "macro": chal_macro,
                }

                print(f"[CHALLENGE][{Path(ck).stem}] "
                      f"acc={m_chal['acc']:.4f}  qwk={m_chal['qwk']:.4f}  "
                      f"within1={m_chal['within1']:.4f}  mae={m_chal['mae']:.4f}\n"
                      f"           BA(macro recall)={chal_macro['balanced_acc']:.4f}  "
                      f"macroF1={chal_macro['macro_f1']:.4f}")

        # ---- aggregate across ckpts ----
        def _agg(split: str):
            keys = ["acc", "qwk", "within1", "mae", "BA", "macroF1"]
            vals = {k: [] for k in keys}
            conf_sum = None
            n = 0
            for ck, rec in per_ckpt_metrics.items():
                if split not in rec:
                    continue
                m = rec[split]["metrics"]
                macro = rec[split]["macro"]
                vals["acc"].append(float(m["acc"]))
                vals["qwk"].append(float(m["qwk"]))
                vals["within1"].append(float(m["within1"]))
                vals["mae"].append(float(m["mae"]))
                vals["BA"].append(float(macro["balanced_acc"]))
                vals["macroF1"].append(float(macro["macro_f1"]))
                conf = rec[split]["conf"]
                conf_sum = conf.astype(np.float64) if conf_sum is None else (conf_sum + conf)
                n += 1
            if n == 0:
                return None
            mean = {k: float(np.mean(vals[k])) for k in keys}
            std  = {k: float(np.std(vals[k], ddof=0)) for k in keys}
            avg_conf = conf_sum / n if conf_sum is not None else None
            return {"mean": mean, "std": std, "avg_conf": avg_conf, "n": n}

        summary: Dict[str, Any] = {}
        for sp in splits:
            ag = _agg(sp)
            if ag is None:
                continue
            print(f"[{sp.upper()}][{case}] mean over {ag['n']} ckpts: " +
                  "  ".join([f"{k}={ag['mean'][k]:.4f}±{ag['std'][k]:.4f}"
                            for k in ["acc", "qwk", "within1", "mae", "BA", "macroF1"]]))
            summary[sp] = {
                "n_ckpts": ag["n"],
                "mean": ag["mean"],
                "std": ag["std"],
                "avg_conf": ag["avg_conf"].tolist() if ag["avg_conf"] is not None else None,
            }

        # Save case-level summary JSON
        case_sum_path = os.path.join(out_root, f"{case}__summary.json")
        with open(case_sum_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Save per-ckpt raw metrics (including macro + confusion)
        raw_path = os.path.join(out_root, f"{case}__per_ckpt_metrics.json")
        serializable: Dict[str, Any] = {}
        for ck, rec in per_ckpt_metrics.items():
            ck_name = Path(ck).name
            serializable[ck_name] = {}
            for sp, rr in rec.items():
                serializable[ck_name][sp] = {
                    "metrics": rr["metrics"],
                    "macro": rr["macro"],
                    "conf": rr["conf"].tolist(),
                }
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

    print("\nDone.\n")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    main()

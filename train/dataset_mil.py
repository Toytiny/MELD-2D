import os
import glob
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch
import sys
from torch.utils.data import Dataset

# from your codebase
# from utils.data_processing import Preprocess_Module

# Get the path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the path of the parent directory
parent_dir = os.path.dirname(script_dir)

# Add the parent directory to the Python path
sys.path.append(parent_dir)


def _even_sample(starts: List[int], cap: int) -> List[int]:
    """Evenly sample 'cap' indices from 'starts' (inclusive range)."""
    if cap is None or len(starts) <= cap:
        return starts
    idx = np.linspace(0, len(starts) - 1, cap).astype(int)
    return [starts[i] for i in idx]

class BagDatasetSingleNpy(Dataset):
    """
    Bag-level dataset for video-level labels with per-video .npy files (shape [T, 17, 3]).
    Each __getitem__ returns a list of window tensors + video label.

    Window generation:
    - mode='train': random offset jitter + stride coverage or uniform-K sampling
    - mode='val'/'test': deterministic offset (0) + stride coverage or uniform-K sampling

    The produced window tensor matches your ST-GCN preprocessing:
        tmp = {
            'img_shape': (H, W),
            'label': -1,
            'start_index': 0,
            'modality': 'Pose',
            'total_frames': win_len,
            'keypoint': np.tile(win[np.newaxis, :, :, :2], (2,1,1,1)),
            'keypoint_score': np.tile(win[np.newaxis, :, :, 2], (2,1,1)),
        }
        out = transform(tmp)
        clip = out['keypoint'][0]   # Tensor [C,T,V,2]
    """

    def __init__(
        self,
        data_list: List[str],
        root_dir: str,
        win_len: int = 124,
        stride: int = 62,
        mode: str = "train",
        cap_windows: Optional[int] = 64,
        sample_mode: str = "stride",         # 'stride' or 'uniform_k'
        frame_step: int = 1,                 # temporal downsample within the raw video (e.g., 2 keeps every 2nd frame)
        pad_short: bool = True,              # pad short videos to win_len
        transform=None,                      # Preprocess_Module(data_augmentation=(mode=='train'))
        img_shape: Tuple[int,int] = (320,480),
        conf_threshold: float = 0.2, 
        keep_at_least_one: bool = True,
        balance_videos: bool = False,
        target_mul: float = 1.0,       # 1.0 => upsample each class to max_class_count
        max_repeat: int = 5):          # cap per-video repetition
        """
        data_list: ["video_rel_path+++label", ...]  where file will be root_dir/<video_rel_path>.npy
        root_dir : directory containing <video>.npy files
        win_len  : window length (frames)
        stride   : sliding stride (frames), used when sample_mode='stride'
        cap_windows: cap the number of windows per video after generation (evenly sampled)
        sample_mode: 'stride' or 'uniform_k'
                     - 'stride': generate all starts by stride
                     - 'uniform_k': ignore stride; evenly select K=cap_windows window starts across timeline
        frame_step: sub-sample frames from the raw video before windowing
        """
        assert sample_mode in ("stride", "uniform_k")
        self.root_dir   = root_dir
        self.win_len    = win_len
        self.stride     = stride
        self.mode       = mode
        self.cap        = cap_windows
        self.sample_mode= sample_mode
        self.frame_step = max(1, frame_step)
        self.pad_short  = pad_short
        self.transform  = transform
        self.img_shape  = img_shape
        
        self.balance_videos = balance_videos and (mode == "train")
        self.target_mul = float(target_mul)
        self.max_repeat = int(max_repeat)

        self.items: List[Dict] = []  # each: {'path': <abs_path>, 'label': int, 'T': int}

        self.conf_threshold = float(conf_threshold)
        self.keep_at_least_one = bool(keep_at_least_one)

        self.base_seed = 1234
        self._rng = np.random.RandomState(self.base_seed)
        
        
        # Build index over videos
        for row in data_list:
            vid_id, lab_txt = row.split('+++')[0], row.split('+++')[1].strip()
            label = int(lab_txt) - 1
            npy_path = os.path.join(self.root_dir, f"{vid_id}.npy")
            if not os.path.exists(npy_path):
                # try direct path if user already includes .npy in the list
                alt_path = os.path.join(self.root_dir, vid_id)
                if alt_path.endswith('.npy') and os.path.exists(alt_path):
                    npy_path = alt_path
                else:
                    print(f"[WARN] Missing video npy: {npy_path}; skipped.")
                    continue

            # Read header to get T cheaply
            try:
                arr = np.load(npy_path, mmap_mode='r')  # [T, 17, 3]
                T_raw = int(arr.shape[0])
            except Exception as e:
                print(f"[WARN] Failed to load {npy_path}: {e}; skipped.")
                continue

            # Effective length after frame_step
            T = (T_raw + self.frame_step - 1) // self.frame_step
            if T <= 0:
                print(f"[WARN] Empty after frame_step: {npy_path}; skipped.")
                continue

            self.items.append(dict(path=npy_path, label=label, T=T, T_raw=T_raw))

        # NEW: build an index list, possibly oversampled per class
        self._build_index()
        
    def set_epoch(self, epoch:int):
            self._rng = np.random.RandomState(self.base_seed + epoch)
            # 若启用 oversample，顺带重新洗牌
            if hasattr(self, "indexes") and self.balance_videos:
                self._rng.shuffle(self.indexes)

    def _build_index(self):
        """Create a list of indices (with oversampling if requested)."""
        n = len(self.items)
        self.indexes = list(range(n))
        if not self.balance_videos or n == 0:
            return

        # class counts
        labels = np.array([it['label'] for it in self.items], dtype=int)
        classes = np.unique(labels)
        counts = {c: int((labels == c).sum()) for c in classes}
        max_cnt = max(counts.values())
        target = int(self.target_mul * max_cnt)  # target per class

        new_index_list = []
        for c in classes:
            idx_c = np.where(labels == c)[0].tolist()
            if len(idx_c) == 0: 
                continue
            # how many we want for class c
            want = max(target, len(idx_c))  # never downsample here
            # repeat-with-cap
            reps = min(self.max_repeat, max(1, int(np.ceil(want / len(idx_c)))))
            pool = idx_c * reps
            # if still short, sample extra (with replacement)
            if len(pool) < want:
                extra = self._rng.choice(idx_c, size=want - len(pool), replace=True).tolist()
                pool.extend(extra)
            # if longer, trim
            pool = pool[:want]
            new_index_list.extend(pool)

        self.indexes = new_index_list
        self._rng.shuffle(self.indexes)
        

    def __len__(self) -> int:
        return len(getattr(self, "indexes", [])) if self.balance_videos else len(self.items)

    def _iter_starts_stride(self, T: int) -> List[int]:
        """Generate window starts with stride; with optional random offset for train."""
        if T < self.win_len:
            return [0] if self.pad_short else []
        if self.mode == "train":
            # jitter offset in [0, stride//2]
            off = self._rng.randint(0, max(1, self.stride // 2 + 1))
        else:
            off = 0
        starts = []
        s = off
        while s + self.win_len <= T:
            starts.append(s)
            s += self.stride
        if not starts and self.pad_short:
            starts = [max(0, T - self.win_len)]
        return starts

    def _window_conf_mean(self, arr: np.ndarray, st: int) -> float:
        """Mean confidence over frames/joints for window starting at 'st' (using frame_step)."""
        st_raw = st * self.frame_step
        end_raw = st_raw + self.win_len * self.frame_step
        if end_raw <= arr.shape[0]:
            sl = arr[st_raw:end_raw:self.frame_step, :, 2]
        else:
            sl = arr[st_raw::self.frame_step, :, 2]
            # if too short, pad by repeating last frame's conf (zeros if empty)
            if sl.shape[0] == 0:
                return 0.0
            if sl.shape[0] < self.win_len:
                pad = self.win_len - sl.shape[0]
                last = sl[-1:]
                sl = np.concatenate([sl, np.repeat(last, pad, axis=0)], axis=0)
        m = np.nanmean(sl)
        if np.isnan(m): m = 0.0
        return float(m)

    def _iter_starts_uniform_k(self, T: int, K: int) -> List[int]:
        """Evenly choose K starts; add random phase/jitter during training."""
        if T < self.win_len:
            return [0] if self.pad_short else []
        if K is None or K <= 0:
            return self._iter_starts_stride(T)

        max_start = T - self.win_len
        if max_start < 0:
            return [0] if self.pad_short else []
        if K == 1:
            return [0]

        # base evenly spaced starts
        starts = np.linspace(0, max_start, K).astype(int)

        if self.mode == "train":
            # 1) random phase (shift all starts by the same small offset)
            #    step ≈ average distance between starts
            step = max(1, int(round(max_start / max(K - 1, 1))))
            off  = self._rng.randint(0, step)  # uniform in [0, step)
            starts = starts + off

            # 2) small local jitter per start (optional but helpful)
            jitter_span = max(1, step // 4)      # e.g., ±step/4
            jitter = self._rng.randint(-jitter_span, jitter_span + 1, size=K)
            starts = starts + jitter

            # clip to valid range
            starts = np.clip(starts, 0, max_start)

        # remove duplicates in rare edge cases and make list
        starts = np.unique(starts).astype(int).tolist()

        # fallback: always ensure at least one window if padding is allowed
        if not starts and self.pad_short:
            starts = [0]
        return starts

    def _load_window(self, arr: np.ndarray, st: int) -> torch.Tensor:
        """
        Slice/Pad one window and run your preprocessing transform to produce ST-GCN input tensor.
        arr is the raw array [T_raw, 17, 3]; but we've planned with T = ceil(T_raw/frame_step).
        We need to gather frames with the given frame_step.
        """
        # gather with frame_step
        # To avoid indexing overflow, compute the raw start index:
        st_raw = st * self.frame_step
        end_raw = st_raw + self.win_len * self.frame_step

        if end_raw <= arr.shape[0]:
            win = arr[st_raw:end_raw:self.frame_step]  # [win_len, 17, 3]
        else:
            # Need padding at the tail (repeat last frame)
            slice_arr = arr[st_raw::self.frame_step]
            if slice_arr.shape[0] == 0:
                # edge case: st_raw beyond arr length due to rounding -> back off
                st_raw = max(0, arr.shape[0] - self.win_len * self.frame_step)
                slice_arr = arr[st_raw::self.frame_step]
            pad = self.win_len - slice_arr.shape[0]
            if pad > 0:
                last = slice_arr[-1:] if slice_arr.shape[0] > 0 else np.zeros((1,17,3), dtype=np.float32)
                win = np.concatenate([slice_arr, np.repeat(last, pad, axis=0)], axis=0)
            else:
                win = slice_arr[:self.win_len]

        win = np.nan_to_num(win, nan=0.0).astype(np.float32)  # [T,17,3]

        # Build dict for your Preprocess_Module
        tmp = {
            'img_shape': self.img_shape,
            'label': -1,
            'start_index': 0,
            'modality': 'Pose',
            'total_frames': self.win_len,
            'keypoint': np.tile(win[np.newaxis, :, :, :2], (2,1,1,1)),   # [2,T,V,2]
            'keypoint_score': np.tile(win[np.newaxis, :, :, 2], (2,1,1)),# [2,T,V]
        }

        if self.transform is not None:
            out = self.transform(tmp)
            clip = out['keypoint'][0]  # Tensor [C,T,V,2]
        else:
            # Fallback: convert to tensor directly
            clip = torch.from_numpy(tmp['keypoint'][0])
        return clip

    def __getitem__(self, idx: int):
        real_idx = self.indexes[idx] if self.balance_videos else idx
        it = self.items[real_idx]
        path, label, T = it['path'], it['label'], it['T']

        # Load raw array lazily (mmap)
        arr = np.load(path, mmap_mode='r')  # [T_raw,17,3]

        # Generate window starts
        if self.sample_mode == 'uniform_k' and self.cap is not None:
            starts = self._iter_starts_uniform_k(T, self.cap)
        else:
            starts = self._iter_starts_stride(T)
            # After stride generation, optionally cap evenly
            if self.cap is not None and len(starts) > self.cap:
                starts = _even_sample(starts, self.cap)

        # Safety: ensure at least one window if padding is allowed
        if not starts and self.pad_short:
            starts = [0]

        # 2) HARD CONFIDENCE FILTER (drop windows below threshold)
        kept = []
        kept_scores = []
        for st in starts:
            mean_conf = self._window_conf_mean(arr, st)
            if mean_conf >= self.conf_threshold:
                kept.append(st)
                kept_scores.append(mean_conf)

        # 3) fallback to avoid empty bags
        if len(kept) == 0:
            if self.keep_at_least_one:
                # keep the best among candidates (even if < threshold)
                best_st = max(starts, key=lambda s: self._window_conf_mean(arr, s))
                kept = [best_st]
            else:
                # return an empty bag (rarely recommended; training code must handle)
                return [], label

        # Build window list
        clips: List[torch.Tensor] = [self._load_window(arr, st) for st in kept]

        return clips, it['label']


def collate_bags(batch):
    """
    Collate function for BagDatasetSingleNpy.
    Pads to the max #windows across the batch and produces a boolean mask.

    Input:
        batch: List of (List[Tensor clip_i], label_int)

    Output:
        X: [B, Wmax, C, T, V, 2]
        M: [B, Wmax]  (True for valid windows)
        Y: [B]
    """
    B = len(batch)
    maxW = max(len(clips) for clips, _ in batch)
    if maxW == 0:
        raise RuntimeError("All bags are empty. Check pad_short/sample_mode settings.")

    # infer clip shape from first available clip
    c0 = None
    for clips, _ in batch:
        if len(clips) > 0:
            c0 = clips[0]
            break
    assert c0 is not None, "Cannot infer clip shape."

    X = torch.zeros((B, maxW, *c0.shape), dtype=c0.dtype)
    M = torch.zeros((B, maxW), dtype=torch.bool)
    Y = torch.tensor([lab for _, lab in batch], dtype=torch.long)

    for b, (clips, _) in enumerate(batch):
        for w, clip in enumerate(clips):
            X[b, w] = clip
            M[b, w] = True

    return X, M, Y

# core.py

import os
import io
import re
import math
import time
import random
import hashlib
import platform
import subprocess

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Config
# ============================================================

SEEDS = [21, 22, 23]
RATIOS = [1, 2, 5, 10, 25, 50, 100]
PROTOCOL_A_SIZES = [2000, 4000, 8000, 16000, 32000, 64000, 128000]
N_F = 64000
STEP_BUDGET = 5000
WARMUP = 500
BATCH_SIZE = 128
ECE_BINS = 15
LOG_EVERY = 250
SAVE_MODELS = False
USE_AMP = False
EVAL_BALANCE_SEED = 12345
RAM_BUDGET_BYTES = 24 * 1024 ** 3
MODEL_NAMES = ["logistic", "cnn", "resnet18"]
HP = {
    "logistic": {"lr": 3e-4, "wd": 1e-2},
    "cnn": {"lr": 3e-4, "wd": 1e-4},
    "resnet18": {"lr": 3e-4, "wd": 1e-4},
}
NORM_MEAN = [0.700756, 0.538358, 0.691621]
NORM_STD = [0.234978, 0.277409, 0.212895]
AUG_SPEC = "hflip0.5|vflip0.5|rot90{0,90,180,270}"

DATAWISE_RESULTS_SLUG = "hrushisanap/datawise-results"

WORKING_ROOT = "/kaggle/working"


def _default_paths(root):
    j = os.path.join
    return {
        "root": root,
        "permutations": j(root, "random_permutations"),
        "balanced_eval": j(root, "balanced_eval"),
        "subsets": j(root, "subsets"),
        "registry": j(root, "condition_registry.csv"),
        "results": j(root, "results"),
        "ledger": j(root, "results", "runs_ledger.csv"),
        "master": j(root, "results", "master_results.csv"),
        "logits": j(root, "logits"),
        "models": j(root, "models"),
        "curves": j(root, "training_curves"),
        "logs": j(root, "logs"),
        "frozen_config": j(root, "frozen_config.md"),
        "figures": j(root, "figures"),
        "summaries": j(root, "summaries"),
        "cache": j(root, "cache"),
    }


PATHS = _default_paths(WORKING_ROOT)

IMG_H, IMG_W, IMG_C = 96, 96, 3
FLAT_DIM = IMG_H * IMG_W * IMG_C
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

LEDGER_COLUMNS = [
    "run_id", "condition_id", "protocol", "model", "n_min", "n_maj",
    "target_ratio", "actual_ratio", "data_seed", "init_seed", "loader_seed",
    "step_budget", "actual_steps", "effective_epochs", "exposures_per_image",
    "status", "config_hash", "timestamp",
]


# ============================================================
# Config hashing and frozen config
# ============================================================

def _canon(obj):
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, float):
        return f"{obj:.10g}"
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_canon(x) for x in obj) + "]"
    if isinstance(obj, dict):
        return "{" + ",".join(f"{k}:{_canon(obj[k])}" for k in sorted(obj)) + "}"
    raise TypeError(f"uncanonicalisable type: {type(obj)}")


def compute_config_hash(norm_mean=None, norm_std=None):
    nm = NORM_MEAN if norm_mean is None else norm_mean
    ns = NORM_STD if norm_std is None else norm_std
    payload = {
        "SEEDS": list(SEEDS),
        "RATIOS": list(RATIOS),
        "PROTOCOL_A_SIZES": list(PROTOCOL_A_SIZES),
        "N_F": N_F,
        "STEP_BUDGET": STEP_BUDGET,
        "WARMUP": WARMUP,
        "BATCH_SIZE": BATCH_SIZE,
        "USE_AMP": USE_AMP,
        "MODEL_NAMES": list(MODEL_NAMES),
        "HP": {k: {hk: float(hv) for hk, hv in v.items()} for k, v in HP.items()},
        "NORM_MEAN": [float(x) for x in nm],
        "NORM_STD": [float(x) for x in ns],
        "EVAL_BALANCE_SEED": EVAL_BALANCE_SEED,
        "AUG_SPEC": AUG_SPEC,
    }
    return hashlib.sha256(_canon(payload).encode("utf-8")).hexdigest()[:16]


CONFIG_HASH = compute_config_hash()


def _parse_frozen_config(path):
    values = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("- ") or ":" not in line:
                continue
            key, rest = line[2:].split(":", 1)
            values[key.strip()] = rest.strip()
    return values


def refresh_config():
    """Re-read frozen_config.md if present, override norm stats, recompute hash."""
    global NORM_MEAN, NORM_STD, CONFIG_HASH
    path = PATHS["frozen_config"]
    if os.path.exists(path):
        vals = _parse_frozen_config(path)
        if "norm_mean" in vals:
            NORM_MEAN = [float(x) for x in vals["norm_mean"].split(",")]
        if "norm_std" in vals:
            NORM_STD = [float(x) for x in vals["norm_std"].split(",")]
    CONFIG_HASH = compute_config_hash()
    return CONFIG_HASH


def write_frozen_config(mean, std, path=None):
    """Write the authoritative constants for the runs. Called by notebook 00."""
    path = path or PATHS["frozen_config"]
    mean = [round(float(x), 6) for x in mean]
    std = [round(float(x), 6) for x in std]
    h = compute_config_hash(mean, std)
    lines = [
        "# Frozen configuration",
        "",
        "Written by notebook 00 at setup. Authoritative for the runs.",
        "",
        f"- step_budget: {STEP_BUDGET}",
        f"- n_f: {N_F}",
        f"- batch_size: {BATCH_SIZE}",
        f"- warmup: {WARMUP}",
        f"- use_amp: {str(USE_AMP).lower()}",
        f"- eval_balance_seed: {EVAL_BALANCE_SEED}",
        f"- seeds: {', '.join(str(s) for s in SEEDS)}",
        f"- hp_logistic: lr={HP['logistic']['lr']:.10g}, wd={HP['logistic']['wd']:.10g}",
        f"- hp_cnn: lr={HP['cnn']['lr']:.10g}, wd={HP['cnn']['wd']:.10g}",
        f"- hp_resnet18: lr={HP['resnet18']['lr']:.10g}, wd={HP['resnet18']['wd']:.10g}",
        f"- norm_mean: {mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}",
        f"- norm_std: {std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}",
        f"- config_hash: {h}",
        "",
    ]
    _ensure(os.path.dirname(path))
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


# ============================================================
# Seeding (three streams)
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _loader_generator(loader_seed):
    g = torch.Generator()
    g.manual_seed(int(loader_seed))
    return g


# ============================================================
# Filesystem and logging helpers
# ============================================================

_SESSION_TS = None


def _now():
    return time.strftime("%Y%m%d_%H%M%S")


def _session_ts():
    global _SESSION_TS
    if _SESSION_TS is None:
        _SESSION_TS = _now()
    return _SESSION_TS


def _ensure(directory):
    if directory:
        os.makedirs(directory, exist_ok=True)


def log(message, stage="run", also_print=True):
    _ensure(PATHS["logs"])
    path = os.path.join(PATHS["logs"], f"{stage}_{_session_ts()}.txt")
    stamp = time.strftime("%H:%M:%S")
    with open(path, "a") as f:
        f.write(f"[{stamp}] {message}\n")
    if also_print:
        print(f"[{stamp}] {message}")


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _git_commit():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], stderr=subprocess.DEVNULL
        ) != 0
        return commit, "dirty" if dirty else "clean"
    except Exception:
        return "unavailable", "unavailable"


def log_environment():
    import numpy as _np
    env = {}
    env["python"] = platform.python_version()
    env["torch"] = torch.__version__
    env["torchvision"] = torchvision.__version__
    env["numpy"] = _np.__version__
    try:
        import sklearn
        env["sklearn"] = sklearn.__version__
    except Exception:
        env["sklearn"] = "unavailable"
    env["pandas"] = pd.__version__
    env["cuda"] = torch.version.cuda if torch.cuda.is_available() else "cpu"
    env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    env["os"] = platform.platform()
    commit, state = _git_commit()
    env["git_commit"] = commit
    env["git_state"] = state
    env["config_hash"] = CONFIG_HASH
    for k, v in env.items():
        log(f"env {k}: {v}", stage="env")
    return env


# ============================================================
# Data loading (PCam parquet, PNG decode, in-RAM)
# ============================================================

class PCamData:
    def __init__(self, train_images, train_labels, train_pos_idx, train_neg_idx,
                 valid_images, valid_labels, valid_kept,
                 test_images, test_labels, test_kept, counts):
        self.train_images = train_images
        self.train_labels = train_labels
        self.train_pos_idx = train_pos_idx
        self.train_neg_idx = train_neg_idx
        self.valid_images = valid_images
        self.valid_labels = valid_labels
        self.valid_kept = valid_kept
        self.test_images = test_images
        self.test_labels = test_labels
        self.test_kept = test_kept
        self.counts = counts


def _split_shards(root, split):
    pat = re.compile(rf"^{split}-\d{{5}}-of-\d{{5}}.*\.parquet$")
    matches = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if pat.match(fn):
                matches.append(os.path.join(dirpath, fn))
    matches.sort(key=os.path.basename)
    return matches


def _split_rowcount(shards):
    import pyarrow.parquet as pq
    return sum(pq.ParquetFile(sp).metadata.num_rows for sp in shards)


def _decode_split(shards, out_images, out_labels, check_png=False):
    import pyarrow.parquet as pq
    from PIL import Image
    pos = 0
    checked = not check_png
    for sp in shards:
        table = pq.read_table(sp, columns=["image", "label"])
        cells = table.column("image").to_pylist()
        labs = table.column("label").to_pylist()
        for cell, lab in zip(cells, labs):
            raw = cell["bytes"]
            if not checked:
                if raw[:8] != PNG_SIGNATURE:
                    raise ValueError("first image is not PNG")
                checked = True
            im = Image.open(io.BytesIO(raw)).convert("RGB")
            arr = np.asarray(im, dtype=np.uint8)
            if arr.shape != (IMG_H, IMG_W, IMG_C):
                raise ValueError(f"unexpected image shape {arr.shape}")
            out_images[pos] = arr
            out_labels[pos] = int(lab)
            pos += 1
        del table, cells, labs
    return pos


def _alloc_images(n, memmap_path=None):
    shape = (n, IMG_H, IMG_W, IMG_C)
    if memmap_path is None:
        return np.empty(shape, dtype=np.uint8)
    _ensure(os.path.dirname(memmap_path))
    return np.memmap(memmap_path, dtype=np.uint8, mode="w+", shape=shape)


def balance_eval_indices(labels, seed):
    """Trim the larger class to the size of the smaller by a seeded random draw.

    Keeps all minority positions plus an equal-size random subset of the
    majority. Returns sorted global positions of length 2k.
    """
    labels = np.asarray(labels)
    pos_positions = np.where(labels == 1)[0]
    neg_positions = np.where(labels == 0)[0]
    if len(pos_positions) <= len(neg_positions):
        min_pos, maj_pos = pos_positions, neg_positions
    else:
        min_pos, maj_pos = neg_positions, pos_positions
    k = len(min_pos)
    keep_maj = np.sort(np.random.default_rng(seed).choice(maj_pos, size=k, replace=False))
    kept = np.sort(np.concatenate([min_pos, keep_maj]))
    return kept


def _save_index_csv(path, indices, column="global_index"):
    _ensure(os.path.dirname(path))
    pd.DataFrame({column: np.asarray(indices)}).to_csv(path, index=False)


def load_pcam(root=None, cache=False):
    search_root = root or "/kaggle/input"
    shards = {s: _split_shards(search_root, s) for s in ("train", "valid", "test")}
    for split, sp in shards.items():
        if not sp:
            raise FileNotFoundError(f"no {split} parquet shards under {search_root}")

    expected = {"train": 262144, "valid": 32768, "test": 32768}
    counts_rows = {s: _split_rowcount(shards[s]) for s in shards}
    for split, n in counts_rows.items():
        if n != expected[split]:
            raise ValueError(f"{split} size {n} != expected {expected[split]}")

    cache_dir = PATHS["cache"]
    train_cache = os.path.join(cache_dir, "train_images.dat")
    train_bytes = counts_rows["train"] * IMG_H * IMG_W * IMG_C
    use_memmap = train_bytes > RAM_BUDGET_BYTES

    train_images = _alloc_images(
        counts_rows["train"], memmap_path=train_cache if use_memmap else None
    )
    train_labels = np.empty(counts_rows["train"], dtype=np.int64)
    _decode_split(shards["train"], train_images, train_labels, check_png=True)
    if use_memmap:
        train_images.flush()

    valid_full = _alloc_images(counts_rows["valid"])
    valid_labels_full = np.empty(counts_rows["valid"], dtype=np.int64)
    _decode_split(shards["valid"], valid_full, valid_labels_full)

    test_full = _alloc_images(counts_rows["test"])
    test_labels_full = np.empty(counts_rows["test"], dtype=np.int64)
    _decode_split(shards["test"], test_full, test_labels_full)

    counts = {
        "train_pos": int((train_labels == 1).sum()),
        "train_neg": int((train_labels == 0).sum()),
        "valid_pos_raw": int((valid_labels_full == 1).sum()),
        "valid_neg_raw": int((valid_labels_full == 0).sum()),
        "test_pos_raw": int((test_labels_full == 1).sum()),
        "test_neg_raw": int((test_labels_full == 0).sum()),
    }
    log(f"train pos/neg {counts['train_pos']}/{counts['train_neg']}", stage="data")
    log(f"valid pos/neg (raw) {counts['valid_pos_raw']}/{counts['valid_neg_raw']}", stage="data")
    log(f"test pos/neg (raw) {counts['test_pos_raw']}/{counts['test_neg_raw']}", stage="data")

    if counts["train_pos"] < 128000 or counts["train_neg"] < 128000:
        raise ValueError("train pool has fewer than 128,000 of a class")

    train_pos_idx = np.where(train_labels == 1)[0]
    train_neg_idx = np.where(train_labels == 0)[0]

    valid_kept = balance_eval_indices(valid_labels_full, EVAL_BALANCE_SEED)
    test_kept = balance_eval_indices(test_labels_full, EVAL_BALANCE_SEED)
    _save_index_csv(os.path.join(PATHS["balanced_eval"], "valid_kept.csv"), valid_kept)
    _save_index_csv(os.path.join(PATHS["balanced_eval"], "test_kept.csv"), test_kept)

    valid_images = np.ascontiguousarray(valid_full[valid_kept])
    valid_labels = valid_labels_full[valid_kept]
    test_images = np.ascontiguousarray(test_full[test_kept])
    test_labels = test_labels_full[test_kept]
    del valid_full, test_full

    counts["valid_balanced"] = int(len(valid_kept))
    counts["test_balanced"] = int(len(test_kept))
    log(f"valid balanced {counts['valid_balanced']} "
        f"({int((valid_labels == 1).sum())}/{int((valid_labels == 0).sum())})", stage="data")
    log(f"test balanced {counts['test_balanced']} "
        f"({int((test_labels == 1).sum())}/{int((test_labels == 0).sum())})", stage="data")

    if cache:
        _ensure(cache_dir)
        np.save(os.path.join(cache_dir, "valid_images.npy"), valid_images)
        np.save(os.path.join(cache_dir, "valid_labels.npy"), valid_labels)
        np.save(os.path.join(cache_dir, "test_images.npy"), test_images)
        np.save(os.path.join(cache_dir, "test_labels.npy"), test_labels)
        np.save(os.path.join(cache_dir, "train_labels.npy"), train_labels)

    return PCamData(
        train_images, train_labels, train_pos_idx, train_neg_idx,
        valid_images, valid_labels, valid_kept,
        test_images, test_labels, test_kept, counts,
    )


def compute_norm_stats(data, chunk=1024):
    """Per-channel mean/std over the full train pool on [0,1], streamed.

    Accumulates per-channel sum and sum-of-squares in float64; peak memory is
    one float64 chunk, well under 1 GB at the default chunk size.
    """
    n = data.shape[0]
    total_px = n * IMG_H * IMG_W
    s = np.zeros(3, dtype=np.float64)
    ss = np.zeros(3, dtype=np.float64)
    for start in range(0, n, chunk):
        block = np.asarray(data[start:start + chunk], dtype=np.float64) / 255.0
        flat = block.reshape(-1, 3)
        s += flat.sum(axis=0)
        for c in range(3):
            col = flat[:, c]
            ss[c] += float(np.dot(col, col))
    mean = s / total_px
    var = ss / total_px - mean ** 2
    std = np.sqrt(np.clip(var, 0.0, None))
    return mean.tolist(), std.tolist()


# ============================================================
# Nested sampling (permutations, subsets)
# ============================================================

_PERM_CACHE = {}


def get_perms(data_seed, data, save=False):
    """Return (pos_perm, neg_perm): permutations of the train-pool positive and
    negative global indices, a deterministic function of data_seed."""
    key = (int(data_seed), len(data.train_pos_idx), len(data.train_neg_idx))
    if key not in _PERM_CACHE:
        rng = np.random.default_rng(data_seed)
        pos_perm = data.train_pos_idx[rng.permutation(len(data.train_pos_idx))]
        neg_perm = data.train_neg_idx[rng.permutation(len(data.train_neg_idx))]
        _PERM_CACHE[key] = (pos_perm, neg_perm)
    pos_perm, neg_perm = _PERM_CACHE[key]
    if save:
        _save_index_csv(
            os.path.join(PATHS["permutations"], f"pos_perm_d{data_seed}.csv"), pos_perm)
        _save_index_csv(
            os.path.join(PATHS["permutations"], f"neg_perm_d{data_seed}.csv"), neg_perm)
    return pos_perm, neg_perm


def make_subset(n_min, n_maj, data_seed, data, save=False):
    pos_perm, neg_perm = get_perms(data_seed, data, save=False)
    if n_min > len(pos_perm):
        raise ValueError(f"n_min {n_min} exceeds positive pool {len(pos_perm)}")
    if n_maj > len(neg_perm):
        raise ValueError(f"n_maj {n_maj} exceeds negative pool {len(neg_perm)}")
    subset = np.concatenate([pos_perm[:n_min], neg_perm[:n_maj]])
    if save:
        _save_index_csv(
            os.path.join(PATHS["subsets"], f"s_m{n_min}_M{n_maj}_d{data_seed}.csv"), subset)
    return subset


# ============================================================
# Datasets and loaders
# ============================================================

def _norm_tensors(mean, std):
    m = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return m, s


class _TrainDataset(Dataset):
    def __init__(self, images, labels, subset_idx, mean, std, aug_rng=None):
        self.images = images
        self.labels = labels
        self.subset_idx = np.asarray(subset_idx)
        self.mean, self.std = _norm_tensors(mean, std)
        self.aug_rng = aug_rng

    def __len__(self):
        return len(self.subset_idx)

    def __getitem__(self, i):
        g = int(self.subset_idx[i])
        img = self.images[g]
        if self.aug_rng is not None:
            if self.aug_rng.random() < 0.5:
                img = img[:, ::-1, :]
            if self.aug_rng.random() < 0.5:
                img = img[::-1, :, :]
            k = int(self.aug_rng.integers(0, 4))
            if k:
                img = np.rot90(img, k)
        x = torch.from_numpy(np.ascontiguousarray(img)).float().div_(255.0)
        x = x.permute(2, 0, 1)
        x = (x - self.mean) / self.std
        return x, int(self.labels[g])


class _EvalDataset(Dataset):
    def __init__(self, images, labels, mean, std):
        self.images = images
        self.labels = labels
        self.mean, self.std = _norm_tensors(mean, std)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = torch.from_numpy(np.ascontiguousarray(self.images[i])).float().div_(255.0)
        x = x.permute(2, 0, 1)
        x = (x - self.mean) / self.std
        return x, int(self.labels[i])


def build_loaders(data, subset_idx, loader_seed, eval_batch=512):
    aug_rng = np.random.default_rng(loader_seed)
    train_ds = _TrainDataset(
        data.train_images, data.train_labels, subset_idx, NORM_MEAN, NORM_STD, aug_rng)
    val_ds = _EvalDataset(data.valid_images, data.valid_labels, NORM_MEAN, NORM_STD)
    test_ds = _EvalDataset(data.test_images, data.test_labels, NORM_MEAN, NORM_STD)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=0, generator=_loader_generator(loader_seed))
    val_loader = DataLoader(
        val_ds, batch_size=eval_batch, shuffle=False, num_workers=0)
    test_loader = DataLoader(
        test_ds, batch_size=eval_batch, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


# ============================================================
# Models
# ============================================================

_PRINTED_PARAMS = set()


class _CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, 2)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.head(x)


def build_model(name):
    if name == "logistic":
        model = nn.Sequential(nn.Flatten(), nn.Linear(FLAT_DIM, 2))
    elif name == "cnn":
        model = _CNN()
    elif name == "resnet18":
        model = torchvision.models.resnet18(weights=None, num_classes=2)
    else:
        raise ValueError(f"unknown model {name}")
    if name not in _PRINTED_PARAMS:
        n_params = sum(p.numel() for p in model.parameters())
        log(f"{name} parameter count: {n_params}", stage="model")
        _PRINTED_PARAMS.add(name)
    return model


# ============================================================
# Training (test-blind) and logit dump
# ============================================================

def _scheduler(optimiser, step_budget):
    def lr_lambda(step):
        if step < WARMUP:
            return (step + 1) / WARMUP
        progress = (step - WARMUP) / max(1, step_budget - WARMUP)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)


def train_model(model, loader, hp, step_budget, log_fn):
    device = _device()
    model.to(device).train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    scheduler = _scheduler(optimiser, step_budget)
    criterion = nn.CrossEntropyLoss()
    use_amp = USE_AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    curve = []
    n_samples = len(loader.dataset)
    step = 0
    while step < step_budget:
        for x, y in loader:
            if step >= step_budget:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(x)
                    loss = criterion(logits, y)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step {step + 1}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimiser)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step {step + 1}")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
            scheduler.step()
            step += 1
            if step % LOG_EVERY == 0 or step == step_budget:
                lv = float(loss.detach().item())
                curve.append((step, lv))
                if log_fn is not None:
                    log_fn(step, lv)

    actual_steps = step
    effective_epochs = actual_steps * BATCH_SIZE / n_samples
    return model, curve, actual_steps, effective_epochs


@torch.no_grad()
def dump_logits(model, loader):
    device = next(model.parameters()).device
    model.eval()
    ys, diffs = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).float()
        d = (logits[:, 1] - logits[:, 0]).cpu().numpy()
        diffs.append(d)
        ys.append(np.asarray(y))
    return np.concatenate(ys), np.concatenate(diffs)


# ============================================================
# Bookkeeping (protocol-independent IDs, registry, ledger)
# ============================================================

def make_condition_id(n_min, n_maj):
    return f"m{int(n_min)}_M{int(n_maj)}"


def make_run_id(model, n_min, n_maj, data_seed, init_seed, loader_seed):
    return (f"{model}__m{int(n_min)}_M{int(n_maj)}"
            f"__d{int(data_seed)}_i{int(init_seed)}_l{int(loader_seed)}")


def _merge_protocols(existing, new):
    parts = set()
    if isinstance(existing, str) and existing:
        parts.update(p for p in existing.split("|") if p)
    if new:
        parts.add(new)
    return "|".join(sorted(parts))


def register_condition(condition_id, n_min, n_maj, protocol):
    path = PATHS["registry"]
    cols = ["condition_id", "n_min", "n_maj", "protocols"]
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=cols)
    mask = df["condition_id"] == condition_id
    if mask.any():
        idx = df.index[mask][0]
        df.at[idx, "protocols"] = _merge_protocols(df.at[idx, "protocols"], protocol)
    else:
        row = {"condition_id": condition_id, "n_min": int(n_min),
               "n_maj": int(n_maj), "protocols": protocol}
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _ensure(os.path.dirname(path))
    df.to_csv(path, index=False)


def save_logits(run_id, which, y_true, logit_diff):
    assert which in ("test", "val")
    _ensure(PATHS["logits"])
    path = os.path.join(PATHS["logits"], f"{run_id}_{which}.csv")
    pd.DataFrame({"y_true": np.asarray(y_true).astype(int),
                  "logit_diff": np.asarray(logit_diff).astype(np.float32)}
                 ).to_csv(path, index=False)
    return path


def save_curve(run_id, curve):
    _ensure(PATHS["curves"])
    path = os.path.join(PATHS["curves"], f"{run_id}.csv")
    pd.DataFrame(curve, columns=["step", "train_loss"]).to_csv(path, index=False)
    return path


def save_model(run_id, model):
    if not SAVE_MODELS:
        return None
    _ensure(PATHS["models"])
    path = os.path.join(PATHS["models"], f"{run_id}.pt")
    torch.save(model.state_dict(), path)
    return path


def save_subset(subset, n_min, n_maj, data_seed):
    path = os.path.join(PATHS["subsets"], f"s_m{n_min}_M{n_maj}_d{data_seed}.csv")
    _save_index_csv(path, subset)
    return path


def _read_ledger():
    path = PATHS["ledger"]
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=LEDGER_COLUMNS)


def _write_ledger(df):
    _ensure(PATHS["results"])
    df.to_csv(PATHS["ledger"], index=False)


def _append_ledger_row(row):
    df = _read_ledger()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _write_ledger(df)


# ============================================================
# Runner (single shared, test-blind)
# ============================================================

def run_training(protocol, model, n_min, n_maj, data_seed, init_seed, loader_seed,
                 data, target_ratio="", resume=True):
    run_id = make_run_id(model, n_min, n_maj, data_seed, init_seed, loader_seed)
    condition_id = make_condition_id(n_min, n_maj)
    n_total = n_min + n_maj
    actual_ratio = n_maj / n_min
    exposures_per_image = STEP_BUDGET * BATCH_SIZE / n_total
    tag = f"[{protocol}][{model}][m={n_min} M={n_maj} r={actual_ratio:.2f}][seed={data_seed}]"

    if resume:
        df = _read_ledger()
        hit = df[(df["run_id"] == run_id)
                 & (df["status"] == "success")
                 & (df["config_hash"] == CONFIG_HASH)]
        if len(hit) > 0:
            idx = hit.index[0]
            df.at[idx, "protocol"] = _merge_protocols(df.at[idx, "protocol"], protocol)
            _write_ledger(df)
            register_condition(condition_id, n_min, n_maj, protocol)
            log(f"{tag} reuse of {run_id}, no retrain", stage=protocol)
            return run_id

    try:
        get_perms(data_seed, data, save=True)
        subset = make_subset(n_min, n_maj, data_seed, data, save=True)

        set_seed(init_seed)
        net = build_model(model)
        train_loader, val_loader, test_loader = build_loaders(data, subset, loader_seed)

        def _log_fn(step, loss):
            log(f"{tag} step {step}/{STEP_BUDGET} loss={loss:.4f}", stage=protocol)

        net, curve, actual_steps, effective_epochs = train_model(
            net, train_loader, HP[model], STEP_BUDGET, _log_fn)

        y_test, d_test = dump_logits(net, test_loader)
        y_val, d_val = dump_logits(net, val_loader)
        save_logits(run_id, "test", y_test, d_test)
        save_logits(run_id, "val", y_val, d_val)
        save_curve(run_id, curve)
        save_model(run_id, net)
        register_condition(condition_id, n_min, n_maj, protocol)

        _append_ledger_row({
            "run_id": run_id, "condition_id": condition_id, "protocol": protocol,
            "model": model, "n_min": n_min, "n_maj": n_maj,
            "target_ratio": target_ratio, "actual_ratio": actual_ratio,
            "data_seed": data_seed, "init_seed": init_seed, "loader_seed": loader_seed,
            "step_budget": STEP_BUDGET, "actual_steps": actual_steps,
            "effective_epochs": effective_epochs, "exposures_per_image": exposures_per_image,
            "status": "success", "config_hash": CONFIG_HASH, "timestamp": _now(),
        })
        log(f"{tag} eff_epochs={effective_epochs:.1f} status=success", stage=protocol)
        return run_id

    except Exception as exc:
        import traceback
        log(f"{tag} FAILED: {exc}", stage=protocol)
        log(traceback.format_exc(), stage=protocol, also_print=False)
        _append_ledger_row({
            "run_id": run_id, "condition_id": condition_id, "protocol": protocol,
            "model": model, "n_min": n_min, "n_maj": n_maj,
            "target_ratio": target_ratio, "actual_ratio": actual_ratio,
            "data_seed": data_seed, "init_seed": init_seed, "loader_seed": loader_seed,
            "step_budget": STEP_BUDGET, "actual_steps": 0,
            "effective_epochs": 0.0, "exposures_per_image": exposures_per_image,
            "status": "failed", "config_hash": CONFIG_HASH, "timestamp": _now(),
        })
        return run_id


# ============================================================
# Kaggle persistence
# ============================================================

_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9][A-Za-z0-9-]*$")


def _check_slug(slug):
    if not slug:
        raise RuntimeError("DATAWISE_RESULTS_SLUG is unset; set it to 'owner/slug' first")
    if not _SLUG_RE.match(slug):
        raise ValueError(f"results slug '{slug}' is not in owner/slug form")
    return slug


def _kaggle_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as exc:
        raise RuntimeError("kaggle package not available") from exc
    api = KaggleApi()
    api.authenticate()
    return api


def _write_dataset_metadata(folder, slug):
    import json
    title = slug.split("/", 1)[1].replace("-", " ").title()
    meta = {"title": title, "id": slug, "licenses": [{"name": "CC0-1.0"}]}
    with open(os.path.join(folder, "dataset-metadata.json"), "w") as f:
        json.dump(meta, f)


def create_results_store(folder=None, slug=None):
    slug = _check_slug(slug or DATAWISE_RESULTS_SLUG)
    folder = folder or PATHS["root"]
    api = _kaggle_api()
    _write_dataset_metadata(folder, slug)
    api.dataset_create_new(folder, dir_mode="zip", public=False)
    return slug


def push_results_store(folder=None, slug=None, version_notes=None):
    slug = _check_slug(slug or DATAWISE_RESULTS_SLUG)
    folder = folder or PATHS["root"]
    api = _kaggle_api()
    _write_dataset_metadata(folder, slug)
    notes = version_notes or f"update {_now()} hash {CONFIG_HASH}"
    try:
        api.dataset_create_version(folder, version_notes=notes, dir_mode="zip")
    except Exception:
        return create_results_store(folder, slug)
    return slug


def pull_results_store(slug=None, dest=None):
    slug = _check_slug(slug or DATAWISE_RESULTS_SLUG)
    dest = dest or PATHS["root"]
    _ensure(dest)
    api = _kaggle_api()
    api.dataset_download_files(slug, path=dest, unzip=True)
    return dest
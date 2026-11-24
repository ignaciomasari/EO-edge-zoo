"""Task-agnostic data helpers: split types, stratified label subsets, batching, band stats."""
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .config import RunConfig

Split = tuple[torch.Tensor, torch.Tensor]  # (x int16 [N,C,H,W] in task.bands order, y long [N])
Data = dict[str, Split]                   # split name ("train" | "val" | "test") -> Split
EVAL_BS = 256  # eval / feature extraction batch


def subset_idx(y: np.ndarray | torch.Tensor, frac: float, seed: int, num_classes: int) -> np.ndarray:
    """Stratified train subset; same (frac, seed) -> same idx for every model (paired comparison)."""
    y = y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
    if frac >= 1.0:
        return np.arange(len(y))
    rng = np.random.default_rng(10_000 + seed)
    per_class = [rng.choice(np.flatnonzero(y == label), max(1, round(frac * (y == label).sum())), replace=False)
                 for label in range(num_classes)]
    return np.sort(np.concatenate(per_class))


def train_subset(data: Data, cfg: "RunConfig") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> (idx, x, y) of cfg.frac stratified train subset for cfg.seed."""
    idx = torch.from_numpy(subset_idx(data["train"][1], cfg.frac, cfg.seed, cfg.num_classes))
    return idx, data["train"][0][idx], data["train"][1][idx]


def band_stats(x: torch.Tensor, max_samples: int = 2000) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-band DN mean/std on first max_samples (for imagenet_13ch z-score)."""
    x_float = x[:max_samples].float()
    return x_float.mean((0, 2, 3)), x_float.std((0, 2, 3))


def iterate(num_samples: int, batch_size: int, shuffle: bool, drop_last: bool = False) -> Iterator[torch.Tensor]:
    """Index batches; drop_last only if num_samples >= batch_size."""
    order = torch.randperm(num_samples) if shuffle else torch.arange(num_samples)
    stop = num_samples - num_samples % batch_size if (drop_last and num_samples >= batch_size) else num_samples
    for start in range(0, stop, batch_size):
        yield order[start:start + batch_size]

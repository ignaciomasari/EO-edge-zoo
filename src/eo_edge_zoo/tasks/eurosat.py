"""EuroSAT: Sentinel-2 L1C, 13 bands, 10 land-cover classes, torchgeo splits."""
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..bands import S2_SSL_ORDER, band_index
from .base import ClassificationTask

if TYPE_CHECKING:
    from ..data import Data, Split


@dataclass(frozen=True)
class EuroSAT(ClassificationTask):
    name: str = "eurosat"
    num_classes: int = 10
    bands: tuple[str, ...] = field(default=S2_SSL_ORDER)  # files store B8A last -> reordered on cache

    def load(self, root: str, num_workers: int = 4) -> "Data":
        return {split: self._cache_split(root, split, num_workers) for split in self.splits}

    def _cache_split(self, root: str, split: str, num_workers: int) -> "Split":
        """-> (int16 x [N,13,64,64] in `bands` order, long y); cached as .npy under root."""
        import numpy as np
        import torch

        x_path, y_path = (os.path.join(root, f"cache_{split}_{name}.npy") for name in ("x", "y"))
        if not os.path.isfile(x_path):
            from torchgeo.datasets import EuroSAT as TorchgeoEuroSAT  # heavy: only for first build
            from tqdm.auto import tqdm

            dataset = TorchgeoEuroSAT(root, split=split, bands=TorchgeoEuroSAT.all_band_names, download=True)
            order = band_index(TorchgeoEuroSAT.all_band_names, self.bands)
            loader = torch.utils.data.DataLoader(dataset, batch_size=256, num_workers=num_workers)
            x_batches, y_batches = [], []
            for batch in tqdm(loader, desc=f"cache {split}"):
                x_batches.append(batch["image"][:, order].round().clamp(0, 32767).to(torch.int16).numpy())
                y_batches.append(batch["label"].numpy())
            np.save(x_path, np.concatenate(x_batches))
            np.save(y_path, np.concatenate(y_batches))
        return torch.from_numpy(np.load(x_path)), torch.from_numpy(np.load(y_path)).long()

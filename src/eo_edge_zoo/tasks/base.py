"""Task interface. Light module: torch only imported inside methods."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from ..bands import RGB, S2_SSL_ORDER, band_index
from ..config import Init

if TYPE_CHECKING:
    import torch

    from ..data import Data


@dataclass(frozen=True)
class Task(ABC):
    primary_metric: ClassVar[str]           # metrics() key -> top1_val/top1_test cols, val model selection
    name: str
    num_classes: int                        # head size
    bands: tuple[str, ...]                  # channel order of load()'s x
    splits: tuple[str, ...] = ("train", "val", "test")
    rgb_dn_max: float = 3000.0              # DN -> [0,1] for RGB/ImageNet input (S2 L1C default)
    reflectance_scale: float = 10_000.0     # DN -> reflectance for S2 SSL weights (S2 L1C default)

    @abstractmethod
    def load(self, root: str, num_workers: int = 4) -> "Data":
        """{split: (x int16 [N,C,H,W] in `bands` order, y)}; downloads + caches under root."""

    @abstractmethod
    def metrics(self, preds: "torch.Tensor", targets: "torch.Tensor") -> dict[str, float]:
        """Metric name -> value over a whole split (higher = better)."""

    def score(self, preds: "torch.Tensor", targets: "torch.Tensor") -> float:
        """Primary metric value."""
        return self.metrics(preds, targets)[self.primary_metric]

    def input_idx(self, init: Init | str) -> list[int]:
        """Channels of x fed to the model for init: RGB -> B4,B3,B2; 13ch -> all (own order); S2 SSL -> SSL4EO order."""
        init = Init(init)
        if init is Init.IMAGENET_RGB:
            return band_index(self.bands, RGB)
        if init is Init.S2_SSL:
            return band_index(self.bands, S2_SSL_ORDER)
        return list(range(len(self.bands)))

    def in_chans(self, init: Init | str) -> int:
        return len(self.input_idx(init))


@dataclass(frozen=True)
class ClassificationTask(Task):
    """Scene classification: one label per image, timm classifier head."""
    primary_metric: ClassVar[str] = "top1"

    def metrics(self, preds: "torch.Tensor", targets: "torch.Tensor") -> dict[str, float]:
        """preds = class ids. Empty split -> 0.0."""
        return {"top1": (preds == targets).sum().item() / max(1, len(targets))}

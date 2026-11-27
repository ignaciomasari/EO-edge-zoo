"""FT weights: out_dir/ckpt/<config_hash>.pt -> rebuild the exact tested model (ONNX / INT8 export)."""
import dataclasses
import json
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .config import DEFAULT_TASK, RunConfig, config_hash, recipe
from .models import InputSpec, build
from .results import Metrics


def ckpt_path(cfg: RunConfig) -> str:
    return os.path.join(cfg.out_dir, "ckpt", f"{config_hash(cfg)}.pt")


def save_checkpoint(cfg: RunConfig, model: nn.Module, spec: InputSpec, metrics: Metrics) -> str:
    """Weights (fp32, CPU) + recipe + input spec + metrics -> ckpt_path(cfg). Atomic (tmp + rename). -> path."""
    path = ckpt_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = dict(
        config_hash=config_hash(cfg),
        recipe=json.loads(json.dumps(recipe(cfg))),  # enums -> str: loadable w/ weights_only
        state_dict={name: tensor.detach().float().cpu() if tensor.is_floating_point() else tensor.detach().cpu()
                    for name, tensor in model.state_dict().items()},
        spec=dict(idx=list(spec.idx), scale=float(spec.scale), clip=bool(spec.clip),
                  mean=spec.mean.flatten().cpu(), std=spec.std.flatten().cpu()),
        metrics=dataclasses.asdict(metrics),
    )
    torch.save(payload, path + ".tmp")
    os.replace(path + ".tmp", path)
    return path


@dataclass
class Checkpoint:
    model: nn.Module          # eval mode, on device
    spec: InputSpec           # feed models.prep(x, spec, train=False, img_size=recipe["img_size"])
    recipe: dict[str, Any]
    metrics: dict[str, Any]   # as logged (top1_val, top1_test, ...)
    config_hash: str


def load_checkpoint(path: str, device: torch.device | str = "cpu") -> Checkpoint:
    """Rebuild arch from recipe (no pretrained download), load weights strictly."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    run, saved_spec = payload["recipe"], payload["spec"]
    model, _ = build(run["backbone"], run["init"], run["num_classes"], task=run.get("task", DEFAULT_TASK),
                     pretrained=False, band_mean=saved_spec["mean"], band_std=saved_spec["std"])
    model.load_state_dict(payload["state_dict"])
    spec = InputSpec(saved_spec["idx"], saved_spec["scale"], saved_spec["clip"], saved_spec["mean"], saved_spec["std"])
    return Checkpoint(model.to(device).eval(), spec.to(device), run, payload["metrics"], payload["config_hash"])

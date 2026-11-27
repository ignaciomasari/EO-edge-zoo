# Extending

Back to [README](../README.md) · see also [Usage](usage.md)

## Adding a task

A task says how to load its data and which bands it has; models adapt to it (RGB init takes B04/B03/B02, the 13-band init tiles its stem over all bands, S2 SSL reorders to the SSL4EO band order and needs all 13 Sentinel-2 bands):

```python
from dataclasses import dataclass
from eo_edge_zoo.tasks import ClassificationTask

@dataclass(frozen=True)
class MyTask(ClassificationTask):
    name: str = "my_task"
    num_classes: int = 5
    bands: tuple[str, ...] = ("B02", "B03", "B04", "B08")

    def load(self, root, num_workers=4):
        return {split: (x_int16, y) for split in self.splits}  # x: [N, C, H, W] in `bands` order
```

Register it in `TASKS` in `src/eo_edge_zoo/tasks/__init__.py` (grid runs are subprocesses, so registration must happen on import), then use `task=my_task`. `ClassificationTask` reports top-1; override `metrics()` for other metrics.

## Adding a backbone

Register it in `src/eo_edge_zoo/models.py`, with any timm architecture and the inits it supports:

```python
BACKBONES["effnet_b0"] = Backbone("efficientnet_b0", "ra_in1k", None, (Init.IMAGENET_RGB, Init.IMAGENET_13CH))
```

Then use `backbone=effnet_b0` in a config or grid; the benchmark and edge metrics pick it up too. For S2 SSL support, pass the torchgeo weights name (e.g. `"ResNet50_Weights.SENTINEL2_ALL_MOCO"`).

## Layout

| module (`src/eo_edge_zoo/`) | role |
|---|---|
| `tasks/` | `Task` / `ClassificationTask`, registry (`get_task`), `EuroSAT` |
| `config.py` | `RunConfig`, `Mode` / `Init` / `Aug` enums, `load_config`, `config_hash` |
| `bands.py`, `data.py` | Sentinel-2 band order; stratified label subsets (same subset for every model at a given fraction and seed), batching |
| `models.py` | `Backbone` registry, `build()` per init and task, GPU preprocessing + D4 augmentation |
| `train.py`, `lp.py` | one fine-tune (Lightning) or linear-probe run → `ResultRow` |
| `checkpoint.py` | save fine-tuned weights + recipe + input spec (`save_ckpt=true`); `load_checkpoint` rebuilds the tested model |
| `results.py` | `ResultRow` / `FTMetrics` / `LPMetrics`; append-only results CSV |
| `grid.py`, `tune.py` | `GridSpec`, `Rule`, `launch`; val-based recipe selection |
| `analysis.py`, `bench.py` | tables, paired deltas, Pareto plots; throughput / memory benchmark |
| `edge.py` | edge proxy metrics: params, MACs, peak activation memory, ONNX export, CPU ONNX Runtime latency, Jetson roofline estimate (`Device`, `op_costs`, `roofline_ms`) |

Tests: `pytest` (CPU only, fake data, no downloads); CI runs them with `ruff` on Python 3.11 and 3.12.

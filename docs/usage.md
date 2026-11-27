# Usage

Back to [README](../README.md) · see also [Extending](extending.md)

## Setup

```bash
pip install -e ".[geo,nb,edge,test]"    # geo = torchgeo (data download + S2 weights), nb = notebook/plots, edge = ONNX
export EO_EDGE_ZOO_WORK_DIR=/data/work  # optional: root for data cache + outputs (default: ..)
```

Data is cached under `$EO_EDGE_ZOO_WORK_DIR/data/<task>`, and results go to `$EO_EDGE_ZOO_WORK_DIR/saved_models/eo_edge_zoo/<task>`.

## Configs and single runs

`configs/default.yaml` lists every field. Overrides use `key=value` with YAML values:

```bash
python -m eo_edge_zoo.train --set mode=lp backbone=vit_s16 init=s2_ssl frac=0.01 seed=1
python -m eo_edge_zoo.train --config my.yaml --set lr=5e-5 drop_path_rate=0.1 epochs=20
```

```python
from eo_edge_zoo.config import load_config, config_hash
from eo_edge_zoo.train import run

cfg = load_config("configs/default.yaml", ["backbone=resnet50", "init=s2_ssl", "frac=0.1"])
config_hash(cfg)   # run id: depends only on recipe fields (not paths, workers, logging)
row = run(cfg)     # loads the task's data, trains, appends a ResultRow to cfg.results_csv, returns it
```

`RunConfig` validates on construction: unknown keys, an unknown `task`, invalid `mode` / `init` / `aug` values and bad types raise before anything trains. `mode` is `ft` (fine-tune) or `lp` (linear probe), and only the fields of that mode enter the hash.

## Grids

```yaml
base: default.yaml                     # relative to this file
set: {frac: 0.1, epochs: 30}           # fixed overrides for every run
grid:                                  # cartesian product
  mode: [lp, ft]
  backbone: [resnet50, vit_s16]
  init: [imagenet_rgb, s2_ssl]
  seed: [0, 1, 2]
rules:                                 # optional, per-run overrides (later wins)
  - when: {backbone: vit_s16, init: s2_ssl}
    set: {lr: 5.0e-5}
```

```bash
python -m eo_edge_zoo.grid my_grid.yaml --dry-run           # list pending runs
python -m eo_edge_zoo.grid my_grid.yaml --timeout-h 6       # run; rerun any time to resume
python -m eo_edge_zoo.grid my_grid.yaml --set epochs=2      # override for every run
```

Combinations a backbone doesn't support (e.g. ConvNeXt + S2 SSL) are dropped. Each run logs to `out_dir/logs/<hash>.log`, and failures go to `failures.csv`. From Python: `GridSpec.load(path).expand()` returns the `Run`s, and `launch(path)` runs them.

## Saving fine-tuned weights

Fine-tune runs keep no weights by default. With `save_ckpt=true`, a run saves its final weights, recipe, input normalisation and metrics to `out_dir/ckpt/<config_hash>.pt`. A ResNet-50 checkpoint is about 100 MB, so turn it on for the runs you want to export, not for sweeps. Linear-probe runs never save weights.

```bash
python -m eo_edge_zoo.train --set backbone=resnet50 init=s2_ssl frac=1.0 save_ckpt=true
python -m eo_edge_zoo.grid my_export_grid.yaml --set save_ckpt=true   # reruns finished FT runs that lack weights
```

`save_ckpt` is not part of the config hash. With it on, the grid launcher also reruns finished fine-tune runs whose checkpoint file is missing. The rerun appends a new results row, and `read_results` keeps only the last row per config hash, so seed counts stay correct.

```python
from eo_edge_zoo.checkpoint import load_checkpoint
from eo_edge_zoo.models import prep

ckpt = load_checkpoint(path)                                      # rebuilds the arch (no download), loads weights
logits = ckpt.model(prep(x_int16, ckpt.spec, False, ckpt.recipe["img_size"]))
ckpt.metrics["top1_test"]                                         # accuracy as logged
```

## Tuning recipes on validation data

```bash
python -m eo_edge_zoo.grid configs/sweep.yaml                    # sweep (e.g. lr × drop_path × wd, one seed)
python -m eo_edge_zoo.tune configs/sweep.yaml configs/grid.yaml  # best val top-1 per (backbone, init) -> tuned.yaml
python -m eo_edge_zoo.grid <out_dir>/tuned.yaml                  # grid + rules with the picks
```

Selection uses `top1_val` only. Runs whose tuned recipe equals the original share its hash and are not rerun.

## Edge metrics

```bash
python -m eo_edge_zoo.edge --out edge.csv --img 64 224          # all backbones × {3, 13} bands × image sizes
python -m eo_edge_zoo.edge --out edge.csv --threads 4 --onnx-dir onnx/   # 4 ORT threads, keep the .onnx files
taskset -c 0 python -m eo_edge_zoo.edge --out edge.csv                  # pin to one core: steadier CPU latency
```

One row per (backbone, `in_chans`, `img_size`), batch size 1, random weights (these metrics depend only on the architecture and input shape):

| column | meaning |
|---|---|
| `params_M` | parameters (millions) |
| `macs_G` | multiply-accumulates of one forward pass: conv, matmul and attention; elementwise ops ignored |
| `act_mem_MB` | peak bytes of input + activations alive at once in an eager FP32 forward pass, weights excluded. A proxy: runtimes that fuse ops or plan memory use less |
| `orin_fp16_est_ms`, `orin_int8_est_ms` | **estimated, not measured**: roofline latency on a Jetson Orin Nano Super (see below) |
| `ort_ms`, `ort_p95_ms` | median / p95 latency of the FP32 ONNX model (dynamo export, opset 18) on ONNX Runtime's CPU provider, `ort_threads` intra-op threads (default 1) |
| `cpu` | CPU model the `ort_*` columns were measured on |

If export or ONNX Runtime fails for a model, its latency is NaN, a warning is logged, and the other columns are still filled. From Python: `edge_metrics(model, in_chans, img)` for any `nn.Module`.

Which columns hold where: `params_M`, `macs_G`, `act_mem_MB` and the `orin_*_est_ms` estimates are computed, so they are identical on any machine. `ort_ms` / `ort_p95_ms` are measured, so they are only comparable between rows from the same `cpu`. Keep the machine quiet (plugged in, other apps closed) and pin one core on CPUs with mixed core types.

### Jetson latency estimate (disclaimer)

> **The `orin_*_est_ms` columns are a model-ranking aid, not Jetson measurements.** They are uncalibrated: no number in this repo has been checked against a real Jetson yet. Use them to compare architectures against each other and to shortlist models; do not quote them as device latency. Measured on-device numbers will replace them (roadmap #16).

How the estimate works: `edge.op_costs` records every kernel of one eager forward pass with its MACs and the elements it reads and writes (inputs, weights, outputs). `edge.roofline_ms` then sums, per kernel, the slower of compute time and memory time, plus a fixed launch overhead:

`time = Σ max(MACs / (peak × compute_eff), bytes / (bandwidth × bw_eff)) + overhead`

The target is `edge.ORIN_NANO_SUPER`: 8.4 T FP16 / 16.7 T INT8 dense tensor-core MAC/s, derived from NVIDIA's 67 sparse INT8 TOPS; 2.1 / 4.2 T MAC/s on CUDA cores, used for depthwise convolutions, which don't map to tensor cores; and 102 GB/s memory bandwidth. Bytes per element are 2 for FP16. For INT8, they are 1 for MAC ops and 2 for the rest (norms, softmax), which TensorRT usually keeps in FP16.

What it captures: at batch size 1 with ~100 GB/s, many layers are limited by memory traffic or launch overhead rather than MACs. That's why depthwise models such as MobileNetV3 and EfficientNet-B0 come out barely faster than ResNet-18 despite 5–10× fewer MACs, and why every model at 64 px is mostly overhead.

What it does not capture, so where it can be wrong:
- **Derates are guesses:** `compute_eff=0.5`, `bw_eff=0.7` and `overhead_s=5 µs` are not fitted to any measurement, so absolute values can be off by a large factor.
- **Fusion is only partly modelled:** views, in-place activations and eval BatchNorm cost nothing, matching TensorRT's usual folding. Every other elementwise op, norm or softmax is charged a full memory pass, whereas TensorRT fuses many of them, so the estimate overstates models with many small ops (ViTs, ConvNeXt, squeeze-excite blocks).
- **No caches, kernel choice or tiling:** small tensors that stay in the 4 MB L2 cache are still charged DRAM bandwidth, and kernel efficiency is assumed equal for every layer shape.
- **No quantization detail:** INT8 assumes every MAC layer quantizes. Layers TensorRT keeps in FP16, and the quantize/dequantize ops between precisions, are ignored.
- **Power mode:** the specs assume MAXN SUPER. Lower power modes run at lower clocks.
- **Attention:** attention is charged its Q, K, V and output traffic, like a fused kernel. Whether TensorRT fuses it on the device depends on the TensorRT version.

To see where a model's estimate comes from kernel by kernel (stage, MACs, activation vs weight bytes, output shape), run `python notebooks/roofline_data.py data.json`. It traces six backbones with the same costing as `op_costs` and writes the JSON behind the roofline explainer page.

To calibrate later: measure a handful of models on the device, check that the ranking agrees (e.g. Spearman correlation), then fit `compute_eff`, `bw_eff` and `overhead_s` with `dataclasses.replace(ORIN_NANO_SUPER, ...)`. For another device, define a new `Device` from its datasheet.

## Analysis

```python
import pandas as pd
from eo_edge_zoo import analysis
from eo_edge_zoo.results import read_results

res = analysis.tag_by_grid(read_results(csv), {"fixed": "configs/grid.yaml", "tuned": tuned_yaml})
agg = analysis.summary(res, bench, split_by=["recipe"])   # mean ± std over seeds (+ throughput)
analysis.table(agg, ["recipe"])                           # wide table: FT/LP × label fraction
analysis.paired_delta(res, "init", "s2_ssl", "imagenet_rgb", split_by=["recipe"])  # seed-paired diff
analysis.ft_lp_gap(res, split_by=["recipe"])
analysis.plot_pareto(agg, ["recipe"])

agg = analysis.summary(res, edge=pd.read_csv("edge.csv"))       # + edge metrics, matched on each run's img_size
analysis.plot_pareto(agg, x="orin_int8_est_ms", xlabel="est. Orin Nano INT8 latency (ms)", maximize_x=False)
```

With `split_by=None`, every hyperparameter that varies in the data becomes its own column, so ad-hoc sweeps are summarised without extra code. Throughput comes from `python -m eo_edge_zoo.bench --out bench.csv`, and edge metrics from `python -m eo_edge_zoo.edge`. When both are merged, `params_M` comes from the edge table, which accounts for image size (a ViT's position embeddings grow with it).

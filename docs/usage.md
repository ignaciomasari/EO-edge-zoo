# Usage

Back to [README](../README.md) · see also [Extending](extending.md)

## Setup

```bash
pip install -e ".[geo,nb,test]"         # geo = torchgeo (data download + S2 weights), nb = notebook/plots
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

## Tuning recipes on validation data

```bash
python -m eo_edge_zoo.grid configs/sweep.yaml                    # sweep (e.g. lr × drop_path × wd, one seed)
python -m eo_edge_zoo.tune configs/sweep.yaml configs/grid.yaml  # best val top-1 per (backbone, init) -> tuned.yaml
python -m eo_edge_zoo.grid <out_dir>/tuned.yaml                  # grid + rules with the picks
```

Selection uses `top1_val` only. Runs whose tuned recipe equals the original share its hash and are not rerun.

## Analysis

```python
from eo_edge_zoo import analysis
from eo_edge_zoo.results import read_results

res = analysis.tag_by_grid(read_results(csv), {"fixed": "configs/grid.yaml", "tuned": tuned_yaml})
agg = analysis.summary(res, bench, split_by=["recipe"])   # mean ± std over seeds (+ throughput)
analysis.table(agg, ["recipe"])                           # wide table: FT/LP × label fraction
analysis.paired_delta(res, "init", "s2_ssl", "imagenet_rgb", split_by=["recipe"])  # seed-paired diff
analysis.ft_lp_gap(res, split_by=["recipe"])
analysis.plot_pareto(agg, ["recipe"])
```

With `split_by=None`, every hyperparameter that varies in the data becomes its own column, so ad-hoc sweeps are summarised without extra code. Throughput comes from `python -m eo_edge_zoo.bench --out bench.csv`.

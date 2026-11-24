"""Edge proxy metrics @bs1: params, MACs, peak activation mem, CPU ONNX Runtime FP32 latency, Jetson Orin Nano
roofline estimate (uncalibrated, see docs/usage.md). `python -m eo_edge_zoo.edge --out edge.csv --img 64 224`.
Arch + input shape only (weights don't matter)."""
import argparse
import gc
import logging
import os
import platform
import tempfile
import time
import weakref
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves
from torch.utils.flop_counter import FlopCounterMode, flop_registry

from . import setup_logging
from .models import BACKBONES

log = logging.getLogger("eo_edge_zoo.edge")  # not __name__: "__main__" under `python -m`


def count_params(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def count_macs(model: nn.Module, x: torch.Tensor) -> int:
    """Multiply-accumulates of one fwd (conv, matmul, attention; elementwise ignored) = torch FLOPs / 2."""
    with torch.inference_mode(), FlopCounterMode(display=False) as counter:
        model.eval()(x)
    return counter.get_total_flops() // 2


class _LiveStorages(TorchDispatchMode):
    """Bytes of storages allocated by ops, alive at once (in-place / views reuse storage -> not counted)."""

    def __init__(self, ignore: set[int], start: int = 0):
        super().__init__()
        self.ignore, self.live, self.cur, self.peak = ignore, set(), start, start

    def _free(self, ptr: int, nbytes: int) -> None:
        self.live.discard(ptr)
        self.cur -= nbytes

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for tensor in tree_leaves(out):
            if not isinstance(tensor, torch.Tensor):
                continue
            storage = tensor.untyped_storage()
            ptr, nbytes = storage.data_ptr(), storage.nbytes()
            if ptr in self.live or ptr in self.ignore or nbytes == 0:
                continue
            self.live.add(ptr)
            self.cur += nbytes
            self.peak = max(self.peak, self.cur)
            weakref.finalize(storage, self._free, ptr, nbytes)
        return out


def peak_act_bytes(model: nn.Module, x: torch.Tensor) -> int:
    """Peak bytes of input + activations during one eager fwd (weights excluded). Proxy: no fusion / mem planning,
    op-internal workspaces not seen."""
    weights = {tensor.untyped_storage().data_ptr() for tensor in [*model.parameters(), *model.buffers()]}
    with torch.inference_mode():
        model.eval()
        tracker = _LiveStorages(weights | {x.untyped_storage().data_ptr()}, start=x.untyped_storage().nbytes())
        with tracker:
            out = model(x)
        peak = tracker.peak
        del out
    return peak


@dataclass(frozen=True)
class OpCost:
    """One kernel of an eager fwd: MACs + elements moved (inputs incl. weights + outputs)."""
    op: str
    macs: int
    float_elems: int  # float tensors (int ones, e.g. max-pool indices, not produced by TensorRT -> ignored)
    depthwise: bool   # grouped conv w/ groups == in_ch: CUDA cores, not tensor cores


def _is_free(func, args) -> bool:
    """No kernel on a TensorRT-like runtime: views / reshapes, in-place ops (activation fused into producer),
    BN (folded), allocations (no tensor input)."""
    return (func.is_view or func._schema.is_mutable or "batch_norm" in func.__name__ or "_unsafe_view" in func.__name__
            or not any(isinstance(leaf, torch.Tensor) for leaf in tree_leaves(args)))


class _OpCosts(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops: list[OpCost] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        packet = func._overloadpacket
        if packet not in flop_registry:  # composite (conv2d, linear, dropout, ...) -> re-dispatch its parts
            with self:  # mode is popped inside its own handler
                decomposed = func.decompose(*args, **kwargs)
            if decomposed is not NotImplemented:
                return decomposed
        out = func(*args, **kwargs)
        if _is_free(func, args):
            return out
        macs = flop_registry[packet](*args, **kwargs, out_val=out) // 2 if packet in flop_registry else 0
        tensors = [leaf for leaf in tree_leaves((args, kwargs, out)) if isinstance(leaf, torch.Tensor)]
        depthwise = packet is torch.ops.aten.convolution and args[8] > 1 and args[8] == args[0].shape[1]
        self.ops.append(OpCost(str(packet), macs, sum(t.numel() for t in tensors if t.is_floating_point()), depthwise))
        return out


def op_costs(model: nn.Module, x: torch.Tensor) -> list[OpCost]:
    """Kernels of one eager fwd (free ops dropped). No other fusion modelled (elementwise / norms cost a pass)."""
    with torch.inference_mode(), _OpCosts() as recorder:
        model.eval()(x)
    return recorder.ops


@dataclass(frozen=True)
class Device:
    """Roofline target: dense peaks + derates. Derates / overhead = guesses, uncalibrated."""
    name: str
    tensor_macs: dict[str, float]  # precision -> tensor-core MAC/s
    simt_macs: dict[str, float]    # precision -> CUDA-core MAC/s (depthwise convs)
    bandwidth: float               # DRAM B/s
    compute_eff: float = 0.5       # attainable fraction of MAC peak
    bw_eff: float = 0.7            # attainable fraction of bandwidth
    overhead_s: float = 5e-6       # per-kernel launch / sync


# Orin Nano Super 8 GB (MAXN SUPER): 8 SM Ampere @1.02 GHz, 67 sparse INT8 TOPS -> 16.7 T dense INT8 MAC/s,
# FP16 tensor = INT8 / 2; CUDA cores 1024 x 1.02 GHz x (2 HFMA2 | 4 DP4A); LPDDR5 102 GB/s
ORIN_NANO_SUPER = Device("orin_nano_super", tensor_macs={"fp16": 8.4e12, "int8": 16.7e12},
                         simt_macs={"fp16": 2.1e12, "int8": 4.2e12}, bandwidth=102e9)
BYTES = {"fp16": 2, "int8": 1}


def roofline_ms(ops: Sequence[OpCost], device: Device = ORIN_NANO_SUPER, precision: str = "fp16") -> float:
    """Σ max(MACs / derated peak, bytes / derated bandwidth) + overhead per kernel, in ms. INT8: MAC ops 1 B/elem,
    rest (norms, softmax, ...) stay FP16."""
    bandwidth, total_s = device.bandwidth * device.bw_eff, 0.0
    for op in ops:
        rate = (device.simt_macs if op.depthwise else device.tensor_macs)[precision] * device.compute_eff
        elem_bytes = BYTES[precision] if op.macs else BYTES["fp16"]
        total_s += max(op.macs / rate, op.float_elems * elem_bytes / bandwidth) + device.overhead_s
    return 1e3 * total_s


def cpu_name() -> str:
    """CPU model (where ort_ms was measured)."""
    try:
        with open("/proc/cpuinfo") as file:
            for line in file:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def export_onnx(model: nn.Module, x: torch.Tensor, path: str, opset: int = 18) -> str:
    """FP32 ONNX, static shape of x (dynamo exporter). -> path."""
    model = model.eval().to(memory_format=torch.contiguous_format)
    with torch.inference_mode(False):  # exporter traces, needs normal tensors
        torch.onnx.export(model, (x.contiguous(),), path, dynamo=True, opset_version=opset, verbose=False)
    return path


def ort_latency_ms(path: str, x: np.ndarray, threads: int = 1, warmup: int = 10,
                   iters: int = 100) -> tuple[float, float]:
    """(median, p95) ms per run of ONNX model at path on CPU EP, `threads` intra-op threads, sequential."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads, opts.inter_op_num_threads = threads, 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    feed = {session.get_inputs()[0].name: x}
    for _ in range(warmup):
        session.run(None, feed)
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        session.run(None, feed)
        times.append(1e3 * (time.perf_counter() - start))
    return float(np.median(times)), float(np.percentile(times, 95))


def edge_metrics(model: nn.Module, in_chans: int, img: int, name: str = "model", onnx_dir: str | None = None,
                 threads: int = 1, warmup: int = 10, iters: int = 100) -> dict[str, float]:
    """EDGE_COLS for model @bs1 (counts + roofline: CPU-independent; ort_*: this CPU). onnx_dir: keep .onnx there
    (default: temp). Export/ORT fail -> NaN latency, logged."""
    model = model.cpu().eval()
    x = torch.randn(1, in_chans, img, img)
    ops = op_costs(model, x)
    row = dict(params_M=count_params(model) / 1e6, macs_G=count_macs(model, x) / 1e9,
               act_mem_MB=peak_act_bytes(model, x) / 2**20,
               orin_fp16_est_ms=roofline_ms(ops, precision="fp16"), orin_int8_est_ms=roofline_ms(ops, precision="int8"),
               ort_ms=float("nan"), ort_p95_ms=float("nan"))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(onnx_dir or tmp, f"{name}_{in_chans}ch_{img}px.onnx")
        try:
            export_onnx(model, x, path)
            row["ort_ms"], row["ort_p95_ms"] = ort_latency_ms(path, x.numpy(), threads, warmup, iters)
        except Exception as error:  # one failing arch must not kill the table
            log.warning("ONNX export/ORT failed (%s ch, %s px): %s: %s", in_chans, img, type(error).__name__, error)
    return row


def create(arch: str, in_chans: int, img: int, num_classes: int) -> nn.Module:
    """Random-init timm model; ViT-like archs built for img (pos-embed size), others size-agnostic."""
    kwargs = dict(pretrained=False, in_chans=in_chans, num_classes=num_classes)
    try:
        return timm.create_model(arch, img_size=img, **kwargs)
    except TypeError:  # arch has no img_size arg
        return timm.create_model(arch, **kwargs)


def bench_edge(backbones: Sequence[str] | None = None, in_chans: Sequence[int] = (3, 13),
               imgs: Sequence[int] = (224,), num_classes: int = 10, onnx_dir: str | None = None,
               **edge_kwargs) -> pd.DataFrame:
    """One row per (backbone, in_chans, img_size). Random init."""
    if onnx_dir:
        os.makedirs(onnx_dir, exist_ok=True)
    rows = []
    for backbone in backbones or BACKBONES:
        for chans in in_chans:
            for img in imgs:
                model = create(BACKBONES[backbone].arch, chans, img, num_classes)
                metrics = edge_metrics(model, chans, img, name=backbone, onnx_dir=onnx_dir, **edge_kwargs)
                rows.append(dict(backbone=backbone, in_chans=chans, img_size=img, **metrics,
                                 ort_threads=edge_kwargs.get("threads", 1), cpu=cpu_name()))
                log.info("%s %sch %spx: %s", backbone, chans, img, {k: round(v, 2) for k, v in metrics.items()})
                del model
                gc.collect()
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--backbones", nargs="+", choices=list(BACKBONES), default=list(BACKBONES))
    parser.add_argument("--in-chans", nargs="+", type=int, default=[3, 13])
    parser.add_argument("--img", nargs="+", type=int, default=[224])
    parser.add_argument("--threads", type=int, default=1, help="ORT intra-op threads (1 = edge-like, stable)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--onnx-dir", help="keep exported .onnx here (default: temp, deleted)")
    args = parser.parse_args(argv)
    setup_logging()
    edge = bench_edge(args.backbones, args.in_chans, args.img, onnx_dir=args.onnx_dir, threads=args.threads,
                      warmup=args.warmup, iters=args.iters)
    edge.to_csv(args.out, index=False)
    print(edge.round(2).to_string(index=False))
    return edge


if __name__ == "__main__":
    main()

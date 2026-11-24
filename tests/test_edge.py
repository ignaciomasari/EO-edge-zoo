import math

import pandas as pd
import pytest
import torch
import torch.nn as nn

from eo_edge_zoo.analysis import EDGE_COLS
from eo_edge_zoo.edge import (
    Device,
    OpCost,
    count_macs,
    count_params,
    cpu_name,
    create,
    edge_metrics,
    main,
    op_costs,
    peak_act_bytes,
    roofline_ms,
)


def conv_relu(inplace: bool) -> nn.Module:
    return nn.Sequential(nn.Conv2d(3, 4, 1, bias=False), nn.ReLU(inplace=inplace))


X = torch.randn(1, 3, 8, 8)  # 768 B
CONV_OUT = 4 * 8 * 8 * 4     # 1024 B


def test_macs_of_conv_and_linear_match_hand_count():
    model = nn.Sequential(nn.Conv2d(3, 4, 1, bias=False), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 10))
    assert count_macs(model, X) == 3 * 4 * 64 + 4 * 10
    assert count_params(model) == 12 + 50


def test_peak_act_counts_input_plus_live_outputs_not_weights():
    # in-place ReLU reuses conv output storage; out-of-place ReLU adds a second live buffer
    assert peak_act_bytes(conv_relu(inplace=True), X) == X.nbytes + CONV_OUT
    assert peak_act_bytes(conv_relu(inplace=False), X) == X.nbytes + 2 * CONV_OUT


def test_peak_act_drops_freed_intermediates():
    # conv1 output freed after conv2 -> peak = input + max pair alive, not sum of all
    model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.Conv2d(4, 4, 1), nn.Conv2d(4, 4, 1))
    assert peak_act_bytes(model, X) == X.nbytes + 2 * CONV_OUT


def test_op_costs_drop_fused_and_folded_ops():
    # conv: in 192 + weight 12 + out 256 elems; in-place ReLU fused, eval BN folded, out-of-place ReLU = own pass
    conv = OpCost("aten.convolution", 768, 192 + 12 + 256, False)
    assert op_costs(conv_relu(inplace=True), X) == [conv]
    assert op_costs(nn.Sequential(nn.Conv2d(3, 4, 1, bias=False), nn.BatchNorm2d(4), nn.ReLU(True)), X) == [conv]
    assert op_costs(conv_relu(inplace=False), X) == [conv, OpCost("aten.relu", 0, 512, False)]


def test_op_costs_flag_depthwise_and_decompose_linear():
    ops = op_costs(nn.Sequential(nn.Conv2d(3, 3, 3, groups=3), nn.Flatten(), nn.Linear(3 * 36, 2)), X)
    assert ops[0].depthwise and not ops[-1].depthwise
    assert ops[-1].macs == 3 * 36 * 2  # linear -> addmm: counted, not skipped as composite


def test_roofline_takes_slower_of_compute_and_memory_plus_overhead():
    device = Device("toy", tensor_macs={"fp16": 2e9, "int8": 4e9}, simt_macs={"fp16": 1e9, "int8": 1e9},
                    bandwidth=1e9, compute_eff=0.5, bw_eff=1.0, overhead_s=1e-3)
    compute_bound = OpCost("conv", macs=10**6, float_elems=10, depthwise=False)        # 1 ms compute
    memory_bound = OpCost("add", macs=0, float_elems=10**6, depthwise=False)           # 2 MB fp16 -> 2 ms
    depthwise = OpCost("dwconv", macs=10**6, float_elems=10, depthwise=True)           # CUDA cores: 2 ms
    assert roofline_ms([compute_bound, memory_bound, depthwise], device) == pytest.approx(1 + 2 + 2 + 3)
    # int8: MAC op 2x faster; non-MAC op stays fp16 bytes
    assert roofline_ms([compute_bound, memory_bound], device, "int8") == pytest.approx(0.5 + 2 + 2)


def test_create_builds_vit_for_img_and_cnn_without_img_arg():
    vit, cnn = create("vit_small_patch16_224", 13, 64, 10), create("resnet18", 13, 64, 10)
    for model in (vit, cnn):
        assert model.eval()(torch.randn(1, 13, 64, 64)).shape == (1, 10)


def test_edge_metrics_ort_latency(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    row = edge_metrics(conv_relu(inplace=True), 3, 8, onnx_dir=str(tmp_path), warmup=1, iters=3)
    assert list(row) == EDGE_COLS
    assert 0 < row["ort_ms"] <= row["ort_p95_ms"]
    assert (tmp_path / "model_3ch_8px.onnx").is_file()


def test_edge_metrics_export_failure_gives_nan_latency_not_crash(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("unsupported op")
    monkeypatch.setattr("eo_edge_zoo.edge.export_onnx", boom)
    row = edge_metrics(conv_relu(inplace=True), 3, 8)
    assert math.isnan(row["ort_ms"]) and row["macs_G"] > 0


def test_cli_writes_edge_csv(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    out = tmp_path / "edge.csv"
    main(["--out", str(out), "--backbones", "resnet50", "--in-chans", "13", "--img", "32", "64",
          "--warmup", "1", "--iters", "2"])
    df = pd.read_csv(out)
    assert list(df.columns) == ["backbone", "in_chans", "img_size", *EDGE_COLS, "ort_threads", "cpu"]
    assert (df.cpu == cpu_name()).all() and cpu_name()
    assert (df.orin_int8_est_ms < df.orin_fp16_est_ms).all()
    assert df.img_size.tolist() == [32, 64]
    assert df.macs_G.iloc[1] == pytest.approx(4 * df.macs_G.iloc[0], rel=0.01)  # MACs ∝ pixels for a CNN
    assert (df.ort_ms > 0).all()

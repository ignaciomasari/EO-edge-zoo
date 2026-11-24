"""Per-kernel data for the roofline explainer page: `python notebooks/roofline_data.py data.json`.
Same kernels + costs as edge.op_costs (subclass), plus stage, kind, act/weight split, output shape."""
import json
import sys

import torch
from torch.utils._pytree import tree_leaves

from eo_edge_zoo.edge import _OpCosts, create

MODELS = [("resnet18", "ResNet-18", "resnet18"), ("resnet50", "ResNet-50", "resnet50"),
          ("vit_s16", "ViT-S/16", "vit_small_patch16_224"), ("convnext_t", "ConvNeXt-T", "convnext_tiny"),
          ("mobilenetv3", "MobileNetV3-L", "mobilenetv3_large_100"), ("effnet_b0", "EfficientNet-B0", "efficientnet_b0")]
IN_CHANS, IMGS = 13, (224, 64)
CONTAINERS = ("blocks", "stages", "layers")  # timm stage containers -> report child (e.g. blocks.3)


class _Annotated(_OpCosts):
    """edge._OpCosts + per-op extras; module path stack from global fwd hooks."""

    def __init__(self, weight_ptrs: set[int]):
        super().__init__()
        self.weight_ptrs, self.module_stack, self.extras = weight_ptrs, [], []

    def stage(self) -> str:
        parts = [name for name in self.module_stack if name]
        if not parts:
            return "embed"  # ops in model.forward itself (cls token, pos embed)
        if parts[0] in CONTAINERS:
            return ".".join(parts[1].split(".")[:2]) if len(parts) > 1 else parts[0]
        return parts[0]

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = super().__torch_dispatch__(func, types, args, kwargs)
        if len(self.ops) > len(self.extras):  # this op recorded (composites: children already annotated)
            op = self.ops[-1]
            floats = [t for t in tree_leaves((args, kwargs or {})) if isinstance(t, torch.Tensor) and t.is_floating_point()]
            weights = sum(t.numel() for t in floats if t.untyped_storage().data_ptr() in self.weight_ptrs)
            outs = [t for t in tree_leaves(out) if isinstance(t, torch.Tensor) and t.is_floating_point()]
            name = op.op.replace("aten.", "")
            kind = ("glue" if not op.macs else "dw" if op.depthwise else "conv" if name == "convolution"
                    else "attn" if "attention" in name else "mm")
            self.extras.append(dict(op=name, stage=self.stage(), kind=kind, macs=op.macs, w=weights,
                                    a=op.float_elems - weights, shape=list(outs[0].shape) if outs else []))
        return out


def trace(arch: str, img: int) -> list[dict]:
    model = create(arch, IN_CHANS, img, 10).eval()
    names = {id(module): name for name, module in model.named_modules()}
    recorder = _Annotated({t.untyped_storage().data_ptr() for t in [*model.parameters(), *model.buffers()]})
    hooks = [torch.nn.modules.module.register_module_forward_pre_hook(
                 lambda module, inputs: recorder.module_stack.append(names.get(id(module), ""))),
             torch.nn.modules.module.register_module_forward_hook(
                 lambda module, inputs, output: (recorder.module_stack.pop(), None)[1])]  # None: keep output
    try:
        with torch.inference_mode(), recorder:
            model(torch.randn(1, IN_CHANS, img, img))
    finally:
        for hook in hooks:
            hook.remove()
    return recorder.extras


def main(out_path: str) -> None:
    data = {f"{key}@{img}": dict(label=label, img=img, ops=trace(arch, img)) for key, label, arch in MODELS for img in IMGS}
    with open(out_path, "w") as file:
        json.dump(data, file, separators=(",", ":"))
    for key, entry in data.items():
        print(f"{key:16s} {len(entry['ops']):4d} kernels  {sum(op['macs'] for op in entry['ops']) / 1e9:.2f} GMACs")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data.json")

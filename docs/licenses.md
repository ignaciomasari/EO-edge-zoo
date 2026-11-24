# Licenses

Back to [README](../README.md)

Code: [Apache-2.0](../LICENSE). Pretrained weights and datasets keep their own licenses:

| item | license | source | note |
|---|---|---|---|
| timm `resnet50.a1_in1k` | Apache-2.0 | [model card](https://huggingface.co/timm/resnet50.a1_in1k) | ImageNet-trained, see below |
| timm `vit_small_patch16_224.augreg_in21k_ft_in1k` | Apache-2.0 | [model card](https://huggingface.co/timm/vit_small_patch16_224.augreg_in21k_ft_in1k) | ImageNet-trained, see below |
| timm `convnext_tiny.fb_in22k` | Apache-2.0 (timm card); MIT (original repo) | [model card](https://huggingface.co/timm/convnext_tiny.fb_in22k), [ConvNeXt](https://github.com/facebookresearch/ConvNeXt/blob/main/LICENSE) | ImageNet-trained, see below |
| SSL4EO-S12 MoCo ResNet-50 / ViT-S/16 (torchgeo `SENTINEL2_ALL_MOCO`) | CC-BY-4.0 | [torchgeo weights table](https://github.com/torchgeo/torchgeo/blob/main/docs/api/weights/sentinel2.csv), [SSL4EO-S12](https://github.com/zhu-xlab/SSL4EO-S12) | attribution required: cite [arXiv:2211.07044](https://arxiv.org/abs/2211.07044) |
| EuroSAT | MIT | [EuroSAT](https://github.com/phelber/EuroSAT) | Copernicus Sentinel data terms also apply |
| timm (library) | Apache-2.0 | [timm](https://github.com/huggingface/pytorch-image-models/blob/main/LICENSE) | |
| torchgeo (library) | MIT | [torchgeo](https://github.com/torchgeo/torchgeo/blob/main/LICENSE) | |

**ImageNet-trained weights:** ImageNet was released for non-commercial research only, and the timm README advises assuming its terms carry over to weights trained on it ([timm licenses](https://github.com/huggingface/pytorch-image-models#licenses)). Check before using those weights commercially.

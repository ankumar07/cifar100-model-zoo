# CIFAR-100 Model Zoo

This repo is a small "model zoo" for CIFAR-100 experiments on macOS (Apple Silicon / M1 Pro) and other platforms.

It supports multiple architectures trained with a common pipeline:

- ResNet-18
- ResNet-34
- WideResNet-28x10
- ConvNeXt-Tiny (timm)
- ViT / DeiT Tiny (timm)

The goal is to compare models on:
- Top-1 / Top-5 accuracy
- Calibration quality (ECE, reliability diagrams)
- Per-class performance (confusion matrix, worst classes, etc.)

---

## 1. Project Structure

```text
cifar100-model-zoo/
├─ experiments/
│  ├─ cifar100_capstone.py   # main training/eval script
│  └─ utils.py               # helpers: device selection, metrics, plots, dirs
├─ data/                     # CIFAR-100 is downloaded here by torchvision
└─ outputs/
   └─ <run_name>/
      ├─ checkpoints/        # .pt checkpoints (best model etc.)
      ├─ plots/              # training curves, confusion matrix, reliability
      └─ results/            # JSON metrics, ECE bins, confusion matrix .npy

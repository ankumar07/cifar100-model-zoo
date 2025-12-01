"""Utility functions for CIFAR-100 training.

Includes:
- seeding helpers
- accuracy / calibration metrics
- plotting helpers
- device / output directory helpers
"""

import os
import random
from typing import Tuple, List, Dict
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Only attempt CUDA seeding if CUDA is actually available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For full determinism (slower):
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def topk_accuracy(logits: torch.Tensor, target: torch.Tensor, ks=(1, 5)) -> List[float]:
    maxk = max(ks)
    with torch.no_grad():
        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in ks:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append((correct_k / target.size(0)).item())
        return res


def compute_ece(confidences: np.ndarray, correctness: np.ndarray, n_bins: int = 15) -> Tuple[float, Dict]:
    """
    confidences: (N,) predicted confidence (prob of predicted class)
    correctness: (N,) 1 if correct else 0
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_stats = {
        "bin_left": [],
        "bin_right": [],
        "bin_acc": [],
        "bin_conf": [],
        "bin_count": [],
    }
    N = len(confidences)
    for i in range(n_bins):
        l, r = bins[i], bins[i + 1]
        idx = (confidences > l) & (confidences <= r) if i > 0 else (confidences >= l) & (confidences <= r)
        if idx.sum() > 0:
            acc = correctness[idx].mean()
            conf = confidences[idx].mean()
            ece += (idx.sum() / N) * abs(acc - conf)
            bin_stats["bin_left"].append(float(l))
            bin_stats["bin_right"].append(float(r))
            bin_stats["bin_acc"].append(float(acc))
            bin_stats["bin_conf"].append(float(conf))
            bin_stats["bin_count"].append(int(idx.sum()))
        else:
            bin_stats["bin_left"].append(float(l))
            bin_stats["bin_right"].append(float(r))
            bin_stats["bin_acc"].append(0.0)
            bin_stats["bin_conf"].append(0.0)
            bin_stats["bin_count"].append(0)
    return float(ece), bin_stats


def plot_training_curves(history: Dict[str, List[float]], out_path: str, title: str):
    # Accuracy
    plt.figure(figsize=(7, 5))
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title} - Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "training_accuracy.png"), dpi=150)
    plt.close()

    # Loss
    plt.figure(figsize=(7, 5))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} - Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "training_loss.png"), dpi=150)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, out_path: str, title: str):
    plt.figure(figsize=(9, 8))
    im = plt.imshow(cm, interpolation="nearest", aspect="auto")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "confusion_matrix.png"), dpi=160)
    plt.close()


def plot_per_class_accuracy(cm: np.ndarray, out_path: str, title: str, topn: int = 25):
    per_class = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    # Show lowest topn classes
    idx_sorted = np.argsort(per_class)
    worst_idx = idx_sorted[:topn]
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(topn), per_class[worst_idx])
    plt.ylim(0, 1.0)
    plt.xticks(np.arange(topn), [str(i) for i in worst_idx], rotation=90)
    plt.ylabel("Per-class accuracy")
    plt.title(f"{title} - {topn} Lowest Classes")
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "per_class_accuracy_worst.png"), dpi=150)
    plt.close()


def plot_reliability(bin_stats: Dict, out_path: str, title: str):
    left = np.array(bin_stats["bin_left"])
    right = np.array(bin_stats["bin_right"])
    acc = np.array(bin_stats["bin_acc"])
    conf = np.array(bin_stats["bin_conf"])
    centers = (left + right) / 2.0
    width = (right - left) * 0.9

    plt.figure(figsize=(6.5, 6))
    # Bars show accuracy in each confidence bin; diagonal is perfect calibration
    plt.bar(
        centers,
        acc,
        align="center",
        width=width,
        edgecolor="black",
        alpha=0.7,
        label="Accuracy per bin",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Perfect calibration")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"{title} - Reliability Diagram")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "reliability_diagram.png"), dpi=160)
    plt.close()


def get_device() -> torch.device:
    """
    Selects the best available device.

    On Apple Silicon (M1/M2) with a recent PyTorch this will prefer the
    Metal Performance Shaders (MPS) backend, which lets you use the
    Apple GPU. Otherwise it falls back to CUDA or CPU.
    """
    # Apple GPU (MPS)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using Apple M1/M2 GPU via torch.backends.mps (MPS).")
        return torch.device("mps")
    # NVIDIA GPU (CUDA)
    if torch.cuda.is_available():
        print("Using NVIDIA GPU via CUDA.")
        return torch.device("cuda")
    # CPU fallback
    print("Using CPU.")
    return torch.device("cpu")


def prepare_output_dirs(run_name: str, base_dir: str = "outputs"):
    """
    Creates an experiment directory structure like:

        outputs/<run_name>/
            checkpoints/
            plots/
            results/

    Returns (out_dir, ckpt_dir, plots_dir, results_dir) as *strings*.
    """
    base = Path(base_dir)
    out_dir = base / run_name
    ckpt_dir = out_dir / "checkpoints"
    plots_dir = out_dir / "plots"
    results_dir = out_dir / "results"

    for d in (out_dir, ckpt_dir, plots_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    return str(out_dir), str(ckpt_dir), str(plots_dir), str(results_dir)

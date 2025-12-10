# cifar100_capstone.py
# CIFAR-100 training for ResNet / WideResNet-28x10 / ConvNeXt-Tiny / ViT-Tiny
# Saves: checkpoints, training curves, confusion matrix, per-class accuracy,
# reliability diagram, JSON metrics and a train.log file per run.

import os
import json
import math
import argparse
import time
import logging
from copy import deepcopy
from datetime import datetime
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix

from utils import (
    seed_everything,
    topk_accuracy,
    compute_ece,
    plot_training_curves,
    plot_confusion_matrix,
    plot_per_class_accuracy,
    plot_reliability,
    get_device,
    prepare_output_dirs,
)

import timm
import matplotlib

matplotlib.use("Agg")

VALID_MODELS = [
    "resnet18",
    "resnet34",
    "resnet50",
    "wrn28x10",
    "convnext_tiny",
    "vit_tiny",
    "deit_tiny",
]


# ---------------------------
# WideResNet (28x10) for CIFAR-style data (works with any input size via AdaptiveAvgPool)
# ---------------------------


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super().__init__()
        self.equalInOut = in_planes == out_planes
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.drop_rate = drop_rate
        self.convShortcut = None
        if not self.equalInOut:
            self.convShortcut = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        if self.equalInOut:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        if not self.equalInOut:
            return out + self.convShortcut(x)
        else:
            return out + x


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super().__init__()
        layers = []
        for i in range(nb_layers):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    drop_rate,
                )
            )
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(
            self,
            depth=28,
            widen_factor=10,
            num_classes=100,
            drop_rate=0.0,
    ):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        k = widen_factor
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(
            3,
            nStages[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.block1 = NetworkBlock(
            n, nStages[0], nStages[1], BasicBlock, 1, drop_rate
        )
        self.block2 = NetworkBlock(
            n, nStages[1], nStages[2], BasicBlock, 2, drop_rate
        )
        self.block3 = NetworkBlock(
            n, nStages[2], nStages[3], BasicBlock, 2, drop_rate
        )
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(nStages[3], num_classes)
        self.nStages = nStages

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# ---------------------------
# Model factory
# ---------------------------


def create_model(
        name: str,
        num_classes: int = 100,
        pretrained: bool = False,
        drop_rate: float = 0.0
) -> Tuple[nn.Module, int]:
    """
    Returns (model, img_size).
    """
    name = name.lower()

    if name in ["resnet18", "resnet34", "resnet50"]:
        if name == "resnet18":
            model = torchvision.models.resnet18(weights=None)
        elif name == "resnet34":
            model = torchvision.models.resnet34(weights=None)
        else:
            model = torchvision.models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        img_size = 224

    elif name == "wrn28x10":
        model = WideResNet(
            depth=28,
            widen_factor=10,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
        img_size = 32

    elif name == "convnext_tiny":
        model = timm.create_model(
            "convnext_tiny",
            pretrained=pretrained,
            num_classes=num_classes,
        )
        img_size = model.default_cfg.get("input_size", (3, 224, 224))[1]

    elif name == "vit_tiny":
        model = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )
        img_size = 224

    elif name == "deit_tiny":
        model = timm.create_model(
            "deit_tiny_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )
        img_size = 224

    else:
        raise ValueError(f"Unknown model: {name}")

    return model, img_size


# ---------------------------
# Data
# ---------------------------


def get_transforms(img_size: int):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # stronger aug: RandAugment
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std=(0.2675, 0.2565, 0.2761),
            ),
            # Cutout-ish: RandomErasing operates on the tensor
            transforms.RandomErasing(
                p=0.5,
                scale=(0.02, 0.2),  # rough "cutout" size range
                ratio=(0.3, 3.3),
                inplace=False,
            ),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std=(0.2675, 0.2565, 0.2761),
            ),
        ]
    )
    return train_transform, test_transform


def get_dataloaders(
        data_root: str,
        img_size: int,
        batch_size: int,
        val_split: float,
        num_workers: int,
        seed: int = 42,
):
    train_tf, test_tf = get_transforms(img_size)

    full_train = datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=train_tf
    )
    test_set = datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=test_tf
    )

    n_train = len(full_train)
    n_val = int(val_split * n_train)
    n_train_ = n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        full_train, [n_train_, n_val], generator=generator
    )

    dl_train = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    dl_val = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    dl_test = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dl_train, dl_val, dl_test


# ---------------------------
# Implement mixup
# ---------------------------
def mixup_data(x, y, alpha=0.2, device="cuda"):
    if alpha <= 0:
        return x, y, None, None

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


#Loss helper
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ---------------------------
# Train / Eval loops
# ---------------------------


def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
):
    model.train()
    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0
    n_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        # Mixed precision is only enabled on CUDA; on MPS/CPU this is automatically off
        # with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        #     logits = model(images)
        #     loss = criterion(logits, targets)

        #Implement mixup
        # apply mixup
        if hasattr(args, "mixup_alpha") and args.mixup_alpha > 0:
            images, targets_a, targets_b, lam = mixup_data(images, targets, alpha=args.mixup_alpha, device=device)
        else:
            targets_a, targets_b, lam = targets, None, None

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(images)
            if lam is not None:
                loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
            else:
                loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        n_samples += bs
        running_loss += loss.item() * bs
        top1, top5 = topk_accuracy(logits, targets, ks=(1, 5))
        running_top1 += top1 * bs
        running_top5 += top5 * bs

    epoch_loss = running_loss / n_samples
    epoch_top1 = running_top1 / n_samples
    epoch_top5 = running_top5 / n_samples

    return epoch_loss, epoch_top1, epoch_top5


def evaluate(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        criterion: nn.Module,
):
    model.eval()
    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0
    n_samples = 0

    all_conf = []
    all_pred = []
    all_true = []
    all_corr = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, targets)

            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            correct = pred.eq(targets)

            bs = images.size(0)
            n_samples += bs
            running_loss += loss.item() * bs
            top1, top5 = topk_accuracy(logits, targets, ks=(1, 5))
            running_top1 += top1 * bs
            running_top5 += top5 * bs

            all_conf.append(conf.detach().cpu().numpy())
            all_pred.append(pred.detach().cpu().numpy())
            all_true.append(targets.detach().cpu().numpy())
            all_corr.append(correct.detach().cpu().numpy().astype(np.float32))

    epoch_loss = running_loss / n_samples
    epoch_top1 = running_top1 / n_samples
    epoch_top5 = running_top5 / n_samples

    confidences = np.concatenate(all_conf)
    preds = np.concatenate(all_pred)
    trues = np.concatenate(all_true)
    correctness = np.concatenate(all_corr)

    return (
        epoch_loss,
        epoch_top1,
        epoch_top5,
        confidences,
        preds,
        trues,
        correctness,
    )


# ---------------------------
# Per-model training run (with logging)
# ---------------------------


def run_experiment_for_model(model_name: str, base_args):
    # Copy args so per-model runs don't clobber each other
    args = deepcopy(base_args)
    args.model = model_name

    # Seed & device
    seed_everything(args.seed)
    device = get_device()

    # Model
    model, img_size = create_model(
        model_name,
        num_classes=100,
        pretrained=args.pretrained,
        drop_rate=args.drop_rate
    )
    model = model.to(device)

    # Dataloaders
    dl_train, dl_val, dl_test = get_dataloaders(
        data_root=args.data_root,
        img_size=img_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Loss
    criterion = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing
    ).to(device)

    # Optimizer choice
    if args.optimizer is None:
        if model_name.startswith("resnet") or model_name == "wrn28x10":
            opt_name = "sgd"
        else:
            opt_name = "adamw"
    else:
        opt_name = args.optimizer

    # LR / WD defaults by optimizer
    if args.lr is None:
        if opt_name == "sgd":
            lr = 0.1
        else:
            lr = 3e-4
    else:
        lr = args.lr

    if args.weight_decay is None:
        if opt_name == "sgd":
            wd = 5e-4
        else:
            wd = 0.05
    else:
        wd = args.weight_decay

    # Optimizer
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=wd,
            nesterov=True,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
        )

    # Cosine schedule with linear warmup
    # Cosine LR schedule with linear warmup (epoch-based)
    warmup_epochs = max(1, int(0.03 * args.epochs))

    def cosine_lr_lambda(epoch: int) -> float:
        """
        epoch: 0, 1, ..., args.epochs-1
        Returns a multiplier for the base LR.
        """
        # epoch counting in the function is 0-based.
        # We'll treat epoch+1 as the current "1-based" progress.
        current_epoch = epoch + 1

        if current_epoch <= warmup_epochs:
            # Linear warmup from 1/warmup_epochs to 1.0
            return current_epoch / float(max(1, warmup_epochs))

        # Cosine decay from warmup_epochs -> args.epochs
        progress = float(current_epoch - warmup_epochs) / float(
            max(1, args.epochs - warmup_epochs)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=cosine_lr_lambda
    )

    # Output directories & logging
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = (
            args.run_name
            or f"{model_name}_e{args.epochs}_bs{args.batch_size}_{timestamp}"
    )
    out_dir, ckpt_dir, plots_dir, results_dir = prepare_output_dirs(run_name)

    # Set up logger: to file + console
    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers if called multiple times
    logger.handlers.clear()

    log_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # File handler
    fh = logging.FileHandler(os.path.join(out_dir, "train.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(log_formatter)
    logger.addHandler(fh)
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(log_formatter)
    logger.addHandler(ch)

    # Log run config
    logger.info(
        f"==> Starting training for model '{model_name}' "
        f"on CIFAR-100 (img_size={img_size}), device={device}"
    )
    logger.info(f"Run name: {run_name}")
    logger.info(f"Outputs will be saved under: {out_dir}")
    logger.info(f"Optimizer={opt_name}, lr={lr}, weight_decay={wd}")
    logger.info(f"Label smoothing={args.label_smoothing}, epochs={args.epochs}")
    if model_name.lower() == "wrn28x10":
        logger.info(f"Dropout rate (WideResNet blocks)={args.drop_rate}")
    logger.info(
        f"Dataset sizes: train={len(dl_train.dataset)}, "
        f"val={len(dl_val.dataset)}, test={len(dl_test.dataset)}"
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters: {n_params:,}")

    # Train
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    best_val = -1.0
    best_path = os.path.join(ckpt_dir, f"{model_name}_best.pt")
    #global_step = 0

    try:
        for epoch in range(args.epochs):
            t0 = time.time()
            train_loss, train_top1, train_top5 = train_one_epoch(
                model,
                dl_train,
                device,
                criterion,
                optimizer,
                scaler,
            )
            val_loss, val_top1, val_top5, *_ = evaluate(
                model, dl_val, device, criterion
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_top1)
            history["val_acc"].append(val_top1)

            scheduler.step()
            #global_step += len(dl_train)

            is_best = val_top1 > best_val
            if is_best:
                best_val = val_top1
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "val_top1": val_top1,
                    },
                    best_path,
                )

            dt = time.time() - t0
            logger.info(
                f"Epoch {epoch + 1:03d}/{args.epochs:03d} "
                f"- {dt:6.1f}s - "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"train_top1={train_top1:.4f}, val_top1={val_top1:.4f}"
            )

        # Load best checkpoint
        logger.info(f"Loading best checkpoint from: {best_path}")
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])

        # Final eval on validation + test
        (
            val_loss,
            val_top1,
            val_top5,
            val_conf,
            val_pred,
            val_true,
            val_corr,
        ) = evaluate(model, dl_val, device, criterion)
        (
            test_loss,
            test_top1,
            test_top5,
            test_conf,
            test_pred,
            test_true,
            test_corr,
        ) = evaluate(model, dl_test, device, criterion)

        # Confusion Matrix (test)
        cm = confusion_matrix(test_true, test_pred, labels=list(range(100)))

        # ECE + reliability (test)
        ece, bin_stats = compute_ece(test_conf, test_corr, n_bins=15)

        # Save metrics
        metrics = {
            "model": model_name,
            "run_name": run_name,
            "img_size": img_size,
            "val_loss": float(val_loss),
            "val_top1": float(val_top1),
            "val_top5": float(val_top5),
            "test_loss": float(test_loss),
            "test_top1": float(test_top1),
            "test_top5": float(test_top5),
            "ece": float(ece),
            "best_val_top1": float(best_val),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        }
        with open(os.path.join(results_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # Save confusion matrix & bin stats
        with open(os.path.join(results_dir, "reliability_bins.json"), "w") as f:
            json.dump(bin_stats, f, indent=2)
        np.save(os.path.join(results_dir, "confusion_matrix.npy"), cm)

        # Plots
        plot_training_curves(history, plots_dir, title=model_name.upper())
        plot_confusion_matrix(
            cm,
            plots_dir,
            title=f"{model_name.upper()} - Confusion Matrix (Test)",
        )
        plot_per_class_accuracy(
            cm,
            plots_dir,
            title=model_name.upper(),
            topn=25,
        )
        plot_reliability(bin_stats, plots_dir, title=model_name.upper())

        logger.info("==> Finished training and evaluation.")
        logger.info("Final metrics:\n" + json.dumps(metrics, indent=2))
        logger.info(f"Outputs saved under: {out_dir}")
        logger.info(f"- Best checkpoint: {best_path}")
        logger.info(f"- Plots: {plots_dir}")
        logger.info(f"- Metrics/CM/ECE: {results_dir}")

    except Exception:
        # Log full stack trace for debugging
        logger.exception("Unhandled exception during training run.")
        # Re-raise so you still see it in the console / debugger
        raise


# ---------------------------
# Main: multi-model orchestration with confirmation
# ---------------------------


def main():
    parser = argparse.ArgumentParser(description="CIFAR-100 Capstone Trainer")

    # You can use either --model (single) or --models (multiple)
    parser.add_argument(
        "--model",
        type=str,
        choices=VALID_MODELS,
        help="Single model architecture",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=VALID_MODELS,
        help="One or more models to train sequentially. "
             "You will be asked to confirm each one.",
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Where to download/load CIFAR-100",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="If None, auto-select per optimizer/model",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="If None, auto-select per optimizer/model",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=["sgd", "adamw"],
        help="If None, choose by model",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained weights when available (timm/torchvision)",
    )
    parser.add_argument(
        "--drop-rate",
        type=float,
        default=0.0,
        help="Dropout rate for WideResNet blocks",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default=None)

    args = parser.parse_args()

    # Decide which models to run
    if args.model is None and args.models is None:
        parser.error("You must specify either --model or --models.")
    if args.model is not None and args.models is not None:
        parser.error("Use either --model or --models, not both.")

    if args.model is not None:
        model_names = [args.model]
    else:
        model_names = args.models

    # Confirmation loop per model
    for model_name in model_names:
        answer = input(
            f"\nDo you want to start training model '{model_name}'? [y/N]: "
        )
        if answer.strip().lower() not in ("y", "yes"):
            print(f"Skipping model '{model_name}'.")
            continue

        print(f"Starting training for model '{model_name}'...\n")
        try:
            run_experiment_for_model(model_name, args)
        except Exception as e:
            # Top-level catch so that one failing model doesn't kill the whole script
            print(
                f"[ERROR] Training for model '{model_name}' failed: {e}. "
                f"Check the corresponding train.log (if created) for details."
            )


if __name__ == "__main__":
    main()

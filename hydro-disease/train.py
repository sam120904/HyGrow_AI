import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch  # pyrefly: ignore[missing-import]
import torch.nn as nn  # pyrefly: ignore[missing-import]
from torch.utils.data import DataLoader  # pyrefly: ignore[missing-import]
from torchvision import models  # pyrefly: ignore[missing-import]
from tqdm import tqdm  # pyrefly: ignore[missing-import]
from dataset import AlbumentationsDataset, TRAIN_TRANSFORM, VAL_TRANSFORM  # pyrefly: ignore[missing-import]

# ── CONFIG ──────────────────────────────────
BASE = Path(__file__).resolve().parent
CLASS_NAMES = json.loads((BASE / "class_names.json").read_text(encoding="utf-8"))
NUM_CLASSES = len(CLASS_NAMES)
BATCH_SIZE  = 64       # safe for 8GB VRAM; use 32 if OOM
EPOCHS_HEAD = 5        # train classifier head only first
EPOCHS_FULL = 20       # then unfreeze all layers
LR_HEAD     = 1e-3
LR_FULL     = 1e-4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, criterion, optimizer, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            if train: optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
    return total_loss / total, correct / total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune MobileNetV3 on prepared PlantVillage data")
    parser.add_argument("--data-dir", type=Path, default=BASE / "data" / "plantvillage-hf")
    parser.add_argument("--checkpoint", type=Path, default=BASE / "models" / "checkpoints" / "mobilenetv3_best.pth")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs-head", type=int, default=EPOCHS_HEAD)
    parser.add_argument("--epochs-full", type=int, default=EPOCHS_FULL)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"Training on: {DEVICE}")

    # ── DATA ────────────────────────────────────
    train_ds = AlbumentationsDataset(args.data_dir / "train", TRAIN_TRANSFORM)
    val_ds   = AlbumentationsDataset(args.data_dir / "val", VAL_TRANSFORM)
    if train_ds.classes != CLASS_NAMES or val_ds.classes != CLASS_NAMES:
        raise ValueError("Dataset class order differs from class_names.json")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=DEVICE.type == "cuda")
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=DEVICE.type == "cuda")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    # ── MODEL ───────────────────────────────────
    model = models.mobilenet_v3_large(weights="IMAGENET1K_V2")
    # Replace classifier head
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    # ── PHASE A: Freeze backbone, train head only ──
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR_HEAD)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print("Phase A — training head")
    best_val_acc = -1.0
    for ep in range(args.epochs_head):
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, optimizer, train=True)
        vl_loss, vl_acc = run_epoch(model, val_dl, criterion, optimizer, train=False)
        print(f"Ep {ep+1}/{args.epochs_head} | train acc {tr_acc:.3f} | val acc {vl_acc:.3f}")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), args.checkpoint)

    # ── PHASE B: Unfreeze all, fine-tune ──
    print("\nPhase B — full fine-tune")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_FULL, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs_full))

    for ep in range(args.epochs_full):
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, optimizer, train=True)
        vl_loss, vl_acc = run_epoch(model, val_dl, criterion, optimizer, train=False)
        scheduler.step()
        print(f"Ep {ep+1}/{args.epochs_full} | train {tr_acc:.3f} | val {vl_acc:.3f}")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), args.checkpoint)
            print(f"  ✓ saved checkpoint (val acc {vl_acc:.3f})")

    print(f"\nDone. Best val accuracy: {best_val_acc:.3f}")

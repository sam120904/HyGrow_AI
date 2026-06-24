import torch  # pyrefly: ignore[missing-import]
import torch.nn as nn  # pyrefly: ignore[missing-import]
from torch.utils.data import DataLoader  # pyrefly: ignore[missing-import]
from torchvision import models  # pyrefly: ignore[missing-import]
from tqdm import tqdm  # pyrefly: ignore[missing-import]
from dataset import AlbumentationsDataset, TRAIN_TRANSFORM, VAL_TRANSFORM  # pyrefly: ignore[missing-import]

# ── CONFIG ──────────────────────────────────
NUM_CLASSES = 15       # actual class folders in data/processed/train
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
    print(f"Training on: {DEVICE}")

    # ── DATA ────────────────────────────────────
    train_ds = AlbumentationsDataset("data/processed/train", TRAIN_TRANSFORM)
    val_ds   = AlbumentationsDataset("data/processed/val",   VAL_TRANSFORM)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

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
    for ep in range(EPOCHS_HEAD):
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, optimizer, train=True)
        vl_loss, vl_acc = run_epoch(model, val_dl, criterion, optimizer, train=False)
        print(f"Ep {ep+1}/{EPOCHS_HEAD} | train acc {tr_acc:.3f} | val acc {vl_acc:.3f}")

    # ── PHASE B: Unfreeze all, fine-tune ──
    print("\nPhase B — full fine-tune")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_FULL, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_FULL)

    best_val_acc = 0
    for ep in range(EPOCHS_FULL):
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, optimizer, train=True)
        vl_loss, vl_acc = run_epoch(model, val_dl, criterion, optimizer, train=False)
        scheduler.step()
        print(f"Ep {ep+1}/{EPOCHS_FULL} | train {tr_acc:.3f} | val {vl_acc:.3f}")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), "models/checkpoints/mobilenetv3_best.pth")
            print(f"  ✓ saved checkpoint (val acc {vl_acc:.3f})")

    print(f"\nDone. Best val accuracy: {best_val_acc:.3f}")
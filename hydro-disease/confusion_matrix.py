"""Generate a visual confusion matrix from the test set."""
import json
import torch  # pyrefly: ignore[missing-import]
import torch.nn as nn  # pyrefly: ignore[missing-import]
import numpy as np  # pyrefly: ignore[missing-import]
import matplotlib.pyplot as plt  # pyrefly: ignore[missing-import]
from torch.utils.data import DataLoader  # pyrefly: ignore[missing-import]
from torchvision import models  # pyrefly: ignore[missing-import]
from pathlib import Path
from dataset import AlbumentationsDataset, VAL_TRANSFORM  # pyrefly: ignore[missing-import]

BASE_DIR = Path(__file__).resolve().parent
CKPT_PATH = BASE_DIR / "models" / "checkpoints" / "mobilenetv3_best.pth"
NUM_CLASSES = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Load model
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    # Load test data
    test_ds = AlbumentationsDataset(str(BASE_DIR / "data" / "plantvillage-hf" / "test"), VAL_TRANSFORM)
    class_names = test_ds.classes
    expected_names = json.loads((BASE_DIR / "class_names.json").read_text(encoding="utf-8"))
    if class_names != expected_names:
        raise ValueError("Dataset class order differs from class_names.json")
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Collect predictions
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            preds = model(imgs).argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    # Build confusion matrix
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t][p] += 1

    # Normalize to percentages (row-wise = recall per class)
    cm_pct = np.divide(cm.astype(float), cm.sum(axis=1, keepdims=True),
                       out=np.zeros_like(cm, dtype=float), where=cm.sum(axis=1, keepdims=True) != 0) * 100

    # Shorten class names for display
    short_names = [name.replace("___", ": ").replace("__", ": ").replace("_", " ") for name in class_names]

    # Plot
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix — Test Set (Recall %)", fontsize=14, fontweight="bold")

    # Add percentage annotations
    for i in range(n):
        for j in range(n):
            val = cm_pct[i, j]
            if val > 0.05:
                color = "white" if val > 50 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall (%)")
    plt.tight_layout()

    out_path = BASE_DIR / "models" / "confusion_matrix.png"
    fig.savefig(out_path, dpi=150)
    print(f"✅ Confusion matrix saved → {out_path}")
    plt.show()


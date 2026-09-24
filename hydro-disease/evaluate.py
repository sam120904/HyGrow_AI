"""
evaluate.py — Evaluate a trained checkpoint on the test split.

Usage
-----
    python evaluate.py                                       # uses best checkpoint
    python evaluate.py --checkpoint models/checkpoints/mobilenetv3_best.pth
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import torch  # pyrefly: ignore[missing-import]
import torch.nn as nn  # pyrefly: ignore[missing-import]
import numpy as np  # pyrefly: ignore[missing-import]
from torch.utils.data import DataLoader  # pyrefly: ignore[missing-import]
from torchvision import models  # pyrefly: ignore[missing-import]

from dataset import AlbumentationsDataset, VAL_TRANSFORM  # pyrefly: ignore[missing-import]

BASE_DIR = Path(__file__).resolve().parent
CKPT_DIR = BASE_DIR / "models" / "checkpoints"
NUM_CLASSES = 15


# ──────────────────────────────────────────────
# Metrics helpers
# ──────────────────────────────────────────────
def compute_metrics(y_true: list, y_pred: list, class_names: list) -> dict:
    """Compute per-class precision, recall, F1 and overall accuracy."""
    num_classes = len(class_names)
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    per_class = {}
    for i in range(num_classes):
        precision = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
        recall = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0.0)
        per_class[class_names[i]] = {
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1": round(f1 * 100, 2),
            "support": tp[i] + fn[i],
        }

    accuracy = sum(tp.values()) / len(y_true) if y_true else 0.0
    macro_f1 = np.mean([v["f1"] for v in per_class.values()])

    return {
        "accuracy": round(accuracy * 100, 2),
        "macro_f1": round(float(macro_f1), 2),
        "per_class": per_class,
    }


def print_report(metrics: dict) -> None:
    """Pretty-print a classification report."""
    print(f"\n{'─' * 72}")
    print(f"{'Class':40s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s} {'Support':>8s}")
    print(f"{'─' * 72}")
    for cls, m in metrics["per_class"].items():
        print(f"{cls:40s} {m['precision']:7.2f} {m['recall']:7.2f} "
              f"{m['f1']:7.2f} {m['support']:>8d}")
    print(f"{'─' * 72}")
    print(f"{'Overall Accuracy':40s} {metrics['accuracy']:7.2f}%")
    print(f"{'Macro F1':40s} {metrics['macro_f1']:7.2f}%")
    print(f"{'─' * 72}\n")


# ──────────────────────────────────────────────
# Confusion matrix (text-based)
# ──────────────────────────────────────────────
def print_confusion_matrix(y_true, y_pred, class_names):
    """Print a simple text confusion matrix."""
    n = len(class_names)
    matrix = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1

    # Abbreviate class names for display
    abbr = [name[:18] for name in class_names]
    col_w = max(len(a) for a in abbr) + 2

    print("\n📊 Confusion Matrix (rows=true, cols=predicted):\n")
    header = " " * col_w + "".join(f"{a:>{col_w}}" for a in abbr)
    print(header)
    for i, row in enumerate(matrix):
        row_str = f"{abbr[i]:<{col_w}}" + "".join(f"{v:>{col_w}}" for v in row)
        print(row_str)
    print()


# ──────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────
@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Device : {device}")

    # --- Load checkpoint ---
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = BASE_DIR / ckpt_path

    print(f"📂 Checkpoint : {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)

    # --- Build model ---
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # --- Data ---
    data_dir = args.data_dir if args.data_dir else str(args.dataset_root / args.split)
    test_ds = AlbumentationsDataset(data_dir, VAL_TRANSFORM)
    class_names = test_ds.classes
    expected_names = json.loads((BASE_DIR / "class_names.json").read_text(encoding="utf-8"))
    if class_names != expected_names:
        raise ValueError("Dataset class order differs from class_names.json")
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    print(f"   Classes    : {len(class_names)}")
    print(f"\n📦 Evaluating on '{args.split}' split  ({len(test_ds)} images)")

    # --- Inference ---
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())

    # --- Metrics ---
    metrics = compute_metrics(all_labels, all_preds, class_names)
    print_report(metrics)

    if args.confusion_matrix and len(class_names) <= 20:
        print_confusion_matrix(all_labels, all_preds, class_names)

    # --- Save results ---
    results_path = CKPT_DIR / f"eval_{args.split}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Results saved → {results_path}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate plant-disease model")
    parser.add_argument("--checkpoint", type=str,
                        default=str(CKPT_DIR / "mobilenetv3_best.pth"),
                        help="Path to .pth checkpoint")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test", "custom"])
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Custom data directory (only with --split custom)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dataset-root", type=Path,
                        default=BASE_DIR / "data" / "plantvillage-hf")
    parser.add_argument("--confusion-matrix", action="store_true", default=True)
    parser.add_argument("--no-confusion-matrix", dest="confusion_matrix",
                        action="store_false")
    args = parser.parse_args()

    main(args)

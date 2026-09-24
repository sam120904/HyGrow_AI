"""Export a trained checkpoint to the ONNX file consumed by app.py."""

import argparse
import json
from pathlib import Path

import torch
from torchvision import models


BASE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=BASE / "models" / "checkpoints" / "mobilenetv3_best.pth")
    parser.add_argument("--output", type=Path, default=BASE / "disease_detector.onnx")
    args = parser.parse_args()
    class_names = json.loads((BASE / "class_names.json").read_text(encoding="utf-8"))
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(class_names))
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.randn(1, 3, 224, 224), args.output,
                      input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17)
    print(f"Exported {args.output}")


if __name__ == "__main__":
    main()

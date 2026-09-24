"""Gradio Space entrypoint. The public /predict endpoint returns class probabilities."""

import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image


BASE = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("HYGROW_MODEL_PATH", str(BASE / "disease_detector.onnx"))).resolve()
CLASS_NAMES = json.loads((BASE / "class_names.json").read_text(encoding="utf-8"))
SESSION = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
INPUT_NAME = SESSION.get_inputs()[0].name
OUTPUT_WIDTH = SESSION.get_outputs()[0].shape[-1]
if isinstance(OUTPUT_WIDTH, int) and OUTPUT_WIDTH != len(CLASS_NAMES):
    raise ValueError("ONNX output width differs from class_names.json")


def preprocess(image: Image.Image) -> np.ndarray:
    rgb = image.convert("RGB").resize((224, 224))
    pixels = np.asarray(rgb, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return np.transpose((pixels - mean) / std, (2, 0, 1))[None].astype(np.float32)


def predict(image: Image.Image) -> dict[str, float]:
    if image is None:
        raise gr.Error("Upload a leaf image to predict a disease class.")
    logits = np.asarray(SESSION.run(None, {INPUT_NAME: preprocess(image)})[0])[0]
    if logits.shape != (len(CLASS_NAMES),):
        raise ValueError(f"Unexpected model output shape: {logits.shape}")
    shifted = logits - np.max(logits)
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    return dict(zip(CLASS_NAMES, probabilities.astype(float).tolist()))


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    title="HyGrow Leaf Disease Detector",
    description="PlantVillage trained classifier for bell pepper, potato, and tomato leaves. Predictions are informational, especially outside controlled image conditions.",
    api_name="predict",
)


if __name__ == "__main__":
    demo.launch()

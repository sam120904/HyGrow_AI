import gradio as gr
import onnxruntime as ort
import numpy as np
from PIL import Image
import json

# Load model
session = ort.InferenceSession("disease_detector.onnx")

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

input_name = session.get_inputs()[0].name


def preprocess(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image = np.array(image).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    return image.astype(np.float32)


def predict(image):
    image = preprocess(image)

    outputs = session.run(None, {input_name: image})
    logits = outputs[0][0]

    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()

    return {
        class_names[i]: float(probs[i])
        for i in range(len(class_names))
    }


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    title="Hydro Disease Detector",
    description="Upload a plant leaf image to identify disease."
)

demo.launch()
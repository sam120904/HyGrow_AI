import torch  # pyrefly: ignore[missing-import]
from torchvision import models  # pyrefly: ignore[missing-import]

model = models.mobilenet_v3_large()
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 15)
model.load_state_dict(torch.load("models/checkpoints/mobilenetv3_best.pth"))
model.eval()

dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy,
    "models/disease_detector.onnx",
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={"image": {0: "batch"}},
    opset_version=17
)
print("Exported to ONNX")

# Then push to HuggingFace Hub:
# pip install huggingface_hub
# huggingface-cli login
# huggingface-cli upload your-org/hydro-disease models/disease_detector.onnx
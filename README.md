# HyGrow AI

Fifteen-class bell pepper, potato, and tomato leaf classifier. `hydro-disease/app.py` exposes a Gradio `/predict` endpoint that accepts one image and returns class probabilities. The HyGrow software backend calls a deployed Hugging Face Space through `HF_SPACE`.

## Dataset provenance and setup

The [PlantVillage authors' repository](https://github.com/spMohanty/PlantVillage-Dataset) recommends its [Hugging Face `mohanty/PlantVillage` dataset](https://huggingface.co/datasets/mohanty/PlantVillage). This project uses its `color` configuration at revision `9e97599868962bd0079b8db4b7f1efa9185fa1e7` (listed license: CC BY-SA 3.0). Cite Mohanty, Hughes, and Salathé, *Using Deep Learning for Image-Based Plant Disease Detection*, Frontiers in Plant Science (2016), DOI: 10.3389/fpls.2016.01419, when using the data or reporting results.

The source's predefined test partition groups images by physical leaf. `prepare_data.py` keeps that test partition and derives validation from training by a seeded leaf-ID hash, so images from one leaf cannot cross partitions. It selects the 15 classes used by the existing HyGrow model and maps source labels to `class_names.json` in the same order. It saves a source/seed/count manifest. The generated data and historical duplicate image trees are ignored by Git; they are no longer needed in a clone.

From `HyGrow_AI/hydro-disease`:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
.\.venv\Scripts\python.exe prepare_data.py
```

On macOS/Linux, use `.venv/bin/python`. The download needs internet and several GB of free disk space. To use another location, pass `--output PATH` to `prepare_data.py` and the matching `--data-dir PATH` to `train.py`, `--dataset-root PATH` to `evaluate.py`. Setup refuses to overwrite a nonempty destination. Cached Hugging Face downloads can be reused.

In a notebook such as Colab, this single cell runs setup through export after cloning the parent repository. Adjust the first path if the clone is elsewhere, and use a GPU runtime if available:

```python
%cd /content/HyGrow/HyGrow_AI/hydro-disease
%pip install -r requirements-train.txt
!python prepare_data.py && python train.py && python evaluate.py --split test && python export.py
```

The cell can take substantial time and disk space. `prepare_data.py` refuses to overwrite an existing populated output directory; for a repeat run, keep the prepared data and run the later commands separately.

## Train, evaluate, export

```powershell
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe evaluate.py --split test
.\.venv\Scripts\python.exe export.py
```

`train.py` fine-tunes MobileNetV3 Large and writes the best validation checkpoint to `models/checkpoints/mobilenetv3_best.pth`. `evaluate.py` writes per-class precision, recall, F1, support, overall accuracy, and macro F1 to `models/checkpoints/eval_test.json`; metrics are percentages. `confusion_matrix.py` renders an optional plot. `export.py` writes `disease_detector.onnx` and may create a `.data` sidecar; deploy both together. Training dependencies include PyTorch and may require a platform-specific CPU/CUDA installation. Seeds and dataset revision are fixed, but GPU kernels and package versions can still vary. No full retraining or new benchmark is claimed here.

The previously tracked checkpoint, ONNX files, and `eval_test.json` predate this new leaf-grouped setup. In particular, the old reported test accuracy should not be treated as a result from the new partition. Retrain and evaluate before quoting new metrics. PlantVillage images were collected under controlled conditions; performance on hydroponic, in-field, or ESP32 camera images needs separate validation.

## Run or deploy inference

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

`app.py` resolves `class_names.json` and the ONNX model relative to itself, so it works from any current directory. Set `HYGROW_MODEL_PATH` to override the model path. The output is a Gradio `Label` with the original 15 label strings and probability scores; its named API endpoint is `/predict`.

For a Hugging Face Gradio Space, place `app.py`, `class_names.json`, `requirements.txt`, `disease_detector.onnx`, and any `disease_detector.onnx.data` sidecar in the Space root. Set the HyGrow backend's `HF_SPACE` environment variable to that Space's `owner/name` identifier. Deploying a newly trained checkpoint requires running `export.py` and updating the model files in the Space. The local app does not automatically update a hosted Space.

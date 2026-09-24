"""Download author-maintained PlantVillage color data and create leaf-safe splits."""

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset


BASE = Path(__file__).resolve().parent
DATASET = "mohanty/PlantVillage"
REVISION = "9e97599868962bd0079b8db4b7f1efa9185fa1e7"
SOURCE_LABELS = [
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def validation_split(leaf_id: str, seed: int, fraction: float) -> str:
    value = hashlib.sha256(f"{seed}:{leaf_id}".encode()).digest()
    return "val" if int.from_bytes(value[:8], "big") / 2**64 < fraction else "train"


def prepare(destination: Path, seed: int = 42, val_fraction: float = 0.15) -> dict:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"{destination} is not empty; choose a new output directory")

    names = json.loads((BASE / "class_names.json").read_text(encoding="utf-8"))
    if len(names) != len(SOURCE_LABELS) or len(set(names)) != len(names):
        raise ValueError("class_names.json must contain 15 distinct labels")
    source_to_target = dict(zip(SOURCE_LABELS, names))
    dataset = load_dataset(DATASET, "color", revision=REVISION)
    labels = dataset["train"].features["label"].names
    missing = set(SOURCE_LABELS) - set(labels)
    if missing:
        raise ValueError(f"Source dataset is missing expected classes: {sorted(missing)}")

    counts = {split: {name: 0 for name in names} for split in ("train", "val", "test")}
    leaf_splits = {}
    for source_split in ("train", "test"):
        for index, item in enumerate(dataset[source_split]):
            source_name = labels[item["label"]]
            if source_name not in source_to_target:
                continue
            leaf_id = str(item["leaf_id"])
            split = "test" if source_split == "test" else validation_split(leaf_id, seed, val_fraction)
            if leaf_id in leaf_splits and leaf_splits[leaf_id] != split:
                raise ValueError(f"Leaf {leaf_id} crosses splits")
            leaf_splits[leaf_id] = split
            target_name = source_to_target[source_name]
            directory = destination / split / target_name
            directory.mkdir(parents=True, exist_ok=True)
            item["image"].convert("RGB").save(directory / f"{index:06d}.jpg", quality=95)
            counts[split][target_name] += 1

    empty = [(split, name) for split, classes in counts.items() for name, count in classes.items() if not count]
    if empty:
        raise ValueError(f"Empty split/class combinations: {empty}")
    manifest = {"dataset": DATASET, "configuration": "color", "revision": REVISION,
                "license": "CC BY-SA 3.0", "seed": seed, "val_fraction": val_fraction,
                "class_names": names, "counts": counts}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "data" / "plantvillage-hf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    args = parser.parse_args()
    result = prepare(args.output, args.seed, args.val_fraction)
    print(json.dumps({split: sum(classes.values()) for split, classes in result["counts"].items()}, indent=2))

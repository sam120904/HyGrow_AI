import os, shutil, random
from pathlib import Path

SRC = Path("data/raw/PlantVillage")
DST = Path("data/processed")
SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}

for cls_dir in SRC.iterdir():
    images = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.JPG"))
    random.shuffle(images)
    n = len(images)
    cuts = [int(n * SPLITS["train"]), int(n * (SPLITS["train"] + SPLITS["val"]))]
    for split, subset in zip(["train","val","test"],
                              [images[:cuts[0]], images[cuts[0]:cuts[1]], images[cuts[1]:]]):
        dest = DST / split / cls_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        for img in subset:
            shutil.copy(img, dest / img.name)

print("Split complete.")

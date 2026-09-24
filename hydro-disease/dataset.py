import albumentations as A
import numpy as np
import torch
from torchvision.datasets import ImageFolder


class AlbumentationsDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root)
        self.alb_transform = transform

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        image = np.asarray(image.convert("RGB"))
        if self.alb_transform is not None:
            image = self.alb_transform(image=image)["image"]
        return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float(), label


TRAIN_TRANSFORM = A.Compose([
    A.Resize(224, 224), A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3), A.GaussianBlur(blur_limit=(3, 5), p=0.1),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
VAL_TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

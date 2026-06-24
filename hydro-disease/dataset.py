import albumentations as A  # pyrefly: ignore[missing-import]
import torch  # pyrefly: ignore[missing-import]
from torchvision.datasets import ImageFolder  # pyrefly: ignore[missing-import]
import numpy as np  # pyrefly: ignore[missing-import]


def _to_tensor(image, **kwargs):
    """Convert HWC numpy array to CHW float tensor."""
    return torch.from_numpy(image.transpose(2, 0, 1)).float()


class AlbumentationsDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root)
        self.alb_transform = transform

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        img = np.array(img)
        if self.alb_transform:
            img = self.alb_transform(image=img)["image"]
        # Convert to tensor if still a numpy array
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        return img, label


TRAIN_TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.4),
    A.HueSaturationValue(p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),  # simulates ESP32 blur
    A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(10, 20), hole_width_range=(10, 20), p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

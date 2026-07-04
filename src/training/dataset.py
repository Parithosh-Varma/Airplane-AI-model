import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from config import TRAINING_CONFIG

logger = logging.getLogger(__name__)


class AircraftImageDataset(Dataset):
    def __init__(
        self,
        image_paths: list[str],
        labels: Optional[list[int]] = None,
        augment: bool = False,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            if path.endswith(".npy"):
                tensor = torch.from_numpy(np.load(path))
            else:
                from PIL import Image
                from torchvision import transforms
                img = Image.open(path).convert("RGB")
                transform = transforms.Compose([
                    transforms.Resize(TRAINING_CONFIG["image_size"]),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ])
                tensor = transform(img)

            if tensor.shape[0] == 3:
                pass
            elif tensor.shape[-1] == 3:
                tensor = tensor.permute(2, 0, 1)
            else:
                tensor = tensor.expand(3, -1, -1)

            if self.augment:
                tensor = self._augment(tensor)

            if self.labels is not None:
                return tensor, torch.tensor(self.labels[idx], dtype=torch.long)
            return tensor, torch.tensor(0, dtype=torch.long)

        except Exception as e:
            logger.warning("Failed to load %s: %s", path, e)
            dummy = torch.zeros(3, *TRAINING_CONFIG["image_size"])
            if self.labels is not None:
                return dummy, torch.tensor(self.labels[idx], dtype=torch.long)
            return dummy, torch.tensor(0, dtype=torch.long)

    def _augment(self, tensor):
        from torchvision import transforms as T
        aug = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])
        return aug(tensor)

import os
from typing import Optional, Callable, Any

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image


class UtkFaceDataset(Dataset):

    def __init__(self, faces_dir: str, transform: Optional[Callable[[Image.Image], Any]]) -> None:
        super(UtkFaceDataset, self).__init__()
        self.faces_dir = faces_dir
        self.image_names = os.listdir(faces_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> tuple[Image.Image, torch.tensor]:
        image_name = self.image_names[index]
        image_path = os.path.join(self.faces_dir, image_name)
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        age, gender, race, *_ = image_name.split("_")
        target_tensor = torch.tensor([
            int(age),
            *F.one_hot(torch.tensor(int(gender)), num_classes=2),
            *F.one_hot(torch.tensor(int(race)), num_classes=5)
        ]).to(torch.float32)
        return image, target_tensor

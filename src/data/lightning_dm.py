import os
import shutil
from typing import Optional, Callable, Any

import pytorch_lightning as pl
import cv2
from PIL import Image
import opendatasets as od
from torch.utils.data import random_split, DataLoader
from dotenv import load_dotenv

from data.dataset import UtkFaceDataset


DATASET_URL = "https://www.kaggle.com/datasets/moritzm00/utkface-cropped"


def is_gray_scale(image_path: str) -> bool:
    img = cv2.imread(image_path)
    if len(img.shape) < 3:
        return True
    return False


class UtkFaceDataModule(pl.LightningDataModule):

    def __init__(
            self,
            root_dir: str,
            batch_size: int,
            num_workers: int = 1,
            train_ratio: float = .7,
            val_ratio: float = .15,
            test_ratio: float = .15,
            transform: Optional[Callable[[Image.Image], Any]] = None,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.test_set = None
        self.val_set = None
        self.train_set = None
        self.root_dir = root_dir
        self.transform = transform
        self.faces_dir = os.path.join(self.root_dir, "faces")
        os.makedirs(self.faces_dir, exist_ok=True)
        self.sets_ratios = (train_ratio, val_ratio, test_ratio)
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        if os.listdir(self.faces_dir):
            return
        load_dotenv()
        downloading_dir = os.path.join(self.root_dir, "raw")
        downloaded_images_dir = os.path.join(downloading_dir, "utkface-cropped", "UTKFace")
        self._download(downloading_dir)
        self._clean(downloaded_images_dir, self.faces_dir)
        shutil.rmtree(downloading_dir)

    def _clean(self, src_dir: str, dst_dir: str) -> None:
        for image_name in os.listdir(src_dir):
            image_path = os.path.join(src_dir, image_name)
            if is_gray_scale(image_path) or len(image_name.split("_")) != 4:
                continue
            image_dst_path = os.path.join(dst_dir, image_name)
            os.rename(image_path, image_dst_path)

    def _not_valid_image(self, image_name: str, image_path: str) -> bool:
        return is_gray_scale(image_path) or len(image_name.split("_")) != 4

    def _download(self, downloading_dir: str):
        print("PASSING KAGGLE CREDENTIALS IS OPTIONAL. (PASS DOTS '.' INSTEAD)")
        od.download(DATASET_URL, data_dir=downloading_dir)

    def setup(self, stage: str) -> None:
        dataset = UtkFaceDataset(self.faces_dir, self.transform)
        self.train_set, self.val_set, self.test_set = random_split(dataset, self.sets_ratios)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)

from typing import Callable, Literal

import lightning as L  # noqa: N812
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.transforms.v2 import Transform

from pytorchcocotools.torch.dataset import CocoDetection
from pytorchcocotools.torch.download import CocoDownloader
from pytorchcocotools.torch.transform import default_transform
from pytorchcocotools.utils.logging import get_logger


class COCODataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str,
        annotation_files: dict[Literal["train", "val", "test", "predict"], str],
        batch_size: int,
        num_workers: int = 2,
        transform: Callable | Transform = default_transform,
        download: bool = False,
    ) -> None:
        super().__init__()
        self.root = root
        self.annotation_files = annotation_files
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.logger = get_logger(self.__class__.__name__)
        self.download = download

    def prepare_data(self) -> None:
        # download
        if self.download:
            self.logger.info("Downloading COCO dataset")
            downloader = CocoDownloader(self.root)
            downloader.download()
            self.annotation_files = {
                "train": "annotations/instances_train2017.json",
                "val": "annotations/instances_val2017.json",
                "test": "annotations/image_info_test2017.json",
                "predict": "annotations/image_info_unlabeled2017.json",
            }

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.logger.debug("Setting up train and val datasets")
            self.train_dataset: CocoDetection = CocoDetection(self.root, self.annotation_files["train"])
            self.val_dataset: CocoDetection = CocoDetection(self.root, self.annotation_files["val"])
        if stage == "test":
            self.test_dataset: CocoDetection = CocoDetection(self.root, self.annotation_files["test"])
        if stage == "predict":
            self.predict_dataset: CocoDetection = CocoDetection(self.root, self.annotation_files["predict"])

    def train_dataloader(self) -> DataLoader[CocoDetection]:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader[CocoDetection]:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader[CocoDetection]:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self) -> DataLoader[CocoDetection]:
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

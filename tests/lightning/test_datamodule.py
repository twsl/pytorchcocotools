import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import tv_tensors as tvt

from pytorchcocotools.lightning.datamodule import COCODataModule
from pytorchcocotools.torch.dataset import CocoDetection
from pytorchcocotools.utils.stage import StageStore

ANNOTATION_FILE = "./data/example.json"
ROOT = "./data"
IMAGE_ROOT = "./data/coco"


@pytest.fixture
def annotation_files() -> StageStore[str]:
    return StageStore(
        train=ANNOTATION_FILE,
        val=ANNOTATION_FILE,
        test=ANNOTATION_FILE,
        predict=ANNOTATION_FILE,
    )


@pytest.fixture
def datamodule(annotation_files: StageStore[str]) -> COCODataModule:
    return COCODataModule(root=ROOT, annotation_files=annotation_files, batch_size=2)


@pytest.fixture
def datamodule_with_images(annotation_files: StageStore[str]) -> COCODataModule:
    return COCODataModule(root=IMAGE_ROOT, annotation_files=annotation_files, batch_size=2)


def test_init(datamodule: COCODataModule) -> None:
    assert datamodule.root == ROOT
    assert datamodule.batch_size == 2
    assert datamodule.num_workers == 2
    assert datamodule.download is False


def test_transforms_wrapped_in_stage_store(annotation_files: StageStore[str]) -> None:
    dm = COCODataModule(root=ROOT, annotation_files=annotation_files, batch_size=1)
    assert isinstance(dm.transforms, StageStore)
    assert dm.transforms.train is not None
    assert dm.transforms.val is not None
    assert dm.transforms.test is None
    assert dm.transforms.predict is None


def test_transforms_stage_store_passthrough(annotation_files: StageStore[str]) -> None:
    custom = StageStore(train=None, val=None, test=None, predict=None)
    dm = COCODataModule(root=ROOT, annotation_files=annotation_files, batch_size=1, transforms=custom)
    assert dm.transforms is custom


def test_setup_fit(datamodule: COCODataModule) -> None:
    datamodule.setup("fit")
    assert isinstance(datamodule.train_dataset, CocoDetection)
    assert len(datamodule.train_dataset) > 0


def test_setup_validate(datamodule: COCODataModule) -> None:
    datamodule.setup("validate")
    assert isinstance(datamodule.val_dataset, CocoDetection)
    assert len(datamodule.val_dataset) > 0


def test_setup_test(datamodule: COCODataModule) -> None:
    datamodule.setup("test")
    assert isinstance(datamodule.test_dataset, CocoDetection)
    assert len(datamodule.test_dataset) > 0


def test_setup_predict(datamodule: COCODataModule) -> None:
    datamodule.setup("predict")
    assert isinstance(datamodule.predict_dataset, CocoDetection)
    assert len(datamodule.predict_dataset) > 0


def test_train_dataloader(datamodule: COCODataModule) -> None:
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()
    assert isinstance(loader, DataLoader)
    assert loader.batch_size == datamodule.batch_size


def test_val_dataloader(datamodule: COCODataModule) -> None:
    datamodule.setup("validate")
    loader = datamodule.val_dataloader()
    assert isinstance(loader, DataLoader)
    assert loader.batch_size == datamodule.batch_size


def test_test_dataloader(datamodule: COCODataModule) -> None:
    datamodule.setup("test")
    loader = datamodule.test_dataloader()
    assert isinstance(loader, DataLoader)
    assert loader.batch_size == datamodule.batch_size


def test_predict_dataloader(datamodule: COCODataModule) -> None:
    datamodule.setup("predict")
    loader = datamodule.predict_dataloader()
    assert isinstance(loader, DataLoader)
    assert loader.batch_size == datamodule.batch_size


def test_prepare_data_no_download(datamodule: COCODataModule) -> None:
    original_files = datamodule.annotation_files
    datamodule.prepare_data()
    assert datamodule.annotation_files is original_files


def test_sample_from_dataset(datamodule_with_images: COCODataModule) -> None:
    datamodule_with_images.setup("fit")
    image, target = datamodule_with_images.train_dataset[0]
    assert isinstance(image, tvt.Image)
    assert image.ndim == 3  # C x H x W
    assert image.shape[0] in (1, 3, 4)


def test_sample_target_keys(datamodule_with_images: COCODataModule) -> None:
    datamodule_with_images.setup("fit")
    _, target = datamodule_with_images.train_dataset[0]
    assert "image_id" in target
    assert "boxes" in target
    assert "masks" in target
    assert "labels" in target


def test_sample_boxes_format(datamodule_with_images: COCODataModule) -> None:
    datamodule_with_images.setup("fit")
    _, target = datamodule_with_images.train_dataset[0]
    boxes = target["boxes"]
    assert isinstance(boxes, tvt.BoundingBoxes)
    assert boxes.ndim == 2
    assert boxes.shape[-1] == 4
    assert boxes.format == tvt.BoundingBoxFormat.XYXY


def test_sample_labels_dtype(datamodule_with_images: COCODataModule) -> None:
    datamodule_with_images.setup("fit")
    _, target = datamodule_with_images.train_dataset[0]
    labels = target["labels"]
    assert isinstance(labels, torch.Tensor)
    assert labels.ndim == 1
    assert labels.shape[0] == target["boxes"].shape[0]


def test_sample_masks_shape(datamodule_with_images: COCODataModule) -> None:
    datamodule_with_images.setup("fit")
    image, target = datamodule_with_images.train_dataset[0]
    masks = target["masks"]
    assert isinstance(masks, tvt.Mask)
    assert masks.ndim == 4  # N x C x H x W
    assert masks.shape[0] == target["boxes"].shape[0]

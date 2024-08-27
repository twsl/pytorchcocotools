from collections.abc import Callable
from pathlib import Path
import re
from typing import Any, cast

import torch
from torchvision import tv_tensors as tvt
from torchvision.datasets.vision import VisionDataset
from torchvision.io import read_image
from torchvision.transforms.v2 import functional as F  # noqa: N812
from torchvision.tv_tensors._dataset_wrapper import list_of_dicts_to_dict_of_lists

from pytorchcocotools import mask
from pytorchcocotools.coco import COCO
from pytorchcocotools.internal.entities import RleObjs
from pytorchcocotools.internal.structure.annotations import (
    CocoAnnotationKeypointDetection,
    CocoAnnotationObjectDetection,
    CocoDetectionAnnotation,
)


class CocoDetection(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2020>`_ Dataset.

    It requires the `PyTorch COCO API to be installed <https://github.com/twsl/pytorchcocotools>`_.

    Args:
        root: Root directory where images are downloaded to.
        annFile: Path to json annotation file.
        transform: A function/transform that takes in a ``torchvision.tv_tensors.Image`` image
            and returns a transformed version.
        target_transform: A function/transform that takes in the
            target and transforms it.
        transforms: A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,  # noqa: N803
        transforms: Callable | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        out_bbox_fmt: tvt.BoundingBoxFormat = tvt.BoundingBoxFormat.XYXY,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.coco = COCO(annFile)
        self.ids = sorted(self.coco.imgs.keys())
        self.out_bbox_fmt = out_bbox_fmt

    @property
    def num_classes(self) -> int:
        return len(self.coco.dataset.categories)

    @property
    def version(self) -> str:
        info = self.coco.dataset["info"]
        # https://semver.org/lang/de/
        build_identifier = re.sub("[^0-9a-zA-Z]+", "_", info["date_created"])
        return f"{info['version']}+{build_identifier}"

    def _load_image(self, id: int) -> tvt.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        img = read_image(str(Path(self.root) / path))
        return tvt.Image(img)

    def _load_target(self, id: int) -> list[CocoDetectionAnnotation]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def _segmentation_to_mask(self, segmentation, *, canvas_size) -> torch.Tensor:
        segmentation = (
            mask.frPyObjects(segmentation, *canvas_size)
            if isinstance(segmentation, dict) and "counts" in segmentation
            else mask.merge(cast(RleObjs, mask.frPyObjects(segmentation, *canvas_size)))
        )
        return mask.decode(segmentation)

    def __getitem__(self, index: int) -> tuple[tvt.Image, dict[str, Any]]:
        id = self.ids[index]
        image = self._load_image(id)
        og_target = self._load_target(id)

        canvas_size = cast(tuple[int, int], tuple(F.get_size(image)))

        batched_target = list_of_dicts_to_dict_of_lists(og_target)

        target: dict[str, Any] = {}
        target["image_id"] = id

        target["boxes"] = F.convert_bounding_box_format(
            tvt.BoundingBoxes(
                batched_target["bbox"],
                format=tvt.BoundingBoxFormat.XYWH,
                canvas_size=canvas_size,
            ),  # pyright: ignore[reportCallIssue]
            new_format=self.out_bbox_fmt,
        )
        target["masks"] = tvt.Mask(
            torch.stack(
                [
                    self._segmentation_to_mask(segmentation, canvas_size=canvas_size)
                    for segmentation in batched_target["segmentation"]
                ]
            ),
        )
        target["labels"] = torch.tensor(batched_target["category_id"])

        if "area" in batched_target:
            target["area"] = torch.tensor(batched_target["area"])
        if "iscrowd" in batched_target:
            target["iscrowd"] = torch.tensor(batched_target["iscrowd"])
        if "keypoints" in batched_target:
            target["keypoints"] = torch.tensor(batched_target["keypoints"])

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.ids)

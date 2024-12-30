from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal, TypeAlias
from urllib.error import URLError

from torchvision.datasets.utils import (
    _flip_byte_order,
    check_integrity,
    download_and_extract_archive,
    extract_archive,
    verify_str_arg,
)

from pytorchcocotools.utils.logging import get_logger
from pytorchcocotools.utils.stage import StageStore

CocoYear: TypeAlias = Literal[2014, 2015, 2017]


@dataclass
class FileEntity:
    filename: str
    md5: str


class CocoFiles(Enum):
    Train2014 = FileEntity(filename="train2014.zip", md5="0da8c0bd3d6becc4dcb32757491aca88")
    Val2014 = FileEntity(filename="val2014.zip", md5="a3d79f5ed8d289b7a7554ce06a5782b3")
    Test2014 = FileEntity(filename="test2014.zip", md5="04127eef689ceac55e3a572c2c92f264")
    Test2015 = FileEntity(filename="test2015.zip", md5="65562e58af7d695cc47356951578c041")
    AnnotationsTrainVal2014 = FileEntity(
        filename="annotations_trainval2014.zip", md5="0a379cfc70b0e71301e0f377548639bd"
    )
    ImageInfoTest2014 = FileEntity(filename="image_info_test2014.zip", md5="25304cbbafb2117fb801c0f7218fdbba")
    ImageInfoTest2015 = FileEntity(filename="image_info_test2015.zip", md5="d9616c9742182f530e2e6d8c6ccd916a")
    Val2017 = FileEntity(filename="val2017.zip", md5="442b8da7639aecaf257c1dceb8ba8c80")
    Test2017 = FileEntity(filename="test2017.zip", md5="77ad2c53ac5d0aea611d422c0938fb35")
    Train2017 = FileEntity(filename="train2017.zip", md5="cced6f7f71b7629ddf16f17bbcfab6b2")
    Unlabeled2017 = FileEntity(filename="unlabeled2017.zip", md5="7ebc562819fdb32847aab79530457326")
    AnnotationsTrainVal2017 = FileEntity(
        filename="annotations_trainval2017.zip", md5="f4bbac642086de4f52a3fdda2de5fa2c"
    )
    StuffAnnotationsTrainVal2017 = FileEntity(
        filename="stuff_annotations_trainval2017.zip", md5="2a27c15a2dfcbd2e1c9276dc23cac101"
    )
    PanopticAnnotationsTrainVal2017 = FileEntity(
        filename="panoptic_annotations_trainval2017.zip", md5="4170db65fc022c9c296af880dbca6055"
    )
    ImageInfoTest2017 = FileEntity(filename="image_info_test2017.zip", md5="85da7065e5e600ebfee8af1edb634eb5")
    ImageInfoUnlabeled2017 = FileEntity(filename="image_info_unlabeled2017.zip", md5="ede38355d5c3e5251bb7f8b68e2c068f")


@dataclass
class BaseCocoFileConfig:
    year: CocoYear
    images: list[CocoFiles]
    annotations: list[CocoFiles]
    stage_files: StageStore[str]


@dataclass
class Coco2014FileConfig(BaseCocoFileConfig):
    year: CocoYear = 2014
    images: list[CocoFiles] = field(
        default_factory=lambda: [
            CocoFiles.Train2014,
            CocoFiles.Val2014,
            CocoFiles.Test2014,
        ]
    )
    annotations: list[CocoFiles] = field(
        default_factory=lambda: [
            CocoFiles.AnnotationsTrainVal2014,
            CocoFiles.ImageInfoTest2014,
        ]
    )
    stage_files: StageStore[str] = field(
        default_factory=lambda: StageStore(
            train="instances_train2014.json",
            val="instances_val2014.json",
            test="image_info_test2014.json",
        )
    )


@dataclass
class Coco2015FileConfig(BaseCocoFileConfig):
    year: CocoYear = 2015
    images: list[CocoFiles] = field(
        default_factory=lambda: [
            CocoFiles.Train2014,
            CocoFiles.Val2014,
            CocoFiles.Test2015,
        ]
    )
    annotations: list[CocoFiles] = field(
        default_factory=lambda: [
            CocoFiles.AnnotationsTrainVal2014,
            CocoFiles.ImageInfoTest2015,
        ]
    )
    stage_files: StageStore[str] = field(
        default_factory=lambda: StageStore(
            train="instances_train2014.json",
            val="instances_val2014.json",
            test="image_info_test2014.json",
        )
    )


@dataclass
class Coco2017FileConfig(BaseCocoFileConfig):
    year: CocoYear = 2017
    images: list[CocoFiles] = field(
        default_factory=lambda: [
            CocoFiles.Val2017,
            CocoFiles.Test2017,
            CocoFiles.Train2017,
            CocoFiles.Unlabeled2017,
        ]
    )
    annotations: list[CocoFiles] = field(
        default_factory=lambda: [
            CocoFiles.AnnotationsTrainVal2017,
            CocoFiles.ImageInfoTest2017,
            CocoFiles.ImageInfoUnlabeled2017,
        ]
    )
    stage_files: StageStore[str] = field(
        default_factory=lambda: StageStore(
            train="annotations_trainval2017.json",
            val="annotations_val2017.json",
            test="image_info_test2017.json",
            predict="image_info_unlabeled2017.json",
        )
    )

from pathlib import Path
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


class CocoDownloader:
    images = [
        ("train2017.zip", "cced6f7f71b7629ddf16f17bbcfab6b2"),
        ("val2017.zip", "442b8da7639aecaf257c1dceb8ba8c80"),
        ("test2017.zip", "77ad2c53ac5d0aea611d422c0938fb35"),
        ("unlabeled2017.zip", "7ebc562819fdb32847aab79530457326"),
    ]
    annotations = [
        ("annotations_trainval2017.zip", "f4bbac642086de4f52a3fdda2de5fa2c"),
        ("stuff_annotations_trainval2017.zip", "2a27c15a2dfcbd2e1c9276dc23cac101"),
        ("panoptic_annotations_trainval2017.zip", "4170db65fc022c9c296af880dbca6055"),
        ("image_info_test2017.zip", "85da7065e5e600ebfee8af1edb634eb5"),
        ("image_info_unlabeled2017.zip", "ede38355d5c3e5251bb7f8b68e2c068f"),
    ]

    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.zip_url = "http://images.cocodataset.org/zips/"
        self.ann_url = "http://images.cocodataset.org/annotations/"
        self.logger = get_logger(self.__class__.__name__)
        self._image_folder = self.root / "images"
        self._annotation_folder = self.root / "annotations"
        self._annotations_files: StageStore[str] | None = None

    @property
    def image_folder(self) -> Path:
        return self._image_folder

    @property
    def annotation_folder(self) -> Path:
        return self._annotation_folder

    @property
    def annotation_files(self) -> StageStore[str]:
        if not self._annotations_files:
            train_path = self.annotation_folder / "annotations_trainval2017.json"
            val_path = self.annotation_folder / "annotations_val2017.json"
            test_path = self.annotation_folder / "image_info_test2017.json"
            predict_path = self.annotation_folder / "image_info_unlabeled2017.json"
            self._annotations_files = StageStore[str](
                train=train_path.absolute().as_posix(),
                val=val_path.absolute().as_posix(),
                test=test_path.absolute().as_posix(),
                predict=predict_path.absolute().as_posix(),
            )
        return self._annotations_files

    def _check_exists(self) -> bool:
        return all(check_integrity(self.root / Path(filename), md5) for filename, md5 in self.images)

    def _download_assets(self, resources: list[tuple[str, str]], base_url: str, extract_folder: Path) -> None:
        Path.mkdir(self.root, exist_ok=True)

        # download files
        for filename, md5 in resources:
            url = base_url + filename
            try:
                self.logger.info(f"Downloading {url}")
                download_and_extract_archive(
                    url, download_root=self.root, extract_root=extract_folder, filename=filename, md5=md5
                )
            except URLError:
                self.logger.exception(f"Failed to download {filename}.")

    def download(self) -> None:
        """Download the COCO data if it doesn't exist already."""
        if self._check_exists():
            return

        Path.mkdir(self.root, exist_ok=True)

        self._download_assets(self.images, self.zip_url, self.image_folder)
        self._download_assets(self.annotations, self.ann_url, self.annotation_folder)

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


class CocoDownloader:
    resources = [
        ("train2017.zip", ""),
        ("val2017.zip", ""),
        ("test2017.zip", ""),
        ("unlabeled2017.zip", ""),
        ("annotations_trainval2017.zip", "f4bbac642086de4f52a3fdda2de5fa2c"),
        ("stuff_annotations_trainval2017.zip", "2a27c15a2dfcbd2e1c9276dc23cac101"),
        ("stuff_image_info_test2017.zip", "2f8b1a5ac3c4362ae678d0374b5cd3be"),
        ("image_info_test2017.zip", "85da7065e5e600ebfee8af1edb634eb5"),
        ("image_info_unlabeled2017.zip", "ede38355d5c3e5251bb7f8b68e2c068f"),
    ]

    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.zip_url = "http://images.cocodataset.org/zips/"
        self.ann_url = "http://images.cocodataset.org/annotations/"
        self.logger = get_logger(self.__class__.__name__)

    def _check_exists(self) -> bool:
        return all(check_integrity(self.root / Path(filename), md5) for filename, md5 in self.resources)

    def download(self) -> None:
        """Download the COCO data if it doesn't exist already."""
        if self._check_exists():
            return

        Path.mkdir(self.root, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            url = f"{self.ann_url if ' ' in filename else self.zip_url}{filename}"
            try:
                self.logger.info(f"Downloading {url}")
                download_and_extract_archive(url, download_root=self.root, filename=filename, md5=md5)
            except URLError:
                self.logger.exception(f"Failed to download {filename}.")

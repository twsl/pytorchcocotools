from pathlib import Path
from urllib.error import URLError

from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
)

from pytorchcocotools.utils.coco.files import (
    BaseCocoFileConfig,
    Coco2017FileConfig,
    FileEntity,
)
from pytorchcocotools.utils.logging import get_logger
from pytorchcocotools.utils.stage import StageStore


class CocoDownloader:
    def __init__(self, root: str, config: BaseCocoFileConfig | None = None) -> None:
        self.root = Path(root)
        self.config: BaseCocoFileConfig = config if config is not None else Coco2017FileConfig()
        self.zip_url = "http://images.cocodataset.org/zips/"
        self.ann_url = "http://images.cocodataset.org/annotations/"
        self.logger = get_logger(self.__class__.__name__)
        self._image_folder = self.root / "images"
        self._annotation_folder = self.root / "annotations"
        self._annotations_files: StageStore[Path] | None = None
        self._images: list[FileEntity] | None = None
        self._annotations: list[FileEntity] | None = None

    @property
    def images(self) -> list[FileEntity]:
        if self._images is None:
            self._images = [file.value for file in self.config.images]
        return self._images

    @property
    def annotations(self) -> list[FileEntity]:
        if self._annotations is None:
            self._annotations = [file.value for file in self.config.annotations]
        return self._annotations

    @property
    def image_folder(self) -> Path:
        return self._image_folder

    @property
    def annotation_folder(self) -> Path:
        return self._annotation_folder

    @property
    def annotation_files(self) -> StageStore[Path]:
        if not self._annotations_files:
            self._annotations_files = StageStore[Path](
                train=self.annotation_folder / self.config.stage_files.train if self.config.stage_files.train else None,
                val=self.annotation_folder / self.config.stage_files.val if self.config.stage_files.val else None,
                test=self.annotation_folder / self.config.stage_files.test if self.config.stage_files.test else None,
                predict=self.annotation_folder / self.config.stage_files.predict
                if self.config.stage_files.predict
                else None,
            )
        return self._annotations_files

    def _check_exists(self, resources: list[FileEntity]) -> bool:
        return all(check_integrity(self.root / Path(file.filename), file.md5) for file in resources)

    def _download_assets(self, resources: list[FileEntity], base_url: str, extract_folder: Path) -> None:
        Path.mkdir(self.root, exist_ok=True)

        # download files
        for file in resources:
            url = base_url + file.filename
            try:
                self.logger.info(f"Downloading {url}")
                download_and_extract_archive(
                    url, download_root=self.root, extract_root=extract_folder, filename=file.filename, md5=file.md5
                )
                self.logger.info(f"Completed {file.filename}")
            except RuntimeError:
                self.logger.exception(f"Failed to verify {file.filename}.")
            except URLError:
                self.logger.exception(f"Failed to download {file.filename}.")

    def download_images(self) -> None:
        """Download the COCO images if they don't exist already."""
        Path.mkdir(self.root, exist_ok=True)
        if self._check_exists(self.images):
            return

        self._download_assets(self.images, self.zip_url, self.image_folder)

    def download_annotations(self) -> None:
        """Download the COCO annotations if they don't exist already."""
        Path.mkdir(self.root, exist_ok=True)
        if self._check_exists(self.annotations):
            return

        self._download_assets(self.annotations, self.ann_url, self.root)

    def download(self) -> None:
        """Download the COCO data if it doesn't exist already."""
        self.download_images()
        self.download_annotations()

    def gsutil_commands(self) -> list[str]:
        commands = []
        commands.append("curl https://sdk.cloud.google.com | bash")
        commands.append(f"mkdir {self.root}")
        commands.append(f"mkdir {self._image_folder}")
        commands.append(f"mkdir {self._annotation_folder}")
        for file in self.images:
            commands.append(f"gsutil -m rsync gs://images.cocodataset.org/{file.filename} {self.root}")
        commands.append(f"gsutil -m rsync gs://images.cocodataset.org/annotations {self.root}")
        return commands

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(root={self.root}, config={self.config})"

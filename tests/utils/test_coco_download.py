from pathlib import Path
from unittest.mock import Mock, call
from urllib.error import URLError

import pytest

import pytorchcocotools.utils.coco.download as download_module
from pytorchcocotools.utils.coco.download import CocoDownloader
from pytorchcocotools.utils.coco.files import BaseCocoFileConfig, CocoFiles, FileEntity
from pytorchcocotools.utils.stage import StageStore


@pytest.fixture(autouse=True)
def block_real_downloads(monkeypatch) -> None:
    def fail_download(*args, **kwargs) -> None:
        raise AssertionError("download_and_extract_archive must be mocked in unit tests")

    monkeypatch.setattr(download_module, "download_and_extract_archive", fail_download)


def make_config(
    *, val: str | None = "val.json", test: str | None = "test.json", predict: str | None = "predict.json"
) -> BaseCocoFileConfig:
    return BaseCocoFileConfig(
        year=2017,
        images=[CocoFiles.Train2017, CocoFiles.Val2017],
        annotations=[CocoFiles.AnnotationsTrainVal2017],
        stage_files=StageStore(train="train.json", val=val, test=test, predict=predict),
    )


def test_images_annotations_and_paths_are_cached(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    downloader = CocoDownloader(str(root), config=make_config())

    assert downloader.images == [CocoFiles.Train2017.value, CocoFiles.Val2017.value]
    assert downloader.images is downloader.images
    assert downloader.annotations == [CocoFiles.AnnotationsTrainVal2017.value]
    assert downloader.annotations is downloader.annotations
    assert downloader.image_folder == root / "images"
    assert downloader.annotation_folder == root / "annotations"


def test_annotation_files_builds_optional_paths_once(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    downloader = CocoDownloader(str(root), config=make_config(val=None, predict=None))

    annotation_files = downloader.annotation_files

    assert annotation_files.train == root / "annotations" / "train.json"
    assert annotation_files.val is None
    assert annotation_files.test == root / "annotations" / "test.json"
    assert annotation_files.predict is None
    assert downloader.annotation_files is annotation_files


def test_check_exists_uses_root_and_md5(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "dataset"
    downloader = CocoDownloader(str(root), config=make_config())
    calls: list[tuple[Path, str]] = []
    outcomes = iter([True, False])

    def fake_check_integrity(path: Path, md5: str) -> bool:
        calls.append((path, md5))
        return next(outcomes)

    monkeypatch.setattr(download_module, "check_integrity", fake_check_integrity)

    assert downloader._check_exists(downloader.images) is False
    assert calls == [
        (root / CocoFiles.Train2017.value.filename, CocoFiles.Train2017.value.md5),
        (root / CocoFiles.Val2017.value.filename, CocoFiles.Val2017.value.md5),
    ]


def test_download_assets_logs_success_and_failures(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "dataset"
    downloader = CocoDownloader(str(root), config=make_config())
    downloader.logger = Mock()
    resources = [
        FileEntity(filename="ok.zip", md5="ok-md5"),
        FileEntity(filename="corrupt.zip", md5="bad-md5"),
        FileEntity(filename="offline.zip", md5="offline-md5"),
    ]
    calls: list[tuple[str, Path, Path, str, str]] = []

    def fake_download(url: str, download_root: Path, extract_root: Path, filename: str, md5: str) -> None:
        calls.append((url, download_root, extract_root, filename, md5))
        if filename == "corrupt.zip":
            raise RuntimeError("checksum mismatch")
        if filename == "offline.zip":
            raise URLError("offline")

    monkeypatch.setattr(download_module, "download_and_extract_archive", fake_download)

    downloader._download_assets(resources, "https://example.invalid/", root / "extract")

    assert root.exists()
    assert calls == [
        ("https://example.invalid/ok.zip", root, root / "extract", "ok.zip", "ok-md5"),
        ("https://example.invalid/corrupt.zip", root, root / "extract", "corrupt.zip", "bad-md5"),
        ("https://example.invalid/offline.zip", root, root / "extract", "offline.zip", "offline-md5"),
    ]
    assert downloader.logger.info.mock_calls == [
        call("Downloading https://example.invalid/ok.zip"),
        call("Completed ok.zip"),
        call("Downloading https://example.invalid/corrupt.zip"),
        call("Downloading https://example.invalid/offline.zip"),
    ]
    assert downloader.logger.exception.mock_calls == [
        call("Failed to verify corrupt.zip."),
        call("Failed to download offline.zip."),
    ]


def test_download_images_delegates_when_assets_are_missing(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "dataset"
    downloader = CocoDownloader(str(root), config=make_config())
    check_exists = Mock(return_value=False)
    download_assets = Mock()
    monkeypatch.setattr(downloader, "_check_exists", check_exists)
    monkeypatch.setattr(downloader, "_download_assets", download_assets)

    downloader.download_images()

    assert root.exists()
    check_exists.assert_called_once_with(downloader.images)
    download_assets.assert_called_once_with(downloader.images, downloader.zip_url, downloader.image_folder)


def test_download_annotations_skips_when_assets_exist(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "dataset"
    downloader = CocoDownloader(str(root), config=make_config())
    check_exists = Mock(return_value=True)
    download_assets = Mock()
    monkeypatch.setattr(downloader, "_check_exists", check_exists)
    monkeypatch.setattr(downloader, "_download_assets", download_assets)

    downloader.download_annotations()

    assert root.exists()
    check_exists.assert_called_once_with(downloader.annotations)
    download_assets.assert_not_called()


def test_download_calls_images_and_annotations(tmp_path: Path, monkeypatch) -> None:
    downloader = CocoDownloader(str(tmp_path / "dataset"), config=make_config())
    download_images = Mock()
    download_annotations = Mock()
    monkeypatch.setattr(downloader, "download_images", download_images)
    monkeypatch.setattr(downloader, "download_annotations", download_annotations)

    downloader.download()

    download_images.assert_called_once_with()
    download_annotations.assert_called_once_with()


def test_gsutil_commands_and_repr_include_expected_paths(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    downloader = CocoDownloader(str(root), config=make_config())

    commands = downloader.gsutil_commands()

    assert commands == [
        "curl https://sdk.cloud.google.com | bash",
        f"mkdir {root}",
        f"mkdir {root / 'images'}",
        f"mkdir {root / 'annotations'}",
        f"gsutil -m rsync gs://images.cocodataset.org/{CocoFiles.Train2017.value.filename} {root}",
        f"gsutil -m rsync gs://images.cocodataset.org/{CocoFiles.Val2017.value.filename} {root}",
        f"gsutil -m rsync gs://images.cocodataset.org/annotations {root}",
    ]
    assert repr(downloader) == f"CocoDownloader(root={root}, config={downloader.config})"

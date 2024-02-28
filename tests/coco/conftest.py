from pycocotools.coco import COCO as COCO
from pytest_cases import fixture
from pytorchcocotools.coco import COCO as COCOpt  # noqa: N811


@fixture(scope="session")
def path() -> str:
    path = "./data/example.json"
    return path


@fixture(scope="session")
def coco_np(path: str) -> COCO:
    return COCO(path)


@fixture(scope="session")
def coco_pt(path: str) -> COCOpt:
    return COCOpt(path)

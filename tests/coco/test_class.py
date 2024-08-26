from pycocotools.coco import COCO as COCOnp  # noqa: N811

from pytorchcocotools.coco import COCO as COCOpt  # noqa: N811


def test_class(coco_np: COCOnp, coco_pt: COCOpt) -> None:
    assert coco_np
    assert coco_pt

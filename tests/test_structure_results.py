import pytest

from pytorchcocotools.internal.structure.results import (
    CocoDensePoseResult,
    CocoImageCaptioningResult,
    CocoInstanceSegmentationnResult,
    CocoKeypointDetectionResult,
    CocoObjectDetectionResult,
    CocoPanopticSegmentationResult,
    CocoSegmentInfo,
    CocoStuffSegmentationResult,
)
from pytorchcocotools.internal.structure.rle import CocoRLE


@pytest.mark.parametrize(
    ("result_cls", "expected_keys"),
    [
        (CocoObjectDetectionResult, {"image_id", "category_id", "bbox", "score"}),
        (CocoInstanceSegmentationnResult, {"image_id", "category_id", "segmentation", "score"}),
        (CocoKeypointDetectionResult, {"image_id", "category_id", "keypoints", "score"}),
        (CocoStuffSegmentationResult, {"image_id", "category_id", "segmentation"}),
        (CocoSegmentInfo, {"id", "category_id"}),
        (CocoPanopticSegmentationResult, {"image_id", "file_name", "segments_info"}),
        (CocoImageCaptioningResult, {"image_id", "caption"}),
        (CocoDensePoseResult, {"image_id", "category_id", "bbox", "score", "uv_shape", "uv_data"}),
    ],
)
def test_result_entities_expose_expected_mapping_keys(result_cls, expected_keys: set[str]) -> None:
    result = result_cls()

    assert set(result.keys()) == expected_keys
    assert len(result) == len(expected_keys)


def test_result_entities_support_mapping_mutation() -> None:
    detection = CocoObjectDetectionResult()

    detection["score"] = 0.75
    detection.update({"bbox": [10.0, 20.0, 30.0, 40.0]})

    assert detection.score == 0.75
    assert detection["bbox"] == [10.0, 20.0, 30.0, 40.0]
    assert "bbox" in detection
    assert detection.pop("score") == 0.75
    assert detection.get("score") is None


def test_result_entities_accept_nested_payloads() -> None:
    segmentation = CocoRLE.from_dict({"counts": [1, 2, 3], "size": (4, 5)})
    instance = CocoInstanceSegmentationnResult(image_id=5, category_id=7, segmentation=segmentation, score=0.9)
    panoptic = CocoPanopticSegmentationResult(
        image_id=9,
        file_name="result.png",
        segments_info=[CocoSegmentInfo(id=3, category_id=11)],
    )
    caption = CocoImageCaptioningResult(image_id=4, caption="a caption")
    densepose = CocoDensePoseResult(
        image_id=1,
        category_id=2,
        bbox=[1.0, 2.0, 3.0, 4.0],
        score=0.5,
        uv_shape=(2, 3, 4),
        uv_data="encoded",
    )

    assert instance.segmentation.size == (4, 5)
    assert instance["segmentation"].counts == [1, 2, 3]
    assert panoptic.segments_info[0].category_id == 11
    assert caption.get("caption") == "a caption"
    densepose["uv_data"] = "updated"
    assert densepose.uv_data == "updated"

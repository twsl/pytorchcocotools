import pytest

from pytorchcocotools.utils.generation import generate_mock_data


def test_generate_mock_data_configures_image_and_category_counts() -> None:
    dataset, _ = generate_mock_data(
        "bbox",
        seed=0,
        num_images=3,
        num_categories=4,
        image_width=32,
        image_height=24,
        annotations_per_image=(1, 1),
        unmatched_detections_per_image=(0, 0),
    )

    assert len(dataset["images"]) == 3
    assert len(dataset["categories"]) == 4
    assert all(image["width"] == 32 and image["height"] == 24 for image in dataset["images"])
    assert len(dataset["annotations"]) == 3
    assert all(0 <= ann["bbox"][0] <= 32 - ann["bbox"][2] for ann in dataset["annotations"])


def test_generate_mock_data_configures_keypoint_categories_and_annotations() -> None:
    dataset, detections = generate_mock_data(
        "keypoints",
        seed=0,
        num_images=2,
        num_categories=3,
        num_keypoints=5,
        annotations_per_image=(1, 1),
        unmatched_detections_per_image=(1, 1),
    )

    assert len(dataset["categories"]) == 3
    assert all(len(category["keypoints"]) == 5 for category in dataset["categories"])
    assert all(len(annotation["keypoints"]) == 15 for annotation in dataset["annotations"])
    assert len(detections) >= 2
    assert all(len(detection["keypoints"]) == 15 for detection in detections)


def test_generate_mock_data_rejects_invalid_dimensions_and_counts() -> None:
    with pytest.raises(ValueError):
        generate_mock_data("keypoints", image_width=0)
    with pytest.raises(ValueError):
        generate_mock_data("keypoints", num_categories=0)
    with pytest.raises(ValueError):
        generate_mock_data("keypoints", num_keypoints=0)

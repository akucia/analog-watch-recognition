import numpy as np
import pytest

from watch_recognition.eval.segmentation_eval import iou_score_from_masks


@pytest.mark.iou_score_from_masks
def test_iou_score_from_masks_empty_target():
    target = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    predicted = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]
    )
    iou_score = iou_score_from_masks(predicted, target)
    assert iou_score == 0.0


@pytest.mark.iou_score_from_masks
def test_iou_score_from_masks_both_empty():
    target = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    predicted = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    iou_score = iou_score_from_masks(predicted, target)
    assert iou_score == 0.0


@pytest.mark.iou_score_from_masks
def test_iou_score_from_masks_exact():
    target = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]
    )
    predicted = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]
    )
    iou_score = iou_score_from_masks(predicted, target)
    assert iou_score == 1.0


@pytest.mark.iou_score_from_masks
def test_iou_score_from_masks_partial():
    target = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]
    )
    predicted = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    iou_score = iou_score_from_masks(predicted, target)
    print(iou_score)
    assert iou_score == pytest.approx(0.5, rel=1e-2)


@pytest.mark.iou_score_from_masks
def test_iou_score_from_masks_partial_2():
    target = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]
    )
    predicted = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 0, 0],
        ]
    )
    iou_score = iou_score_from_masks(predicted, target)
    print(iou_score)
    assert iou_score == pytest.approx(0.67, rel=1e-2)

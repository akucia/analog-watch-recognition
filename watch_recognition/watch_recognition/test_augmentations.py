import random
from unittest import TestCase

import albumentations as A
import numpy as np

from watch_recognition.datasets import DEFAULT_TRANSFORMS


class TestAugmentations(TestCase):
    def test_transformations_no_op(self):
        random.seed(0)
        np.random.seed(0)
        transforms = A.Compose(
            [],
            keypoint_params=A.KeypointParams(format="xyas", remove_invisible=False),
        )
        image = np.zeros((224, 224, 3))
        keypoints = np.array(
            [
                [0, 0, 0, 0],  # center
                [0, 0, 0, 0],  # top
                [0, 0, 0, 0],  # hour
                [0, 0, 0, 0],  # minute
            ]
        )
        keypoints[:, 0] *= image.shape[0]
        keypoints[:, 1] *= image.shape[1]
        data = {
            "image": image,
            "keypoints": keypoints,
        }
        aug_data = transforms(**data)
        aug_kp = np.array(aug_data["keypoints"])
        self.assertEqual(keypoints.shape[0], aug_kp.shape[0])
        self.assertTrue(np.all(keypoints == aug_kp))

    def test_transformations_kp_outside_image(self):
        random.seed(0)
        np.random.seed(0)
        transforms = A.Compose(
            [],
            keypoint_params=A.KeypointParams(format="xyas", remove_invisible=False),
        )
        image = np.zeros((224, 224, 3))
        keypoints = np.array(
            [
                [-1, -1, 0, 0],  # center
                [0, 0, 0, 0],  # top
                [0, 0, 0, 0],  # hour
                [0, 0, 0, 0],  # minute
            ]
        )
        keypoints[:, 0] *= image.shape[0]
        keypoints[:, 1] *= image.shape[1]
        data = {
            "image": image,
            "keypoints": keypoints,
        }
        aug_data = transforms(**data)
        aug_kp = np.array(aug_data["keypoints"])
        self.assertEqual(keypoints.shape[0], aug_kp.shape[0])
        self.assertTrue(np.all(keypoints == aug_kp))

    def test_transformations_kp_RandomSizedCrop(self):
        random.seed(0)
        np.random.seed(0)
        transforms = A.Compose(
            [
                A.RandomSizedCrop(
                    min_max_height=(200, 200), height=224, width=224, p=1
                ),
            ],
            keypoint_params=A.KeypointParams(format="xyas", remove_invisible=False),
        )
        image = np.zeros((224, 224, 3))
        keypoints = np.array(
            [
                [0.5, 0.5, 0, 0],  # center
                [0, 0, 0, 0],  # top
                [0, 0, 0, 0],  # hour
                [0, 0, 0, 0],  # minute
            ]
        )
        keypoints[:, 0] *= image.shape[0]
        keypoints[:, 1] *= image.shape[1]
        data = {
            "image": image,
            "keypoints": keypoints,
        }
        aug_data = transforms(**data)
        aug_kp = np.array(aug_data["keypoints"])
        self.assertEqual(keypoints.shape[0], aug_kp.shape[0])
        self.assertTrue(np.all(aug_kp[1:, 0] < 0))
        self.assertTrue(np.all(aug_kp[:1, 0] > 0))

    def test_transformations_kp_DEFAULT_TRANSFORMS(self):
        random.seed(0)
        np.random.seed(0)
        image = np.zeros((224, 224, 3)).astype("uint8")
        keypoints = np.array(
            [
                [0.5, 0.5, 0, 0],  # center
                [0, 0, 0, 0],  # top
                [0, 0, 0, 0],  # hour
                [0, 0, 0, 0],  # minute
            ]
        )
        keypoints[:, 0] *= image.shape[0]
        keypoints[:, 1] *= image.shape[1]
        data = {
            "image": image,
            "keypoints": keypoints,
        }
        aug_data = DEFAULT_TRANSFORMS(**data)
        aug_kp = np.array(aug_data["keypoints"])
        self.assertEqual(keypoints.shape[0], aug_kp.shape[0])
        self.assertTrue(np.all(aug_kp[:, 0] > 0))
        self.assertTrue(np.all(aug_kp[:, 1] > 0))

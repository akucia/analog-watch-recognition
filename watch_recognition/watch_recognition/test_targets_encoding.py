import unittest
from functools import partial

import numpy as np

from watch_recognition.targets_encoding import (
    convert_mask_outputs_to_keypoints,
    encode_keypoints_to_mask_np,
)
from watch_recognition.utilities import Point


class TestTargetsEncodingDecoding(unittest.TestCase):
    def test_kp_encode_decode_mask(self):
        image_size = (96, 96)
        mask_size = image_size
        encode_kp = partial(
            encode_keypoints_to_mask_np,
            image_size=(*image_size, 3),
            mask_size=mask_size,
            radius=1,
            include_background=False,
            separate_hour_and_minute_hands=False,
            add_perimeter=False,
            sparse=False,
            blur=True,
        )
        kp = np.array([[50, 45], [50, 20], [58, 38], [55, 66]])
        names = ["Center", "Top", "Hour", "Minute"]
        kp_target = [Point(*k, name=n) for k, n in zip(kp, names)]
        mask = encode_kp(kp)

        kp_pred = convert_mask_outputs_to_keypoints(mask)
        for a in kp_pred:
            for b in kp_target:
                if a.name == b.name:
                    self.assertLess(
                        a.distance(b),
                        1e-8,
                        msg=f"keypoint is {a.name} too far from the target",
                    )

    def test_kp_encode_decode_mask_with_perimeter(self):
        image_size = (96, 96)
        mask_size = image_size
        encode_kp = partial(
            encode_keypoints_to_mask_np,
            image_size=(*image_size, 3),
            mask_size=mask_size,
            radius=1,
            include_background=False,
            separate_hour_and_minute_hands=False,
            add_perimeter=True,
            with_perimeter_for_hands=True,
            sparse=False,
            blur=True,
        )
        kp = np.array([[50, 45], [50, 20], [58, 38], [55, 66]])
        names = ["Center", "Top", "Hour", "Minute"]
        kp_target = [Point(*k, name=n) for k, n in zip(kp, names)]
        mask = encode_kp(kp)

        kp_pred = convert_mask_outputs_to_keypoints(mask)
        for a in kp_pred:
            for b in kp_target:
                if a.name == b.name:
                    self.assertLess(
                        a.distance(b),
                        1e-8,
                        msg=f"keypoint is {a.name} too far from the target",
                    )


if __name__ == "__main__":
    unittest.main()

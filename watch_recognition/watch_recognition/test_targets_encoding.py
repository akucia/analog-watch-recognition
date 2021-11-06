import unittest
from functools import partial

import numpy as np
from skimage.draw import line

from watch_recognition.targets_encoding import (
    _blur_mask,
    convert_mask_outputs_to_keypoints,
    decode_keypoints_via_line_fits,
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

    def test_decode_keypoints_via_line_fits(self):
        image = np.zeros((128, 128))
        center = Point(64, 64)
        p1 = Point(100, 100)
        rr, cc = line(*center.as_coordinates_tuple, *p1.as_coordinates_tuple)
        image[rr, cc] = 1

        p2 = Point(25, 50)
        rr, cc = line(*center.as_coordinates_tuple, *p2.as_coordinates_tuple[::-1])
        image[rr, cc] = 1

        image = _blur_mask(image)

        p1_hat, p2_hat = decode_keypoints_via_line_fits(image, center)
        if p1_hat.distance(p1) > p1_hat.distance(p2):
            p1_hat, p2_hat = p2_hat, p1_hat
        self.assertAlmostEqual(p1_hat.x, p1.x, delta=3)
        self.assertAlmostEqual(p1_hat.y, p1.y, delta=3)
        self.assertAlmostEqual(p2_hat.x, p2.x, delta=3)
        self.assertAlmostEqual(p2_hat.y, p2.y, delta=3)


if __name__ == "__main__":
    unittest.main()

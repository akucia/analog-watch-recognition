import unittest
from functools import partial
from itertools import combinations_with_replacement
from typing import Tuple

import numpy as np
import tensorflow as tf
from time_loss import time_loss_np, time_loss_tf


class TestTimeLoss(tf.test.TestCase):
    def setUp(self):
        self.minutes = list(range(0, 60))
        self.hours = list(range(1, 13))

    def _str_time_to_np_arrays(self, s: str) -> Tuple[np.ndarray, np.ndarray]:
        hours, minutes = tuple(map(int, s.split(":")))
        return np.array([hours]), np.array([minutes])

    def test_loss_np_hours(self):
        real_time = self._str_time_to_np_arrays("2:00")
        pred_time = self._str_time_to_np_arrays("2:00")
        self.assertEqual(0, time_loss_np(real_time[0], pred_time[0]))

        real_time = self._str_time_to_np_arrays("2:00")
        pred_time = self._str_time_to_np_arrays("6:00")
        self.assertEqual(4, time_loss_np(real_time[0], pred_time[0]))

        real_time = self._str_time_to_np_arrays("11:00")
        pred_time = self._str_time_to_np_arrays("9:00")
        self.assertEqual(2, time_loss_np(real_time[0], pred_time[0]))

        real_time = self._str_time_to_np_arrays("11:00")
        pred_time = self._str_time_to_np_arrays("2:00")
        self.assertEqual(3, time_loss_np(real_time[0], pred_time[0]))

    def test_loss_np_minutes(self):
        minutes_loss_np = partial(time_loss_np, max_val=60)
        real_time = self._str_time_to_np_arrays("2:00")
        pred_time = self._str_time_to_np_arrays("2:00")
        self.assertEqual(0, minutes_loss_np(real_time[1], pred_time[1]))

        real_time = self._str_time_to_np_arrays("2:10")
        pred_time = self._str_time_to_np_arrays("2:00")
        self.assertEqual(10, minutes_loss_np(real_time[1], pred_time[1]))

        real_time = self._str_time_to_np_arrays("2:00")
        pred_time = self._str_time_to_np_arrays("2:10")
        self.assertEqual(10, minutes_loss_np(real_time[1], pred_time[1]))

        real_time = self._str_time_to_np_arrays("11:00")
        pred_time = self._str_time_to_np_arrays("10:59")
        # non zero loss on hour is intentional in this case, it might be changed
        # in the future
        self.assertEqual(1, time_loss_np(real_time[0], pred_time[0]))
        self.assertEqual(1, minutes_loss_np(real_time[1], pred_time[1]))

        real_time = self._str_time_to_np_arrays("12:15")
        pred_time = self._str_time_to_np_arrays("11:45")
        self.assertEqual(1, time_loss_np(real_time[0], pred_time[0]))
        self.assertEqual(30, minutes_loss_np(real_time[1], pred_time[1]))

    def test_time_loss_tf_equal_to_np(self):
        hours_combinations = np.array(
            [x for x in combinations_with_replacement(self.hours, 2)]
        ).astype(np.float)
        y_t = hours_combinations[:, 0].reshape(-1, 1)
        y_pred = hours_combinations[:, 1].reshape(-1, 1)

        np_losses = time_loss_np(y_t, y_pred)
        tf_losses = time_loss_tf(y_t, y_pred)
        diff = tf.reduce_sum(np_losses - tf_losses)

        self.assertAlmostEqual(0, diff)

        minutes_combinations = np.array(
            [x for x in combinations_with_replacement(self.minutes, 2)]
        ).astype(np.float)
        y_t = minutes_combinations[:, 0].reshape(-1, 1)
        y_pred = minutes_combinations[:, 1].reshape(-1, 1)

        np_losses = time_loss_np(y_t, y_pred, max_val=60)
        tf_losses = time_loss_tf(y_t, y_pred, max_val=60)
        diff = tf.reduce_sum(np_losses - tf_losses)

        self.assertAlmostEqual(0, diff)


if __name__ == "__main__":
    unittest.main()

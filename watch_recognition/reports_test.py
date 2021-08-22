import unittest

import numpy as np

from watch_recognition.reports import time_diff, time_diff_np


class ReportsTests(unittest.TestCase):
    def test_time_diff(self):
        a = 12, 00
        b = 12, 00
        expected_diff = 0
        result_diff = time_diff(a, b)
        self.assertEqual(expected_diff, result_diff)

        result_diff = time_diff(b, a)
        self.assertEqual(expected_diff, result_diff)

        a = 12, 58
        b = 1, 7
        expected_diff = 9
        result_diff = time_diff(a, b)
        self.assertEqual(expected_diff, result_diff)

        result_diff = time_diff(b, a)
        self.assertEqual(expected_diff, result_diff)

        a = 11, 7
        b = 1, 7
        expected_diff = 120
        result_diff = time_diff(a, b)
        self.assertEqual(expected_diff, result_diff)

        result_diff = time_diff(b, a)
        self.assertEqual(expected_diff, result_diff)

        a = 11, 7
        b = 10, 53
        expected_diff = 14
        result_diff = time_diff(a, b)
        self.assertEqual(expected_diff, result_diff)

        result_diff = time_diff(b, a)
        self.assertEqual(expected_diff, result_diff)

        a = 11, 59
        b = 12, 1
        expected_diff = 2
        result_diff = time_diff(a, b)
        self.assertEqual(expected_diff, result_diff)

        result_diff = time_diff(b, a)
        self.assertEqual(expected_diff, result_diff)

        a = 12, 59
        b = 1, 1
        expected_diff = 2
        result_diff = time_diff(a, b)
        self.assertEqual(expected_diff, result_diff)

        result_diff = time_diff(b, a)
        self.assertEqual(expected_diff, result_diff)

        a = 12, 59
        b = 12, 1
        expected_diff = 58
        result_diff = time_diff(a, b)
        self.assertEqual(expected_diff, result_diff)

        result_diff = time_diff(b, a)
        self.assertEqual(expected_diff, result_diff)

    def test_time_diff_np(self):
        a = np.array(
            [
                [12, 00],
                [12, 58],
                [11, 7],
                [11, 7],
                [11, 59],
                [12, 59],
                [12, 59],
            ]
        )
        b = np.array(
            [
                [12, 00],
                [1, 7],
                [1, 7],
                [10, 53],
                [12, 1],
                [1, 1],
                [12, 1],
            ]
        )
        expected = np.array(
            [
                [0],
                [9],
                [120],
                [14],
                [2],
                [2],
                [58],
            ]
        )
        result = time_diff_np(a, b)
        self.assertTrue(np.all(expected == result))

import unittest

from watch_recognition.models import points_to_time
from watch_recognition.utilities import Point


class TestPoints_to_time(unittest.TestCase):
    def test_points_to_time(self):
        with self.subTest("Test 9:45"):
            center = Point(0, 0).rename("Center")
            top = center.translate(0, 1).rename("Top")
            hour = center.translate(1, 0).rename("Hour")
            minute = center.translate(1, 0).rename("Minute")
            hour, minute = points_to_time(
                center=center, hour=hour, top=top, minute=minute
            )
            self.assertEqual(9.0, hour)
            self.assertEqual(45.0, minute)

        with self.subTest("Test 12:00"):
            center = Point(0, 0).rename("Center")
            top = center.translate(0, 1).rename("Top")
            hour = center.translate(0, 1).rename("Hour")
            minute = center.translate(0, 1).rename("Minute")
            hour, minute = points_to_time(
                center=center, hour=hour, top=top, minute=minute
            )
            self.assertEqual(12.0, hour)
            self.assertEqual(0.0, minute)

        with self.subTest("Test 12:00 rotated"):
            center = Point(0, 0).rename("Center")
            top = (
                center.translate(0, 1)
                .rename("Top")
                .rotate_around_origin_point(center, 90)
            )
            hour = (
                center.translate(0, 1)
                .rename("Hour")
                .rotate_around_origin_point(center, 90)
            )
            minute = (
                center.translate(0, 1)
                .rename("Minute")
                .rotate_around_origin_point(center, 90)
            )
            hour, minute = points_to_time(
                center=center, hour=hour, top=top, minute=minute
            )
            self.assertEqual(12.0, hour)
            self.assertEqual(0.0, minute)

        with self.subTest("Test 6:30"):
            center = Point(0, 0).rename("Center")
            top = center.translate(0, 1).rename("Top")
            hour = center.translate(0, -1).rename("Hour")
            minute = center.translate(0, -1).rename("Minute")
            hour, minute = points_to_time(
                center=center, hour=hour, top=top, minute=minute
            )
            self.assertEqual(6.0, hour)
            self.assertEqual(30.0, minute)

        with self.subTest("Test 3:15"):
            center = Point(0, 0).rename("Center")
            top = center.translate(0, 1).rename("Top")
            hour = center.translate(-1, 0).rename("Hour")
            minute = center.translate(-1, 0).rename("Minute")
            hour, minute = points_to_time(
                center=center, hour=hour, top=top, minute=minute
            )
            self.assertEqual(3.0, hour)
            self.assertEqual(15.0, minute)

        with self.subTest("Test 10:08 rotated"):
            center = Point(x=115, y=125.5, name="Center", score=1.0)
            top = Point(x=49, y=95, name="Top", score=1.0)
            hour = Point(x=63, y=162, name="Hour", score=None)
            minute = Point(x=96, y=41, name="Minute", score=None)

            hour, minute = points_to_time(
                center=center, hour=hour, top=top, minute=minute
            )
            self.assertEqual(10.0, hour, msg="Hour")
            self.assertEqual(8.0, minute, msg="Minute")


if __name__ == "__main__":
    unittest.main()

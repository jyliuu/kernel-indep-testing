import unittest

from src import calibration


class TestCalibration(unittest.TestCase):
    def test_calibration(self):
        print(calibration())


if __name__ == '__main__':
    unittest.main()

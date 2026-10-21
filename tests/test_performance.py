import unittest
from unittest.mock import patch

from gesture_manager.performance import ProcessMetricsSampler


class PerformanceSamplerTests(unittest.TestCase):
    def test_cpu_sample_accumulates_until_minimum_interval(self) -> None:
        sampler = ProcessMetricsSampler()
        sampler._last_wall = 10.0
        sampler._last_cpu = 4.0
        with patch("gesture_manager.performance.monotonic", return_value=10.2), \
             patch("gesture_manager.performance.process_time", return_value=4.1):
            self.assertIsNone(sampler.sample().cpu_percent)
        with patch("gesture_manager.performance.monotonic", return_value=10.6), \
             patch("gesture_manager.performance.process_time", return_value=4.3):
            sample = sampler.sample()
        self.assertAlmostEqual(sample.cpu_percent, 50.0)


if __name__ == "__main__":
    unittest.main()

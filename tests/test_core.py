"""CPU checks for metric conventions and release configuration; no dataset required."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
import common
from analysis_lib import angle, skel

spec = importlib.util.spec_from_file_location("run_experiments", REPO / "scripts" / "run_experiments.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class MetricTests(unittest.TestCase):
    def test_original_pixel_spacing_and_valid_landmark_pooling(self):
        predicted = np.array([[[3., 4.], [np.nan, np.nan]], [[0., 2.], [0., 0.]]])
        truth = np.zeros_like(predicted)
        mean, errors = common.mre_mm(predicted, truth, [0.1, 0.5])
        self.assertAlmostEqual(mean, 0.5)
        np.testing.assert_allclose(errors, [[0.5, np.nan], [1., 0.]], equal_nan=True)

    def test_sdr_includes_threshold_and_ignores_missing_landmarks(self):
        self.assertAlmostEqual(common.sdr(np.array([[2., 2.01, np.nan, 1.]]), 2.), 200. / 3.)

    def test_angles_and_class_boundaries(self):
        self.assertAlmostEqual(float(angle(np.array([1., 0.]), np.zeros(2), np.array([0., 1.]))), 90.)
        np.testing.assert_array_equal(skel(np.array([-1., 0., 4., 5.])), [3, 1, 1, 2])

    def test_main_and_controls_have_distinct_augmentation(self):
        main = launcher.configurations("main")
        self.assertEqual({c[1] for c in main}, {"aariz", "isbi"})
        self.assertTrue(all(c[0] == "s1_letter" and c[2] == "letter" and "--aug_photo" in c[3] for c in main))
        self.assertTrue(all(c[3] == [] for c in launcher.configurations("controls")))
        self.assertEqual(len(launcher.configurations("lodo")), 7)


if __name__ == "__main__":
    unittest.main()

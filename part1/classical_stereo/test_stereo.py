"""Numerical regression checks for units, PFM layout, masks and stereo signs."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import stereo


class StereoCorrectnessTests(unittest.TestCase):
    def test_middlebury_pfm_bottom_up_endianness_without_radiometric_scale(self):
        values = np.array([[1, 2, np.inf], [4, 5, 6]], np.float32)
        with tempfile.TemporaryDirectory(dir=stereo.ROOT, prefix="test_pfm_") as tmp:
            for endian, scale in [("<", -1.0), (">", 1.0), ("<", -0.003922)]:
                p = Path(tmp) / "test.pfm"
                p.write_bytes(f"Pf\n# comment\n3 2\n{scale}\n".encode()
                              + np.flipud(values).astype(endian + "f4").tobytes())
                np.testing.assert_array_equal(stereo.read_pfm(p), values)

    def test_metric_units_doffs_and_invalid_values(self):
        d = np.array([[20, 0, -10, np.nan]], np.float32)
        calib = {"cam0": [[1000, 0, 0], [0, 1000, 0], [0, 0, 1]],
                 "baseline": 100, "doffs": 10}
        z = stereo.depth_from_disparity(d, np.ones(d.shape, bool), calib)
        np.testing.assert_allclose(z[0, :2], [1000 * 100 / 30 / 1000, 10])
        self.assertTrue(np.isnan(z[0, 2:]).all())

    def test_fixed_point_invalid_sentinel_and_zero_disparity(self):
        raw = np.array([[-16, 0, 16, 255, 256]], np.int16)
        valid = stereo.prediction_valid(raw, {"min_disparity": 0}, 16)
        np.testing.assert_array_equal(valid, [[False, True, True, True, False]])
        self.assertEqual(float(raw[0, 2] / 16), 1.0)

    def test_mae_and_coverage_have_different_denominators(self):
        gt = np.array([[10, 20, 0, np.inf, 30]], np.float32)
        pred = np.array([[12, np.nan, 99, 99, 27]], np.float32)
        valid = np.isfinite(pred)
        metrics, error = stereo.evaluate(pred, valid, gt)
        self.assertEqual(metrics["gt_valid_pixels"], 3)
        self.assertEqual(metrics["evaluated_pixels"], 2)
        self.assertAlmostEqual(metrics["valid_fraction"], 2 / 3)
        self.assertEqual(metrics["mae_px"], 2.5)
        self.assertTrue(np.isnan(error[0, 1:4]).all())
        empty, _ = stereo.evaluate(pred, np.zeros(pred.shape, bool), gt)
        self.assertIsNone(empty["mae_px"])
        self.assertEqual(empty["valid_fraction"], 0)

    def test_left_right_sign_and_out_of_bounds(self):
        left = np.full((2, 8), 2.0, np.float32)
        right_raw = np.full((2, 8), -2 * 16, np.int16)
        residual, status = stereo.lr_diagnostic(left, np.ones(left.shape, bool), right_raw, -15, 1.0)
        self.assertTrue((status[:, :2] == 0).all())
        self.assertTrue((status[:, 2:] == 1).all())
        self.assertTrue((residual[:, 2:] == 0).all())
        right_raw[:, 0] = -6 * 16
        _, status = stereo.lr_diagnostic(left, np.ones(left.shape, bool), right_raw, -15, 1.0)
        self.assertTrue((status[:, 2] == 2).all())

    def test_actual_sgbm_recovers_known_horizontal_shift(self):
        # Independent geometry oracle: a textured plane at an exact 8px shift.
        rng = np.random.default_rng(42)
        left = rng.integers(0, 256, (80, 192), dtype=np.uint8)
        right = np.zeros_like(left)
        right[:, :-8] = left[:, 8:]
        config = json.loads((stereo.ROOT / "config.json").read_text())
        matcher = stereo.create_matcher(config, 16)
        raw = matcher.compute(left, right)
        recovered = raw[10:-10, 32:-16].astype(np.float32) / 16
        self.assertGreater(float((np.abs(recovered - 8) < .25).mean()), .99)


if __name__ == "__main__":
    unittest.main(verbosity=2)

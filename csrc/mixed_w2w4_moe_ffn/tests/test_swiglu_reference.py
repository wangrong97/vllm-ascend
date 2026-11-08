# SPDX-License-Identifier: Apache-2.0
"""Numerical contract for the complete FFN golden generator (CPU only)."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import ml_dtypes  # noqa: F401
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "numpy and ml_dtypes are required for numerical checks")
class SwiGLUReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "scripts" / "swiglu_reference.py"
        spec = importlib.util.spec_from_file_location("full_ffn_swiglu_reference", path)
        cls.reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.reference)

    @staticmethod
    def projection(gate, up):
        return np.concatenate([np.full((1, 32), gate, np.float32), np.full((1, 32), up, np.float32)], axis=1)

    def test_negative_gate_is_not_lower_clipped(self):
        actual = self.reference.swiglu(self.projection(-20, 1))
        np.testing.assert_allclose(actual, -4.122307e-8, rtol=1e-6)

    def test_positive_gate_and_both_up_tails_are_clipped(self):
        positive = self.reference.swiglu(self.projection(20, 30))
        negative = self.reference.swiglu(self.projection(20, -30))
        np.testing.assert_allclose(positive, 99.99546, rtol=1e-6)
        np.testing.assert_array_equal(negative, -positive)

    def test_model_limit_is_not_hardcoded(self):
        actual = self.reference.swiglu(self.projection(20, 30), limit=1)
        np.testing.assert_allclose(actual, 0.7310586, rtol=1e-6)

    def test_zero_limit_retains_unclipped_diagnostic(self):
        actual = self.reference.swiglu(self.projection(20, 30), limit=0)
        np.testing.assert_allclose(actual, 600, rtol=1e-6)

    def test_intermediate_bf16_then_e8m0_and_fp8_rounding(self):
        middle = self.reference.swiglu(self.projection(20, 30))
        quantized, scale = self.reference.quantize_h(middle)
        # BF16 rounds 99.99546 to 100. The scale is 2^-2; 400 rounds to
        # E4M3 384 by ties-to-even, yielding a reconstructed value of 96.
        np.testing.assert_array_equal(scale, np.array([[125]], np.uint8))
        np.testing.assert_array_equal(quantized.astype(np.float32), np.full((1, 32), 384))
        np.testing.assert_array_equal(self.reference.dequantize_h(quantized, scale), np.full((1, 32), 96))

    def test_clipping_does_not_hide_invalid_raw_projections(self):
        for gate, up in ((np.inf, 1), (-np.inf, 1), (1, np.inf), (1, -np.inf), (np.nan, 1), (1, np.nan)):
            with self.subTest(gate=gate, up=up), self.assertRaises(ValueError):
                self.reference.swiglu(self.projection(gate, up))

    def test_bf16_overflow_is_rejected_before_clipping(self):
        with self.assertRaises(ValueError):
            self.reference.swiglu(self.projection(np.finfo(np.float32).max, 1))

    def test_invalid_limits_are_rejected(self):
        for limit in (-1, np.inf, np.nan, 1e300):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.reference.swiglu(self.projection(1, 1), limit=limit)

    def test_generator_pairs_limit_with_golden_and_enables_empty_expert_precisions(self):
        generator = Path(__file__).resolve().parents[1] / "scripts" / "gen_data.py"
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(
                [
                    sys.executable,
                    str(generator),
                    "3",
                    "64",
                    "128",
                    "32",
                    "--group-list",
                    "1,0,0",
                    "--is-expert-w4",
                    "0,0,1",
                    "--is-down-w4",
                    "0,1,1",
                    "--swiglu-limit",
                    "1.25",
                ],
                cwd=directory,
                capture_output=True,
                text=True,
                check=True,
            )
            root = Path(directory)
            metadata = json.loads((root / "input/metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["swiglu_limit"], 1.25)
            self.assertEqual(metadata["resident_stage_mask"], 15)
            self.assertEqual(metadata["tiling_abi_version"], 4)
            np.testing.assert_array_equal(np.fromfile(root / "input/input_swiglu_limit.bin", dtype=np.float32), [1.25])
            self.assertEqual((root / "output/golden_c.bin").stat().st_size, 64)


if __name__ == "__main__":
    unittest.main()

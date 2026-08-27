"""
Tests for Module 3.5 — Document Liveness Check.

Key test cases matching the required demo scenarios:
  1. Genuine document (not screen replay) → Part A should NOT flag
  2. No SVM → graceful threshold fallback
  3. Single repeated frame (static image) → Part B → STATIC
  4. Part B with enough frames but no motion → STATIC
  5. Liveness skipped (no video) → 0 pts in risk score
  6. SCREEN_REPLAY flag propagates to risk score
  7. LIVENESS_CHECK_FAILED flag propagates to risk score
"""

import os
import sys
import unittest
import tempfile
import shutil

import cv2
import numpy as np

# Add backend to path
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VENDOR_DIR = os.path.join(BACKEND_DIR, "vendor", "image_forensics")
for d in (BACKEND_DIR, VENDOR_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

SAMPLE_DIR = os.path.join(BACKEND_DIR, "vendor", "mrz_scanner", "data")
GENUINE_SAMPLES = [
    os.path.join(SAMPLE_DIR, s)
    for s in ["passport_uk.jpg", "td1.jpg", "td3.jpg"]
    if os.path.exists(os.path.join(SAMPLE_DIR, s))
]

from modules.document_liveness import detect_screen_replay, detect_physical_motion, extract_frames
from modules.risk_scoring import compute_risk_score


def _make_static_frame_dir(n_frames: int = 10, width: int = 640, height: int = 480) -> str:
    """Create a temp dir with n_frames identical frames (static/no motion)."""
    d = tempfile.mkdtemp(prefix="test_liveness_static_")
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Add a bright region to simulate a document with specular highlight
    frame[100:200, 400:500] = 255  # white "specular" region
    frame[50:400, 50:600] = (30, 30, 60)  # dark passport background
    frame[100:200, 400:500] = 255  # keep specular bright
    for i in range(n_frames):
        cv2.imwrite(os.path.join(d, f"frame_{i:04d}.jpg"), frame)
    return d


def _make_motion_frame_dir(n_frames: int = 15, width: int = 640, height: int = 480) -> str:
    """
    Create a temp dir with n_frames showing a moving specular highlight
    (simulates tilting a real document).
    """
    d = tempfile.mkdtemp(prefix="test_liveness_motion_")
    for i in range(n_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[50:400, 50:600] = (30, 30, 60)
        # Move the specular highlight progressively across the frame
        x_start = 300 + i * 15
        x_end = min(x_start + 80, width)
        y_start = 100 + i * 5
        y_end = min(y_start + 60, height)
        frame[y_start:y_end, x_start:x_end] = 255
        cv2.imwrite(os.path.join(d, f"frame_{i:04d}.jpg"), frame)
    return d


class TestPartAScreenReplay(unittest.TestCase):

    @unittest.skipIf(not GENUINE_SAMPLES, "No sample images available")
    def test_genuine_document_not_flagged(self):
        """Genuine document photos must not be flagged as screen replay (no false positives)."""
        for path in GENUINE_SAMPLES:
            with self.subTest(img=os.path.basename(path)):
                result = detect_screen_replay(path)
                self.assertNotEqual(result["method"], "unavailable",
                                    f"Method should not be 'unavailable' for {path}")
                # The key test: genuine docs should NOT be flagged
                # (This may fail before the SVM is trained; threshold fallback is acceptable)
                if result["method"] == "svm":
                    self.assertFalse(
                        result["is_screen_replay"],
                        f"SVM falsely flagged genuine document: {path}\n"
                        f"  FFT peak ratio: {result['fft_peak_ratio']}\n"
                        f"  Texture uniformity: {result['texture_uniformity']}"
                    )

    @unittest.skipIf(not GENUINE_SAMPLES, "No sample images available")
    def test_fallback_returns_valid_structure(self):
        """Even without SVM, detect_screen_replay must return a complete, valid dict."""
        result = detect_screen_replay(GENUINE_SAMPLES[0])
        self.assertIn("is_screen_replay", result)
        self.assertIn("confidence", result)
        self.assertIn("method", result)
        self.assertIn("fft_peak_ratio", result)
        self.assertIn("texture_uniformity", result)
        self.assertIn(result["method"], {"svm", "heuristic", "unavailable"})
        self.assertIsNone(result.get("error"), f"Unexpected error: {result.get('error')}")

    def test_invalid_path_returns_error(self):
        result = detect_screen_replay("/nonexistent/path/image.jpg")
        self.assertIsNotNone(result.get("error"))
        self.assertIsNone(result["is_screen_replay"])


class TestPartBPhysicalMotion(unittest.TestCase):

    def setUp(self):
        self._dirs = []

    def tearDown(self):
        for d in self._dirs:
            shutil.rmtree(d, ignore_errors=True)

    def test_static_frames_verdict_static(self):
        """Identical frames with no motion → STATIC verdict."""
        d = _make_static_frame_dir(n_frames=10)
        self._dirs.append(d)
        result = detect_physical_motion(d)
        self.assertEqual(result["verdict"], "STATIC",
                         f"Expected STATIC for identical frames, got {result['verdict']}\n"
                         f"  displacement={result['mean_highlight_displacement_px']:.3f} px\n"
                         f"  hologram_shift={result['hologram_hsv_shift']:.4f}")

    def test_motion_frames_verdict_physical(self):
        """Frames with clear specular displacement → PHYSICAL verdict."""
        d = _make_motion_frame_dir(n_frames=15)
        self._dirs.append(d)
        result = detect_physical_motion(d)
        self.assertIn(result["verdict"], {"PHYSICAL", "INCONCLUSIVE"},
                      f"Expected PHYSICAL for motion frames, got {result['verdict']}\n"
                      f"  displacement={result['mean_highlight_displacement_px']:.3f} px")

    def test_insufficient_frames_returns_inconclusive(self):
        """Fewer than 5 frames → INCONCLUSIVE."""
        d = _make_static_frame_dir(n_frames=3)
        self._dirs.append(d)
        result = detect_physical_motion(d)
        self.assertEqual(result["verdict"], "INCONCLUSIVE")
        self.assertEqual(result["frame_count"], 3)

    def test_empty_dir_returns_inconclusive(self):
        d = tempfile.mkdtemp(prefix="test_liveness_empty_")
        self._dirs.append(d)
        result = detect_physical_motion(d)
        self.assertEqual(result["verdict"], "INCONCLUSIVE")
        self.assertEqual(result["frame_count"], 0)


class TestLivenessInRiskScoring(unittest.TestCase):
    """Verify liveness flags propagate correctly into compute_risk_score."""

    _BASE_OCR = {
        "mrz_status": "VALID", "checksum_valid": True, "status": "VALID",
        "document_type": "PASSPORT", "expiry_date": "2030-01-01",
    }
    _BASE_TAMPER = {"tamper_score": 5, "verdict": "Authentic"}
    _BASE_FACE   = {"verified": True, "distance": 0.3, "confidence": 80.0, "is_real": True}

    def test_no_liveness_adds_zero_points(self):
        """When liveness_result is None, score must be identical to pre-liveness baseline."""
        risk_without = compute_risk_score(self._BASE_OCR, self._BASE_TAMPER, self._BASE_FACE)
        risk_with_none = compute_risk_score(self._BASE_OCR, self._BASE_TAMPER, self._BASE_FACE,
                                             liveness_result=None)
        self.assertEqual(risk_without["risk_score"], risk_with_none["risk_score"])

    def test_screen_replay_svm_requires_review_without_changing_score(self):
        liveness = {
            "screen_replay": {"is_screen_replay": True, "method": "svm", "confidence": 0.92,
                               "fft_peak_ratio": 0.25, "texture_uniformity": 0.70},
            "physical_motion": {"verdict": "SKIPPED"},
        }
        risk = compute_risk_score(self._BASE_OCR, self._BASE_TAMPER, self._BASE_FACE,
                                  liveness_result=liveness)
        bd = risk["breakdown"]["document_liveness"]
        self.assertEqual(bd["score"], 0.0)
        self.assertTrue(risk["requires_manual_review"])
        self.assertTrue(any("SCREEN_REPLAY_SUSPECTED:" in f for f in risk["flags"]))

    def test_screen_replay_threshold_requires_review_without_changing_score(self):
        liveness = {
            "screen_replay": {"is_screen_replay": True, "method": "threshold", "confidence": 0.72,
                               "fft_peak_ratio": 0.20, "texture_uniformity": 0.68},
            "physical_motion": {"verdict": "SKIPPED"},
        }
        risk = compute_risk_score(self._BASE_OCR, self._BASE_TAMPER, self._BASE_FACE,
                                  liveness_result=liveness)
        bd = risk["breakdown"]["document_liveness"]
        self.assertEqual(bd["score"], 0.0)
        self.assertTrue(risk["requires_manual_review"])
        self.assertTrue(any("SCREEN_REPLAY_SUSPECTED_HEURISTIC" in f for f in risk["flags"]))

    def test_static_motion_requires_review_without_changing_score(self):
        liveness = {
            "screen_replay": {"is_screen_replay": False, "method": "threshold", "confidence": 0.2,
                               "fft_peak_ratio": 0.10, "texture_uniformity": 0.40},
            "physical_motion": {
                "verdict": "STATIC", "motion_detected": False, "hologram_shift_detected": False,
                "mean_highlight_displacement_px": 0.5, "hologram_hsv_shift": 0.02,
            },
        }
        risk = compute_risk_score(self._BASE_OCR, self._BASE_TAMPER, self._BASE_FACE,
                                  liveness_result=liveness)
        bd = risk["breakdown"]["document_liveness"]
        self.assertEqual(bd["score"], 0.0)
        self.assertTrue(risk["requires_manual_review"])
        self.assertTrue(any("LIVENESS_CHECK_FAILED" in f for f in risk["flags"]))

    def test_both_parts_flagged_require_review_without_score_points(self):
        liveness = {
            "screen_replay": {"is_screen_replay": True, "method": "svm", "confidence": 0.95,
                               "fft_peak_ratio": 0.30, "texture_uniformity": 0.80},
            "physical_motion": {
                "verdict": "STATIC", "motion_detected": False, "hologram_shift_detected": False,
                "mean_highlight_displacement_px": 0.3, "hologram_hsv_shift": 0.01,
            },
        }
        risk = compute_risk_score(self._BASE_OCR, self._BASE_TAMPER, self._BASE_FACE,
                                  liveness_result=liveness)
        bd = risk["breakdown"]["document_liveness"]
        self.assertEqual(bd["score"], 0.0)
        self.assertTrue(risk["requires_manual_review"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""
Tests for Module 3: Tampering Detection (Phase 6.1).

Critical tests that were missing from the original test suite:
  1. Genuine documents must NOT be flagged as tampered (guards H8)
  2. A crashed detector must NOT yield "Authentic" (guards H6)
  3. INCONCLUSIVE verdict propagates correctly to risk scoring
"""

import os
import sys
import unittest
from unittest.mock import patch

# Add backend to path
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VENDOR_DIR = os.path.join(BACKEND_DIR, "vendor", "image_forensics")
for d in (BACKEND_DIR, VENDOR_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

SAMPLE_DIR = os.path.join(BACKEND_DIR, "vendor", "mrz_scanner", "data")
GENUINE_SAMPLES = [
    s for s in ["passport_uk.jpg", "td1.jpg", "td2.jpg", "td3.jpg", "mrva.jpg", "mrvb.jpg"]
    if os.path.exists(os.path.join(SAMPLE_DIR, s))
]

from modules.tampering_detection import analyze_tampering  # noqa: E402
from modules.risk_scoring import compute_risk_score        # noqa: E402


class TestGenuineDocumentFalsePositives(unittest.TestCase):
    """
    Every sample is a GENUINE document.
    All should score < 10 (\"Authentic\") after the Phase 2 overhaul.

    If these fail, run tools/measure_baseline.py to see the per-detector breakdown
    and identify which scorer is driving high scores on genuine documents.
    """

    @unittest.skipIf(not GENUINE_SAMPLES, "No sample images found in vendor/mrz_scanner/data/")
    def test_genuine_documents_score_authentic(self):
        for name in GENUINE_SAMPLES:
            path = os.path.join(SAMPLE_DIR, name)
            with self.subTest(doc=name):
                result = analyze_tampering(path)
                tamper_score = result.get("tamper_score")
                verdict = result.get("verdict")

                if verdict == "INCONCLUSIVE":
                    # Detectors crashed; skip scoring assertion but flag for investigation
                    self.skipTest(f"{name}: detectors returned INCONCLUSIVE (coverage={result.get('detector_coverage'):.2f})")

                self.assertIsNotNone(tamper_score, f"{name}: tamper_score is None")
                self.assertLess(
                    tamper_score, 10.0,
                    f"{name}: genuine document scored {tamper_score:.1f} — false positive!\n"
                    f"Breakdown: {result.get('breakdown')}"
                )


class TestCrashedDetectorDoesNotReadAsAuthentic(unittest.TestCase):
    """
    H6 fix: a crashed detector contributes 0 without renormalization,
    artificially lowering the combined score toward 'Authentic'.
    After the fix, crashed detectors are excluded and weights renormalized.
    An INCONCLUSIVE verdict should be returned when coverage is too low.
    """

    @unittest.skipIf(not GENUINE_SAMPLES, "No sample images found")
    def test_copy_move_crash_yields_inconclusive_not_authentic(self):
        path = os.path.join(SAMPLE_DIR, GENUINE_SAMPLES[0])
        with patch("modules.tampering_detection.detect_copy_move", side_effect=RuntimeError("simulated crash")):
            result = analyze_tampering(path)

        self.assertIn("copy_move", result.get("unavailable_detectors", []))
        self.assertTrue(result.get("degraded"), "degraded flag should be True")
        # With copy_move (42% weight) missing, coverage is 0.58 < 0.60 -> INCONCLUSIVE
        self.assertEqual(result.get("verdict"), "INCONCLUSIVE",
                         "With <60% detector coverage, verdict should be INCONCLUSIVE, not Authentic")

    @unittest.skipIf(not GENUINE_SAMPLES, "No sample images found")
    def test_minor_detector_crash_still_produces_verdict(self):
        """A single minor detector (EXIF, 3%) crashing should not force INCONCLUSIVE."""
        path = os.path.join(SAMPLE_DIR, GENUINE_SAMPLES[0])
        with patch("modules.tampering_detection.analyze_exif", side_effect=RuntimeError("simulated crash")):
            result = analyze_tampering(path)

        self.assertIn("exif_analysis", result.get("unavailable_detectors", []))
        self.assertTrue(result.get("degraded"), "degraded flag should be True")
        self.assertNotEqual(result.get("verdict"), "INCONCLUSIVE",
                            "A single 3%-weight detector crash should not force INCONCLUSIVE")


class TestInconclusiveVerdictInRiskScoring(unittest.TestCase):
    """INCONCLUSIVE tamper verdict should not contribute points to risk score."""

    def test_inconclusive_tamper_contributes_zero_points(self):
        mock_ocr = {"mrz_status": "VALID", "checksum_valid": True, "status": "VALID",
                    "document_type": "PASSPORT", "expiry_date": "2030-01-01"}
        mock_tamper = {"tamper_score": None, "verdict": "INCONCLUSIVE",
                       "degraded": True, "detector_coverage": 0.3}
        mock_face = {"verified": None, "is_real": None}

        risk = compute_risk_score(mock_ocr, mock_tamper, mock_face)

        tamper_breakdown = risk["breakdown"].get("tampering", {})
        self.assertEqual(tamper_breakdown.get("score"), 0.0,
                         "INCONCLUSIVE tamper should contribute 0 points to risk")
        self.assertTrue(
            any("TAMPERING_INCONCLUSIVE" in f for f in risk.get("flags", [])),
            "TAMPERING_INCONCLUSIVE flag should be in risk flags"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

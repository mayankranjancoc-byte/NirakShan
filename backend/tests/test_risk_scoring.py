"""
Focused automated tests for backend/modules/risk_scoring.py

Validates:
1. Case 1: No selfie supplied (verified=None) -> SKIPPED (0 penalty, no mismatch flag)
2. Case 2: Matching selfie (verified=True) -> MATCH (0 penalty, verified=True)
3. Case 3: Mismatching selfie (verified=False) -> MISMATCH (+20 penalty, FACE_MISMATCH flag)
4. Case 4: Face verification error -> ERROR (+10 penalty, FACE_VERIFICATION_ERROR flag)
5. Case 5: Face spoofing -> SPOOF (+10 penalty, FACE_SPOOF_DETECTED flag)
"""

import os
import sys
import unittest

# Ensure backend root is on sys.path
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from modules.risk_scoring import compute_risk_score


class TestRiskScoringFaceVerification(unittest.TestCase):
    def setUp(self):
        # Baseline mock OCR result (valid passport, not expired)
        self.mock_ocr_valid = {
            "status": "SUCCESS",
            "checksum_valid": True,
            "expiry_date": "2030-01-01",
            "document_number": "123456789",
        }
        # Baseline mock tamper result (authentic document)
        self.mock_tamper_authentic = {
            "tamper_score": 5.0,
            "verdict": "Authentic",
            "breakdown": {},
        }

    def test_case_1_no_selfie_supplied(self):
        """
        Case 1: No selfie supplied (verified=None)
        Must NOT penalize as a face mismatch.
        Must contribute 0.0 risk points for face verification.
        """
        face_result = {
            "verified": None,
            "confidence": None,
            "distance": None,
            "is_real": None,
            "note": "No selfie provided -- face verification skipped",
        }

        result = compute_risk_score(
            self.mock_ocr_valid,
            self.mock_tamper_authentic,
            face_result,
        )

        face_breakdown = result["breakdown"]["face_verification"]

        # Assertions
        self.assertEqual(face_breakdown["score"], 0.0)
        self.assertIsNone(face_breakdown["verified"])
        self.assertIsNone(face_breakdown["is_real"])

        # Flags assertions
        for flag in result["flags"]:
            self.assertNotIn("FACE_MISMATCH", flag)
            self.assertNotIn("FACE_SPOOF_DETECTED", flag)
            self.assertNotIn("FACE_VERIFICATION_ERROR", flag)

        # Baseline risk should be purely tamper (5 * 0.3 = 1.5).
        # No selfie means face liveness is skipped, not failed.
        self.assertEqual(result["risk_score"], 1.5)
        self.assertEqual(result["verdict"], "LOW")

    def test_case_2_selfie_matching(self):
        """
        Case 2: Selfie supplied and faces match (verified=True)
        """
        face_result = {
            "verified": True,
            "confidence": 100.0,
            "distance": 0.0,
            "threshold": 0.68,
            "is_real": None,
        }

        result = compute_risk_score(
            self.mock_ocr_valid,
            self.mock_tamper_authentic,
            face_result,
        )

        face_breakdown = result["breakdown"]["face_verification"]
        self.assertEqual(face_breakdown["score"], 3.0)
        self.assertTrue(face_breakdown["verified"])
        self.assertFalse(any("FACE_MISMATCH" in f for f in result["flags"]))

    def test_case_3_selfie_mismatch(self):
        """
        Case 3: Selfie supplied and faces do NOT match (verified=False)
        Must apply the 20.0 penalty and emit a FACE_MISMATCH flag.
        """
        face_result = {
            "verified": False,
            "confidence": 0.28,
            "distance": 0.9687,
            "threshold": 0.68,
            "is_real": None,
        }

        result = compute_risk_score(
            self.mock_ocr_valid,
            self.mock_tamper_authentic,
            face_result,
        )

        face_breakdown = result["breakdown"]["face_verification"]
        self.assertEqual(face_breakdown["score"], 23.0)
        self.assertFalse(face_breakdown["verified"])
        self.assertTrue(any("FACE_MISMATCH" in f for f in result["flags"]))

    def test_case_4_face_verification_error(self):
        """
        Case 4: Face verification encountered an error
        Must apply 10.0 error penalty and emit a FACE_VERIFICATION_ERROR flag,
        without being converted into a clean mismatch.
        """
        face_result = {
            "verified": False,
            "confidence": 0.0,
            "is_real": None,
            "error": "Face could not be detected in selfie",
        }

        result = compute_risk_score(
            self.mock_ocr_valid,
            self.mock_tamper_authentic,
            face_result,
        )

        face_breakdown = result["breakdown"]["face_verification"]
        self.assertEqual(face_breakdown["score"], 13.0)
        self.assertTrue(any("FACE_VERIFICATION_ERROR" in f for f in result["flags"]))
        self.assertFalse(any("FACE_MISMATCH" in f for f in result["flags"]))

    def test_case_5_spoof_detected(self):
        """
        Case 5: Anti-spoofing flags a fake face (is_real=False)
        Must apply 10.0 spoof penalty and emit FACE_SPOOF_DETECTED flag.
        """
        face_result = {
            "verified": True,
            "confidence": 98.0,
            "distance": 0.1,
            "is_real": False,
        }

        result = compute_risk_score(
            self.mock_ocr_valid,
            self.mock_tamper_authentic,
            face_result,
        )

        face_breakdown = result["breakdown"]["face_verification"]
        self.assertEqual(face_breakdown["score"], 10.0)
        self.assertTrue(any("FACE_SPOOF_DETECTED" in f for f in result["flags"]))


if __name__ == "__main__":
    unittest.main()

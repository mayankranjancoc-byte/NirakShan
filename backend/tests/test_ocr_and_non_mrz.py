"""
Focused automated tests for OCR, FastMRZ, non-MRZ document handling, and risk scoring integration.

Test cases:
1. Valid passport with MRZ -> status=VALID, mrz_status=VALID, checksum_valid=True, 0 penalty
2. Invalid passport MRZ -> status=INVALID, mrz_status=INVALID, checksum_valid=False, +25 penalty, MRZ_CHECKSUM_INVALID flag
3. Passport where MRZ cannot be read -> status=EXTRACTION_FAILED, mrz_status=EXTRACTION_FAILED, +25 penalty, MRZ_EXTRACTION_FAILED flag
4. Legitimate non-MRZ document (NON_MRZ_ID) -> status=NOT_APPLICABLE, mrz_status=NOT_APPLICABLE, 0 penalty, NO MRZ_EXTRACTION_FAILED flag
5. Unknown document type (no MRZ found) -> status=DOCUMENT_TYPE_UNKNOWN, mrz_status=NOT_DETERMINED, 0 penalty, NO fraud penalty
"""

import os
import sys
import unittest

# Ensure backend root is on sys.path
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from modules.ocr_extraction import extract_document_fields
from modules.risk_scoring import compute_risk_score

SAMPLE_DIR = os.path.join(BACKEND_DIR, "vendor", "fastmrz", "data")
PASSPORT_IMG = os.path.join(SAMPLE_DIR, "passport_uk.jpg")
NOMRZ_IMG = os.path.join(SAMPLE_DIR, "nomrz.jpg")


class TestNonMRZAndOCRHandling(unittest.TestCase):
    def setUp(self):
        # Baseline authentic tamper and skipped face results
        self.mock_tamper = {
            "tamper_score": 0.0,
            "verdict": "Authentic",
            "breakdown": {},
        }
        self.mock_face = {
            "verified": None,
            "confidence": None,
            "is_real": None,
        }

    def test_case_1_valid_passport_with_mrz(self):
        """
        Case 1: Valid passport with MRZ
        FastMRZ extracts fields and confirms checksum validity.
        Must receive 0 penalty for MRZ.
        """
        if not os.path.exists(PASSPORT_IMG):
            self.skipTest(f"Sample image not found: {PASSPORT_IMG}")

        ocr_result = extract_document_fields(PASSPORT_IMG, document_type="PASSPORT")

        self.assertEqual(ocr_result["status"], "VALID")
        self.assertEqual(ocr_result["mrz_status"], "VALID")
        self.assertTrue(ocr_result["checksum_valid"])
        self.assertIn("document_number", ocr_result)

        risk = compute_risk_score(ocr_result, self.mock_tamper, self.mock_face)
        # MRZ itself should have 0 penalty (only expiry if date in past)
        self.assertFalse(any("MRZ_EXTRACTION_FAILED" in f for f in risk["flags"]))
        self.assertFalse(any("MRZ_CHECKSUM_INVALID" in f for f in risk["flags"]))

    def test_case_2_invalid_passport_mrz(self):
        """
        Case 2: Passport MRZ with invalid check digits
        Must receive +25 penalty and MRZ_CHECKSUM_INVALID flag.
        """
        mock_invalid_ocr = {
            "document_type": "PASSPORT",
            "mrz_type": "TD3",
            "status": "INVALID",
            "mrz_status": "INVALID",
            "checksum_valid": False,
            "document_number": "707797979",
            "error": "Document number checksum is not matching",
        }

        risk = compute_risk_score(mock_invalid_ocr, self.mock_tamper, self.mock_face)

        self.assertEqual(risk["breakdown"]["mrz_validation"]["score"], 25.0)
        self.assertTrue(any("MRZ_CHECKSUM_INVALID" in f for f in risk["flags"]))
        self.assertFalse(any("MRZ_EXTRACTION_FAILED" in f for f in risk["flags"]))

    def test_case_3_passport_mrz_unreadable(self):
        """
        Case 3: Document expected to be a PASSPORT, but MRZ cannot be read
        Must receive +25 penalty and MRZ_EXTRACTION_FAILED flag.
        """
        if not os.path.exists(NOMRZ_IMG):
            self.skipTest(f"Sample image not found: {NOMRZ_IMG}")

        ocr_result = extract_document_fields(NOMRZ_IMG, document_type="PASSPORT")

        self.assertEqual(ocr_result["status"], "EXTRACTION_FAILED")
        self.assertEqual(ocr_result["mrz_status"], "EXTRACTION_FAILED")
        self.assertFalse(ocr_result["checksum_valid"])

        risk = compute_risk_score(ocr_result, self.mock_tamper, self.mock_face)

        self.assertEqual(risk["breakdown"]["mrz_validation"]["score"], 25.0)
        self.assertTrue(any("MRZ_EXTRACTION_FAILED" in f for f in risk["flags"]))

    def test_case_4_legitimate_non_mrz_document(self):
        """
        Case 4: Legitimate Non-MRZ Document (NON_MRZ_ID / Driving License)
        Must NOT receive MRZ_EXTRACTION_FAILED flag.
        Must contribute 0 fraud penalty for MRZ.
        Tesseract OCR extracts visible text.
        """
        if not os.path.exists(NOMRZ_IMG):
            self.skipTest(f"Sample image not found: {NOMRZ_IMG}")

        ocr_result = extract_document_fields(NOMRZ_IMG, document_type="NON_MRZ_ID")

        self.assertEqual(ocr_result["document_type"], "NON_MRZ_ID")
        self.assertEqual(ocr_result["status"], "NOT_APPLICABLE")
        self.assertEqual(ocr_result["mrz_status"], "NOT_APPLICABLE")
        self.assertIsNone(ocr_result["checksum_valid"])
        self.assertIn("extracted_text", ocr_result)
        self.assertTrue(len(ocr_result["extracted_text"]) > 0)

        risk = compute_risk_score(ocr_result, self.mock_tamper, self.mock_face)

        # Crucial assertions: 0 penalty, no extraction failure flag
        self.assertEqual(risk["breakdown"]["mrz_validation"]["score"], 0.0)
        self.assertFalse(any("MRZ_EXTRACTION_FAILED" in f for f in risk["flags"]))
        self.assertFalse(any("MRZ_CHECKSUM_INVALID" in f for f in risk["flags"]))
        self.assertEqual(risk["risk_score"], 0.0)
        self.assertEqual(risk["verdict"], "LOW")

    def test_case_5_unknown_document_type(self):
        """
        Case 5: Unknown document type with no MRZ detected
        Must NOT automatically assign a fraud penalty.
        Pipeline continues with status DOCUMENT_TYPE_UNKNOWN / NOT_DETERMINED.
        """
        if not os.path.exists(NOMRZ_IMG):
            self.skipTest(f"Sample image not found: {NOMRZ_IMG}")

        ocr_result = extract_document_fields(NOMRZ_IMG, document_type="UNKNOWN")

        self.assertEqual(ocr_result["document_type"], "UNKNOWN")
        self.assertEqual(ocr_result["status"], "DOCUMENT_TYPE_UNKNOWN")
        self.assertEqual(ocr_result["mrz_status"], "NOT_DETERMINED")
        self.assertIsNone(ocr_result["checksum_valid"])
        self.assertIn("extracted_text", ocr_result)

        risk = compute_risk_score(ocr_result, self.mock_tamper, self.mock_face)

        # Crucial assertions: 0 penalty, no extraction failure flag
        self.assertEqual(risk["breakdown"]["mrz_validation"]["score"], 0.0)
        self.assertFalse(any("MRZ_EXTRACTION_FAILED" in f for f in risk["flags"]))
        self.assertFalse(any("MRZ_CHECKSUM_INVALID" in f for f in risk["flags"]))
        self.assertEqual(risk["risk_score"], 0.0)
        self.assertEqual(risk["verdict"], "LOW")


if __name__ == "__main__":
    unittest.main()

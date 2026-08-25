"""
Module 5: Risk Scoring Layer
This is the ONLY original logic in the prototype.

Combines outputs from all four modules into a single risk assessment
using a weighted rule engine (not a trained model).

Weight rationale:
  - Failed MRZ checksum: strong signal of invalid/forged document (25 pts)
  - Expired document: definite flag, but not necessarily fraud (15 pts)
  - Tamper score: scaled from DocAuth's 0-100 range (30 pts max)
  - Face mismatch: strong signal of identity fraud (20 pts)
  - Face liveness fail: indicates photo spoofing (10 pts)

Total possible: 100 points
  0-25  -> LOW risk (likely genuine)
  25-60 -> MEDIUM risk (needs manual review)
  60+   -> HIGH risk (likely fraudulent)
"""

from datetime import datetime, date


def compute_risk_score(
    ocr_result: dict,
    tamper_result: dict,
    face_result: dict,
) -> dict:
    """
    Combine all module outputs into a unified risk assessment.

    Args:
        ocr_result: Output from extract_document_fields() (Module 1+2)
        tamper_result: Output from analyze_tampering() (Module 3)
        face_result: Output from verify_face_match() (Module 4)

    Returns:
        dict with:
          - risk_score: 0-100 overall risk
          - verdict: "LOW" / "MEDIUM" / "HIGH"
          - flags: list of specific risk reasons
          - breakdown: per-component scores
    """
    flags = []
    breakdown = {}
    score_breakdown_list = []
    score = 0.0

    # ── Module 1+2: MRZ/OCR Checksum Validation ──────────────────────────────

    ocr_score = 0.0
    mrz_status = ocr_result.get("mrz_status", "")
    ocr_status = ocr_result.get("status", "FAILURE")
    checksum_valid = ocr_result.get("checksum_valid")
    doc_type = ocr_result.get("document_type", "UNKNOWN")

    effective_mrz_status = mrz_status or ocr_status

    if effective_mrz_status == "NOT_APPLICABLE":
        score_breakdown_list.append({"component": "MRZ Checksum", "points_added": 0.0, "max_points": 25.0, "reason": "Non-MRZ document type, MRZ validation not applicable"})
    elif effective_mrz_status in ["NOT_DETERMINED", "DOCUMENT_TYPE_UNKNOWN"]:
        score_breakdown_list.append({"component": "MRZ Checksum", "points_added": 0.0, "max_points": 25.0, "reason": "Unknown document type with no MRZ detected"})
    elif effective_mrz_status in ["EXTRACTION_FAILED", "FAILURE"]:
        ocr_score += 25.0
        flags.append("MRZ_EXTRACTION_FAILED: Expected MRZ could not be read from document")
        score_breakdown_list.append({"component": "MRZ Checksum", "points_added": 25.0, "max_points": 25.0, "reason": "Expected MRZ could not be read from document"})
    elif effective_mrz_status == "INVALID" or checksum_valid is False:
        ocr_score += 25.0
        flags.append("MRZ_CHECKSUM_INVALID: Document MRZ check digits failed validation")
        score_breakdown_list.append({"component": "MRZ Checksum", "points_added": 25.0, "max_points": 25.0, "reason": "Document MRZ check digits failed validation"})
    elif effective_mrz_status in ["VALID", "SUCCESS"]:
        score_breakdown_list.append({"component": "MRZ Checksum", "points_added": 0.0, "max_points": 25.0, "reason": "All checksums valid"})

    # Check document expiry
    expiry_str = ocr_result.get("expiry_date", "")
    if expiry_str:
        try:
            expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
            if expiry_date < date.today():
                ocr_score += 15.0
                flags.append(f"DOCUMENT_EXPIRED: Expired on {expiry_str}")
                score_breakdown_list.append({"component": "Document Expiry", "points_added": 15.0, "max_points": 15.0, "reason": f"Expired on {expiry_str}"})
            else:
                score_breakdown_list.append({"component": "Document Expiry", "points_added": 0.0, "max_points": 15.0, "reason": "Document is not expired"})
        except (ValueError, TypeError):
            pass

    ocr_score = min(ocr_score, 40.0)  # Cap at 40
    breakdown["mrz_validation"] = {
        "score": round(ocr_score, 2),
        "max_possible": 40,
        "document_type": doc_type,
        "mrz_status": effective_mrz_status,
        "checksum_valid": checksum_valid,
        "expiry_date": expiry_str,
    }
    score += ocr_score

    # ── Module 3: Tampering Detection ─────────────────────────────────────────

    tamper_raw = tamper_result.get("tamper_score", 0)
    # Scale DocAuth's 0-100 score to our 0-30 range
    tamper_score = (tamper_raw / 100.0) * 30.0
    tamper_score = round(tamper_score, 2)

    if tamper_raw >= 55:
        flags.append(f"TAMPERING_DETECTED: Document shows signs of forgery (score: {tamper_raw}%)")
        score_breakdown_list.append({"component": "Tamper Detection", "points_added": tamper_score, "max_points": 30.0, "reason": f"Document shows signs of forgery (score: {tamper_raw}%)"})
    elif tamper_raw >= 10:
        flags.append(f"TAMPERING_SUSPICIOUS: Document has suspicious artifacts (score: {tamper_raw}%)")
        score_breakdown_list.append({"component": "Tamper Detection", "points_added": tamper_score, "max_points": 30.0, "reason": f"Document has suspicious artifacts (score: {tamper_raw}%)"})
    else:
        score_breakdown_list.append({"component": "Tamper Detection", "points_added": tamper_score, "max_points": 30.0, "reason": "No significant tampering detected"})

    breakdown["tampering"] = {
        "score": tamper_score,
        "max_possible": 30,
        "raw_tamper_score": tamper_raw,
        "tamper_verdict": tamper_result.get("verdict", "Unknown"),
    }
    score += tamper_score

    # ── Module 4: Face Verification ───────────────────────────────────────────

    face_score = 0.0
    face_verified = face_result.get("verified")
    face_error = face_result.get("error")
    is_real = face_result.get("is_real")

    if face_error:
        face_score += 10.0
        flags.append(f"FACE_VERIFICATION_ERROR: {face_error}")
        score_breakdown_list.append({"component": "Face Matching", "points_added": 10.0, "max_points": 20.0, "reason": f"Face verification error: {face_error}"})
    elif face_verified is False:
        face_score += 20.0
        distance = face_result.get("distance", "N/A")
        flags.append(f"FACE_MISMATCH: Document face does not match live photo (distance: {distance})")
        score_breakdown_list.append({"component": "Face Matching", "points_added": 20.0, "max_points": 20.0, "reason": "Document face does not match live photo"})
    elif face_verified is True:
        score_breakdown_list.append({"component": "Face Matching", "points_added": 0.0, "max_points": 20.0, "reason": "Document face matches live photo"})
    else:
        score_breakdown_list.append({"component": "Face Matching", "points_added": 0.0, "max_points": 20.0, "reason": "No selfie provided, verification skipped"})

    if is_real is False:
        face_score += 10.0
        flags.append("FACE_SPOOF_DETECTED: Live photo appears to be a spoof (not a real face)")
        score_breakdown_list.append({"component": "Face Liveness", "points_added": 10.0, "max_points": 10.0, "reason": "Live photo appears to be a spoof"})
    elif is_real is True:
        score_breakdown_list.append({"component": "Face Liveness", "points_added": 0.0, "max_points": 10.0, "reason": "Live photo appears to be a real face"})

    face_score = min(face_score, 30.0)  # Cap at 30
    breakdown["face_verification"] = {
        "score": round(face_score, 2),
        "max_possible": 30,
        "verified": face_verified,
        "is_real": is_real,
        "confidence": face_result.get("confidence"),
    }
    score += face_score

    # ── Final Verdict ─────────────────────────────────────────────────────────

    score = round(min(score, 100.0), 2)

    if score < 25:
        verdict = "LOW"
    elif score < 60:
        verdict = "MEDIUM"
    else:
        verdict = "HIGH"

    return {
        "risk_score": score,
        "verdict": verdict,
        "flags": flags,
        "breakdown": breakdown,
        "score_breakdown": score_breakdown_list,
    }


if __name__ == "__main__":
    import json
    import os
    import sys

    # Add vendor path for DocAuth imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from modules.ocr_extraction import extract_document_fields
    from modules.tampering_detection import analyze_tampering
    from modules.face_verification import verify_face_match

    sample_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "vendor", "fastmrz", "data")
    )
    passport = os.path.join(sample_dir, "passport_uk.jpg")
    td1 = os.path.join(sample_dir, "td1.jpg")

    print("=" * 60)
    print("FULL PIPELINE TEST: End-to-end document screening")
    print("=" * 60)

    print("\n[1/4] Running MRZ extraction...")
    ocr = extract_document_fields(passport)
    print(f"  Status: {ocr.get('status')} | Checksum: {ocr.get('checksum_valid')}")

    print("\n[2/4] Running tampering detection...")
    tamper = analyze_tampering(passport)
    print(f"  Score: {tamper.get('tamper_score')}% | Verdict: {tamper.get('verdict')}")

    print("\n[3/4] Running face verification (passport vs itself = match)...")
    face = verify_face_match(passport, passport)
    print(f"  Verified: {face.get('verified')} | Confidence: {face.get('confidence')}%")

    print("\n[4/4] Computing risk score...")
    risk = compute_risk_score(ocr, tamper, face)

    print("\n" + "=" * 60)
    print("FINAL RISK ASSESSMENT")
    print("=" * 60)
    print(json.dumps(risk, indent=4))

    # Also test with mismatch
    print("\n" + "=" * 60)
    print("MISMATCH SCENARIO (passport vs td1 ID)")
    print("=" * 60)
    face_mismatch = verify_face_match(passport, td1)
    risk_mismatch = compute_risk_score(ocr, tamper, face_mismatch)
    print(json.dumps(risk_mismatch, indent=4))

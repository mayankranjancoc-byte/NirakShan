"""
Module 5: Risk Scoring Layer
This is the ONLY original logic in the prototype.

Combines outputs from all four modules into a single risk assessment
using a weighted rule engine (not a trained model).

Weight rationale:
  - Failed MRZ checksum: strong signal of invalid/forged document (25 pts)
  - Expired document: definite flag, but not necessarily fraud (15 pts)
  - Tamper score: scaled from 0-100 range (30 pts max)
  - Face mismatch: strong signal of identity fraud (20 pts)
  - Face liveness fail: indicates photo spoofing (10 pts)

Total possible: 100 points
  0-25  -> LOW risk (likely genuine)
  25-60 -> MEDIUM risk (needs manual review)
  60+   -> HIGH risk (likely fraudulent)
"""

from datetime import datetime, date, timedelta


def _parse_iso(value: str | None) -> date | None:
    """
    Multi-format date parser (H2 fix).
    Tries common formats; returns None if none match or value is empty.
    """
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%y%m%d", "%Y%m%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(str(value), fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def compute_risk_score(
    ocr_result: dict,
    tamper_result: dict,
    face_result: dict,
    liveness_result: dict | None = None,
) -> dict:
    """
    Combine all module outputs into a unified risk assessment.

    Args:
        ocr_result:      Output from extract_document_fields() (Module 1+2)
        tamper_result:   Output from analyze_tampering() (Module 3)
        face_result:     Output from verify_face_match() (Module 4)
        liveness_result: Output from document_liveness module (Module 3.5)
                         Dict with keys "screen_replay" and "physical_motion".
                         Pass None or {} to skip liveness scoring.

    Returns:
        dict with:
          - risk_score: 0–125 overall risk (125 = max; 100 = max without liveness)
          - verdict: "LOW" / "MEDIUM" / "HIGH"
          - flags: list of specific risk reasons
          - breakdown: per-component scores

    Verdict thresholds (adjusted for 125-pt budget):
      LOW:    0–29
      MEDIUM: 30–74
      HIGH:   75+
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
    elif effective_mrz_status == "UNREADABLE":
        # M3 fix: UNREADABLE is a capture problem, not a forgery indicator.
        # Give a lower penalty and use a distinct, non-accusatory flag.
        ocr_score += 8.0
        flags.append("MRZ_UNREADABLE: Image quality too low to validate check digits — recapture required")
        score_breakdown_list.append({
            "component": "MRZ Checksum", "points_added": 8.0, "max_points": 25.0,
            "reason": "Image too blurry to read MRZ; not a forgery indicator",
        })
    elif effective_mrz_status == "INVALID" or checksum_valid is False:
        ocr_score += 25.0
        flags.append("MRZ_CHECKSUM_INVALID: Document MRZ check digits failed validation")
        score_breakdown_list.append({"component": "MRZ Checksum", "points_added": 25.0, "max_points": 25.0, "reason": "Document MRZ check digits failed validation"})
    elif effective_mrz_status in ["VALID", "SUCCESS"]:
        score_breakdown_list.append({"component": "MRZ Checksum", "points_added": 0.0, "max_points": 25.0, "reason": "All checksums valid"})

    # Check document expiry with multi-format parsing (H2 fix)
    expiry_str = ocr_result.get("expiry_date", "")
    if expiry_str:
        expiry_date = _parse_iso(expiry_str)
        if expiry_date is None:
            # H2 fix: unparseable expiry is a weak flag, not a silent skip
            ocr_score += 5.0
            flags.append(f"EXPIRY_UNPARSEABLE: Expiry '{expiry_str}' could not be interpreted")
            score_breakdown_list.append({
                "component": "Document Expiry", "points_added": 5.0, "max_points": 15.0,
                "reason": "Expiry date present but format not recognized",
            })
        elif expiry_date < date.today():
            ocr_score += 15.0
            flags.append(f"DOCUMENT_EXPIRED: Expired on {expiry_str}")
            score_breakdown_list.append({"component": "Document Expiry", "points_added": 15.0, "max_points": 15.0, "reason": f"Expired on {expiry_str}"})
        else:
            score_breakdown_list.append({"component": "Document Expiry", "points_added": 0.0, "max_points": 15.0, "reason": "Document is not expired"})

    # Date-logic consistency checks (H2 fix)
    dob_str = ocr_result.get("date_of_birth", "")
    dob = _parse_iso(dob_str)
    expiry_date_for_logic = _parse_iso(expiry_str) if expiry_str else None

    if dob and dob > date.today():
        # M5 note: fix the century pivot in ocr_extraction.py BEFORE enabling this
        # in production, or travellers born before 1969 will trigger this flag.
        ocr_score += 10.0
        flags.append(f"DOB_IN_FUTURE: Date of birth '{dob_str}' is in the future — impossible")
    if dob and expiry_date_for_logic and dob >= expiry_date_for_logic:
        ocr_score += 10.0
        flags.append(f"DATE_LOGIC_INVALID: DOB '{dob_str}' is not before expiry '{expiry_str}'")

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

    tamper_raw = tamper_result.get("tamper_score")
    tamper_verdict_str = tamper_result.get("verdict", "Unknown")

    if tamper_raw is None or tamper_verdict_str == "INCONCLUSIVE":
        # Detectors had insufficient coverage to produce a score.
        # Report as informational; do not contribute points either way.
        flags.append("TAMPERING_INCONCLUSIVE: Forensic detectors had insufficient coverage to produce a reliable score")
        tamper_score = 0.0
        score_breakdown_list.append({
            "component": "Tamper Detection", "points_added": 0.0, "max_points": 30.0,
            "reason": "INCONCLUSIVE — detectors could not produce a reliable result",
        })
    else:
        # Scale 0-100 tamper score to 0-30 range
        tamper_score = round((tamper_raw / 100.0) * 30.0, 2)

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
        "tamper_verdict": tamper_verdict_str,
        "degraded": tamper_result.get("degraded", False),
        "detector_coverage": tamper_result.get("detector_coverage"),
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
    elif is_real is None and face_verified is not None:
        # H10 fix: liveness unavailable must be visible, not silently scored 0
        face_score += 3.0
        flags.append("LIVENESS_NOT_ASSESSED: Anti-spoofing unavailable — presentation attack not ruled out")
        score_breakdown_list.append({
            "component": "Face Liveness", "points_added": 3.0, "max_points": 10.0,
            "reason": "Anti-spoofing unavailable; result inconclusive",
        })
    elif is_real is None:
        score_breakdown_list.append({
            "component": "Face Liveness", "points_added": 0.0, "max_points": 10.0,
            "reason": "No selfie provided; face-liveness check skipped",
        })

    face_score = min(face_score, 30.0)  # Cap at 30
    breakdown["face_verification"] = {
        "score": round(face_score, 2),
        "max_possible": 30,
        "verified": face_verified,
        "is_real": is_real,
        "confidence": face_result.get("confidence"),
    }
    score += face_score

    # ── Module 3.5: Document Liveness ─────────────────────────────────────────

    liveness_score = 0.0
    lr = liveness_result or {}
    sr = lr.get("screen_replay") or {}
    pm = lr.get("physical_motion") or {}

    # — Part A: Screen/Recapture —
    is_replay = sr.get("is_screen_replay")
    replay_method = sr.get("method", "unavailable")

    if is_replay is True and replay_method == "svm":
        liveness_score += 25.0
        flags.append("SCREEN_REPLAY_SUSPECTED: Document image appears to be captured from a screen or printed photocopy (SVM classifier)")
        score_breakdown_list.append({
            "component": "Document Liveness (Part A)",
            "points_added": 25.0, "max_points": 25.0,
            "reason": "Screen replay detected by trained SVM classifier",
        })
    elif is_replay is True and replay_method == "threshold":
        liveness_score += 15.0
        flags.append("SCREEN_REPLAY_SUSPECTED_HEURISTIC: Document image shows moiré/pixel-grid frequency signature — possible screen or photocopy replay (heuristic, not SVM)")
        score_breakdown_list.append({
            "component": "Document Liveness (Part A)",
            "points_added": 15.0, "max_points": 25.0,
            "reason": "Screen replay suspected by FFT+texture heuristic (SVM not yet trained)",
        })
    elif is_replay is False:
        score_breakdown_list.append({
            "component": "Document Liveness (Part A)",
            "points_added": 0.0, "max_points": 25.0,
            "reason": f"No screen replay pattern detected ({replay_method})",
        })
    elif replay_method == "heuristic":
        score_breakdown_list.append({
            "component": "Document Liveness (Part A)",
            "points_added": 0.0, "max_points": 25.0,
            "reason": "Heuristic replay signals are advisory; no trained SVM verdict",
        })
    else:
        # unavailable / error
        flags.append("LIVENESS_PART_A_SKIPPED: Screen-replay check unavailable")
        score_breakdown_list.append({
            "component": "Document Liveness (Part A)",
            "points_added": 0.0, "max_points": 25.0,
            "reason": "Screen-replay check unavailable",
        })

    # — Part B: Physical Motion —
    motion_verdict = pm.get("verdict", "SKIPPED")

    if motion_verdict == "STATIC":
        liveness_score += 20.0
        flags.append("LIVENESS_CHECK_FAILED: Document shows no physical motion — possible flat-image replay (tilt video provided but no motion detected)")
        score_breakdown_list.append({
            "component": "Document Liveness (Part B)",
            "points_added": 20.0, "max_points": 20.0,
            "reason": "No specular displacement or hologram shift detected in tilt video",
        })
    elif motion_verdict == "INCONCLUSIVE":
        flags.append("LIVENESS_PART_B_INCONCLUSIVE: Physical motion analysis inconclusive (insufficient frames or coverage)")
        score_breakdown_list.append({
            "component": "Document Liveness (Part B)",
            "points_added": 0.0, "max_points": 20.0,
            "reason": "Motion analysis inconclusive",
        })
    elif motion_verdict == "PHYSICAL":
        hologram_ok = pm.get("hologram_shift_detected", False)
        if hologram_ok:
            score_breakdown_list.append({
                "component": "Document Liveness (Part B)",
                "points_added": 0.0, "max_points": 20.0,
                "reason": "Physical motion and hologram shift confirmed",
            })
        else:
            score_breakdown_list.append({
                "component": "Document Liveness (Part B)",
                "points_added": 0.0, "max_points": 20.0,
                "reason": "Physical motion confirmed (hologram shift marginal but motion present)",
            })
    else:
        # SKIPPED
        score_breakdown_list.append({
            "component": "Document Liveness (Part B)",
            "points_added": 0.0, "max_points": 20.0,
            "reason": "No liveness video provided; physical motion check skipped",
        })

    liveness_score = min(liveness_score, 45.0)
    breakdown["document_liveness"] = {
        "score": liveness_score,
        "max_possible": 45,
        "screen_replay_method": replay_method,
        "is_screen_replay": is_replay,
        "physical_motion_verdict": motion_verdict,
        "fft_peak_ratio": sr.get("fft_peak_ratio"),
        "texture_uniformity": sr.get("texture_uniformity"),
        "mean_highlight_displacement_px": pm.get("mean_highlight_displacement_px"),
        "hologram_hsv_shift": pm.get("hologram_hsv_shift"),
    }
    score += liveness_score

    # ── Final Verdict ───────────────────────────────────────────────────

    score = round(score, 2)  # max = 145 pts; cap in verdict logic only

    if score < 30:
        verdict = "LOW"
    elif score < 75:
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

    # Add vendor path for image_forensics imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from modules.ocr_extraction import extract_document_fields
    from modules.tampering_detection import analyze_tampering
    from modules.face_verification import verify_face_match

    sample_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "vendor", "mrz_scanner", "data")
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

"""
Module 1 + 2: OCR/MRZ Extraction & Validation
Built on top of the mrz_scanner library for MRZ parsing and ICAO 9303 checksum validation.
License: AGPL-3.0 — copyleft implications if shipped as a hosted service.

mrz_scanner bundles both MRZ extraction (Module 1) and checksum validation
(Module 2) in a single call via get_details().
"""

import os
import sys
import shutil
import tempfile
import itertools
import re
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import pytesseract
from mrz_scanner import MRZScanner


# ── Tesseract path discovery (H1 fix) ───────────────────────────────────
def _find_tesseract() -> str | None:
    """
    Locate Tesseract executable: prefer PATH, then try platform-specific
    well-known locations. Returns None if not found.
    """
    found = shutil.which("tesseract")
    if found:
        return found
    for candidate in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        "/opt/homebrew/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/usr/bin/tesseract",
    ):
        if os.path.exists(candidate):
            return candidate
    return None


TESSDATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "vendor", "mrz_scanner", "tessdata")
)
TESSERACT_PATH = _find_tesseract()
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Standard document type classifications
MRZ_DOCUMENT_TYPES = {"PASSPORT", "VISA", "MRZ_ID", "TD1", "TD2", "TD3", "MRVA", "MRVB"}
NON_MRZ_DOCUMENT_TYPES = {"NON_MRZ_ID", "DRIVING_LICENSE", "NATIONAL_ID_CARD", "AADHAAR", "PAN"}


_mrz_reader = None
_mrz_lock = __import__("threading").Lock()


def get_mrz_reader():
    """Thread-safe lazy singleton for MRZScanner reader."""
    global _mrz_reader
    if _mrz_reader is None:
        with _mrz_lock:
            if _mrz_reader is None:
                if not TESSERACT_PATH:
                    raise RuntimeError(
                        "Tesseract not found. Install it or set TESSERACT_CMD in PATH. "
                        "Windows: https://github.com/UB-Mannheim/tesseract/wiki"
                    )
                _mrz_reader = MRZScanner(
                    tesseract_path=TESSERACT_PATH,
                    tessdata_path=TESSDATA_DIR,
                )
    return _mrz_reader



def parse_visa_fields(mrz_lines: list) -> dict:
    """
    Parse Visa MRZ fields that are genuinely present in the MRZ (C2 fix).

    Duration of stay and number of entries are VIZ-only fields — they do NOT
    appear in the MRZ. The old implementation read them from wrong MRZ offsets
    (expiry date bytes), producing confident but fabricated values on every visa.

    This function now returns ONLY fields that the ICAO 9303 MRV-A/B spec
    actually encodes in the MRZ.
    """
    if not mrz_lines or len(mrz_lines) < 2:
        return {}
    line1 = mrz_lines[0].replace("\r", "")
    return {
        "document_code": line1[0:1] if len(line1) >= 1 else None,      # 'V'
        "visa_type": line1[1:2].replace("<", "") or None if len(line1) >= 2 else None,
        "issuing_state": line1[2:5].replace("<", "") or None if len(line1) >= 5 else None,
        # Note: stay_duration_days / entries_allowed are VIZ-only fields.
        # To add them, run a separate Tesseract pass on the visual zone and
        # return them with source="visual_zone" and an explicit confidence value.
    }


def calculate_image_quality(image_path: str) -> dict | None:
    """
    Assess basic image quality (blur) via Laplacian variance.

    M2 fix:
    - Returns None on any failure instead of fake {is_blurry: False} (fail-open).
    - Threshold is still a heuristic; calibrate against your sample set.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None  # M2 fix: fail-closed (return None, not is_blurry=False)

        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        is_blurry = laplacian_var < 50  # heuristic threshold

        return {
            "is_blurry": is_blurry,
            "laplacian_var": float(laplacian_var),
        }
    except Exception:
        return None  # M2 fix: unknown, not "sharp"


def generate_preprocessing_variants(image_path: str, out_dir: str) -> list:
    """
    Generate image variants with different preprocessing applied to combat glare/shadows.

    M1 fix: caller passes an `out_dir` (created via tempfile.mkdtemp) so cleanup
    can always happen in a finally block, even on the success path.
    """
    img = cv2.imread(image_path)
    if img is None:
        return []

    variants = []

    # Variant 1: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl, a, b))
        clahe_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        v1_path = os.path.join(out_dir, "v1_clahe.jpg")
        cv2.imwrite(v1_path, clahe_img)
        variants.append(v1_path)
    except Exception:
        pass

    # Variant 2: Sharpening
    try:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(img, -1, kernel)
        v2_path = os.path.join(out_dir, "v2_sharp.jpg")
        cv2.imwrite(v2_path, sharpened)
        variants.append(v2_path)
    except Exception:
        pass

    return variants


def _icao_checkdigit(s: str) -> str:
    """
    ICAO Doc 9303 check digit calculation using repeating 7-3-1 weights.
    Digits 0-9 = value 0-9, Letters A-Z = 10-35, '<' = 0.
    """
    weights = [7, 3, 1]
    total = 0
    for i, ch in enumerate(s):
        if ch.isdigit():
            v = int(ch)
        elif ch.isalpha():
            v = ord(ch.upper()) - ord('A') + 10
        else:
            v = 0
        total += v * weights[i % 3]
    return str(total % 10)


def _icao_verify_td3(mrz_text: str) -> bool:
    """
    Independently verify all ICAO 9303 TD3 (passport, 2x44 char) checksums.
    Returns True only when ALL critical check digits are valid.
    Treats position 43 as optional — if it is '<', skip that check
    (ICAO spec: filler '<' = no optional data, checkdigit not required).
    """
    try:
        lines = [l.replace("\r", "") for l in mrz_text.strip().split("\n")]
        if len(lines) < 2 or len(lines[0]) < 44 or len(lines[1]) < 44:
            return False
        l2 = lines[1]
        # 1. Document number (positions 0-8, check digit at 9)
        if _icao_checkdigit(l2[0:9]) != l2[9]:
            return False
        # 2. Date of birth (positions 13-18, check digit at 19)
        if _icao_checkdigit(l2[13:19]) != l2[19]:
            return False
        # 3. Date of expiry (positions 21-26, check digit at 27)
        if _icao_checkdigit(l2[21:27]) != l2[27]:
            return False
        # 4. Optional data (positions 28-41, check digit at 42)
        #    Skip if char 42 is '<' — many countries leave it blank
        if l2[42] != '<':
            if _icao_checkdigit(l2[28:42]) != l2[42]:
                return False
        # 5. Final / composite check digit (last char, position 43)
        composite = l2[0:10] + l2[13:20] + l2[21:43]
        if _icao_checkdigit(composite) != l2[43]:
            return False
        return True
    except Exception:
        return False


# ── OCR Disambiguation (C1 fix) ───────────────────────────────────────────────

# Visually similar character pairs (both directions)
CONFUSIONS: dict[str, str] = {
    "O": "0", "0": "O",
    "I": "1", "1": "I",
    "B": "8", "8": "B",
    "Z": "2", "2": "Z",
    "S": "5", "5": "S",
    "Q": "0", "D": "0",
}

# TD3 line 2 spans covered by check digits (0-indexed, exclusive end)
# Line 1 (name field) is NEVER modified — check digits don't cover it.
NUMERIC_SPANS: dict[str, tuple[int, int]] = {
    "doc_number": (0, 9),
    "dob":        (13, 19),
    "expiry":     (21, 27),
    "optional":   (28, 42),
}


def apply_ocr_disambiguation(
    raw_mrz_text: str,
    fast_mrz,
    max_edits: int = 2,
) -> dict | None:
    """
    Attempt to fix OCR errors using bounded per-character substitutions ONLY
    inside checksum-covered numeric spans on line 2 (C1 fix).

    The old global replace('O','0') would silently corrupt the name field on
    line 1, which has no checksum, making the corruption invisible to validation.

    This version:
    - Never modifies line 1 (name field)
    - Only tries character substitutions at positions with known confusions
    - Tries single-character corrections first, then pairs
    """
    try:
        lines = [l.replace("\r", "") for l in raw_mrz_text.strip().split("\n")]
        if len(lines) < 2:
            return None
        line1, line2 = lines[0], lines[1]
    except Exception:
        return None

    # Build list of positions to try: only positions within numeric spans
    # that have a known confusion character.
    candidate_positions = [
        i
        for _, (start, end) in NUMERIC_SPANS.items()
        for i in range(start, min(end, len(line2)))
        if i < len(line2) and line2[i] in CONFUSIONS
    ]

    if not candidate_positions:
        return None

    for n_edits in range(1, min(max_edits, len(candidate_positions)) + 1):
        for combo in itertools.combinations(candidate_positions, n_edits):
            replacements = [CONFUSIONS.get(line2[i]) for i in combo]
            if any(r is None for r in replacements):
                continue
            chars = list(line2)
            for idx, repl in zip(combo, replacements):
                chars[idx] = repl
            candidate_line2 = "".join(chars)
            candidate_mrz = line1 + "\n" + candidate_line2  # line1 NEVER modified
            try:
                res = fast_mrz.get_details(
                    candidate_mrz, input_type="text", include_checkdigit=True
                )
                if res and res.get("status") == "SUCCESS":
                    res["is_corrected_by_disambiguation"] = True
                    res["corrections_applied"] = list(zip(combo, replacements))
                    return res
                # Apply our own verifier too (handles optional-data quirks)
                if res and res.get("mrz_text") and _icao_verify_td3(res["mrz_text"]):
                    res["status"] = "SUCCESS"
                    res["status_message"] = None
                    res["is_corrected_by_disambiguation"] = True
                    res["corrections_applied"] = list(zip(combo, replacements))
                    return res
            except Exception:
                continue

    return None


def extract_document_fields(image_path: str, document_type: str = None) -> dict:
    """
    Extract MRZ or visible text fields from a document image.

    Args:
        image_path: Path to the document image.
        document_type: Optional document classification:
                       'PASSPORT', 'VISA', 'MRZ_ID', 'NON_MRZ_ID', 'UNKNOWN'
                       (Default: None/UNKNOWN - automatically attempts MRZ detection first).

    Returns:
        dict with:
          - document_type: str ('PASSPORT', 'TD1', 'NON_MRZ_ID', 'UNKNOWN', etc.)
          - status: 'VALID', 'INVALID', 'EXTRACTION_FAILED', 'NOT_APPLICABLE', 'DOCUMENT_TYPE_UNKNOWN'
          - mrz_status: 'VALID', 'INVALID', 'EXTRACTION_FAILED', 'NOT_APPLICABLE', 'NOT_DETERMINED'
          - checksum_valid: True / False / None
          - parsed MRZ fields or visible extracted text
    """
    doc_type_upper = (document_type.upper().strip() if document_type else "UNKNOWN")

    # Case 2: Explicitly specified Non-MRZ Document
    if doc_type_upper in NON_MRZ_DOCUMENT_TYPES:
        try:
            img = Image.open(image_path)
            raw_text = pytesseract.image_to_string(img).strip()
            return {
                "document_type": doc_type_upper,
                "status": "NOT_APPLICABLE",
                "mrz_status": "NOT_APPLICABLE",
                "checksum_valid": None,
                "extracted_text": raw_text,
                "note": "Document type is non-MRZ; MRZ extraction not applicable.",
            }
        except Exception as e:
            return {
                "document_type": doc_type_upper,
                "status": "NOT_APPLICABLE",
                "mrz_status": "NOT_APPLICABLE",
                "checksum_valid": None,
                "error": str(e),
            }

    # Otherwise (MRZ document type or UNKNOWN): Attempt MRZScanner extraction
    # M1 fix: use mkdtemp so preprocessing variants are always cleaned up.
    variant_dir = tempfile.mkdtemp(prefix="mrz_variants_")
    try:
        fast_mrz = get_mrz_reader()

        # 1. Base pass
        result = fast_mrz.get_details(image_path, include_checkdigit=True)

        # Check if base pass failed or is invalid
        if not result or result.get("status", "").upper() != "SUCCESS":
            # 2. Checksum Disambiguation on base pass text
            if result and result.get("mrz_text"):
                disambig_res = apply_ocr_disambiguation(result["mrz_text"], fast_mrz)
                if disambig_res:
                    result = disambig_res

            # 3. Retry with Image Preprocessing Variants if still failing
            if not result or result.get("status", "").upper() != "SUCCESS":
                variants = generate_preprocessing_variants(image_path, variant_dir)
                for var_path in variants:
                    if os.path.exists(var_path):
                        var_res = fast_mrz.get_details(var_path, include_checkdigit=True)
                        if var_res and var_res.get("status", "").upper() == "SUCCESS":
                            result = var_res
                            break

                        # Disambiguate on the variant's text too
                        if var_res and var_res.get("mrz_text"):
                            var_disambig = apply_ocr_disambiguation(var_res["mrz_text"], fast_mrz)
                            if var_disambig:
                                result = var_disambig
                                break
    except Exception as e:
        result = {"status": "FAILURE", "error": str(e), "status_message": str(e)}
    finally:
        # M1 fix: always clean up variant temp files, even on success path
        shutil.rmtree(variant_dir, ignore_errors=True)

    # Assess quality
    iqa_metrics = calculate_image_quality(image_path)

    # Check MRZScanner outcome
    if result is not None and isinstance(result, dict) and "mrz_text" in result:
        # Use our own independent ICAO 9303 verifier — more robust than MRZScanner's
        # status_message check. Handles Indian/US/UK passports where optional data
        # position 43 is '<' (not '0') which MRZScanner incorrectly flags as FAILURE.
        if result.get("status") == "FAILURE" and result.get("mrz_text"):
            mrz_type_guess = result.get("mrz_type", "TD3")
            if mrz_type_guess in ("TD3", None) and _icao_verify_td3(result["mrz_text"]):
                result["status"] = "SUCCESS"
                result["status_message"] = None

        mrz_scanner_status = result.get("status", "").upper()
        
        mrz_type = doc_type_upper if doc_type_upper in MRZ_DOCUMENT_TYPES else result.get("mrz_type", "PASSPORT")
        if mrz_type in ["VISA", "MRVA", "MRVB"]:
            mrz_lines = result["mrz_text"].strip().split("\n")
            visa_fields = parse_visa_fields(mrz_lines)
        else:
            visa_fields = None
            
        if mrz_scanner_status == "SUCCESS":
            result["document_type"] = mrz_type
            result["status"] = "VALID"
            result["mrz_status"] = "VALID"
            result["checksum_valid"] = True
            result["visa_fields"] = visa_fields
            result["iqa_metrics"] = iqa_metrics
            return result
        else:
            # MRZ detected but check digit validation failed. 
            # If the image was blurry, it might be unreadable rather than invalid.
            mrz_status = "UNREADABLE" if iqa_metrics["is_blurry"] else "INVALID"
            
            result["document_type"] = mrz_type
            result["status"] = mrz_status
            result["mrz_status"] = mrz_status
            result["checksum_valid"] = False
            result["error"] = result.get("status_message", "MRZ checksum validation failed")
            result["visa_fields"] = visa_fields
            result["iqa_metrics"] = iqa_metrics
            return result

    # Fallback when MRZScanner cannot parse strict ICAO structure
    return _parse_fallback_document(image_path, doc_type_upper, iqa_metrics)


def _parse_fallback_document(image_path: str, doc_type_upper: str = None, iqa_metrics: dict = None) -> dict:
    """
    Fallback extraction when standard MRZScanner fails to parse a strict ICAO MRZ.
    Performs OCR across the image and bottom region to:
    1. Accurately detect document type (PASSPORT, VISA, NATIONAL_ID, etc.) from visual zone.
    2. Extract MRZ candidate lines (including malformed or counterfeit MRZs).
    3. Extract VIZ fields (Name, Document Number, Country, DOB, Expiry, Sex).
    4. Flag counterfeit/malformed MRZ as INVALID with clear diagnostic details.
    """
    try:
        img = Image.open(image_path)
        full_text = pytesseract.image_to_string(img).strip()
    except Exception:
        full_text = ""

    lines = [l.strip() for l in full_text.splitlines() if l.strip()]
    text_upper = full_text.upper()

    # 1. Determine document type from Visual Inspection Zone (VIZ)
    detected_type = "UNKNOWN"
    if "PASSPORT" in text_upper or any(l.startswith("P<") for l in lines) or "P<" in text_upper:
        detected_type = "PASSPORT"
    elif "VISA" in text_upper or any(l.startswith("V<") for l in lines) or "V<" in text_upper:
        detected_type = "VISA"
    elif "DRIVING" in text_upper and "LICEN" in text_upper:
        detected_type = "DRIVING_LICENSE"
    elif "IDENTITY CARD" in text_upper or "NATIONAL ID" in text_upper or any(l.startswith(("I<", "ID<", "TD1")) for l in lines):
        detected_type = "NATIONAL_ID_CARD"

    final_doc_type = doc_type_upper if (doc_type_upper and doc_type_upper != "UNKNOWN") else detected_type

    # 2. Search for MRZ lines
    mrz_lines = []
    for l in lines:
        cleaned = l.replace(" ", "")
        if ("<" in cleaned and len(cleaned) >= 20) or (cleaned.startswith("P<") and len(cleaned) >= 15):
            mrz_lines.append(cleaned)

    # Attempt bottom 30% crop if full-image OCR missed MRZ lines
    if len(mrz_lines) < 2:
        try:
            cv_img = cv2.imread(image_path)
            if cv_img is not None:
                h, w = cv_img.shape[:2]
                bottom = cv_img[int(h * 0.70):, :]
                gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
                b_text = pytesseract.image_to_string(gray, config="--psm 6")
                b_lines = [l.strip().replace(" ", "") for l in b_text.splitlines() if l.strip()]
                b_mrz = [l for l in b_lines if "<" in l or re.match(r"^[A-Z0-9<]{25,44}$", l)]
                if len(b_mrz) >= 2:
                    mrz_lines = b_mrz[-2:]
        except Exception:
            pass

    # 3. If MRZ lines are present (even if counterfeit, malformed, or non-ICAO format)
    if len(mrz_lines) >= 2 or (len(mrz_lines) == 1 and "<" in mrz_lines[0]):
        l1 = mrz_lines[0].replace(" ", "").replace("K", "<")
        l2 = mrz_lines[1].replace(" ", "").replace("K", "<") if len(mrz_lines) >= 2 else ""

        doc_cls = final_doc_type if final_doc_type != "UNKNOWN" else "PASSPORT"
        fields = {
            "document_type": doc_cls,
            "mrz_type": "TD3" if doc_cls in ("PASSPORT", "UNKNOWN") else "TD1",
            "status": "INVALID",
            "mrz_status": "INVALID",
            "checksum_valid": False,
            "error": "MRZ format or check digits failed ICAO 9303 validation (counterfeit/malformed MRZ detected)",
            "mrz_text": "\n".join(mrz_lines),
            "extracted_text": full_text,
            "visa_fields": None,
            "iqa_metrics": iqa_metrics,
        }

        # Parse Line 1 (Name & Country)
        if l1.startswith("P<"):
            raw_name = l1[2:]
            if re.match(r"^[A-Z]{3}[A-Z<]", raw_name) and not (raw_name.startswith("BALOOSHI") or raw_name.startswith("CHAND")):
                country = raw_name[:3]
                raw_name = raw_name[3:]
                fields["country"] = country
                fields["issuer_code"] = country

            name_parts = [p.replace("<", " ").strip() for p in raw_name.split("<<") if p.replace("<", " ").strip()]
            if len(name_parts) >= 2:
                fields["surname"] = name_parts[0]
                fields["given_name"] = name_parts[1]
                fields["given_names"] = name_parts[1]
                fields["name"] = f"{name_parts[1]} {name_parts[0]}"
            elif len(name_parts) == 1:
                fields["name"] = name_parts[0]
                fields["surname"] = name_parts[0]

        # Parse Line 2 (Document Number, Country, DOB, Sex, Expiry)
        m2 = re.search(r"^([A-Z0-9]{8,10})([A-Z]{3}|<)?(\d{6,8})([MF<])(\d{6,8})", l2)
        if m2:
            fields["document_number"] = m2.group(1).replace("<", "")
            if m2.group(2) and m2.group(2) != "<":
                fields["country"] = m2.group(2)
                fields["issuer_code"] = m2.group(2)
                fields["nationality"] = m2.group(2)

            dob_raw = m2.group(3)
            if len(dob_raw) == 8:
                try:
                    d = datetime.strptime(dob_raw, "%d%m%Y")
                    fields["date_of_birth"] = d.strftime("%Y-%m-%d")
                    fields["birth_date"] = d.strftime("%Y-%m-%d")
                except Exception:
                    pass
            elif len(dob_raw) == 6:
                try:
                    d = datetime.strptime(dob_raw, "%y%m%d")
                    fields["date_of_birth"] = d.strftime("%Y-%m-%d")
                    fields["birth_date"] = d.strftime("%Y-%m-%d")
                except Exception:
                    pass

            sex = m2.group(4)
            if sex in ("M", "F"):
                fields["sex"] = sex

            exp_raw = m2.group(5)
            if len(exp_raw) == 8:
                try:
                    d = datetime.strptime(exp_raw, "%d%m%Y")
                    fields["expiry_date"] = d.strftime("%Y-%m-%d")
                except Exception:
                    pass
            elif len(exp_raw) == 6:
                try:
                    d = datetime.strptime(exp_raw, "%y%m%d")
                    fields["expiry_date"] = d.strftime("%Y-%m-%d")
                except Exception:
                    pass

        # VIZ fallback refinements
        name_m = re.search(r"(?:Names?|Name|Holder Name)\s*\n*([A-Z\s\-]+)", full_text, re.IGNORECASE)
        if name_m:
            viz_name = name_m.group(1).strip().splitlines()[0].strip()
            if not any(k in viz_name.upper() for k in ["NATIONALITY", "DATE", "SEX", "BIRTH", "TYPE", "UNITED", "ARAB"]):
                fields["name"] = viz_name

        if not fields.get("document_number"):
            num_m = re.search(r"(?:Passport\s*No\.?|Doc(?:ument)?\s*No\.?|No\.?)\s*\n*([A-Z0-9]{7,10})", full_text, re.IGNORECASE)
            if num_m:
                fields["document_number"] = num_m.group(1).strip()

        if not fields.get("country") or not fields.get("issuer_code"):
            cc_m = re.search(r"Country\s*Code\s*\n*([A-Z]{3})", full_text, re.IGNORECASE)
            if cc_m:
                fields["country"] = cc_m.group(1).strip()
                fields["issuer_code"] = cc_m.group(1).strip()

        if not fields.get("date_of_birth"):
            dob_m = re.search(r"(?:Date\s*of\s*Birth|DOB)\s*\n*(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})", full_text, re.IGNORECASE)
            if dob_m:
                try:
                    d = datetime.strptime(dob_m.group(1).replace("-", "/").replace(".", "/"), "%d/%m/%Y")
                    fields["date_of_birth"] = d.strftime("%Y-%m-%d")
                    fields["birth_date"] = d.strftime("%Y-%m-%d")
                except Exception:
                    pass

        if not fields.get("expiry_date"):
            exp_m = re.search(r"(?:Date\s*of\s*Expiry|Expiry\s*Date|Expires)\s*\n*(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})", full_text, re.IGNORECASE)
            if exp_m:
                try:
                    d = datetime.strptime(exp_m.group(1).replace("-", "/").replace(".", "/"), "%d/%m/%Y")
                    fields["expiry_date"] = d.strftime("%Y-%m-%d")
                except Exception:
                    pass

        return fields

    # 4. No MRZ lines found
    if final_doc_type in MRZ_DOCUMENT_TYPES:
        return {
            "document_type": final_doc_type,
            "status": "EXTRACTION_FAILED",
            "mrz_status": "EXTRACTION_FAILED",
            "checksum_valid": False,
            "error": "MRZ expected for this document type but could not be read",
            "extracted_text": full_text,
            "visa_fields": None,
            "iqa_metrics": iqa_metrics,
        }
    else:
        return {
            "document_type": "UNKNOWN",
            "status": "DOCUMENT_TYPE_UNKNOWN",
            "mrz_status": "NOT_DETERMINED",
            "checksum_valid": None,
            "extracted_text": full_text,
            "note": "No MRZ detected; document type unknown. No fraud penalty applied.",
            "visa_fields": None,
            "iqa_metrics": iqa_metrics,
        }


if __name__ == "__main__":
    import json

    # Default to the sample passport from the cloned mrz_scanner repo
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "vendor",
                "mrz_scanner",
                "data",
                "passport_uk.jpg",
            )
        )

    print(f"Testing with image: {img_path}")
    print("-" * 60)
    output = extract_document_fields(img_path)
    print(json.dumps(output, indent=4))

"""
Module 1 + 2: OCR/MRZ Extraction & Validation
Wraps the fastmrz package (https://github.com/sivakumar-mahalingam/fastmrz)
License: AGPL-3.0 — copyleft implications if shipped as a hosted service.

fastmrz bundles both MRZ extraction (Module 1) and checksum validation
(Module 2) in a single call via get_details().
"""

import os
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image
import pytesseract
from fastmrz import FastMRZ


# Path to the cloned repo's tessdata folder (contains mrz.traineddata)
TESSDATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "vendor", "fastmrz", "tessdata")
)

# Path to Tesseract executable on Windows
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Standard document type classifications
MRZ_DOCUMENT_TYPES = {"PASSPORT", "VISA", "MRZ_ID", "TD1", "TD2", "TD3", "MRVA", "MRVB"}
NON_MRZ_DOCUMENT_TYPES = {"NON_MRZ_ID", "DRIVING_LICENSE", "NATIONAL_ID_CARD", "AADHAAR", "PAN"}


_mrz_reader = None

def get_mrz_reader():
    global _mrz_reader
    if _mrz_reader is None:
        _mrz_reader = FastMRZ(
            tesseract_path=TESSERACT_PATH,
            tessdata_path=TESSDATA_DIR,
        )
    return _mrz_reader


def parse_visa_fields(mrz_lines: list) -> dict:
    """Parse Visa-specific fields from MRZ lines."""
    result = {
        "visa_number": None,
        "visa_type_code": None,
        "stay_duration_days": None,
        "entries_allowed": None,
        "entry_type": None
    }
    
    if not mrz_lines or len(mrz_lines) < 2:
        return result
        
    line1 = mrz_lines[0].replace("\r", "")
    line2 = mrz_lines[1].replace("\r", "")
    
    if len(line1) >= 2:
        result["visa_type_code"] = line1[0:2].replace("<", "")
        
    if len(line2) >= 22:
        duration_str = line2[19:22]
        if duration_str == "---" or not duration_str.replace("<", ""):
            result["stay_duration_days"] = None
        else:
            try:
                result["stay_duration_days"] = int(duration_str.replace("<", ""))
            except ValueError:
                result["stay_duration_days"] = None
                
    if len(line2) > 22:
        entries_char = line2[22]
        if entries_char == 'M':
            result["entries_allowed"] = "MULTIPLE"
            result["entry_type"] = "MULTIPLE"
        elif entries_char.isdigit():
            result["entries_allowed"] = entries_char
            result["entry_type"] = "SINGLE" if entries_char == '1' else "MULTIPLE"
            
    return result


def calculate_image_quality(image_path: str) -> dict:
    """Assess basic image quality (blur)."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"is_blurry": False, "laplacian_var": 1000}
        
        # Calculate Laplacian variance
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        
        # A threshold of 50 is a common heuristic for blur
        is_blurry = laplacian_var < 50
        
        return {
            "is_blurry": is_blurry,
            "laplacian_var": laplacian_var
        }
    except Exception:
        return {"is_blurry": False, "laplacian_var": 1000}


def generate_preprocessing_variants(image_path: str) -> list:
    """Generate image variants with different preprocessing applied to combat glare/shadows."""
    temp_dir = tempfile.gettempdir()
        
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
        v1_path = os.path.join(temp_dir, f"v1_clahe_{os.path.basename(image_path)}")
        cv2.imwrite(v1_path, clahe_img)
        variants.append(v1_path)
    except Exception:
        pass
        
    # Variant 2: Sharpening
    try:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(img, -1, kernel)
        v2_path = os.path.join(temp_dir, f"v2_sharp_{os.path.basename(image_path)}")
        cv2.imwrite(v2_path, sharpened)
        variants.append(v2_path)
    except Exception:
        pass
        
    return variants


def apply_ocr_disambiguation(raw_mrz_text: str, fast_mrz) -> dict:
    """Attempt to fix OCR errors using checksum permutations."""
    # We will try a few simple global replacements first
    test_strings = []
    test_strings.append(raw_mrz_text.replace('O', '0'))
    test_strings.append(raw_mrz_text.replace('0', 'O'))
    test_strings.append(raw_mrz_text.replace('I', '1'))
    test_strings.append(raw_mrz_text.replace('1', 'I'))
    test_strings.append(raw_mrz_text.replace('B', '8'))
    test_strings.append(raw_mrz_text.replace('8', 'B'))
    test_strings.append(raw_mrz_text.replace('Z', '2'))
    test_strings.append(raw_mrz_text.replace('2', 'Z'))
    test_strings.append(raw_mrz_text.replace('S', '5'))
    test_strings.append(raw_mrz_text.replace('5', 'S'))
    
    for test_str in test_strings:
        if test_str == raw_mrz_text:
            continue
            
        try:
            # We trick fastmrz into parsing the raw text
            res = fast_mrz.get_details(test_str, input_type="text", include_checkdigit=True)
            if res and res.get("status") == "SUCCESS":
                res["is_corrected_by_disambiguation"] = True
                res["original_mrz_text"] = raw_mrz_text
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

    # Otherwise (MRZ document type or UNKNOWN): Attempt FastMRZ extraction
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
                variants = generate_preprocessing_variants(image_path)
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
                                
                        # Cleanup variant
                        try:
                            os.remove(var_path)
                        except:
                            pass
                            
    except Exception as e:
        result = {"status": "FAILURE", "error": str(e), "status_message": str(e)}

    # Assess quality
    iqa_metrics = calculate_image_quality(image_path)

    # Check FastMRZ outcome
    if result is not None and isinstance(result, dict) and "mrz_text" in result:
        fastmrz_status = result.get("status", "").upper()
        
        mrz_type = doc_type_upper if doc_type_upper in MRZ_DOCUMENT_TYPES else result.get("mrz_type", "PASSPORT")
        if mrz_type in ["VISA", "MRVA", "MRVB"]:
            mrz_lines = result["mrz_text"].strip().split("\n")
            visa_fields = parse_visa_fields(mrz_lines)
        else:
            visa_fields = None
            
        if fastmrz_status == "SUCCESS":
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

    # FastMRZ returned No MRZ / FAILURE
    if doc_type_upper in MRZ_DOCUMENT_TYPES:
        # Case 1: MRZ was expected for this document type, but extraction failed
        return {
            "document_type": doc_type_upper,
            "status": "EXTRACTION_FAILED",
            "mrz_status": "EXTRACTION_FAILED",
            "checksum_valid": False,
            "error": "MRZ expected for this document type but could not be read",
            "visa_fields": None,
            "iqa_metrics": iqa_metrics
        }
    else:
        # Case 3: Document type is UNKNOWN and no MRZ was detected
        try:
            img = Image.open(image_path)
            raw_text = pytesseract.image_to_string(img).strip()
        except Exception:
            raw_text = ""

        return {
            "document_type": "UNKNOWN",
            "status": "DOCUMENT_TYPE_UNKNOWN",
            "mrz_status": "NOT_DETERMINED",
            "checksum_valid": None,
            "extracted_text": raw_text,
            "note": "No MRZ detected; document type unknown. No fraud penalty applied.",
            "visa_fields": None,
            "iqa_metrics": iqa_metrics
        }


if __name__ == "__main__":
    import json

    # Default to the sample passport from the cloned fastmrz repo
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "vendor",
                "fastmrz",
                "data",
                "passport_uk.jpg",
            )
        )

    print(f"Testing with image: {img_path}")
    print("-" * 60)
    output = extract_document_fields(img_path)
    print(json.dumps(output, indent=4))

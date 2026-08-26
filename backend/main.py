"""
FastAPI Backend for Document Screening Prototype

Endpoints:
  POST /screen-document  -- Upload document image + optional selfie, get risk assessment
  GET  /health           -- Liveness check
"""

import os
import sys
import uuid
import shutil
import logging

import numpy as np

# Set environment variables before importing tensorflow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONIOENCODING"] = "utf-8"

# Add vendor directory to path for DocAuth imports
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
VENDOR_DIR = os.path.join(BACKEND_DIR, "vendor", "docauth")
if VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from modules.ocr_extraction import extract_document_fields, get_mrz_reader
from modules.tampering_detection import analyze_tampering
from modules.face_verification import verify_face_match
from modules.risk_scoring import compute_risk_score
from modules.audit_logger import log_screening_result, get_recent_screenings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
UPLOAD_DIR = os.path.join(BACKEND_DIR, "uploads")
FRONTEND_DIR = os.path.abspath(os.path.join(BACKEND_DIR, "..", "frontend"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="Document Screening Prototype",
    description=(
        "AI-powered border document screening system. "
        "Combines MRZ extraction (fastmrz), tampering detection (DocAuth), "
        "and face verification (deepface) into a unified risk assessment."
    ),
    version="0.1.0",
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _save_upload(upload: UploadFile) -> str:
    """Save an uploaded file to disk and return the path."""
    ext = os.path.splitext(upload.filename or "file.jpg")[1] or ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    with open(filepath, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return filepath


def _json_safe(value):
    """Convert NumPy values, bytes, and custom objects into JSON-compatible values."""
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="ignore")
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


@app.on_event("startup")
async def preload_models():
    # 1. Warm up the FastMRZ singleton
    get_mrz_reader()
    # 2. Force DeepFace to download ArcFace + anti-spoof weights
    # by running a dummy verify on a blank white image pair in a background thread
    import threading
    def _bg_preload():
        try:
            import numpy as np
            from deepface import DeepFace
            blank = np.ones((112, 112, 3), dtype=np.uint8) * 255
            DeepFace.verify(
                img1_path=blank,
                img2_path=blank,
                model_name="ArcFace",
                detector_backend="retinaface",
                anti_spoofing=True,
                enforce_detection=False,
                silent=True
            )
        except Exception:
            pass  # weights download happens, dummy verify may fail — that's fine
            
    threading.Thread(target=_bg_preload, daemon=True).start()


@app.get("/health")
async def health_check():
    """Basic liveness check."""
    return {"status": "ok", "service": "document-screening-prototype", "models_loaded": ["FastMRZ", "ArcFace", "RetinaFace"]}


@app.get("/audit-log")
async def audit_log(limit: int = 20):
    """Returns last N screening records from the DB as JSON."""
    return get_recent_screenings(limit=limit)


@app.post("/screen-document")
async def screen_document(
    document: UploadFile = File(..., description="Document image (passport, visa, ID card)"),
    selfie: UploadFile = File(None, description="Optional live selfie for face matching"),
    document_type: str = Form(None, description="Optional document type: PASSPORT, VISA, MRZ_ID, NON_MRZ_ID, UNKNOWN"),
):
    """
    Upload a document image (and optional selfie) for full screening.

    Runs all four modules in sequence:
      1. MRZ/OCR extraction + checksum validation
      2. Tampering detection (ELA, edge, wavelet, copy-move)
      3. Face verification (if selfie provided)
      4. Risk scoring (combines all results)

    Returns partial results if any module fails (doesn't crash).
    """
    doc_path = None
    selfie_path = None
    results = {
        "ocr": None,
        "tampering": None,
        "face": None,
        "risk": None,
        "errors": [],
    }

    try:
        session_id = uuid.uuid4().hex
        results["session_id"] = session_id
        
        # Save uploaded files
        doc_path = _save_upload(document)
        logger.info(f"Document saved: {doc_path}")

        if selfie and selfie.filename:
            selfie_path = _save_upload(selfie)
            logger.info(f"Selfie saved: {selfie_path}")

        # ── Module 1+2: MRZ Extraction ────────────────────────────────────
        try:
            logger.info(f"Running OCR/MRZ extraction (type: {document_type or 'AUTO'})...")
            ocr_result = extract_document_fields(doc_path, document_type=document_type)
            results["ocr"] = ocr_result
            logger.info(f"MRZ status: {ocr_result.get('status')}")
        except Exception as e:
            logger.error(f"MRZ extraction failed: {e}")
            results["ocr"] = {"status": "FAILURE", "error": str(e), "checksum_valid": False}
            results["errors"].append(f"MRZ extraction: {e}")

        # ── Module 3: Tampering Detection ─────────────────────────────────
        try:
            logger.info("Running tampering detection...")
            tamper_result = analyze_tampering(doc_path)
            # Remove non-serializable numpy arrays from response
            tamper_serializable = {
                "tamper_score": tamper_result.get("tamper_score"),
                "verdict": tamper_result.get("verdict"),
                "breakdown": {},
                "errors": tamper_result.get("errors"),
            }
            if "ela_heatmap_b64" in tamper_result:
                tamper_serializable["ela_heatmap_b64"] = tamper_result["ela_heatmap_b64"]
            for key, val in tamper_result.get("breakdown", {}).items():
                tamper_serializable["breakdown"][key] = {
                    k: v for k, v in val.items()
                    if not hasattr(v, "shape")  # Skip numpy arrays
                }
            results["tampering"] = tamper_serializable
            logger.info(f"Tamper score: {tamper_result.get('tamper_score')}%")
        except Exception as e:
            logger.error(f"Tampering detection failed: {e}")
            results["tampering"] = {"tamper_score": 0, "verdict": "Unknown", "error": str(e)}
            results["errors"].append(f"Tampering detection: {e}")

        # ── Module 4: Face Verification ───────────────────────────────────
        if selfie_path:
            try:
                logger.info("Running face verification...")
                passport_number = results.get("ocr", {}).get("document_number")
                face_result = verify_face_match(doc_path, selfie_path, session_id=session_id, passport_number=passport_number)
                results["face"] = face_result
                logger.info(f"Face verified: {face_result.get('verified')}")
            except Exception as e:
                logger.error(f"Face verification failed: {e}")
                results["face"] = {
                    "verified": False, "confidence": 0, "is_real": None,
                    "error": str(e),
                }
                results["errors"].append(f"Face verification: {e}")
        else:
            results["face"] = {
                "verified": None,
                "confidence": None,
                "is_real": None,
                "note": "No selfie provided -- face verification skipped",
            }

        # ── Risk Scoring ──────────────────────────────────────────────────
        try:
            logger.info("Computing risk score...")
            risk_result = compute_risk_score(
                ocr_result=results["ocr"],
                tamper_result=results.get("tampering", {"tamper_score": 0}),
                face_result=results.get("face", {"verified": None}),
            )
            results["risk"] = risk_result
            logger.info(f"Risk: {risk_result.get('risk_score')} ({risk_result.get('verdict')})")
            
            log_screening_result(results)
        except Exception as e:
            logger.error(f"Risk scoring or audit logging failed: {e}")
            results["risk"] = {"error": str(e)}
            results["errors"].append(f"Risk scoring: {e}")

        # Clean up errors list if empty
        if not results["errors"]:
            results["errors"] = None

        return JSONResponse(content=_json_safe(results))

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up uploaded files
        for path in [doc_path, selfie_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


# Mount Frontend UI at root
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    # Ensure Tesseract is in PATH
    tesseract_dir = r"C:\Program Files\Tesseract-OCR"
    if tesseract_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = tesseract_dir + ";" + os.environ.get("PATH", "")

    uvicorn.run(app, host="0.0.0.0", port=8000)

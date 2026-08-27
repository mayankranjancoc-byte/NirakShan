"""
FastAPI Backend for Document Screening Prototype

Endpoints:
  POST /screen-document  -- Upload document image + optional selfie, get risk assessment
  GET  /health           -- Liveness check
  GET  /audit-log        -- Recent screening records (auth required)
"""

import os
import sys
import uuid
import shutil
import hashlib
import logging
import tempfile
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image

# Set environment variables before importing tensorflow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

# ── Upload security constants ──────────────────────────────────────────────
MAX_UPLOAD_BYTES = 15 * 1024 * 1024   # 15 MB (images)
MAX_VIDEO_BYTES  = 50 * 1024 * 1024   # 50 MB (liveness video)
ALLOWED_CONTENT_TYPES = {
    "image/jpeg": ".jpg",
    "image/jpg":  ".jpg",
    "image/png":  ".png",
    "image/webp": ".webp",
}
ALLOWED_VIDEO_TYPES = {
    "video/mp4":       ".mp4",
    "video/webm":      ".webm",
    "video/quicktime": ".mov",
    "video/x-msvideo": ".avi",
}
# Decompression-bomb guard: refuse images larger than 80 megapixels
Image.MAX_IMAGE_PIXELS = 80_000_000

# ── API key auth ───────────────────────────────────────────────────────────
# Set NIRAKSHAN_API_KEY env var to enable. If unset, auth is bypassed (dev mode).
API_KEY = os.environ.get("NIRAKSHAN_API_KEY", "")

# Ensure backend directory is in sys.path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Add vendor directory to path for image_forensics imports
VENDOR_DIR = os.path.join(BACKEND_DIR, "vendor", "image_forensics")
if VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from modules.ocr_extraction import extract_document_fields, get_mrz_reader
from modules.tampering_detection import analyze_tampering
from modules.face_verification import verify_face_match
from modules.risk_scoring import compute_risk_score
from modules.audit_logger import log_screening_result, get_recent_screenings
from modules.document_liveness import detect_screen_replay, detect_physical_motion, extract_frames

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
UPLOAD_DIR = os.path.join(BACKEND_DIR, "uploads")
FRONTEND_DIR = os.path.abspath(os.path.join(BACKEND_DIR, "..", "frontend"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Lifespan: replaces deprecated @app.on_event("startup") ───────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up the lightweight OCR singleton. Face models are loaded lazily by
    # Module 4 so an unavailable optional ArcFace download cannot delay or
    # destabilise application startup.
    get_mrz_reader()
    yield


app = FastAPI(
    title="NirakShan Document Screening",
    description=(
        "AI-powered border document screening system. "
        "Combines MRZ extraction, tampering detection, "
        "and face verification into a unified risk assessment."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

# CORS configuration
ALLOWED_ORIGINS = os.environ.get("CORS_ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if "*" not in ALLOWED_ORIGINS else ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


# ── Auth dependency ────────────────────────────────────────────────────────
_bearer = HTTPBearer(auto_error=False)

async def _require_auth(
    cred: HTTPAuthorizationCredentials = Security(_bearer),
) -> None:
    """Reject requests without a valid Bearer token when API_KEY is set."""
    if not API_KEY:
        return  # Dev mode: auth disabled
    if cred is None or cred.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _save_upload(upload: UploadFile) -> str:
    """
    Validate and save an uploaded file to disk.
    Enforces: content-type allowlist, 15 MB size cap, decodability check.
    Path traversal is not possible — basename is a fresh uuid4.
    """
    ct = (upload.content_type or "").split(";")[0].strip().lower()
    if ct not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{ct}'. Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )
    ext = ALLOWED_CONTENT_TYPES[ct]
    filepath = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")

    written = 0
    try:
        with open(filepath, "wb") as f:
            while True:
                chunk = upload.file.read(1024 * 1024)  # 1 MB chunks
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    f.close()
                    os.remove(filepath)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB limit",
                    )
                f.write(chunk)

        # Verify the bytes are actually a decodable image
        try:
            with Image.open(filepath) as probe:
                probe.verify()
        except Exception as img_err:
            os.remove(filepath)
            raise HTTPException(
                status_code=400,
                detail=f"File is not a readable image: {img_err}",
            )
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    return filepath


def _save_upload_video(upload: UploadFile) -> str:
    """
    Validate and save a liveness video upload.
    Accepts mp4/webm/mov/avi up to 50 MB.
    """
    ct = (upload.content_type or "").split(";")[0].strip().lower()
    if ct not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported video type '{ct}'. Accepted: {list(ALLOWED_VIDEO_TYPES)}",
        )
    ext = ALLOWED_VIDEO_TYPES[ct]
    filepath = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
    written = 0
    try:
        with open(filepath, "wb") as f:
            while True:
                chunk = upload.file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_VIDEO_BYTES:
                    f.close()
                    os.remove(filepath)
                    raise HTTPException(
                        status_code=413,
                        detail=f"Video exceeds {MAX_VIDEO_BYTES // (1024*1024)} MB limit",
                    )
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(status_code=500, detail=f"Video upload failed: {e}")
    return filepath


def _sha256(path: str) -> str:
    """Compute SHA-256 of a file for chain-of-custody audit records."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


# (startup logic moved to lifespan() above)


@app.get("/health")
async def health_check():
    """Real component health check with an explicit face-model fallback."""
    components: dict[str, str] = {}
    optional_components: dict[str, str] = {}
    try:
        get_mrz_reader()
        components["mrz_scanner"] = "ok"
    except Exception as e:
        components["mrz_scanner"] = f"unavailable: {e}"

    try:
        from deepface import DeepFace as _DF
        # We no longer load the face models in the health check, because Keras models
        # use too much RAM (500MB+) and will cause OOM crashes on the free tier.
        components["face_verification"] = "ok (lazy-loaded)"
    except Exception as _e:
        components["face_biometrics"] = f"import error: {_e}"

    degraded = components.get("mrz_scanner") != "ok" or components.get("face_verification", "").startswith("unavailable")
    return JSONResponse(
        status_code=503 if components.get("mrz_scanner") != "ok" else 200,
        content={
            "status": "degraded" if degraded else "ok",
            "components": components,
            "optional_components": optional_components,
        },
    )


@app.get("/audit-log")
async def audit_log(limit: int = 20, _auth=Security(_require_auth)):
    """Returns last N screening records. Requires valid API key."""
    return get_recent_screenings(limit=min(limit, 100))


@app.post("/screen-document")
def screen_document(
    document: UploadFile = File(..., description="Document image (passport, visa, ID card)"),
    selfie:   UploadFile = File(None, description="Optional live selfie for face matching"),
    video:    UploadFile = File(None, description="Optional 2-3s tilt video for physical liveness check (mp4/webm/mov)"),
    document_type: str = Form(None, description="Optional document type: PASSPORT, VISA, MRZ_ID, NON_MRZ_ID, UNKNOWN"),
    _auth=Security(_require_auth),
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
    video_path = None
    frames_dir = None
    results = {
        "ocr": None,
        "tampering": None,
        "face": None,
        "liveness": None,
        "risk": None,
        "errors": [],
    }

    try:
        session_id = uuid.uuid4().hex
        results["session_id"] = session_id

        # Save uploaded files (validation inside _save_upload)
        doc_path = _save_upload(document)
        logger.info(f"Document saved: {doc_path}")

        # Compute SHA-256 for chain-of-custody before any processing
        results["document_sha256"] = _sha256(doc_path)

        if selfie and selfie.filename:
            selfie_path = _save_upload(selfie)
            logger.info(f"Selfie saved: {selfie_path}")
            results["selfie_sha256"] = _sha256(selfie_path)

        if video and video.filename:
            video_path = _save_upload_video(video)
            logger.info(f"Liveness video saved: {video_path}")
            results["video_sha256"] = _sha256(video_path)

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

        # ── Module 3.5: Document Liveness ─────────────────────────────────
        liveness_result = {
            "screen_replay": None,
            "physical_motion": None,
        }
        try:
            logger.info("Running document liveness check (Part A: screen replay)...")
            replay = detect_screen_replay(doc_path)
            liveness_result["screen_replay"] = replay
            logger.info(f"Screen replay: {replay.get('is_screen_replay')} ({replay.get('method')})")
        except Exception as e:
            logger.error(f"Screen replay detection failed: {e}")
            liveness_result["screen_replay"] = {"error": str(e), "is_screen_replay": None}
            results["errors"].append(f"Screen replay detection: {e}")

        if video_path:
            try:
                logger.info("Running document liveness check (Part B: physical motion)...")
                frames_dir = tempfile.mkdtemp(prefix="liveness_frames_")
                frame_paths = extract_frames(video_path, frames_dir)
                logger.info(f"Extracted {len(frame_paths)} frames for motion analysis")
                motion = detect_physical_motion(frames_dir)
                liveness_result["physical_motion"] = motion
                logger.info(f"Motion verdict: {motion.get('verdict')}")
            except Exception as e:
                logger.error(f"Physical motion detection failed: {e}")
                liveness_result["physical_motion"] = {"error": str(e), "verdict": "INCONCLUSIVE"}
                results["errors"].append(f"Physical motion detection: {e}")
        else:
            liveness_result["physical_motion"] = {
                "verdict": "SKIPPED",
                "note": "No liveness video uploaded",
            }

        results["liveness"] = liveness_result

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
                liveness_result=results.get("liveness", {}),
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
        # Clean up uploaded files and liveness temp frames
        for path in [doc_path, selfie_path, video_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        if frames_dir and os.path.exists(frames_dir):
            shutil.rmtree(frames_dir, ignore_errors=True)


# Mount Frontend UI at root
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    import argparse

    # Ensure Tesseract is in PATH on Windows
    tesseract_dir = r"C:\Program Files\Tesseract-OCR"
    if os.name == "nt" and tesseract_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = tesseract_dir + ";" + os.environ.get("PATH", "")

    parser = argparse.ArgumentParser(description="NirakShan Document Screening API")
    parser.add_argument("--expose", action="store_true",
                        help="Bind to 0.0.0.0 (external). Default: 127.0.0.1 only.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    host = "0.0.0.0" if args.expose else "127.0.0.1"
    if args.expose:
        logger.warning("Server bound to 0.0.0.0 — ensure API key auth is configured.")

    uvicorn.run(app, host=host, port=args.port)

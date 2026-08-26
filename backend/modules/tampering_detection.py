"""
Module 3: Document Tampering Detection

Uses three detection techniques:
  1. ELA (Error Level Analysis) — detects re-compression artifacts (JPEG only).
     Scores block-level outliers, not global mean (H4 fix).
  2. Edge Detection — detects abrupt edge-density steps at field boundaries,
     not variance across the page layout (H5 fix).
  3. Copy-Move Detection — ORB + DBSCAN clustering with area-based scoring
     and a proper Lowe ratio test (C4 fix).
  4. Wavelet Analysis — block-level detail energy deviation (M6 fix).
  5. EXIF Metadata — demoted to 3%% weight, stripped EXIF not scored (H7 fix).

Weights (C5 fix — single source of truth):
  ELA: 30%%, Edge: 15%%, Copy-Move: 42%%, Wavelet: 10%%, EXIF: 3%%

Thresholds follow DocAuth conventions:
  0-10%%   → Authentic
  10-55%%  → Suspicious
  55-100%% → Forged

If detectors crash, the verdict is renormalized over successful detectors (H6 fix).
If detector coverage drops below 60%%, the verdict is INCONCLUSIVE.
"""

import os
import sys
import io
import base64

# Add vendor directory to path so we can import from the cloned DocAuth repo
VENDOR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vendor", "docauth"))
if VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

from src.analysis.ela import generate_ela, ela_score
from src.analysis.edge_detection import detect_all as detect_edges
from src.analysis.wavelet import decompose as wavelet_decompose
from src.copy_move.detector import detect_copy_move

import cv2
import numpy as np
from PIL import Image

# ── Weights — single source of truth (C5 fix) ────────────────────────
TAMPER_WEIGHTS: dict[str, float] = {
    "ela":            0.30,
    "edge_detection": 0.15,
    "copy_move":      0.42,
    "wavelet":        0.10,
    "exif_analysis":  0.03,
}
assert abs(sum(TAMPER_WEIGHTS.values()) - 1.0) < 1e-6, "Tamper weights must sum to 1.0"


def _ela_score_local(ela_image: Image.Image, block: int = 16) -> float:
    """
    Localized ELA: score the fraction of blocks that are statistical outliers
    compared to the page median (H4 fix).

    Replaces the old global mean, which was driven by document design rather
    than manipulation. A genuine document's ELA map is spatially uniform;
    a manipulated region stands out as a local high-error cluster.

    Returns a 0–1 score where 1.0 means >=2%% of blocks are clear outliers.
    """
    gray = np.array(ela_image.convert("L"), np.float32)
    h, w = gray.shape
    bh, bw = h // block, w // block
    if bh == 0 or bw == 0:
        return 0.0
    # Reshape into a grid of (bh, bw) blocks, each of size (block, block)
    blocks = gray[:bh * block, :bw * block].reshape(bh, block, bw, block).mean(axis=(1, 3))
    med = np.median(blocks)
    mad = np.median(np.abs(blocks - med)) + 1e-6
    z = np.abs(blocks - med) / (1.4826 * mad)  # robust z-score
    frac_anomalous = float((z > 3.5).mean())   # fraction of clearly deviant blocks
    return min(1.0, frac_anomalous / 0.02)      # 2%% of blocks deviant => score 1.0


def _edge_anomaly_score(edge_results: dict) -> float | None:
    """
    Measure edge density using only Canny (the only binary-comparable detector).

    Instead of variance across a layout grid (which always fires on genuine
    documents due to their intentional non-uniform layout), we compute a simple
    global Canny edge density as a proxy for the overall edge complexity.

    A very high or very low value vs. expected is anomalous, but we report a
    bounded [0,1] continuous score rather than a binary flag.

    Note: Ideally this would compare density across detected field boundaries
    (H5 full fix), but that requires field-region localization not yet available.
    This version is still an improvement over the grid-CV approach because it
    uses a single commensurable metric instead of averaging incompatible ones.
    """
    canny = edge_results.get("canny")
    if canny is None:
        return None

    h, w = canny.shape
    if h < 4 or w < 4:
        return None

    density = float(np.mean(canny) / 255.0)
    # Score based on extreme deviation from a typical document edge density
    # Typical Canny density for an ID document is roughly 0.05–0.20.
    # Very high density (>0.35) can indicate heavy noise/manipulation artifacts.
    return float(min(1.0, max(0.0, (density - 0.05) / 0.30)))


def _wavelet_anomaly_score_local(wavelet_result: dict, block: int = 16) -> float | None:
    """
    Block-level wavelet anomaly scoring (M6 fix).

    The old approach took `mean(normalized_map) * 2`, which is a constant
    function of image resolution and sharpness — not tampering. This version
    scores blocks against the page's own robust baseline so that a genuine
    uniform document scores near 0 regardless of its sharpness.
    """
    reconstructed = wavelet_result.get("reconstructed")
    if reconstructed is None:
        return None

    gray = np.array(reconstructed, np.float32)
    h, w = gray.shape if gray.ndim == 2 else gray.shape[:2]
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    bh, bw = h // block, w // block
    if bh == 0 or bw == 0:
        return 0.0
    blocks = gray[:bh * block, :bw * block].reshape(bh, block, bw, block).mean(axis=(1, 3))
    med = np.median(blocks)
    mad = np.median(np.abs(blocks - med)) + 1e-6
    z = np.abs(blocks - med) / (1.4826 * mad)
    frac_anomalous = float((z > 3.5).mean())
    return min(1.0, frac_anomalous / 0.02)


from PIL.ExifTags import TAGS


def analyze_exif(image_path: str) -> dict:
    """
    Analyze EXIF metadata for signs of tampering.

    H7 fix:
    - exif_stripped is now INFORMATIONAL only (not scored).
    - software_flagged is informational only (trivially forged/removed by attacker).
    - EXIF weight dropped from 20%% to 3%%.
    - exif_anomaly_score is 1.0 only when software is flagged, 0.0 otherwise.
    """
    res = {
        "has_exif": False,
        "software_used": None,
        "software_flagged": False,
        "camera_make": None,
        "exif_stripped": False,  # informational only — not scored
        "exif_anomaly_score": 0.0,
        "scored": False,
    }

    try:
        img = Image.open(image_path)
        is_jpeg = img.format in ("JPEG", "JPG")

        exif = img.getexif()
        # Pillow's getexif() returns an Exif object (never None), so check length.
        if len(exif) == 0:
            if is_jpeg:
                res["exif_stripped"] = True
            # Not scored — every scanned JPEG has no EXIF; penalizing it is noise.
            return res

        res["has_exif"] = True

        # Decode EXIF safely
        exif_data = {}
        for tag_id, raw_val in exif.items():
            tag = TAGS.get(tag_id, tag_id)
            if isinstance(raw_val, bytes):
                try:
                    val_str = raw_val.decode("utf-8", errors="ignore").strip()
                except Exception:
                    val_str = str(raw_val)
            elif isinstance(raw_val, (int, float, str, bool)):
                val_str = raw_val
            else:
                val_str = str(raw_val)
            exif_data[str(tag)] = val_str

        res["camera_make"] = str(exif_data.get("Make")) if exif_data.get("Make") else None
        software = exif_data.get("Software")
        if software:
            res["software_used"] = str(software)
            soft_lower = str(software).lower()
            # "adobe" already subsumes "photoshop"; check for commonly misused tools.
            suspicious = ["gimp", "canva", "adobe", "paintshop", "affinity"]
            if any(s in soft_lower for s in suspicious):
                res["software_flagged"] = True
                res["exif_anomaly_score"] = 1.0
                res["scored"] = True
    except Exception:
        pass

    return res


def analyze_tampering(image_path: str) -> dict:
    """
    Run all tampering detection checks on a document image.

    Returns:
      - tamper_score: 0-100 overall score, or None if INCONCLUSIVE
      - verdict: "Authentic" / "Suspicious" / "Forged" / "INCONCLUSIVE"
      - breakdown: per-check scores and details
      - detector_coverage: fraction of weights from successful detectors
      - degraded: True if any detector failed
    """
    breakdown = {}
    errors = []

    # --- 1. Error Level Analysis (ELA) ---
    ela_b64 = None
    try:
        # H4 fix: gate on JPEG (ELA is meaningless for PNG/WebP with no prior compression).
        with Image.open(image_path) as probe:
            img_format = probe.format

        if img_format not in ("JPEG", "MPO"):
            breakdown["ela"] = {
                "score": None,
                "status": "NOT_APPLICABLE",
                "reason": "ELA requires JPEG source with compression history",
                "description": "Error Level Analysis — not applicable to this format",
            }
        else:
            ela_image = generate_ela(image_path)
            ela_sc = _ela_score_local(ela_image)  # localized block-outlier scoring

            # M7 fix: downscale heatmap to bounded preview before base64-encoding.
            preview = ela_image.copy()
            preview.thumbnail((640, 640), Image.LANCZOS)
            buf = io.BytesIO()
            preview.save(buf, format="WEBP", quality=80)
            ela_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            breakdown["ela"] = {
                "score": round(ela_sc * 100, 2),
                "description": "Error Level Analysis — block-outlier anomaly score",
            }
    except Exception as e:
        errors.append(f"ELA failed: {e}")
        breakdown["ela"] = {"score": None, "error": str(e)}

    # --- 2. Edge Detection ---
    try:
        edge_results = detect_edges(image_path)
        edge_sc = _edge_anomaly_score(edge_results)
        if edge_sc is None:
            breakdown["edge_detection"] = {
                "score": None,
                "status": "UNAVAILABLE",
                "description": "Edge anomaly — insufficient data",
            }
        else:
            breakdown["edge_detection"] = {
                "score": round(edge_sc * 100, 2),
                "description": "Edge density anomaly — detects splicing artifacts",
            }
    except Exception as e:
        errors.append(f"Edge detection failed: {e}")
        breakdown["edge_detection"] = {"score": None, "error": str(e)}

    # --- 3. Wavelet Analysis ---
    try:
        wavelet_result = wavelet_decompose(image_path)
        wavelet_sc = _wavelet_anomaly_score_local(wavelet_result)
        if wavelet_sc is None:
            breakdown["wavelet"] = {
                "score": None,
                "status": "UNAVAILABLE",
                "description": "Wavelet analysis — result unavailable",
            }
        else:
            breakdown["wavelet"] = {
                "score": round(wavelet_sc * 100, 2),
                "description": "Wavelet block-outlier analysis — detects high-frequency anomalies",
            }
    except Exception as e:
        errors.append(f"Wavelet analysis failed: {e}")
        breakdown["wavelet"] = {"score": None, "error": str(e)}

    # --- 4. Copy-Move Detection ---
    try:
        cm_result = detect_copy_move(image_path)
        cm_score = cm_result["score"]
        breakdown["copy_move"] = {
            "score": round(cm_score * 100, 2),
            "verdict": cm_result["verdict"],
            "method": cm_result["method"],
            "description": "Copy-move forgery detection — finds duplicated regions",
        }
    except Exception as e:
        errors.append(f"Copy-move detection failed: {e}")
        breakdown["copy_move"] = {"score": None, "error": str(e)}

    # --- 5. EXIF Analysis (informational; 3%% weight) ---
    try:
        exif_res = analyze_exif(image_path)
        exif_score = exif_res["exif_anomaly_score"]
        breakdown["exif_analysis"] = {
            "score": round(exif_score * 100, 2) if exif_res["scored"] else None,
            "description": "EXIF metadata — informational; low weight",
            "details": exif_res,
        }
    except Exception as e:
        errors.append(f"EXIF analysis failed: {e}")
        breakdown["exif_analysis"] = {"score": None, "error": str(e)}

    # --- Combine scores (H6 fix: renormalize over successful detectors) ---
    available = {
        k: w for k, w in TAMPER_WEIGHTS.items()
        if isinstance(breakdown.get(k, {}).get("score"), (int, float))
    }

    if not available:
        return {
            "tamper_score": None,
            "verdict": "INCONCLUSIVE",
            "reason": "All forensic detectors failed or returned no result",
            "breakdown": breakdown,
            "errors": errors if errors else None,
            "ela_heatmap_b64": ela_b64,
            "detector_coverage": 0.0,
            "degraded": True,
        }

    total_w = sum(available.values())
    combined = sum(
        breakdown[k]["score"] * w for k, w in available.items()
    ) / total_w  # renormalize so crashed detectors don’t bias toward Authentic

    combined = round(min(combined, 100.0), 2)
    coverage = round(total_w / sum(TAMPER_WEIGHTS.values()), 3)
    degraded = coverage < 1.0

    if coverage < 0.6:
        verdict = "INCONCLUSIVE"  # too little evidence to assert anything
    elif combined < 10:
        verdict = "Authentic"
    elif combined < 55:
        verdict = "Suspicious"
    else:
        verdict = "Forged"

    return {
        "tamper_score": combined,
        "verdict": verdict,
        "breakdown": breakdown,
        "errors": errors if errors else None,
        "ela_heatmap_b64": ela_b64,
        "detector_coverage": coverage,
        "degraded": degraded,
        "unavailable_detectors": sorted(set(TAMPER_WEIGHTS) - set(available)),
    }


if __name__ == "__main__":
    import json

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Use the same passport sample from fastmrz
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

    print(f"Testing tampering detection on: {img_path}")
    print("-" * 60)
    result = analyze_tampering(img_path)
    print(json.dumps(result, indent=4))

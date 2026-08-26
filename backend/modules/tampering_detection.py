"""
Module 3: Document Tampering Detection
Wraps forensic analysis from DocAuth (https://github.com/trinity652/DocAuth)
License: MIT

Uses three detection techniques from the cloned repo:
  1. ELA (Error Level Analysis) — detects re-compression artifacts
  2. Edge Detection — detects inconsistent edges from splicing
  3. Copy-Move Detection — detects duplicated regions (ORB+RANSAC)

Also includes wavelet analysis from DocAuth for texture anomaly detection.

Thresholds follow DocAuth conventions:
  0-10%   → Authentic
  10-55%  → Suspicious
  55-100% → Forged
"""

import os
import sys

# Add vendor directory to path so we can import from the cloned DocAuth repo
VENDOR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vendor", "docauth"))
if VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

from src.analysis.ela import generate_ela, ela_score
from src.analysis.edge_detection import detect_all as detect_edges
from src.analysis.wavelet import decompose as wavelet_decompose
from src.copy_move.detector import detect_copy_move

import numpy as np


def _edge_anomaly_score(edge_results: dict) -> float:
    """
    Compute an anomaly score from edge detection results.

    Higher scores suggest inconsistent edge patterns (potential splicing).
    Measures variance of edge density across regions — a uniform document
    should have relatively consistent edge density, while spliced regions
    tend to show abrupt transitions.
    """
    scores = []
    for name, edge_map in edge_results.items():
        if edge_map is None:
            continue
        # Divide image into a 4x4 grid and measure edge density variance
        h, w = edge_map.shape
        grid_h, grid_w = h // 4, w // 4
        densities = []
        for i in range(4):
            for j in range(4):
                block = edge_map[i * grid_h:(i + 1) * grid_h, j * grid_w:(j + 1) * grid_w]
                density = np.mean(block) / 255.0
                densities.append(density)
        # Coefficient of variation — higher means more inconsistent edges
        densities = np.array(densities)
        if densities.mean() > 0:
            cv = densities.std() / densities.mean()
        else:
            cv = 0.0
        scores.append(min(cv, 1.0))

    return float(np.mean(scores)) if scores else 0.0


def _wavelet_anomaly_score(wavelet_result: dict) -> float:
    """
    Compute anomaly score from wavelet decomposition.

    High-frequency detail energy in manipulated regions tends to differ
    from the document background.
    """
    reconstructed = wavelet_result.get("reconstructed")
    if reconstructed is None:
        return 0.0

    # Normalized energy of the detail-only reconstruction
    energy = np.mean(reconstructed.astype(np.float32)) / 255.0
    return float(min(energy * 2.0, 1.0))  # Scale up, cap at 1.0


from PIL import Image
from PIL.ExifTags import TAGS

def analyze_exif(image_path: str) -> dict:
    """Analyze EXIF metadata for signs of tampering."""
    res = {
        "has_exif": False,
        "software_used": None,
        "software_flagged": False,
        "camera_make": None,
        "exif_stripped": False,
        "exif_anomaly_score": 0.0
    }
    
    try:
        img = Image.open(image_path)
        
        # Check if JPEG
        is_jpeg = img.format in ("JPEG", "JPG")
        
        exif = img.getexif()
        if exif is None or len(exif) == 0:
            if is_jpeg:
                res["exif_stripped"] = True
                res["exif_anomaly_score"] = 0.5
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
            suspicious = ["photoshop", "gimp", "canva", "adobe", "paint"]
            if any(s in soft_lower for s in suspicious):
                res["software_flagged"] = True
                res["exif_anomaly_score"] = 1.0
                
    except Exception:
        pass
        
    return res


def analyze_tampering(image_path: str) -> dict:
    """
    Run all tampering detection checks on a document image.

    Args:
        image_path: Path to the document image.

    Returns:
        dict with:
          - tamper_score: 0-100 overall score
          - verdict: "Authentic" / "Suspicious" / "Forged"
          - breakdown: per-check scores and details
    """
    breakdown = {}
    errors = []

    # --- 1. Error Level Analysis (ELA) ---
    ela_b64 = None
    try:
        ela_image = generate_ela(image_path)
        ela_sc = ela_score(ela_image)
        
        import base64, io
        buffer = io.BytesIO()
        ela_image.save(buffer, format="PNG")
        ela_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        breakdown["ela"] = {
            "score": round(ela_sc * 100, 2),
            "description": "Error Level Analysis — detects re-compression artifacts",
        }
    except Exception as e:
        errors.append(f"ELA failed: {e}")
        breakdown["ela"] = {"score": 0, "error": str(e)}

    # --- 2. Edge Detection ---
    try:
        edge_results = detect_edges(image_path)
        edge_sc = _edge_anomaly_score(edge_results)
        breakdown["edge_detection"] = {
            "score": round(edge_sc * 100, 2),
            "description": "Edge consistency analysis — detects splicing artifacts",
            "detectors_used": list(edge_results.keys()),
        }
    except Exception as e:
        errors.append(f"Edge detection failed: {e}")
        breakdown["edge_detection"] = {"score": 0, "error": str(e)}

    # --- 3. Wavelet Analysis ---
    try:
        wavelet_result = wavelet_decompose(image_path)
        wavelet_sc = _wavelet_anomaly_score(wavelet_result)
        breakdown["wavelet"] = {
            "score": round(wavelet_sc * 100, 2),
            "description": "Wavelet texture analysis — detects high-frequency anomalies",
        }
    except Exception as e:
        errors.append(f"Wavelet analysis failed: {e}")
        breakdown["wavelet"] = {"score": 0, "error": str(e)}

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
        breakdown["copy_move"] = {"score": 0, "error": str(e)}

    # --- 5. EXIF Analysis ---
    try:
        exif_res = analyze_exif(image_path)
        exif_score = exif_res["exif_anomaly_score"]
        breakdown["exif_analysis"] = {
            "score": round(exif_score * 100, 2),
            "description": "EXIF metadata analysis — detects stripped or altered metadata",
            "details": exif_res
        }
    except Exception as e:
        errors.append(f"EXIF analysis failed: {e}")
        breakdown["exif_analysis"] = {"score": 0, "error": str(e)}

    # --- Combine scores ---
    # Weighted combination:
    #   ELA: 25%, Edge: 15%, Wavelet: 15%, Copy-Move: 25%, EXIF: 20%
    weights = {
        "ela": 0.25,
        "edge_detection": 0.15,
        "wavelet": 0.15,
        "copy_move": 0.25,
        "exif_analysis": 0.20,
    }

    combined = 0.0
    for key, weight in weights.items():
        score = breakdown.get(key, {}).get("score", 0)
        combined += score * weight

    combined = round(min(combined, 100.0), 2)

    # DocAuth thresholds
    if combined < 10:
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

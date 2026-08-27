"""
Module 3.5 — Document Liveness Check

Two sub-components address different attack vectors:

Part A — Screen/Recapture Detection
    Catches an attacker photographing a phone screen, monitor, or printed photocopy.
    Method: Feature-based SVM classifier using colour moments + difference histograms
    + gradient-texture descriptors derived from the liveness_core repo.
    Fallback: if no trained SVM exists yet, returns a threshold-based score
    (explicitly labelled "method": "threshold") so the system degrades gracefully.

Part B — Physical Motion / Hologram Tracking
    Confirms the document is a real 3D object being tilted, not a flat static image.
    Method: Dense optical flow across video frames (Farneback) tracking specular
    highlight displacement + HSV histogram shift in the document's bright region.
    Returns INCONCLUSIVE if fewer than 5 frames are available.
"""

import os
import sys
import shutil
import tempfile
import logging
import threading

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

_MODULE_DIR = os.path.dirname(__file__)
_VENDOR_DIR = os.path.join(_MODULE_DIR, "..", "vendor", "liveness_core")
_MODEL_PATH = os.path.join(_MODULE_DIR, "..", "models", "screen_replay_svm.joblib")

# Add vendor dir to path so we can import features.py
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_VENDOR_DIR))

# ── SVM singleton ─────────────────────────────────────────────────────────────

_svm = None
_svm_lock = threading.Lock()


def _load_svm():
    """Lazy-load the trained SVM. Returns None if not yet trained."""
    global _svm
    if _svm is None:
        with _svm_lock:
            if _svm is None:
                model_abs = os.path.abspath(_MODEL_PATH)
                if not os.path.exists(model_abs):
                    return None
                try:
                    import joblib  # scikit-learn includes this
                    _svm = joblib.load(model_abs)
                    logger.info("Loaded screen-replay SVM from %s", model_abs)
                except Exception as e:
                    logger.warning("Failed to load screen-replay SVM: %s", e)
                    return None
    return _svm


# ── Threshold constants (Part A fallback) ────────────────────────────────────

# When the SVM is absent, we use gradient texture uniformity as a heuristic.
# Screen-reproduced images have highly uniform gradient histograms because the
# pixel grid introduces a periodic low-variance structure.
_TEXTURE_UNIFORMITY_THRESHOLD = 0.65   # values above this → likely screen replay
_FFT_PEAK_RATIO_THRESHOLD = 0.18        # secondary signal: peak / mean in mid-freq ring

# ── Optical flow constants (Part B) ──────────────────────────────────────────

_MIN_FRAMES_FOR_VERDICT = 5
_SPECULAR_BRIGHTNESS_THRESHOLD = 230   # pixel brightness to be "specular highlight"
_MOTION_DISPLACEMENT_THRESHOLD = 3.0  # pixels — minimum highlight displacement per frame pair
_HOLOGRAM_SHIFT_THRESHOLD = 0.12       # HSV histogram chi-square distance first vs last frame


# ─────────────────────────────────────────────────────────────────────────────
# Part A: detect_screen_replay
# ─────────────────────────────────────────────────────────────────────────────

def _fft_peak_ratio(grey: np.ndarray) -> float:
    """
    Compute the ratio of dominant mid-frequency peak to background in the FFT
    spectrum. Screen/printed-on-screen patterns leave a periodic grid that
    shows up as concentrated energy in the mid-frequency annular band.
    """
    f = np.fft.fft2(grey.astype("float32"))
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log1p(np.abs(fshift))

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    # Mid-frequency annular band: pixels between r1 and r2 from centre
    r1 = int(min(h, w) * 0.05)
    r2 = int(min(h, w) * 0.30)

    y_idx, x_idx = np.ogrid[:h, :w]
    dist = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
    ring_mask = (dist >= r1) & (dist <= r2)
    ring_values = magnitude[ring_mask]

    if ring_values.size == 0:
        return 0.0

    peak = float(np.percentile(ring_values, 99))
    mean = float(np.mean(ring_values))
    return peak / (mean + 1e-7)


def _texture_uniformity(grey: np.ndarray) -> float:
    """
    Gradient histogram uniformity. A flat histogram (low std, high entropy-like
    uniformity) indicates a periodic structured pattern (screen pixels).
    Returns a 0–1 score where 1.0 = perfectly uniform = likely screen.
    """
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    hist, _ = np.histogram(mag.ravel(), bins=64, range=(0, 512))
    hist = hist.astype("float") / (hist.sum() + 1e-7)
    uniformity = 1.0 - float(np.std(hist)) * 64.0  # normalised std
    return float(np.clip(uniformity, 0.0, 1.0))


def detect_screen_replay(image_path: str) -> dict:
    """
    Part A: Detect whether the document image was captured from a screen or
    photocopy rather than a genuine physical document.

    Returns:
        {
          "is_screen_replay": bool | None,   # None = inconclusive / unavailable
          "confidence": float,               # 0.0–1.0  (SVM probability or heuristic)
          "method": "svm" | "threshold" | "unavailable",
          "fft_peak_ratio": float,           # mid-frequency ring peak/mean
          "texture_uniformity": float,       # gradient histogram uniformity
          "error": str | None,               # set if processing failed
        }
    """
    result = {
        "is_screen_replay": None,
        "confidence": 0.0,
        "method": "unavailable",
        "fft_peak_ratio": 0.0,
        "texture_uniformity": 0.0,
        "error": None,
    }

    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            result["error"] = "Could not read image"
            return result

        grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Always compute signal features for logging / fallback
        fft_ratio = _fft_peak_ratio(grey)
        tex_uni = _texture_uniformity(grey)
        result["fft_peak_ratio"] = round(fft_ratio, 4)
        result["texture_uniformity"] = round(tex_uni, 4)

        # ── Attempt SVM path ──────────────────────────────────────────────
        svm = _load_svm()
        if svm is not None:
            try:
                from features import extract_features  # vendored
                feat = extract_features(image_path)
                if feat is not None:
                    prob = svm.predict_proba(feat.reshape(1, -1))[0]
                    # Label 1 = screen replay by training convention
                    label = int(svm.predict(feat.reshape(1, -1))[0])
                    confidence = float(prob[label])
                    result["is_screen_replay"] = bool(label == 1)
                    result["confidence"] = round(confidence, 4)
                    result["method"] = "svm"
                    return result
            except Exception as e:
                logger.warning("SVM inference failed, falling back to threshold: %s", e)

        # ── Heuristic fallback ────────────────────────────────────────────
        # These signals are useful diagnostics, but they are not calibrated
        # enough to make an automated fraud decision. In particular, normal
        # passport scans can contain strong periodic print patterns. Only a
        # trained SVM is permitted to return a positive replay verdict.
        #
        # Keep the calculated suspicion score for the UI and audit trail, but
        # mark the decision inconclusive so a genuine document is never
        # penalised solely by this fallback.
        combined = (fft_ratio / _FFT_PEAK_RATIO_THRESHOLD +
                    tex_uni / _TEXTURE_UNIFORMITY_THRESHOLD) / 2.0
        result["is_screen_replay"] = None
        result["confidence"] = round(float(np.clip(combined / 2.0, 0.0, 1.0)), 4)
        result["method"] = "heuristic"
        result["note"] = "Advisory heuristic only; a trained SVM is required for a replay verdict."

    except Exception as e:
        result["error"] = str(e)
        logger.exception("detect_screen_replay failed for %s", image_path)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Part B: detect_physical_motion
# ─────────────────────────────────────────────────────────────────────────────

def _find_specular_region(frame: np.ndarray) -> np.ndarray | None:
    """
    Return a binary mask of high-brightness (specular) pixels.
    Returns None if fewer than 50 specular pixels are found.
    """
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(grey, _SPECULAR_BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)
    if cv2.countNonZero(mask) < 50:
        return None
    return mask


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return None
    return M["m10"] / M["m00"], M["m01"] / M["m00"]


def _hsv_histogram(frame: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """32-bin H + 32-bin S HSV histogram, optionally masked to a region."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], mask, [32], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], mask, [32], [0, 256])
    combined = np.concatenate([h_hist.ravel(), s_hist.ravel()])
    cv2.normalize(combined, combined)
    return combined


def _chi_square(h1: np.ndarray, h2: np.ndarray) -> float:
    return float(cv2.compareHist(h1.astype("float32"), h2.astype("float32"),
                                 cv2.HISTCMP_CHISQR_ALT))


def extract_frames(video_path: str, out_dir: str, max_frames: int = 60) -> list[str]:
    """
    Extract up to `max_frames` evenly-spaced frames from a video.
    Saves as JPEG to `out_dir`. Returns list of saved frame paths.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, total // max_frames) if total > 0 else 1

    paths = []
    idx = 0
    saved = 0
    while saved < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        p = os.path.join(out_dir, f"frame_{saved:04d}.jpg")
        cv2.imwrite(p, frame)
        paths.append(p)
        saved += 1
        idx += step

    cap.release()
    return paths


def detect_physical_motion(frames_dir: str) -> dict:
    """
    Part B: Confirm the document is a real 3D object being tilted.

    Analyses a directory of sequential JPEG frames (from extract_frames()).

    Returns:
        {
          "verdict": "PHYSICAL" | "STATIC" | "INCONCLUSIVE",
          "motion_detected": bool | None,
          "hologram_shift_detected": bool | None,
          "mean_highlight_displacement_px": float,
          "hologram_hsv_shift": float,        # chi-square distance 0 = identical
          "frame_count": int,
          "note": str,
        }
    """
    result = {
        "verdict": "INCONCLUSIVE",
        "motion_detected": None,
        "hologram_shift_detected": None,
        "mean_highlight_displacement_px": 0.0,
        "hologram_hsv_shift": 0.0,
        "frame_count": 0,
        "note": "",
    }

    try:
        frame_files = sorted(
            f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        result["frame_count"] = len(frame_files)

        if len(frame_files) < _MIN_FRAMES_FOR_VERDICT:
            result["note"] = (
                f"Only {len(frame_files)} frames available; "
                f"need ≥{_MIN_FRAMES_FOR_VERDICT} for a verdict."
            )
            return result

        frames = []
        for fn in frame_files:
            f = cv2.imread(os.path.join(frames_dir, fn))
            if f is not None:
                frames.append(f)

        if len(frames) < _MIN_FRAMES_FOR_VERDICT:
            result["note"] = "Insufficient readable frames."
            return result

        # ── Specular highlight tracking ───────────────────────────────────
        prev_grey = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        displacements = []

        highlight_centroids = []
        spec_mask_0 = _find_specular_region(frames[0])
        if spec_mask_0 is None:
            result["note"] = "No specular highlights found; using full-frame optical flow."
        c0 = _centroid(spec_mask_0) if spec_mask_0 is not None else None
        if c0:
            highlight_centroids.append(c0)

        for frame in frames[1:]:
            curr_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Dense optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_grey, curr_grey, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2,
                flags=0,
            )

            # Measure flow specifically in the specular region if detected
            spec_mask = _find_specular_region(frame)
            if spec_mask is not None and spec_mask_0 is not None:
                region_flow = flow[spec_mask > 0]
            else:
                region_flow = flow.reshape(-1, 2)

            if region_flow.size > 0:
                magnitudes = np.linalg.norm(region_flow, axis=1)
                displacements.append(float(np.mean(magnitudes)))

            # Track specular centroid
            c = _centroid(spec_mask) if spec_mask is not None else None
            if c:
                highlight_centroids.append(c)

            prev_grey = curr_grey

        mean_disp = float(np.mean(displacements)) if displacements else 0.0
        result["mean_highlight_displacement_px"] = round(mean_disp, 3)

        # ── Hologram HSV shift (first vs last frame) ──────────────────────
        # Use the brightest quadrant of the first and last frame as "hologram region"
        h, w = frames[0].shape[:2]
        # Holographic foils are typically top-right of passport; compare that quadrant
        roi_first = frames[0][:h // 2, w // 2:]
        roi_last = frames[-1][:h // 2, w // 2:]
        hist_first = _hsv_histogram(roi_first)
        hist_last = _hsv_histogram(roi_last)
        hsv_shift = _chi_square(hist_first, hist_last)
        result["hologram_hsv_shift"] = round(hsv_shift, 4)

        # ── Verdict ───────────────────────────────────────────────────────
        motion_ok = mean_disp >= _MOTION_DISPLACEMENT_THRESHOLD
        hologram_ok = hsv_shift >= _HOLOGRAM_SHIFT_THRESHOLD

        result["motion_detected"] = motion_ok
        result["hologram_shift_detected"] = hologram_ok

        if motion_ok and hologram_ok:
            result["verdict"] = "PHYSICAL"
            result["note"] = "Consistent specular motion and hologram colour shift detected."
        elif motion_ok and not hologram_ok:
            result["verdict"] = "PHYSICAL"
            result["note"] = (
                "Specular motion confirmed; hologram shift below threshold "
                "(may be valid if document lacks overt hologram)."
            )
        elif not motion_ok and hologram_ok:
            result["verdict"] = "INCONCLUSIVE"
            result["note"] = (
                "Hologram shift present but highlight displacement too small; "
                "may be minimal tilt or background artefact."
            )
        else:
            result["verdict"] = "STATIC"
            result["note"] = (
                "No specular displacement or hologram shift — document appears static. "
                "Possible flat photo replay or insufficient tilt during capture."
            )

    except Exception as e:
        result["error"] = str(e)
        result["note"] = f"Processing failed: {e}"
        logger.exception("detect_physical_motion failed in %s", frames_dir)

    return result

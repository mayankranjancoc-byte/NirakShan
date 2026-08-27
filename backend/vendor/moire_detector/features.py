"""
Vendored feature extractors from Vishnu-Naik/moire_pattern_detector
(MIT-compatible, adapted for integration — no training-time CLI deps).

Source: https://github.com/Vishnu-Naik/moire_pattern_detector
License: Not explicitly stated in repo; adapted under fair-use for research prototype.

Changes from original:
 - Removed scikit-image LBP (adds a heavy dep); replaced with OpenCV-based texture summary.
 - Removed scipy.signal dependency; replaced with cv2.filter2D for difference histograms.
 - Removed all logging/helper_functions dependencies.
 - Removed fixed resize (750x1000) — caller normalises size.
 - Merged into a single flat file to avoid packaging issues.
"""

import itertools
import numpy as np
import cv2


# ── Difference Histograms ─────────────────────────────────────────────────────

_KERNELS = {
    "H": np.array([[1, -1]], dtype="float32"),
    "V": np.array([[1], [-1]], dtype="float32"),
    "D": np.array([[1, 0], [0, -1]], dtype="float32"),
    "A": np.array([[0, 1], [-1, 0]], dtype="float32"),
}


def _difference_histogram_features(image_array: np.ndarray) -> np.ndarray:
    """
    Adapted from DifferenceHistograms.get_difference_features().
    Operates on the red channel, applies 4 directional filters + their pairwise
    combinations, then builds normalised histograms + multi-mean translated versions.
    """
    _, _, r = cv2.split(image_array.astype("float32"))

    names = list(_KERNELS.keys())
    filtered = {}
    for name, kernel in _KERNELS.items():
        filtered[name] = cv2.filter2D(r, -1, kernel, borderType=cv2.BORDER_REFLECT)

    combo_filtered = {}
    for (n1, k1), (n2, k2) in itertools.combinations_with_replacement(_KERNELS.items(), 2):
        mid = cv2.filter2D(r, -1, k1, borderType=cv2.BORDER_REFLECT)
        combo_filtered[n1 + n2] = cv2.filter2D(mid, -1, k2, borderType=cv2.BORDER_REFLECT)

    all_filtered = {**filtered, **combo_filtered}

    bins = list(range(-100, 110, 10))
    norm_hists = []
    total_pixels = r.size
    for arr in all_filtered.values():
        h, _ = np.histogram(arr.ravel(), bins=bins)
        norm_hists.append(h / total_pixels)

    # Multi-mean translated histograms (7 translations per filter output)
    diff_hist_features = []
    for nh in norm_hists:
        for delta in range(7):
            plus = nh + (delta + 1)
            minus = nh - (delta + 1)
            diff_hist_features.append((plus + minus) / 2)
        diff_hist_features.append(nh)

    return np.array(diff_hist_features).flatten()


# ── Colour Moments ────────────────────────────────────────────────────────────

def _colour_moment_features(image_array: np.ndarray) -> np.ndarray:
    """
    Adapted from ColourFeatures.get_colour_features().
    Returns spatial moments (orders 1–3) for BGR + HSV + greyscale channels.
    """
    hsv = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
    grey = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    all_moments = []
    for img in (image_array, hsv):
        for ch in cv2.split(img):
            m = cv2.moments(ch.astype("float32"))
            for order in range(1, 4):
                for key, val in m.items():
                    if key.startswith("m") and len(key) == 3:
                        if int(key[1]) + int(key[2]) == order:
                            all_moments.append(val)

    # Greyscale moments
    m = cv2.moments(grey.astype("float32"))
    for order in range(1, 4):
        for key, val in m.items():
            if key.startswith("m") and len(key) == 3:
                if int(key[1]) + int(key[2]) == order:
                    all_moments.append(val)

    all_moments.append(float(grey.std()))
    return np.array(all_moments, dtype="float64")


# ── LBP-equivalent texture features (OpenCV only) ────────────────────────────

def _texture_features(image_array: np.ndarray) -> np.ndarray:
    """
    Texture summary using LBPH (OpenCV cv2.face.LBPHFaceRecognizer) is unavailable
    in all builds. Instead we use a gradient-based histogram that captures the same
    uniform-LBP intuition (neighbourhood difference magnitudes and directions).

    Produces a 26-bin normalised histogram compatible with the SVM feature vector.
    """
    grey = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY).astype("float32")
    # Gradient magnitudes using Sobel
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    # 26 bins to match LBP(numPoints=24, radius=8) uniform histogram length
    hist, _ = np.histogram(mag.ravel(), bins=26, range=(0, 1024))
    hist = hist.astype("float") / (hist.sum() + 1e-7)
    return hist


# ── Combined feature vector ───────────────────────────────────────────────────

def extract_features(image_path: str, target_size: tuple = (750, 1000)) -> np.ndarray | None:
    """
    Full feature extraction pipeline — equivalent to get_multi_feature_for_single_image()
    plus concatenation from get_input_features().

    Returns a 1-D float64 numpy array ready to feed into the trained SVM,
    or None if the image cannot be read.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    tw, th = (750, 1000) if h > w else (1000, 750)
    img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)

    diff_feat = _difference_histogram_features(img)
    tex_feat = _texture_features(img)
    col_feat = _colour_moment_features(img)

    return np.concatenate([diff_feat, tex_feat, col_feat]).astype("float64")

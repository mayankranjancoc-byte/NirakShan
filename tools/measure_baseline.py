"""
Baseline measurement script (Phase 2.0).

Run this BEFORE and AFTER the Module 3 forensic scorer fixes.
All sample documents are GENUINE \u2014 every verdict should be "Authentic".
If most read "Suspicious", the H4/H5/M6/C4 false-positive predictions are confirmed.

Usage:
    cd backend
    python ../tools/measure_baseline.py
"""

import os
import sys
import json

# Add backend to path
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
VENDOR_DIR = os.path.join(BACKEND_DIR, "vendor", "docauth")
for d in (BACKEND_DIR, VENDOR_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

SAMPLE_DIR = os.path.join(BACKEND_DIR, "vendor", "fastmrz", "data")
SAMPLES = ["passport_uk.jpg", "td1.jpg", "td2.jpg", "td3.jpg", "mrva.jpg", "mrvb.jpg"]

from modules.tampering_detection import analyze_tampering  # noqa: E402

def main():
    print(f"\n{'document':<22} {'tamper':>7} {'verdict':<14} {'ela':>7} {'edge':>7} {'wave':>7} {'cm':>7} {'exif':>7} {'cov':>6}")
    print("-" * 95)

    all_authentic = True
    for name in SAMPLES:
        path = os.path.join(SAMPLE_DIR, name)
        if not os.path.exists(path):
            print(f"{name:<22} {'NOT FOUND'}")
            continue
        r = analyze_tampering(path)
        b = r.get("breakdown", {})

        def s(key):
            val = b.get(key, {}).get("score")
            return f"{val:>7.1f}" if isinstance(val, (int, float)) else f"{'N/A':>7}"

        ts = r.get("tamper_score")
        ts_str = f"{ts:>7.1f}" if isinstance(ts, (int, float)) else f"{'N/A':>7}"
        verdict = r.get("verdict", "?")
        cov = r.get("detector_coverage", 0.0)

        if verdict != "Authentic":
            all_authentic = False

        print(f"{name:<22} {ts_str} {verdict:<14} {s('ela')} {s('edge_detection')} {s('wavelet')} {s('copy_move')} {s('exif_analysis')} {cov:>6.2f}")

    print("-" * 95)
    if all_authentic:
        print("\n\u2705 All genuine documents scored 'Authentic'. False-positive risk is LOW.")
    else:
        print("\n\u274c Some genuine documents scored non-Authentic. FALSE POSITIVE RISK CONFIRMED.")
        print("   Check the breakdown columns to identify which detectors are driving high scores.")
        print("   Expected culprits: edge_detection (layout) and wavelet (normalization).")


if __name__ == "__main__":
    main()

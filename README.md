# NirakShan — AI Document Screening Prototype

NirakShan is a local FastAPI prototype for screening travel and identity documents. It combines OCR/MRZ validation, image-forensics signals, document liveness checks, face matching, and an explainable risk score.

## Verified status

- Live bundled-passport request: OCR valid, checksum valid, tamper verdict authentic, fixed-scale risk response successful.
- Face matching: uses strict portrait crops and currently falls back to **VGG-Face** when optional ArcFace weights are unavailable.
## Modules

| Module | Capability | Current state |
|---|---|---|
| 1 | OCR extraction | MRZScanner reads TD1/TD2/TD3/MRV documents; Tesseract supplies a text fallback for non-MRZ documents. |
| 2 | Document validation | ICAO check digits, expiry handling, document status, and visa-field parsing. |
| 3 | Static tamper forensics | ELA, edge, wavelet, copy-move, and EXIF signals. ELA/wavelet require structural corroboration before affecting the automated score. |
| 3.5 | Document liveness | Short tilt-video upload/recording, frame extraction, optical flow, highlight motion, and HSV colour-shift analysis. Screen-replay heuristics are advisory until an SVM is trained. |
| 4 | Face verification | Strict document-portrait/selfie cropping, VGG-Face comparison, quality flags, and cross-attempt identity matching. |
| 5 | Risk scoring | Explainable fixed 0–100 score: OCR/validation (40), tampering (30), and face checks (30). Optional clean checks never change the score; hard signals produce a separate manual-review requirement. |

## Run locally

Requirements:

- Python 3.13 with the existing `backend/.venv`
- Tesseract OCR installed at `C:\Program Files\Tesseract-OCR\tesseract.exe`

```powershell
cd C:\SPACE\HACKATHONS\SIH\doc-screening-prototype\backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open:

- Web UI: http://127.0.0.1:8000/
- API docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

## Test

```powershell
cd C:\SPACE\HACKATHONS\SIH\doc-screening-prototype\backend
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

## Important limitations

- The screen-replay fallback is **not** a trained classifier. It exposes FFT/texture signals but does not automatically mark a document as forged; only a trained SVM may issue that verdict.
- The physical liveness workflow needs real tilted-document and replay-video validation before it is presented as a reliable security control.
- Document-liveness results are review-only signals until calibrated against a labelled operational dataset; they do not change the fixed risk score.
- Face anti-spoofing is currently unavailable because PyTorch is not installed. Ordinary document-to-selfie matching remains available through VGG-Face.
- ArcFace is optional and presently unavailable because its local weights are incomplete; the application deliberately uses VGG-Face instead of failing.
- This is a decision-support prototype, not a replacement for issuing-authority, blacklist, or ICAO-chip verification systems.

## Technologies Used

- **Backend:** FastAPI, Uvicorn, Python 3.13, SQLite
- **Computer Vision & Forensics:** OpenCV, NumPy, Pillow, PyWavelets
- **OCR:** Tesseract OCR, ICAO 9303 MRZ parsing logic
- **Biometrics & Liveness:** VGG-Face architecture, Farneback Dense Optical Flow, Fast Fourier Transform (FFT)
- **Frontend:** HTML5, CSS3, Vanilla JavaScript

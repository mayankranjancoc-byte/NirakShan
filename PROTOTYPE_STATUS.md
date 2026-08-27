# NirakShan Prototype Status

> Last verified: **27 August 2026**
>
> Project: `doc-screening-prototype/` · Problem statement: `SIH26188`

## Verification summary

| Check | Result |
|---|---|
| Python syntax checks | Passed |
| Automated tests | **35/35 passed** |
| `GET /health` | API starts; MRZScanner and a VGG-Face fallback are operational |
| `POST /screen-document` with bundled passport | Successful JSON response; OCR valid, checksum valid, tampering authentic, LOW risk |
| Face-match smoke test | Same document portrait compared to itself returned a match after strict cropping |

## Module stages

| Module | Stage | What is implemented | What remains |
|---|---|---|---|
| **1 — OCR** | Verified | MRZScanner MRZ extraction, preprocessing retry, blur/IQA signal, and Tesseract fallback. | Measure extraction accuracy on representative real-world images. |
| **2 — Validation** | Verified | Check digits, format/status handling, expiry checks, visa parsing, and non-MRZ routing. | Add issuing-authority/blacklist integration if available. |
| **3 — Tampering** | Verified for bundled samples | ELA preview, edge, wavelet, copy-move, EXIF, detector coverage, and inconclusive handling. ELA/wavelet are gated by structural corroboration to avoid genuine-document false positives. | Validate against a labelled tampered-document dataset. |
| **3.5A — Replay** | Advisory prototype | FFT/texture indicators are returned for screen/print replay analysis. | Train and validate the SVM before enabling an automatic replay verdict. |
| **3.5B — Optical liveness** | Implemented; synthetic tests passed | Video capture/upload, frame extraction, Farneback optical flow, specular motion, HSV colour shift, and explainable verdicts. | Validate with real passport tilt videos, printed-copy attacks, and phone-screen replays. |
| **4 — Face verification** | Functional fallback | Strict face crops, quality flags, VGG-Face comparison, audit embeddings, and cross-attempt deduplication. | Reinstall optional ArcFace weights; install PyTorch only if face anti-spoofing is required. Test with consented real match/mismatch pairs. |
| **5 — Risk scoring** | Verified | Explainable point breakdown and LOW/MEDIUM/HIGH verdict; skipped selfie adds no liveness penalty. | Calibrate thresholds against a labelled evaluation set. |
| **UI/API** | Verified | Drag/drop document and selfie upload, optional tilt-video recording/upload, results dashboard, audit endpoint, upload validation, and static frontend mounting. | Conduct browser usability testing on the target demo device. |

## Current operational decisions

- **VGG-Face is the active face model.** The app no longer attempts a broken ArcFace download during startup.
- **Face liveness is not claimed as operational.** If PyTorch is unavailable, the output is explicitly marked inconclusive.
- **Heuristic replay signals do not add fraud points.** This prevents false-positive penalties on genuine documents. A trained SVM may add replay risk points.
- **A missing selfie is not treated as a face-liveness failure.** It is reported as skipped with zero face-risk points.

## How to reproduce the verified checks

```powershell
cd C:\SPACE\HACKATHONS\SIH\doc-screening-prototype\backend
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Then open http://127.0.0.1:8000/ and upload `backend/vendor/mrz_scanner/data/passport_uk.jpg`.

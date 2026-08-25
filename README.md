# AI-Based Fake Identity & Document Screening Prototype

A multi-modal border document screening and identity verification prototype. This system integrates open-source optical character recognition (OCR), image forensics, and facial biometrics to compute a composite risk score for travel and identity documents.

---

## 🏗 Architecture & Modules

The system acts as glue code orchestrating three open-source repositories and a custom risk-scoring engine:

| Module | Purpose | Source Repo / Dependency | License |
|---|---|---|---|
| **Module 1 & 2** | MRZ Extraction & Checksum Validation | [fastmrz](https://github.com/sivakumar-mahalingam/fastmrz) | **AGPL-3.0** |
| **Module 3** | Document Tampering & Forensics | [DocAuth](https://github.com/trinity652/DocAuth) | **MIT** |
| **Module 4** | 1:1 Facial Biometric Verification | [deepface](https://github.com/serengil/deepface) | **MIT** |
| **Module 5** | Rule-based Composite Risk Scoring | *Custom implementation* (`backend/modules/risk_scoring.py`) | MIT |

---

## ⚡ How to Run Locally

### Prerequisites
1. **Python**: Python 3.11 – 3.13 (recommended: Python 3.13)
2. **Tesseract OCR**:
   - Windows: Install via Chocolatey (`choco install tesseract -y`) or from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).
   - Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`

### Setup & Launch

1. **Navigate to the prototype directory:**
   ```bash
   cd doc-screening-prototype/backend
   ```

2. **Activate the Virtual Environment:**
   - Windows (PowerShell):
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - Linux / macOS:
     ```bash
     source .venv/bin/activate
     ```

3. **Start the FastAPI Server & Frontend:**
   ```bash
   python -m uvicorn main:app --host 0.0.0.0 --port 8000
   ```

4. **Access the Web Interface & API Docs:**
   - **Web UI**: Open [http://localhost:8000/](http://localhost:8000/) in your browser.
   - **Swagger / OpenAPI Documentation**: Open [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 🛡️ Screening Pipeline & Risk Scoring

### 1. Document Extraction (MRZ / OCR)
- Reads Machine Readable Zones compliant with ICAO Doc 9303 (TD1, TD2, TD3, MRV-A, MRV-B).
- Validates mathematical check digits for document number, date of birth, expiration date, and optional data.

### 2. Forensic Tamper Detection
- **Error Level Analysis (ELA)**: Detects compression discrepancies from localized photo/text edits.
- **Edge Consistency**: Analyzes Canny, Sobel, Laplacian, and Prewitt gradient uniformity to flag spliced artifacts.
- **Wavelet Decomposition**: Scans high-frequency detail bands for texture irregularities.
- **Copy-Move Detection**: Employs ORB feature descriptors and RANSAC homography estimation to detect duplicated or cloned sections.

### 3. Biometric Face Verification
- Computes facial embeddings using deep convolutional networks (VGG-Face).
- Compares passport/ID portrait against live selfie with distance metrics and threshold gating.

### 4. Composite Risk Scoring Engine
- **0 – 24**: **LOW RISK** (Document verified, valid checksums, low tamper heuristics, biometric match confirmed).
- **25 – 59**: **MEDIUM RISK** (Minor flags such as expired validity, moderate edge anomaly, or facial verification discrepancy).
- **60 – 100**: **HIGH RISK** (Checksum failure, high tampering/cloning score, or severe biometric mismatch).

---

## ⚠️ Known Limitations & Scope

1. **Document Format Scope**:
   - `fastmrz` specifically parses ICAO-compliant MRZ formats (passports, visas, standard national IDs). Non-MRZ documents (certain driver licenses, permits) will not return structured MRZ fields.
2. **Forensics Sensitivity**:
   - `DocAuth` algorithms operate on visual and compression heuristics. Heavily compressed social-media scans or low-resolution camera captures may exhibit baseline noise variance.
3. **Biometric Input Quality**:
   - Face matching accuracy relies on illumination, pose, and resolution in the document portrait and live capture.
4. **License Notice**:
   - `fastmrz` is licensed under **AGPL-3.0**. While suitable for prototypes and internal screening tools, derivative works hosted as a public network service are subject to AGPL copyleft requirements.

---

## 📚 Attribution & Acknowledgements

- **FastMRZ**: [https://github.com/sivakumar-mahalingam/fastmrz](https://github.com/sivakumar-mahalingam/fastmrz) (AGPL-3.0)
- **DocAuth**: [https://github.com/trinity652/DocAuth](https://github.com/trinity652/DocAuth) (MIT)
- **DeepFace**: [https://github.com/serengil/deepface](https://github.com/serengil/deepface) (MIT)

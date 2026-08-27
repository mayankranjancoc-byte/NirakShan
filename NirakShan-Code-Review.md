# NirakShan — Code Review

**Repo:** [Project-Core](https://github.com/Project-Core) @ `c5e51fb` (shallow clone, `main`)
**Scope:** Modules 1–4 (+ Module 5 risk scoring and the orchestration layer, since module defects surface there)
**Method:** full read of `backend/modules/*`, `backend/main.py`, `backend/tests/*`, and the vendored primitives in `backend/vendor/image_forensics/src/*` and `backend/vendor/mrz_scanner/mrz_scanner/mrz_scanner.py`. Claims below are marked **[verified]** where confirmed against source, **[inferred]** where reasoned from code but not executed.

**Not run.** No Python environment was provisioned, so nothing here comes from executing the pipeline. Several findings (notably the false-positive rates in Module 3) are predicted from the mathematics of the scoring functions and should be confirmed by running the measurements in [Recommended validation](#recommended-validation).

---

## Summary

The integration work is competent — clean module boundaries, per-module exception isolation, graceful partial results, an audit table, and a test suite. The problems are not in the plumbing.

**Three findings dominate everything else:**

1. **Module 3's four forensic scores are global-average statistics, not localized-anomaly statistics.** ELA, edge, and wavelet all reduce to "average intensity of a normalized map over the whole page." Tampering is by definition *local*. As written, these scores are driven by document design, texture, and JPEG quality rather than by manipulation — so genuine documents score high and the `< 10 → Authentic` band is close to unreachable. **[inferred — high confidence, verify empirically]**
2. **The identity-deduplication path can flag two unrelated travellers as the same person**, via a permissive threshold (cosine similarity `> 0.40`) combined with an exclusion key (`passport_number`) that is `None` whenever MRZ extraction fails. **[verified]** This is the most damaging false positive the system can produce.
3. **No authentication anywhere, and `GET /audit-log` returns every prior screening record to any unauthenticated caller** on a service bound to `0.0.0.0`. **[verified]**

And the structural gap: **there is no chip/NFC layer.** For a compliant ePassport the authoritative anti-substitution check is comparing the printed portrait against the chip's signed DG2 image. Without it, the photo-substitution attack the problem statement names first — re-laminated booklet, impostor's own photo — passes cleanly: MRZ checksums are untouched, the printed face matches the live face, and Module 3's heuristics are not reliable enough to catch it. See [Gap A](#gap-a--no-chip-layer).

**Credit where due, verified in source:** `mrz_scanner` sets `status = "SUCCESS"` only if no individual check digit failed (`mrz_scanner.py:357-359`), so Module 1's `checksum_valid = True` on SUCCESS is **correct** — I expected this to be conflated and it is not. Per-module `try/except` isolation in `main.py` is genuinely well done. The missing-selfie and non-MRZ penalty fixes recorded in `PROTOTYPE_STATUS.md` §9 are real and correctly implemented.

**Counts:** 9 critical · 14 high · 11 medium/low.

---

# Module 1 & 2 — OCR / MRZ Extraction & Validation

`backend/modules/ocr_extraction.py`

### 🔴 C1 — `apply_ocr_disambiguation` can silently corrupt the holder's name and still report `VALID`

`ocr_extraction.py:141-170`. The function tries global character substitutions across the **entire MRZ text**:

```python
test_strings.append(raw_mrz_text.replace('O', '0'))
test_strings.append(raw_mrz_text.replace('1', 'I'))
```

Two compounding problems. **[verified]**

- MRZ check digits cover only the numeric fields (document number, DOB, expiry, optional data, and the composite). **They do not cover the name field.** In TD3 the name is on line 1; the composite check digit is computed over line 2 only.
- A global `replace('O','0')` rewrites `JOHNSON` → `J0HNS0N` on line 1. That corruption is invisible to every checksum, so if the substitution happens to fix a genuine OCR error in the *document number* on line 2, the function returns `status: SUCCESS` — and the pipeline reports `checksum_valid: True`, `status: VALID` with a mangled name.

So a traveller can be cleared with a name that does not match their document. The reverse also holds: `replace('1','I')` and `replace('5','S')` corrupt dates and document numbers, making valid documents fail.

**Fix.** Constrain the search to the checksum-covered numeric fields and do a bounded per-character candidate search rather than a global replace:

```python
CONFUSIONS = {'O':'0','0':'O','I':'1','1':'I','B':'8','8':'B','Z':'2','2':'Z','S':'5','5':'S','Q':'0','D':'0'}
NUMERIC_SPANS = {  # 0-indexed, TD3 line 2
    'doc_number': (0, 9), 'dob': (13, 19), 'expiry': (21, 27), 'optional': (28, 42),
}

def disambiguate(mrz_lines, fast_mrz, max_edits=2):
    """Try confusion substitutions ONLY inside checksum-covered numeric spans."""
    line1, line2 = mrz_lines[0], mrz_lines[1]
    positions = [i for _, (s, e) in NUMERIC_SPANS.items() for i in range(s, min(e, len(line2)))]
    for n_edits in range(1, max_edits + 1):
        for combo in itertools.combinations(positions, n_edits):
            for repls in itertools.product(*[[CONFUSIONS.get(line2[i])] for i in combo]):
                if any(r is None for r in repls):
                    continue
                chars = list(line2)
                for idx, r in zip(combo, repls):
                    chars[idx] = r
                candidate = line1 + "\n" + "".join(chars)   # line1 NEVER modified
                res = fast_mrz.get_details(candidate, input_type="text", include_checkdigit=True)
                if res and res.get("status") == "SUCCESS":
                    res["is_corrected_by_disambiguation"] = True
                    res["corrections_applied"] = list(zip(combo, repls))
                    return res
    return None
```

Also surface `is_corrected_by_disambiguation` to the officer. A document that only validates *after* automated correction is a weaker result than one that validates as-read, and the UI currently does not distinguish them.

### 🔴 C2 — `parse_visa_fields` reads the wrong MRZ offsets and reports fabricated fields

`ocr_extraction.py:47-84`. **[verified against ICAO 9303 Part 7 MRV-A layout]**

MRV-A line 2 is: positions 1-9 document number, 10 check digit, 11-13 nationality, 14-19 DOB, 20 check digit, 21 sex, 22-27 expiry, 28 check digit, 29-44 optional data. Zero-indexed: `[0:9]` docnum, `[9]` cd, `[10:13]` nat, `[13:19]` dob, `[19]` cd, `[20]` sex, `[21:27]` expiry, `[27]` cd.

The code reads:

```python
duration_str = line2[19:22]      # = DOB check digit + sex + first char of expiry
entries_char = line2[22]         # = second char of expiry
```

Both are garbage. Worse, **duration of stay and number of entries are not MRZ fields at all** — they appear only in the visual zone. So `entries_allowed` is populated from a digit of the expiry date: since expiry chars are numeric, the `entries_char == 'M'` branch never fires and the `isdigit()` branch reports `"SINGLE"` when the expiry's second digit happens to be `1` and `"MULTIPLE"` otherwise. The system emits a confident, wrong value on essentially every visa.

**Fix.** Delete the `stay_duration_days` / `entries_allowed` / `entry_type` fields, or move them to a VIZ-OCR path with an explicit `source: "visual_zone"` marker and a confidence value. Keep only what the MRZ actually contains:

```python
def parse_visa_fields(mrz_lines: list) -> dict:
    """MRV-A/B: only fields genuinely present in the MRZ. Duration of stay and
    entry count are VIZ-only and must not be inferred from line 2."""
    if not mrz_lines or len(mrz_lines) < 2:
        return {}
    line1 = mrz_lines[0].replace("\r", "")
    return {
        "document_code": line1[0:1],                      # 'V'
        "visa_type": line1[1:2].replace("<", "") or None, # subtype
        "issuing_state": line1[2:5].replace("<", "") or None,
    }
```

Then add the check that *does* matter and is currently missing: **visa validity must fall inside passport validity**, and stay duration (once read from the VIZ) must be consistent with the visa type.

### 🔴 C3 — AGPL-3.0 dependency vendored into a repo with no root licence

**[verified]** `backend/vendor/mrz_scanner/` is a full copy of an AGPL-3.0 project, imported directly by `ocr_extraction.py` and served over HTTP by FastAPI. There is **no `LICENSE` file at the repository root** (confirmed: no match for `LICENSE*`), while `PROTOTYPE_STATUS.md` §3 labels Module 5 "MIT".

AGPL §13 extends copyleft to network use: offering this as a hosted service obliges you to offer the complete corresponding source of the whole work under AGPL. The README's "License Notice" acknowledges the risk but the repository state does not resolve it.

For an SIH project targeting government deployment this is a genuine adoption blocker, not a formality — a ministry evaluator or procurement reviewer will raise it, and it sits in the *core* MRZ path.

**Fix.** Pick one, now rather than later:
1. Add a root `LICENSE` that is AGPL-3.0-compatible and accept the copyleft obligation. Simplest, but constrains downstream deployment.
2. Replace `mrz_scanner` with a permissively licensed MRZ path. The MRZ parse and the 7-3-1 check-digit algorithm are ~150 lines and fully specified in ICAO 9303 — writing them yourself removes the dependency entirely and is a *net gain* for the project's technical story, because check-digit validation is deterministic logic you should own and be able to explain.
3. Isolate `mrz_scanner` behind a process boundary and document the separation. Legally murkier; I would not rely on it.

**Recommendation: option 2.** It converts a compliance liability into a demonstrable, self-owned deterministic layer.

### 🟠 H1 — Hardcoded Windows Tesseract path is passed unconditionally on all platforms

`ocr_extraction.py:26-45`. **[verified]**

```python
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
...
_mrz_reader = MRZScanner(tesseract_path=TESSERACT_PATH, tessdata_path=TESSDATA_DIR)
```

The `os.path.exists` guard protects `pytesseract`, but `TESSERACT_PATH` is then handed to `MRZScanner` **unguarded**. On macOS and Linux that is a non-existent path, despite the README documenting `brew install tesseract` and `apt-get install tesseract-ocr`. `main.py:280-283` has the same Windows-only `PATH` manipulation using `;` as separator.

**Fix.**

```python
import shutil

def _find_tesseract() -> str | None:
    found = shutil.which("tesseract")
    if found:
        return found
    for candidate in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        "/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract", "/usr/bin/tesseract",
    ):
        if os.path.exists(candidate):
            return candidate
    return None

TESSERACT_PATH = _find_tesseract()
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def get_mrz_reader():
    global _mrz_reader
    if _mrz_reader is None:
        if not TESSERACT_PATH:
            raise RuntimeError("Tesseract not found; install it or set TESSERACT_CMD")
        _mrz_reader = MRZScanner(tesseract_path=TESSERACT_PATH, tessdata_path=TESSDATA_DIR)
    return _mrz_reader
```

### 🟠 H2 — Expiry validation fails open; no date-logic validation at all

`risk_scoring.py:97-111`. **[verified]**

```python
try:
    expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
    ...
except (ValueError, TypeError):
    pass          # <-- silent
```

If the date format differs at all, the expiry check is skipped entirely: no penalty, no flag, and **no entry in `score_breakdown`** — so the officer sees no indication that expiry was never evaluated. In a fraud system the default must be "could not verify," not "verified fine."

Separately, the problem statement's date-logic requirement is unimplemented. There is no check that `DOB < issue_date < expiry_date`, no sanity bound on DOB (a parsed DOB in the future should be a hard flag — see M6), and no visa-inside-passport-validity check.

**Fix.**

```python
def _parse_iso(value):
    for fmt in ("%Y-%m-%d", "%y%m%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except (ValueError, TypeError):
            continue
    return None

expiry_date = _parse_iso(expiry_str)
if expiry_str and expiry_date is None:
    flags.append("EXPIRY_UNPARSEABLE: Expiry present but could not be interpreted")
    score_breakdown_list.append({"component": "Document Expiry", "points_added": 5.0,
                                 "max_points": 15.0, "reason": "Expiry date unparseable"})
    ocr_score += 5.0
elif expiry_date:
    ...  # existing comparison

# date-logic consistency
dob = _parse_iso(ocr_result.get("date_of_birth", ""))
if dob and dob > date.today():
    flags.append("DOB_IN_FUTURE: Date of birth is in the future — impossible")
    ocr_score += 10.0
if dob and expiry_date and dob >= expiry_date:
    flags.append("DATE_LOGIC_INVALID: DOB is not before expiry date")
    ocr_score += 10.0
```

### 🟠 H3 — No VIZ ↔ MRZ cross-check

**[verified — absent]** Tesseract is invoked for non-MRZ documents (`ocr_extraction.py:196-198`) and for unknown types (`:283`), but **never for MRZ documents**, and its output is never compared against the parsed MRZ fields.

This is the largest cheap win available in the codebase. Forgers typically alter the printed visual zone and miss the MRZ, or alter both inconsistently. Comparing OCR'd VIZ name / DOB / document number / expiry against the MRZ equivalents is deterministic, needs no training data, is fully explainable, and catches exactly the "altered DOB" and "text manipulation" cases the problem statement names.

**Fix.** Add a module between 2 and 3:

```python
def cross_check_viz_mrz(image_path: str, mrz_fields: dict) -> dict:
    """Compare printed visual-zone OCR against parsed MRZ fields.
    Deterministic; no model required. Returns per-field agreement."""
    viz_text = pytesseract.image_to_string(Image.open(image_path)).upper()
    viz_norm = re.sub(r"[^A-Z0-9]", "", viz_text)
    findings, mismatches = {}, 0
    checks = {
        "document_number": mrz_fields.get("document_number"),
        "surname":         mrz_fields.get("surname"),
        "given_name":      mrz_fields.get("given_name"),
    }
    for field, mrz_value in checks.items():
        if not mrz_value:
            findings[field] = "MRZ_VALUE_ABSENT"
            continue
        needle = re.sub(r"[^A-Z0-9]", "", str(mrz_value).upper())
        if needle and needle in viz_norm:
            findings[field] = "AGREES"
        else:
            findings[field] = "NOT_FOUND_IN_VIZ"   # weaker than MISMATCH: OCR may have failed
            mismatches += 1
    return {"findings": findings, "fields_not_corroborated": mismatches,
            "viz_ocr_confidence": "low" if len(viz_norm) < 40 else "normal"}
```

Score it carefully: `NOT_FOUND_IN_VIZ` is **not** proof of mismatch, because VIZ OCR fails often. Treat a single non-corroboration as a weak signal and *multiple* non-corroborations on an otherwise-readable page as a strong one. Do not let a Tesseract failure manufacture a fraud flag.

### 🟡 M1 — Temp-file leak on the success path

`ocr_extraction.py:243-263`. **[verified]** Preprocessing variants are written to `tempfile.gettempdir()` and `os.remove` is only reached when *both* the variant parse **and** its disambiguation fail. Every `break` on success leaks the file, and remaining unprocessed variants are never cleaned.

Names are also predictable (`v1_clahe_<basename>`), which in a shared `/tmp` is a collision and symlink-attack surface.

**Fix.** Use `tempfile.mkdtemp()` and a `try/finally` with `shutil.rmtree`:

```python
variant_dir = tempfile.mkdtemp(prefix="mrz_variants_")
try:
    variants = generate_preprocessing_variants(image_path, out_dir=variant_dir)
    for var_path in variants:
        ...
finally:
    shutil.rmtree(variant_dir, ignore_errors=True)
```

### 🟡 M2 — `calculate_image_quality` fails open, and measures the wrong region

`ocr_extraction.py:87-107`. **[verified]** On any exception or unreadable image it returns `{"is_blurry": False, "laplacian_var": 1000}` — i.e. "sharp." Fail-open in a fraud pipeline is the wrong default.

It also measures Laplacian variance over the **whole page**. MRZ readability depends on the MRZ band; a sharp page with a blurry MRZ strip reports as sharp. The threshold `50` is an unsourced heuristic with no calibration against the sample set.

**Fix.** Return `None` on failure and treat it as unknown; compute variance on the MRZ ROI once `mrz_scanner` has located it; calibrate the threshold against the eight images in `backend/vendor/mrz_scanner/data/`.

### 🟡 M3 — The `UNREADABLE` state is computed, then rendered meaningless

`ocr_extraction.py:322` sets `mrz_status = "UNREADABLE" if iqa_metrics["is_blurry"] else "INVALID"`, with `checksum_valid = False`.

In `risk_scoring.py:88` the branch is `elif effective_mrz_status == "INVALID" or checksum_valid is False:` — so `UNREADABLE` **does** fall through to the same 25-point penalty via the `checksum_valid is False` clause. **[verified — I initially suspected a scoring hole here; there is none.]**

The actual defect is semantic: the officer-facing flag reads `MRZ_CHECKSUM_INVALID: Document MRZ check digits failed validation` for a document that was merely too blurry to read. That is an accusation of forgery where the correct statement is "insufficient image quality — recapture." `PROTOTYPE_STATUS.md` §4 advertises the IQA/`UNREADABLE` distinction as a working feature; it has no observable effect.

**Fix.** Give `UNREADABLE` its own branch, a distinct flag, and a lower penalty, since it is a capture problem rather than a fraud indicator:

```python
elif effective_mrz_status == "UNREADABLE":
    ocr_score += 8.0
    flags.append("MRZ_UNREADABLE: Image quality too low to validate check digits — recapture required")
    score_breakdown_list.append({"component": "MRZ Checksum", "points_added": 8.0,
        "max_points": 25.0, "reason": "Image too blurry to read MRZ; not a forgery indicator"})
elif effective_mrz_status == "INVALID" or checksum_valid is False:
    ...
```

Ordering matters — the `UNREADABLE` branch must precede the `checksum_valid is False` clause.

### 🟡 M4 — Visa `stay_duration_days` is parsed but never scored

Even setting aside C2, nothing in `risk_scoring.py` reads `visa_fields`. The problem statement asks for stay-duration and visa-validity checks; the plumbing exists and terminates in nothing.

### 🟡 M5 — Inherited: two-digit MRZ years use Python's fixed 1969 pivot

`mrz_scanner.py:145`: `datetime.strptime(input_date, "%y%m%d")`. **[verified]** Python's `%y` maps `00-68 → 2000-2068` and `69-99 → 1969-1999`.

ICAO 9303 does not encode a century in MRZ dates; correct implementations use a sliding window per field (DOB must be in the past; expiry within roughly ±15 years). With a fixed pivot, **a traveller born in 1960 (`600101`) is parsed as 2060** — a birth date a century in the future.

Today this has no scoring effect, because `risk_scoring.py` never validates DOB (see H2). But the wrong DOB *is displayed to the officer*, so anyone born before 1969 shows a birth date that disagrees with the printed page. If you adopt the H2 `DOB_IN_FUTURE` check, this bug starts flagging every such traveller as fraudulent — so **fix M5 before shipping H2.**

**Fix** (in your own parser, per C3 option 2):

```python
def _mrz_date(yymmdd: str, kind: str) -> date | None:
    yy, mm, dd = int(yymmdd[0:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
    today = date.today()
    for century in (1900, 2000):
        try:
            candidate = date(century + yy, mm, dd)
        except ValueError:
            continue
        if kind == "dob" and candidate <= today:
            return candidate
        if kind == "expiry" and today - timedelta(days=365 * 15) <= candidate:
            return candidate
    return None
```

---

# Module 3 — Tampering Detection

`backend/modules/tampering_detection.py` + `backend/vendor/image_forensics/src/analysis/*`

This module has the deepest problems. All four forensic scores share one root error: **they measure a global average where the phenomenon is local.**

### 🔴 C4 — Copy-move detection will fire on genuine documents

`vendor/image_forensics/src/copy_move/detector.py:44-95`. **[verified in source; false-positive rate inferred]**

Three independent defects compound:

**(a) The claimed ratio test does not exist.**

```python
raw_matches = bf.knnMatch(des, des, k=3)
for m_list in raw_matches:
    for m in m_list[1:]:  # skip self-match
        if np.linalg.norm(pt1 - pt2) > MIN_SPATIAL_DIST:
            good_matches.append(m)
            break
```

The comment says "ratio test" but no ratio is computed. The first neighbour past the self-match that is >20px away is accepted regardless of descriptor distance. Weak matches count as strongly as good ones.

**(b) The score is a ratio, which inverts the intended meaning.**

```python
score = min(1.0, inlier_count / max(1, len(good_matches)))
```

10 matches with 10 inliers → `1.0` ("Forged"). 500 matches with 100 inliers → `0.2` ("Suspicious"). A document with *few* repeated features scores higher than one with many. What matters for copy-move is the absolute extent of duplicated area, not the inlier fraction.

**(c) The fatal one — identity homography on self-matched descriptors.** `cv2.findHomography(src_pts, dst_pts, RANSAC)` is fitted on ORB matches from an image against **itself**. Every ID document is full of legitimately repeated structure: guilloche security patterns, repeated glyphs, border rules, and above all the MRZ band's long runs of `<` filler characters. These self-match densely and fit a clean translational homography, producing a high inlier ratio on a perfectly genuine passport.

Combined with (b), a genuine passport with a modest number of repeated features can score near `1.0` → `verdict: "Forged"` → 25% weight → `TAMPERING_DETECTED` in the risk engine.

**Fix.** Score by duplicated *area* under multiple local transforms, exclude the MRZ band, and add the missing ratio test:

```python
def _orb_ransac(gray, nfeatures=5000, min_match_count=10, mrz_roi=None):
    if mrz_roi:                       # exclude the MRZ band: repeated '<' fillers self-match
        x, y, w, h = mrz_roi
        gray = gray.copy()
        gray[y:y+h, x:x+w] = 0
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp, des = orb.detectAndCompute(gray, None)
    if des is None or len(kp) < 2:
        return {"score": 0.0, "mask": np.zeros_like(gray), "method": "orb_ransac"}

    diag = np.hypot(*gray.shape)
    min_dist = max(20, 0.02 * diag)              # scale-relative, not absolute
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    good = []
    for m_list in bf.knnMatch(des, des, k=3):
        cands = [m for m in m_list[1:]
                 if np.linalg.norm(np.array(kp[m.queryIdx].pt) - np.array(kp[m.trainIdx].pt)) > min_dist]
        if len(cands) >= 2 and cands[0].distance < 0.75 * cands[1].distance:   # real ratio test
            good.append(cands[0])

    # Cluster by offset vector; a genuine repeated texture yields many small
    # inconsistent clusters, a copy-move yields one large coherent cluster.
    if len(good) < min_match_count:
        return {"score": 0.0, "mask": np.zeros_like(gray), "method": "orb_ransac"}
    offsets = np.array([np.subtract(kp[m.trainIdx].pt, kp[m.queryIdx].pt) for m in good])
    labels = DBSCAN(eps=max(5, 0.01 * diag), min_samples=min_match_count).fit_predict(offsets)
    mask = np.zeros(gray.shape, np.uint8)
    for lbl in set(labels) - {-1}:
        for m in (m for m, l in zip(good, labels) if l == lbl):
            cv2.circle(mask, tuple(map(int, kp[m.queryIdx].pt)), 8, 255, -1)
            cv2.circle(mask, tuple(map(int, kp[m.trainIdx].pt)), 8, 255, -1)
    # score = fraction of page area implicated, capped
    score = min(1.0, (mask > 0).sum() / mask.size / 0.05)
    return {"score": round(float(score), 4), "mask": mask, "method": "orb_ransac_clustered"}
```

### 🔴 C5 — Documented tamper weights contradict the implemented weights

**[verified]** `tampering_detection.py:253-259` implements:

```python
weights = {"ela": 0.25, "edge_detection": 0.15, "wavelet": 0.15, "copy_move": 0.25, "exif_analysis": 0.20}
```

`PROTOTYPE_STATUS.md` §6 documents: *"ELA (30%), Edge Variance (15%), Wavelet Energy (15%), and Copy-Move (40%)"* — different numbers, and **EXIF omitted entirely** despite carrying 20% in code.

For a system whose selling point is a transparent, explainable rule engine, a published formula that does not match the code is a credibility problem, and it is exactly the kind of discrepancy a judge finds by reading two files. **Fix:** single source of truth — define weights in one module, have the docs generated from or explicitly reference it, and add a test asserting `sum(weights.values()) == 1.0`.

### 🟠 H4 — `ela_score` measures mean brightness, not manipulation

`vendor/image_forensics/src/analysis/ela.py:66-76`. **[verified]**

```python
max_possible = 255.0 * arr.shape[0] * arr.shape[1] * arr.shape[2]
return float(arr.sum() / max_possible)
```

This is the normalized mean ELA intensity across the whole image. The ELA forgery signal is a **localized discontinuity** — a region whose error level differs from its surroundings. A global mean discards exactly that information.

Consequences: a busy, high-contrast, high-detail document yields a high score with no tampering; a smooth forged patch *lowers* the mean. Scores are dominated by document design and source JPEG quality, and are not comparable between two different genuine documents.

Compounding, in `generate_ela`: `quality=95, scale=15` means any per-pixel difference ≥ 17 saturates at 255, compressing dynamic range before the sum is taken. And there is **no format gate** — for a PNG source with no prior JPEG history, the difference map is uniform quantization noise and the score is meaningless.

**Fix.** Score block-level outliers, not the global mean:

```python
def ela_score_local(ela_image, block=16):
    """Localized ELA: how far do the most anomalous blocks deviate from the page median?
    Returns (score, per_block_map) so the officer can be shown WHERE."""
    gray = np.array(ela_image.convert("L"), np.float32)
    h, w = gray.shape
    bh, bw = h // block, w // block
    if bh == 0 or bw == 0:
        return 0.0, None
    blocks = gray[:bh * block, :bw * block].reshape(bh, block, bw, block).mean(axis=(1, 3))
    med = np.median(blocks)
    mad = np.median(np.abs(blocks - med)) + 1e-6
    z = np.abs(blocks - med) / (1.4826 * mad)          # robust z-score
    frac_anomalous = float((z > 3.5).mean())            # fraction of clearly deviant blocks
    return min(1.0, frac_anomalous / 0.02), z           # 2% of blocks deviant => score 1.0
```

And gate on format:

```python
if Image.open(path).format not in ("JPEG", "MPO"):
    breakdown["ela"] = {"score": None, "status": "NOT_APPLICABLE",
                        "reason": "ELA requires a JPEG source with compression history"}
```

`None` must then be excluded from the weighted combination, not coerced to 0 — see H6.

### 🟠 H5 — `_edge_anomaly_score` is a document-layout detector

`tampering_detection.py:38-70`. **[verified]**

It computes the coefficient of variation of edge density across a 4×4 grid, averaged over five detector outputs. Every genuine ID document is *designed* to be spatially non-uniform: a portrait region, a dense MRZ band, blank margins, a signature strip. **High CV is the normal state of a genuine document**, so this fires on layout, not splicing.

Three further defects:
- It averages `canny` (binary 0/255) with `sobel`/`laplacian`/`prewitt_x`/`prewitt_y` (each min-max normalized to its own range). These are not commensurable, and min-max normalization makes each value relative to that image's own maximum, so densities are not comparable across images.
- `prewitt_x` and `prewitt_y` are counted as two of five detectors, giving Prewitt double weight.
- `grid_h, grid_w = h // 4, w // 4` → if either dimension is < 4, the divisor is 0, blocks are empty, and `np.mean` of an empty slice yields `nan` with a RuntimeWarning. `nan` then propagates silently into the combined score.

**Fix.** Compare edge *statistics across a boundary* rather than variance across a layout grid — and restrict analysis to detected field regions:

```python
def _edge_anomaly_score(edge_results, field_rois):
    """Splicing leaves abrupt edge-density steps at region borders.
    Measure the discontinuity ACROSS each field boundary, not variance across the page."""
    canny = edge_results.get("canny")
    if canny is None or not field_rois:
        return None                       # unknown, not zero
    steps = []
    for (x, y, w, h) in field_rois:
        pad = 6
        inner = canny[y:y+h, x:x+w]
        outer = canny[max(0, y-pad):y+h+pad, max(0, x-pad):x+w+pad]
        if inner.size == 0 or outer.size <= inner.size:
            continue
        d_in = inner.mean() / 255.0
        d_ring = (outer.sum() - inner.sum()) / max(1, outer.size - inner.size) / 255.0
        steps.append(abs(d_in - d_ring) / (d_in + d_ring + 1e-6))
    return float(np.clip(max(steps), 0, 1)) if steps else None
```

### 🟠 H6 — A crashed detector contributes `0`, biasing every verdict toward "Authentic"

`tampering_detection.py:261-266`. **[verified]**

```python
except Exception as e:
    breakdown["copy_move"] = {"score": 0, "error": str(e)}
...
score = breakdown.get(key, {}).get("score", 0)
combined += score * weight
```

If copy-move throws, its 25% weight silently contributes zero and the remaining weights are **not renormalized** — the maximum achievable score drops to 75 while the `< 10 → Authentic` / `< 55 → Suspicious` thresholds stay put. A pipeline failure therefore reads as evidence of authenticity. In a fraud system that is the wrong direction to fail.

**Fix.** Renormalize over detectors that actually produced a score, and mark degradation explicitly:

```python
available = {k: w for k, w in weights.items()
             if isinstance(breakdown.get(k, {}).get("score"), (int, float))}
if not available:
    return {"tamper_score": None, "verdict": "INCONCLUSIVE",
            "reason": "All forensic detectors failed", "breakdown": breakdown}

total_w = sum(available.values())
combined = sum(breakdown[k]["score"] * w for k, w in available.items()) / total_w
coverage = total_w / sum(weights.values())

result = {"tamper_score": round(min(combined, 100.0), 2),
          "detector_coverage": round(coverage, 2),
          "degraded": coverage < 1.0,
          "unavailable_detectors": sorted(set(weights) - set(available))}
if coverage < 0.6:
    result["verdict"] = "INCONCLUSIVE"     # too little evidence to assert anything
```

Then propagate `INCONCLUSIVE` into `risk_scoring` as an explicit "could not assess" state rather than 0 points.

### 🟠 H7 — EXIF carries 20% weight on trivially forged, routinely absent metadata

`tampering_detection.py:104-160`. **[verified]**

- `exif_stripped = True → score 0.5` for **any** JPEG without EXIF. Every image that has passed through a messaging app, been re-saved by an editor, produced by a flatbed scanner, or captured as a screenshot has no EXIF. Legitimate scans therefore accrue `0.5 × 100 × 0.20 = 10` tamper points → 3 risk points, as pure noise.
- `software_flagged → score 1.0`, the maximum, if EXIF `Software` contains `photoshop`/`gimp`/`canva`/`adobe`/`paint`. This is 20 of 100 tamper points → 6 risk points, driven by a string an attacker removes with `exiftool -all=` in one command. Meanwhile a *genuine* document legitimately cropped in Photoshop by a border operator is flagged at maximum.
- `"adobe"` already subsumes `"photoshop"`; `"paint"` substring-matches anything containing it.
- `PROTOTYPE_STATUS.md` claims EXIF flags "missing timestamps." **No timestamp logic exists in the code.**
- `if exif is None` is dead: modern Pillow's `getexif()` returns an empty `Exif` object, never `None`.

Metadata is the weakest possible evidence tier here, and it is weighted equal to ELA and copy-move.

**Fix.** Demote EXIF to a non-scoring context signal, or cap it at ~3% of the tamper score:

```python
weights = {"ela": 0.30, "edge_detection": 0.15, "wavelet": 0.10, "copy_move": 0.42, "exif_analysis": 0.03}
```

and drop `exif_stripped` from scoring entirely — report it as informational context (`"exif_absent": true, "scored": false`), because absence of EXIF carries essentially no evidential weight for a scanned document. Retain `software_flagged` as an informational note, not a score contribution.

### 🟠 H8 — Practical consequence: "Authentic" is close to unreachable

**[inferred from H4/H5/H7 + C4 — verify empirically]**

Stacking the four metric defects: edge CV is high on any real document (H5), ELA mean is driven by document texture (H4), wavelet is min-max-normalized so its mean is roughly texture-determined (H9), a stripped-EXIF scan adds a flat 10 points (H7), and copy-move can spike on repeated typography (C4).

The `combined < 10 → "Authentic"` band is therefore very unlikely to be reached by a genuine document, and `risk_scoring.py:127` flags `TAMPERING_SUSPICIOUS` at `tamper_raw >= 10`. **The predicted outcome is that nearly every genuine document receives a tampering flag** — which will be plainly visible the moment you demo across several documents, and is the fastest way to lose a judge's confidence in the whole pipeline.

Nothing in the repo measures this: there is no ROC curve, no false-positive rate, no threshold justification, and no labelled evaluation anywhere. See [Recommended validation](#recommended-validation) — this is the single highest-value thing to run before any demo.

### 🟡 M6 — `_wavelet_anomaly_score` measures the mean of a min-max-normalized map

`tampering_detection.py:73-87` with `vendor/.../wavelet.py:76-79`. **[verified]**

`decompose` returns `reconstructed = cv2.normalize(np.abs(...), 0, 255, NORM_MINMAX)`, which forces the maximum to 255 for **every** image. The consumer then takes `np.mean(...)/255 * 2`.

So the score is a pure function of the *shape* of the high-frequency distribution — driven by resolution, sharpness, and paper texture, not tampering. It is also non-comparable across images by construction, since each is normalized to its own extremes. The docstring describes a localized claim ("detail energy in manipulated regions differs from the background") and the implementation computes a global mean — the same category error as H4.

**Fix.** Have `decompose` also return the un-normalized `reconstructed_raw`, and score block-level deviation of detail energy against the page's own robust baseline (same pattern as the H4 fix).

### 🟡 M7 — Full-resolution ELA heatmap base64-encoded into every response

`tampering_detection.py:184-190`. **[verified]** The ELA PNG is base64-encoded at source resolution and returned in the JSON on every request. For a 4000×3000 scan that is tens of megabytes of base64 per response, with no downscaling and no opt-out.

**Fix.** Downscale to a bounded preview and gate behind a query parameter:

```python
preview = ela_image.copy()
preview.thumbnail((640, 640), Image.LANCZOS)
buffer = io.BytesIO()
preview.save(buffer, format="WEBP", quality=80)
```

### 🟡 M8 — Missing forensic signals, and no localization

**[verified — absent]** The signals that matter most for document forgery are not implemented:

- **Per-field font / glyph and baseline geometry** — the single highest-value signal for altered text, and the one most likely to catch a modified DOB. Substituted characters rarely match stroke weight, spacing, and baseline exactly.
- **JPEG ghost** (multi-quality sweep) — a much stronger localized recompression signal than single-quality ELA.
- **CFA / demosaicing inconsistency** and **noise-residual (PRNU/SRM-style)** analysis.
- **Guilloche / security-pattern periodicity** via FFT — a designed, verifiable property of real documents.
- **Photo-region boundary and re-lamination artefacts** — the direct signature of photo substitution.
- **Recapture / presentation-attack detection** (screen moiré, print texture) — at a real counter the common attack is a photo of a photo, and nothing here detects it.

Equally important: **every output is a global scalar.** There is no per-region localization, so the officer is told *that* something is suspicious but never *where*. Localization is what makes a forensic verdict actionable and defensible; without it the module cannot support the explainability the problem statement asks for.

---

# Module 4 — Face Verification

`backend/modules/face_verification.py` + `backend/modules/audit_logger.py`

### 🔴 C6 — Identity dedup threshold is far too permissive for 1:N search

`audit_logger.py:120-155`. **[verified]**

```python
def find_similar_identity(embedding, exclude_passport, threshold: float = 0.40):
    sim = np.dot(emb_a, emb_b) / (norm_a * norm_b)
    if sim > threshold:      # cosine SIMILARITY > 0.40 => "same person"
```

Cosine similarity `> 0.40` is used as a same-person decision. Two problems.

First, **1:N gallery search requires a materially stricter threshold than 1:1 verification**, because the probability of at least one false match grows with gallery size. A threshold tuned (even correctly) for a single comparison is wrong for a scan across hundreds of rows.

Second, the threshold is **model-independent** while the code can produce embeddings from either ArcFace or VGG-Face (see C7). Their similarity distributions differ, so one constant cannot be right for both.

The consequence is false `MULTIPLE_IDENTITY_SUSPECTED` flags — accusing an innocent traveller of holding a second identity, which is the most damaging output this system can produce.

**Fix.** Model-specific thresholds, calibrated at a stated FMR against a labelled set, and return the *best* match rather than the first:

```python
# Calibrate these against a labelled gallery at a target false-match rate,
# then record the FMR you calibrated at. Do not ship uncalibrated constants.
DEDUP_THRESHOLDS = {"ArcFace": 0.68, "VGG-Face": 0.75}   # cosine similarity, 1:N

def find_similar_identity(embedding, exclude_session, model_name, gallery_fmr="1e-4"):
    threshold = DEDUP_THRESHOLDS.get(model_name)
    if threshold is None:
        return {"status": "NO_THRESHOLD_FOR_MODEL", "model": model_name}
    rows = _load_gallery(model_name)          # filter by model — see C7
    best = None
    for session, passport, emb in rows:
        if session == exclude_session:        # exclude by SESSION, not passport — see C8
            continue
        if len(emb) != len(embedding):        # never silently skip a shape mismatch
            logger.warning("embedding dim mismatch: %s vs %s", len(emb), len(embedding))
            continue
        sim = float(np.dot(embedding, emb) / (np.linalg.norm(embedding) * np.linalg.norm(emb)))
        if sim > threshold and (best is None or sim > best["similarity"]):
            best = {"matched_session": session, "matched_passport": passport,
                    "similarity": sim, "model": model_name, "calibrated_fmr": gallery_fmr}
    return best
```

### 🔴 C7 — Embeddings from different models are compared, and the failure is silently swallowed

`audit_logger.py:111-118` and `:130-153`. **[verified]**

`store_embedding` records `(session_id, passport_number, embedding_json, timestamp)` — **no model name.** But `face_verification.py:163-166` selects the model at runtime:

```python
model_name = "ArcFace" if has_arcface else "VGG-Face"
```

ArcFace embeddings are 512-dimensional; VGG-Face are 4096. Both land in the same table. `np.dot` on mismatched dimensions raises `ValueError`, which is caught by:

```python
except Exception:
    continue
```

So after any model switch — including the silent downgrade in C8 — dedup silently stops matching against every previously stored row, with no log line and no flag. The feature appears to work and does nothing.

**Fix.** Add `model_name` and `embedding_dim` columns, filter the gallery by model, and never swallow a dimension mismatch (see the C6 snippet). Existing rows have no model attribution and cannot be reliably back-filled — drop and rebuild the table.

```sql
CREATE TABLE IF NOT EXISTS identity_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    passport_number TEXT,
    model_name TEXT NOT NULL,
    embedding_dim INTEGER NOT NULL,
    embedding_json TEXT NOT NULL,
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_emb_model ON identity_embeddings(model_name);
CREATE INDEX IF NOT EXISTS idx_emb_session ON identity_embeddings(session_id);
```

### 🔴 C8 — Dedup excludes by `passport_number`, which is `None` whenever MRZ extraction fails

`main.py:224-225` → `face_verification.py:171-186` → `audit_logger.py:120`. **[verified]**

```python
passport_number = results.get("ocr", {}).get("document_number")   # None if MRZ failed
face_result = verify_face_match(doc_path, selfie_path, session_id=session_id,
                                passport_number=passport_number)
...
if session_id and passport_number:        # dedup only runs when both are truthy
    similar = find_similar_identity(embedding, exclude_passport=passport_number)
    store_embedding(session_id, passport_number, embedding)
```

Two distinct defects.

**(a) Wrong exclusion semantics.** Excluding rows with the *same passport number* is intended as "don't match me against myself," but it also excludes the **exact case the problem statement asks you to catch**: the same person presenting a second document. If the impostor reuses a document number, the match is suppressed. Exclusion should be by `session_id`.

**(b) `None` poisons the comparison.** When MRZ extraction fails, `document_number` is `None`. The `if session_id and passport_number` guard means dedup is skipped — so *no* embedding is stored and no check is run, meaning **every traveller whose MRZ is unreadable is silently exempt from duplicate-identity screening.** That is a trivially exploitable evasion: degrade the MRZ and the biometric dedup never runs.

If that guard were relaxed without fixing the exclusion key, the opposite failure appears: two different travellers both stored with `passport_number = NULL` would be compared against each other and could match.

**Fix.** Always store and always search, keyed on `session_id`; treat a missing document number as metadata, not a gate:

```python
try:
    similar = find_similar_identity(embedding, exclude_session=session_id, model_name=model_name)
    if similar and similar.get("matched_session"):
        identity_cross_check = {"duplicate_detected": True, **similar}
        flags.append("MULTIPLE_IDENTITY_SUSPECTED")
    store_embedding(session_id, passport_number, model_name, embedding)   # passport may be None
except Exception as error:
    logger.exception("identity dedup failed")
    flags.append(f"DEDUP_UNAVAILABLE: {error}")      # visible, not silent
```

### <a name="gap-a--no-chip-layer"></a>🔴 C9 — No chip / NFC layer: photo substitution is undetectable

**[verified — absent from the entire repo]**

This is the central capability gap, not a bug in existing code.

A compliant ICAO ePassport carries a contactless chip whose data groups are protected by a Document Security Object signed by the issuing state, chaining to that state's CSCA. **DG2 holds the holder's facial image, signed.** The authoritative test for photo substitution is therefore: compare the printed portrait against the chip's signed DG2 image. It is cryptographic, not statistical.

Trace the photo-substitution attack through the current pipeline:

| Layer | Result on a re-laminated, photo-substituted passport |
|---|---|
| Module 1/2 MRZ | ✅ Passes — MRZ was never touched, all check digits valid |
| Module 3 tampering | ⚠️ Unreliable — heuristics operate on a photograph of a physically altered booklet, and per H4/H5/H8 they cannot separate this from baseline |
| Module 4 face | ✅ **Matches** — the printed photo *is* the impostor's own face |
| Module 5 risk | 🟢 **LOW RISK** |

The system clears the attack it exists to stop. Adding DG2 comparison turns this into an immediate, explainable `REFER`: *"printed photograph does not match the issuing state's signed facial image."*

**Fix.** This is a new module, not a patch — and it should be prioritized above every remaining item in this review:

```python
# modules/chip_authentication.py  (Layer 0 — strongest evidence, run FIRST)
def read_and_verify_chip(mrz_lines: list[str]) -> dict:
    """
    BAC/PACE session from MRZ-derived key -> read DG1/DG2 -> Passive Authentication:
      1. recompute each data-group hash, compare against the SOD
      2. verify the SOD signature
      3. chain the Document Signer cert to the issuing CSCA
    Returns a verdict tier, never a bare boolean.
    """
    return {
        "chip_present": bool,
        "passive_auth": "PASS" | "FAIL" | "UNAVAILABLE",
        "dg_hashes_match": bool,
        "sod_signature_valid": bool,
        "csca_chain_valid": bool,
        "dg1_fields": {...},           # cross-check against MRZ and VIZ
        "dg2_face_image": bytes | None, # third face for comparison
        "evidence_tier": "CRYPTOGRAPHIC" | "NONE",
    }
```

Then make face verification three-way — live ↔ printed ↔ chip — and weight the chip result far above every heuristic in `risk_scoring`. Note that Passive Authentication needs the issuing CSCA certificates; production requires ICAO PKD access, and a local trust store is the right POC substitute **provided you state that dependency openly**.

You can legitimately develop and test this on team members' own passports with written consent.

### 🟠 H9 — Silent ArcFace → VGG-Face downgrade, while `/health` claims ArcFace is loaded

`face_verification.py:162-166` and `main.py:159-162`. **[verified]**

```python
arcface_weights = os.path.expanduser("~/.face_biometrics/weights/arcface_weights.h5")
has_arcface = os.path.exists(arcface_weights) and os.path.getsize(arcface_weights) > 100_000_000
model_name = "ArcFace" if has_arcface else "VGG-Face"
```

This hardcodes the filename, the `.h5` extension, the home-relative path (ignoring `DEEPFACE_HOME`), and a magic 100 MB size. Any deviation silently selects VGG-Face — a materially weaker model whose **distance thresholds have different meaning**, so the downstream `distance`/`threshold`/`confidence` values change semantics with no signal.

Meanwhile `/health` unconditionally reports:

```python
return {"status": "ok", ..., "models_loaded": ["MRZScanner", "ArcFace", "RetinaFace"]}
```

A health endpoint that always claims success is worse than none — it actively misleads. Note also that `preload_models` warms `detector_backend="retinaface"` while `DETECTION_BACKENDS` prefers `opencv`, so the warm-up may not warm the path actually used.

**Fix.** Resolve the model by attempting to build it, record which model was actually used in the response *and* the audit row, and have `/health` report real per-component state:

```python
def _resolve_face_model() -> str:
    for candidate in ("ArcFace", "VGG-Face"):
        try:
            FaceBiometrics.build_model(candidate)
            return candidate
        except Exception as error:
            logger.warning("face model %s unavailable: %s", candidate, error)
    raise RuntimeError("No face recognition model available")

@app.get("/health")
async def health_check():
    components = {}
    try:
        get_mrz_reader(); components["mrz_scanner"] = "ok"
    except Exception as e:
        components["mrz_scanner"] = f"unavailable: {e}"
    for model in ("ArcFace", "VGG-Face"):
        try:
            FaceBiometrics.build_model(model); components[model] = "ok"
        except Exception as e:
            components[model] = f"unavailable: {e}"
    degraded = any(v != "ok" for v in components.values())
    return JSONResponse(
        status_code=503 if components.get("mrz_scanner") != "ok" else 200,
        content={"status": "degraded" if degraded else "ok", "components": components},
    )
```

### 🟠 H10 — Liveness evaluates a different face than the one being matched, and fails open

`face_verification.py:99-113`. **[verified]**

```python
faces = FaceBiometrics.extract_faces(img_path=selfie_path, ..., anti_spoofing=True)
return faces[0].get("is_real"), None
```

`_extract_face_crop` selects `max(candidates, key=_face_area)` — the **largest** face. `_liveness_check` takes `faces[0]` — the **first** face in detection order. In any image containing more than one face these can differ, so liveness may be assessed on a bystander while the match is performed on the traveller.

It also re-runs full detection (doubling detector cost), and on failure returns `is_real = None`. In `risk_scoring.py:186-193`, `is_real is None` yields **0 points and no liveness entry in `score_breakdown` at all** — so if anti-spoofing fails to load, presentation attacks become entirely unpenalized and the officer-facing report shows no indication that liveness was never assessed.

**Fix.** Reuse the already-selected crop and its index, and make unavailability visible in the score:

```python
face, index = _select_primary_face(faces)       # same selection rule for both paths
liveness = face.get("is_real")
...
# risk_scoring.py
if is_real is None:
    face_score += 3.0
    flags.append("LIVENESS_NOT_ASSESSED: Anti-spoofing unavailable — presentation attack not ruled out")
    score_breakdown_list.append({"component": "Face Liveness", "points_added": 3.0,
        "max_points": 10.0, "reason": "Anti-spoofing unavailable; result inconclusive"})
```

There is also **no presentation-attack detection on the document image** — a printed or screen-displayed document passes unchallenged, which is the common attack at a physical counter.

### 🟠 H11 — `session_id` in `screening_log` never matches `identity_embeddings`

`main.py:186` vs `audit_logger.py:32-35`. **[verified]**

```python
# main.py
session_id = uuid.uuid4().hex
results["session_id"] = session_id
verify_face_match(..., session_id=session_id, ...)     # -> identity_embeddings

# audit_logger.py
def log_screening_result(result_dict) -> str:
    session_id = uuid.uuid4().hex        # generates a SECOND, different id
```

`log_screening_result` ignores `result_dict["session_id"]` and mints a new one, then `main.py:262` discards the returned value. So for any given screening the id in `screening_log` and the id in `identity_embeddings` are different, and **the audit trail cannot be joined to the biometric record.**

**Fix.** One line:

```python
session_id = result_dict.get("session_id") or uuid.uuid4().hex
```

### 🟡 M9 — `MIN_FACE_SIDE_PX = 40` is too small for reliable biometric comparison

`face_verification.py:16`. A 40×40 crop provides on the order of ~15 px inter-eye distance — far below what face recognition needs for a dependable embedding, and well below ISO/ICAO portrait guidance. The gate prevents the worst case (comparing a whole page) but still admits crops that will yield confident-looking, unreliable matches. Raise to ~80–112 px and emit an explicit `LOW_QUALITY_PORTRAIT` flag between the two bounds rather than proceeding silently.

### 🟡 M10 — `confidence` is an uncalibrated linear rescale presented as a percentage

`face_verification.py:196-199`. **[verified]**

```python
confidence = result.get("confidence", 0)
if not confidence and threshold > 0:
    confidence = max(0, (1 - distance / threshold)) * 100
```

FaceBiometrics's `verify` does not return a `confidence` key, so this fallback is effectively unconditional (and `if not confidence` would also trigger on a legitimate `0`). The result is a linear function of distance-to-boundary, displayed to an officer as `confidence: 87%` — an uncalibrated number wearing the clothes of a probability.

**Fix.** Either report the raw distance and threshold only, or calibrate genuine/impostor distance distributions on a labelled set and convert via a fitted logistic. Label it `calibrated: false` until you do.

### 🟡 M11 — `os.replace` across filesystems

`face_verification.py:80-88`. `tempfile` frequently sits on a different device than the destination; `os.replace` then raises `OSError: Invalid cross-device link`, which the `except` re-raises as a hard failure. Use `shutil.move`.

---

# Cross-cutting — API, security, audit, tests

### 🔴 C10 — No authentication; `/audit-log` exposes all prior screening records

`main.py:150-157`. **[verified]**

```python
@app.get("/audit-log")
async def audit_log(limit: int = 20):
    return get_recent_screenings(limit=limit)
```

No authentication, no authorization, no rate limiting, on a server bound to `0.0.0.0:8000`. Any caller on the network can retrieve every prior screening record — document types, MRZ statuses, risk scores, verdicts, and flags — and can submit unlimited screening requests.

For a system whose stated purpose is border screening with an audit trail, this is the most serious security defect in the repository. The problem statement also calls for officer/supervisor role separation, which does not exist in any form.

**Fix.** Minimum viable, before any networked demo:

```python
from fastapi import Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

bearer = HTTPBearer()
ROLES = {"officer": {"screen"}, "supervisor": {"screen", "audit"}}

async def require(scope: str):
    async def _dep(cred: HTTPAuthorizationCredentials = Security(bearer)):
        claims = verify_token(cred.credentials)          # JWT verify, exp, issuer
        if scope not in ROLES.get(claims.get("role"), set()):
            raise HTTPException(403, "insufficient role")
        return claims
    return _dep

@app.get("/audit-log")
async def audit_log(limit: int = 20, user=Depends(require("audit"))):
    log_access(user["sub"], "audit-log", limit)           # log who read the audit log
    return get_recent_screenings(limit=min(limit, 100))
```

Also bind to `127.0.0.1` by default and require an explicit flag to expose externally.

### 🔴 C11 — No upload validation: unbounded size, unchecked content

`main.py:97-104`. **[verified]**

```python
def _save_upload(upload: UploadFile) -> str:
    ext = os.path.splitext(upload.filename or "file.jpg")[1] or ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    with open(filepath, "wb") as f:
        shutil.copyfileobj(upload.file, f)
```

No size limit, no content-type check, no magic-byte verification, no decodability check, and no pixel-dimension bound. Consequences:

- **Unbounded upload → disk exhaustion / DoS.** `copyfileobj` streams whatever is sent.
- The extension is attacker-controlled and the bytes are unvalidated, then handed to `cv2.imread`, `PIL.Image.open`, and Tesseract — three native decoders receiving untrusted input with no gating.
- **`Image.MAX_IMAGE_PIXELS` is never set**, so a decompression-bomb PNG will be expanded by Pillow.

**Path traversal is *not* exploitable** — only the extension is taken from `filename` and the basename is a fresh `uuid4` — so that specific concern is already handled. **[verified]**

**Fix.**

```python
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
ALLOWED = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
Image.MAX_IMAGE_PIXELS = 80_000_000          # decompression-bomb guard

def _save_upload(upload: UploadFile) -> str:
    if upload.content_type not in ALLOWED:
        raise HTTPException(415, f"Unsupported type: {upload.content_type}")
    filepath = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ALLOWED[upload.content_type]}")
    written = 0
    with open(filepath, "wb") as f:
        while chunk := upload.file.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                f.close(); os.remove(filepath)
                raise HTTPException(413, "File exceeds 15 MB limit")
            f.write(chunk)
    try:                                      # verify it really is a decodable image
        with Image.open(filepath) as probe:
            probe.verify()
    except Exception:
        os.remove(filepath)
        raise HTTPException(400, "File is not a readable image")
    return filepath
```

### 🟠 H12 — Blocking CPU work inside `async def` serializes the whole server

`main.py:165-170`. **[verified]** `screen_document` is declared `async def` but performs entirely synchronous, CPU-heavy work — Tesseract, ONNX, TensorFlow, OpenCV, PyWavelets — directly in the coroutine. That blocks the event loop for the full duration of a screening, so concurrent requests queue behind each other and `/health` becomes unresponsive during any screening.

**Fix.** The one-character version: drop `async`, and FastAPI runs the handler in its threadpool.

```python
@app.post("/screen-document")
def screen_document(...):        # sync def -> executed in a threadpool
```

Then bound concurrency deliberately (TensorFlow and Tesseract are memory-hungry) and confirm `get_mrz_reader()`'s lazy global is guarded:

```python
_mrz_lock = threading.Lock()

def get_mrz_reader():
    global _mrz_reader
    if _mrz_reader is None:
        with _mrz_lock:
            if _mrz_reader is None:
                _mrz_reader = MRZScanner(...)
    return _mrz_reader
```

### 🟠 H13 — Uploads are deleted with no hash retained: no chain of custody

`main.py:274-280`. **[verified]** The `finally` block deletes both uploads, and `audit_logger.log_screening_result` records scores and flags but **no hash of the input image.**

So there is no way to prove which image produced a verdict, no way to re-run analysis on a disputed case, and no way to detect that the audit row was altered. The `screening_log` table has no hash chaining, so any row can be edited or deleted without trace.

For a border-screening system this undermines the audit-trail requirement entirely.

**Fix.** Hash before analysis, chain the log:

```python
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

results["document_sha256"] = _sha256(doc_path)
if selfie_path:
    results["selfie_sha256"] = _sha256(selfie_path)
```

```sql
ALTER TABLE screening_log ADD COLUMN document_sha256 TEXT;
ALTER TABLE screening_log ADD COLUMN selfie_sha256   TEXT;
ALTER TABLE screening_log ADD COLUMN prev_row_hash   TEXT;
ALTER TABLE screening_log ADD COLUMN row_hash        TEXT;
```

with `row_hash = sha256(prev_row_hash || canonical_json(row))`, plus an integrity-verification endpoint that walks the chain. Also add a documented retention policy — biometric embeddings are sensitive personal data under the DPDP Act, 2023, and the repo currently stores them indefinitely with no stated basis or expiry.

### 🟠 H14 — Modules 3 and 4 have no tests; the "invalid MRZ" test is a hand-written mock

**[verified]** `backend/tests/` contains `test_risk_scoring.py`, `test_ocr_and_non_mrz.py`, `test_audit_and_identity.py`. There is **no test for `tampering_detection.py` and none for `face_verification.py`** — the two modules carrying the deepest defects in this review.

Of the OCR tests, `test_case_2_invalid_passport_mrz` does not use a document at all:

```python
mock_invalid_ocr = {"status": "INVALID", "checksum_valid": False, ...}
risk = compute_risk_score(mock_invalid_ocr, self.mock_tamper, self.mock_face)
```

That asserts the risk engine's `if/else` branch, not the system's ability to detect a bad check digit. There is no test anywhere that takes a genuinely invalid MRZ image and confirms detection.

The tests do import and exercise real modules — no `unittest.mock` anywhere, which is good — but `PROTOTYPE_STATUS.md` presents "10 Automated Tests — 10/10 Passing" as evidence of correctness. That claim is much weaker than it appears: the passing tests are consistent with every critical finding in this review.

**Fix.** Highest-value tests to add, in order:

```python
# 1. Genuine documents must NOT be flagged as tampered  (guards H8 — run this first)
def test_genuine_documents_score_authentic(self):
    for name in ("passport_uk.jpg", "td1.jpg", "td2.jpg", "td3.jpg", "mrva.jpg", "mrvb.jpg"):
        with self.subTest(doc=name):
            result = analyze_tampering(os.path.join(SAMPLE_DIR, name))
            self.assertLess(result["tamper_score"], 10.0,
                            f"{name}: genuine document scored {result['tamper_score']} — false positive")

# 2. A real MRZ with a corrupted check digit must be caught end to end (guards H14)
def test_corrupted_check_digit_detected_on_real_image(self):
    forged = _render_mrz_with_bad_check_digit(SAMPLE_DIR / "passport_uk.jpg")
    result = extract_document_fields(forged, document_type="PASSPORT")
    self.assertFalse(result["checksum_valid"])

# 3. Disambiguation must never alter the name field (guards C1)
def test_disambiguation_preserves_name(self):
    original = extract_document_fields(PASSPORT_IMG, document_type="PASSPORT")
    noisy = _inject_ocr_noise(PASSPORT_IMG, field="document_number")
    corrected = extract_document_fields(noisy, document_type="PASSPORT")
    self.assertEqual(original["surname"], corrected["surname"])
    self.assertEqual(original["given_name"], corrected["given_name"])

# 4. Two different people must not be flagged as duplicates (guards C6/C8)
def test_distinct_identities_not_flagged_as_duplicate(self):
    store_embedding("s1", None, "ArcFace", _embedding_of(PERSON_A))
    hit = find_similar_identity(_embedding_of(PERSON_B), exclude_session="s2", model_name="ArcFace")
    self.assertIsNone(hit, "distinct people flagged as the same identity")

# 5. A failed detector must not read as 'Authentic' (guards H6)
def test_detector_failure_yields_inconclusive_not_authentic(self):
    with patch("modules.tampering_detection.detect_copy_move", side_effect=RuntimeError("boom")):
        result = analyze_tampering(PASSPORT_IMG)
    self.assertNotEqual(result["verdict"], "Authentic")
    self.assertTrue(result["degraded"])
```

### 🟡 M12 — `allow_origins=["*"]` with `allow_credentials=True`

`main.py:78-84`. **[verified]** This combination is invalid per the CORS specification — browsers reject a wildcard origin when credentials are allowed — so the configuration is simultaneously non-functional and maximally permissive in intent. Replace with an explicit origin allowlist.

### 🟡 M13 — `@app.on_event("startup")` is deprecated

`main.py:126`. Superseded by the lifespan context manager in current FastAPI/Starlette; currently emits a `DeprecationWarning` and will eventually be removed.

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_mrz_reader()
    threading.Thread(target=_bg_preload, daemon=True).start()
    yield

app = FastAPI(..., lifespan=lifespan)
```

### 🟡 M14 — `tamper_serializable` filter is redundant and shallow

`main.py:207-215`. The `if not hasattr(v, "shape")` filter predates `_json_safe`, which already handles `np.ndarray` and `np.generic` correctly. It is now redundant — and because it only inspects **top-level** values, it never looked inside `breakdown["exif_analysis"]["details"]`, which is precisely why the EXIF serialization bug in commit `c5e51fb` occurred. Note also that numpy scalars *do* expose `.shape` (`np.float64(1.0).shape == ()`), so if any score is ever left as a numpy scalar this filter will silently delete it. **[inferred — I could not execute numpy to confirm the current code path is unaffected; the current code casts scores to Python floats, so it appears benign today.]** Remove the filter and rely on `_json_safe`.

### 🟡 M15 — SQLite: no indices, no WAL, no busy timeout

`audit_logger.py`. `identity_embeddings` has no primary key and no indices; `screening_log` has none on `session_id`. Once H12 puts requests in a threadpool, concurrent writes against a default-configured SQLite file will produce `database is locked`. Add the indices from C7, plus:

```python
conn = sqlite3.connect(DB_PATH, timeout=30)
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA busy_timeout=30000")
```

Also `find_similar_identity` currently does a full table scan with a Python-loop cosine and a `json.loads` per row — fine at 10 rows, unusable at 10k. Load the gallery as a single matrix and use one vectorized `numpy` dot product, or move to FAISS as the problem statement's 1:N requirement implies.

### 🟡 M16 — Vendored third-party dataset committed to the repo

`backend/vendor/image_forensics/Signature Detection and Analysis/data/test/**` contains hundreds of PNG signature specimens from someone else's dataset. This bloats the clone, and the licence and consent basis for redistributing signature images is unexamined. `.gitignore` has the vendor exclusions **commented out**. Remove the data directory from version control and fetch vendored dependencies via a setup script or submodule.

---

## Priority order

Fix in this sequence. Items are ordered by *risk removed per hour spent*, not by severity alone.

**Before any networked demo (hours, not days)**
1. **C10** — add auth to `/screen-document` and `/audit-log`; bind to `127.0.0.1`
2. **C11** — upload size, type, and decodability validation; set `Image.MAX_IMAGE_PIXELS`
3. **H11** — one-line `session_id` fix so the audit trail actually joins
4. **C5** — reconcile documented vs implemented tamper weights

**Before trusting any output (this is where the real work is)**
5. **H8 / [validation](#recommended-validation)** — measure false-positive rate on the six genuine sample documents. Do this *first*; it tells you how bad H4/H5/M6/C4 actually are, and everything downstream depends on the answer
6. **C4** — copy-move: add the ratio test, switch to area-based scoring, exclude the MRZ band
7. **H4 / H5 / M6** — convert ELA, edge, and wavelet from global means to localized block-anomaly scores
8. **H7** — demote EXIF from 20% to ~3%, drop `exif_stripped` from scoring
9. **H6** — renormalize weights over successful detectors; add an `INCONCLUSIVE` verdict
10. **C6 / C7 / C8** — dedup: model-aware gallery, calibrated 1:N threshold, exclude by session, always store
11. **C1** — constrain OCR disambiguation to numeric spans; never touch line 1

**Before claiming the problem statement is addressed**
12. **C9** — the chip/NFC layer. This is the difference between a heuristic prototype and a system that detects photo substitution
13. **H3** — VIZ ↔ MRZ cross-check (cheap, deterministic, high signal)
14. **C2** — remove the fabricated visa fields; add visa-inside-passport validity
15. **H2 / M5** — date-logic validation, and fix the century pivot *before* enabling `DOB_IN_FUTURE`
16. **H14** — tests for Modules 3 and 4, starting with the genuine-document false-positive test
17. **H9 / H10** — honest model resolution, real `/health`, liveness on the matched face, penalize unassessed liveness
18. **H13** — document hashes and a hash-chained audit log
19. **C3** — resolve the AGPL position; ideally replace `mrz_scanner` with your own ICAO 9303 parser
20. **H12** — drop `async` so screening runs in the threadpool

**Cleanup**
21. M1–M4, M7–M16

---

## <a name="recommended-validation"></a>Recommended validation

The repository contains no measurement of detector accuracy — no ROC, no false-positive rate, no threshold justification. Before fixing anything in Module 3, establish the baseline. This is one afternoon's work and it will reorder your priorities:

```python
# tools/measure_baseline.py — run BEFORE and AFTER the Module 3 fixes
SAMPLES = ["passport_uk.jpg", "td1.jpg", "td2.jpg", "td3.jpg", "mrva.jpg", "mrvb.jpg"]

print(f"{'document':<20} {'tamper':>7} {'verdict':<12} {'ela':>7} {'edge':>7} {'wave':>7} {'cm':>7} {'exif':>7}")
for name in SAMPLES:                       # all six are GENUINE documents
    r = analyze_tampering(os.path.join(SAMPLE_DIR, name))
    b = r["breakdown"]
    print(f"{name:<20} {r['tamper_score']:>7} {r['verdict']:<12} "
          f"{b['ela']['score']:>7} {b['edge_detection']['score']:>7} "
          f"{b['wavelet']['score']:>7} {b['copy_move']['score']:>7} "
          f"{b['exif_analysis']['score']:>7}")
```

**Every row is a genuine document, so every `verdict` should read `Authentic`.** My prediction is that most will read `Suspicious` and the per-detector columns will show edge and wavelet dominating. If that holds, H4/H5/M6/C4 are confirmed and they are your top priority. If it does not hold, I am wrong about the severity and you should tell me — the prediction is falsifiable on purpose.

Then build the minimum labelled set you need to justify any threshold:

| Class | How to obtain | Purpose |
|---|---|---|
| Genuine | The 6 vendored samples + team documents (written consent) | False-positive rate |
| Photo substitution | Documented synthetic pipeline over the genuine set | Detection rate on the primary attack |
| Field alteration (DOB / name / number) | Same pipeline, per field, varied magnitude | Per-field sensitivity |
| Recapture | Photograph a screen; print and re-photograph | Presentation-attack coverage |

Then report **precision, recall, and the operating threshold** per detector — and evaluate **held out by generation method** (train or tune on one alteration method, test on another) so you are measuring generalization rather than recognition of your own synthetic artefacts. Reporting the in-method → cross-method drop honestly is worth more than a high number.

---

*Review based on static analysis of commit `c5e51fb`. Nothing was executed. Findings marked **[inferred]** — particularly the Module 3 false-positive predictions — should be confirmed with the baseline measurement above before being treated as settled.*

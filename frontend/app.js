// API endpoint configuration (uses current origin whether localhost:8000 or production Render domain)
const API_BASE = (window.location.origin && window.location.origin !== 'null' && !window.location.protocol.startsWith('file'))
    ? window.location.origin 
    : 'http://localhost:8000';

const API_SCREEN = `${API_BASE}/screen-document`;
const API_HEALTH = `${API_BASE}/health`;

// DOM Elements
const docInput = document.getElementById('document-input');
const docDropzone = document.getElementById('doc-dropzone');
const docPreviewWrap = document.getElementById('doc-preview-wrap');
const docPreview = document.getElementById('doc-preview');
const docRemoveBtn = document.getElementById('doc-remove-btn');

const selfieInput = document.getElementById('selfie-input');
const selfieDropzone = document.getElementById('selfie-dropzone');
const selfiePreviewWrap = document.getElementById('selfie-preview-wrap');
const selfiePreview = document.getElementById('selfie-preview');
const selfieRemoveBtn = document.getElementById('selfie-remove-btn');

const videoInput = document.getElementById('video-input');
const videoDropzone = document.getElementById('video-dropzone');
const videoDropContent = document.getElementById('video-drop-content');
const videoFilename = document.getElementById('video-filename');
const videoRemoveBtn = document.getElementById('video-remove-btn');
const recordBtn = document.getElementById('record-btn');
const countdownEl = document.getElementById('liveness-countdown');

const screeningForm = document.getElementById('screening-form');
const submitBtn = document.getElementById('submit-btn');
const btnText = submitBtn.querySelector('.btn-text');
const loadingSpinner = document.getElementById('loading-spinner');

const resultsPlaceholder = document.getElementById('results-placeholder');
const resultsContent = document.getElementById('results-content');
const backendStatus = document.getElementById('backend-status');

let recordedVideoBlob = null;
let mediaRecorder = null;

// Setup Dropzones
function setupDropzone(input, dropzone, previewWrap, previewImg, removeBtn) {
    function showPreview(file) {
        if (!file || !file.type.startsWith('image/')) return;
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            previewWrap.classList.remove('hidden');
            dropzone.querySelector('.dropzone-content').classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }

    function clearPreview(e) {
        if (e) e.stopPropagation();
        input.value = '';
        previewImg.src = '';
        previewWrap.classList.add('hidden');
        dropzone.querySelector('.dropzone-content').classList.remove('hidden');
    }

    input.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            showPreview(e.target.files[0]);
        }
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        if (dt.files && dt.files.length > 0) {
            input.files = dt.files;
            showPreview(dt.files[0]);
        }
    });

    removeBtn.addEventListener('click', clearPreview);
}

setupDropzone(docInput, docDropzone, docPreviewWrap, docPreview, docRemoveBtn);
setupDropzone(selfieInput, selfieDropzone, selfiePreviewWrap, selfiePreview, selfieRemoveBtn);

// ── Liveness Video: File Upload & MediaRecorder Capture ─────────────────────
if (videoInput) {
    videoInput.addEventListener('change', () => {
        if (videoInput.files && videoInput.files[0]) {
            recordedVideoBlob = null;
            videoFilename.textContent = '📹 ' + videoInput.files[0].name;
            videoDropContent.classList.add('hidden');
            videoFilename.classList.remove('hidden');
            videoRemoveBtn.classList.remove('hidden');
        }
    });
}
if (videoRemoveBtn) {
    videoRemoveBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        recordedVideoBlob = null;
        if (videoInput) videoInput.value = '';
        videoFilename.classList.add('hidden');
        videoRemoveBtn.classList.add('hidden');
        videoDropContent.classList.remove('hidden');
    });
}

if (recordBtn) {
    recordBtn.addEventListener('click', async () => {
        if (mediaRecorder && mediaRecorder.state === 'recording') return;
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            const chunks = [];
            const mimeType = MediaRecorder.isTypeSupported('video/webm') ? 'video/webm' : 'video/mp4';
            mediaRecorder = new MediaRecorder(stream, { mimeType });
            mediaRecorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
            mediaRecorder.onstop = () => {
                recordedVideoBlob = new Blob(chunks, { type: mimeType });
                stream.getTracks().forEach(t => t.stop());
                videoFilename.textContent = `📹 Recorded liveness clip (${(recordedVideoBlob.size / 1024).toFixed(0)} KB)`;
                videoDropContent.classList.add('hidden');
                videoFilename.classList.remove('hidden');
                videoRemoveBtn.classList.remove('hidden');
                countdownEl.textContent = '✓ Done';
                setTimeout(() => { countdownEl.textContent = ''; }, 2000);
            };
            mediaRecorder.start();
            recordBtn.disabled = true;
            let secs = 3;
            countdownEl.textContent = `🔴 Recording... ${secs}s`;
            const iv = setInterval(() => {
                secs--;
                if (secs <= 0) {
                    clearInterval(iv);
                    mediaRecorder.stop();
                    recordBtn.disabled = false;
                } else {
                    countdownEl.textContent = `🔴 Recording... ${secs}s`;
                }
            }, 1000);
        } catch (err) {
            countdownEl.textContent = `Camera error: ${err.message}`;
            console.error(err);
        }
    });
}

// Health check on load
async function checkHealth() {
    try {
        const res = await fetch(API_HEALTH);
        if (res.ok) {
            backendStatus.textContent = 'Backend Active';
            backendStatus.style.color = '#10b981';
        } else {
            backendStatus.textContent = 'Backend Issue';
            backendStatus.style.color = '#f59e0b';
        }
    } catch (err) {
        backendStatus.textContent = 'Backend Offline';
        backendStatus.style.color = '#ef4444';
    }
}
checkHealth();

// Form Submit Handler
screeningForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    if (!docInput.files || docInput.files.length === 0) {
        alert('Please select or upload a document image first.');
        return;
    }

    const formData = new FormData();
    formData.append('document', docInput.files[0]);
    if (selfieInput.files && selfieInput.files.length > 0) {
        formData.append('selfie', selfieInput.files[0]);
    }
    // Liveness video: prefer recorded blob, then file upload
    if (recordedVideoBlob) {
        const ext = recordedVideoBlob.type.includes('webm') ? 'webm' : 'mp4';
        formData.append('video', recordedVideoBlob, `liveness.${ext}`);
    } else if (videoInput && videoInput.files && videoInput.files.length > 0) {
        formData.append('video', videoInput.files[0]);
    }

    // Set Loading State
    submitBtn.disabled = true;
    btnText.textContent = 'Screening in progress...';
    loadingSpinner.classList.remove('hidden');

    try {
        const response = await fetch(API_SCREEN, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || `Server returned status ${response.status}`);
        }

        const data = await response.json();
        renderResults(data);
    } catch (err) {
        alert(`Screening failed: ${err.message}`);
        console.error(err);
    } finally {
        submitBtn.disabled = false;
        btnText.textContent = 'Run Security Screening';
        loadingSpinner.classList.add('hidden');
    }
});

// Render Results
function renderResults(data) {
    resultsPlaceholder.classList.add('hidden');
    resultsContent.classList.remove('hidden');

    const risk = data.risk || {};
    const ocr = data.ocr || {};
    const tampering = data.tampering || {};
    const face = data.face || {};
    const liveness = data.liveness || {};

    // 1. Overall Verdict & Score Dial
    const scoreVal = document.getElementById('risk-score-val');
    const scoreDial = document.querySelector('.score-dial');
    const verdictBadge = document.getElementById('verdict-badge');
    const flagsList = document.getElementById('flags-list');

    const score = risk.risk_score !== undefined ? risk.risk_score : 0;
    const verdict = risk.verdict || 'UNKNOWN';
    const reviewRequired = risk.requires_manual_review === true;

    scoreVal.textContent = Math.round(score);
    verdictBadge.textContent = reviewRequired ? 'MANUAL REVIEW REQUIRED' : `${verdict} RISK`;

    // Clear previous classes
    scoreDial.className = 'score-dial';
    verdictBadge.className = 'verdict-badge';

    if (reviewRequired) {
        scoreDial.style.borderColor = 'var(--danger)';
        verdictBadge.classList.add('verdict-high');
    } else if (verdict === 'LOW') {
        scoreDial.style.borderColor = 'var(--success)';
        verdictBadge.classList.add('verdict-low');
    } else if (verdict === 'MEDIUM') {
        scoreDial.style.borderColor = 'var(--warning)';
        verdictBadge.classList.add('verdict-medium');
    } else {
        scoreDial.style.borderColor = 'var(--danger)';
        verdictBadge.classList.add('verdict-high');
    }

    // Flags
    flagsList.innerHTML = '';
    const flags = risk.flags || [];
    if (flags.length === 0) {
        flagsList.innerHTML = '<span class="flag-item" style="color: var(--success);">✓ No critical risk flags detected</span>';
    } else {
        flags.forEach(flag => {
            const div = document.createElement('div');
            div.className = 'flag-item';
            div.textContent = flag;
            flagsList.appendChild(div);
        });
    }

    // 1.5 Score Breakdown Table
    const breakdownBody = document.getElementById('score-breakdown-body');
    const breakdownContainer = document.getElementById('score-breakdown-container');
    if (breakdownBody && breakdownContainer) {
        breakdownBody.innerHTML = '';
        const scoreBreakdowns = risk.score_breakdown || [];
        if (scoreBreakdowns.length > 0) {
            scoreBreakdowns.forEach(item => {
                const tr = document.createElement('tr');
                if (item.points_added > 0) {
                    tr.style.color = 'var(--danger)';
                    tr.style.fontWeight = '500';
                }
                tr.innerHTML = `
                    <td>${item.component}</td>
                    <td style="text-align: right;">+${item.points_added.toFixed(2)}</td>
                    <td style="text-align: right;">${item.max_points.toFixed(2)}</td>
                    <td>${item.reason}</td>
                `;
                breakdownBody.appendChild(tr);
            });
            breakdownContainer.classList.remove('hidden');
        } else {
            breakdownContainer.classList.add('hidden');
        }
    }

    // 2. Module 1+2: MRZ / OCR Details
    const ocrBadge = document.getElementById('ocr-badge');
    const ocrTable = document.getElementById('ocr-table-body');
    ocrTable.innerHTML = '';

    if (ocr.status === 'SUCCESS' || ocr.status === 'VALID' || ocr.status === 'INVALID' || ocr.status === 'NOT_APPLICABLE') {
        let isSuccess = (ocr.status === 'SUCCESS' || ocr.status === 'VALID');
        let badgeText = isSuccess ? 'VALID MRZ' : (ocr.status === 'NOT_APPLICABLE' ? 'NON-MRZ ID' : 'INVALID CHECKSUM');
        
        ocrBadge.textContent = badgeText;
        ocrBadge.className = `module-tag ${isSuccess || ocr.status === 'NOT_APPLICABLE' ? 'badge-success' : 'badge-danger'}`;

        const rows = [
            ['Document Type', ocr.mrz_type || ocr.document_code || ocr.document_type || 'N/A'],
            ['Issuer Country', ocr.issuer_code || 'N/A'],
            ['Holder Name', `${ocr.given_name || ''} ${ocr.surname || ''}`.trim() || 'N/A'],
            ['Document Number', ocr.document_number || 'N/A'],
            ['Date of Birth', ocr.birth_date || 'N/A'],
            ['Sex / Gender', ocr.sex || 'N/A'],
            ['Expiry Date', ocr.expiry_date || 'N/A'],
            ['Checksum Validity', ocr.checksum_valid === true ? '✓ Passed (All check digits valid)' : (ocr.checksum_valid === false ? '✗ FAILED validation' : 'N/A')]
        ];

        rows.forEach(([label, val]) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `<td>${label}</td><td>${val}</td>`;
            ocrTable.appendChild(tr);
        });
    } else {
        ocrBadge.textContent = 'NO MRZ / FAILED';
        ocrBadge.className = 'module-tag badge-danger';
        ocrTable.innerHTML = `<tr><td colspan="2" class="muted-text">Could not extract MRZ: ${ocr.error || 'Unknown error'}</td></tr>`;
    }

    // 3. Module 3: Tampering Forensics
    const tamperBadge = document.getElementById('tamper-badge');
    const tamperScore = tampering.tamper_score !== undefined ? tampering.tamper_score : 0;
    const tamperVerdict = tampering.verdict || 'Unknown';

    tamperBadge.textContent = `${tamperVerdict.toUpperCase()} (${tamperScore}%)`;
    if (tamperVerdict === 'Authentic') {
        tamperBadge.className = 'module-tag badge-success';
    } else if (tamperVerdict === 'Suspicious') {
        tamperBadge.className = 'module-tag badge-warning';
    } else {
        tamperBadge.className = 'module-tag badge-danger';
    }

    const breakdown = tampering.breakdown || {};
    document.getElementById('metric-ela').textContent = `${breakdown.ela?.score ?? 0}%`;
    document.getElementById('metric-edge').textContent = `${breakdown.edge_detection?.score ?? 0}%`;
    document.getElementById('metric-wavelet').textContent = `${breakdown.wavelet?.score ?? 0}%`;
    document.getElementById('metric-copymove').textContent = `${breakdown.copy_move?.score ?? 0}%`;

    // EXIF
    const exifMetric = document.getElementById('exif-metric-item');
    if (exifMetric) {
        if (breakdown.exif_analysis) {
            document.getElementById('metric-exif').textContent = `${breakdown.exif_analysis.score ?? 0}%`;
            exifMetric.style.display = 'flex';
        } else {
            exifMetric.style.display = 'none';
        }
    }

    // ELA Heatmap
    const elaContainer = document.getElementById('ela-heatmap-container');
    const elaImg = document.getElementById('ela-heatmap-img');
    if (elaContainer && elaImg) {
        if (tampering.ela_heatmap_b64) {
            elaImg.src = `data:image/png;base64,${tampering.ela_heatmap_b64}`;
            elaContainer.classList.remove('hidden');
        } else {
            elaContainer.classList.add('hidden');
        }
    }

    // 4. Module 4: Face Verification
    const faceBadge = document.getElementById('face-badge');
    const faceDetails = document.getElementById('face-details');

    if (face.verified === true) {
        faceBadge.textContent = 'MATCHED';
        faceBadge.className = 'module-tag badge-success';
        faceDetails.innerHTML = `
            <table class="data-table">
                <tr><td>Verification</td><td>✓ Match Confirmed</td></tr>
                <tr><td>Confidence</td><td>${face.confidence}%</td></tr>
                <tr><td>Distance</td><td>${face.distance} (Threshold: ${face.threshold})</td></tr>
                <tr><td>Model</td><td>${face.model || 'VGG-Face'}</td></tr>
                ${face.warning ? `<tr><td>Notice</td><td class="muted-text">${face.warning}</td></tr>` : ''}
            </table>
        `;
    } else if (face.verified === false) {
        faceBadge.textContent = 'MISMATCH';
        faceBadge.className = 'module-tag badge-danger';
        faceDetails.innerHTML = `
            <table class="data-table">
                <tr><td>Verification</td><td>✗ Identity Mismatch</td></tr>
                <tr><td>Confidence</td><td>${face.confidence}%</td></tr>
                <tr><td>Distance</td><td>${face.distance}</td></tr>
                <tr><td>Model</td><td>${face.model || 'VGG-Face'}</td></tr>
                ${face.error ? `<tr><td>Error</td><td class="muted-text">${face.error}</td></tr>` : ''}
            </table>
        `;
    } else {
        faceBadge.textContent = 'SKIPPED';
        faceBadge.className = 'module-tag';
        faceDetails.innerHTML = `<p class="muted-text">${face.note || 'No selfie uploaded for face matching.'}</p>`;
    }

    // 5. Module 3.5: Document Liveness
    const livenessBadge = document.getElementById('liveness-badge');
    const livenessDetails = document.getElementById('liveness-details');

    if (livenessBadge && livenessDetails) {
        const sr = liveness.screen_replay || {};
        const pm = liveness.physical_motion || {};

        // Part A badge
        const replayOk = sr.is_screen_replay === false;
        const replayFail = sr.is_screen_replay === true;
        const motionVerdict = pm.verdict || 'SKIPPED';

        let lBadgeText, lBadgeClass;
        if (replayFail) {
            lBadgeText = 'REPLAY DETECTED';
            lBadgeClass = 'module-tag badge-danger';
        } else if (motionVerdict === 'STATIC') {
            lBadgeText = 'STATIC IMAGE';
            lBadgeClass = 'module-tag badge-danger';
        } else if (motionVerdict === 'PHYSICAL' && replayOk) {
            lBadgeText = 'PASS';
            lBadgeClass = 'module-tag badge-success';
        } else if (motionVerdict === 'SKIPPED') {
            lBadgeText = replayOk ? 'PARTIAL PASS' : (sr.is_screen_replay === null ? 'SKIPPED' : 'CHECK');
            lBadgeClass = replayOk ? 'module-tag badge-success' : 'module-tag';
        } else {
            lBadgeText = 'INCONCLUSIVE';
            lBadgeClass = 'module-tag badge-warning';
        }

        livenessBadge.textContent = lBadgeText;
        livenessBadge.className = lBadgeClass;

        const methodLabel = sr.method === 'svm' ? 'Trained SVM' :
                            sr.method === 'threshold' ? 'FFT+Texture Heuristic' : 'Unavailable';
        const replayLabel = sr.is_screen_replay === true ? '⚠ Replay Detected' :
                            sr.is_screen_replay === false ? '✓ No Replay' : 'N/A';

        livenessDetails.innerHTML = `
            <table class="data-table">
                <tr><td><b>Part A: Screen Replay</b></td><td>${replayLabel} <span class="muted-text">(${methodLabel})</span></td></tr>
                <tr><td>FFT Peak Ratio</td><td>${sr.fft_peak_ratio ?? 'N/A'}</td></tr>
                <tr><td>Texture Uniformity</td><td>${sr.texture_uniformity ?? 'N/A'}</td></tr>
                <tr><td><b>Part B: Physical Motion</b></td><td>${motionVerdict}</td></tr>
                <tr><td>Highlight Displacement</td><td>${pm.mean_highlight_displacement_px != null ? pm.mean_highlight_displacement_px.toFixed(2) + ' px' : 'N/A'}</td></tr>
                <tr><td>Hologram HSV Shift</td><td>${pm.hologram_hsv_shift ?? 'N/A'}</td></tr>
                <tr><td>Frames Analysed</td><td>${pm.frame_count ?? 0}</td></tr>
                ${pm.note ? `<tr><td>Note</td><td class="muted-text">${pm.note}</td></tr>` : ''}
            </table>
        `;
    }

    // 6. Raw JSON
    document.getElementById('raw-json-output').textContent = JSON.stringify(data, null, 2);
}

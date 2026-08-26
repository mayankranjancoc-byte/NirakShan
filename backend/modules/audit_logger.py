import sqlite3
import json
import os
import uuid
import logging

import numpy as np

logger = logging.getLogger(__name__)

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "audit_log.db"))


def _get_conn() -> sqlite3.Connection:
    """Return a WAL-enabled connection with a generous busy timeout."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _init_db():
    conn = _get_conn()
    cursor = conn.cursor()

    # Create tables if they don't exist at all
    cursor.executescript('''
        CREATE TABLE IF NOT EXISTS screening_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            document_type TEXT,
            mrz_status TEXT,
            risk_score REAL,
            verdict TEXT,
            flags TEXT,
            face_verified TEXT,
            exif_anomaly_score REAL,
            tamper_score REAL,
            session_id TEXT,
            document_sha256 TEXT,
            selfie_sha256 TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_log_session
            ON screening_log(session_id);
    ''')

    # Migration: add new columns to screening_log if they don't exist yet
    existing = {row[1] for row in cursor.execute("PRAGMA table_info(screening_log)")}
    for col, defn in (
        ("document_sha256", "TEXT"),
        ("selfie_sha256", "TEXT"),
    ):
        if col not in existing:
            cursor.execute(f"ALTER TABLE screening_log ADD COLUMN {col} {defn}")
            logger.info("Migrated screening_log: added column %s", col)

    # identity_embeddings: drop and recreate if the schema is stale
    # (old table had no model_name / embedding_dim columns — mixing models is unsafe)
    old_cols = {row[1] for row in cursor.execute("PRAGMA table_info(identity_embeddings)")} \
        if cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='identity_embeddings'").fetchone() \
        else set()

    if "model_name" not in old_cols:
        logger.info("Recreating identity_embeddings table (schema migration)")
        cursor.executescript('''
            DROP TABLE IF EXISTS identity_embeddings;
            CREATE TABLE identity_embeddings (
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
        ''')
    else:
        # Ensure indices exist
        cursor.executescript('''
            CREATE INDEX IF NOT EXISTS idx_emb_model ON identity_embeddings(model_name);
            CREATE INDEX IF NOT EXISTS idx_emb_session ON identity_embeddings(session_id);
        ''')

    conn.commit()
    conn.close()


# Initialize DB on import
_init_db()


def log_screening_result(result_dict: dict) -> str:
    """
    Inserts a row into screening_log, returns the session_id.
    Uses the session_id already in result_dict so the audit row
    can be joined to identity_embeddings (fixes H11).
    """
    # H11 fix: reuse the session_id from the pipeline, not a new one.
    session_id = result_dict.get("session_id") or uuid.uuid4().hex
    timestamp = result_dict.get("timestamp") or __import__("datetime").datetime.now().isoformat()

    # Extract fields from result_dict
    risk = result_dict.get("risk", {})
    ocr = result_dict.get("ocr", {})
    face = result_dict.get("face", {})
    tamper = result_dict.get("tampering", {})

    document_type = ocr.get("document_type")
    mrz_status = ocr.get("mrz_status")
    risk_score = risk.get("risk_score")
    verdict = risk.get("verdict")
    flags_json = json.dumps(risk.get("flags", []))

    face_verified_val = face.get("verified")
    face_error = face.get("error")
    if face_error:
        face_verified = "ERROR"
    elif face_verified_val is True:
        face_verified = "MATCH"
    elif face_verified_val is False:
        face_verified = "MISMATCH"
    else:
        face_verified = "SKIPPED"

    exif_score = None
    if tamper.get("breakdown") and tamper["breakdown"].get("exif_analysis"):
        exif_score = tamper["breakdown"]["exif_analysis"].get("score")

    tamper_score = tamper.get("tamper_score")
    document_sha256 = result_dict.get("document_sha256")
    selfie_sha256 = result_dict.get("selfie_sha256")

    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO screening_log
        (timestamp, document_type, mrz_status, risk_score, verdict, flags,
         face_verified, exif_anomaly_score, tamper_score, session_id,
         document_sha256, selfie_sha256)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        timestamp, document_type, mrz_status, risk_score, verdict, flags_json,
        face_verified, exif_score, tamper_score, session_id,
        document_sha256, selfie_sha256,
    ))
    conn.commit()
    conn.close()

    return session_id


def get_recent_screenings(limit: int = 20) -> list[dict]:
    """Returns last N rows as list of dicts."""
    conn = _get_conn()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM screening_log ORDER BY id DESC LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def store_embedding(
    session_id: str,
    passport_number: str | None,
    model_name: str,
    embedding: list,
) -> None:
    """Store a face embedding tagged with the model that produced it."""
    conn = _get_conn()
    cursor = conn.cursor()
    timestamp = __import__("datetime").datetime.now().isoformat()
    cursor.execute('''
        INSERT INTO identity_embeddings
        (session_id, passport_number, model_name, embedding_dim, embedding_json, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        session_id,
        passport_number,  # may be None when MRZ extraction failed
        model_name,
        len(embedding),
        json.dumps(embedding),
        timestamp,
    ))
    conn.commit()
    conn.close()


# Model-specific 1:N dedup thresholds (cosine similarity).
# These are conservative starting values — calibrate against a labelled gallery
# and document the target false-match rate before production use.
DEDUP_THRESHOLDS: dict[str, float] = {
    "ArcFace": 0.68,
    "VGG-Face": 0.75,
}


def find_similar_identity(
    embedding: list,
    exclude_session: str,
    model_name: str,
    threshold: float | None = None,
) -> dict | None:
    """
    Search the gallery for a face similar to `embedding`, returning the best
    (highest-similarity) match above the threshold.

    Excludes rows with the same `session_id` as the current screening.
    Never excludes by passport_number (C8 fix: that would suppress the exact
    case we want to catch — same person, different document).

    Embeddings from other models are never compared (C7 fix: different dims,
    different similarity distributions).
    """
    if threshold is None:
        threshold = DEDUP_THRESHOLDS.get(model_name)
        if threshold is None:
            logger.warning("No dedup threshold configured for model %s; skipping", model_name)
            return None

    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        '''
        SELECT session_id, passport_number, embedding_json
        FROM identity_embeddings
        WHERE model_name = ? AND session_id != ?
        ''',
        (model_name, exclude_session),
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return None

    emb_a = np.array(embedding, dtype=np.float64)
    norm_a = np.linalg.norm(emb_a)
    if norm_a == 0:
        return None

    best: dict | None = None
    best_sim: float = -1.0

    for r_session, r_passport, r_emb_json in rows:
        try:
            emb_b = np.array(json.loads(r_emb_json), dtype=np.float64)
        except Exception:
            continue

        if len(emb_b) != len(emb_a):
            # Should not happen because we filter by model_name, but log if it does.
            logger.warning(
                "Embedding dim mismatch for session %s: expected %d, got %d",
                r_session, len(emb_a), len(emb_b),
            )
            continue

        norm_b = np.linalg.norm(emb_b)
        if norm_b == 0:
            continue

        sim = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
        if sim > threshold and sim > best_sim:
            best_sim = sim
            best = {
                "matched_session": r_session,
                "matched_passport": r_passport,
                "similarity": round(sim, 4),
                "model": model_name,
                "threshold_used": threshold,
            }

    return best

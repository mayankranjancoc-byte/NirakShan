import sqlite3
import json
import os
import uuid

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "audit_log.db"))

def _init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
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
            session_id TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS identity_embeddings (
            session_id TEXT,
            passport_number TEXT,
            embedding_json TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Initialize DB on import
_init_db()

def log_screening_result(result_dict: dict) -> str:
    """
    Inserts a row into screening_log, returns the session_id.
    """
    session_id = uuid.uuid4().hex
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
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO screening_log 
        (timestamp, document_type, mrz_status, risk_score, verdict, flags, face_verified, exif_anomaly_score, tamper_score, session_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        timestamp, document_type, mrz_status, risk_score, verdict, flags_json, face_verified, exif_score, tamper_score, session_id
    ))
    conn.commit()
    conn.close()
    
    return session_id

def get_recent_screenings(limit: int = 20) -> list[dict]:
    """
    Returns last N rows as list of dicts.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM screening_log ORDER BY id DESC LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def store_embedding(session_id: str, passport_number: str, embedding: list) -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = __import__("datetime").datetime.now().isoformat()
    cursor.execute('''
        INSERT INTO identity_embeddings 
        (session_id, passport_number, embedding_json, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (session_id, passport_number, json.dumps(embedding), timestamp))
    conn.commit()
    conn.close()

def find_similar_identity(embedding: list, exclude_passport: str, threshold: float = 0.40) -> dict:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT session_id, passport_number, embedding_json FROM identity_embeddings')
    rows = cursor.fetchall()
    conn.close()
    
    import numpy as np
    emb_a = np.array(embedding)
    norm_a = np.linalg.norm(emb_a)
    
    for row in rows:
        r_session = row[0]
        r_passport = row[1]
        r_emb_json = row[2]
        
        if r_passport == exclude_passport:
            continue
            
        try:
            emb_b = np.array(json.loads(r_emb_json))
            norm_b = np.linalg.norm(emb_b)
            if norm_a > 0 and norm_b > 0:
                sim = np.dot(emb_a, emb_b) / (norm_a * norm_b)
                if sim > threshold:
                    return {
                        "matched_session": r_session,
                        "matched_passport": r_passport,
                        "similarity": float(sim)
                    }
        except Exception:
            continue
            
    return None

"""
Module 4: Face Verification
Wraps deepface (https://github.com/serengil/deepface)
License: MIT

Provides face matching between a document photo and a live selfie,
with optional anti-spoofing (liveness detection).
"""

import os
import sys

from deepface import DeepFace


def extract_face_from_document(image_path: str, output_path: str = None) -> str:
    """
    Extract the face from a document image using deepface's built-in detector.

    Args:
        image_path: Path to the document image.
        output_path: Optional path to save the extracted face. If None,
                     saves to the same directory with '_face' suffix.

    Returns:
        Path to the extracted face image.
    """
    import cv2
    import numpy as np

    faces = DeepFace.extract_faces(
        img_path=image_path,
        detector_backend="opencv",
        enforce_detection=False,
    )

    if not faces:
        raise ValueError("No face detected in the document image")

    # Use the first (largest/most confident) face
    face_data = faces[0]
    face_array = face_data["face"]

    # Convert from normalized float [0,1] to uint8 [0,255]
    if face_array.max() <= 1.0:
        face_array = (face_array * 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV saving
    face_bgr = cv2.cvtColor(face_array, cv2.COLOR_RGB2BGR)

    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_face{ext}"

    cv2.imwrite(output_path, face_bgr)
    return output_path


def verify_face_match(doc_face_path: str, live_photo_path: str, session_id: str = None, passport_number: str = None) -> dict:
    """
    Verify whether the face in a document matches a live photo.

    Uses the full document/photo images directly (deepface handles
    face detection internally). anti_spoofing is attempted first but
    falls back gracefully if the model can't be loaded.

    Args:
        doc_face_path: Path to the document image (or extracted face).
        live_photo_path: Path to the live selfie/photo.
        session_id: The unique ID for this screening session.
        passport_number: The passport number extracted from the document.

    Returns:
        dict with:
          - verified: bool -- whether faces match
          - distance: float -- face embedding distance
          - confidence: float -- match confidence percentage
          - threshold: float -- distance threshold used
          - model: str -- face recognition model used
          - is_real: bool -- anti-spoofing result (if available)
          - error: str -- error message if verification failed
          - identity_cross_check: dict -- result of deduplication check
    """
    flags = []
    
    # Check if ArcFace model weights are downloaded
    arcface_weights = os.path.expanduser("~/.deepface/weights/arcface_weights.h5")
    has_arcface = os.path.exists(arcface_weights) and os.path.getsize(arcface_weights) > 100_000_000
    
    model_name = "ArcFace" if has_arcface else "VGG-Face"
    detector_backend = "retinaface" if has_arcface else "opencv"
    
    # ── Cross-Document Identity Deduplication ──
    identity_cross_check = {
        "duplicate_detected": False,
        "matched_session_id": None,
        "matched_passport_number": None,
        "similarity_score": None
    }
    
    try:
        try:
            embedding = DeepFace.represent(
                img_path=live_photo_path,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=False
            )[0]["embedding"]
        except Exception:
            embedding = DeepFace.represent(
                img_path=live_photo_path,
                model_name="VGG-Face",
                detector_backend="opencv",
                enforce_detection=False
            )[0]["embedding"]
        
        if session_id and passport_number:
            from modules.audit_logger import find_similar_identity, store_embedding
            
            similar = find_similar_identity(embedding, exclude_passport=passport_number)
            if similar:
                identity_cross_check = {
                    "duplicate_detected": True,
                    "matched_session_id": similar["matched_session"],
                    "matched_passport_number": similar["matched_passport"],
                    "similarity_score": similar["similarity"]
                }
                flags.append("MULTIPLE_IDENTITY_SUSPECTED")
                
            store_embedding(session_id, passport_number, embedding)
            
    except Exception as e:
        flags.append(f"EMBEDDING_ERROR: {e}")

    result = None
    # 1. Try with anti_spoofing
    try:
        result = DeepFace.verify(
            img1_path=doc_face_path,
            img2_path=live_photo_path,
            model_name=model_name,
            detector_backend=detector_backend,
            anti_spoofing=True,
            enforce_detection=False
        )
    except Exception:
        flags.append("ANTI_SPOOF_UNAVAILABLE")
        # 2. Try without anti_spoofing
        try:
            result = DeepFace.verify(
                img1_path=doc_face_path,
                img2_path=live_photo_path,
                model_name=model_name,
                detector_backend=detector_backend,
                anti_spoofing=False,
                enforce_detection=False
            )
        except Exception:
            # 3. Fallback to VGG-Face + opencv
            try:
                result = DeepFace.verify(
                    img1_path=doc_face_path,
                    img2_path=live_photo_path,
                    model_name="VGG-Face",
                    detector_backend="opencv",
                    anti_spoofing=False,
                    enforce_detection=False
                )
            except Exception as e:
                return {
                    "verified": False,
                    "distance": None,
                    "confidence": 0.0,
                    "is_real": None,
                    "error": f"Face verification failed: {e}",
                    "identity_cross_check": identity_cross_check,
                }

    distance = result.get("distance", 0)
    threshold = result.get("threshold", 0.68)
    confidence = result.get("confidence", 0)

    # If confidence not in result, calculate from distance
    if confidence == 0 and threshold > 0:
        confidence = max(0, (1 - distance / threshold)) * 100

    response = {
        "verified": result.get("verified", False),
        "distance": round(distance, 4),
        "confidence": round(confidence, 2),
        "threshold": threshold,
        "model": result.get("model", model_name),
        "detector": result.get("detector_backend", detector_backend),
        "is_real": result.get("is_real", None),
        "flags": flags,
        "identity_cross_check": identity_cross_check,
    }
    
    if "ANTI_SPOOF_UNAVAILABLE" in flags:
        response["warning"] = "Anti-spoofing unavailable; liveness not checked"

    return response


if __name__ == "__main__":
    import json

    if len(sys.argv) >= 3:
        doc_path = sys.argv[1]
        live_path = sys.argv[2]

        print(f"Document face: {doc_path}")
        print(f"Live photo:    {live_path}")
        print("-" * 60)
        result = verify_face_match(doc_path, live_path)
        print(json.dumps(result, indent=4))
    else:
        # Self-test using sample data from fastmrz
        sample_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "vendor", "fastmrz", "data")
        )
        passport = os.path.join(sample_dir, "passport_uk.jpg")
        td1 = os.path.join(sample_dir, "td1.jpg")

        if not os.path.exists(passport):
            print(f"Sample passport not found at: {passport}")
            sys.exit(1)

        print("=== MATCH TEST (passport vs itself) ===")
        result1 = verify_face_match(passport, passport)
        print(json.dumps(result1, indent=4))

        print()
        print("=== MISMATCH TEST (passport vs td1 ID) ===")
        result2 = verify_face_match(passport, td1)
        print(json.dumps(result2, indent=4))

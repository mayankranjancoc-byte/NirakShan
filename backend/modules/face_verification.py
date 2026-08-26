"""Module 4: strict document-portrait and selfie face verification."""

import os
import tempfile

import cv2
import numpy as np
from deepface import DeepFace


# A face smaller than this in the original image is too low-detail for a
# reliable biometric comparison. It is better to ask for a clearer image than
# to silently compare an entire passport page as though it were a face.
MIN_FACE_SIDE_PX = 40
# OpenCV is already bundled with DeepFace and avoids an unexpected first-run
# RetinaFace download. RetinaFace remains a strict fallback for harder images.
DETECTION_BACKENDS = ("opencv", "retinaface")


def _face_area(face: dict) -> int:
    area = face.get("facial_area", {})
    return int(area.get("w", 0)) * int(area.get("h", 0))


def _extract_face_crop(image_path: str, label: str) -> tuple[str, dict]:
    """Strictly detect the largest usable face and write an aligned temp crop."""
    failures = []
    for backend in DETECTION_BACKENDS:
        try:
            faces = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend=backend,
                enforce_detection=True,
                align=True,
            )
            candidates = [
                face for face in faces
                if min(
                    int(face.get("facial_area", {}).get("w", 0)),
                    int(face.get("facial_area", {}).get("h", 0)),
                ) >= MIN_FACE_SIDE_PX
            ]
            if not candidates:
                failures.append(f"{backend}: no face at least {MIN_FACE_SIDE_PX}px")
                continue

            face = max(candidates, key=_face_area)
            face_rgb = face["face"]
            if face_rgb.max() <= 1.0:
                face_rgb = face_rgb * 255
            face_rgb = np.clip(face_rgb, 0, 255).astype(np.uint8)

            handle = tempfile.NamedTemporaryFile(
                prefix=f"screening_{label}_face_", suffix=".jpg", delete=False
            )
            handle.close()
            if not cv2.imwrite(handle.name, cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)):
                os.remove(handle.name)
                raise RuntimeError("could not save detected face crop")

            area = face.get("facial_area", {})
            return handle.name, {
                "backend": backend,
                "faces_detected": len(faces),
                "original_face_size": {
                    "width": int(area.get("w", 0)),
                    "height": int(area.get("h", 0)),
                },
            }
        except Exception as error:
            failures.append(f"{backend}: {error}")

    raise ValueError(
        f"Could not reliably detect a usable face in the {label} image. "
        "Use a sharper, front-facing image with one clearly visible face. "
        f"Detection details: {'; '.join(failures)}"
    )


def extract_face_from_document(image_path: str, output_path: str = None) -> str:
    """Extract a real detected portrait from a document; never use the whole page."""
    crop_path, _ = _extract_face_crop(image_path, "document")
    if output_path is None:
        return crop_path
    try:
        os.replace(crop_path, output_path)
        return output_path
    except Exception:
        if os.path.exists(crop_path):
            os.remove(crop_path)
        raise


def _liveness_check(selfie_path: str, detector_backend: str) -> tuple[bool | None, str | None]:
    """Return an optional liveness result without allowing it to break matching."""
    try:
        faces = DeepFace.extract_faces(
            img_path=selfie_path,
            detector_backend=detector_backend,
            enforce_detection=True,
            anti_spoofing=True,
        )
        if not faces:
            return None, "Anti-spoofing returned no face result"
        return faces[0].get("is_real"), None
    except Exception as error:
        return None, f"Anti-spoofing unavailable: {error}"


def _verify_crops(document_crop: str, selfie_crop: str, preferred_model: str) -> dict:
    """Compare already-detected face crops; skip a second, unreliable detection pass."""
    models = [preferred_model]
    if preferred_model != "VGG-Face":
        models.append("VGG-Face")

    failures = []
    for model_name in models:
        try:
            result = DeepFace.verify(
                img1_path=document_crop,
                img2_path=selfie_crop,
                model_name=model_name,
                detector_backend="skip",
                enforce_detection=False,
                anti_spoofing=False,
            )
            result["model"] = result.get("model", model_name)
            return result
        except Exception as error:
            failures.append(f"{model_name}: {error}")
    raise RuntimeError("Face comparison failed. " + "; ".join(failures))


def verify_face_match(
    document_path: str,
    live_photo_path: str,
    session_id: str = None,
    passport_number: str = None,
) -> dict:
    """Compare a strictly detected document portrait to a strictly detected selfie."""
    flags = []
    temporary_files = []
    identity_cross_check = {
        "duplicate_detected": False,
        "matched_session_id": None,
        "matched_passport_number": None,
        "similarity_score": None,
    }

    arcface_weights = os.path.expanduser("~/.deepface/weights/arcface_weights.h5")
    has_arcface = os.path.exists(arcface_weights) and os.path.getsize(arcface_weights) > 100_000_000
    model_name = "ArcFace" if has_arcface else "VGG-Face"

    try:
        document_crop, document_detection = _extract_face_crop(document_path, "document")
        temporary_files.append(document_crop)
        selfie_crop, selfie_detection = _extract_face_crop(live_photo_path, "selfie")
        temporary_files.append(selfie_crop)

        liveness, liveness_warning = _liveness_check(live_photo_path, selfie_detection["backend"])
        if liveness_warning:
            flags.append("ANTI_SPOOF_UNAVAILABLE")

        try:
            embedding = DeepFace.represent(
                img_path=selfie_crop,
                model_name=model_name,
                detector_backend="skip",
                enforce_detection=False,
            )[0]["embedding"]
            if session_id and passport_number:
                from modules.audit_logger import find_similar_identity, store_embedding

                similar = find_similar_identity(embedding, exclude_passport=passport_number)
                if similar:
                    identity_cross_check = {
                        "duplicate_detected": True,
                        "matched_session_id": similar["matched_session"],
                        "matched_passport_number": similar["matched_passport"],
                        "similarity_score": similar["similarity"],
                    }
                    flags.append("MULTIPLE_IDENTITY_SUSPECTED")
                store_embedding(session_id, passport_number, embedding)
        except Exception as error:
            flags.append(f"EMBEDDING_ERROR: {error}")

        result = _verify_crops(document_crop, selfie_crop, model_name)
        distance = float(result.get("distance", 0))
        threshold = float(result.get("threshold", 0.68))
        confidence = result.get("confidence", 0)
        if not confidence and threshold > 0:
            confidence = max(0, (1 - distance / threshold)) * 100

        response = {
            "verified": bool(result.get("verified", False)),
            "distance": round(distance, 4),
            "confidence": round(float(confidence), 2),
            "threshold": threshold,
            "model": result.get("model", model_name),
            "detector": "strict-crop",
            "is_real": liveness,
            "flags": flags,
            "identity_cross_check": identity_cross_check,
            "detection": {
                "document": document_detection,
                "selfie": selfie_detection,
            },
        }
        if liveness_warning:
            response["warning"] = liveness_warning
        return response
    except Exception as error:
        return {
            "verified": False,
            "distance": None,
            "confidence": 0.0,
            "is_real": None,
            "error": f"Face verification failed: {error}",
            "flags": flags,
            "identity_cross_check": identity_cross_check,
        }
    finally:
        for path in temporary_files:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

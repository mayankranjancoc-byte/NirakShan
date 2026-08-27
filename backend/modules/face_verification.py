"""Module 4: strict document-portrait and selfie face verification."""

import os
import shutil
import tempfile
import logging

import cv2
import numpy as np
from deepface import DeepFace as FaceBiometrics

logger = logging.getLogger(__name__)

# A face smaller than this in the original image is too low-detail for a
# reliable biometric comparison. Faces between MIN and WARN are accepted
# but emit a LOW_QUALITY_PORTRAIT flag.
MIN_FACE_SIDE_PX = 80
WARN_FACE_SIDE_PX = 112
# OpenCV is already bundled with FaceBiometrics and avoids an unexpected first-run
# RetinaFace download. RetinaFace remains a strict fallback for harder images.
DETECTION_BACKENDS = ("opencv", "retinaface")


def _face_area(face: dict) -> int:
    area = face.get("facial_area", {})
    return int(area.get("w", 0)) * int(area.get("h", 0))


def _extract_face_crop(
    image_path: str, label: str
) -> tuple[str, dict, bool]:
    """Strictly detect the largest usable face and write an aligned temp crop.

    Returns (crop_path, detection_info, low_quality_warning).
    """
    failures = []
    for backend in DETECTION_BACKENDS:
        try:
            faces = FaceBiometrics.extract_faces(
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
            face_w = int(area.get("w", 0))
            face_h = int(area.get("h", 0))
            low_quality = min(face_w, face_h) < WARN_FACE_SIDE_PX

            return handle.name, {
                "backend": backend,
                "faces_detected": len(faces),
                "original_face_size": {"width": face_w, "height": face_h},
            }, low_quality
        except Exception as error:
            failures.append(f"{backend}: {error}")

    raise ValueError(
        f"Could not reliably detect a usable face in the {label} image. "
        "Use a sharper, front-facing image with one clearly visible face. "
        f"Detection details: {'; '.join(failures)}"
    )


def extract_face_from_document(image_path: str, output_path: str = None) -> str:
    """Extract a real detected portrait from a document; never use the whole page."""
    crop_path, _, _ = _extract_face_crop(image_path, "document")
    if output_path is None:
        return crop_path
    try:
        shutil.move(crop_path, output_path)  # M11: shutil.move handles cross-device
        return output_path
    except Exception:
        if os.path.exists(crop_path):
            os.remove(crop_path)
        raise


def _liveness_check(
    selfie_crop_path: str, detector_backend: str
) -> tuple[bool | None, str | None]:
    """Run anti-spoofing on the ALREADY CROPPED selfie face (H10 fix: same
    face selection as biometric match, not a second independent detection)."""
    try:
        # Pass the crop directly with skip so we assess the exact same face
        # used for matching — avoids a bystander being checked instead.
        faces = FaceBiometrics.extract_faces(
            img_path=selfie_crop_path,
            detector_backend="skip",
            enforce_detection=False,
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
    if preferred_model != "Facenet":
        models.append("Facenet")

    failures = []
    for model_name in models:
        try:
            result = FaceBiometrics.verify(
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


def _resolve_face_model() -> str:
    """Try to build each candidate model; return the first available one."""
    for candidate in ("Facenet", "ArcFace"):
        try:
            FaceBiometrics.build_model(candidate)
            return candidate
        except Exception as error:
            logger.warning("Face model %s unavailable: %s", candidate, error)
    raise RuntimeError("No face recognition model available; install face_biometrics weights.")


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

    # H9 fix: resolve the actual model instead of guessing via file path.
    try:
        model_name = _resolve_face_model()
    except RuntimeError as e:
        return {
            "verified": False, "distance": None, "confidence": 0.0,
            "is_real": None, "error": str(e), "flags": flags,
            "identity_cross_check": identity_cross_check,
        }

    try:
        document_crop, document_detection, doc_low_q = _extract_face_crop(document_path, "document")
        temporary_files.append(document_crop)
        selfie_crop, selfie_detection, selfie_low_q = _extract_face_crop(live_photo_path, "selfie")
        temporary_files.append(selfie_crop)

        if doc_low_q:
            flags.append("LOW_QUALITY_PORTRAIT_DOCUMENT: Document portrait is small; result may be less reliable")
        if selfie_low_q:
            flags.append("LOW_QUALITY_PORTRAIT_SELFIE: Selfie is small; result may be less reliable")

        # H10 fix: run liveness on the already-extracted selfie crop,
        # not on the original image with a fresh independent detection.
        liveness, liveness_warning = _liveness_check(selfie_crop, selfie_detection["backend"])
        if liveness_warning:
            flags.append("ANTI_SPOOF_UNAVAILABLE")

        try:
            embedding = FaceBiometrics.represent(
                img_path=selfie_crop,
                model_name=model_name,
                detector_backend="skip",
                enforce_detection=False,
            )[0]["embedding"]

            # C6/C7/C8 fix: always store + always search, keyed by session_id.
            # passport_number may be None when MRZ failed — that’s fine.
            if session_id:
                from modules.audit_logger import find_similar_identity, store_embedding

                try:
                    similar = find_similar_identity(
                        embedding,
                        exclude_session=session_id,
                        model_name=model_name,
                    )
                    if similar:
                        identity_cross_check = {
                            "duplicate_detected": True,
                            "matched_session_id": similar["matched_session"],
                            "matched_passport_number": similar["matched_passport"],
                            "similarity_score": similar["similarity"],
                        }
                        flags.append("MULTIPLE_IDENTITY_SUSPECTED")
                    store_embedding(session_id, passport_number, model_name, embedding)
                except Exception as dedup_error:
                    logger.exception("Identity dedup failed")
                    flags.append(f"DEDUP_UNAVAILABLE: {dedup_error}")
        except Exception as error:
            flags.append(f"EMBEDDING_ERROR: {error}")

        result = _verify_crops(document_crop, selfie_crop, model_name)
        distance = float(result.get("distance", 0))
        threshold = float(result.get("threshold", 0.68))
        # M10 fix: label confidence as uncalibrated to avoid misleading officers.
        raw_distance_ratio = max(0.0, 1.0 - distance / threshold) if threshold > 0 else 0.0

        response = {
            "verified": bool(result.get("verified", False)),
            "distance": round(distance, 4),
            "confidence_uncalibrated": round(raw_distance_ratio * 100, 2),
            "confidence": round(raw_distance_ratio * 100, 2),  # kept for UI compatibility
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

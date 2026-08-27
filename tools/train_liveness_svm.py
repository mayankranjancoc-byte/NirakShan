"""
Training script for the screen-replay SVM (Part A of Document Liveness).

Usage:
    cd doc-screening-prototype/backend
    .venv/Scripts/python ../tools/train_liveness_svm.py ^
        --genuine path/to/genuine_photos/ ^
        --replay  path/to/screen_photos/ ^
        --out     models/screen_replay_svm.joblib

Image collection guide:
    genuine/ — Real physical document photographed directly
    replay/  — Same document shown on a phone/laptop screen and re-photographed,
               plus printed photocopies re-photographed
    Aim for ≥ 20 images per class; 50+ per class for reliable SVM training.
"""

import os
import sys
import argparse
import numpy as np

# Add backend dir so we can import feature extractor
_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
_VENDOR = os.path.join(_BACKEND, "vendor", "moire_detector")
for _p in (_BACKEND, _VENDOR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from features import extract_features  # noqa: E402


def load_dataset(genuine_dir: str, replay_dir: str):
    X, y = [], []
    _EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def _collect(directory, label):
        for fn in os.listdir(directory):
            if os.path.splitext(fn)[1].lower() not in _EXTS:
                continue
            feat = extract_features(os.path.join(directory, fn))
            if feat is not None:
                X.append(feat)
                y.append(label)
            else:
                print(f"  [skip] {fn} — could not extract features")

    print(f"Loading genuine images from {genuine_dir}...")
    _collect(genuine_dir, label=0)
    print(f"Loading replay images from {replay_dir}...")
    _collect(replay_dir, label=1)

    return np.array(X), np.array(y)


def main():
    parser = argparse.ArgumentParser(description="Train screen-replay SVM")
    parser.add_argument("--genuine", required=True, help="Dir of genuine document images")
    parser.add_argument("--replay",  required=True, help="Dir of screen/print replay images")
    parser.add_argument("--out",     required=True, help="Output path for .joblib model")
    parser.add_argument("--cv",      type=int, default=5, help="Cross-validation folds (default 5)")
    args = parser.parse_args()

    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib

    X, y = load_dataset(args.genuine, args.replay)
    print(f"\nDataset: {len(X)} samples ({sum(y == 0)} genuine, {sum(y == 1)} replay)")

    if len(X) < 10:
        print("ERROR: Need at least 10 images total. Collect more samples.")
        sys.exit(1)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True)),
    ])

    param_grid = {
        "svm__C": [1, 10, 100],
        "svm__gamma": ["scale", "auto"],
    }

    print(f"\nRunning {args.cv}-fold cross-validated grid search...")
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
    gs = GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=1)
    gs.fit(X, y)

    print(f"\nBest params: {gs.best_params_}")
    print(f"Best CV F1:  {gs.best_score_:.4f}")

    best = gs.best_estimator_
    scores = cross_val_score(best, X, y, cv=cv, scoring="accuracy")
    print(f"CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    # Final full-fit evaluation
    best.fit(X, y)
    y_pred = best.predict(X)
    print("\nFull-training classification report:")
    print(classification_report(y, y_pred, target_names=["genuine", "replay"]))
    print("Confusion matrix:")
    print(confusion_matrix(y, y_pred))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    joblib.dump(best, args.out)
    print(f"\nSaved model to {args.out}")
    print("To use: copy this file to backend/models/screen_replay_svm.joblib")


if __name__ == "__main__":
    main()

import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


def _get_resample_method():
    return getattr(Image, "Resampling", Image).BILINEAR


def extract_handcrafted_features(image_path: str, image_size: int = 16, hist_bins: int = 16) -> np.ndarray:
    """Convert an image into a compact handcrafted feature vector."""
    with Image.open(image_path) as img:
        gray = img.convert("L")
        gray = gray.resize((image_size, image_size), resample=_get_resample_method())
        arr = np.asarray(gray, dtype=np.float32) / 255.0

    flat = arr.reshape(-1)
    hist, _ = np.histogram(arr, bins=hist_bins, range=(0.0, 1.0))
    stats = np.array(
        [
            float(arr.mean()),
            float(arr.std()),
            float(arr.min()),
            float(arr.max()),
            float(np.percentile(arr, 25)),
            float(np.percentile(arr, 75)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([flat, hist.astype(np.float32), stats])


def infer_label_from_mask(mask_path: str) -> int:
    """Infer a binary label from a grayscale mask image.

    The dataset uses separate image folders for images and labels. Since the
    label files are image masks, this helper turns any visible foreground into
    a positive class and an all-black label into a negative class.
    """
    with Image.open(mask_path) as img:
        mask = np.asarray(img.convert("L"), dtype=np.float32)

    if mask.size == 0:
        return 0

    foreground_ratio = np.count_nonzero(mask) / mask.size
    return int(foreground_ratio > 0.001)


def list_image_ids(image_dir: str) -> List[str]:
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    files = [
        os.path.splitext(name)[0]
        for name in os.listdir(image_dir)
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]
    return sorted(files)


def load_image_dataset(image_dir: str, label_dir: str, max_samples: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load image features and binary labels from paired image/label directories."""
    image_ids = list_image_ids(image_dir)
    label_ids = set(list_image_ids(label_dir))

    shared_ids = [image_id for image_id in image_ids if image_id in label_ids]
    if not shared_ids:
        raise ValueError(f"No matching image/label pairs found between {image_dir} and {label_dir}")

    if max_samples is not None:
        shared_ids = shared_ids[:max_samples]

    feature_rows = []
    labels = []
    for image_id in shared_ids:
        image_path = os.path.join(image_dir, f"{image_id}.png")
        label_path = os.path.join(label_dir, f"{image_id}.png")

        if not os.path.exists(image_path):
            image_path = os.path.join(image_dir, f"{image_id}.jpg")
        if not os.path.exists(label_path):
            label_path = os.path.join(label_dir, f"{image_id}.jpg")

        if not os.path.exists(image_path) or not os.path.exists(label_path):
            continue

        feature_rows.append(extract_handcrafted_features(image_path))
        labels.append(infer_label_from_mask(label_path))

    if not feature_rows:
        raise ValueError("No image features were loaded from the provided dataset directories")

    return np.vstack(feature_rows), np.asarray(labels, dtype=np.int64)


def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )


def train_and_evaluate(
    image_dir: str,
    label_dir: str,
    model_dir: str,
    max_samples: int | None = None,
    test_size: float = 0.2,
) -> Dict[str, float]:
    os.makedirs(model_dir, exist_ok=True)

    X, y = load_image_dataset(image_dir, label_dir, max_samples=max_samples)
    stratify = y if np.unique(y).size > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=stratify,
        random_state=42,
    )

    model = build_model()
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    roc_auc = float(roc_auc_score(y_test, probs)) if np.unique(y_test).size > 1 else float("nan")
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "roc_auc": roc_auc,
    }

    config = {"image_size": 16, "hist_bins": 16}
    joblib.dump(model, os.path.join(model_dir, "image_model.joblib"))
    joblib.dump(config, os.path.join(model_dir, "image_feature_config.joblib"))

    print("Validation accuracy:", metrics["accuracy"])
    print("Validation ROC-AUC:", metrics["roc_auc"])
    print("\nClassification report:")
    labels = [0, 1] if np.unique(y_test).size > 1 else [0]
    print(classification_report(y_test, preds, labels=labels, target_names=["Negative", "Positive"], zero_division=0))

    return metrics

import argparse
import os
from typing import List

import joblib
import numpy as np
import pandas as pd

from image_feature_pipeline import extract_handcrafted_features, list_image_ids


def predict_from_directory(model_dir: str, image_dir: str, output_csv: str | None = None) -> pd.DataFrame:
    model = joblib.load(os.path.join(model_dir, "image_model.joblib"))
    config = joblib.load(os.path.join(model_dir, "image_feature_config.joblib"))

    image_ids = list_image_ids(image_dir)
    feature_rows = []
    loaded_ids = []
    for image_id in image_ids:
        image_path = os.path.join(image_dir, f"{image_id}.png")
        if not os.path.exists(image_path):
            image_path = os.path.join(image_dir, f"{image_id}.jpg")
        if not os.path.exists(image_path):
            continue
        loaded_ids.append(image_id)
        feature_rows.append(extract_handcrafted_features(image_path, image_size=config.get("image_size", 16), hist_bins=config.get("hist_bins", 16)))

    if not feature_rows:
        raise ValueError(f"No image files were found in {image_dir}")

    features = np.vstack(feature_rows)
    probs = model.predict_proba(features)[:, 1]
    labels = (probs >= 0.5).astype(int)

    results = pd.DataFrame({"image_id": loaded_ids, "probability": probs, "predicted_label": labels})
    if output_csv:
        results.to_csv(output_csv, index=False)
        print(f"Saved predictions to {output_csv}")
    else:
        print(results.head().to_string(index=False))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a directory of images with the handcrafted feature classifier.")
    parser.add_argument("--model-dir", type=str, default="model_artifacts", help="Directory containing the trained image model artifacts.")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory containing the images to score.")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional CSV path for the prediction results.")
    args = parser.parse_args()

    predict_from_directory(args.model_dir, args.image_dir, args.output_csv)


if __name__ == "__main__":
    main()

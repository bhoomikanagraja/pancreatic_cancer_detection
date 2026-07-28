import argparse
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from data_utils import load_data, get_target_label, build_preprocessor, preprocess_features


def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=15,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )


def main(data_path: str, model_dir: str, threshold: float = 0.2):
    os.makedirs(model_dir, exist_ok=True)

    df = load_data(data_path)
    y = get_target_label(df)
    preprocessor = build_preprocessor()
    X = preprocess_features(df, preprocessor, fit=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = build_model()
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_val)[:, 1]
    pred_labels = (probs >= threshold).astype(int)

    print("\nValidation accuracy:", accuracy_score(y_val, pred_labels))
    print(f"Validation ROC-AUC: {roc_auc_score(y_val, probs):.4f}")
    print(f"\nUsing decision threshold: {threshold}")
    print("\nClassification report:")
    print(classification_report(y_val, pred_labels, target_names=["No Cancer", "Cancer"], zero_division=0))

    model_path = os.path.join(model_dir, "pancreatic_cancer_model.joblib")
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, os.path.join(model_dir, "preprocessor.joblib"))

    print(f"Model saved to {model_path}")
    print(f"Preprocessor saved to {os.path.join(model_dir, 'preprocessor.joblib')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a pancreatic cancer detection model.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="archive/healthcare_dataset.csv",
        help="Path to the CSV dataset.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="model_artifacts",
        help="Directory where the trained model and preprocessing artifacts will be saved.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Probability threshold for predicting the positive cancer class.",
    )
    args = parser.parse_args()

    main(args.data_path, args.model_dir, args.threshold)

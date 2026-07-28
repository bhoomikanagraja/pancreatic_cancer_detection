import argparse

from image_feature_pipeline import train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a handcrafted-image classifier for the provided image/label folders.")
    parser.add_argument("--image-dir", type=str, default="archive/Dataset/X_train", help="Directory containing the training images.")
    parser.add_argument("--label-dir", type=str, default="archive/Dataset/Y_train", help="Directory containing the corresponding label images.")
    parser.add_argument("--model-dir", type=str, default="model_artifacts", help="Directory to save the trained image model artifacts.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit for the number of image/label pairs to load for a quick run.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of samples to reserve for validation.")
    args = parser.parse_args()

    train_and_evaluate(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        model_dir=args.model_dir,
        max_samples=args.max_samples,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()

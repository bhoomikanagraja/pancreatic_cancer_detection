import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from image_feature_pipeline import extract_handcrafted_features, infer_label_from_mask, load_image_dataset


class ImageFeaturePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="image-feature-test-", dir=".")
        self.image_dir = Path(self.temp_dir) / "images"
        self.label_dir = Path(self.temp_dir) / "labels"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.label_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_extract_handcrafted_features_returns_expected_shape(self) -> None:
        img_path = self.image_dir / "sample.png"
        Image.fromarray((255 * __import__("numpy").ones((8, 8), dtype="uint8")).astype("uint8")).save(img_path)

        features = extract_handcrafted_features(str(img_path))
        self.assertEqual(features.shape[0], 16 * 16 + 16 + 6)

    def test_infer_label_from_mask_detects_foreground(self) -> None:
        mask_path = self.label_dir / "sample.png"
        Image.fromarray(__import__("numpy").zeros((8, 8), dtype="uint8")).save(mask_path)
        self.assertEqual(infer_label_from_mask(str(mask_path)), 0)

        Image.fromarray(__import__("numpy").full((8, 8), 255, dtype="uint8")).save(mask_path)
        self.assertEqual(infer_label_from_mask(str(mask_path)), 1)

    def test_load_image_dataset_uses_matching_pairs(self) -> None:
        image_path = self.image_dir / "pair.png"
        label_path = self.label_dir / "pair.png"
        Image.fromarray(__import__("numpy").zeros((8, 8), dtype="uint8")).save(image_path)
        Image.fromarray(__import__("numpy").full((8, 8), 255, dtype="uint8")).save(label_path)

        X, y = load_image_dataset(str(self.image_dir), str(self.label_dir), max_samples=None)
        self.assertEqual(X.shape[0], 1)
        self.assertEqual(y.shape[0], 1)
        self.assertEqual(int(y[0]), 1)


if __name__ == "__main__":
    unittest.main()

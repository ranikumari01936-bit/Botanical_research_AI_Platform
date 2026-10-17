import json
import unittest
from pathlib import Path

from PIL import Image

from flower_classifier import predict_flower


PROJECT_DIR = Path(__file__).resolve().parent
EXPECTED_CLASSES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]


class FlowerClassifierTests(unittest.TestCase):
    def test_model_uses_the_preserved_five_class_mapping(self):
        classes_path = PROJECT_DIR / "plant_ml" / "models" / "flower_classes.json"
        with classes_path.open(encoding="utf-8") as file:
            self.assertEqual(json.load(file), EXPECTED_CLASSES)

    def test_prediction_uses_one_of_the_existing_classes(self):
        with Image.new("RGB", (320, 240), color=(71, 123, 85)) as image:
            result = predict_flower(image.copy())

        self.assertIn(result["flower"], EXPECTED_CLASSES)
        self.assertGreaterEqual(result["confidence"], 0)
        self.assertLessEqual(result["confidence"], 100)


if __name__ == "__main__":
    unittest.main()
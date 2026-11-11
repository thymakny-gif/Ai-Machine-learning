"""
Predict whether an image contains a Car, Motorbike, or Bicycle.
Usage:
    python predict.py path/to/your_image.jpg
"""
from pathlib import Path
import json
import sys
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "vehicle_cnn.keras"
LABEL_PATH = BASE_DIR / "models" / "class_names.json"
IMG_SIZE = (200, 200)

if len(sys.argv) < 2:
    raise SystemExit("Usage: python predict.py path/to/image.jpg")
image_path = Path(sys.argv[1])

if not image_path.is_file():
    raise SystemExit(f"Image not found: {image_path}")
if not MODEL_PATH.is_file() or not LABEL_PATH.is_file():
    raise SystemExit("Trained model not found. Run `python train.py` first.")

model = tf.keras.models.load_model(MODEL_PATH)
class_names = json.loads(LABEL_PATH.read_text(encoding="utf-8"))

image = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
image_array = tf.keras.utils.img_to_array(image)
image_array = tf.expand_dims(image_array, 0)

predictions = model.predict(image_array, verbose=0)[0]
best_index = int(tf.argmax(predictions))
print(f"Prediction: {class_names[best_index]}")
print(f"Confidence: {predictions[best_index] * 100:.2f}%")
print("\nAll class probabilities:")
for name, score in zip(class_names, predictions):
    print(f"  {name}: {score * 100:.2f}%")

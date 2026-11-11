"""
Vehicle Image Classification using CNN
Classes currently included: Car, Motorbike
Run: python train.py
"""
from pathlib import Path
import json
import tensorflow as tf
from tensorflow.keras import layers, models

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Dataset"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

IMG_SIZE = (200, 200)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42
CLASS_NAMES = ["Car", "Motorbike"]

def count_images(folder):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in extensions)

for split in ("train", "validation", "test"):
    for class_name in CLASS_NAMES:
        folder = DATA_DIR / split / class_name
        folder.mkdir(parents=True, exist_ok=True)
        if count_images(folder) == 0:
            raise SystemExit(
                f"No images found in: {folder}\n"
                "Add images to each class folder before training. "
                "See README.md for the required structure."
            )

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=CLASS_NAMES,
    shuffle=True,
    seed=SEED,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "validation",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=CLASS_NAMES,
    shuffle=False,
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=CLASS_NAMES,
    shuffle=False,
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(500).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
])

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    data_augmentation,
    layers.Rescaling(1.0 / 255),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(len(CLASS_NAMES), activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_DIR / "vehicle_cnn.keras",
        monitor="val_accuracy",
        save_best_only=True,
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
print(f"\nTest accuracy: {test_accuracy * 100:.2f}%")
print(f"Test loss: {test_loss:.4f}")

# Save class names in the same order used by the model.
(MODEL_DIR / "class_names.json").write_text(
    json.dumps(CLASS_NAMES, indent=2), encoding="utf-8"
)

# Save training history for later graphing.
import csv
history_path = BASE_DIR / "history.csv"
history_keys = list(history.history.keys())
with history_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch"] + history_keys)
    for i in range(len(history.history["loss"])):
        writer.writerow([i + 1] + [history.history[k][i] for k in history_keys])

print(f"\nSaved model to: {MODEL_DIR / 'vehicle_cnn.keras'}")
print(f"Saved training history to: {history_path}")

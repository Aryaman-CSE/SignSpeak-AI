import json
import shutil
import numpy as np
import tensorflow as tf

from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets" / "asl_alphabet"
MODEL_DIR = ROOT / "models" / "asl_mobilenetv2"

ORIGINAL = MODEL_DIR / "final_model.h5"
OUTPUT = MODEL_DIR / "improved_model.h5"
BEST = MODEL_DIR / "improved_best.weights.h5"

with open(MODEL_DIR / "class_indices.json") as f:
    class_indices = json.load(f)

print("Loading original classifier...")
model = tf.keras.models.load_model(ORIGINAL)

train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.12,
    brightness_range=[0.85, 1.15]
)

val_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train = train_gen.flow_from_directory(
    str(DATA / "train"),
    target_size=(160, 160),
    batch_size=32,
    class_mode="categorical"
)

val = val_gen.flow_from_directory(
    str(DATA / "validation"),
    target_size=(160, 160),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# Fine-tune the final MobileNetV2 layers.
base = model.layers[1]
base.trainable = True

for layer in base.layers:
    layer.trainable = False

for layer in base.layers[-30:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

# Give extra importance to classes with weak recall.
weights = {i: 1.0 for i in range(len(class_indices))}
for letter, weight in {
    "U": 1.6,
    "E": 1.5,
    "N": 1.4,
    "W": 1.35,
    "T": 1.3,
    "G": 1.3
}.items():
    if letter in class_indices:
        weights[class_indices[letter]] = weight

model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    ModelCheckpoint(
        str(BEST),
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=1e-7,
        verbose=1
    )
]

print("\nStarting targeted fine-tuning...")
model.fit(
    train,
    validation_data=val,
    epochs=3,
    class_weight=weights,
    callbacks=callbacks
)

if BEST.exists():
    model.load_weights(BEST)

model.save(OUTPUT)

print("\nFine-tuning completed.")
print("Improved model:", OUTPUT)
print("Original model preserved:", ORIGINAL)

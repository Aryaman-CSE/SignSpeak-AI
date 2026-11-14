import random
import shutil
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets" / "asl_alphabet"
WEBCAM = ROOT / "datasets" / "webcam_samples"
MODEL_DIR = ROOT / "models" / "asl_mobilenetv2"

SOURCE_MODEL = MODEL_DIR / "webcam_finetuned_model.h5"
OUTPUT_MODEL = MODEL_DIR / "webcam_finetuned_v2.h5"
BEST_WEIGHTS = MODEL_DIR / "webcam_v2_best.weights.h5"
TEMP = MODEL_DIR / "webcam_training_data_v2"

webcam_letters = [
    "E", "K", "G", "I", "H", "P", "S",
    "M", "N", "D", "Q", "T", "X",
    "U", "V", "W", "R"
]

random.seed(42)
tf.random.set_seed(42)

if not SOURCE_MODEL.exists():
    raise FileNotFoundError(f"Source model missing: {SOURCE_MODEL}")

if not (DATA / "train").is_dir():
    raise FileNotFoundError(DATA / "train")

letters = sorted(
    p.name for p in (DATA / "train").iterdir() if p.is_dir()
)

if len(letters) != 26:
    raise RuntimeError(f"Expected 26 classes, found {len(letters)}: {letters}")

for letter in webcam_letters:
    folder = WEBCAM / letter
    if not folder.is_dir():
        raise RuntimeError(f"Missing webcam folder: {folder}")

    count = len(list(folder.glob("*.jpg")))
    if count < 20:
        raise RuntimeError(f"Insufficient webcam images for {letter}: {count}")

print("Static classes:", letters)
print("Webcam-enhanced classes:", webcam_letters)
print("Loading previous working model...")

model = tf.keras.models.load_model(SOURCE_MODEL)

if model.output_shape[-1] != len(letters):
    raise RuntimeError(
        f"Model has {model.output_shape[-1]} outputs, "
        f"but dataset has {len(letters)} classes."
    )

# Rebuild only the new temporary training directory.
if TEMP.exists():
    shutil.rmtree(TEMP)

for split in ["train", "validation"]:
    for letter in letters:
        (TEMP / split / letter).mkdir(parents=True, exist_ok=True)

print("\nPreparing original-data replay...")

# Retain examples from every original class.
for letter in letters:
    files = list((DATA / "train" / letter).glob("*"))
    random.shuffle(files)

    for i, f in enumerate(files[:150]):
        if f.is_file():
            shutil.copy2(
                f, TEMP / "train" / letter / f"original_{i}_{f.name}"
            )

    val_files = list((DATA / "validation" / letter).glob("*"))
    random.shuffle(val_files)

    for i, f in enumerate(val_files[:30]):
        if f.is_file():
            shutil.copy2(
                f, TEMP / "validation" / letter / f"original_{i}_{f.name}"
            )

print("\nAdding all webcam samples...")

for letter in webcam_letters:
    files = list((WEBCAM / letter).glob("*.jpg"))
    random.shuffle(files)

    split_at = int(len(files) * 0.8)
    train_files = files[:split_at]
    val_files = files[split_at:]

    for i, f in enumerate(train_files):
        shutil.copy2(
            f, TEMP / "train" / letter / f"webcam_{i}_{f.name}"
        )

    for i, f in enumerate(val_files):
        shutil.copy2(
            f, TEMP / "validation" / letter / f"webcam_{i}_{f.name}"
        )

    print(
        f"{letter}: {len(train_files)} training, "
        f"{len(val_files)} validation images"
    )

print("\nLoading image generators...")

train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=8,
    width_shift_range=0.06,
    height_shift_range=0.06,
    zoom_range=0.10,
    brightness_range=[0.90, 1.10]
)

val_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train = train_gen.flow_from_directory(
    str(TEMP / "train"),
    target_size=(160, 160),
    batch_size=32,
    classes=letters,
    class_mode="categorical"
)

validation = val_gen.flow_from_directory(
    str(TEMP / "validation"),
    target_size=(160, 160),
    batch_size=32,
    classes=letters,
    class_mode="categorical",
    shuffle=False
)

# Fine-tune the final MobileNetV2 feature layers.
base = model.layers[1]
base.trainable = True

for layer in base.layers:
    layer.trainable = False

for layer in base.layers[-20:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=0.000001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    ModelCheckpoint(
        str(BEST_WEIGHTS),
        monitor="val_loss",
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

print("\nStarting second-round webcam fine-tuning...")

model.fit(
    train,
    validation_data=validation,
    epochs=5,
    callbacks=callbacks
)

if BEST_WEIGHTS.exists():
    model.load_weights(BEST_WEIGHTS)

model.save(OUTPUT_MODEL)

print("\nFINE-TUNING COMPLETE")
print("New model:", OUTPUT_MODEL)
print("Starting model preserved:", SOURCE_MODEL)
print("Original improved model preserved.")

import os, json, math, shutil, random
from pathlib import Path
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets" / "asl_alphabet"
WEBCAM = ROOT / "datasets" / "webcam_samples"
CROPS = ROOT / "datasets" / "webcam_live_crops"
MODEL_DIR = ROOT / "models" / "asl_mobilenetv2"
SOURCE = MODEL_DIR / "webcam_finetuned_v2.h5"
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BATCH_ORIGINAL = 28
BATCH_WEBCAM = 4
EPOCHS = 2
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def image_count(folder):
    if not folder.exists():
        return 0
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXT)

for split in ("train", "validation", "test", "official_test"):
    if not (DATA / split).is_dir():
        if split == "official_test":
            continue
        raise FileNotFoundError(DATA / split)

if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)

class_file = MODEL_DIR / "class_indices.json"
class_map = json.loads(class_file.read_text(encoding="utf-8"))
expected_map = {letter: i for i, letter in enumerate(LETTERS)}
if class_map != expected_map:
    raise RuntimeError("Model class order is not A-Z. Stopping to prevent incorrect labels.")

print("TensorFlow:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))
print("Loading existing production v2 model...")
model = tf.keras.models.load_model(SOURCE)
print("Production model loaded; source file will not be modified.")

# Prefer existing live-crop images if they are already present.
crop_train = CROPS / "train"
crop_val = CROPS / "validation"
use_crops = image_count(crop_train) >= 300 and image_count(crop_val) >= 50

# If no usable live-crop set exists, split the already-collected webcam images.
# No new photos are collected.
if use_crops:
    webcam_train_dir = crop_train
    webcam_val_dir = crop_val
    print("Using existing webcam_live_crops.")
else:
    print("Using existing webcam_samples; making an 80/20 split.")
    raw_split = MODEL_DIR / "webcam_existing_images_split"
    if raw_split.exists():
        shutil.rmtree(raw_split)
    webcam_train_dir = raw_split / "train"
    webcam_val_dir = raw_split / "validation"

    for split_dir in (webcam_train_dir, webcam_val_dir):
        for letter in LETTERS:
            (split_dir / letter).mkdir(parents=True, exist_ok=True)

    for letter in LETTERS:
        folder = WEBCAM / letter
        files = sorted(
            [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXT]
        ) if folder.is_dir() else []
        if not files:
            continue
        n_val = max(1, int(len(files) * 0.20))
        train_files = files[:-n_val]
        val_files = files[-n_val:]
        for i, p in enumerate(train_files):
            shutil.copy2(p, webcam_train_dir / letter / f"{i}_{p.name}")
        for i, p in enumerate(val_files):
            shutil.copy2(p, webcam_val_dir / letter / f"{i}_{p.name}")

for split_dir in (webcam_train_dir, webcam_val_dir):
    for letter in LETTERS:
        (split_dir / letter).mkdir(parents=True, exist_ok=True)

print("Original training images:", image_count(DATA / "train"))
print("Original validation images:", image_count(DATA / "validation"))
print("Webcam training images:", image_count(webcam_train_dir))
print("Webcam validation images:", image_count(webcam_val_dir))

if image_count(webcam_train_dir) == 0 or image_count(webcam_val_dir) == 0:
    raise RuntimeError("No webcam training/validation images found.")

aug = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=6,
    width_shift_range=0.04,
    height_shift_range=0.04,
    zoom_range=0.08,
    brightness_range=(0.90, 1.10),
    horizontal_flip=False
)
plain = ImageDataGenerator(preprocessing_function=preprocess_input)

original_train = aug.flow_from_directory(
    str(DATA / "train"), target_size=(160,160),
    batch_size=BATCH_ORIGINAL, classes=LETTERS,
    class_mode="categorical", shuffle=True, seed=SEED
)
original_val = plain.flow_from_directory(
    str(DATA / "validation"), target_size=(160,160),
    batch_size=BATCH_ORIGINAL, classes=LETTERS,
    class_mode="categorical", shuffle=False
)
webcam_train = aug.flow_from_directory(
    str(webcam_train_dir), target_size=(160,160),
    batch_size=BATCH_WEBCAM, classes=LETTERS,
    class_mode="categorical", shuffle=True, seed=SEED
)
webcam_val = plain.flow_from_directory(
    str(webcam_val_dir), target_size=(160,160),
    batch_size=BATCH_WEBCAM, classes=LETTERS,
    class_mode="categorical", shuffle=False
)

if original_train.class_indices != original_val.class_indices:
    raise RuntimeError("Original training/validation class order mismatch.")
if original_train.class_indices != webcam_train.class_indices:
    raise RuntimeError("Webcam/original class order mismatch.")

class MixedSequence(tf.keras.utils.Sequence):
    def __init__(self, original, webcam):
        self.original = original
        self.webcam = webcam
    def __len__(self):
        return len(self.original)
    def __getitem__(self, index):
        xo, yo = self.original[index]
        xw, yw = self.webcam[index % len(self.webcam)]
        return np.concatenate((xo, xw), axis=0), np.concatenate((yo, yw), axis=0)
    def on_epoch_end(self):
        self.original.on_epoch_end()
        self.webcam.on_epoch_end()

train_mix = MixedSequence(original_train, webcam_train)
val_mix = MixedSequence(original_val, webcam_val)

# Evaluate the untouched source model before fine-tuning.
report = {"source_model": str(SOURCE), "epochs_requested": EPOCHS, "baseline": {}, "candidate": {}}
eval_flows = {}
for split in ("test", "official_test"):
    folder = DATA / split
    if folder.is_dir() and image_count(folder) > 0:
        flow = plain.flow_from_directory(
            str(folder), target_size=(160,160), batch_size=32,
            classes=LETTERS, class_mode="categorical", shuffle=False
        )
        eval_flows[split] = flow
        flow.reset()
        report["baseline"][split] = {
            k: float(v) for k, v in model.evaluate(flow, verbose=1, return_dict=True).items()
        }

webcam_val.reset()
report["baseline"]["webcam_validation"] = {
    k: float(v) for k, v in model.evaluate(webcam_val, verbose=1, return_dict=True).items()
}

# Fine-tune only the later MobileNetV2 layers.
base = model.layers[1]
base.trainable = True
for layer in base.layers:
    layer.trainable = False
for layer in base.layers[-20:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-6),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
candidate_path = MODEL_DIR / f"webcam_full_dataset_candidate_{stamp}.h5"
weights_path = MODEL_DIR / f"webcam_full_dataset_best_{stamp}.weights.h5"

callbacks = [
    ModelCheckpoint(str(weights_path), monitor="val_loss",
                    save_best_only=True, save_weights_only=True, verbose=1),
    EarlyStopping(monitor="val_loss", patience=1,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=1, min_lr=1e-7, verbose=1)
]

print("\nSTARTING FULL-DATASET FINE-TUNING")
print("All original training images are included.")
print("Original validation is used for checkpoint selection.")
print("Webcam samples are mixed into every training batch.")
print("Test and official_test remain untouched.")
model.fit(
    train_mix,
    validation_data=val_mix,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

if weights_path.exists():
    model.load_weights(weights_path)

model.save(candidate_path)
print("\nCandidate saved:", candidate_path)

for split, flow in eval_flows.items():
    flow.reset()
    report["candidate"][split] = {
        k: float(v) for k, v in model.evaluate(flow, verbose=1, return_dict=True).items()
    }

webcam_val.reset()
report["candidate"]["webcam_validation"] = {
    k: float(v) for k, v in model.evaluate(webcam_val, verbose=1, return_dict=True).items()
}

report["candidate_model"] = str(candidate_path)
report_path = MODEL_DIR / f"webcam_full_dataset_report_{stamp}.json"
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\n===== FINAL COMPARISON =====")
print(json.dumps(report, indent=2))
print("\nReport:", report_path)
print("Production v2 remains unchanged:", SOURCE)

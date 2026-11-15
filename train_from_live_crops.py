import json
import random
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets" / "asl_alphabet"
LIVE = ROOT / "datasets" / "webcam_live_crops"
MODEL_DIR = ROOT / "models" / "asl_mobilenetv2"
SOURCE_MODEL = MODEL_DIR / "webcam_finetuned_v2.h5"
CLASS_MAP = MODEL_DIR / "class_indices.json"
TRAIN_ROOT = MODEL_DIR / "training_from_live_crops"
OUTPUT_MODEL = MODEL_DIR / "webcam_finetuned_live_crops.h5"
BEST_WEIGHTS = MODEL_DIR / "webcam_live_crops_best.weights.h5"
REPORT = MODEL_DIR / "webcam_live_crops_report.json"
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SIZE = 160
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def files_in(folder):
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])


def make_generator(directory, training, shuffle):
    if training:
        gen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=4,
            width_shift_range=0.025,
            height_shift_range=0.025,
            zoom_range=0.04,
            brightness_range=(0.92, 1.08),
            horizontal_flip=False,
        )
    else:
        gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    return gen.flow_from_directory(
        str(directory), target_size=(SIZE, SIZE), color_mode="rgb",
        batch_size=32, classes=LETTERS, class_mode="categorical",
        shuffle=shuffle, seed=SEED, interpolation="nearest",
    )


def evaluate(model, generator):
    generator.reset()
    loss, acc = model.evaluate(generator, verbose=0)
    generator.reset()
    pred = np.argmax(model.predict(generator, verbose=0), axis=1)
    return float(loss), float(acc), generator.classes.copy(), pred


def main():
    required = [SOURCE_MODEL, CLASS_MAP, DATA / "train", DATA / "validation", DATA / "test", LIVE / "train", LIVE / "validation"]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required path: {path}")
    for path in [TRAIN_ROOT, OUTPUT_MODEL, BEST_WEIGHTS, REPORT]:
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}. Nothing was overwritten.")

    class_map = json.loads(CLASS_MAP.read_text(encoding="utf-8"))
    if class_map != {letter: i for i, letter in enumerate(LETTERS)}:
        raise RuntimeError(f"Unexpected class map: {class_map}")

    webcam_counts = {}
    for letter in LETTERS:
        tr_dir, va_dir = LIVE / "train" / letter, LIVE / "validation" / letter
        tr = files_in(tr_dir) if tr_dir.is_dir() else []
        va = files_in(va_dir) if va_dir.is_dir() else []
        webcam_counts[letter] = {"train": len(tr), "validation": len(va)}
        if letter not in "JZ" and (len(tr) < 30 or len(va) < 8):
            raise RuntimeError(f"Insufficient live crops for {letter}: train={len(tr)}, validation={len(va)}. Complete live collection before training.")

    print("Loading production model; it will not be modified.")
    production = tf.keras.models.load_model(SOURCE_MODEL)
    if production.output_shape[-1] != 26:
        raise RuntimeError(f"Expected 26 outputs, got {production.output_shape[-1]}")

    (TRAIN_ROOT / "train").mkdir(parents=True)
    (TRAIN_ROOT / "validation").mkdir(parents=True)
    for split in ("train", "validation"):
        for letter in LETTERS:
            (TRAIN_ROOT / split / letter).mkdir(parents=True, exist_ok=True)

    print("Preparing balanced training folders: 150 original + live crops per letter.")
    for letter in LETTERS:
        source_train = files_in(DATA / "train" / letter)
        source_val = files_in(DATA / "validation" / letter)
        if len(source_train) < 150 or len(source_val) < 30:
            raise RuntimeError(f"Not enough original images for {letter}: train={len(source_train)}, validation={len(source_val)}")
        random.shuffle(source_train)
        random.shuffle(source_val)
        for i, src in enumerate(source_train[:150]):
            shutil.copy2(src, TRAIN_ROOT / "train" / letter / f"original_{i:03d}{src.suffix.lower()}")
        for i, src in enumerate(source_val[:30]):
            shutil.copy2(src, TRAIN_ROOT / "validation" / letter / f"original_{i:03d}{src.suffix.lower()}")
        for split in ("train", "validation"):
            source_live = LIVE / split / letter
            if source_live.is_dir():
                for i, src in enumerate(files_in(source_live)):
                    shutil.copy2(src, TRAIN_ROOT / split / letter / f"live_{i:04d}.jpg")

    train = make_generator(TRAIN_ROOT / "train", True, True)
    validation = make_generator(TRAIN_ROOT / "validation", False, False)
    test = make_generator(DATA / "test", False, False)
    webcam_validation = make_generator(LIVE / "validation", False, False)

    candidate = tf.keras.models.clone_model(production)
    candidate.set_weights(production.get_weights())
    if len(candidate.layers) < 2:
        raise RuntimeError("Unexpected model architecture; refusing to fine-tune.")
    base = candidate.layers[1]
    base.trainable = True
    for layer in base.layers:
        layer.trainable = False
    for layer in base.layers[-20:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    candidate.compile(optimizer=Adam(learning_rate=0.000001), loss="categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        ModelCheckpoint(str(BEST_WEIGHTS), monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-7, verbose=1),
    ]
    print("Starting live-crop aligned fine-tuning (maximum 5 epochs).")
    candidate.fit(train, validation_data=validation, epochs=5, callbacks=callbacks, verbose=1)
    if not BEST_WEIGHTS.is_file():
        raise RuntimeError("No best checkpoint saved; candidate was not created.")
    candidate.load_weights(BEST_WEIGHTS)
    candidate.save(OUTPUT_MODEL)

    print("Comparing production and candidate on original test + live webcam holdout...")
    p_orig_loss, p_orig_acc, _, _ = evaluate(production, test)
    c_orig_loss, c_orig_acc, _, _ = evaluate(candidate, test)
    p_web_loss, p_web_acc, y, p_web_pred = evaluate(production, webcam_validation)
    c_web_loss, c_web_acc, _, c_web_pred = evaluate(candidate, webcam_validation)
    live_report = classification_report(y, c_web_pred, labels=list(range(26)), target_names=LETTERS, output_dict=True, zero_division=0)
    report = {
        "webcam_sample_counts": webcam_counts,
        "metrics": {
            "production_original_test_accuracy": p_orig_acc,
            "candidate_original_test_accuracy": c_orig_acc,
            "production_live_webcam_validation_accuracy": p_web_acc,
            "candidate_live_webcam_validation_accuracy": c_web_acc,
            "production_original_test_loss": p_orig_loss,
            "candidate_original_test_loss": c_orig_loss,
            "production_live_webcam_validation_loss": p_web_loss,
            "candidate_live_webcam_validation_loss": c_web_loss,
        },
        "candidate_live_webcam_classification_report": live_report,
        "candidate_live_webcam_confusion_matrix": confusion_matrix(y, c_web_pred, labels=list(range(26))).tolist(),
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n================ FINAL COMPARISON ================")
    print(f"Original test | production v2: {p_orig_acc * 100:.2f}%")
    print(f"Original test | live-crop candidate: {c_orig_acc * 100:.2f}%")
    print(f"Live webcam holdout | production v2: {p_web_acc * 100:.2f}%")
    print(f"Live webcam holdout | candidate: {c_web_acc * 100:.2f}%")
    print("Candidate:", OUTPUT_MODEL)
    print("Report:", REPORT)
    if c_web_acc > p_web_acc and c_orig_acc >= p_orig_acc - 0.005:
        print("RESULT: Candidate improved live-crop validation without a large original-test regression. Run a live A/B test before promotion.")
    else:
        print("RESULT: Do NOT promote. Production v2 remains unchanged.")


if __name__ == "__main__":
    main()

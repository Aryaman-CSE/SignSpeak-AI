import json
import random
import shutil
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets" / "asl_alphabet"
WEBCAM = ROOT / "datasets" / "webcam_samples"
MODEL_DIR = ROOT / "models" / "asl_mobilenetv2"

SOURCE_MODEL = MODEL_DIR / "webcam_finetuned_v2.h5"
OUTPUT_MODEL = MODEL_DIR / "webcam_finetuned_crop_corrected.h5"
BEST_WEIGHTS = MODEL_DIR / "webcam_crop_corrected_best.weights.h5"
CROP_ROOT = MODEL_DIR / "webcam_crops_corrected"
TRAIN_ROOT = MODEL_DIR / "webcam_training_data_crop_corrected"
REPORT_PATH = MODEL_DIR / "webcam_crop_corrected_report.json"

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
INPUT_SIZE = 320
CLASSIFIER_SIZE = 160
MAX_HANDS = 2
HAND_CROP_SCALE = 1.75
MIN_HAND_SIZE = 45
MIN_DETECTION_CONFIDENCE = 0.55
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def fail_if_missing(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required path missing: {path}")


def natural_key(path: Path):
    import re
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)]


def extract_hand_like_live(frame, landmarks):
    """Same crop geometry as realtime_signspeak.py::extract_hand."""
    height, width = frame.shape[:2]
    xs = np.asarray([p.x * width for p in landmarks.landmark], dtype=np.float32)
    ys = np.asarray([p.y * height for p in landmarks.landmark], dtype=np.float32)

    left = float(xs.min())
    right = float(xs.max())
    top = float(ys.min())
    bottom = float(ys.max())
    box_width = right - left
    box_height = bottom - top

    if max(box_width, box_height) < MIN_HAND_SIZE:
        return None

    side = max(box_width, box_height) * HAND_CROP_SCALE
    side = max(side, MIN_HAND_SIZE * 2)
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0

    x1 = int(round(center_x - side / 2.0))
    y1 = int(round(center_y - side / 2.0))
    x2 = int(round(center_x + side / 2.0))
    y2 = int(round(center_y + side / 2.0))

    sx1 = max(0, x1)
    sy1 = max(0, y1)
    sx2 = min(width, x2)
    sy2 = min(height, y2)
    if sx2 <= sx1 or sy2 <= sy1:
        return None

    crop = frame[sy1:sy2, sx1:sx2]
    crop = cv2.copyMakeBorder(
        crop,
        max(0, -y1), max(0, y2 - height),
        max(0, -x1), max(0, x2 - width),
        cv2.BORDER_REPLICATE
    )
    crop = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    return crop


def prepare_webcam_crops():
    if CROP_ROOT.exists():
        raise FileExistsError(
            f"Crop directory already exists: {CROP_ROOT}\n"
            "Nothing was overwritten. Rename it before rerunning."
        )

    for split in ("train", "validation"):
        for letter in LETTERS:
            (CROP_ROOT / split / letter).mkdir(parents=True, exist_ok=True)

    hands_api = mp.solutions.hands
    stats = {}
    total_detected = 0
    total_failed = 0

    print("\nCropping webcam samples with the same MediaPipe hand-box logic as live inference...")
    with hands_api.Hands(
        static_image_mode=True,
        max_num_hands=MAX_HANDS,
        model_complexity=0,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE
    ) as tracker:
        for letter in LETTERS:
            folder = WEBCAM / letter
            files = sorted(
                [p for p in folder.iterdir()
                 if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
                key=natural_key
            ) if folder.is_dir() else []

            if not files:
                stats[letter] = {"source": 0, "train_crops": 0, "validation_crops": 0, "failed": 0}
                continue

            split_at = max(1, int(len(files) * 0.80))
            # Keep later sequential captures for validation to reduce near-duplicate leakage.
            groups = {"train": files[:split_at], "validation": files[split_at:]}
            class_stats = {"source": len(files), "train_crops": 0, "validation_crops": 0, "failed": 0}

            for split, split_files in groups.items():
                for image_path in split_files:
                    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    if frame is None:
                        class_stats["failed"] += 1
                        total_failed += 1
                        continue

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    result = tracker.process(rgb)
                    hand_landmarks = result.multi_hand_landmarks or []

                    valid_crops = []
                    for landmarks in hand_landmarks:
                        crop320 = extract_hand_like_live(frame, landmarks)
                        if crop320 is None:
                            continue
                        # Live classifier uses INTER_AREA from 320x320 to 160x160.
                        crop160 = cv2.resize(
                            crop320, (CLASSIFIER_SIZE, CLASSIFIER_SIZE),
                            interpolation=cv2.INTER_AREA
                        )
                        valid_crops.append(crop160)

                    if not valid_crops:
                        class_stats["failed"] += 1
                        total_failed += 1
                        continue

                    for hand_index, crop in enumerate(valid_crops):
                        out = CROP_ROOT / split / letter / f"{image_path.stem}_h{hand_index}.png"
                        if not cv2.imwrite(str(out), crop):
                            raise RuntimeError(f"Could not write crop: {out}")
                        class_stats[f"{split}_crops"] += 1
                        total_detected += 1

            stats[letter] = class_stats
            print(
                f"{letter}: source={class_stats['source']}, "
                f"train_crops={class_stats['train_crops']}, "
                f"validation_crops={class_stats['validation_crops']}, "
                f"no_hand={class_stats['failed']}"
            )

    print(f"\nWebcam crop extraction: {total_detected} crops saved; {total_failed} images had no usable hand crop.")

    required_classes = [letter for letter in LETTERS if (WEBCAM / letter).is_dir()]
    missing_classes = [letter for letter in LETTERS if letter not in required_classes]
    if missing_classes:
        print("No webcam samples for:", ", ".join(missing_classes))

    for letter in required_classes:
        tr = len(list((CROP_ROOT / "train" / letter).glob("*.jpg")))
        va = len(list((CROP_ROOT / "validation" / letter).glob("*.jpg")))
        if tr < 20 or va < 5:
            raise RuntimeError(
                f"Not enough detected webcam crops for {letter}: train={tr}, validation={va}. "
                "Collect more images for this letter before fine-tuning."
            )
    return stats


def prepare_training_folders():
    if TRAIN_ROOT.exists():
        raise FileExistsError(
            f"Training directory already exists: {TRAIN_ROOT}\n"
            "Nothing was overwritten. Rename it before rerunning."
        )

    for split in ("train", "validation"):
        for letter in LETTERS:
            (TRAIN_ROOT / split / letter).mkdir(parents=True, exist_ok=True)

    print("\nAdding original-data replay (150 train and 30 validation images per class)...")
    for letter in LETTERS:
        train_files = sorted(
            [p for p in (DATA / "train" / letter).iterdir()
             if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
            key=natural_key
        )
        val_files = sorted(
            [p for p in (DATA / "validation" / letter).iterdir()
             if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
            key=natural_key
        )
        if len(train_files) < 150 or len(val_files) < 30:
            raise RuntimeError(
                f"Insufficient original data for {letter}: "
                f"train={len(train_files)}, validation={len(val_files)}"
            )
        random.shuffle(train_files)
        random.shuffle(val_files)
        for i, src in enumerate(train_files[:150]):
            shutil.copy2(src, TRAIN_ROOT / "train" / letter / f"original_{i}_{src.name}")
        for i, src in enumerate(val_files[:30]):
            shutil.copy2(src, TRAIN_ROOT / "validation" / letter / f"original_{i}_{src.name}")

    print("\nAdding correctly cropped webcam samples...")
    for letter in LETTERS:
        for split in ("train", "validation"):
            src_dir = CROP_ROOT / split / letter
            for src in src_dir.glob("*.jpg"):
                shutil.copy2(src, TRAIN_ROOT / split / letter / f"webcam_{src.name}")


def make_generator(directory: Path, training: bool, shuffle: bool):
    if training:
        image_gen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=4,
            width_shift_range=0.03,
            height_shift_range=0.03,
            zoom_range=0.05,
            brightness_range=(0.90, 1.10),
            horizontal_flip=False
        )
    else:
        image_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    return image_gen.flow_from_directory(
        str(directory),
        target_size=(CLASSIFIER_SIZE, CLASSIFIER_SIZE),
        color_mode="rgb",
        batch_size=32,
        classes=LETTERS,
        class_mode="categorical",
        shuffle=shuffle,
        seed=RANDOM_SEED,
        interpolation="nearest"
    )


def evaluate(model, generator):
    generator.reset()
    loss, accuracy = model.evaluate(generator, verbose=0)
    generator.reset()
    probs = model.predict(generator, verbose=0)
    preds = np.argmax(probs, axis=1)
    labels = generator.classes.copy()
    return float(loss), float(accuracy), labels, preds


def main():
    for required in (
        SOURCE_MODEL,
        MODEL_DIR / "class_indices.json",
        DATA / "train",
        DATA / "validation",
        DATA / "test",
        WEBCAM
    ):
        fail_if_missing(required)

    for path in (OUTPUT_MODEL, BEST_WEIGHTS, REPORT_PATH, CROP_ROOT, TRAIN_ROOT):
        if path.exists():
            raise FileExistsError(
                f"Output already exists: {path}\n"
                "Nothing was overwritten. Rename the old output before rerunning."
            )

    class_map = json.loads((MODEL_DIR / "class_indices.json").read_text(encoding="utf-8"))
    expected_map = {letter: i for i, letter in enumerate(LETTERS)}
    if class_map != expected_map:
        raise RuntimeError(f"Class map mismatch. Expected {expected_map}, found {class_map}")

    print("Loading production model. It will remain unchanged.")
    production_model = tf.keras.models.load_model(SOURCE_MODEL)
    if production_model.output_shape[-1] != 26:
        raise RuntimeError(f"Expected 26 model outputs, got {production_model.output_shape[-1]}")

    stats = prepare_webcam_crops()
    prepare_training_folders()

    train = make_generator(TRAIN_ROOT / "train", training=True, shuffle=True)
    validation = make_generator(TRAIN_ROOT / "validation", training=False, shuffle=False)
    webcam_validation = make_generator(CROP_ROOT / "validation", training=False, shuffle=False)
    original_test = make_generator(DATA / "test", training=False, shuffle=False)

    if train.class_indices != validation.class_indices:
        raise RuntimeError("Training and validation class order differs.")

    # Clone from exact production weights; all candidate training happens on the clone.
    candidate = tf.keras.models.clone_model(production_model)
    candidate.set_weights(production_model.get_weights())

    base = candidate.layers[1]
    base.trainable = True
    for layer in base.layers:
        layer.trainable = False
    for layer in base.layers[-20:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    candidate.compile(
        optimizer=Adam(learning_rate=0.000001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        ModelCheckpoint(
            str(BEST_WEIGHTS), monitor="val_loss", save_best_only=True,
            save_weights_only=True, verbose=1
        ),
        EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=1,
            min_lr=0.0000001, verbose=1
        )
    ]

    print("\nStarting corrected crop-aligned fine-tuning (maximum 5 epochs).")
    print("Production model remains unchanged:", SOURCE_MODEL)
    candidate.fit(
        train,
        validation_data=validation,
        epochs=5,
        callbacks=callbacks,
        verbose=1
    )

    if not BEST_WEIGHTS.is_file():
        raise RuntimeError("No best checkpoint was produced; candidate was not saved.")
    candidate.load_weights(BEST_WEIGHTS)
    candidate.save(OUTPUT_MODEL)

    print("\nEvaluating original test set and held-out webcam crops on BOTH models...")
    base_orig_loss, base_orig_acc, _, _ = evaluate(production_model, original_test)
    cand_orig_loss, cand_orig_acc, _, _ = evaluate(candidate, original_test)
    base_web_loss, base_web_acc, y_true, base_web_pred = evaluate(production_model, webcam_validation)
    cand_web_loss, cand_web_acc, _, cand_web_pred = evaluate(candidate, webcam_validation)

    names = LETTERS
    web_report = classification_report(
        y_true, cand_web_pred, labels=list(range(26)), target_names=names,
        output_dict=True, zero_division=0
    )
    report = {
        "source_model": str(SOURCE_MODEL),
        "candidate_model": str(OUTPUT_MODEL),
        "webcam_crop_stats": stats,
        "metrics": {
            "production_original_test_accuracy": base_orig_acc,
            "candidate_original_test_accuracy": cand_orig_acc,
            "production_webcam_validation_accuracy": base_web_acc,
            "candidate_webcam_validation_accuracy": cand_web_acc,
            "production_original_test_loss": base_orig_loss,
            "candidate_original_test_loss": cand_orig_loss,
            "production_webcam_validation_loss": base_web_loss,
            "candidate_webcam_validation_loss": cand_web_loss
        },
        "candidate_webcam_classification_report": web_report,
        "candidate_webcam_confusion_matrix": confusion_matrix(
            y_true, cand_web_pred, labels=list(range(26))
        ).tolist()
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n================ FINAL COMPARISON ================")
    print(f"Original test | production v2: {base_orig_acc * 100:.2f}%")
    print(f"Original test | crop candidate: {cand_orig_acc * 100:.2f}%")
    print(f"Webcam holdout | production v2: {base_web_acc * 100:.2f}%")
    print(f"Webcam holdout | crop candidate: {cand_web_acc * 100:.2f}%")
    print("\nCandidate model:", OUTPUT_MODEL)
    print("Report:", REPORT_PATH)
    print("Production model was not modified.")

    if cand_web_acc > base_web_acc and cand_orig_acc >= base_orig_acc - 0.005:
        print("RESULT: Candidate improved held-out webcam accuracy without a large original-test regression.")
        print("Next step: run the live webcam A/B check before promotion.")
    else:
        print("RESULT: Do NOT promote the candidate. Production v2 remains the recommended model.")


if __name__ == "__main__":
    main()

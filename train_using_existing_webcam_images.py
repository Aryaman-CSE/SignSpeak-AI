import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import json
import random
import re
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets" / "asl_alphabet"
WEBCAM = ROOT / "datasets" / "webcam_samples"
MODEL_DIR = ROOT / "models" / "asl_mobilenetv2"
SOURCE_MODEL = MODEL_DIR / "webcam_finetuned_v2.h5"
CLASS_MAP = MODEL_DIR / "class_indices.json"
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CROP_SIZE = 320
MODEL_SIZE = 160
CROP_SCALE = 1.75
MIN_HAND_SIZE = 35
SEED = 73
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

RUN = datetime.now().strftime("%Y%m%d_%H%M%S")
CROP_ROOT = MODEL_DIR / f"existing_webcam_crops_{RUN}"
TRAIN_ROOT = MODEL_DIR / f"existing_webcam_training_{RUN}"
OUTPUT_MODEL = MODEL_DIR / f"webcam_existing_data_candidate_{RUN}.h5"
BEST_WEIGHTS = MODEL_DIR / f"webcam_existing_data_best_{RUN}.weights.h5"
REPORT = MODEL_DIR / f"webcam_existing_data_report_{RUN}.json"


def natural_key(p):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]


def image_files(folder):
    if not folder.is_dir():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTS], key=natural_key)


def live_style_crop(frame, landmarks):
    """Same geometry as realtime_signspeak.py, applied in original image coordinates."""
    h, w = frame.shape[:2]
    xs = np.asarray([p.x * w for p in landmarks.landmark], np.float32)
    ys = np.asarray([p.y * h for p in landmarks.landmark], np.float32)
    left, right, top, bottom = float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())
    bw, bh = right - left, bottom - top
    if max(bw, bh) < MIN_HAND_SIZE:
        return None
    side = max(max(bw, bh) * CROP_SCALE, MIN_HAND_SIZE * 2)
    cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
    x1, y1 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x2, y2 = int(round(cx + side / 2)), int(round(cy + side / 2))
    sx1, sy1, sx2, sy2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if sx2 <= sx1 or sy2 <= sy1:
        return None
    crop = frame[sy1:sy2, sx1:sx2]
    crop = cv2.copyMakeBorder(crop, max(0, -y1), max(0, y2-h), max(0, -x1), max(0, x2-w), cv2.BORDER_REPLICATE)
    crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(crop, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_AREA)


def center_padded_fallback(frame):
    """Fallback for already tightly framed hand photos; retain hand scale with replicated padding."""
    h, w = frame.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    square = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_REPLICATE)
    pad = max(1, int(round(side * (1.50 - 1.0) / 2.0)))
    square = cv2.copyMakeBorder(square, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    square = cv2.resize(square, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(square, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_AREA)


def detect_crop(frame, hands):
    """Try enlarged, contrast-normalized images before using a centered fallback."""
    h, w = frame.shape[:2]
    scale = min(3.0, max(1.8, 640.0 / max(h, w)))
    enlarged = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(enlarged, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    attempts = (enlarged, enhanced, frame)
    for candidate in attempts:
        rgb = cv2.cvtColor(candidate, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = hands.process(rgb)
        for landmarks in (result.multi_hand_landmarks or []):
            crop = live_style_crop(frame, landmarks)
            if crop is not None:
                return crop, "mediapipe"
    return center_padded_fallback(frame), "fallback"


def make_gen(directory, training=False, shuffle=False):
    if training:
        aug = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=4,
            width_shift_range=0.025,
            height_shift_range=0.025,
            zoom_range=0.04,
            brightness_range=(0.90, 1.10),
            horizontal_flip=False,
        )
    else:
        aug = ImageDataGenerator(preprocessing_function=preprocess_input)
    return aug.flow_from_directory(
        str(directory), target_size=(MODEL_SIZE, MODEL_SIZE), color_mode="rgb",
        batch_size=32, classes=LETTERS, class_mode="categorical",
        shuffle=shuffle, seed=SEED, interpolation="nearest"
    )


def evaluate(model, gen):
    gen.reset()
    loss, accuracy = model.evaluate(gen, verbose=0)
    gen.reset()
    pred = np.argmax(model.predict(gen, verbose=0), axis=1)
    return float(loss), float(accuracy), gen.classes.copy(), pred


def main():
    for p in (SOURCE_MODEL, CLASS_MAP, WEBCAM, DATA / "train", DATA / "validation", DATA / "test"):
        if not p.exists():
            raise FileNotFoundError(f"Required project file/folder missing: {p}")
    class_map = json.loads(CLASS_MAP.read_text(encoding="utf-8"))
    expected = {letter: i for i, letter in enumerate(LETTERS)}
    if class_map != expected:
        raise RuntimeError(f"Class mapping is not A-Z in expected order: {class_map}")

    present = [letter for letter in LETTERS if image_files(WEBCAM / letter)]
    absent = [letter for letter in LETTERS if letter not in present]
    if not present:
        raise RuntimeError(f"No webcam images found under {WEBCAM}")
    print("Using existing webcam images. No new collection required.")
    print("Available webcam classes:", ", ".join(present))
    if absent:
        print("No personal webcam images for:", ", ".join(absent), "(these keep original training data)")

    for split in ("train", "validation"):
        for letter in LETTERS:
            (CROP_ROOT / split / letter).mkdir(parents=True, exist_ok=True)
            (TRAIN_ROOT / split / letter).mkdir(parents=True, exist_ok=True)

    # One static detector per letter; low threshold and 2-3x upscaling are for the small stored frames.
    mp_hands = mp.solutions.hands
    summary = {}
    print("\nExtracting hand crops from existing photos (upscaled + contrast retry)...")
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, model_complexity=1,
                        min_detection_confidence=0.15) as hands:
        for letter in present:
            files = image_files(WEBCAM / letter)
            cut = max(1, int(len(files) * 0.80))
            splits = {"train": files[:cut], "validation": files[cut:]}
            summary[letter] = {"source": len(files), "mediapipe": 0, "fallback": 0, "train": 0, "validation": 0}
            for split, group in splits.items():
                for index, path in enumerate(group):
                    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    crop, mode = detect_crop(frame, hands)
                    out = CROP_ROOT / split / letter / f"{path.stem}_{index:04d}.jpg"
                    if not cv2.imwrite(str(out), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                        raise RuntimeError(f"Could not save crop: {out}")
                    summary[letter][mode] += 1
                    summary[letter][split] += 1
            s = summary[letter]
            print(f"{letter}: {s['source']} existing | {s['mediapipe']} hand-detected | {s['fallback']} centered fallback | {s['train']} train | {s['validation']} validation")

    # Keep a small balanced original-data replay so webcam adaptation does not wipe general ASL knowledge.
    print("\nBuilding balanced training folders...")
    for letter in LETTERS:
        tr = image_files(DATA / "train" / letter)
        va = image_files(DATA / "validation" / letter)
        if len(tr) < 150 or len(va) < 30:
            raise RuntimeError(f"Original dataset too small for {letter}: train={len(tr)}, validation={len(va)}")
        random.shuffle(tr)
        random.shuffle(va)
        for i, src in enumerate(tr[:150]):
            shutil.copy2(src, TRAIN_ROOT / "train" / letter / f"original_{i:03d}{src.suffix.lower()}")
        for i, src in enumerate(va[:30]):
            shutil.copy2(src, TRAIN_ROOT / "validation" / letter / f"original_{i:03d}{src.suffix.lower()}")
        # Limit per-letter webcam replay to avoid overwhelming original examples.
        for split, limit in (("train", 100), ("validation", 40)):
            crops = sorted((CROP_ROOT / split / letter).glob("*.jpg"), key=natural_key)
            random.shuffle(crops)
            for i, src in enumerate(crops[:limit]):
                shutil.copy2(src, TRAIN_ROOT / split / letter / f"webcam_{i:03d}.jpg")

    train = make_gen(TRAIN_ROOT / "train", training=True, shuffle=True)
    val = make_gen(TRAIN_ROOT / "validation", training=False, shuffle=False)
    original_test = make_gen(DATA / "test", training=False, shuffle=False)
    webcam_val = make_gen(CROP_ROOT / "validation", training=False, shuffle=False)

    print("\nLoading production v2 (source file will not be modified)...")
    model = tf.keras.models.load_model(SOURCE_MODEL)
    if model.output_shape[-1] != 26:
        raise RuntimeError(f"Expected 26 model outputs, got {model.output_shape}")
    if len(model.layers) < 2:
        raise RuntimeError("Unexpected model architecture; stopping without saving")
    base = model.layers[1]
    base.trainable = True
    for layer in base.layers:
        layer.trainable = False
    for layer in base.layers[-20:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    model.compile(optimizer=Adam(learning_rate=0.000001), loss="categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        ModelCheckpoint(str(BEST_WEIGHTS), monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-7, verbose=1),
    ]
    print("\nFine-tuning from existing webcam_finetuned_v2.h5. Maximum 5 epochs.")
    model.fit(train, validation_data=val, epochs=5, callbacks=callbacks, verbose=1)
    if not BEST_WEIGHTS.is_file():
        raise RuntimeError("Training did not produce a best checkpoint; production model remains untouched")
    model.load_weights(BEST_WEIGHTS)
    model.save(OUTPUT_MODEL)

    print("\nEvaluating both models on the untouched original test set and held-out webcam crops...")
    production = tf.keras.models.load_model(SOURCE_MODEL)
    p_orig_loss, p_orig_acc, _, _ = evaluate(production, original_test)
    c_orig_loss, c_orig_acc, _, _ = evaluate(model, original_test)
    p_web_loss, p_web_acc, y, p_web_pred = evaluate(production, webcam_val)
    c_web_loss, c_web_acc, _, c_web_pred = evaluate(model, webcam_val)
    report_by_class = classification_report(y, c_web_pred, labels=list(range(26)), target_names=LETTERS, output_dict=True, zero_division=0)
    results = {
        "webcam_sample_summary": summary,
        "metrics": {
            "production_original_test_accuracy": p_orig_acc,
            "candidate_original_test_accuracy": c_orig_acc,
            "production_webcam_holdout_accuracy": p_web_acc,
            "candidate_webcam_holdout_accuracy": c_web_acc,
            "production_original_test_loss": p_orig_loss,
            "candidate_original_test_loss": c_orig_loss,
            "production_webcam_holdout_loss": p_web_loss,
            "candidate_webcam_holdout_loss": c_web_loss,
        },
        "candidate_webcam_per_letter": report_by_class,
        "candidate_webcam_confusion_matrix": confusion_matrix(y, c_web_pred, labels=list(range(26))).tolist(),
        "candidate_model": str(OUTPUT_MODEL),
    }
    REPORT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n================= FINAL COMPARISON =================")
    print(f"Original test | production v2: {p_orig_acc*100:.2f}%")
    print(f"Original test | candidate:     {c_orig_acc*100:.2f}%")
    print(f"Webcam holdout | production:   {p_web_acc*100:.2f}%")
    print(f"Webcam holdout | candidate:    {c_web_acc*100:.2f}%")
    print("Candidate model:", OUTPUT_MODEL)
    print("Evaluation report:", REPORT)
    if c_web_acc > p_web_acc and c_orig_acc >= p_orig_acc - 0.005:
        print("DECISION: Candidate is eligible for a live A/B test; do not auto-replace production.")
    else:
        print("DECISION: Do NOT promote. Production v2 remains unchanged.")
    print("Crop summary saved in report; 'fallback' images are approximate crops, not true landmarks.")


if __name__ == "__main__":
    main()

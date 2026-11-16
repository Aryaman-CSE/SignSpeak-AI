"""Background-robust fine-tuning for SignSpeak AI.

This script:
- finds labelled A-Z image folders in the known SignSpeak project locations;
- derives foreground masks using MediaPipe Selfie Segmentation, with a border-colour fallback;
- trains on original images and foregrounds composited over many synthetic/real backgrounds;
- evaluates the original and candidate models on the same held-out raw and composited images;
- promotes the candidate only when robust validation improves without materially hurting raw validation.

The active model is backed up before promotion. Dynamic J/Z model files are not modified.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "asl_mobilenetv2"
CLASS_FILE = MODEL_DIR / "class_indices.json"
LIVE_SCRIPT = ROOT / "realtime_signspeak.py"
MASK_CACHE = MODEL_DIR / "background_mask_cache"
REAL_BACKGROUND_DIRS = [ROOT / "backgrounds", ROOT / "datasets" / "backgrounds"]

SEED = 20261116
MAX_TRAIN_PER_CLASS = 220
MAX_VAL_PER_CLASS = 55
BATCH_SIZE = 24
EPOCHS = 8
LEARNING_RATE = 2e-6
BACKGROUND_PROBABILITY = 0.75
MIN_TRAIN_PER_CLASS = 12
MIN_VAL_PER_CLASS = 8

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def stop(message: str, code: int = 1) -> None:
    print("\n[STOP] " + message)
    raise SystemExit(code)


def resolve_active_model() -> Path:
    if not CLASS_FILE.is_file():
        stop(f"Missing class map: {CLASS_FILE}")
    if not LIVE_SCRIPT.is_file():
        stop(f"Missing live app: {LIVE_SCRIPT}")

    source = LIVE_SCRIPT.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r'model_path\s*=\s*SCRIPT_DIR\s*/\s*"models"\s*/\s*"asl_mobilenetv2"\s*/\s*"([^"]+\.h5)"',
        source,
    )
    if match:
        active = MODEL_DIR / match.group(1)
    else:
        choices = [MODEL_DIR / "webcam_finetuned_v2.h5", MODEL_DIR / "webcam_finetuned_model.h5"]
        active = next((p for p in choices if p.is_file()), choices[0])
    if not active.is_file():
        stop(f"The model referenced by realtime_signspeak.py does not exist: {active}")
    return active


def read_class_map() -> tuple[list[str], dict[str, int]]:
    data = json.loads(CLASS_FILE.read_text(encoding="utf-8"))
    mapping = {str(k).upper(): int(v) for k, v in data.items()}
    if len(mapping) != 26 or set(mapping) != set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        stop("class_indices.json must contain all 26 letters A-Z exactly once.")
    if sorted(mapping.values()) != list(range(26)):
        stop("class_indices.json indices must be exactly 0 through 25.")
    classes = [letter for letter, _ in sorted(mapping.items(), key=lambda item: item[1])]
    return classes, mapping


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def class_folders(folder: Path, classes: list[str]) -> dict[str, Path]:
    if not folder.is_dir():
        return {}
    found = {}
    for child in folder.iterdir():
        if child.is_dir() and child.name.upper() in classes:
            found[child.name.upper()] = child
    return found


def split_names(folder: Path) -> tuple[Path | None, Path | None]:
    train = next((folder / n for n in ("train", "training") if (folder / n).is_dir()), None)
    val = next((folder / n for n in ("validation", "val", "valid", "test") if (folder / n).is_dir()), None)
    return train, val


def collect_records(classes: list[str]) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    train_records: list[tuple[Path, int]] = []
    val_records: list[tuple[Path, int]] = []
    seen_train: set[str] = set()
    seen_val: set[str] = set()

    known_roots = [
        ROOT / "datasets" / "asl_alphabet",
        ROOT / "models" / "asl_mobilenetv2" / "webcam_existing_images_split",
        ROOT / "datasets" / "webcam_samples",
        ROOT / "models" / "asl_mobilenetv2" / "webcam_crops_corrected",
        ROOT / "models" / "asl_mobilenetv2" / "existing_webcam_crops_20261115_164246",
        ROOT / "models" / "asl_mobilenetv2" / "existing_webcam_training_20261115_164246",
    ]

    for root in known_roots:
        if not root.is_dir():
            continue
        train_root, val_root = split_names(root)
        if train_root is not None:
            train_map = class_folders(train_root, classes)
            val_map = class_folders(val_root, classes) if val_root else {}
            for letter in classes:
                files = image_files(train_map.get(letter, Path("/__not_found__")))
                random.shuffle(files)
                for path in files[:MAX_TRAIN_PER_CLASS]:
                    key = str(path.resolve()).lower()
                    if key not in seen_train:
                        train_records.append((path, classes.index(letter)))
                        seen_train.add(key)
                vfiles = image_files(val_map.get(letter, Path("/__not_found__")))
                random.shuffle(vfiles)
                for path in vfiles[:MAX_VAL_PER_CLASS]:
                    key = str(path.resolve()).lower()
                    if key not in seen_val:
                        val_records.append((path, classes.index(letter)))
                        seen_val.add(key)
        else:
            direct_map = class_folders(root, classes)
            if not direct_map:
                continue
            for letter in classes:
                files = image_files(direct_map.get(letter, Path("/__not_found__")))
                if not files:
                    continue
                random.shuffle(files)
                cut = max(1, int(len(files) * 0.80)) if len(files) > 1 else len(files)
                train_files = files[:cut]
                val_files = files[cut:]
                for path in train_files[:MAX_TRAIN_PER_CLASS]:
                    key = str(path.resolve()).lower()
                    if key not in seen_train:
                        train_records.append((path, classes.index(letter)))
                        seen_train.add(key)
                for path in val_files[:MAX_VAL_PER_CLASS]:
                    key = str(path.resolve()).lower()
                    if key not in seen_val:
                        val_records.append((path, classes.index(letter)))
                        seen_val.add(key)

    # The raw ASL alphabet corpus may be named "train" plus "test" rather than "validation".
    train_counts = Counter(label for _, label in train_records)
    val_counts = Counter(label for _, label in val_records)
    missing_train = [classes[i] for i in range(26) if train_counts[i] < MIN_TRAIN_PER_CLASS]
    missing_val = [classes[i] for i in range(26) if val_counts[i] < MIN_VAL_PER_CLASS]
    if missing_train or missing_val:
        detail = []
        if missing_train:
            detail.append("too few training images: " + ", ".join(missing_train))
        if missing_val:
            detail.append("too few validation images: " + ", ".join(missing_val))
        stop(
            "Not enough labelled images to safely fine-tune all 26 classes (" + "; ".join(detail) + "). "
            "Expected datasets/asl_alphabet/{train,validation}/A..Z or an equivalent split in "
            "models/asl_mobilenetv2/webcam_existing_images_split. No model was changed."
        )
    return train_records, val_records


def border_colour_mask(image_bgr: np.ndarray) -> np.ndarray | None:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    border = np.concatenate((lab[0, :, :], lab[-1, :, :], lab[:, 0, :], lab[:, -1, :]), axis=0)
    bg = np.median(border, axis=0)
    border_dist = np.linalg.norm(border - bg, axis=1)
    threshold = max(17.0, float(np.percentile(border_dist, 97)) + 9.0)
    dist = np.linalg.norm(lab - bg, axis=2)
    mask = (dist > threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [largest], -1, 255, thickness=cv2.FILLED)
    ratio = float(np.mean(filled > 0))
    if not 0.035 <= ratio <= 0.88:
        return None
    return cv2.GaussianBlur(filled, (5, 5), 0)


class MaskMaker:
    def __init__(self) -> None:
        MASK_CACHE.mkdir(parents=True, exist_ok=True)
        self.segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.cache_hits = 0
        self.segmented = 0
        self.fallback = 0
        self.failed = 0

    def mask(self, path: Path, image_bgr: np.ndarray) -> np.ndarray | None:
        stat = path.stat()
        key = hashlib.sha1(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()
        cache_path = MASK_CACHE / f"{key}.png"
        if cache_path.is_file():
            cached = cv2.imread(str(cache_path), cv2.IMREAD_GRAYSCALE)
            if cached is not None:
                self.cache_hits += 1
                return cached

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self.segmenter.process(rgb)
        alpha = None
        if result.segmentation_mask is not None:
            probability = cv2.resize(result.segmentation_mask, (160, 160), interpolation=cv2.INTER_LINEAR)
            fg_ratio = float(np.mean(probability > 0.5))
            if 0.025 <= fg_ratio <= 0.96:
                alpha = np.clip((probability - 0.12) / 0.68, 0.0, 1.0)
                alpha = cv2.GaussianBlur(alpha.astype(np.float32), (5, 5), 0.8)
                self.segmented += 1

        if alpha is None:
            mask = border_colour_mask(image_bgr)
            if mask is not None:
                alpha = mask.astype(np.float32) / 255.0
                self.fallback += 1

        if alpha is None:
            self.failed += 1
            return None

        out = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(cache_path), out)
        return out

    def close(self) -> None:
        self.segmenter.close()


def make_background(height: int, width: int, rng: np.random.Generator, real_backgrounds: list[Path]) -> np.ndarray:
    if real_backgrounds and rng.random() < 0.40:
        p = real_backgrounds[int(rng.integers(0, len(real_backgrounds)))]
        image = cv2.imread(str(p))
        if image is not None:
            return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    c1 = rng.integers(0, 256, size=3).astype(np.float32)
    c2 = rng.integers(0, 256, size=3).astype(np.float32)
    horizontal = np.linspace(0, 1, width, dtype=np.float32)[None, :, None]
    vertical = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    blend = (horizontal * 0.55 + vertical * 0.45)
    base = c1[None, None, :] * (1 - blend) + c2[None, None, :] * blend
    texture = rng.normal(0, rng.uniform(8, 28), size=(height, width, 1)).astype(np.float32)
    bg = np.repeat(base, height, axis=0) if base.shape[0] == 1 else base
    bg = np.clip(bg + texture, 0, 255).astype(np.uint8)

    # Add varied edges, shapes, and texture so the network sees more than flat colour fields.
    for _ in range(int(rng.integers(5, 18))):
        colour = tuple(int(x) for x in rng.integers(0, 256, size=3))
        p1 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        p2 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        thickness = int(rng.integers(1, 7))
        if rng.random() < 0.5:
            cv2.line(bg, p1, p2, colour, thickness, cv2.LINE_AA)
        else:
            radius = int(rng.integers(4, max(5, min(width, height) // 3)))
            cv2.circle(bg, p1, radius, colour, -1, cv2.LINE_AA)
    if rng.random() < 0.5:
        bg = cv2.GaussianBlur(bg, (int(rng.choice([3, 5, 7])),) * 2, 0)
    return bg


def compose_foreground(image: np.ndarray, alpha_u8: np.ndarray | None, bg: np.ndarray) -> np.ndarray:
    if alpha_u8 is None:
        return image.copy()
    alpha = alpha_u8.astype(np.float32)[:, :, None] / 255.0
    return np.clip(image.astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)


def transform_image(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = image.shape[:2]
    angle = float(rng.uniform(-8.0, 8.0))
    scale = float(rng.uniform(0.92, 1.08))
    tx, ty = float(rng.uniform(-0.035, 0.035) * w), float(rng.uniform(-0.035, 0.035) * h)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    result = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    if rng.random() < 0.8:
        alpha = float(rng.uniform(0.78, 1.25))
        beta = float(rng.uniform(-18, 18))
        result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
    if rng.random() < 0.15:
        result = cv2.GaussianBlur(result, (3, 3), 0)
    return result


class SignSequence(tf.keras.utils.Sequence):
    def __init__(self, records, mask_maker, training: bool, robust_validation: bool = False):
        self.records = list(records)
        self.mask_maker = mask_maker
        self.training = training
        self.robust_validation = robust_validation
        self.indices = np.arange(len(self.records))
        self.real_backgrounds = []
        for folder in REAL_BACKGROUND_DIRS:
            if folder.is_dir():
                self.real_backgrounds.extend(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.records) / BATCH_SIZE))

    def on_epoch_end(self):
        if self.training:
            np.random.shuffle(self.indices)

    def __getitem__(self, batch_index):
        ids = self.indices[batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE]
        images, labels = [], []
        for record_index in ids:
            path, label = self.records[int(record_index)]
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f"Could not read image: {path}")
            image = cv2.resize(image, (160, 160), interpolation=cv2.INTER_AREA)
            alpha = self.mask_maker.mask(path, image)

            if self.training:
                rng = np.random.default_rng()
                if alpha is not None and rng.random() < BACKGROUND_PROBABILITY:
                    bg = make_background(160, 160, rng, self.real_backgrounds)
                    image = compose_foreground(image, alpha, bg)
                image = transform_image(image, rng)
            elif self.robust_validation and alpha is not None:
                stable_seed = int(hashlib.sha1(str(path).encode()).hexdigest()[:8], 16)
                rng = np.random.default_rng(stable_seed)
                bg = make_background(160, 160, rng, self.real_backgrounds)
                image = compose_foreground(image, alpha, bg)

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            images.append(preprocess_input(rgb))
            labels.append(label)

        return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def mask_coverage(records: list[tuple[Path, int]], maker: MaskMaker) -> tuple[float, dict[str, float]]:
    per_class = Counter()
    good = Counter()
    total = Counter()
    for path, label in records:
        image = cv2.imread(str(path))
        if image is None:
            continue
        image = cv2.resize(image, (160, 160), interpolation=cv2.INTER_AREA)
        mask = maker.mask(path, image)
        total[label] += 1
        if mask is not None:
            good[label] += 1
    coverage = sum(good.values()) / max(1, sum(total.values()))
    by_class = {str(i): good[i] / max(1, total[i]) for i in range(26)}
    return coverage, by_class


def find_backbone(model: tf.keras.Model):
    nested = [layer for layer in model.layers if isinstance(layer, tf.keras.Model)]
    if not nested:
        return None
    return max(nested, key=lambda layer: len(layer.layers))


def evaluate(model, sequence, label: str) -> float:
    probabilities = model.predict(sequence, verbose=0)
    predicted = np.argmax(probabilities, axis=1)
    truth = np.asarray(
        [class_index for _, class_index in sequence.records],
        dtype=np.int32
    )
    if len(predicted) != len(truth):
        raise RuntimeError("Evaluation prediction count does not match labels.")
    accuracy = float(np.mean(predicted == truth))
    print(f"{label}: {accuracy * 100:.2f}%")
    return accuracy


def evaluate_per_class(model, sequence, records, classes, label: str) -> dict[str, float]:
    probabilities = model.predict(sequence, verbose=0)
    predicted = np.argmax(probabilities, axis=1)
    truth = np.asarray([class_index for _, class_index in records], dtype=np.int32)
    if len(predicted) != len(truth):
        raise RuntimeError("Validation prediction count does not match validation labels.")
    scores = {}
    print(f"\n{label} per-letter accuracy:")
    for index, letter in enumerate(classes):
        mask = truth == index
        score = float(np.mean(predicted[mask] == truth[mask])) if np.any(mask) else 0.0
        scores[letter] = score
        print(f"  {letter}: {score * 100:.1f}% ({int(np.sum(predicted[mask] == truth[mask]))}/{int(mask.sum())})")
    return scores


def patch_temporal_smoothing() -> None:
    """Install bounded weighted temporal voting without touching model weights."""
    import py_compile

    if not LIVE_SCRIPT.is_file():
        print("[WARN] Live script not found; temporal smoother was not changed.")
        return
    source = LIVE_SCRIPT.read_text(encoding="utf-8", errors="replace")
    function_pattern = re.compile(
        r"(?ms)^def stable_prediction\(history\):.*?(?=^def (?:load_dynamic_model|main)\s*\()"
    )
    replacement = '''def stable_prediction(history):
    recent = list(history)[-HISTORY_SIZE:]
    if not recent:
        return None, 0.0

    evidence = {}
    confidence_total = {}
    weight_total = {}
    support = {}
    recent_support = {}
    last_age = {}

    for index, (label, score) in enumerate(recent):
        if label is None:
            continue
        age = len(recent) - 1 - index
        score = float(np.clip(score, 0.0, 1.0))
        weight = (0.82 ** age) * (0.55 + 0.45 * score)
        evidence[label] = evidence.get(label, 0.0) + weight
        confidence_total[label] = confidence_total.get(label, 0.0) + score * weight
        weight_total[label] = weight_total.get(label, 0.0) + weight
        support[label] = support.get(label, 0) + 1
        if age <= 3:
            recent_support[label] = recent_support.get(label, 0) + 1
        last_age[label] = age

    if not evidence:
        return None, 0.0

    ranked = sorted(evidence.items(), key=lambda item: item[1], reverse=True)
    best_label, best_weight = ranked[0]
    runner_up_weight = ranked[1][1] if len(ranked) > 1 else 0.0
    total_weight = sum(evidence.values())
    dominance = best_weight / max(total_weight, 1e-8)
    average_confidence = confidence_total[best_label] / max(weight_total[best_label], 1e-8)

    # Require repeated support, including recent evidence, but tolerate brief missed frames.
    if support[best_label] < max(3, VOTES_REQUIRED):
        return None, 0.0
    if recent_support.get(best_label, 0) < 2 or last_age[best_label] > 2:
        return None, 0.0
    if dominance < 0.48 or best_weight < runner_up_weight * 1.10:
        return None, 0.0
    if average_confidence < MIN_SIGN_CONFIDENCE:
        return None, 0.0

    return best_label, float(average_confidence)

'''
    if not function_pattern.search(source):
        print("[WARN] Could not safely locate stable_prediction(); live script left unchanged.")
        return

    updated = function_pattern.sub(lambda _: replacement, source, count=1)
    for name, value in (("HISTORY_SIZE", "7"), ("VOTES_REQUIRED", "3")):
        updated, count = re.subn(rf"(?m)^\s*{name}\s*=.*$", f"{name} = {value}", updated, count=1)
        if count != 1:
            print(f"[WARN] Could not update {name}; live script left unchanged.")
            return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = LIVE_SCRIPT.with_name(f"realtime_signspeak_before_weighted_smoothing_{stamp}.py")
    shutil.copy2(LIVE_SCRIPT, backup)
    LIVE_SCRIPT.write_text(updated, encoding="utf-8")
    try:
        py_compile.compile(str(LIVE_SCRIPT), doraise=True)
    except Exception:
        shutil.copy2(backup, LIVE_SCRIPT)
        print("[WARN] Temporal patch failed syntax validation; restored original live script.")
        return
    print("[PATCHED] Weighted 7-frame temporal smoothing installed.")
    print("Live-script backup:", backup)


def main() -> None:
    print("\n=== SignSpeak background-robust fine-tuning ===")
    print("Project:", ROOT)
    classes, class_map = read_class_map()
    active_model = resolve_active_model()
    train_records, val_records = collect_records(classes)

    print("Active production model:", active_model.name)
    print("Training images:", len(train_records), "| Validation images:", len(val_records))
    for i, letter in enumerate(classes):
        tc = sum(1 for _, y in train_records if y == i)
        vc = sum(1 for _, y in val_records if y == i)
        print(f"  {letter}: train={tc}, validation={vc}")

    model = tf.keras.models.load_model(active_model)
    if int(model.output_shape[-1]) != 26:
        stop(f"Expected 26 classifier outputs, found {model.output_shape[-1]}. No model was changed.")

    maker = MaskMaker()
    try:
        print("\nPreparing foreground masks (first run can take time)...")
        train_coverage, _ = mask_coverage(train_records, maker)
        val_coverage, per_class_val = mask_coverage(val_records, maker)
        print(f"Training mask coverage: {train_coverage * 100:.1f}%")
        print(f"Validation mask coverage: {val_coverage * 100:.1f}%")
        if val_coverage < 0.45 or train_coverage < 0.45:
            stop(
                "Foreground masks were reliable for fewer than 45% of images. "
                "I will not train or replace the working model using unreliable masks. "
                "Use the project's close-up hand crops or add clean labelled hand images."
            )
        weak = [classes[int(i)] for i, score in per_class_val.items() if score < 0.20]
        if weak:
            stop("Segmentation is unreliable for these validation classes: " + ", ".join(weak) + ". No model was changed.")

        raw_val = SignSequence(val_records, maker, training=False, robust_validation=False)
        robust_val = SignSequence(val_records, maker, training=False, robust_validation=True)
        baseline_raw = evaluate(model, raw_val, "Original model / held-out raw validation")
        baseline_robust = evaluate(model, robust_val, "Original model / held-out varied-background validation")
        baseline_letter_scores = evaluate_per_class(
            model, robust_val, val_records, classes, "Original model / varied-background"
        )

        backbone = find_backbone(model)
        if backbone is None:
            stop("Could not identify the nested MobileNetV2 backbone safely. No model was changed.")
        backbone.trainable = True
        for layer in backbone.layers:
            layer.trainable = False
        for layer in backbone.layers[-30:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True

        # Keep the classifier head trainable while updating only the final feature layers.
        for layer in model.layers:
            if layer is not backbone:
                layer.trainable = True

        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = MODEL_DIR / f"webcam_background_robust_candidate_{stamp}.h5"
        best_weights = MODEL_DIR / f"webcam_background_robust_best_{stamp}.weights.h5"
        train_seq = SignSequence(train_records, maker, training=True)

        callbacks = [
            ModelCheckpoint(str(best_weights), monitor="val_accuracy", mode="max", save_best_only=True, save_weights_only=True, verbose=1),
            EarlyStopping(monitor="val_accuracy", mode="max", patience=3, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=2e-7, verbose=1),
        ]
        print("\nFine-tuning with randomized foreground/background composites...")
        model.fit(train_seq, validation_data=robust_val, epochs=EPOCHS, callbacks=callbacks, verbose=1)
        if best_weights.is_file():
            model.load_weights(best_weights)
        model.save(candidate)
        print("Candidate saved:", candidate)

        raw_after = evaluate(model, raw_val, "Candidate / held-out raw validation")
        robust_after = evaluate(model, robust_val, "Candidate / held-out varied-background validation")
        candidate_letter_scores = evaluate_per_class(
            model, robust_val, val_records, classes, "Candidate / varied-background"
        )

        # Promote only when the unseen-background validation improves and no letter regresses materially.
        robust_gain = robust_after - baseline_robust
        raw_drop = baseline_raw - raw_after
        worst_letter_change = min(
            candidate_letter_scores[letter] - baseline_letter_scores[letter]
            for letter in classes
        )
        print(f"\nRobust validation change: {robust_gain * 100:+.2f} percentage points")
        print(f"Raw validation change: {(-raw_drop) * 100:+.2f} percentage points")
        print(f"Worst per-letter change: {worst_letter_change * 100:+.2f} percentage points")

        if robust_gain >= 0.01 and raw_drop <= 0.02 and worst_letter_change >= -0.05:
            backup = active_model.with_name(active_model.stem + f"_before_background_robust_{stamp}" + active_model.suffix)
            shutil.copy2(active_model, backup)
            shutil.copy2(candidate, active_model)
            print("\n[PROMOTED] Candidate improved varied-background validation and preserved raw validation.")
            print("Production backup:", backup)
            print("Updated production model:", active_model)
        else:
            print("\n[NOT PROMOTED] Candidate did not meet the safe validation gate.")
            print("The current production model was left untouched.")
            print("Candidate retained for inspection:", candidate)

        patch_temporal_smoothing()
        print("\nJ/Z LSTM files were not modified.")
        print("Next validation: test the live app on at least 3 physical backgrounds not used in training.")
    finally:
        maker.close()


if __name__ == "__main__":
    main()

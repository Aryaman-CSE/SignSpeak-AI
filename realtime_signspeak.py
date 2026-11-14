
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

import re
import json
import glob
import time
import shutil
import threading
from pathlib import Path
from datetime import datetime
from collections import deque, Counter

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import mediapipe as mp


CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
INPUT_SIZE = 320
MAX_HANDS = 2
HAND_DETECTION_CONFIDENCE = 0.55
HAND_TRACKING_CONFIDENCE = 0.55
HAND_CROP_SCALE = 1.75
MIN_HAND_SIZE = 45
MIN_SIGN_CONFIDENCE = 0.40
MIN_CLASS_MARGIN = 0.025
HISTORY_SIZE = 5
VOTES_REQUIRED = 3
MIRROR = True
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

SCRIPT_DIR = Path(__file__).resolve().parent

POSSIBLE_ROOTS = [
    SCRIPT_DIR,
    SCRIPT_DIR / "SignSpeak-AI-main",
    SCRIPT_DIR.parent / "SignSpeak-AI-main"
]

ROOT = next(
    (
        p for p in POSSIBLE_ROOTS
        if (p / "Tensorflow/workspace/models/pipeline.config").is_file()
    ),
    None
)

if ROOT is None:
    raise FileNotFoundError(
        "Cannot find Tensorflow/workspace/models/pipeline.config. "
        "Save this file in the outer SignSpeak-AI-main project folder."
    )

CONFIG_PATH = ROOT / "Tensorflow/workspace/models/pipeline.config"
MODEL_DIR = ROOT / "Tensorflow/workspace/models/my_ssd_mobnet"
LABEL_PATH = ROOT / "Tensorflow/workspace/annotations/label_map.pbtxt"


def parse_label_map(path):
    if not path.exists():
        return {}

    content = path.read_text(encoding="utf-8-sig")
    labels = {}

    for block in re.findall(r"item\s*\{([^}]*)\}", content, re.S):
        id_match = re.search(r"\bid\s*:\s*(\d+)", block)
        name_match = re.search(
            r"\bname\s*:\s*['\"]([^'\"]+)['\"]",
            block
        )

        if id_match and name_match:
            labels[int(id_match.group(1))] = name_match.group(1).strip()

    return labels


def ensure_alphabet_labels(num_classes):
    if num_classes != 26:
        raise RuntimeError(
            f"pipeline.config specifies {num_classes} classes, not 26. "
            "This detector is configured for ASL letters A-Z."
        )

    expected = {
        i: letter
        for i, letter in enumerate(LETTERS, 1)
    }

    current = parse_label_map(LABEL_PATH)

    if current != expected:
        LABEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        if LABEL_PATH.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = LABEL_PATH.with_name(
                f"label_map_backup_{stamp}.pbtxt"
            )
            shutil.copy2(LABEL_PATH, backup)
            print("[BACKUP] Previous label map:", backup)

        content = "\n".join(
            f"item {{\n    id: {idx}\n    name: '{letter}'\n}}"
            for idx, letter in expected.items()
        ) + "\n"

        LABEL_PATH.write_text(content, encoding="utf-8")
        print("[FIX] Label map configured for A-Z.")

    return expected


def get_latest_checkpoint():
    files = glob.glob(str(MODEL_DIR / "ckpt-*.index"))
    checkpoints = []

    for path in files:
        match = re.search(r"ckpt-(\d+)\.index$", path)

        if match:
            checkpoints.append((int(match.group(1)), path[:-6]))

    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found in {MODEL_DIR}"
        )

    return max(checkpoints, key=lambda item: item[0])[1]



def create_detector():
    model_path = SCRIPT_DIR / "models" / "asl_mobilenetv2" / "webcam_finetuned_model.h5"
    class_path = SCRIPT_DIR / "models" / "asl_mobilenetv2" / "class_indices.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Fine-tuned model not found: {model_path}")

    with open(class_path, "r", encoding="utf-8") as f:
        class_indices = json.load(f)

    labels = {int(index): letter for letter, index in class_indices.items()}

    print("[INFO] Project:", ROOT)
    print("[INFO] Loading MobileNetV2:", model_path)
    print("[INFO] Classes: A-Z")

    model = tf.keras.models.load_model(model_path)

    print("[SUCCESS] Webcam-fine-tuned MobileNetV2 loaded.")
    return model, labels



def extract_hand(frame, landmarks):
    height, width = frame.shape[:2]

    xs = np.array(
        [point.x * width for point in landmarks.landmark],
        dtype=np.float32
    )

    ys = np.array(
        [point.y * height for point in landmarks.landmark],
        dtype=np.float32
    )

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
        max(0, -y1),
        max(0, y2 - height),
        max(0, -x1),
        max(0, x2 - width),
        cv2.BORDER_REPLICATE
    )

    crop = cv2.resize(
        crop,
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    points = [
        (
            int(np.clip(point.x * width, 0, width - 1)),
            int(np.clip(point.y * height, 0, height - 1))
        )
        for point in landmarks.landmark
    ]

    box = (
        int(np.clip(left, 0, width - 1)),
        int(np.clip(top, 0, height - 1)),
        int(np.clip(right, 0, width - 1)),
        int(np.clip(bottom, 0, height - 1))
    )

    return crop, box, points



def classify_hands(crops, detector, labels):
    batch = []

    for crop in crops:
        image = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        batch.append(image)

    batch = np.asarray(batch, dtype=np.float32)
    batch = preprocess_input(batch)

    probabilities = detector.predict(batch, verbose=0)
    results = []

    for scores in probabilities:
        ranked = np.argsort(scores)[::-1]
        best_id = int(ranked[0])
        best_score = float(scores[best_id])

        second_score = (
            float(scores[ranked[1]]) if len(ranked) > 1 else 0.0
        )

        if (
            best_id in labels
            and best_score >= MIN_SIGN_CONFIDENCE
            and best_score - second_score >= MIN_CLASS_MARGIN
        ):
            results.append((labels[best_id], best_score))
        else:
            results.append((None, best_score))

    return results



def stable_prediction(history):
    valid = [
        (label, score)
        for label, score in history
        if label is not None
    ]

    if not valid:
        return None, 0.0

    counts = Counter(label for label, _ in valid)
    ranked = counts.most_common(2)

    best_label, best_count = ranked[0]
    runner_up_count = ranked[1][1] if len(ranked) > 1 else 0

    if best_count < VOTES_REQUIRED or best_count <= runner_up_count:
        return None, 0.0

    confidence_values = [
        score for label, score in valid
        if label == best_label
    ]

    return best_label, float(np.mean(confidence_values))


def main():
    detector, labels = create_detector()
    hands_solution = mp.solutions.hands

    frame_lock = threading.Lock()
    state_lock = threading.Lock()

    stop_event = threading.Event()
    reset_event = threading.Event()

    latest_frame = {
        "frame": None,
        "sequence": 0
    }

    state = {
        "hands": [],
        "fps": 0.0,
        "status": "Starting hand tracker...",
        "error": None
    }

    def inference_worker():
        histories = {}
        previous_sequence = -1
        fps_average = 0.0

        try:
            with hands_solution.Hands(
                static_image_mode=False,
                max_num_hands=MAX_HANDS,
                model_complexity=0,
                min_detection_confidence=HAND_DETECTION_CONFIDENCE,
                min_tracking_confidence=HAND_TRACKING_CONFIDENCE
            ) as hand_tracker:

                while not stop_event.is_set():
                    with frame_lock:
                        frame = latest_frame["frame"]
                        sequence = latest_frame["sequence"]

                    if frame is None or sequence == previous_sequence:
                        time.sleep(0.002)
                        continue

                    previous_sequence = sequence

                    if reset_event.is_set():
                        histories.clear()
                        reset_event.clear()

                    started = time.perf_counter()

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False

                    hand_results = hand_tracker.process(rgb)

                    detected_hands = (
                        hand_results.multi_hand_landmarks or []
                    )

                    handedness = (
                        hand_results.multi_handedness or []
                    )

                    crops = []
                    boxes = []
                    point_sets = []
                    keys = []

                    for index, hand_landmarks in enumerate(detected_hands):
                        extracted = extract_hand(frame, hand_landmarks)

                        if extracted is None:
                            continue

                        crop, box, points = extracted
                        key = f"Hand-{index}"

                        if (
                            index < len(handedness)
                            and handedness[index].classification
                        ):
                            key = handedness[index].classification[0].label

                        crops.append(crop)
                        boxes.append(box)
                        point_sets.append(points)
                        keys.append(key)

                    predictions = (
                        classify_hands(crops, detector, labels)
                        if crops
                        else []
                    )

                    visible_hands = []
                    active_keys = set(keys)

                    for key, box, points, prediction in zip(
                        keys,
                        boxes,
                        point_sets,
                        predictions
                    ):
                        history = histories.setdefault(
                            key,
                            deque(maxlen=HISTORY_SIZE)
                        )

                        history.append(prediction)

                        stable_label, stable_score = stable_prediction(
                            history
                        )

                        raw_label, raw_score = prediction

                        visible_hands.append({
                            "key": key,
                            "box": box,
                            "points": points,
                            "raw_label": raw_label,
                            "raw_score": raw_score,
                            "label": stable_label,
                            "score": stable_score
                        })

                    for old_key in list(histories):
                        if old_key not in active_keys:
                            histories.pop(old_key, None)

                    elapsed = max(
                        time.perf_counter() - started,
                        0.000001
                    )

                    instant_fps = 1.0 / elapsed

                    fps_average = (
                        instant_fps
                        if fps_average == 0
                        else 0.75 * fps_average + 0.25 * instant_fps
                    )

                    if not visible_hands:
                        message = "NO HAND - FACE AND BACKGROUND IGNORED"
                    elif any(hand["label"] for hand in visible_hands):
                        message = "SIGN STABLE - HOLD YOUR HAND SIGN"
                    else:
                        message = "HAND FOUND - ANALYZING SIGN"

                    with state_lock:
                        state["hands"] = visible_hands
                        state["fps"] = fps_average
                        state["status"] = message
                        state["error"] = None

        except Exception as exc:
            import traceback
            traceback.print_exc()

            with state_lock:
                state["error"] = str(exc)
                state["status"] = "INFERENCE ERROR - SEE TERMINAL"

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError(
            "Webcam could not be opened. Close other camera applications."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    worker = threading.Thread(
        target=inference_worker,
        daemon=True
    )

    worker.start()
    mirror = MIRROR

    print("\n[SIGNSPEAK AI] HAND-ONLY DETECTION ACTIVE")
    print("Controls: R = reset | F = mirror toggle | Q / ESC = quit\n")

    try:
        while camera.isOpened():
            success, frame = camera.read()

            if not success:
                print("[ERROR] Camera frame unavailable.")
                break

            # Keep the original camera frame for model inference.
            with frame_lock:
                latest_frame["frame"] = frame.copy()
                latest_frame["sequence"] += 1

            with state_lock:
                hands = list(state["hands"])
                fps = float(state["fps"])
                message = state["status"]
                error = state["error"]

            height, width = frame.shape[:2]

            for hand in hands:
                x1, y1, x2, y2 = hand["box"]

                color = (
                    (40, 230, 40)
                    if hand["label"]
                    else (0, 190, 255)
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                for px, py in hand["points"]:
                    cv2.circle(
                        frame,
                        (px, py),
                        2,
                        (255, 180, 0),
                        -1
                    )

                if hand["label"]:
                    text = (
                        f"{hand['label']} "
                        f"{hand['score'] * 100:.0f}%"
                    )
                elif hand["raw_label"]:
                    text = f"Checking {hand['raw_label']}..."
                else:
                    text = "Uncertain sign"

                cv2.putText(
                    frame,
                    text,
                    (x1, max(24, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    color,
                    2,
                    cv2.LINE_AA
                )

            stable_hands = [
                hand for hand in hands
                if hand["label"]
            ]

            if stable_hands:
                best = max(
                    stable_hands,
                    key=lambda item: item["score"]
                )

                headline = (
                    f"DETECTED LETTER: {best['label']} "
                    f"({best['score'] * 100:.0f}%)"
                )

                headline_color = (40, 240, 40)

            elif hands:
                headline = "HAND FOUND - ANALYZING"
                headline_color = (0, 210, 255)

            else:
                headline = "SHOW YOUR HAND TO THE CAMERA"
                headline_color = (220, 220, 220)

            cv2.rectangle(
                frame,
                (0, 0),
                (width, 88),
                (20, 20, 20),
                -1
            )

            cv2.putText(
                frame,
                headline,
                (14, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.76,
                headline_color,
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                message[:100],
                (14, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (220, 220, 220),
                1,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                f"INFERENCE FPS: {fps:.1f}",
                (width - 180, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (210, 210, 210),
                1
            )

            cv2.putText(
                frame,
                "R: RESET | F: FLIP | Q: QUIT",
                (14, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (240, 240, 240),
                1
            )

            if error:
                cv2.putText(
                    frame,
                    "ERROR: " + error[:80],
                    (14, height - 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1
                )

            # Mirror only the display, not the classifier input.
            if mirror:
                frame = cv2.flip(frame, 1)

            cv2.imshow(
                "SignSpeak AI - Hand-Only ASL Detection",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

            if key in (ord("r"), ord("R")):
                reset_event.set()

            if key in (ord("f"), ord("F")):
                mirror = not mirror

    finally:
        stop_event.set()
        worker.join(timeout=3)
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

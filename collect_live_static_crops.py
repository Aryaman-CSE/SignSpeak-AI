import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "datasets" / "webcam_live_crops"
LETTERS = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in "JZ"]
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
INPUT_SIZE = 320
CLASSIFIER_SIZE = 160
MAX_HANDS = 2
HAND_DETECTION_CONFIDENCE = 0.55
HAND_TRACKING_CONFIDENCE = 0.55
HAND_CROP_SCALE = 1.75
MIN_HAND_SIZE = 45
TRAIN_TARGET = 40
VALIDATION_TARGET = 10
CAPTURE_INTERVAL = 0.35


def extract_hand(frame, landmarks):
    height, width = frame.shape[:2]
    xs = np.asarray([p.x * width for p in landmarks.landmark], dtype=np.float32)
    ys = np.asarray([p.y * height for p in landmarks.landmark], dtype=np.float32)
    left, right = float(xs.min()), float(xs.max())
    top, bottom = float(ys.min()), float(ys.max())
    box_width, box_height = right - left, bottom - top
    if max(box_width, box_height) < MIN_HAND_SIZE:
        return None

    side = max(box_width, box_height) * HAND_CROP_SCALE
    side = max(side, MIN_HAND_SIZE * 2)
    center_x, center_y = (left + right) / 2.0, (top + bottom) / 2.0
    x1, y1 = int(round(center_x - side / 2.0)), int(round(center_y - side / 2.0))
    x2, y2 = int(round(center_x + side / 2.0)), int(round(center_y + side / 2.0))
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(width, x2), min(height, y2)
    if sx2 <= sx1 or sy2 <= sy1:
        return None

    crop = frame[sy1:sy2, sx1:sx2]
    crop = cv2.copyMakeBorder(
        crop,
        max(0, -y1), max(0, y2 - height),
        max(0, -x1), max(0, x2 - width),
        cv2.BORDER_REPLICATE,
    )
    crop320 = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    crop160 = cv2.resize(crop320, (CLASSIFIER_SIZE, CLASSIFIER_SIZE), interpolation=cv2.INTER_AREA)
    box = (int(sx1), int(sy1), int(sx2), int(sy2))
    return crop160, box


def count_files(folder):
    return len(list(folder.glob("*.jpg"))) if folder.exists() else 0


def main():
    # Resume an existing collection instead of discarding partial progress.
    for split in ("train", "validation"):
        for letter in LETTERS:
            (OUT / split / letter).mkdir(parents=True, exist_ok=True)

    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam. Close other camera applications and retry.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    current_letter = None
    recording = False
    last_save = 0.0
    validation_phase = False
    counts = {
        letter: {
            "train": count_files(OUT / "train" / letter),
            "validation": count_files(OUT / "validation" / letter),
        }
        for letter in LETTERS
    }

    print("\nSIGNSPEAK LIVE CROP COLLECTION")
    print("Use the real webcam. Select any static ASL letter with its keyboard key A-Z, except J/Z.")
    print("SPACE = start/stop capture | Q or ESC = quit")
    print("For each letter: collect 40 training crops, then change angle/distance and collect 10 validation crops.")
    print("Keep the entire hand in view. The collector saves only when the live MediaPipe detector finds a hand.\n")

    hands_api = mp.solutions.hands
    try:
        with hands_api.Hands(
            static_image_mode=False,
            max_num_hands=MAX_HANDS,
            model_complexity=0,
            min_detection_confidence=HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=HAND_TRACKING_CONFIDENCE,
        ) as tracker:
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Webcam frame could not be read.")

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                result = tracker.process(rgb)
                landmarks_list = result.multi_hand_landmarks or []

                candidates = []
                for lm in landmarks_list:
                    extracted = extract_hand(frame, lm)
                    if extracted is None:
                        continue
                    crop, box = extracted
                    area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
                    candidates.append((area, crop, box))
                candidates.sort(key=lambda item: item[0], reverse=True)

                display = frame.copy()
                if candidates:
                    _, current_crop, box = candidates[0]
                    cv2.rectangle(display, (box[0], box[1]), (box[2], box[3]), (0, 220, 0), 2)
                    cv2.putText(display, "HAND DETECTED", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
                else:
                    current_crop = None
                    cv2.putText(display, "NO HAND DETECTED - adjust position/light", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

                if current_letter:
                    tr = counts[current_letter]["train"]
                    va = counts[current_letter]["validation"]
                    phase = "VALIDATION" if validation_phase else "TRAIN"
                    msg = f"LETTER {current_letter} | {phase} {va if validation_phase else tr}/{VALIDATION_TARGET if validation_phase else TRAIN_TARGET}"
                else:
                    msg = "Press a letter key A-Z to select (J/Z use dynamic model)"
                cv2.putText(display, msg, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                status = "RECORDING" if recording else "PAUSED"
                cv2.putText(display, f"{status} | SPACE toggle | Q quit", (12, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                now = time.monotonic()
                if recording and current_letter and current_crop is not None and now - last_save >= CAPTURE_INTERVAL:
                    split = "validation" if validation_phase else "train"
                    limit = VALIDATION_TARGET if validation_phase else TRAIN_TARGET
                    if counts[current_letter][split] < limit:
                        index = counts[current_letter][split] + 1
                        dest = OUT / split / current_letter / f"{current_letter}_{index:04d}.jpg"
                        if not cv2.imwrite(str(dest), current_crop, [cv2.IMWRITE_JPEG_QUALITY, 96]):
                            raise RuntimeError(f"Could not save {dest}")
                        counts[current_letter][split] += 1
                        last_save = now
                    if counts[current_letter][split] >= limit:
                        recording = False
                        if not validation_phase:
                            print(f"{current_letter}: 40 training crops complete. Change hand angle/distance, then press SPACE for validation.")
                        else:
                            print(f"{current_letter}: complete (40 train, 10 validation). Select the next letter.")
                            current_letter = None
                            validation_phase = False

                cv2.imshow("SignSpeak - Live Crop Collection", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                elif key == 32:
                    if not current_letter:
                        print("Select a letter first using A-X (excluding J/Z).")
                        recording = False
                    else:
                        if counts[current_letter]["train"] >= TRAIN_TARGET and counts[current_letter]["validation"] < VALIDATION_TARGET:
                            validation_phase = True
                        if counts[current_letter]["validation"] >= VALIDATION_TARGET:
                            print(f"{current_letter} is already complete. Select another letter.")
                            recording = False
                        else:
                            recording = not recording
                            last_save = 0.0
                elif ord("a") <= key <= ord("z") or ord("A") <= key <= ord("Z"):
                    letter = chr(key).upper()
                    if letter in LETTERS:
                        current_letter = letter
                        recording = False
                        validation_phase = counts[letter]["train"] >= TRAIN_TARGET
                        print(f"Selected {letter}: train {counts[letter]['train']}/{TRAIN_TARGET}, validation {counts[letter]['validation']}/{VALIDATION_TARGET}")
                    elif letter in "JZ":
                        print(f"{letter} is dynamic and remains on the J/Z LSTM pipeline.")

    finally:
        camera.release()
        cv2.destroyAllWindows()

    summary = {
        letter: counts[letter] for letter in LETTERS
    }
    (OUT / "collection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSaved live hand crops to:", OUT)
    print(json.dumps(summary, indent=2))
    incomplete = [letter for letter in LETTERS if counts[letter]["train"] < TRAIN_TARGET or counts[letter]["validation"] < VALIDATION_TARGET]
    if incomplete:
        print("Incomplete letters:", ", ".join(incomplete))
    else:
        print("All 24 static letters have 40 training and 10 validation crops.")


if __name__ == "__main__":
    main()

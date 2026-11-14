
import cv2
import json
import numpy as np
import tensorflow as tf
import mediapipe as mp

from pathlib import Path
from collections import deque, Counter
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "asl_mobilenetv2" / "webcam_finetuned_model.h5"
CLASS_PATH = ROOT / "models" / "asl_mobilenetv2" / "class_indices.json"

IMG_SIZE = 160
PADDING = 0.35
SMOOTHING_FRAMES = 10
STABILITY_FRAMES = 7
REQUIRED_VOTES = 5
CONFIDENCE_THRESHOLD = 0.60

print("Loading SignSpeak AI...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.60,
    min_tracking_confidence=0.65
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

probability_history = deque(maxlen=SMOOTHING_FRAMES)
prediction_history = deque(maxlen=STABILITY_FRAMES)

frame_count = 0
smoothed_probs = None
stable_label = "Show your hand"
mirror = True

print("F = Mirror ON/OFF | Q = Quit")

while True:
    ret, raw_frame = cap.read()
    if not ret:
        break

    frame_count += 1
    height, width = raw_frame.shape[:2]

    # Detect on the original camera frame.
    rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    display = raw_frame.copy()
    last_top3 = []

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        xs = [p.x * width for p in hand_landmarks.landmark]
        ys = [p.y * height for p in hand_landmarks.landmark]

        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))

        hand_w = max(1, x_max - x_min)
        hand_h = max(1, y_max - y_min)

        # Make a square crop without stretching the hand.
        side = int(max(hand_w, hand_h) * (1 + 2 * PADDING))
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2

        half = side // 2
        sx1, sy1 = cx - half, cy - half
        sx2, sy2 = sx1 + side, sy1 + side

        # Pad image edges if the hand is near a camera boundary.
        crop = raw_frame[
            max(0, sy1):min(height, sy2),
            max(0, sx1):min(width, sx2)
        ]

        if crop.size > 0:
            top = max(0, -sy1)
            bottom = max(0, sy2 - height)
            left = max(0, -sx1)
            right = max(0, sx2 - width)

            crop = cv2.copyMakeBorder(
                crop, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )

            # Square crop retains the original hand proportions.
            image = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = preprocess_input(image.astype(np.float32))
            image = np.expand_dims(image, axis=0)

            # Run prediction on alternate frames to reduce CPU load.
            if frame_count % 2 == 0:
                probabilities = model.predict(image, verbose=0)[0]
                probability_history.append(probabilities)

                smoothed_probs = np.mean(
                    np.array(probability_history), axis=0
                )

                top_index = int(np.argmax(smoothed_probs))
                confidence = float(smoothed_probs[top_index])

                if confidence >= CONFIDENCE_THRESHOLD:
                    prediction_history.append(labels[top_index])
                else:
                    prediction_history.append("Uncertain")

            if smoothed_probs is not None:
                last_top3 = np.argsort(smoothed_probs)[-3:][::-1]

                counts = Counter(prediction_history)
                candidate, votes = counts.most_common(1)[0]

                if candidate != "Uncertain" and votes >= REQUIRED_VOTES:
                    stable_label = candidate
                else:
                    stable_label = "Stabilizing..."

        # Draw the hand box and landmarks on the original frame.
        bx1 = max(0, sx1)
        by1 = max(0, sy1)
        bx2 = min(width - 1, sx2)
        by2 = min(height - 1, sy2)

        cv2.rectangle(
            display, (bx1, by1), (bx2, by2), (0, 255, 0), 2
        )

        mp_draw.draw_landmarks(
            display, hand_landmarks, mp_hands.HAND_CONNECTIONS
        )

        cv2.putText(
            display, f"Sign: {stable_label}", (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )

        if smoothed_probs is not None:
            y = 75
            for idx in last_top3:
                text = (
                    f"{labels[int(idx)]}: "
                    f"{smoothed_probs[idx] * 100:.1f}%"
                )
                cv2.putText(
                    display, text, (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2
                )
                y += 28

    else:
        probability_history.clear()
        prediction_history.clear()
        smoothed_probs = None
        stable_label = "No hand detected"

        cv2.putText(
            display, stable_label, (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
        )

    # Mirror only the displayed image, never the classifier input.
    if mirror:
        display = cv2.flip(display, 1)

    cv2.putText(
        display,
        f"Mirror: {'ON' if mirror else 'OFF'} | F: Toggle | Q: Quit",
        (15, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
    )

    cv2.imshow("SignSpeak AI - Improved Classifier", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("f"):
        mirror = not mirror
    elif key == ord("q"):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()



import cv2
import mediapipe as mp
import numpy as np
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "datasets" / "dynamic_signs"

SEQUENCE_LENGTH = 30
TARGET = 100
LETTERS = ["J", "Z"]

for letter in LETTERS:
    (OUT / letter).mkdir(parents=True, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

print("SPACE = record one sequence | N = next letter | Q = quit")

current = 0

while current < len(LETTERS):
    letter = LETTERS[current]
    folder = OUT / letter

    existing = list(folder.glob("*.npy"))
    if len(existing) >= TARGET:
        print(f"{letter} complete: {len(existing)}/{TARGET}")
        current += 1
        continue

    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    count = len(list(folder.glob("*.npy")))
    cv2.putText(display, f"Sign: {letter} | Sequences: {count}/{TARGET}",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2)
    cv2.putText(display, "SPACE = record | N = skip | Q = quit",
                (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)

    cv2.imshow("SignSpeak - Dynamic Sign Collection", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        sequence = []
        print(f"Recording {letter}...")

        while len(sequence) < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                landmarks = result.multi_hand_landmarks[0].landmark
                points = np.array(
                    [[p.x, p.y, p.z] for p in landmarks],
                    dtype=np.float32
                )
                sequence.append(points)
            else:
                sequence.append(np.zeros((21, 3), dtype=np.float32))

            progress = frame.copy()
            cv2.putText(progress,
                        f"Recording {letter}: {len(sequence)}/{SEQUENCE_LENGTH}",
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)
            cv2.imshow("SignSpeak - Dynamic Sign Collection", progress)
            cv2.waitKey(1)

        if len(sequence) == SEQUENCE_LENGTH:
            existing = list(folder.glob("*.npy"))
            numbers = [
                int(p.stem) for p in existing if p.stem.isdigit()
            ]
            next_number = max(numbers, default=0) + 1

            path = folder / f"{next_number:04d}.npy"
            np.save(path, np.array(sequence, dtype=np.float32))
            print(f"Saved {letter}: {next_number}/{TARGET}")

    elif key == ord("n"):
        current += 1

    elif key == ord("q"):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
print("Dynamic collection finished.")

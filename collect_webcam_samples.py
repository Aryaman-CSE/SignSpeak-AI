import cv2
import mediapipe as mp
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "datasets" / "webcam_samples"

targets = {
    "D": 100,
    "Q": 100,
    "T": 100,
    "X": 100
}

letters = list(targets.keys())

for letter in letters:
    (OUT / letter).mkdir(parents=True, exist_ok=True)

def saved_count(letter):
    return len(list((OUT / letter).glob("*.jpg")))

current = 0
cap = cv2.VideoCapture(0)

hands = mp.solutions.hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

print("SPACE = capture | N = skip | Q = quit")

while current < len(letters):
    letter = letters[current]
    target = targets[letter]
    saved = saved_count(letter)

    if saved >= target:
        print(f"Completed {letter}: {saved}/{target}")
        current += 1
        continue

    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    display = frame.copy()
    crop = None

    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0].landmark
        h, w = frame.shape[:2]

        xs = [int(p.x * w) for p in lm]
        ys = [int(p.y * h) for p in lm]

        x1, x2 = max(0, min(xs)), min(w, max(xs))
        y1, y2 = max(0, min(ys)), min(h, max(ys))

        side = max(x2-x1, y2-y1)
        cx, cy = (x1+x2)//2, (y1+y2)//2
        half = int(side * 0.85)

        sx1, sy1 = max(0, cx-half), max(0, cy-half)
        sx2, sy2 = min(w, cx+half), min(h, cy+half)

        candidate = frame[sy1:sy2, sx1:sx2]

        if candidate.size:
            crop = candidate
            cv2.rectangle(display, (x1,y1), (x2,y2), (0,255,0), 2)

    cv2.putText(display, f"Sign: {letter} | Saved: {saved}/{target}",
                (15,35), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0,255,0), 2)

    cv2.imshow("SignSpeak - Data Collection", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(" ") and crop is not None:
        existing = list((OUT / letter).glob("*.jpg"))
        numbers = [int(p.stem) for p in existing if p.stem.isdigit()]
        next_number = max(numbers, default=0) + 1
        path = OUT / letter / f"{next_number:04d}.jpg"

        if cv2.imwrite(str(path), crop):
            print(f"Saved {letter}: {next_number}/{target}")

    elif key == ord("n"):
        current += 1

    elif key == ord("q"):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
print("Collection finished.")

import cv2
import mediapipe as mp
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "datasets" / "webcam_samples"

targets = {"D": 200, "Q": 200, "U": 200, "S": 200, "T": 200}
letters = list(targets)

cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

current = 0
print("SPACE = capture | N = skip | Q = quit")

while current < len(letters):
    letter = letters[current]
    folder = OUT / letter
    folder.mkdir(parents=True, exist_ok=True)
    saved = len(list(folder.glob("*.jpg")))
    target = targets[letter]

    if saved >= target:
        print(f"Completed {letter}: {saved}/{target}")
        current += 1
        continue

    ret, frame = cap.read()
    if not ret:
        break

    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    display = frame.copy()
    crop = None

    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0].landmark
        h, w = frame.shape[:2]
        xs = [int(p.x*w) for p in lm]
        ys = [int(p.y*h) for p in lm]
        x1, x2 = max(0,min(xs)), min(w,max(xs))
        y1, y2 = max(0,min(ys)), min(h,max(ys))
        side = max(x2-x1,y2-y1)
        cx, cy = (x1+x2)//2, (y1+y2)//2
        half = int(side*0.85)
        crop = frame[max(0,cy-half):min(h,cy+half),
                     max(0,cx-half):min(w,cx+half)]
        if crop.size == 0:
            crop = None
        cv2.rectangle(display,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.putText(display,f"Sign: {letter} | {saved}/{target}",
                (15,35),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
    cv2.imshow("SignSpeak Data Collection",display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(" ") and crop is not None:
        existing = [int(p.stem) for p in folder.glob("*.jpg") if p.stem.isdigit()]
        number = max(existing,default=0)+1
        if cv2.imwrite(str(folder/f"{number:04d}.jpg"),crop):
            print(f"Saved {letter}: {number}/{target}")
    elif key == ord("n"):
        current += 1
    elif key == ord("q"):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
print("Collection finished.")

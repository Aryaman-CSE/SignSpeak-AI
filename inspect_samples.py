import cv2
from pathlib import Path

root = Path("datasets/webcam_samples")
letters = ["D", "E", "I", "K", "M", "N", "T", "X"]
tiles = []

for letter in letters:
    files = sorted((root / letter).glob("*.jpg"))[:6]

    for i, path in enumerate(files):
        img = cv2.imread(str(path))
        if img is None:
            continue

        img = cv2.resize(img, (160, 160))
        cv2.putText(img, f"{letter} - {i+1}", (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        tiles.append((letter, img))

canvas = cv2.imread(str(root / "E" / "0001.jpg"))
if canvas is None:
    raise RuntimeError("Could not load sample image")

sheet = 255 * (canvas * 0)
sheet = cv2.resize(sheet, (160*6, 160*8))

for i, (_, img) in enumerate(tiles):
    x = (i % 6) * 160
    y = (i // 6) * 160
    sheet[y:y+160, x:x+160] = img

cv2.imwrite("problematic_sign_samples.jpg", sheet)
print("Saved problematic_sign_samples.jpg")

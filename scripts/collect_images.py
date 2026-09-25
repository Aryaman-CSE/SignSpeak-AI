import cv2
import os

print("=== SignSpeak AI - Image Collection ===")

class_name = input("Enter sign name: ").strip()

if not class_name:
    print("Sign name cannot be empty.")
    exit()

save_dir = os.path.join(
    "Tensorflow",
    "workspace",
    "images",
    "collected",
    class_name
)

os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print(f"\nCollecting images for: {class_name}")
print("Press SPACE to capture an image.")
print("Press Q to quit.\n")

count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam frame.")
        break

    cv2.putText(
        frame,
        f"{class_name} | Images: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("SignSpeak AI - Image Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        filename = os.path.join(save_dir, f"{class_name}.{count:04d}.jpg")
        cv2.imwrite(filename, frame)
        count += 1
        print(f"Saved: {filename}")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nCollection finished.")
print(f"Total images collected for {class_name}: {count}")
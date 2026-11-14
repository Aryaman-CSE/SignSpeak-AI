import cv2
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "asl_mobilenetv2" / "final_model.h5"
CLASS_PATH = ROOT / "models" / "asl_mobilenetv2" / "class_indices.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check camera permissions.")

print("Webcam test started. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Center square crop
    h, w = frame.shape[:2]
    side = min(h, w)
    x1 = (w - side) // 2
    y1 = (h - side) // 2
    crop = frame[y1:y1+side, x1:x1+side]

    image = cv2.resize(crop, (160, 160))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_input(image.astype(np.float32))
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image, verbose=0)[0]
    top_indices = np.argsort(predictions)[-3:][::-1]

    y = 35
    for idx in top_indices:
        text = f"{labels[int(idx)]}: {predictions[idx]*100:.1f}%"
        cv2.putText(frame, text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)
        y += 35

    cv2.rectangle(frame, (x1, y1), (x1+side, y1+side),
                  (255, 0, 0), 2)

    cv2.imshow("SignSpeak AI - Classifier Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "asl_mobilenetv2"
TEMP = MODEL_DIR / "webcam_training_data_v2"

model = tf.keras.models.load_model(
    MODEL_DIR / "webcam_finetuned_v2.h5"
)

gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

data = gen.flow_from_directory(
    str(TEMP / "validation"),
    target_size=(160, 160),
    batch_size=32,
    classes=sorted(p.name for p in (TEMP / "validation").iterdir() if p.is_dir()),
    class_mode="categorical",
    shuffle=False
)

predictions = model.predict(data, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = data.classes
labels = list(data.class_indices.keys())

print("\n========== PER-LETTER ACCURACY ==========")

for i, letter in enumerate(labels):
    mask = y_true == i
    total = int(mask.sum())
    correct = int((y_pred[mask] == i).sum())
    accuracy = 100 * correct / total if total else 0
    print(f"{letter}: {accuracy:.2f}% ({correct}/{total})")

print("\n========== TOP CONFUSIONS ==========")

confusions = []

for actual in range(len(labels)):
    for predicted in range(len(labels)):
        if actual != predicted:
            count = int(np.sum((y_true == actual) & (y_pred == predicted)))
            if count:
                confusions.append((count, labels[actual], labels[predicted]))

for count, actual, predicted in sorted(confusions, reverse=True)[:20]:
    print(f"{actual} predicted as {predicted}: {count}")

print("\nOverall validation accuracy:",
      f"{100*np.mean(y_true == y_pred):.2f}%")

import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "asl_mobilenetv2"
TEMP = MODEL_DIR / "webcam_training_data_round3"

model = tf.keras.models.load_model(
    MODEL_DIR / "webcam_finetuned_round3.h5"
)

gen = ImageDataGenerator(preprocessing_function=preprocess_input)

data = gen.flow_from_directory(
    str(TEMP / "validation"),
    target_size=(160, 160),
    batch_size=32,
    classes=sorted(p.name for p in (TEMP / "validation").iterdir() if p.is_dir()),
    class_mode="categorical",
    shuffle=False
)

pred = np.argmax(model.predict(data, verbose=1), axis=1)
true = data.classes
labels = list(data.class_indices.keys())

print("\nROUND 3 PER-LETTER ACCURACY")
for i, letter in enumerate(labels):
    mask = true == i
    total = int(mask.sum())
    correct = int((pred[mask] == i).sum())
    print(f"{letter}: {100*correct/total:.2f}% ({correct}/{total})")

print("\nTOP CONFUSIONS")
errors = []
for a in range(len(labels)):
    for p in range(len(labels)):
        count = int(np.sum((true == a) & (pred == p)))
        if a != p and count:
            errors.append((count, labels[a], labels[p]))

for count, actual, predicted in sorted(errors, reverse=True)[:20]:
    print(f"{actual} predicted as {predicted}: {count}")

print("\nOverall accuracy:", round(100*np.mean(pred == true), 2), "%")

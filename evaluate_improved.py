import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

root = Path(__file__).resolve().parent
folder = root / "models" / "asl_mobilenetv2"

model = tf.keras.models.load_model(folder / "improved_model.h5")

with open(folder / "class_indices.json") as f:
    indices = json.load(f)

data = ImageDataGenerator(
    preprocessing_function=preprocess_input
).flow_from_directory(
    str(root / "datasets" / "asl_alphabet" / "test"),
    target_size=(160, 160),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

pred = model.predict(data, verbose=1)
actual = data.classes
predicted = np.argmax(pred, axis=1)

names = [k for k, v in sorted(indices.items(), key=lambda x: x[1])]

print("\nIMPROVED TEST ACCURACY:",
      round(np.mean(actual == predicted) * 100, 2), "%")

print("\nPER-LETTER RESULTS:")
print(classification_report(
    actual, predicted,
    target_names=names,
    zero_division=0
))

cm = confusion_matrix(actual, predicted)

print("\nTOP CONFUSIONS:")
errors = []
for i in range(26):
    for j in range(26):
        if i != j and cm[i, j] > 0:
            errors.append((cm[i, j], names[i], names[j]))

for count, a, b in sorted(errors, reverse=True)[:20]:
    print(f"{a} predicted as {b}: {count}")

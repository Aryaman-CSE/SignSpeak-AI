import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parent
TEST_DIR = ROOT / "datasets" / "asl_alphabet" / "test"
MODEL_PATH = ROOT / "models" / "asl_mobilenetv2" / "final_model.h5"
CLASS_PATH = ROOT / "models" / "asl_mobilenetv2" / "class_indices.json"

with open(CLASS_PATH) as f:
    class_indices = json.load(f)

model = tf.keras.models.load_model(MODEL_PATH)

generator = ImageDataGenerator(preprocessing_function=preprocess_input)
data = generator.flow_from_directory(
    str(TEST_DIR),
    target_size=(160, 160),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

pred = model.predict(data, verbose=1)
actual = data.classes
predicted = np.argmax(pred, axis=1)

names = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]
cm = confusion_matrix(actual, predicted)

report = classification_report(
    actual, predicted, target_names=names,
    output_dict=True, zero_division=0
)

print("\n========== LETTER PERFORMANCE ==========")
for letter in names:
    print(f"{letter}: Recall {report[letter]['recall']*100:.2f}% | Precision {report[letter]['precision']*100:.2f}%")

print("\n========== TOP CONFUSIONS ==========")
errors = []
for i in range(len(names)):
    for j in range(len(names)):
        if i != j and cm[i, j] > 0:
            errors.append((cm[i, j], names[i], names[j]))

for count, a, b in sorted(errors, reverse=True)[:30]:
    print(f"{a} predicted as {b}: {count}")

print("\nOverall accuracy:", round(np.mean(actual == predicted)*100, 2), "%")

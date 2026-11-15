import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models/asl_mobilenetv2/webcam_finetuned_round4.h5"
DATA = ROOT / "datasets/asl_alphabet/test"
OUT = ROOT / "models/asl_mobilenetv2"

print("Loading production static model...")
model = tf.keras.models.load_model(MODEL)

generator = ImageDataGenerator(
    preprocessing_function=preprocess_input
).flow_from_directory(
    str(DATA),
    target_size=(160, 160),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

if generator.num_classes != 26:
    raise ValueError(f"Expected 26 classes, found {generator.num_classes}")

loss, accuracy = model.evaluate(generator, verbose=1)
generator.reset()
probabilities = model.predict(generator, verbose=1)

predicted = np.argmax(probabilities, axis=1)
actual = generator.classes
names = list(generator.class_indices.keys())

print("\nOVERALL TEST ACCURACY:", round(accuracy * 100, 2), "%")
print("\nPER-LETTER RESULTS:")
print(classification_report(
    actual, predicted,
    labels=list(range(26)),
    target_names=names,
    zero_division=0
))
print("\nCONFUSION MATRIX:")
print(confusion_matrix(actual, predicted))

report = classification_report(
    actual, predicted,
    labels=list(range(26)),
    target_names=names,
    output_dict=True,
    zero_division=0
)

with open(OUT / "static_evaluation_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("\nReport saved to models/asl_mobilenetv2/static_evaluation_report.json")


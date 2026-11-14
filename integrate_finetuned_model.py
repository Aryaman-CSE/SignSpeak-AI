from pathlib import Path
from datetime import datetime
import re

path = Path("realtime_signspeak.py")

if not path.exists():
    raise FileNotFoundError("realtime_signspeak.py not found in this folder.")

source = path.read_text(encoding="utf-8-sig")

# Preserve the original main application.
backup = path.with_name(
    f"realtime_signspeak_backup_{datetime.now():%Y%m%d_%H%M%S}.py"
)
backup.write_text(source, encoding="utf-8")

# Add required imports.
if "import json" not in source:
    source = source.replace("import re\n", "import re\nimport json\n", 1)

if "from tensorflow.keras.applications.mobilenet_v2 import preprocess_input" not in source:
    source = source.replace(
        "import tensorflow as tf\n",
        "import tensorflow as tf\nfrom tensorflow.keras.applications.mobilenet_v2 import preprocess_input\n",
        1
    )

# Remove the old Object Detection API imports.
source = source.replace(
    "from object_detection.builders import model_builder\n"
    "from object_detection.utils import config_util\n",
    ""
)

# Replace the old SSD loader with the fine-tuned MobileNetV2 loader.
new_loader = '''
def create_detector():
    model_path = ROOT / "models" / "asl_mobilenetv2" / "webcam_finetuned_model.h5"
    class_path = ROOT / "models" / "asl_mobilenetv2" / "class_indices.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Fine-tuned model not found: {model_path}")

    with open(class_path, "r", encoding="utf-8") as f:
        class_indices = json.load(f)

    labels = {int(index): letter for letter, index in class_indices.items()}

    print("[INFO] Project:", ROOT)
    print("[INFO] Loading MobileNetV2:", model_path)
    print("[INFO] Classes: A-Z")

    model = tf.keras.models.load_model(model_path)

    print("[SUCCESS] Webcam-fine-tuned MobileNetV2 loaded.")
    return model, labels


'''

source, count = re.subn(
    r"def create_detector\(\):.*?(?=\ndef extract_hand\()",
    new_loader,
    source,
    count=1,
    flags=re.S
)
if count != 1:
    raise RuntimeError("Could not safely locate create_detector().")

# Replace SSD detection with MobileNetV2 image classification.
new_classifier = '''
def classify_hands(crops, detector, labels):
    batch = []

    for crop in crops:
        image = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        batch.append(image)

    batch = np.asarray(batch, dtype=np.float32)
    batch = preprocess_input(batch)

    probabilities = detector.predict(batch, verbose=0)
    results = []

    for scores in probabilities:
        ranked = np.argsort(scores)[::-1]
        best_id = int(ranked[0])
        best_score = float(scores[best_id])

        second_score = (
            float(scores[ranked[1]]) if len(ranked) > 1 else 0.0
        )

        if (
            best_id in labels
            and best_score >= MIN_SIGN_CONFIDENCE
            and best_score - second_score >= MIN_CLASS_MARGIN
        ):
            results.append((labels[best_id], best_score))
        else:
            results.append((None, best_score))

    return results


'''

source, count = re.subn(
    r"def classify_hands\(crops, detector, labels\):.*?(?=\ndef stable_prediction\()",
    new_classifier,
    source,
    count=1,
    flags=re.S
)
if count != 1:
    raise RuntimeError("Could not safely locate classify_hands().")

# Keep raw frames for inference; mirror only the displayed image.
old_flip = '''            if mirror:
                frame = cv2.flip(frame, 1)

            with frame_lock:
'''

new_flip = '''            # Keep the original camera frame for model inference.
            with frame_lock:
'''

if old_flip not in source:
    raise RuntimeError("Could not locate camera mirror section.")

source = source.replace(old_flip, new_flip, 1)

display_marker = '''            cv2.imshow(
                "SignSpeak AI - Hand-Only ASL Detection",
                frame
            )
'''

display_replacement = '''            # Mirror only the display, not the classifier input.
            if mirror:
                frame = cv2.flip(frame, 1)

            cv2.imshow(
                "SignSpeak AI - Hand-Only ASL Detection",
                frame
            )
'''

if display_marker not in source:
    raise RuntimeError("Could not locate display section.")

source = source.replace(display_marker, display_replacement, 1)

# Check syntax before replacing the original.
compile(source, str(path), "exec")
path.write_text(source, encoding="utf-8")

print("\\n[SUCCESS] Main project updated.")
print("[BACKUP]", backup)
print("[MODEL] models/asl_mobilenetv2/webcam_finetuned_model.h5")
print("[PRESERVED] Original SSD checkpoint and all trained models.")

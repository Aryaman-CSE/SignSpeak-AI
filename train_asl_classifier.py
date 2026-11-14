import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# SignSpeak AI - separate A-Z image classifier.
# This does NOT modify the original SSD MobileNet checkpoint.

DATA_DIR = Path("datasets/asl_alphabet")
OUTPUT_DIR = Path("models/asl_mobilenetv2")
IMG_SIZE = 160
BATCH_SIZE = 32
EPOCHS = 5
SEED = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

required_dirs = [DATA_DIR / split for split in ("train", "validation", "test")]
for folder in required_dirs:
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {folder.resolve()}")

print("TensorFlow:", tf.__version__)
print("Devices:", tf.config.list_physical_devices())
print("Training on CPU if no TensorFlow GPU is available.")
print("Input size:", IMG_SIZE, "Batch size:", BATCH_SIZE)

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=8,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.10,
    brightness_range=(0.85, 1.15),
    horizontal_flip=False
)
eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_datagen.flow_from_directory(
    str(DATA_DIR / "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=SEED
)
val_data = eval_datagen.flow_from_directory(
    str(DATA_DIR / "validation"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)
test_data = eval_datagen.flow_from_directory(
    str(DATA_DIR / "test"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

if train_data.class_indices != val_data.class_indices or train_data.class_indices != test_data.class_indices:
    raise ValueError("Class folders/order differ between train, validation and test.")

if len(train_data.class_indices) != 26:
    raise ValueError(f"Expected 26 classes, found {len(train_data.class_indices)}.")

with (OUTPUT_DIR / "class_indices.json").open("w", encoding="utf-8") as f:
    json.dump(train_data.class_indices, f, indent=2)

base = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.30)(x)
outputs = layers.Dense(26, activation="softmax")(x)
model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

best_weights = OUTPUT_DIR / "best_model.weights.h5"
callbacks = [
    ModelCheckpoint(
        str(best_weights),
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=2,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=1,
        min_lr=1e-5,
        verbose=1
    )
]

print("\nStarting initial transfer-learning run.")
print("This is CPU training if TensorFlow reports no GPU; each epoch may take a while.")

model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

if best_weights.exists():
    model.load_weights(str(best_weights))

print("\nEvaluating on the held-out test split...")
test_loss, test_accuracy = model.evaluate(test_data, verbose=1)
print(f"\nTEST LOSS: {test_loss:.4f}")
print(f"TEST ACCURACY: {test_accuracy * 100:.2f}%")

model.save(str(OUTPUT_DIR / "final_model.h5"))
print("\nTraining run complete.")
print("Saved model:", (OUTPUT_DIR / "final_model.h5").resolve())
print("Best weights:", best_weights.resolve())

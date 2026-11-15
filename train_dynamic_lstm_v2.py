import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

tf.random.set_seed(42)
np.random.seed(42)

DATA_DIR = Path("datasets/dynamic_signs")
OUTPUT_DIR = Path("models/dynamic_signs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

X, y = [], []

for label_dir in sorted(DATA_DIR.iterdir()):
    if not label_dir.is_dir():
        continue

    for file in sorted(label_dir.glob("*.npy")):
        sequence = np.load(file, allow_pickle=False)

        if sequence.shape != (30, 21, 3):
            print("Skipping invalid file:", file)
            continue

        if not np.isfinite(sequence).all():
            print("Skipping non-finite file:", file)
            continue

        X.append(sequence.reshape(30, 63))
        y.append(label_dir.name)

X = np.asarray(X, dtype=np.float32)

encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("Dataset shape:", X.shape)
print("Classes:", list(encoder.classes_))

if len(encoder.classes_) != 2 or len(X) < 20:
    raise ValueError("Unexpected dataset or missing classes.")

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=0.20,
    random_state=42,
    stratify=y_trainval
)

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(30, 63)),
    tf.keras.layers.Masking(mask_value=0.0),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(len(encoder.classes_), activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        OUTPUT_DIR / "dynamic_lstm_v2_best.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=8,
    callbacks=callbacks,
    verbose=1
)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

predictions = np.argmax(
    model.predict(X_test, verbose=0),
    axis=1
)

print("\nTEST ACCURACY:", round(accuracy * 100, 2), "%")
print("TEST LOSS:", round(loss, 4))

print("\nCLASSIFICATION REPORT:")
print(classification_report(
    y_test,
    predictions,
    labels=np.arange(len(encoder.classes_)),
    target_names=encoder.classes_,
    zero_division=0
))

print("CONFUSION MATRIX:")
print(confusion_matrix(y_test, predictions))

model.save(OUTPUT_DIR / "dynamic_lstm_v2.h5")
np.save(OUTPUT_DIR / "classes_v2.npy", encoder.classes_)

print("\nNEW MODEL SAVED:")
print(OUTPUT_DIR / "dynamic_lstm_v2.h5")
print("Original production model was not modified.")

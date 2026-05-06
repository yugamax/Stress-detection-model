from pathlib import Path
from collections import Counter
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# -------------------------------
# MODEL
# -------------------------------
def build_model(img_size, num_classes):
    base = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )

    x = keras.layers.GlobalAveragePooling2D()(base.output)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Dense(256, activation="relu",
                           kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = keras.layers.Dropout(0.4)(x)

    x = keras.layers.Dense(128, activation="relu",
                           kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = keras.layers.Dropout(0.3)(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=base.input, outputs=outputs)
    return model, base


# -------------------------------
# TRAINING
# -------------------------------
def train(data_dir="facesData", img_size=224, batch_size=32):

    data_path = Path(data_dir)
    train_dir = data_path / "train"
    test_dir = data_path / "test"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError("Train/Test folders missing")

    # 🔥 Stronger augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=25,
        zoom_range=0.2,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        brightness_range=(0.7, 1.3),
        shear_range=0.1
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=True,
    )

    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=False,
    )

    class_names = list(train_gen.class_indices.keys())

    print("Train distribution:", Counter(train_gen.classes))
    print("Test distribution:", Counter(test_gen.classes))

    # ✅ CLASS WEIGHTS (IMPORTANT)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_gen.classes),
        y=train_gen.classes,
    )
    class_weights = dict(enumerate(class_weights))
    print("Class weights:", class_weights)

    model, base = build_model(img_size, len(class_names))

    # -------------------------------
    # PHASE 1: Train head
    # -------------------------------
    base.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            "best_model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("\n🚀 Phase 1: Training top layers...")
    model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=12,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # -------------------------------
    # PHASE 2: Fine-tuning
    # -------------------------------
    print("\n🔥 Phase 2: Fine-tuning...")

    base.trainable = True

    # Freeze early layers, train deeper layers
    for layer in base.layers[:-120]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )

    model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=30,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # -------------------------------
    # EVALUATION
    # -------------------------------
    loss, acc, prec, rec = model.evaluate(test_gen, verbose=0)
    print(f"\nFinal Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}, Recall: {rec:.4f}")

    test_gen.reset()
    y_true, y_pred = [], []

    for images, labels in test_gen:
        preds = model.predict(images, verbose=0)

        # 🔥 THRESHOLD TUNING (adjust this)
        threshold = 0.6
        preds_binary = (preds[:, 1] > threshold).astype(int)

        y_pred.extend(preds_binary)
        y_true.extend(labels)

        if len(y_true) >= test_gen.samples:
            break

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    model.save("stress_model.keras")
    print("\n✅ Model saved!")


# -------------------------------
if __name__ == "__main__":
    train()
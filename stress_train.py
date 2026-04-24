from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix


class EpochLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(
            f"Epoch {epoch+1} | "
            f"loss: {logs['loss']:.4f} | "
            f"acc: {logs['accuracy']:.4f} | "
            f"val_loss: {logs['val_loss']:.4f} | "
            f"val_acc: {logs['val_accuracy']:.4f}"
        )


def build_model(img_size, num_classes):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )

    base.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=base.input, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, base


def train(data_dir="facesData", img_size=128, batch_size=32):
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    test_dir = data_path / "test"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError("Train/Test folders missing")

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        horizontal_flip=True,
        rotation_range=25,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=(0.7, 1.3),
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
    print("Classes:", class_names)

    print("Train distribution:", Counter(train_gen.classes))
    print("Test distribution:", Counter(test_gen.classes))

    model, base = build_model(img_size, len(class_names))

    callbacks = [
        EpochLogger(),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
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

    history1 = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=10,
        callbacks=callbacks,
        verbose=0,
    )

    base.trainable = True

    for layer in base.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history2 = model.fit(
        train_gen,
        validation_data=test_gen,
        initial_epoch=history1.epoch[-1] + 1,
        epochs=20,
        callbacks=callbacks,
        verbose=0,
    )

    loss, acc = model.evaluate(test_gen, verbose=0)
    print(f"\nFinal Accuracy: {acc:.4f}")

    test_gen.reset()
    y_true, y_pred = [], []

    for images, labels in test_gen:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
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
    print("\nModel saved!")


if __name__ == "__main__":
    train()
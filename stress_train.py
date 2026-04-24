from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_model(img_size, num_classes):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )

    base.trainable = False  # freeze initially

    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=base.input, outputs=output)

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
        raise FileNotFoundError(
            f"Expected train and test folders inside {data_path}, but one or both are missing."
        )

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=25,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=(0.7, 1.3),
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

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
    num_classes = len(class_names)

    print("Classes:", class_names)

    model, base = build_model(img_size, num_classes)

    callbacks = [
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
            "best_stress_efficientnet.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=10,
        callbacks=callbacks,
        verbose=2,
    )


    base.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),  # VERY important (low LR)
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=10,
        callbacks=callbacks,
        verbose=2,
    )

    # Evaluation
    loss, acc = model.evaluate(test_gen)
    print(f"Accuracy: {acc:.4f}")

    # Predictions
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

    model.save("stress_efficientnet.keras")
    print("Model saved!")


if __name__ == "__main__":
    train()
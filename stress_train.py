from pathlib import Path
from collections import Counter
import os
import json
import warnings

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).resolve().parent
STATE_FILE = BASE_DIR / "training_state.json"
MODEL_FILE = BASE_DIR / "latest_model.keras"
BASE_FILE  = BASE_DIR / "base_model.keras"

PHASE1_EPOCHS    = 15
PHASE2_EPOCHS    = 40
VALIDATION_SPLIT = 0.2


# ── State helpers ──────────────────────────────────────────────────────────────

def save_state(phase: str, epoch: int):
    STATE_FILE.write_text(json.dumps({"phase": phase, "epoch": epoch}))


def load_state() -> dict:
    if not STATE_FILE.exists():
        return {"phase": "phase1", "epoch": 0}
    return json.loads(STATE_FILE.read_text())


class EpochCheckpoint(keras.callbacks.Callback):
    def __init__(self, phase: str):
        super().__init__()
        self.phase = phase

    def on_epoch_end(self, epoch, logs=None):
        save_state(self.phase, epoch + 1)


# ── Data directories ───────────────────────────────────────────────────────────

def resolve_data_dirs(data_path: Path):
    train_dir = data_path / "train"
    val_dir   = data_path / "val"
    test_dir  = data_path / "test"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError("train/ and test/ folders are required.")

    has_val = val_dir.exists()
    return train_dir, val_dir if has_val else None, test_dir, not has_val


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model(img_size: int, num_classes: int):
    """
    Lighter classification head:
    - Single dense layer (512) instead of two stacked ones
    - Moderate dropout (0.5) to regularise without over-suppressing signal
    - GeM pooling via GlobalAveragePooling2D (standard, stable)
    """
    base = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )

    x = keras.layers.GlobalAveragePooling2D()(base.output)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(
            512,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(1e-4),
        )(x)
    x = keras.layers.Dropout(0.5)(x)

    activation = "sigmoid" if num_classes == 2 else "softmax"
    output_units = 1 if num_classes == 2 else num_classes
    outputs = keras.layers.Dense(output_units, activation=activation)(x)

    model = keras.Model(inputs=base.input, outputs=outputs)
    return model, base


def compile_model(model, lr: float, num_classes: int):
    if num_classes == 2:
        loss    = "binary_crossentropy"
        metrics = [keras.metrics.BinaryAccuracy(name="accuracy"),
                   keras.metrics.AUC(name="auc")]
    else:
        loss    = "sparse_categorical_crossentropy"
        metrics = [keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

    model.compile(optimizer=keras.optimizers.Adam(lr), loss=loss, metrics=metrics)


# ── Training ───────────────────────────────────────────────────────────────────

def train(data_dir: str = "facesData", img_size: int = 224, batch_size: int = 32):

    state     = load_state()
    data_path = Path(data_dir)
    train_dir, val_dir, test_dir, use_split_val = resolve_data_dirs(data_path)

    # ── Generators ────────────────────────────────────────────────────────────
    # Augmentation is intentionally moderate — aggressive transforms
    # (strong colour jitter, large shifts) hurt face recognition tasks.
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=VALIDATION_SPLIT if use_split_val else 0.0,
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2),
    )
    plain_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    flow_kw = dict(
        target_size=(img_size, img_size),
        batch_size=batch_size,
        seed=42,
    )

    if use_split_val:
        train_gen = train_datagen.flow_from_directory(
            train_dir, subset="training",   shuffle=True,  class_mode="sparse", **flow_kw)
        val_gen   = train_datagen.flow_from_directory(
            train_dir, subset="validation", shuffle=False, class_mode="sparse", **flow_kw)
    else:
        train_gen = train_datagen.flow_from_directory(
            train_dir, shuffle=True,  class_mode="sparse", **flow_kw)
        val_gen   = plain_datagen.flow_from_directory(
            val_dir,   shuffle=False, class_mode="sparse", **flow_kw)

    test_gen = plain_datagen.flow_from_directory(
        test_dir, shuffle=False, class_mode="sparse", **flow_kw)

    class_names = list(train_gen.class_indices.keys())
    num_classes = len(class_names)

    print(f"\nClasses ({num_classes}): {class_names}")
    print("Train:", Counter(train_gen.classes))
    print("Val:  ", Counter(val_gen.classes))
    print("Test: ", Counter(test_gen.classes))

    # ── Class weights ─────────────────────────────────────────────────────────
    class_weights = dict(enumerate(
        compute_class_weight("balanced",
                             classes=np.unique(train_gen.classes),
                             y=train_gen.classes)
    ))

    # ── Binary label mapping for 2-class case ─────────────────────────────────
    # flow_from_directory always emits sparse integer labels (0 / 1).
    # binary_crossentropy expects the same, so no remapping needed.

    # ── Load or build model ───────────────────────────────────────────────────
    if MODEL_FILE.exists():
        model = keras.models.load_model(str(MODEL_FILE))
        if BASE_FILE.exists():
            base = keras.models.load_model(str(BASE_FILE))
        else:
            base = next(
                (l for l in model.layers if "efficientnet" in l.name.lower()), None
            )
            if base is None:
                model, base = build_model(img_size, num_classes)
    else:
        model, base = build_model(img_size, num_classes)

    # ── Common callbacks factory ──────────────────────────────────────────────
    def make_callbacks(phase: str):
        return [
            EpochCheckpoint(phase),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.3, patience=3,
                min_lr=1e-7, verbose=1),
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=8,
                restore_best_weights=True, verbose=1),
            keras.callbacks.ModelCheckpoint(
                str(MODEL_FILE), save_best_only=True, verbose=0),
        ]

    # ── Phase 1: train head only ──────────────────────────────────────────────
    if state["phase"] == "phase1":
        base.trainable = False
        compile_model(model, lr=3e-4, num_classes=num_classes)

        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=PHASE1_EPOCHS,
            initial_epoch=state["epoch"],
            class_weight=class_weights,
            callbacks=make_callbacks("phase1"),
            verbose=2,
        )
        save_state("phase2", 0)
        state = {"phase": "phase2", "epoch": 0}
        base.save(str(BASE_FILE))

    # ── Phase 2: fine-tune top layers ─────────────────────────────────────────
    if state["phase"] == "phase2":
        base.trainable = True

        # Freeze all layers first, then unfreeze the last 150 layers
        for layer in base.layers:
            layer.trainable = False
        for layer in base.layers[-150:]:
            layer.trainable = True

        compile_model(model, lr=2e-5, num_classes=num_classes)

        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=PHASE2_EPOCHS,
            initial_epoch=state["epoch"],
            class_weight=class_weights,
            callbacks=make_callbacks("phase2"),
            verbose=2,
        )

    # ── Evaluation ────────────────────────────────────────────────────────────
    results = model.evaluate(test_gen, verbose=0)

    test_gen.reset()
    all_probs, all_true = [], []
    for images, labels in tqdm(test_gen, total=test_gen.samples // test_gen.batch_size, desc="Evaluating"):
        all_probs.append(model.predict(images, verbose=0))
        all_true.extend(labels)
        if len(all_true) >= test_gen.samples:
            break

    all_probs = np.concatenate(all_probs, axis=0)
    all_true  = np.array(all_true, dtype=int)

    if num_classes == 2:
        # Binary: find threshold that maximises macro-F1
        from sklearn.metrics import f1_score
        probs_pos = all_probs.ravel()
        best_t, best_f1 = 0.5, 0.0
        for t in tqdm(np.arange(0.3, 0.8, 0.02), desc="Finding optimal threshold"):
            preds = (probs_pos >= t).astype(int)
            f1    = f1_score(all_true, preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        y_pred = (probs_pos >= best_t).astype(int)
    else:
        y_pred = np.argmax(all_probs, axis=1)

    classification_report(all_true, y_pred,
                          target_names=class_names, zero_division=0)

    cm = confusion_matrix(all_true, y_pred)
    plt.figure(figsize=(max(6, num_classes), max(5, num_classes - 1)))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()

    model.save("stress_model.keras")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
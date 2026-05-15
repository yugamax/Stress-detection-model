from pathlib import Path
from collections import Counter
import os
import json
import warnings
import signal
import atexit
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).resolve().parent
STATE_FILE = BASE_DIR / "training_state.json"
MODEL_FILE = BASE_DIR / "latest_model.keras"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Longer Phase 1 to let head fully learn before fine-tuning
PHASE1_EPOCHS    = 25
PHASE2_EPOCHS    = 60
VALIDATION_SPLIT = 0.2

IMG_SIZE = 260   # FIX: B3 native input size (was 224 for B0)

# Graceful interruption state
_INTERRUPTED = False
_CURRENT_MODEL = None


def handle_interrupt(signum, frame):
    """Handle SIGINT/SIGTERM to gracefully save state before exit."""
    global _INTERRUPTED, _CURRENT_MODEL
    _INTERRUPTED = True
    print("\n\n🛑 Received interrupt signal. Saving checkpoint...")
    if _CURRENT_MODEL is not None:
        try:
            _CURRENT_MODEL.save(str(MODEL_FILE))
            print(f"   ✓ Model saved to {MODEL_FILE}")
        except Exception as e:
            print(f"   ⚠️  Error saving model: {e}")
    raise KeyboardInterrupt("Training interrupted by user")


# ── State helpers ──────────────────────────────────────────────────────────────

def save_state(phase: str, epoch: int, best_val_loss: float = 9999.0, batch_idx: int = None, total_batches: int = None):
    """Save training state with checkpoint details. Flushes to disk immediately."""
    state_dict = {
        "phase": phase,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Include batch-level progress for recovery
    if batch_idx is not None and total_batches is not None:
        state_dict["batch_progress"] = {"current": batch_idx, "total": total_batches}
    
    try:
        # Write with flush to ensure data is persisted
        with open(STATE_FILE, 'w') as f:
            json.dump(state_dict, f, indent=2)
            f.flush()  # Flush to buffer
            os.fsync(f.fileno())  # Force disk write (MUST be before file closes)
    except Exception as e:
        print(f"⚠️  Error saving state: {e}")


def load_state() -> dict:
    """Load training state from checkpoint file."""
    if not STATE_FILE.exists():
        return {"phase": "phase1", "epoch": 0, "best_val_loss": 9999.0}
    try:
        state = json.loads(STATE_FILE.read_text())
        # Add batch progress info if it exists
        batch_progress = state.get("batch_progress")
        if batch_progress:
            print(f"   📍 Resuming from batch {batch_progress['current']}/{batch_progress['total']}")
        return state
    except Exception as e:
        print(f"⚠️  Error loading state: {e}")
        return {"phase": "phase1", "epoch": 0, "best_val_loss": 9999.0}


def save_model_checkpoint(model, phase: str, epoch: int, suffix: str = ""):
    """Save backup checkpoint of model weights with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = f"model_checkpoint_{phase}_e{epoch:03d}_{timestamp}{suffix}.keras"
    checkpoint_path = CHECKPOINT_DIR / checkpoint_name
    try:
        model.save(str(checkpoint_path))
        print(f"   💾 Checkpoint saved: {checkpoint_name}")
        
        # Keep only last 5 checkpoints per phase to save space
        checkpoints = sorted(CHECKPOINT_DIR.glob(f"model_checkpoint_{phase}_*.keras"))
        if len(checkpoints) > 5:
            for old_ckpt in checkpoints[:-5]:
                old_ckpt.unlink()
                print(f"   🗑  Removed old checkpoint: {old_ckpt.name}")
    except Exception as e:
        print(f"   ⚠️  Error saving checkpoint: {e}")


class EpochCheckpoint(keras.callbacks.Callback):
    """Callback to save state at epoch end and create model backups periodically."""
    def __init__(self, phase: str):
        super().__init__()
        self.phase = phase
        self.best_val_loss = 9999.0
        self.epoch_count = 0

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss", 9999.0) if logs else 9999.0
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
        
        # Save state after each epoch
        save_state(self.phase, epoch + 1, self.best_val_loss)
        
        # Save model checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_model_checkpoint(self.model, self.phase, epoch + 1, suffix="_best" if val_loss < self.best_val_loss else "")


class BatchCheckpoint(keras.callbacks.Callback):
    """Callback to save state at batch level for fine-grained recovery."""
    def __init__(self, phase: str, save_every_n_batches: int = 100):
        super().__init__()
        self.phase = phase
        self.save_every_n_batches = save_every_n_batches
        self.batch_count = 0
        self.best_val_loss = 9999.0

    def on_train_batch_end(self, batch, logs=None):
        """Save state periodically during training."""
        self.batch_count += 1
        
        # Save batch progress every N batches (less frequent to avoid overhead)
        if self.batch_count % self.save_every_n_batches == 0:
            val_loss = logs.get("val_loss", 9999.0) if logs else 9999.0
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
            # Note: batch info saved but may not be used for simple recovery
            # since full epochs are rerun on recovery


def setup_signal_handlers():
    """Register signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)
    if os.name == 'nt':  # Windows
        # Windows supports SIGTERM but not SIGALRM
        try:
            signal.signal(signal.SIGBREAK, handle_interrupt)
        except (AttributeError, ValueError):
            pass


# ── Label smoothing loss (penalises overconfidence) ───────────────────────────

class LabelSmoothingLoss(keras.losses.Loss):
    """
    FIX: Label smoothing prevents the model from becoming overconfident
    on the majority class, which was causing 9-13% stress confidence.
    smoothing=0.1 means true labels shift from [0,1] → [0.05, 0.95].
    """
    def __init__(self, num_classes: int, smoothing: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.smoothing = smoothing

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true_oh = tf.one_hot(y_true, self.num_classes)
        smooth_labels = (1.0 - self.smoothing) * y_true_oh + (self.smoothing / self.num_classes)
        log_probs = tf.math.log(tf.clip_by_value(y_pred, 1e-9, 1.0))
        return -tf.reduce_mean(tf.reduce_sum(smooth_labels * log_probs, axis=-1))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_classes": self.num_classes, "smoothing": self.smoothing})
        return cfg


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model(img_size: int, num_classes: int):
    """
    FIX: Switch to EfficientNetB3 (was B0).
    B3 has 12M params vs B0's 5.3M — captures subtler facial features.
    Native input size is 300px but 260px works well and trains faster.
    """
    base = keras.applications.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )

    x = keras.layers.GlobalAveragePooling2D()(base.output)
    x = keras.layers.BatchNormalization()(x)

    # FIX: Two-layer head — gives more representational capacity
    x = keras.layers.Dense(
        512,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(x)
    x = keras.layers.Dropout(0.3)(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=base.input, outputs=outputs)
    return model, base


def compile_model(model, lr: float, num_classes: int, label_smoothing: float = 0.0):
    loss = (
        LabelSmoothingLoss(num_classes, label_smoothing)
        if label_smoothing > 0
        else "sparse_categorical_crossentropy"
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


# ── Training ───────────────────────────────────────────────────────────────────

def train(data_dir="facesData", img_size=IMG_SIZE, batch_size=32):
    global _CURRENT_MODEL
    
    # Setup signal handlers for graceful interruption
    setup_signal_handlers()
    
    state = load_state()
    data_path = Path(data_dir)

    train_dir = data_path / "train"
    val_dir   = data_path / "val"
    test_dir  = data_path / "test"

    use_split_val = not val_dir.exists()

    # FIX: Face-appropriate augmentations.
    # Removed large zoom/shift that distort face geometry.
    # Kept horizontal flip (valid for faces), gentle brightness/contrast,
    # and added channel_shift for skin-tone variation.
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=VALIDATION_SPLIT if use_split_val else 0.0,
        rotation_range=10,          # reduced: ±10° instead of ±15°
        zoom_range=0.08,            # reduced: 8% instead of 20%
        width_shift_range=0.05,     # reduced: 5% instead of 15%
        height_shift_range=0.05,    # reduced: 5% instead of 15%
        horizontal_flip=True,       # valid for faces
        brightness_range=(0.85, 1.15),
        channel_shift_range=15.0,   # skin-tone variation
        fill_mode="reflect",        # better for faces than "nearest"
    )

    plain_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    flow_kw = dict(
        target_size=(img_size, img_size),
        batch_size=batch_size,
        seed=42,
        class_mode="sparse",
    )

    if use_split_val:
        train_gen = train_datagen.flow_from_directory(
            train_dir, subset="training", shuffle=True, **flow_kw
        )
        val_gen = train_datagen.flow_from_directory(
            train_dir, subset="validation", shuffle=False, **flow_kw
        )
    else:
        train_gen = train_datagen.flow_from_directory(
            train_dir, shuffle=True, **flow_kw
        )
        val_gen = plain_datagen.flow_from_directory(
            val_dir, shuffle=False, **flow_kw
        )

    test_gen = plain_datagen.flow_from_directory(
        test_dir, shuffle=False, **flow_kw
    )

    class_names = list(train_gen.class_indices.keys())
    num_classes = len(class_names)

    print("\n⚠️  CLASS INDEX MAPPING:")
    for name, idx in train_gen.class_indices.items():
        print(f"   {idx} → '{name}'")

    train_counts = Counter(train_gen.classes)
    val_counts   = Counter(val_gen.classes)
    test_counts  = Counter(test_gen.classes)

    print("\nTrain:", {class_names[k]: v for k, v in train_counts.items()})
    print("Val:  ", {class_names[k]: v for k, v in val_counts.items()})
    print("Test: ", {class_names[k]: v for k, v in test_counts.items()})

    # ── Class weights ─────────────────────────────────────────────────────────
    raw_weights = compute_class_weight(
        "balanced",
        classes=np.unique(train_gen.classes),
        y=train_gen.classes,
    )
    class_weights = dict(enumerate(raw_weights))
    print("\nClass weights:", {class_names[k]: round(v, 3) for k, v in class_weights.items()})

    # Extra boost if imbalance ratio > 1.5
    weights_arr = np.array(list(class_weights.values()))
    if weights_arr.max() / weights_arr.min() > 1.5:
        minority_idx = int(np.argmax(weights_arr))
        class_weights[minority_idx] *= 1.5
        print(f"   ✓ Boosted '{class_names[minority_idx]}' weight to {class_weights[minority_idx]:.3f}")

    # ── Model ────────────────────────────────────────────────────────────────

    if MODEL_FILE.exists():
        print(f"\n📥 Loading model from {MODEL_FILE}...")
        model = keras.models.load_model(
            str(MODEL_FILE),
            custom_objects={"LabelSmoothingLoss": LabelSmoothingLoss},
        )
        print(f"   Total layers: {len(model.layers)}")

        # Verify output shape matches current num_classes
        last_units = model.layers[-1].output_shape[-1]
        if last_units != num_classes:
            print(f"   ⚠️  Output units ({last_units}) ≠ num_classes ({num_classes}). Rebuilding...")
            MODEL_FILE.unlink()
            STATE_FILE.unlink() if STATE_FILE.exists() else None
            model, base = build_model(img_size, num_classes)
            state = {"phase": "phase1", "epoch": 0, "best_val_loss": 9999.0}
        else:
            # Find base/head boundary
            gap_idx = next(
                (i for i, l in enumerate(model.layers)
                 if isinstance(l, keras.layers.GlobalAveragePooling2D)),
                None,
            )
            cutoff = gap_idx if gap_idx is not None else int(len(model.layers) * 0.80)
            base_layers = model.layers[:cutoff]
            print(f"   ✓ Base: {len(base_layers)} layers | Head: {len(model.layers) - cutoff} layers")

            class FakeBase:
                def __init__(self, layers):
                    self.layers = layers
                @property
                def trainable(self):
                    return all(l.trainable for l in self.layers)
                @trainable.setter
                def trainable(self, value):
                    for l in self.layers:
                        l.trainable = value

            base = FakeBase(base_layers)
    else:
        print(f"\n🆕 Building new EfficientNetB3 model...")
        model, base = build_model(img_size, num_classes)

    model.summary(line_length=80, expand_nested=False)
    
    # Set global model reference for interrupt handler
    _CURRENT_MODEL = model

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def make_callbacks(phase):
        return [
            EpochCheckpoint(phase),
            BatchCheckpoint(phase, save_every_n_batches=100),  # Save batch progress periodically
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.4,         # gentler decay (was 0.3)
                patience=5,         # more patience (was 4)
                min_lr=5e-8,
                verbose=1,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,        # more patience (was 8)
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ModelCheckpoint(
                str(MODEL_FILE),
                save_best_only=True,
                monitor="val_loss",
                verbose=1,
            ),
        ]

    # ── Phase 1: Train head only ───────────────────────────────────────────────

    if state["phase"] == "phase1":
        print("\n🔒 Phase 1: Freezing base, training head only...")
        base.trainable = False

        # FIX: LR 3e-4 instead of 1e-3.
        # 1e-3 caused the chaotic loss in your epochs 1-3.
        # Label smoothing 0.1 from the start prevents majority-class collapse.
        compile_model(model, lr=3e-4, num_classes=num_classes, label_smoothing=0.1)

        try:
            model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=PHASE1_EPOCHS,
                initial_epoch=state["epoch"],
                class_weight=class_weights,
                callbacks=make_callbacks("phase1"),
                verbose=2,
            )
        except KeyboardInterrupt:
            print("⚠️  Phase 1 training interrupted. State saved. Exiting...")
            save_model_checkpoint(model, "phase1", state.get("epoch", 0), suffix="_interrupted")
            raise

        save_state("phase2", 0)
        state = {"phase": "phase2", "epoch": 0, "best_val_loss": 9999.0}

    # ── Phase 2: Progressive unfreezing ───────────────────────────────────────

    if state["phase"] == "phase2":
        print("\n🔓 Phase 2: Progressive fine-tuning (3 stages)...")

        # FIX: Progressive unfreezing instead of all-at-once.
        # Stage A: top 30 layers, Stage B: top 80, Stage C: top 150.
        # Each stage runs for a fixed number of epochs then expands.
        # This prevents early catastrophic forgetting of ImageNet features.

        n_base = len(base.layers)

        def set_trainable(n_unfreeze, freeze_bn=True):
            for layer in base.layers:
                layer.trainable = False
            n = min(n_unfreeze, n_base)
            for layer in base.layers[-n:]:
                layer.trainable = True
            if freeze_bn:
                for layer in base.layers:
                    if isinstance(layer, keras.layers.BatchNormalization):
                        layer.trainable = False
            frozen = sum(1 for l in base.layers if not l.trainable)
            unfrozen = sum(1 for l in base.layers if l.trainable)
            print(f"   → {unfrozen} base layers trainable, {frozen} frozen (BN always frozen)")

        stages = [
            # (n_unfreeze, lr,     epochs, label_smooth)
            (30,           2e-5,   15,     0.1),   # Stage A — just top layers
            (80,           1e-5,   20,     0.05),  # Stage B — mid layers
            (150,          5e-6,   25,     0.0),   # Stage C — deep layers
        ]

        # If resuming Phase 2, figure out which stage we're in
        start_epoch = state.get("epoch", 0)
        cumulative = 0
        start_stage = 0
        for i, (_, _, ep, _) in enumerate(stages):
            if start_epoch >= cumulative + ep:
                cumulative += ep
                start_stage = i + 1
            else:
                break

        for stage_i, (n_unfreeze, lr, ep, smooth) in enumerate(stages[start_stage:], start=start_stage):
            stage_start = max(0, start_epoch - cumulative)
            if stage_start >= ep:
                cumulative += ep
                continue

            print(f"\n   Stage {stage_i + 1}/3: {n_unfreeze} layers unfrozen, LR={lr}")
            set_trainable(n_unfreeze)
            compile_model(model, lr=lr, num_classes=num_classes, label_smoothing=smooth)

            try:
                model.fit(
                    train_gen,
                    validation_data=val_gen,
                    epochs=cumulative + ep,
                    initial_epoch=cumulative + stage_start,
                    class_weight=class_weights,
                    callbacks=make_callbacks("phase2"),
                    verbose=2,
                )
            except KeyboardInterrupt:
                print(f"⚠️  Phase 2 Stage {stage_i + 1}/3 interrupted. State saved at epoch {cumulative}. Exiting...")
                save_model_checkpoint(model, "phase2", cumulative, suffix="_interrupted")
                raise

            cumulative += ep
            save_state("phase2", cumulative)

        save_state("done", cumulative)

    # ── Evaluation ────────────────────────────────────────────────────────────

    print("\n📊 Final evaluation on test set...")
    results = model.evaluate(test_gen, verbose=1)
    for name, val in zip(model.metrics_names, results):
        print(f"   {name}: {val:.4f}")

    test_gen.reset()
    all_probs, all_true = [], []

    for images, labels in tqdm(test_gen, desc="Collecting predictions"):
        all_probs.append(model.predict(images, verbose=0))
        all_true.extend(labels.astype(int))
        if len(all_true) >= test_gen.samples:
            break

    all_probs = np.concatenate(all_probs, axis=0)[:test_gen.samples]
    all_true  = np.array(all_true[:test_gen.samples], dtype=int)

    # ── Threshold search on stress class ─────────────────────────────────────
    if num_classes == 2:
        stress_idx = train_gen.class_indices.get(
            "stress",
            train_gen.class_indices.get("Stress", 1)
        )
        print(f"\n   Stress class index: {stress_idx}")
        stress_probs = all_probs[:, stress_idx]

        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.1, 0.9, 0.01):
            preds_raw = (stress_probs >= t).astype(int)
            mapped = np.where(preds_raw == 1, stress_idx, 1 - stress_idx)
            f1 = f1_score(all_true, mapped, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t

        print(f"   Best threshold: {best_t:.2f}  (macro F1: {best_f1:.4f})")
        stress_preds = (stress_probs >= best_t).astype(int)
        y_pred = np.where(stress_preds == 1, stress_idx, 1 - stress_idx)
    else:
        y_pred = np.argmax(all_probs, axis=1)

    print("\n" + classification_report(
        all_true, y_pred,
        target_names=class_names,
        zero_division=0,
    ))

    # ── Confidence diagnostics ────────────────────────────────────────────────
    print("🔍 Confidence diagnostics per true class:")
    for cls_idx, cls_name in enumerate(class_names):
        mask = all_true == cls_idx
        if mask.sum() == 0:
            continue
        p = all_probs[mask, cls_idx]
        print(f"   '{cls_name}': mean={p.mean():.3f}  min={p.min():.3f}  max={p.max():.3f}  "
              f"(samples where model was right: {(np.argmax(all_probs[mask], axis=1) == cls_idx).mean():.1%})")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(all_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("\n✅ Saved confusion_matrix.png")

    model.save("stress_model.keras")
    print("✅ Saved stress_model.keras")
    print("\nDone.")


if __name__ == "__main__":
    # IMPORTANT: Comment out the deletion below if you want to resume from checkpoints
    # Uncomment only when starting fresh training
    
    # For fresh start with B3 architecture:
    # for f in [STATE_FILE, MODEL_FILE]:
    #     if f.exists():
    #         print(f"🗑  Removing old file: {f.name}")
    #         f.unlink()
    
    try:
        train()
    except KeyboardInterrupt:
        print("\n\n🛑 Training interrupted by user. Checkpoints have been saved.")
        print(f"   📁 Resume with: python {Path(__file__).name}")
        print(f"   📝 State: {STATE_FILE}")
        print(f"   💾 Latest model: {MODEL_FILE}")
        print(f"   🔄 Checkpoints: {CHECKPOINT_DIR}")
        import sys
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
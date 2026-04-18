# ocr/train.py
"""
Train the character CNN on MNIST with noise augmentation.

MNIST is the document-recommended dataset (see case brief).
Uses Keras built-in loader — no downloads needed, works on all platforms.
"""
import os
import json
import numpy as np
import tensorflow as tf

from cnn_model import build_cnn, compile_model
from noise import add_gaussian_noise, add_salt_pepper_noise


# ── config ────────────────────────────────────────────────────────
IMAGE_SIZE   = 28
BATCH_SIZE   = 128
EPOCHS       = 12
LEARN_RATE   = 1e-3
NUM_CLASSES  = 10      # MNIST = 10 digit classes
MODEL_PATH   = "artifacts/character_cnn.keras"

os.makedirs("artifacts", exist_ok=True)


# ── data loading ──────────────────────────────────────────────────

def load_mnist():
    """Keras built-in MNIST — downloads once, ~11MB, cached at ~/.keras/datasets."""
    print("Loading MNIST via Keras (first run downloads ~11MB)...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print(f"Train size: {len(x_train)}")
    print(f"Test size:  {len(x_test)}")
    return (x_train, y_train), (x_test, y_test)


def preprocess_arrays(x_train, y_train, x_test, y_test):
    """Normalize to [0,1] and add channel dimension."""
    x_train = x_train.astype(np.float32) / 255.0
    x_test  = x_test.astype(np.float32) / 255.0
    x_train = x_train[..., np.newaxis]   # (N, 28, 28, 1)
    x_test  = x_test[..., np.newaxis]
    return x_train, y_train, x_test, y_test


def apply_training_noise(image, label):
    """50% Gaussian, 25% salt-pepper, 25% clean."""
    rand = tf.random.uniform([], minval=0, maxval=1)
    image = tf.cond(
        rand < 0.5,
        lambda: add_gaussian_noise(image, std=0.15),
        lambda: tf.cond(
            rand < 0.75,
            lambda: add_salt_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02),
            lambda: image
        )
    )
    return image, label


def make_dataset(x, y, batch_size, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(10000)
        ds = ds.map(apply_training_noise, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ── evaluation ────────────────────────────────────────────────────

def evaluate_per_noise_profile(model, x_test, y_test):
    """Measure accuracy separately on clean, Gaussian, salt-pepper."""
    print("\n" + "=" * 55)
    print("  PER-NOISE-PROFILE EVALUATION")
    print("=" * 55)

    # clean
    clean_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)
    _, clean_acc = model.evaluate(clean_ds, verbose=0)
    print(f"  Clean (no noise)       : {clean_acc * 100:.2f}%")

    # Gaussian
    x_gauss = tf.clip_by_value(
        x_test + np.random.normal(0, 0.15, x_test.shape).astype(np.float32),
        0.0, 1.0
    )
    gauss_ds = tf.data.Dataset.from_tensor_slices((x_gauss, y_test)).batch(BATCH_SIZE)
    _, gauss_acc = model.evaluate(gauss_ds, verbose=0)
    print(f"  Gaussian noise (σ=0.15): {gauss_acc * 100:.2f}%")

    # Salt-pepper
    x_sp = x_test.copy()
    rand = np.random.uniform(0, 1, x_sp.shape)
    x_sp = np.where(rand < 0.02, 1.0, x_sp)          # salt
    x_sp = np.where((rand >= 0.02) & (rand < 0.04), 0.0, x_sp)  # pepper
    sp_ds = tf.data.Dataset.from_tensor_slices((x_sp.astype(np.float32), y_test)).batch(BATCH_SIZE)
    _, sp_acc = model.evaluate(sp_ds, verbose=0)
    print(f"  Salt-pepper noise      : {sp_acc * 100:.2f}%")

    print("=" * 55)

    return {
        "clean":       float(clean_acc),
        "gaussian":    float(gauss_acc),
        "salt_pepper": float(sp_acc)
    }


# ── main ──────────────────────────────────────────────────────────

def main():
    (x_train, y_train), (x_test, y_test) = load_mnist()
    x_train, y_train, x_test, y_test = preprocess_arrays(x_train, y_train, x_test, y_test)

    train_ds = make_dataset(x_train, y_train, BATCH_SIZE, training=True)
    test_ds  = make_dataset(x_test,  y_test,  BATCH_SIZE, training=False)

    print("\nBuilding CNN...")
    model = build_cnn(num_classes=NUM_CLASSES)
    model = compile_model(model, learning_rate=LEARN_RATE)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
            mode="max"
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max"
        )
    ]

    print(f"\nTraining for up to {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    metrics = evaluate_per_noise_profile(model, x_test, y_test)

    with open("artifacts/training_metrics.json", "w") as f:
        json.dump({
            "dataset":            "MNIST",
            "num_classes":        NUM_CLASSES,
            "final_val_accuracy": float(history.history["val_accuracy"][-1]),
            "best_val_accuracy":  float(max(history.history["val_accuracy"])),
            "per_noise_accuracy": metrics,
            "total_epochs_run":   len(history.history["loss"])
        }, f, indent=2)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metrics saved to: artifacts/training_metrics.json")

    best_acc = max(history.history["val_accuracy"])
    if best_acc >= 0.95:
        print(f"\n✓ SUCCESS — best validation accuracy {best_acc * 100:.2f}% meets 95% requirement")
    else:
        print(f"\n⚠ Best validation accuracy {best_acc * 100:.2f}% — below 95% target")


if __name__ == "__main__":
    main()
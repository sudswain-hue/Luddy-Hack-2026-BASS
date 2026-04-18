"""
MNIST-focused OCR workflow (TensorFlow) + optional CTC for multi-digit *lines*
synthesized from MNIST digits.

Important mapping to your brief:
- MNIST official images are 28x28 grayscale with ONE digit per image.
  For that setting, the correct head is *single-label softmax* (not CTC).
- CTC is for *sequence* OCR: one wide image -> variable-length digit string.
  The included `run_ctc_synthetic_lines_demo` builds such lines from MNIST
  tiles so you can demonstrate CTC end-to-end on TensorFlow.

SIDD: for `--mode ctc_demo --sidd-dir ...`, we blend random grayscale *patches* from
  your SIDD image tree into synthetic digit-line images to mimic realistic sensor texture.
  Download/unpack SIDD yourself, then point `--sidd-dir` at a folder full of `.png`/`.jpg`.
  See `sidd_evaluation_notes()` for caveats (SIDD is not OCR text labels).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf


# -----------------------------------------------------------------------------
# CTC: conceptual summary (see module docstring + comments in train_ctc_model)
# -----------------------------------------------------------------------------
#
# CTC aligns a *longer* time axis T (CNN outputs one logit vector per time step)
# to a *shorter* label sequence (e.g. digit string "304") without per-timestep
# character labels. It introduces a "blank" class; training marginalizes over
# all valid alignments. TensorFlow expects logits time-major: shape (T, B, C).


@dataclass
class TrainConfig:
    image_size: Tuple[int, int] = (32, 32)
    batch_size: int = 128
    epochs: int = 15
    learning_rate: float = 1e-3
    min_char_acc: float = 0.95
    seed: int = 42


@dataclass
class NoiseProfileEvalConfig:
    """
    Graduate-style requirement: *measurable* accuracy under each corruption.

    We use a fixed RNG seed so the Gaussian noise field and salt/pepper masks are
    reproducible across runs (same corrupted test set each time).
    """

    gaussian_std: float = 0.06
    salt_prob: float = 0.01
    pepper_prob: float = 0.01
    seed: int = 1337


def set_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)


def load_mnist() -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
    """MNIST via TensorFlow/Keras (no separate download manager required)."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return (x_train, y_train), (x_test, y_test)


def to_model_input(
    images: np.ndarray,
    image_size: Tuple[int, int],
    add_channel: bool = True,
) -> np.ndarray:
    """Resize (optional), float32, [0,1]. images: uint8 (N,H,W) or (N,H,W,1)."""
    if images.ndim == 3:
        x = images[..., None]
    else:
        x = images
    x = tf.image.resize(x, image_size, method="bilinear").numpy()
    x = x.astype("float32") / 255.0
    return x


def add_gaussian_noise(x: tf.Tensor, std: float = 0.08) -> tf.Tensor:
    return tf.clip_by_value(x + tf.random.normal(tf.shape(x), stddev=std), 0.0, 1.0)


def add_salt_pepper_noise(x: tf.Tensor, salt_prob: float = 0.01, pepper_prob: float = 0.01) -> tf.Tensor:
    """Randomly set pixels to 1 (salt) or 0 (pepper). x in [0,1]."""
    salt = tf.random.uniform(tf.shape(x)) < salt_prob
    pepper = tf.random.uniform(tf.shape(x)) < pepper_prob
    x = tf.where(salt, 1.0, x)
    x = tf.where(pepper, 0.0, x)
    return x


def make_augment_fn(
    use_gaussian: bool = True,
    use_snp: bool = True,
    gaussian_std: float = 0.06,
) -> Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    def augment(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # Avoid horizontal flips on MNIST (e.g., 6 vs 9 ambiguity).
        if use_gaussian:
            image = add_gaussian_noise(image, std=gaussian_std)
        if use_snp:
            image = add_salt_pepper_noise(image)
        return image, label

    return augment


def build_mnist_cnn(num_classes: int = 10, image_size: Tuple[int, int] = (32, 32)) -> tf.keras.Model:
    """
    Small CNN justified for 28x32-ish digits:
    - 3x3 kernels: standard local receptive field for strokes/edges.
    - Two conv blocks + MaxPool: spatial summarization before classification head.
    - ReLU: cheap, avoids vanishing gradients in deep stacks (vs sigmoid).
    - Dropout on dense: mild regularization on MNIST.
    """
    inputs = tf.keras.Input(shape=(*image_size, 1), name="image")
    x = inputs
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="digit")(x)
    model = tf.keras.Model(inputs, outputs, name="mnist_digit_cnn")
    return model


def make_tf_dataset(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    training: bool,
    image_size: Tuple[int, int],
    augment: Optional[Callable] = None,
) -> tf.data.Dataset:
    """
    Always resize + scale to float32 [0, 1] *before* optional noise augmentation.
    Applying Gaussian / salt-pepper on uint8 then clipping to [0, 1] is incorrect.
    """
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(
        lambda im, lab: (
            tf.cast(tf.image.resize(im, image_size), tf.float32) / 255.0,
            lab,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if training and augment is not None:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(10_000, reshuffle_each_iteration=training)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def character_accuracy_mnist(y_true: np.ndarray, y_pred_logits: np.ndarray) -> float:
    """One character per image -> equals standard classification accuracy."""
    y_hat = np.argmax(y_pred_logits, axis=-1)
    return float(np.mean(y_hat == y_true))


def _mnist_test_float32(x_u8: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """x_u8: (N,H,W) uint8 or (N,H,W,1). Returns (N,h,w,1) float32 in [0,1]."""
    if x_u8.ndim == 3:
        x = x_u8[..., None]
    else:
        x = x_u8
    x = tf.image.resize(x, image_size).numpy().astype(np.float32) / 255.0
    return x


def _predict_logits_in_batches(model: tf.keras.Model, x: np.ndarray, batch_size: int) -> np.ndarray:
    parts = []
    for i in range(0, len(x), batch_size):
        parts.append(model.predict(x[i : i + batch_size], verbose=0))
    return np.concatenate(parts, axis=0)


def apply_gaussian_noise_np(x: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(loc=0.0, scale=std, size=x.shape).astype(np.float32)
    return np.clip(x + noise, 0.0, 1.0)


def apply_salt_pepper_noise_np(
    x: np.ndarray,
    salt_prob: float,
    pepper_prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Independent Bernoulli masks; where both fire, pepper wins (set to 0)."""
    salt = rng.random(x.shape) < salt_prob
    pepper = rng.random(x.shape) < pepper_prob
    out = np.array(x, copy=True)
    out[salt & ~pepper] = 1.0
    out[pepper] = 0.0
    return out


def evaluate_noise_profile_accuracies(
    model: tf.keras.Model,
    x_test_u8: np.ndarray,
    y_test: np.ndarray,
    image_size: Tuple[int, int],
    noise_cfg: NoiseProfileEvalConfig,
    batch_size: int = 512,
) -> dict:
    """
    Report character-level accuracy on the *same* MNIST test labels under:
      - clean inputs
      - Gaussian noise only (fixed draw per run via `noise_cfg.seed`)
      - salt-and-pepper only (fixed masks per run)

    This matches the brief's intent: two noise *profiles*, each with an explicit metric.
    """
    y_true = y_test.astype(np.int64)
    x0 = _mnist_test_float32(x_test_u8, image_size)

    rng_g = np.random.default_rng(noise_cfg.seed)
    rng_s = np.random.default_rng(noise_cfg.seed + 1)
    x_gauss = apply_gaussian_noise_np(x0, noise_cfg.gaussian_std, rng_g)
    x_snp = apply_salt_pepper_noise_np(x0, noise_cfg.salt_prob, noise_cfg.pepper_prob, rng_s)

    logits_clean = _predict_logits_in_batches(model, x0, batch_size)
    logits_g = _predict_logits_in_batches(model, x_gauss, batch_size)
    logits_s = _predict_logits_in_batches(model, x_snp, batch_size)

    return {
        "val_character_accuracy_clean": character_accuracy_mnist(y_true, logits_clean),
        "val_character_accuracy_gaussian_noise": character_accuracy_mnist(y_true, logits_g),
        "val_character_accuracy_salt_pepper_noise": character_accuracy_mnist(y_true, logits_s),
    }


def train_mnist_classifier(
    cfg: TrainConfig,
    noise_eval: Optional[NoiseProfileEvalConfig] = None,
) -> Tuple[tf.keras.Model, dict]:
    set_seed(cfg.seed)
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # Keep raw uint8 for resize in tf.data (GPU-friendly path).
    x_train = x_train[..., None]
    x_test = x_test[..., None]

    augment = make_augment_fn(use_gaussian=True, use_snp=True)
    train_ds = make_tf_dataset(x_train, y_train, cfg.batch_size, True, cfg.image_size, augment=augment)
    val_ds = make_tf_dataset(x_test, y_test, cfg.batch_size, False, cfg.image_size, augment=None)

    model = build_mnist_cnn(num_classes=10, image_size=cfg.image_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_acc", patience=3, restore_best_weights=True, mode="max"),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs, callbacks=callbacks, verbose=1)

    # Final explicit metrics (clean + per-noise-profile), full test set.
    noise_eval = noise_eval or NoiseProfileEvalConfig()
    profile_metrics = evaluate_noise_profile_accuracies(
        model,
        x_test,
        y_test,
        cfg.image_size,
        noise_eval,
        batch_size=max(cfg.batch_size, 256),
    )
    metrics = {**profile_metrics, "history": history.history}
    clean_acc = profile_metrics["val_character_accuracy_clean"]
    print("\n=== Validation character-level accuracy (MNIST test, 10k images) ===")
    print(f"  Clean:               {profile_metrics['val_character_accuracy_clean']:.4f}")
    print(f"  Gaussian noise only: {profile_metrics['val_character_accuracy_gaussian_noise']:.4f}  (std={noise_eval.gaussian_std})")
    print(
        f"  Salt-pepper only:    {profile_metrics['val_character_accuracy_salt_pepper_noise']:.4f}  "
        f"(p_salt={noise_eval.salt_prob}, p_pepper={noise_eval.pepper_prob})"
    )
    print("====================================================================")
    if clean_acc < cfg.min_char_acc:
        print(
            f"WARNING: clean accuracy {clean_acc:.4f} is below brief threshold {cfg.min_char_acc:.2f} "
            "(eligibility is usually stated on validation/clean performance — confirm with organizers)."
        )
    return model, metrics


# -----------------------------------------------------------------------------
# CTC: synthetic multi-digit lines from MNIST (demonstration for your writeup)
# -----------------------------------------------------------------------------

BLANK_INDEX = 10  # digits 0-9 -> logits indices 0-9, blank is index 10 (num_classes-1 pattern)


def synthesize_digit_line_batch(
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    batch_size: int,
    min_len: int = 3,
    max_len: int = 6,
    digit_size: Tuple[int, int] = (28, 28),
    pad: int = 4,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (batch, H, W, 1) line images by concatenating MNIST digits left-to-right.

    Returns:
      images uint8
      y_dense: (batch, max_len) padded with -1 (unused; we use lengths)
      y_length: (batch,) true label lengths
    """
    rng = rng or np.random.default_rng()
    h, w = digit_size
    H = h + 4

    images = []
    labels = []
    lengths = []
    for _ in range(batch_size):
        seq_len = int(rng.integers(min_len, max_len + 1))
        idxs = rng.integers(0, len(x_pool), size=seq_len)
        tiles = [x_pool[i] for i in idxs]
        labs = [int(y_pool[i]) for i in idxs]
        Wtot = seq_len * w + (seq_len + 1) * pad
        canvas = np.zeros((H, Wtot), dtype=np.uint8)
        x0 = pad
        for t in tiles:
            th, tw = t.shape
            y0 = (H - th) // 2
            canvas[y0 : y0 + th, x0 : x0 + tw] = np.maximum(canvas[y0 : y0 + th, x0 : x0 + tw], t)
            x0 += tw + pad
        images.append(canvas[..., None])
        labels.append(labs)
        lengths.append(seq_len)

    # Same batch must share (H, W) for np.stack — width varies with random seq_len.
    max_w = max(im.shape[1] for im in images)
    padded: List[np.ndarray] = []
    for im in images:
        hh, ww, cc = im.shape
        if ww == max_w:
            padded.append(im)
            continue
        plane = np.zeros((H, max_w, 1), dtype=np.uint8)
        plane[:, :ww, :] = im
        padded.append(plane)

    max_len = max(lengths)
    y_dense = np.full((batch_size, max_len), fill_value=-1, dtype=np.int32)
    for i, lab in enumerate(labels):
        y_dense[i, : len(lab)] = np.array(lab, dtype=np.int32)
    y_length = np.array(lengths, dtype=np.int32)
    return np.stack(padded, axis=0), y_dense, y_length


def line_ctc_time_steps(max_width: int, *, halve_width_twice: bool = True) -> int:
    """
    Horizontal length ``T`` of logits from ``build_line_cnn_ctc`` (Keras valid MaxPool, stride = pool).

    If ``halve_width_twice`` is True (MNIST line default): two 2×2 pools → width scales ~×1/4.
    If False (office long lines): second pool is 2×1 → width scales ~×1/2 (more CTC frames).
    """
    w = int(max_width)
    if w < 2:
        return max(w, 1)
    w = (w - 2) // 2 + 1
    if halve_width_twice:
        if w < 2:
            return max(w, 1)
        w = (w - 2) // 2 + 1
    return w


def min_line_canvas_width_for_ctc_time_steps(
    min_t: int, *, halve_width_twice: bool = True, cap: int = 32768
) -> int:
    """Smallest ``max_width`` (step 32) with ``line_ctc_time_steps(width, ...) >= min_t``."""
    need = max(1, int(min_t))
    for w in range(32, cap + 1, 32):
        if line_ctc_time_steps(w, halve_width_twice=halve_width_twice) >= need:
            return w
    return cap


def build_line_cnn_ctc(
    image_height: int,
    max_width: int,
    num_classes: int,
    *,
    halve_width_twice: bool = True,
) -> tf.keras.Model:
    """
    CNN trunk ending in per-time-step logits for CTC.

    num_classes must include blank, e.g. digits 0-9 + blank => num_classes=11, blank index 10.

    Spatial flow (example H=32): two pooling stages shrink height strongly; width shrinks by
    two 2×2 pools (``halve_width_twice=True``, MNIST default) or by one 2×2 then 2×1
    (``halve_width_twice=False``, better for very long transcripts). Then height is collapsed
    with a mean over rows to get one feature vector per horizontal step.
    """
    inputs = tf.keras.Input(shape=(image_height, max_width, 1), name="line_image")
    x = inputs
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    if halve_width_twice:
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    else:
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1))(x)

    # Collapse remaining height to a single row of features per x-position (robust for H=32 -> 8).
    x = tf.keras.layers.Lambda(lambda t: tf.reduce_mean(t, axis=1))(x)  # (B, W', C)
    x = tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    logits = tf.keras.layers.Dense(num_classes, activation=None, name="logits")(x)  # (b, T, C)
    return tf.keras.Model(inputs, logits, name="line_cnn_ctc")


def ctc_batch_loss(
    y_true_dense: tf.Tensor,
    y_true_length: tf.Tensor,
    logits: tf.Tensor,
    blank_index: Optional[int] = None,
) -> tf.Tensor:
    """
    y_true_dense: (B, Lmax) int32, padded with arbitrary value not used beyond label_length.
    y_true_length: (B,) int32
    logits: (B, T, C)
    blank_index: defaults to digit CTC ``BLANK_INDEX`` (10); set to ``len(vocab)`` for char CTC.
    """
    bi = BLANK_INDEX if blank_index is None else blank_index
    y_true_sparse = tf.RaggedTensor.from_tensor(y_true_dense, lengths=y_true_length).to_sparse()
    # time-major logits expected by tf.nn.ctc_loss
    logit_length = tf.fill([tf.shape(logits)[0]], tf.shape(logits)[1])
    logits_t = tf.transpose(logits, [1, 0, 2])  # (T, B, C)
    loss = tf.nn.ctc_loss(
        labels=y_true_sparse,
        logits=logits_t,
        label_length=y_true_length,
        logit_length=logit_length,
        blank_index=bi,
    )
    return tf.reduce_mean(loss)


def ctc_greedy_decode(logits: tf.Tensor) -> tf.Tensor:
    """
    Greedy CTC decode. logits batch-major (B, T, C).

    Uses Keras `ctc_decode` helper (batch-major API) for portability across TF versions.
    `ctc_decode` expects probabilities in many setups; we apply softmax for decoding only.
    """
    y_pred = tf.nn.softmax(logits, axis=-1)
    input_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
    decoded_list, _ = tf.keras.backend.ctc_decode(y_pred, input_length, greedy=True)
    return decoded_list[0]  # int dense (B, T) padded


def train_ctc_model(
    epochs: int = 8,
    steps_per_epoch: int = 200,
    batch_size: int = 64,
    max_width: int = 34 * 8,
    image_height: int = 32,
    seed: int = 42,
    sidd_patch_cache: Optional[Sequence[np.ndarray]] = None,
    sidd_strength: float = 0.0,
) -> tf.keras.Model:
    """
    Train CNN+CTC on synthetic MNIST lines.

    Optional: pass `sidd_patch_cache` (list of (H,W,1) float patches from `build_sidd_patch_cache`)
    and `sidd_strength` > 0 to blend real SIDD texture/noise into each training batch for robustness.

    Note: this is a *demo* trainer using a manual loop for clarity; you can wrap
    it in tf.keras.Model.train_step for integration with model.fit.
    """
    set_seed(seed)
    rng_py = np.random.default_rng(seed + 7)
    (x_train, y_train), _ = load_mnist()
    x_train = x_train.astype(np.uint8)
    num_classes = 11  # 10 digits + blank

    model = build_line_cnn_ctc(image_height=image_height, max_width=max_width, num_classes=num_classes)
    opt = tf.keras.optimizers.Adam(1e-3)

    @tf.function
    def train_step(x, y_dense, y_len):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = ctc_batch_loss(y_dense, y_len, logits, blank_index=None)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    use_sidd = bool(sidd_patch_cache) and sidd_strength > 0
    if use_sidd:
        print(f"[CTC] SIDD patch blend enabled (strength={sidd_strength:.3f}, cache={len(sidd_patch_cache)} patches).")

    for epoch in range(epochs):
        losses = []
        for _ in range(steps_per_epoch):
            imgs, y_dense, y_len = synthesize_digit_line_batch(
                x_train, y_train, batch_size=batch_size, min_len=3, max_len=6
            )
            x = tf.image.resize(imgs, (image_height, max_width)).numpy().astype(np.float32) / 255.0
            if use_sidd:
                x = apply_sidd_patch_blend(x, sidd_patch_cache, rng_py, strength=sidd_strength)
            loss = train_step(tf.constant(x), tf.constant(y_dense), tf.constant(y_len))
            losses.append(float(loss.numpy()))
        print(f"[CTC demo] epoch {epoch+1}/{epochs} mean loss={float(np.mean(losses)):.4f}")

    return model


def _ctc_pred_digits_from_dense_row(row: np.ndarray) -> List[int]:
    pred = [int(v) for v in np.asarray(row).tolist() if int(v) != -1]
    return [p for p in pred if p != BLANK_INDEX and p != -1]


def evaluate_ctc_sequence_accuracies_all_profiles(
    model: tf.keras.Model,
    noise_cfg: Optional[NoiseProfileEvalConfig] = None,
    batch_size: int = 256,
    n_batches: int = 20,
    max_width: int = 34 * 8,
    H: int = 32,
    sidd_patch_cache: Optional[Sequence[np.ndarray]] = None,
    sidd_strength: float = 0.0,
) -> dict:
    """
    Strict *sequence* accuracy (full digit string must match) on synthetic MNIST lines.

    For each batch we build **one** set of line images and labels, then evaluate the **same**
    lines under: clean, Gaussian noise only, salt-and-pepper only, and optionally SIDD blend
    (same strength as training when SIDD is enabled).
    """
    noise_cfg = noise_cfg or NoiseProfileEvalConfig()
    _, (x_test, y_test) = load_mnist()
    rng_g = np.random.default_rng(noise_cfg.seed)
    rng_s = np.random.default_rng(noise_cfg.seed + 1)
    rng_sidd = np.random.default_rng(noise_cfg.seed + 2)

    keys = [
        "sequence_accuracy_clean",
        "sequence_accuracy_gaussian_noise",
        "sequence_accuracy_salt_pepper_noise",
    ]
    if sidd_patch_cache is not None and len(sidd_patch_cache) > 0 and sidd_strength > 0:
        keys.append("sequence_accuracy_sidd_blend")
    counts = {k: [0, 0] for k in keys}

    for _ in range(n_batches):
        imgs, y_dense, y_len = synthesize_digit_line_batch(x_test, y_test, batch_size=batch_size)
        x0 = tf.image.resize(imgs, (H, max_width)).numpy().astype(np.float32) / 255.0
        bundles: List[Tuple[str, np.ndarray]] = [
            ("sequence_accuracy_clean", x0),
            ("sequence_accuracy_gaussian_noise", apply_gaussian_noise_np(x0, noise_cfg.gaussian_std, rng_g)),
            (
                "sequence_accuracy_salt_pepper_noise",
                apply_salt_pepper_noise_np(x0, noise_cfg.salt_prob, noise_cfg.pepper_prob, rng_s),
            ),
        ]
        if "sequence_accuracy_sidd_blend" in counts:
            bundles.append(
                (
                    "sequence_accuracy_sidd_blend",
                    apply_sidd_patch_blend(x0, sidd_patch_cache, rng_sidd, strength=sidd_strength),
                )
            )

        for key, x_in in bundles:
            logits = model.predict(x_in, verbose=0)
            dense = ctc_greedy_decode(tf.constant(logits, dtype=tf.float32)).numpy()
            for i in range(batch_size):
                true = y_dense[i, : y_len[i]].tolist()
                pred = _ctc_pred_digits_from_dense_row(dense[i])
                counts[key][1] += 1
                if pred == true:
                    counts[key][0] += 1

    return {k: counts[k][0] / max(counts[k][1], 1) for k in keys}


def ctc_sequence_accuracy_on_batch(
    model: tf.keras.Model,
    batch_size: int = 256,
    sidd_patch_cache: Optional[Sequence[np.ndarray]] = None,
    sidd_strength: float = 0.0,
) -> float:
    """Backward-compatible single scalar: SIDD-blend acc if SIDD given, else clean."""
    m = evaluate_ctc_sequence_accuracies_all_profiles(
        model,
        batch_size=batch_size,
        n_batches=20,
        sidd_patch_cache=sidd_patch_cache,
        sidd_strength=sidd_strength,
    )
    if sidd_patch_cache is not None and len(sidd_patch_cache) > 0 and sidd_strength > 0:
        return float(m["sequence_accuracy_sidd_blend"])
    return float(m["sequence_accuracy_clean"])


# -----------------------------------------------------------------------------
# Inference service (optional)
# -----------------------------------------------------------------------------

_global_model: Optional[tf.keras.Model] = None
_global_line_model: Optional[tf.keras.Model] = None

DEFAULT_LINE_IMAGE_HEIGHT = 32
DEFAULT_LINE_MAX_WIDTH = 34 * 8


def load_model_for_serving(export_dir: str) -> None:
    global _global_model
    _global_model = tf.keras.models.load_model(export_dir, compile=False)


def load_line_ctc_model_for_serving(export_dir: str) -> None:
    """Load the CTC line OCR model (e.g. `line_ctc_sidd.keras` from --ctc-export)."""
    global _global_line_model
    _global_line_model = tf.keras.models.load_model(export_dir, compile=False)


def predict_digit_proba(image_f32_chw_or_hwc: np.ndarray, image_size: Tuple[int, int] = (32, 32)) -> np.ndarray:
    assert _global_model is not None, "Call load_model_for_serving first."
    x = image_f32_chw_or_hwc
    if x.ndim == 2:
        x = x[..., None]
    if x.ndim == 3:
        x = x[None, ...]
    x = tf.image.resize(x, image_size).numpy().astype(np.float32) / (255.0 if x.max() > 1.5 else 1.0)
    return _global_model.predict(x, verbose=0)[0]


def decode_line_ctc_image_to_digit_string(
    image_u8: np.ndarray,
    image_height: int = DEFAULT_LINE_IMAGE_HEIGHT,
    max_width: int = DEFAULT_LINE_MAX_WIDTH,
) -> str:
    """
    Stage-1 handoff for teammates: image -> **extracted digit string** (ASCII, e.g. ``"304"``).

    Pass this ``text`` to the Stage-2 Huffman microservice as the payload to compress.
    Expects ``_global_line_model`` loaded (``LINE_CTC_MODEL_DIR`` when serving).
    """
    assert _global_line_model is not None, "Call load_line_ctc_model_for_serving first."
    x = np.asarray(image_u8)
    if x.ndim == 2:
        x = x[..., None]
    if x.ndim == 3:
        x = x[None, ...]
    scale = 255.0 if float(np.max(x)) > 1.5 else 1.0
    x = tf.image.resize(tf.constant(x), (image_height, max_width)).numpy().astype(np.float32) / scale
    logits = _global_line_model.predict(x, verbose=0)
    dense = ctc_greedy_decode(tf.constant(logits, dtype=tf.float32)).numpy()[0]
    pred = _ctc_pred_digits_from_dense_row(dense)
    return "".join(str(d) for d in pred)


def create_fastapi_app(image_size: Tuple[int, int] = (32, 32)):
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, File, HTTPException, UploadFile

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        path = os.environ.get("MNIST_MODEL_DIR", "")
        if path:
            load_model_for_serving(path)
        line = os.environ.get("LINE_CTC_MODEL_DIR", "")
        if line:
            load_line_ctc_model_for_serving(line)
        yield

    app = FastAPI(title="OCR Stage-1 (single digit + line CTC)", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {
            "digit_model_loaded": _global_model is not None,
            "line_ctc_model_loaded": _global_line_model is not None,
        }

    @app.post("/ocr")
    async def ocr_digit(file: UploadFile = File(...)):
        if _global_model is None:
            raise HTTPException(status_code=503, detail="MNIST_MODEL_DIR not set or model failed to load.")
        raw = await file.read()
        im = tf.image.decode_image(raw, channels=1, dtype=tf.uint8).numpy()
        prob = predict_digit_proba(im, image_size=image_size)
        digit = int(np.argmax(prob))
        return {"digit": digit, "confidence": float(np.max(prob)), "text": str(digit)}

    @app.post("/ocr/line")
    async def ocr_line(file: UploadFile = File(...)):
        """
        Multi-digit **line** image -> JSON ``{"text": "..."}`` for Stage-2 Huffman compression.

        Input image should resemble training: horizontal digit string on a dark/light background;
        it is resized to (32, 272) the same way as in training.
        """
        if _global_line_model is None:
            raise HTTPException(
                status_code=503,
                detail="LINE_CTC_MODEL_DIR not set or line CTC model failed to load.",
            )
        raw = await file.read()
        im = tf.image.decode_image(raw, channels=1, dtype=tf.uint8).numpy()
        text = decode_line_ctc_image_to_digit_string(im)
        return {"text": text, "length": len(text)}

    return app


# -----------------------------------------------------------------------------
# SIDD patch noise (for robust CTC training — real sensor texture / noise)
# -----------------------------------------------------------------------------

_SIDD_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG", ".JPEG"}


def collect_sidd_image_paths(root: str, max_files: int = 8000) -> List[str]:
    """
    Recursively collect image paths under `root` (typical unpacked SIDD tree of PNGs).

    You must download SIDD yourself (license/size); point `--sidd-dir` at the folder
    that contains the `.png` / `.jpg` files (e.g. extracted SRGB blocks).
    """
    root_p = Path(root).expanduser().resolve()
    if not root_p.is_dir():
        return []
    paths: List[str] = []
    for p in sorted(root_p.rglob("*")):
        if p.is_file() and p.suffix in _SIDD_IMAGE_SUFFIXES and p.stat().st_size > 4096:
            paths.append(str(p))
            if len(paths) >= max_files:
                break
    return paths


def _decode_gray(path: str) -> tf.Tensor:
    data = tf.io.read_file(path)
    im = tf.image.decode_image(data, channels=3, dtype=tf.uint8, expand_animations=False)
    im.set_shape([None, None, 3])
    return tf.image.rgb_to_grayscale(im)


def build_sidd_patch_cache(
    paths: Sequence[str],
    height: int,
    width: int,
    cache_size: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """
    Preload random crops from SIDD images, all resized to (height, width, 1) float32 [0,1].
    Training then *blends* these patches into synthetic line images for domain-style noise.
    """
    if not paths:
        return []
    cache: List[np.ndarray] = []
    tries = 0
    max_tries = max(cache_size * 15, 200)
    while len(cache) < cache_size and tries < max_tries:
        tries += 1
        path = paths[int(rng.integers(0, len(paths)))]
        try:
            g = _decode_gray(path)
            h = int(g.shape[0])
            w = int(g.shape[1])
            if h < 8 or w < 8:
                continue
            if h >= height and w >= width:
                i = int(rng.integers(0, h - height + 1))
                j = int(rng.integers(0, w - width + 1))
                crop = g[i : i + height, j : j + width]
            else:
                crop = tf.image.resize(g, (height, width), method="area")
            crop = tf.cast(crop, tf.float32) / 255.0
            cache.append(crop.numpy().astype(np.float32))
        except Exception:
            continue
    return cache


def apply_sidd_patch_blend(
    x: np.ndarray,
    patch_cache: Sequence[np.ndarray],
    rng: np.random.Generator,
    strength: float,
) -> np.ndarray:
    """
    Per-sample convex blend with a random SIDD patch (same spatial size as x).

    x: (B, H, W, 1) float32 in [0,1]
    strength in [0,1]: 0 = no SIDD, 1 = pure patch (usually too strong; try 0.2–0.5).
    """
    if not patch_cache or strength <= 0:
        return x
    strength = float(np.clip(strength, 0.0, 1.0))
    out = np.empty_like(x, dtype=np.float32)
    for b in range(x.shape[0]):
        patch = patch_cache[int(rng.integers(0, len(patch_cache)))]
        out[b] = np.clip((1.0 - strength) * x[b] + strength * patch, 0.0, 1.0)
    return out


# -----------------------------------------------------------------------------
# SIDD notes
# -----------------------------------------------------------------------------

def sidd_evaluation_notes() -> str:
    return (
        "SIDD contains noisy/clean real image pairs for denoising benchmarks, not OCR transcripts.\n"
        "This repo (ctc_demo): pass --sidd-dir to blend random SIDD *patches* into synthetic digit lines during CTC training.\n"
        "Other reasonable hackathon uses:\n"
        "  - Robustness: report accuracy clean vs SIDD-blended eval (script prints both when SIDD is enabled).\n"
        "  - Preprocessing: denoise before OCR (extra scope).\n"
        "For OCR *accuracy* on real text, you still need (image, transcript) pairs."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["mnist", "ctc_demo", "export", "notes", "office_ctc"],
        default="mnist",
        help="office_ctc: real clean+noisy office PNGs + Tesseract labels (see office_line_ctc.py).",
    )
    parser.add_argument("--export_dir", default="artifacts/mnist_digit_model.keras")
    parser.add_argument(
        "--sidd-dir",
        default="",
        help="Root folder of SIDD (or any) RGB/gray PNGs/JPGs for patch noise during --mode ctc_demo.",
    )
    parser.add_argument(
        "--sidd-strength",
        type=float,
        default=0.35,
        help="Blend weight for SIDD patch (0 disables). Typical 0.2–0.45.",
    )
    parser.add_argument(
        "--sidd-cache",
        type=int,
        default=384,
        help="Number of random (H,W) patches to preload from disk into RAM.",
    )
    parser.add_argument(
        "--ctc-export",
        default="",
        help="Optional path to save the CTC model after ctc_demo (e.g. artifacts/line_ctc.keras).",
    )
    parser.add_argument("--ctc-epochs", type=int, default=8)
    parser.add_argument("--ctc-steps", type=int, default=200)
    parser.add_argument("--office-epochs", type=int, default=25)
    parser.add_argument("--office-steps", type=int, default=40)
    parser.add_argument("--office-sidd-dir", default="", help="With --mode office_ctc: SIDD folder for train blend.")
    parser.add_argument("--office-sidd-strength", type=float, default=0.0)
    parser.add_argument(
        "--office-sidd-eval-dir",
        default="",
        help="SIDD folder for office *eval* four-way (defaults to --office-sidd-dir if set).",
    )
    parser.add_argument("--office-eval-sidd-strength", type=float, default=0.35)
    parser.add_argument(
        "--office-force-transcripts",
        action="store_true",
        help="Regenerate artifacts/office_transcripts.csv with Tesseract.",
    )
    parser.add_argument(
        "--office-no-bootstrap",
        action="store_true",
        help="Do not auto-create office_transcripts.csv when missing (you must supply CSV).",
    )
    parser.add_argument(
        "--office-rebuild-val-tesseract",
        action="store_true",
        help="Regenerate validation-only Tesseract reference CSV.",
    )
    parser.add_argument(
        "--office-max-width",
        type=int,
        default=0,
        help="Office CTC canvas width (default 4096). Must be large enough vs longest transcript.",
    )
    parser.add_argument(
        "--office-image-height",
        type=int,
        default=0,
        help="Office CTC input height (default 128).",
    )
    parser.add_argument(
        "--office-legacy-double-width-pool",
        action="store_true",
        help="Two 2×2 pools on width (fewer CTC time steps; needs a much wider canvas for long text).",
    )
    args = parser.parse_args()

    if args.mode == "notes":
        print(sidd_evaluation_notes())
        return

    if args.mode == "office_ctc":
        import office_line_ctc as ol

        cfg = ol.OfficeTrainConfig(
            epochs=args.office_epochs,
            steps_per_epoch=args.office_steps,
            max_width=args.office_max_width or ol.OfficeTrainConfig.max_width,
            image_height=args.office_image_height or ol.OfficeTrainConfig.image_height,
            halve_width_twice=bool(args.office_legacy_double_width_pool),
            sidd_dir=args.office_sidd_dir,
            train_sidd_strength=float(args.office_sidd_strength),
            sidd_eval_dir=args.office_sidd_eval_dir,
            eval_sidd_strength=float(args.office_eval_sidd_strength),
            bootstrap_tesseract=(not args.office_no_bootstrap) or bool(args.office_force_transcripts),
            force_transcripts=bool(args.office_force_transcripts),
            rebuild_val_tesseract=bool(args.office_rebuild_val_tesseract),
        )
        ol.train_office_ctc(cfg)
        return

    if args.mode == "mnist":
        cfg = TrainConfig()
        model, metrics = train_mnist_classifier(cfg)
        os.makedirs(os.path.dirname(args.export_dir) or ".", exist_ok=True)
        model.save(args.export_dir)
        print(f"Saved model to {args.export_dir}")
        return

    if args.mode == "ctc_demo":
        H, W = 32, 34 * 8
        sidd_cache: Optional[List[np.ndarray]] = None
        if args.sidd_dir:
            paths = collect_sidd_image_paths(args.sidd_dir)
            if not paths:
                print(f"[CTC] WARNING: no images found under --sidd-dir={args.sidd_dir!r}; training clean only.")
            else:
                print(f"[CTC] Found {len(paths)} image files; building {args.sidd_cache} SIDD patches ({H}x{W})…")
                rng_cache = np.random.default_rng(42)
                sidd_cache = build_sidd_patch_cache(paths, H, W, args.sidd_cache, rng_cache)
                if not sidd_cache:
                    print("[CTC] WARNING: SIDD patch cache empty (decode failures?); training clean only.")
                    sidd_cache = None
        strength = float(args.sidd_strength) if sidd_cache else 0.0
        m = train_ctc_model(
            epochs=args.ctc_epochs,
            steps_per_epoch=args.ctc_steps,
            sidd_patch_cache=sidd_cache,
            sidd_strength=strength,
        )
        ne = NoiseProfileEvalConfig()
        ctc_metrics = evaluate_ctc_sequence_accuracies_all_profiles(
            m,
            noise_cfg=ne,
            sidd_patch_cache=sidd_cache,
            sidd_strength=strength,
        )
        print("\n=== CTC strict sequence accuracy (synthetic MNIST lines, same batch / each corruption) ===")
        print(f"  Clean:               {ctc_metrics['sequence_accuracy_clean']:.4f}")
        print(f"  Gaussian noise only: {ctc_metrics['sequence_accuracy_gaussian_noise']:.4f}  (std={ne.gaussian_std})")
        print(
            f"  Salt-pepper only:    {ctc_metrics['sequence_accuracy_salt_pepper_noise']:.4f}  "
            f"(p_salt={ne.salt_prob}, p_pepper={ne.pepper_prob})"
        )
        if "sequence_accuracy_sidd_blend" in ctc_metrics:
            print(f"  SIDD patch blend:      {ctc_metrics['sequence_accuracy_sidd_blend']:.4f}  (strength={strength})")
        print("========================================================================")
        if args.ctc_export:
            os.makedirs(os.path.dirname(args.ctc_export) or ".", exist_ok=True)
            m.save(args.ctc_export)
            print(f"[CTC] Saved model to {args.ctc_export}")
        return

    if args.mode == "export":
        print("Train first with --mode mnist; export path is written by training.")


# Optional HTTP serving (requires `fastapi` + `uvicorn`):
#   PowerShell: $env:MNIST_MODEL_DIR=".../mnist_digit_model.keras"   # POST /ocr -> text single digit
#   $env:LINE_CTC_MODEL_DIR=".../line_ctc_sidd.keras"                 # POST /ocr/line -> {"text":"304"}
#   uvicorn mnist_ocr_workflow:app --host 127.0.0.1 --port 8000 --app-dir "<project dir>"
try:
    import fastapi  # noqa: F401

    app = create_fastapi_app()
except ModuleNotFoundError:  # training-only environments
    app = None  # type: ignore[assignment]


if __name__ == "__main__":
    main()

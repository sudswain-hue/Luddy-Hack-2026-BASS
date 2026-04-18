"""
Train character-level CTC on SimulatedNoisyOffice-style folders (clean + noisy PNGs).

**Training labels:** ``artifacts/office_transcripts.csv`` (path<TAB>transcript). If missing,
Tesseract is run **once** on all clean ``*_Clean_*.png`` files to create it (offline labeling only;
your deployed OCR is still CNN+CTC). Use ``--no-bootstrap`` to disable auto-creation.

**Validation reference:** ``artifacts/office_tesseract_val_reference.csv`` — Tesseract is run on
each **validation** image only, and printed metrics compare the CNN+CTC output to that reference.

**Eval corruptions:** clean, Gaussian-only, salt-pepper-only, and optional **SIDD patch blend**
(eval-only unless you also set train SIDD).

**Stage 2:** ``artifacts/stage2_val_export.jsonl`` — one JSON per val image with
``pred_text_clean``, ``pred_text_gaussian``, ``pred_text_salt_pepper``, ``pred_text_sidd_blend``,
and ``stage2_suggested_payload`` (= clean decode) for Huffman.

Run: ``python office_line_ctc.py train`` or ``python mnist_ocr_workflow.py --mode office_ctc``
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

import mnist_ocr_workflow as m

# -----------------------------------------------------------------------------
# Paths (relative to this file's parent = project root)
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DEFAULT_CLEAN_DIRS = [
    ROOT / "clean_images_grayscale",
    ROOT / "clean_images_grayscale_doubleresolution",
    ROOT / "clean_images_binaryscal (low resolution)",
]
DEFAULT_NOISY_DIR = ROOT / "simulated_noisy_images_grayscale"
TRANSCRIPTS_CSV = ROOT / "artifacts" / "office_transcripts.csv"
# Tesseract run *only* on validation-set images (TE+TR+VA val items) for reporting / strict metrics.
TESSERACT_VAL_CSV = ROOT / "artifacts" / "office_tesseract_val_reference.csv"
STAGE2_EXPORT_JSONL = ROOT / "artifacts" / "stage2_val_export.jsonl"
VOCAB_JSON = ROOT / "artifacts" / "office_line_vocab.json"
MODEL_OUT = ROOT / "artifacts" / "office_line_ctc.keras"


@dataclass
class OfficeTrainConfig:
    image_height: int = 128
    # Default canvas fits long Tesseract dumps when ``halve_width_twice`` is False (T ≈ max_width/2).
    max_width: int = 4096
    batch_size: int = 2  # small: long sequences + wide images
    # False: second pool is 2×1 → ~2× more CTC frames than two 2×2 pools (better for long text).
    halve_width_twice: bool = False
    epochs: int = 25
    steps_per_epoch: int = 40
    lr: float = 3e-4
    seed: int = 42
    # If False: no extra pixel noise in training (only what's already in ``simulated_noisy_*`` images).
    # False = rely on sensor noise in ``simulated_noisy_*`` only (no extra Gaussian/S&P on pixels).
    apply_extra_train_noise: bool = False
    train_gaussian_std: float = 0.04
    train_salt_prob: float = 0.004
    train_pepper_prob: float = 0.004
    train_sidd_strength: float = 0.0
    sidd_dir: str = ""
    # SIDD patches for *evaluation* only (clean / Gaussian / SNP / SIDD four-way); train can stay SIDD-free.
    sidd_eval_dir: str = ""
    eval_sidd_strength: float = 0.35
    # If True and ``office_transcripts.csv`` is missing, build it once with Tesseract (training labels).
    bootstrap_tesseract: bool = True
    force_transcripts: bool = False
    rebuild_val_tesseract: bool = False


@dataclass
class OfficeNoiseEval:
    """Slightly milder than legacy 0.01/0.01 when paired with train-time speckle."""

    gaussian_std: float = 0.05
    salt_prob: float = 0.005
    pepper_prob: float = 0.005
    seed: int = 1337


def _norm_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def noisy_basename_to_clean(noisy_name: str) -> str:
    for tag in ("Noisec", "Noisef", "Noisep", "Noisew"):
        if f"_{tag}_" in noisy_name:
            return noisy_name.replace(f"_{tag}_", "_Clean_")
    return noisy_name


def discover_clean_pngs(clean_dirs: Sequence[Path]) -> List[Path]:
    out: List[Path] = []
    for d in clean_dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.png")):
            if "_Clean_" in p.name:
                out.append(p)
    return out


def discover_noisy_pngs(noisy_dir: Path) -> List[Path]:
    if not noisy_dir.is_dir():
        return []
    return sorted([p for p in noisy_dir.glob("*.png") if "_Noise" in p.name])


def tesseract_transcript(image_path: Path) -> str:
    import pytesseract

    im = Image.open(image_path).convert("L")
    txt = pytesseract.image_to_string(im, lang="eng")
    return _norm_text(txt)


def require_tesseract_bootstrap_stack() -> None:
    """Fail fast before writing CSVs full of empty strings."""
    try:
        import pytesseract
    except ModuleNotFoundError as e:
        raise SystemExit(
            "[office] Python package `pytesseract` is missing.\n"
            "  .venv\\Scripts\\python -m pip install -r requirements-ocr.txt\n"
            "  (or: pip install pytesseract)\n"
            "You also need the Tesseract OCR *engine* installed and on PATH."
        ) from e
    try:
        pytesseract.get_tesseract_version()
    except Exception as e:
        raise SystemExit(
            "[office] Tesseract OCR engine not found or not on PATH "
            "(pytesseract is installed but cannot run `tesseract`).\n"
            "  Windows: install from https://github.com/UB-Mannheim/tesseract/wiki "
            "and add the install folder to PATH, or set Tesseract path in code.\n"
            f"Detail: {e}"
        ) from e


def _write_transcripts_from_tesseract(clean_paths: Sequence[Path], csv_path: Path) -> None:
    require_tesseract_bootstrap_stack()
    print(f"[office] Bootstrap: Tesseract -> {csv_path} ({len(clean_paths)} clean images)…")
    rows = []
    for i, p in enumerate(clean_paths):
        try:
            t = tesseract_transcript(p)
        except Exception as e:
            print(f"[office] WARN: OCR failed for {p.name}: {e}")
            t = ""
        rows.append((str(p.resolve()), t))
        if (i + 1) % 10 == 0:
            print(f"  … {i+1}/{len(clean_paths)}")
    if not any(t.strip() for _, t in rows):
        raise SystemExit(
            "[office] Bootstrap produced only empty transcripts (Tesseract failed every image).\n"
            "Fix Tesseract / PATH, delete the bad CSV if present, then re-run with:\n"
            "  --office-force-transcripts"
        )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("path\ttranscript\n")
        for path, t in rows:
            f.write(path.replace("\t", " ") + "\t" + t.replace("\t", " ").replace("\n", " ") + "\n")
    print("[office] Transcripts CSV ready (bootstrap only; your deployed model stays CNN+CTC).")


def ensure_transcripts_csv(
    clean_paths: Sequence[Path],
    csv_path: Path,
    *,
    bootstrap_tesseract: bool,
    force: bool = False,
) -> None:
    """
    ``office_transcripts.csv`` = **training supervision** (path<TAB>transcript).

    If the file is missing and ``bootstrap_tesseract`` is True (default), it is created by
    running Tesseract once on **all** clean ``*_Clean_*.png`` files (offline label generation only).
    """
    if csv_path.is_file() and not force:
        print(f"[office] Using existing training transcripts: {csv_path}")
        return
    if not bootstrap_tesseract:
        raise SystemExit(
            f"Missing {csv_path} and auto-labeling disabled.\n"
            "Add path<TAB>transcript rows, or re-run with default Tesseract bootstrap enabled.\n"
            "  python mnist_ocr_workflow.py --mode office_ctc"
        )
    _write_transcripts_from_tesseract(clean_paths, csv_path)


def ensure_val_tesseract_reference(
    val_items: Sequence[Tuple[Path, str, bool]],
    csv_path: Path,
    *,
    force: bool,
) -> Dict[str, str]:
    """Tesseract text for each **validation** image path (used only for metrics + export reference)."""
    if csv_path.is_file() and not force:
        return load_transcripts_csv(csv_path)
    print(f"[office] Building validation-only Tesseract references -> {csv_path} ({len(val_items)} images)…")
    require_tesseract_bootstrap_stack()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: Dict[str, str] = {}
    for i, (p, _csv_gt, _) in enumerate(val_items):
        key = str(p.resolve())
        try:
            rows[key] = tesseract_transcript(p)
        except Exception as e:
            print(f"[office] WARN val tess {p.name}: {e}")
            rows[key] = ""
        if (i + 1) % 5 == 0:
            print(f"  … val tess {i+1}/{len(val_items)}")
    if val_items and not any(t.strip() for t in rows.values()):
        raise SystemExit(
            "[office] Validation Tesseract produced only empty strings. "
            "Fix Tesseract / PATH and use --office-rebuild-val-tesseract."
        )
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("path\ttranscript\n")
        for path, t in sorted(rows.items()):
            f.write(path.replace("\t", " ") + "\t" + t.replace("\t", " ").replace("\n", " ") + "\n")
    print("[office] Validation Tesseract reference CSV ready.")
    return rows


def load_transcripts_csv(csv_path: Path) -> Dict[str, str]:
    m_: Dict[str, str] = {}
    with open(csv_path, encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            m_[parts[0]] = _norm_text(parts[1])
    return m_


def build_charset(transcripts: Sequence[str]) -> str:
    chars = set()
    for t in transcripts:
        for c in t:
            if ord(c) < 32:
                continue
            chars.add(c)
    # stable order: printable ascii first seen order then rest sorted
    ordered: List[str] = []
    for c in sorted(chars, key=lambda x: (ord(x) > 126, ord(x))):
        if c not in ordered:
            ordered.append(c)
    return "".join(ordered)


def encode_label(text: str, vocab: str) -> List[int]:
    ids: List[int] = []
    for c in text:
        if c in vocab:
            ids.append(vocab.index(c))
    return ids


def labels_to_dense_batch(labels: List[List[int]], pad: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    max_len = max((len(x) for x in labels), default=0)
    b = len(labels)
    y = np.full((b, max_len), fill_value=pad, dtype=np.int32)
    lengths = np.zeros((b,), dtype=np.int32)
    for i, seq in enumerate(labels):
        lengths[i] = len(seq)
        y[i, : len(seq)] = np.array(seq, dtype=np.int32)
    return y, lengths


def pred_indices_to_text(ids: List[int], vocab: str, blank_index: int) -> str:
    out = []
    for i in ids:
        if i < 0 or i == blank_index or i >= len(vocab):
            continue
        out.append(vocab[i])
    return "".join(out)


def dense_row_to_indices(row: np.ndarray, blank_index: int) -> List[int]:
    pred = [int(v) for v in np.asarray(row).tolist() if int(v) != -1]
    return [p for p in pred if p != blank_index]


def load_gray_u8(path: Path) -> np.ndarray:
    data = tf.io.read_file(str(path))
    im = tf.image.decode_png(data, channels=1, dtype=tf.uint8)
    return im.numpy()


def pad_batch_images(images: List[np.ndarray], target_h: int, target_w: int) -> np.ndarray:
    """images: list of (h,w,1) uint8 -> (B,target_h,target_w,1) float32 [0,1]"""
    b = len(images)
    out = np.zeros((b, target_h, target_w, 1), dtype=np.float32)
    for i, im in enumerate(images):
        t = tf.image.resize(im, (target_h, target_w), method="area").numpy().astype(np.float32) / 255.0
        out[i] = t
    return out


def split_from_name(path: Path) -> str:
    """TE / TR / VA from ``*_TE.png`` etc."""
    stem = path.stem
    if stem.endswith("_TE"):
        return "TE"
    if stem.endswith("_TR"):
        return "TR"
    if stem.endswith("_VA"):
        return "VA"
    return "UNK"


def build_dataset_items(
    clean_dirs: Sequence[Path],
    noisy_dir: Path,
    transcripts: Dict[str, str],
) -> Tuple[List[Tuple[Path, str, bool]], List[Tuple[Path, str, bool]]]:
    train: List[Tuple[Path, str, bool]] = []
    val: List[Tuple[Path, str, bool]] = []
    for p in discover_clean_pngs(clean_dirs):
        key = str(p.resolve())
        t = transcripts.get(key, "")
        if not t:
            continue
        sp = split_from_name(p)
        # TE+TR for training (more samples); VA for validation only.
        (train if sp in ("TE", "TR") else val).append((p, t, False))
    for p in discover_noisy_pngs(noisy_dir):
        clean_name = noisy_basename_to_clean(p.name)
        clean_path = None
        for d in clean_dirs:
            cand = d / clean_name
            if cand.is_file():
                clean_path = cand
                break
        if clean_path is None:
            continue
        key = str(clean_path.resolve())
        t = transcripts.get(key, "")
        if not t:
            continue
        sp = split_from_name(p)
        (train if sp in ("TE", "TR") else val).append((p, t, True))
    return train, val


def train_office_ctc(cfg: OfficeTrainConfig) -> None:
    m.set_seed(cfg.seed)
    clean_dirs = [Path(d) for d in DEFAULT_CLEAN_DIRS]
    noisy_dir = Path(DEFAULT_NOISY_DIR)
    clean_paths = discover_clean_pngs(clean_dirs)
    if not clean_paths:
        raise SystemExit("No clean *_Clean_*.png found under default clean_* folders.")

    if cfg.force_transcripts and TRANSCRIPTS_CSV.is_file():
        TRANSCRIPTS_CSV.unlink()
    ensure_transcripts_csv(
        clean_paths,
        TRANSCRIPTS_CSV,
        bootstrap_tesseract=cfg.bootstrap_tesseract,
        force=cfg.force_transcripts,
    )
    transcripts_map = load_transcripts_csv(TRANSCRIPTS_CSV)
    train_items, val_items = build_dataset_items(clean_dirs, noisy_dir, transcripts_map)
    if not train_items:
        raise SystemExit("No training items (TE+TR split). Check transcripts CSV.")

    val_tess_ref = ensure_val_tesseract_reference(
        val_items,
        TESSERACT_VAL_CSV,
        force=cfg.rebuild_val_tesseract,
    )
    all_text = [t for _, t, _ in train_items] + [t for t in val_tess_ref.values() if t]
    vocab = build_charset([t for t in all_text if t])
    if not vocab:
        raise SystemExit("Empty vocabulary — transcripts missing or empty.")
    blank_index = len(vocab)
    num_classes = len(vocab) + 1
    print(f"[office] Vocab size={len(vocab)} (+blank) train={len(train_items)} val={len(val_items)}")

    max_lab = max(
        (len(encode_label(t, vocab)) for _, t, _ in train_items),
        default=0,
    )
    max_lab = max(max_lab, max((len(encode_label(t, vocab)) for _, t, _ in val_items), default=0))
    # Worst-case CTC alignment length grows ~2× label length (repeated chars need blanks).
    min_t = max(1, 2 * max_lab)
    t_steps = m.line_ctc_time_steps(cfg.max_width, halve_width_twice=cfg.halve_width_twice)
    if t_steps < min_t:
        suggest = m.min_line_canvas_width_for_ctc_time_steps(
            min_t, halve_width_twice=cfg.halve_width_twice
        )
        extra = ""
        if cfg.halve_width_twice:
            extra = (
                "\n  Tip: omit --office-legacy-double-width-pool (default) for ~2× more CTC frames "
                "at the same canvas width."
            )
        raise SystemExit(
            f"[office] Longest encoded label length={max_lab}; CTC needs at least ~{min_t} time steps, "
            f"but max_width={cfg.max_width} yields T={t_steps} (halve_width_twice={cfg.halve_width_twice}).\n"
            f"  Set max_width to at least {suggest} (e.g. --office-max-width {suggest}).{extra}"
        )
    print(
        f"[office] CTC T={t_steps} (max_width={cfg.max_width}, halve_width_twice={cfg.halve_width_twice}); "
        f"longest label len={max_lab}"
    )

    sidd_cache: Optional[List[np.ndarray]] = None
    rng_sidd = np.random.default_rng(cfg.seed + 3)
    if cfg.sidd_dir:
        paths = m.collect_sidd_image_paths(cfg.sidd_dir)
        if paths:
            sidd_cache = m.build_sidd_patch_cache(paths, cfg.image_height, cfg.max_width, 256, rng_sidd)
            print(f"[office] SIDD train cache={len(sidd_cache or [])} patches (train blend if strength>0)")

    sidd_eval_cache: Optional[List[np.ndarray]] = None
    sed = (cfg.sidd_eval_dir or "").strip() or (cfg.sidd_dir or "").strip()
    if sed:
        ep = m.collect_sidd_image_paths(sed)
        if ep:
            sidd_eval_cache = m.build_sidd_patch_cache(
                ep, cfg.image_height, cfg.max_width, 256, np.random.default_rng(cfg.seed + 11)
            )
            print(f"[office] SIDD eval cache={len(sidd_eval_cache)} patches (eval SIDD blend strength={cfg.eval_sidd_strength})")

    model = m.build_line_cnn_ctc(
        cfg.image_height,
        cfg.max_width,
        num_classes,
        halve_width_twice=cfg.halve_width_twice,
    )
    opt = tf.keras.optimizers.Adam(cfg.lr)
    rng = np.random.default_rng(cfg.seed)

    @tf.function
    def train_step(x, y_dense, y_len):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = m.ctc_batch_loss(y_dense, y_len, logits, blank_index=blank_index)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for epoch in range(cfg.epochs):
        losses = []
        for _ in range(cfg.steps_per_epoch):
            idxs = rng.integers(0, len(train_items), size=cfg.batch_size)
            batch_paths = [train_items[i][0] for i in idxs]
            batch_text = [train_items[i][1] for i in idxs]
            imgs_u8 = [load_gray_u8(p) for p in batch_paths]
            x = pad_batch_images(imgs_u8, cfg.image_height, cfg.max_width)
            if cfg.apply_extra_train_noise:
                x = m.apply_gaussian_noise_np(x, cfg.train_gaussian_std, rng)
                x = m.apply_salt_pepper_noise_np(x, cfg.train_salt_prob, cfg.train_pepper_prob, rng)
            if sidd_cache and cfg.train_sidd_strength > 0:
                x = m.apply_sidd_patch_blend(x, sidd_cache, rng, strength=cfg.train_sidd_strength)
            labels = [encode_label(t, vocab) for t in batch_text]
            y_dense, y_len = labels_to_dense_batch(labels)
            if int(np.max(y_len)) == 0:
                continue
            loss = train_step(tf.constant(x), tf.constant(y_dense), tf.constant(y_len))
            losses.append(float(loss.numpy()))
        print(f"[office] epoch {epoch+1}/{cfg.epochs} mean_loss={float(np.mean(losses)):.4f}")

    # --- eval vs Tesseract(val) reference + export strings for Stage 2 ---
    ne = OfficeNoiseEval()
    rng_g = np.random.default_rng(ne.seed)
    rng_s = np.random.default_rng(ne.seed + 1)
    rng_sd = np.random.default_rng(ne.seed + 2)

    def eval_export_val(items: List[Tuple[Path, str, bool]], tag: str) -> None:
        if not items:
            print(f"[office] val[{tag}] empty")
            return
        stats = {
            "clean": [0, 0],
            "gaussian": [0, 0],
            "snp": [0, 0],
            "norm_clean": [0, 0],
            "norm_gaussian": [0, 0],
            "norm_snp": [0, 0],
        }
        use_sidd_eval = bool(sidd_eval_cache) and cfg.eval_sidd_strength > 0
        if use_sidd_eval:
            stats["sidd"] = [0, 0]
            stats["norm_sidd"] = [0, 0]

        export_rows: List[dict] = []
        for p, gt_csv, _is_noisy in items:
            pkey = str(p.resolve())
            gt_eval = _norm_text(val_tess_ref.get(pkey, gt_csv))
            im = load_gray_u8(p)
            x0 = pad_batch_images([im], cfg.image_height, cfg.max_width)
            bundles: List[Tuple[str, np.ndarray]] = [
                ("clean", x0),
                ("gaussian", m.apply_gaussian_noise_np(x0, ne.gaussian_std, rng_g)),
                ("snp", m.apply_salt_pepper_noise_np(x0, ne.salt_prob, ne.pepper_prob, rng_s)),
            ]
            if use_sidd_eval:
                bundles.append(
                    ("sidd", m.apply_sidd_patch_blend(x0, sidd_eval_cache, rng_sd, strength=cfg.eval_sidd_strength))
                )
            preds: Dict[str, str] = {}
            for name, xb in bundles:
                logits = model.predict(xb, verbose=0)
                dense = m.ctc_greedy_decode(tf.constant(logits, dtype=tf.float32)).numpy()[0]
                pred_ids = dense_row_to_indices(dense, blank_index)
                pred = pred_indices_to_text(pred_ids, vocab, blank_index)
                preds[name] = pred
                sk = name
                nk = "norm_" + name
                stats[sk][1] += 1
                stats[nk][1] += 1
                if pred == gt_eval:
                    stats[sk][0] += 1
                if _norm_text(pred).lower() == _norm_text(gt_eval).lower():
                    stats[nk][0] += 1

            export_rows.append(
                {
                    "image": p.name,
                    "image_path": pkey,
                    "reference_train_csv": gt_csv,
                    "reference_tesseract_validation": gt_eval,
                    "pred_text_clean": preds.get("clean", ""),
                    "pred_text_gaussian": preds.get("gaussian", ""),
                    "pred_text_salt_pepper": preds.get("snp", ""),
                    "pred_text_sidd_blend": preds.get("sidd", ""),
                    "stage2_suggested_payload": preds.get("clean", ""),
                }
            )

        print(f"\n=== Office CTC val ({tag}) vs Tesseract(val ref) — {len(items)} images ===")
        print(f"  Clean:               {stats['clean'][0]/max(stats['clean'][1],1):.4f}  (strict)")
        print(f"  Gaussian noise only: {stats['gaussian'][0]/max(stats['gaussian'][1],1):.4f}  (std={ne.gaussian_std})")
        print(f"  Salt-pepper only:    {stats['snp'][0]/max(stats['snp'][1],1):.4f}  (p={ne.salt_prob})")
        if use_sidd_eval:
            print(f"  SIDD patch blend:    {stats['sidd'][0]/max(stats['sidd'][1],1):.4f}  (strength={cfg.eval_sidd_strength})")
        print(f"  Clean (norm+lower):  {stats['norm_clean'][0]/max(stats['norm_clean'][1],1):.4f}")
        print(f"  Gaussian (norm):     {stats['norm_gaussian'][0]/max(stats['norm_gaussian'][1],1):.4f}")
        print(f"  Salt-pepper (norm):  {stats['norm_snp'][0]/max(stats['norm_snp'][1],1):.4f}")
        if use_sidd_eval:
            print(f"  SIDD (norm):         {stats['norm_sidd'][0]/max(stats['norm_sidd'][1],1):.4f}")
        print("================================================================")
        print(f"[office] Metrics use **Tesseract re-read** of each val image as reference (not the training CSV).")

        STAGE2_EXPORT_JSONL.parent.mkdir(parents=True, exist_ok=True)
        with open(STAGE2_EXPORT_JSONL, "w", encoding="utf-8") as f:
            for row in export_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[office] Stage-2 handoff (pred texts per noise) -> {STAGE2_EXPORT_JSONL}")

    eval_export_val(val_items, "VA holdout")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_OUT))
    meta = {
        "vocab": vocab,
        "blank_index": blank_index,
        "image_height": cfg.image_height,
        "max_width": cfg.max_width,
        "halve_width_twice": cfg.halve_width_twice,
        "model_path": str(MODEL_OUT.resolve()),
    }
    with open(VOCAB_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[office] Saved model -> {MODEL_OUT}")
    print(f"[office] Saved vocab  -> {VOCAB_JSON}")


def decode_image_to_text(image_u8: np.ndarray, model: tf.keras.Model, meta: dict) -> str:
    h, w = int(meta["image_height"]), int(meta["max_width"])
    vocab = str(meta["vocab"])
    blank_index = int(meta["blank_index"])
    x = np.asarray(image_u8)
    if x.ndim == 2:
        x = x[..., None]
    if x.ndim == 3:
        x = x[None, ...]
    x = pad_batch_images([x[0]], h, w)
    logits = model.predict(x, verbose=0)
    dense = m.ctc_greedy_decode(tf.constant(logits, dtype=tf.float32)).numpy()[0]
    ids = dense_row_to_indices(dense, blank_index)
    return pred_indices_to_text(ids, vocab, blank_index)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train", "decode_one"], default="train", nargs="?")
    p.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Do not auto-create office_transcripts.csv with Tesseract when missing.",
    )
    p.add_argument("--force-transcripts", action="store_true", help="Regenerate training CSV with Tesseract.")
    p.add_argument(
        "--rebuild-val-tesseract",
        action="store_true",
        help="Regenerate office_tesseract_val_reference.csv for validation metrics.",
    )
    p.add_argument("--sidd-dir", default="", help="Optional SIDD folder for *training* blend.")
    p.add_argument("--sidd-strength", type=float, default=0.0)
    p.add_argument("--sidd-eval-dir", default="", help="SIDD folder for *eval* four-way (defaults to --sidd-dir).")
    p.add_argument("--eval-sidd-strength", type=float, default=0.35)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--max-width", type=int, default=0, help="Input canvas width (0 = config default).")
    p.add_argument("--image-height", type=int, default=0, help="Input height (0 = config default).")
    p.add_argument(
        "--legacy-double-width-pool",
        action="store_true",
        help="Use two 2×2 pools on width (MNIST-style, fewer CTC frames; needs wider canvas for long text).",
    )
    p.add_argument("--image", default="", help="For decode_one: path to PNG.")
    args = p.parse_args()

    if args.cmd == "train":
        cfg = OfficeTrainConfig(
            epochs=args.epochs,
            steps_per_epoch=args.steps,
            max_width=args.max_width or OfficeTrainConfig.max_width,
            image_height=args.image_height or OfficeTrainConfig.image_height,
            halve_width_twice=bool(args.legacy_double_width_pool),
            sidd_dir=args.sidd_dir,
            train_sidd_strength=float(args.sidd_strength),
            sidd_eval_dir=args.sidd_eval_dir,
            eval_sidd_strength=float(args.eval_sidd_strength),
            bootstrap_tesseract=(not args.no_bootstrap) or bool(args.force_transcripts),
            force_transcripts=bool(args.force_transcripts),
            rebuild_val_tesseract=bool(args.rebuild_val_tesseract),
        )
        train_office_ctc(cfg)
        return

    if args.cmd == "decode_one":
        if not args.image or not VOCAB_JSON.is_file() or not MODEL_OUT.is_file():
            raise SystemExit("Need --image and trained artifacts/office_line_* files.")
        meta = json.loads(VOCAB_JSON.read_text(encoding="utf-8"))
        model = tf.keras.models.load_model(str(MODEL_OUT), compile=False)
        im = load_gray_u8(Path(args.image))
        txt = decode_image_to_text(im, model, meta)
        print(txt)
        return


if __name__ == "__main__":
    main()

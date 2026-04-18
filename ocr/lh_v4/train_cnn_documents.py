#!/usr/bin/env python3
"""
OCR training/serving script using only:
Voxel51/scanned-images-dataset-for-ocr-and-vlm-finetuning

Architecture:
[Input Image]
  -> CNN Denoiser / Enhancement Network
  -> CNN Feature Extractor (ResNet)
  -> BiLSTM
  -> CTC (or CTC + Attention)
  -> Text Output

Important:
This script is model-only for OCR training/inference.
Tesseract is used only to generate transcripts for the validation stage.
"""

from __future__ import annotations

import argparse
import io
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import fiftyone as fo
import torch
import torch.nn as nn
import torch.nn.functional as F
from fiftyone.utils.huggingface import load_from_hub
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

DEFAULT_HUB = "Voxel51/scanned-images-dataset-for-ocr-and-vlm-finetuning"
DEFAULT_TRANSCRIPT_FIELD = "ground_truth"
DEFAULT_SPLIT_FIELD = "ground_truth"
PAD_IDX = 0
BLANK_IDX = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/serve OCR model on FiftyOne hub dataset")
    p.add_argument(
        "--mode",
        choices=["train", "serve", "predict", "extract", "eval-extract"],
        default="train",
    )
    p.add_argument("--hub-dataset", default=DEFAULT_HUB)
    p.add_argument("--transcript-field", default=DEFAULT_TRANSCRIPT_FIELD, help="Field containing transcript targets (text or Classification label)")
    p.add_argument("--validation-use-tesseract", action="store_true", default=True, help="Use Tesseract transcripts for validation labels")
    p.add_argument(
        "--target-level",
        choices=["full", "line", "word"],
        default="line",
        help="Target granularity: full page text, a line-like chunk, or first word.",
    )
    p.add_argument("--split-field", default=DEFAULT_SPLIT_FIELD, help="Field for stratified split (default ground_truth)")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--random-sample-size",
        type=int,
        default=200,
        help="After loading, use a random subset of at most this many samples (0 = use the full dataset).",
    )
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ctc-attn-alpha", type=float, default=0.0, help="Set >0 to enable CTC + Attention")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-height", type=int, default=64)
    p.add_argument("--image-width", type=int, default=256)
    p.add_argument(
        "--keep-original-downsampling",
        action="store_true",
        help="Use original ResNet width downsampling (shorter output sequences).",
    )
    p.add_argument("--output-dir", type=Path, default=Path("runs/stage1_ocr"))
    p.add_argument("--model-path", type=Path, default=Path("runs/stage1_ocr/best_stage1_ocr.pt"))
    p.add_argument(
        "--val-predictions-path",
        type=Path,
        default=Path("runs/stage1_ocr/val_predictions.jsonl"),
        help="Where to save validation ground-truth vs predicted text for comparison.",
    )
    p.add_argument("--device", default=None, help="cuda|mps|cpu")
    p.add_argument("--launch-app", action="store_true")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--image-path", type=Path, default=None)
    p.add_argument(
        "--extract-output-path",
        type=Path,
        default=Path("runs/stage1_ocr/extracted_texts.jsonl"),
        help="Output JSONL path for extracted OCR text in extract mode; input for eval-extract.",
    )
    p.add_argument(
        "--eval-reference-jsonl",
        type=Path,
        default=None,
        help="For eval-extract: JSONL with image_path and reference_text (or ground_truth_text) per line.",
    )
    return p.parse_args()


def pick_device(preferred: str | None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_text(s: str) -> str:
    return " ".join(s.strip().split())


def sample_text(sample: fo.Sample, transcript_field: str) -> str:
    value = sample[transcript_field]
    if value is None:
        return ""
    if hasattr(value, "label"):
        return normalize_text(str(value.label))
    return normalize_text(str(value))


def pseudo_text_from_image(path: str) -> str:
    try:
        import pytesseract  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Pseudo OCR requested, but pytesseract is unavailable. "
            "Install it and ensure the Tesseract binary exists."
        ) from exc
    text = pytesseract.image_to_string(Image.open(path).convert("L"))
    return normalize_text(text)


def load_tesseract_cache(cache_path: Path) -> Dict[str, str]:
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_tesseract_cache(cache_path: Path, cache: Dict[str, str]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def get_tesseract_text(path: str, cache: Dict[str, str]) -> str:
    if path in cache:
        return cache[path]
    txt = pseudo_text_from_image(path)
    cache[path] = txt
    return txt


def build_vocab(texts: Sequence[str]) -> Tuple[Dict[str, int], Dict[int, str], int]:
    chars = sorted({ch for t in texts for ch in t})
    char_to_idx = {ch: i + 1 for i, ch in enumerate(chars)}  # blank is 0
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    eos_idx = len(chars) + 1
    return char_to_idx, idx_to_char, eos_idx


def make_target_text(text: str, target_level: str) -> str:
    text = normalize_text(text)
    if not text:
        return ""
    if target_level == "word":
        return text.split(" ", 1)[0]
    if target_level == "line":
        # Approximate a line target from page-level OCR text.
        return " ".join(text.split()[:16])
    return text


def random_sample_fiftyone(
    dataset: fo.Dataset,
    n: int,
    seed: int,
) -> Union[fo.Dataset, fo.DatasetView]:
    """Return a view of up to `n` randomly chosen samples (fewer if the dataset is smaller)."""
    if n <= 0:
        return dataset
    ids = dataset.values("id")
    if len(ids) <= n:
        return dataset
    rng = random.Random(seed)
    chosen = rng.sample(ids, n)
    return dataset.select(chosen)


def stratified_split_ids(
    dataset: Union[fo.Dataset, fo.DatasetView],
    split_field: str,
    train_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    by_label: Dict[str, List[str]] = defaultdict(list)
    for s in dataset.iter_samples(progress=False):
        label = "__all__"
        try:
            value = s[split_field]
            label = str(value.label) if hasattr(value, "label") else str(value)
        except Exception:
            pass
        by_label[label].append(s.id)
    train_ids: List[str] = []
    val_ids: List[str] = []
    for ids in by_label.values():
        ids = list(ids)
        rng.shuffle(ids)
        cut = max(1, min(len(ids) - 1, int(round(len(ids) * train_ratio)))) if len(ids) > 1 else len(ids)
        train_ids.extend(ids[:cut])
        val_ids.extend(ids[cut:])
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    return train_ids, val_ids


class HubOCRDataset(Dataset):
    def __init__(
        self,
        view: fo.DatasetView,
        transcript_field: str,
        force_tesseract: bool,
        tesseract_cache: Dict[str, str],
        ctc_char_to_idx: Dict[str, int],
        eos_idx: int,
        target_level: str,
        transform: transforms.Compose,
    ) -> None:
        self.rows: List[Tuple[str, str]] = []
        self.ctc_char_to_idx = ctc_char_to_idx
        self.eos_idx = eos_idx
        self.transform = transform
        for s in view:
            try:
                text = sample_text(s, transcript_field)
            except (KeyError, AttributeError):
                text = ""
            if force_tesseract:
                text = get_tesseract_text(s.filepath, tesseract_cache)
            text = make_target_text(text, target_level)
            if text:
                self.rows.append((s.filepath, text))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path, text = self.rows[idx]
        image = Image.open(path).convert("L")
        x = self.transform(image)
        ctc_target = torch.tensor([self.ctc_char_to_idx[c] for c in text], dtype=torch.long)
        attn_target = torch.tensor([self.ctc_char_to_idx[c] for c in text] + [self.eos_idx], dtype=torch.long)
        return {"image": x, "ctc_target": ctc_target, "attn_target": attn_target, "text": text, "path": path}


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    ctc_targets = [b["ctc_target"] for b in batch]
    flat = torch.cat(ctc_targets, dim=0)
    lengths = torch.tensor([len(x) for x in ctc_targets], dtype=torch.long)
    attn = pad_sequence([b["attn_target"] for b in batch], batch_first=True, padding_value=PAD_IDX)
    texts = [b["text"] for b in batch]
    paths = [b["path"] for b in batch]
    return {"images": images, "targets_flat": flat, "target_lengths": lengths, "attn_targets": attn, "texts": texts, "paths": paths}


def make_transform(train: bool, h: int, w: int) -> transforms.Compose:
    ops: List[Any] = [transforms.Resize((h, w))]
    if train:
        ops.append(transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.98, 1.02)))
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


class DenoiserEnhancer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x + self.net(x), 0.0, 1.0)


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, reduce_width_downsampling: bool = True) -> None:
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if reduce_width_downsampling:
            # Keep strong vertical downsampling while preserving more horizontal timesteps.
            base.conv1.stride = (2, 1)
            base.maxpool.stride = (2, 1)
            base.layer3[0].conv1.stride = (2, 1)
            base.layer3[0].downsample[0].stride = (2, 1)
            base.layer4[0].conv1.stride = (2, 1)
            base.layer4[0].downsample[0].stride = (2, 1)
        self.features = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool, base.layer1, base.layer2, base.layer3, base.layer4)
        self.out_channels = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class AdditiveAttention(nn.Module):
    def __init__(self, enc_dim: int, dec_dim: int = 256, attn_dim: int = 256) -> None:
        super().__init__()
        self.w_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.w_dec = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, enc: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        score = self.v(torch.tanh(self.w_enc(enc) + self.w_dec(dec).unsqueeze(1))).squeeze(-1)
        attn = torch.softmax(score, dim=1)
        return torch.bmm(attn.unsqueeze(1), enc).squeeze(1)


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size: int, enc_dim: int, dec_dim: int = 256) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dec_dim, padding_idx=PAD_IDX)
        self.attn = AdditiveAttention(enc_dim=enc_dim, dec_dim=dec_dim)
        self.cell = nn.GRUCell(dec_dim + enc_dim, dec_dim)
        self.out = nn.Linear(dec_dim + enc_dim, vocab_size)
        self.dec_dim = dec_dim

    def forward(self, enc_out: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = targets.shape
        state = torch.zeros(bsz, self.dec_dim, device=targets.device)
        prev = torch.zeros(bsz, dtype=torch.long, device=targets.device)
        logits: List[torch.Tensor] = []
        for t in range(seq_len):
            emb = self.embed(prev)
            ctx = self.attn(enc_out, state)
            state = self.cell(torch.cat([emb, ctx], dim=1), state)
            step = self.out(torch.cat([state, ctx], dim=1))
            logits.append(step.unsqueeze(1))
            prev = targets[:, t]
        return torch.cat(logits, dim=1)


class OCRModel(nn.Module):
    def __init__(
        self,
        eos_idx: int,
        use_attention: bool = False,
        hidden: int = 256,
        reduce_width_downsampling: bool = True,
    ) -> None:
        super().__init__()
        # CTC uses blank (0) + character classes (1..eos_idx-1) only — never EOS.
        # Attention uses padding (0) + chars + EOS (eos_idx), so vocab = eos_idx + 1.
        self.eos_idx = eos_idx
        self.denoiser = DenoiserEnhancer()
        self.extractor = ResNetFeatureExtractor(reduce_width_downsampling=reduce_width_downsampling)
        self.project = nn.Linear(self.extractor.out_channels, hidden)
        self.bilstm = nn.LSTM(hidden, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
        self.ctc_head = nn.Linear(hidden * 2, eos_idx)
        self.use_attention = use_attention
        self.attn_decoder = AttentionDecoder(vocab_size=eos_idx + 1, enc_dim=hidden * 2) if use_attention else None

    def forward(self, x: torch.Tensor, attn_targets: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        x = self.denoiser(x)
        feat = self.extractor(x)
        seq = feat.mean(dim=2).transpose(1, 2)
        seq = self.project(seq)
        enc_out, _ = self.bilstm(seq)
        out = {"ctc_logits": self.ctc_head(enc_out)}
        if self.use_attention and self.attn_decoder is not None and attn_targets is not None:
            out["attn_logits"] = self.attn_decoder(enc_out, attn_targets)
        return out


def ctc_decode(log_probs_tbn: torch.Tensor, input_lengths: torch.Tensor, idx_to_char: Dict[int, str]) -> List[str]:
    pred = log_probs_tbn.argmax(dim=2).transpose(0, 1)
    texts: List[str] = []
    for b in range(pred.size(0)):
        prev = BLANK_IDX
        chars: List[str] = []
        for t in range(int(input_lengths[b].item())):
            token = int(pred[b, t].item())
            if token != BLANK_IDX and token != prev:
                chars.append(idx_to_char.get(token, ""))
            prev = token
        text = "".join(chars)
        if text.strip():
            texts.append(text)
            continue

        # Fallback: if greedy path is blank/whitespace-only, pick best non-blank
        # token per timestep to avoid empty outputs in comparisons.
        fallback_chars: List[str] = []
        step_scores = log_probs_tbn[: int(input_lengths[b].item()), b, :]
        prev_fb = BLANK_IDX
        for t in range(step_scores.size(0)):
            row = step_scores[t]
            top_idx = int(row.argmax().item())
            token = top_idx
            if token == BLANK_IDX and row.numel() > 1:
                # choose best non-blank class
                token = int(torch.argmax(row[1:]).item()) + 1
            if token != BLANK_IDX and token != prev_fb:
                ch = idx_to_char.get(token, "")
                if ch:
                    fallback_chars.append(ch)
            prev_fb = token
        fallback_text = "".join(fallback_chars).strip()
        texts.append(fallback_text)
    return texts


def char_edit_distance(ref: str, hyp: str) -> int:
    """Levenshtein distance at character granularity (same dynamic program as CER)."""
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


def char_error_rate(ref: str, hyp: str) -> float:
    m, n = len(ref), len(hyp)
    if m == 0:
        return 0.0 if n == 0 else 1.0
    return char_edit_distance(ref, hyp) / m


def compute_ctc_loss(ctc_fn: nn.Module, log_probs: torch.Tensor, targets: torch.Tensor, input_lengths: torch.Tensor, target_lengths: torch.Tensor, device: torch.device) -> torch.Tensor:
    if device.type == "mps":
        return ctc_fn(log_probs.cpu(), targets.cpu(), input_lengths.cpu(), target_lengths.cpu()).to(device)
    return ctc_fn(log_probs, targets, input_lengths, target_lengths)


def run_eval(
    model: OCRModel,
    loader: DataLoader,
    device: torch.device,
    ctc_attn_alpha: float,
    idx_to_char: Dict[int, str],
) -> Tuple[float, float, List[Dict[str, Any]]]:
    model.eval()
    ctc_fn = nn.CTCLoss(blank=BLANK_IDX, reduction="mean", zero_infinity=True)
    attn_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    comparisons: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["images"].to(device)
            y = batch["targets_flat"].to(device)
            y_len = batch["target_lengths"].to(device)
            attn_targets = batch["attn_targets"].to(device)
            out = model(x, attn_targets if ctc_attn_alpha > 0 else None)
            log_probs = F.log_softmax(out["ctc_logits"], dim=-1).transpose(0, 1)
            in_len = torch.full((x.size(0),), fill_value=log_probs.size(0), dtype=torch.long, device=device)
            loss = compute_ctc_loss(ctc_fn, log_probs, y, in_len, y_len, device)
            if ctc_attn_alpha > 0 and "attn_logits" in out:
                attn = out["attn_logits"]
                loss = loss + ctc_attn_alpha * attn_fn(attn.reshape(-1, attn.size(-1)), attn_targets.reshape(-1))
            preds = ctc_decode(log_probs, in_len, idx_to_char)
            for p, r, path in zip(preds, batch["texts"], batch["paths"]):
                cer = char_error_rate(r, p)
                total_acc += max(0.0, 1.0 - cer)
                comparisons.append(
                    {
                        "ground_truth_text": r,
                        "predicted_text": p,
                        "char_error_rate": cer,
                    }
                )
                n += 1
            total_loss += float(loss.item())
    return total_loss / max(1, len(loader)), total_acc / max(1, n), comparisons


def save_predictions_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_texts(args: argparse.Namespace) -> None:
    dataset = load_dataset_from_hub(args)
    dataset = random_sample_fiftyone(dataset, args.random_sample_size, args.seed)
    device = pick_device(args.device)
    model, idx_to_char, cfg = load_model_for_inference(args.model_path, device)
    h = int(cfg.get("image_height", args.image_height))
    w = int(cfg.get("image_width", args.image_width))
    rows: List[Dict[str, Any]] = []

    for s in dataset.iter_samples(progress=False):
        image = Image.open(s.filepath)
        text = predict_text(model, image, device, idx_to_char, h, w)
        rows.append(
            {
                "sample_id": s.id,
                "image_path": s.filepath,
                "extracted_text": text,
            }
        )

    save_predictions_jsonl(args.extract_output_path, rows)
    print(f"Extracted OCR text for {len(rows)} images using model inference")
    print(f"Saved extraction output to: {args.extract_output_path}")


def load_dataset_from_hub(args: argparse.Namespace) -> fo.Dataset:
    kwargs: Dict[str, Any] = {}
    if args.max_samples is not None:
        kwargs["max_samples"] = args.max_samples
    # Use a filesystem-safe local FiftyOne dataset name instead of raw hub ID.
    local_name = args.hub_dataset.replace("/", "__")
    kwargs["name"] = local_name
    if fo.dataset_exists(local_name):
        dataset = fo.load_dataset(local_name)
        if args.max_samples is not None:
            ids = dataset.values("id")
            dataset = dataset.select(ids[: args.max_samples])
        if args.launch_app:
            session = fo.launch_app(dataset)
            session.wait()
        return dataset
    dataset = load_from_hub(args.hub_dataset, **kwargs)
    if args.launch_app:
        session = fo.launch_app(dataset)
        session.wait()
    return dataset


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    dataset = load_dataset_from_hub(args)
    dataset = random_sample_fiftyone(dataset, args.random_sample_size, args.seed)
    cache_path = args.output_dir / "tesseract_cache.json"
    tesseract_cache = load_tesseract_cache(cache_path)

    has_transcript_field = args.transcript_field in dataset.get_field_schema()
    if not has_transcript_field:
        raise SystemExit(
            f"Transcript field '{args.transcript_field}' not found. "
            "Provide a valid transcript/label field for model-only training."
        )

    train_ids, val_ids = stratified_split_ids(
        dataset,
        args.split_field,
        args.train_ratio,
        args.seed,
    )
    train_view = dataset.select(train_ids)
    val_view = dataset.select(val_ids)

    # Build vocabulary from texts actually used by train/val targets.
    texts: List[str] = []
    for s in train_view.iter_samples(progress=False):
        text = ""
        if has_transcript_field:
            text = make_target_text(sample_text(s, args.transcript_field), args.target_level)
        if text:
            texts.append(text)
    if args.validation_use_tesseract:
        for s in val_view.iter_samples(progress=False):
            txt = make_target_text(get_tesseract_text(s.filepath, tesseract_cache), args.target_level)
            if txt:
                texts.append(txt)
    elif has_transcript_field:
        for s in val_view.iter_samples(progress=False):
            try:
                txt = make_target_text(sample_text(s, args.transcript_field), args.target_level)
            except Exception:
                txt = ""
            if txt:
                texts.append(txt)

    texts = [t for t in texts if t]
    if not texts:
        raise SystemExit("No transcript text found in training targets. Provide transcript field values.")
    char_to_idx, idx_to_char, eos_idx = build_vocab(texts)

    train_ds = HubOCRDataset(
        train_view,
        args.transcript_field,
        False,
        tesseract_cache,
        char_to_idx,
        eos_idx,
        args.target_level,
        make_transform(True, args.image_height, args.image_width),
    )
    val_ds = HubOCRDataset(
        val_view,
        args.transcript_field,
        args.validation_use_tesseract,
        tesseract_cache,
        char_to_idx,
        eos_idx,
        args.target_level,
        make_transform(False, args.image_height, args.image_width),
    )
    save_tesseract_cache(cache_path, tesseract_cache)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_batch)

    model = OCRModel(
        eos_idx=eos_idx,
        use_attention=args.ctc_attn_alpha > 0,
        reduce_width_downsampling=not args.keep_original_downsampling,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ctc_fn = nn.CTCLoss(blank=BLANK_IDX, reduction="mean", zero_infinity=True)
    attn_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            x = batch["images"].to(device)
            y = batch["targets_flat"].to(device)
            y_len = batch["target_lengths"].to(device)
            attn_targets = batch["attn_targets"].to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x, attn_targets if args.ctc_attn_alpha > 0 else None)
            log_probs = F.log_softmax(out["ctc_logits"], dim=-1).transpose(0, 1)
            in_len = torch.full((x.size(0),), fill_value=log_probs.size(0), dtype=torch.long, device=device)
            loss = compute_ctc_loss(ctc_fn, log_probs, y, in_len, y_len, device)
            if args.ctc_attn_alpha > 0 and "attn_logits" in out:
                attn = out["attn_logits"]
                loss = loss + args.ctc_attn_alpha * attn_fn(attn.reshape(-1, attn.size(-1)), attn_targets.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += float(loss.item())

        _, val_acc, val_comparisons = run_eval(
            model,
            val_loader,
            device,
            args.ctc_attn_alpha,
            idx_to_char,
        )
        print(f"Epoch {epoch}/{args.epochs} train_loss={running/max(1, len(train_loader)):.4f} val_char_acc={val_acc:.4f}")
        if val_acc > best:
            best = val_acc
            save_predictions_jsonl(args.val_predictions_path, val_comparisons)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "best_val_char_acc": best,
                    "config": vars(args),
                    "char_to_idx": char_to_idx,
                    "idx_to_char": idx_to_char,
                    "eos_idx": eos_idx,
                    "vocab_size": eos_idx + 1,
                },
                args.model_path,
            )
    print(f"Best validation char accuracy: {best:.4f}")
    print(f"Saved best model to: {args.model_path}")
    print(f"Saved validation comparisons to: {args.val_predictions_path}")


def load_model_for_inference(model_path: Path, device: torch.device) -> Tuple[OCRModel, Dict[int, str], Dict[str, Any]]:
    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt.get("config", {})
    eos_idx = ckpt.get("eos_idx")
    if eos_idx is None:
        eos_idx = int(ckpt["vocab_size"]) - 1
    model = OCRModel(
        eos_idx=int(eos_idx),
        use_attention=cfg.get("ctc_attn_alpha", 0.0) > 0,
        reduce_width_downsampling=not cfg.get("keep_original_downsampling", False),
    ).to(device)
    try:
        model.load_state_dict(ckpt["model_state"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint weights do not match the current model (CTC head size changed). "
            "Delete the old checkpoint and train again, or use a checkpoint saved after this fix."
        ) from exc
    model.eval()
    idx_to_char = {int(k): v for k, v in ckpt["idx_to_char"].items()}
    return model, idx_to_char, cfg


def preprocess_inference_image(image: Image.Image, image_h: int, image_w: int) -> torch.Tensor:
    return transforms.ToTensor()(image.convert("L").resize((image_w, image_h), Image.BILINEAR)).unsqueeze(0)


@torch.no_grad()
def predict_text(model: OCRModel, image: Image.Image, device: torch.device, idx_to_char: Dict[int, str], image_h: int, image_w: int) -> str:
    x = preprocess_inference_image(image, image_h, image_w).to(device)
    out = model(x)
    log_probs = F.log_softmax(out["ctc_logits"], dim=-1).transpose(0, 1)
    in_len = torch.tensor([log_probs.size(0)], dtype=torch.long, device=device)
    return ctc_decode(log_probs, in_len, idx_to_char)[0]


def serve(args: argparse.Namespace) -> None:
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import JSONResponse
    import uvicorn

    device = pick_device(args.device)
    model, idx_to_char, cfg = load_model_for_inference(args.model_path, device)
    h = int(cfg.get("image_height", args.image_height))
    w = int(cfg.get("image_width", args.image_width))
    app = FastAPI(title="Stage1 OCR Microservice")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "model_path": str(args.model_path)}

    @app.post("/ocr")
    async def ocr(file: UploadFile = File(...)) -> JSONResponse:
        image = Image.open(io.BytesIO(await file.read()))
        text = predict_text(model, image, device, idx_to_char, h, w)
        return JSONResponse({"text": text})

    uvicorn.run(app, host=args.host, port=args.port)


def predict_cli(args: argparse.Namespace) -> None:
    if args.image_path is None:
        raise SystemExit("--image-path is required for --mode predict")
    device = pick_device(args.device)
    model, idx_to_char, cfg = load_model_for_inference(args.model_path, device)
    h = int(cfg.get("image_height", args.image_height))
    w = int(cfg.get("image_width", args.image_width))
    text = predict_text(model, Image.open(args.image_path), device, idx_to_char, h, w)
    print(text)


def load_reference_map(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        row = json.loads(raw)
        ip = row.get("image_path")
        if not ip:
            continue
        ref = row.get("reference_text", row.get("ground_truth_text"))
        if ref is None:
            continue
        mapping[str(ip)] = normalize_text(str(ref))
    return mapping


def eval_extract(args: argparse.Namespace) -> None:
    """Character-level scores for extracted text vs reference (needs ground truth per image)."""
    extract_path = args.extract_output_path
    if not extract_path.is_file():
        raise SystemExit(f"eval-extract: file not found: {extract_path}")

    ref_by_path: Dict[str, str] = {}
    if args.eval_reference_jsonl is not None:
        if not args.eval_reference_jsonl.is_file():
            raise SystemExit(f"eval-extract: --eval-reference-jsonl not found: {args.eval_reference_jsonl}")
        ref_by_path = load_reference_map(args.eval_reference_jsonl)

    per_sample_acc: List[float] = []
    per_sample_cer: List[float] = []
    total_edits_nonempty_ref = 0
    total_ref_chars_nonempty = 0
    n_matched = 0
    n_missing_ref = 0

    for raw in extract_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        row = json.loads(raw)
        hyp = normalize_text(str(row.get("extracted_text", "")))
        ip = row.get("image_path")
        ref = row.get("reference_text", row.get("ground_truth_text"))
        if ref is None and ip is not None:
            ref = ref_by_path.get(str(ip))
        if ref is None:
            n_missing_ref += 1
            continue
        ref = normalize_text(str(ref))
        n_matched += 1
        cer = char_error_rate(ref, hyp)
        per_sample_cer.append(cer)
        per_sample_acc.append(max(0.0, 1.0 - cer))
        if len(ref) > 0:
            total_edits_nonempty_ref += char_edit_distance(ref, hyp)
            total_ref_chars_nonempty += len(ref)

    if n_matched == 0:
        raise SystemExit(
            "eval-extract: no samples with reference text. "
            "Add reference_text (or ground_truth_text) to each line of the extract JSONL, "
            "or pass --eval-reference-jsonl with matching image_path keys."
        )

    mean_char_acc = sum(per_sample_acc) / n_matched
    mean_cer = sum(per_sample_cer) / n_matched
    if total_ref_chars_nonempty > 0:
        corpus_cer = total_edits_nonempty_ref / total_ref_chars_nonempty
        corpus_char_acc = max(0.0, 1.0 - corpus_cer)
    else:
        corpus_cer = float("nan")
        corpus_char_acc = float("nan")

    print(f"eval-extract: matched={n_matched} missing_reference={n_missing_ref}")
    print(f"  mean character accuracy (1 - CER per sample, then mean): {mean_char_acc:.4f}")
    print(f"  mean CER (per sample): {mean_cer:.4f}")
    if total_ref_chars_nonempty > 0:
        print(f"  corpus character accuracy (1 - total_edits / total_ref_chars): {corpus_char_acc:.4f}")
        print(f"  corpus CER: {corpus_cer:.4f}")


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "serve":
        serve(args)
    elif args.mode == "predict":
        predict_cli(args)
    elif args.mode == "eval-extract":
        eval_extract(args)
    else:
        extract_texts(args)


if __name__ == "__main__":
    main()

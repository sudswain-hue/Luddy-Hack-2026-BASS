# ocr/run_pipeline.py
"""
End-to-end pipeline — single image, single folder, or multi-folder batch.

Usage:
  python run_pipeline.py                       # single image (default)
  python run_pipeline.py path/to/image.png     # specific image
  python run_pipeline.py --batch               # default noisy folder only
  python run_pipeline.py --all                 # all 4 dataset folders (288 images)
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import requests


OCR_URL     = "http://localhost:8000"
HUFFMAN_URL = "http://localhost:8001"

DEFAULT_FOLDERS = [
    "data/SimulatedNoisyOffice/simulated_noisy_images_grayscale",
    "data/SimulatedNoisyOffice/clean_images_grayscale",
    "data/SimulatedNoisyOffice/clean_images_grayscale_doubleresolution",
    "data/SimulatedNoisyOffice/clean_images_binaryscal (low resolution)",
]


def call_ocr(image_path: Path) -> tuple[str, float]:
    with open(image_path, 'rb') as f:
        files = {'file': (image_path.name, f, 'image/png')}
        t0 = time.perf_counter()
        r  = requests.post(f"{OCR_URL}/ocr/document", files=files)
        latency_ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return r.json()["text"], latency_ms


def call_compress(text: str) -> tuple[dict, float]:
    t0 = time.perf_counter()
    r  = requests.post(f"{HUFFMAN_URL}/compress", json={"text": text})
    latency_ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return r.json()["data"], latency_ms


def call_decompress(compressed_b64: str, padding: int) -> tuple[str, float]:
    t0 = time.perf_counter()
    r  = requests.post(f"{HUFFMAN_URL}/decompress", json={
        "compressed_bytes_b64": compressed_b64,
        "padding":              padding,
    })
    latency_ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return r.json()["recovered_text"], latency_ms


def process_image(image_path: Path, folder_label: str = "") -> dict:
    total_start = time.perf_counter()
    try:
        text, ocr_ms = call_ocr(image_path)
        if not text.strip():
            return {
                "image":    image_path.name,
                "folder":   folder_label,
                "status":   "skipped_empty_ocr",
            }

        comp, comp_ms = call_compress(text)
        recovered, decomp_ms = call_decompress(
            comp["compressed_bytes_b64"], comp["padding"]
        )
        lossless = recovered == text
        total_ms = (time.perf_counter() - total_start) * 1000

        return {
            "image":                 image_path.name,
            "folder":                folder_label,
            "status":                "success",
            "text_length":           len(text),
            "compressed_size":       comp["compressed_size_bytes"],
            "compression_ratio":     comp["metrics"]["compression_ratio"],
            "entropy":               comp["metrics"]["entropy"],
            "encoding_efficiency":   comp["metrics"]["encoding_efficiency"],
            "ocr_latency_ms":        round(ocr_ms, 2),
            "compress_latency_ms":   round(comp_ms, 2),
            "decompress_latency_ms": round(decomp_ms, 2),
            "total_latency_ms":      round(total_ms, 2),
            "lossless":              lossless,
        }
    except Exception as e:
        return {
            "image":   image_path.name,
            "folder":  folder_label,
            "status":  "error",
            "error":   str(e),
        }


def aggregate_metrics(results: list[dict], label: str) -> dict:
    success = [r for r in results if r["status"] == "success"]
    if not success:
        return {"label": label, "count": 0, "error": "no successful runs"}
    ratios      = [r["compression_ratio"]    for r in success]
    efficiency  = [r["encoding_efficiency"]  for r in success]
    total_lat   = [r["total_latency_ms"]     for r in success]
    ocr_lat     = [r["ocr_latency_ms"]       for r in success]
    comp_lat    = [r["compress_latency_ms"]  for r in success]
    decomp_lat  = [r["decompress_latency_ms"] for r in success]
    return {
        "label":                      label,
        "count":                      len(success),
        "total_attempted":            len(results),
        "avg_compression_ratio":      round(statistics.mean(ratios), 4),
        "min_compression_ratio":      round(min(ratios), 4),
        "max_compression_ratio":      round(max(ratios), 4),
        "avg_efficiency":             round(statistics.mean(efficiency), 4),
        "avg_total_latency_ms":       round(statistics.mean(total_lat), 2),
        "avg_ocr_latency_ms":         round(statistics.mean(ocr_lat), 2),
        "avg_compress_latency_ms":    round(statistics.mean(comp_lat), 2),
        "avg_decompress_latency_ms":  round(statistics.mean(decomp_lat), 2),
        "all_lossless":               all(r["lossless"] for r in success),
    }


def print_summary_row(summary: dict):
    if summary["count"] == 0:
        print(f"  {summary['label']:<55} NO SUCCESSFUL RUNS")
        return
    print(f"  {summary['label']:<55} "
          f"n={summary['count']:>3}  "
          f"ratio={summary['avg_compression_ratio']:.3f}x  "
          f"eff={summary['avg_efficiency']*100:.2f}%  "
          f"{summary['avg_total_latency_ms']:.0f}ms")


def run_single(image_path: Path):
    print(f"\n{'=' * 70}")
    print(f"  END-TO-END PIPELINE TEST  ·  {image_path.name}")
    print(f"{'=' * 70}\n")

    r = process_image(image_path)
    if r["status"] != "success":
        print(f"FAILED — {r.get('error', r['status'])}")
        return

    print(f"  Text extracted       : {r['text_length']} chars")
    print(f"  Compressed size      : {r['compressed_size']} bytes")
    print(f"  Compression ratio    : {r['compression_ratio']}x")
    print(f"  Entropy              : {r['entropy']} bits/char")
    print(f"  Encoding efficiency  : {r['encoding_efficiency']*100:.2f}%")
    print(f"  OCR latency          : {r['ocr_latency_ms']}ms")
    print(f"  Compress latency     : {r['compress_latency_ms']}ms")
    print(f"  Decompress latency   : {r['decompress_latency_ms']}ms")
    print(f"  Total latency        : {r['total_latency_ms']}ms")
    print(f"  Lossless             : {'YES ✓' if r['lossless'] else 'NO ✗'}")
    print()


def run_batch_folders(folders: list[Path], output_json: Path):
    all_results = []
    folder_summaries = []

    batch_start = time.perf_counter()

    for folder in folders:
        if not folder.is_dir():
            print(f"WARN: folder not found: {folder}")
            continue

        images = sorted(folder.glob("*.png"))
        if not images:
            print(f"WARN: no PNGs in {folder}")
            continue

        print(f"\n{'=' * 70}")
        print(f"  Processing folder: {folder.name}")
        print(f"  Images found: {len(images)}")
        print(f"{'=' * 70}")

        folder_results = []
        for img in images:
            r = process_image(img, folder_label=folder.name)
            folder_results.append(r)
            if r["status"] == "success":
                print(f"  {img.name:<40} "
                      f"chars={r['text_length']:>4}  "
                      f"ratio={r['compression_ratio']}x  "
                      f"{r['total_latency_ms']:.0f}ms")
            else:
                print(f"  {img.name:<40} SKIPPED ({r['status']})")

        all_results.extend(folder_results)
        folder_summaries.append(aggregate_metrics(folder_results, folder.name))

    batch_ms = (time.perf_counter() - batch_start) * 1000

    # overall aggregate
    overall = aggregate_metrics(all_results, "OVERALL (all folders)")

    # ── print everything ────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  PER-FOLDER SUMMARY")
    print(f"{'=' * 70}")
    for s in folder_summaries:
        print_summary_row(s)

    print(f"\n{'=' * 70}")
    print(f"  OVERALL SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total folders processed     : {len(folder_summaries)}")
    print(f"  Total images processed      : {overall['count']}/{overall['total_attempted']}")
    print(f"  Batch total runtime         : {batch_ms/1000:.1f}s")
    print(f"  Avg compression ratio       : {overall['avg_compression_ratio']}x")
    print(f"  Min/max compression ratio   : {overall['min_compression_ratio']}x / {overall['max_compression_ratio']}x")
    print(f"  Avg encoding efficiency     : {overall['avg_efficiency']*100:.2f}%")
    print(f"  Avg OCR latency             : {overall['avg_ocr_latency_ms']}ms")
    print(f"  Avg compress latency        : {overall['avg_compress_latency_ms']}ms")
    print(f"  Avg decompress latency      : {overall['avg_decompress_latency_ms']}ms")
    print(f"  Avg total latency           : {overall['avg_total_latency_ms']}ms")
    print(f"  All runs lossless?          : {'YES ✓' if overall['all_lossless'] else 'NO ✗'}")
    print(f"{'=' * 70}\n")

    # save report
    output_json.write_text(json.dumps({
        "overall":          overall,
        "per_folder":       folder_summaries,
        "per_image":        all_results,
        "batch_runtime_ms": batch_ms,
    }, indent=2))
    print(f"Full report saved to: {output_json}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", help="Specific image path (single mode)")
    parser.add_argument("--batch",  action="store_true", help="Run on default noisy folder")
    parser.add_argument("--all",    action="store_true", help="Run on ALL 4 dataset folders (~288 images)")
    parser.add_argument("--folder", default="data/SimulatedNoisyOffice/simulated_noisy_images_grayscale")
    parser.add_argument("--output", default="batch_results.json")
    args = parser.parse_args()

    if args.all:
        run_batch_folders([Path(f) for f in DEFAULT_FOLDERS], Path(args.output))
    elif args.batch:
        run_batch_folders([Path(args.folder)], Path(args.output))
    elif args.image:
        run_single(Path(args.image))
    else:
        # default: first image from noisy folder
        folder = Path(args.folder)
        images = sorted(folder.glob("*.png"))
        if images:
            run_single(images[0])
        else:
            print("No images found. Try --batch or --all")
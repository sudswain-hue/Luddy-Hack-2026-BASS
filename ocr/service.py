# ocr/service.py

"""
OCR Microservice — Stage 1 of the compression pipeline.

Exposes HTTP endpoints that accept images and return extracted text.
Two modes:
  - /ocr/digit   → CNN-based single character classification (MNIST)
  - /ocr/document → Tesseract-based full document OCR
  - /ocr          → auto-detect (uses document mode by default)

Model is loaded ONCE at startup, then reused across requests.
"""
import io
import time
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from pathlib import Path
import distutils as _distutils

from tesseract import extract_text_from_pil


# ── config ────────────────────────────────────────────────────────
MODEL_PATH  = Path(__file__).parent / "artifacts" / "character_cnn.keras"
APP_VERSION = "1.0.0"


# ── app + model ───────────────────────────────────────────────────
app = FastAPI(
    title="OCR Microservice (Stage 1)",
    description="CNN + Tesseract OCR for the 2-stage compression pipeline",
    version=APP_VERSION,
)

print(f"Loading CNN model from {MODEL_PATH}...")
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Run `python train.py` first."
    )
model = tf.keras.models.load_model(str(MODEL_PATH))
print(f"✓ Model loaded. Input shape: {model.input_shape}, "
      f"classes: {model.output_shape[-1]}")


# ── helpers ───────────────────────────────────────────────────────
def preprocess_for_cnn(pil_image: Image.Image) -> np.ndarray:
    """
    Resize to 28x28, grayscale, normalize to [0,1], add batch + channel dims.
    """
    img = pil_image.convert("L")             # grayscale
    img = img.resize((28, 28))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr[np.newaxis, ..., np.newaxis]   # (1, 28, 28, 1)
    return arr


# ── endpoints ─────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "service": "OCR Microservice (Stage 1)",
        "version": APP_VERSION,
        "endpoints": {
            "GET  /health":        "health check",
            "POST /ocr/digit":     "CNN classifies a single digit image",
            "POST /ocr/document":  "Tesseract extracts text from document image",
            "POST /ocr":           "Auto-route (defaults to document mode)",
        },
    }


@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "model_loaded": True,
        "model_path":   str(MODEL_PATH),
    }


@app.post("/ocr/digit")
async def ocr_digit(file: UploadFile = File(...)):
    """
    CNN-based single-digit classification.
    Expects a small image (28x28 ideally) containing a single digit.
    Returns predicted digit and confidence.
    """
    try:
        contents = await file.read()
        img      = Image.open(io.BytesIO(contents))

        t0 = time.perf_counter()
        arr = preprocess_for_cnn(img)
        predictions = model.predict(arr, verbose=0)
        latency_ms  = round((time.perf_counter() - t0) * 1000, 2)

        predicted_digit = int(np.argmax(predictions[0]))
        confidence      = float(predictions[0][predicted_digit])

        return {
            "mode":            "cnn_digit",
            "text":            str(predicted_digit),
            "predicted_digit": predicted_digit,
            "confidence":      round(confidence, 4),
            "latency_ms":      latency_ms,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")


@app.post("/ocr/document")
async def ocr_document(file: UploadFile = File(...)):
    """
    Tesseract-based full-document OCR.
    Accepts any size image containing printed text.
    Returns extracted text.
    """
    try:
        contents = await file.read()
        img      = Image.open(io.BytesIO(contents))

        t0 = time.perf_counter()
        text = extract_text_from_pil(img)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "mode":         "tesseract_document",
            "text":         text,
            "char_count":   len(text),
            "latency_ms":   latency_ms,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")


@app.post("/ocr")
async def ocr_auto(file: UploadFile = File(...)):
    """
    Default OCR endpoint — routes to document mode.
    Most batch pipeline usage goes here.
    """
    return await ocr_document(file)
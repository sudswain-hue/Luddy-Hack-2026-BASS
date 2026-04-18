# huffman/service.py
"""
Stage 2 — Adaptive Huffman Compression Microservice
Exposes two endpoints for the pipeline:
  POST /compress    - compress text using adaptive Huffman
  POST /decompress  - recover the original text (lossless)
"""
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pipeline import huffman_compress, huffman_decompress
from cache_manager import CompressionCache

app   = FastAPI(
            title       = "Stage 2 - Adaptive Huffman Compression",
            description = "Lossless compression microservice for OCR text output",
            version     = "1.0.0"
        )
cache = CompressionCache()


# ── Request models ─────────────────────────────────────────────────

class CompressRequest(BaseModel):
    text: str

class DecompressRequest(BaseModel):
    compressed_bytes_b64: str     # base64-encoded bytes over JSON
    padding: int                  # trailing bits added during encoding


# ── Endpoints ──────────────────────────────────────────────────────

@app.get("/")
def root():
    """Service info."""
    return {
        "service":   "Stage 2 - Adaptive Huffman Compression",
        "status":    "running",
        "endpoints": ["/compress", "/decompress", "/health", "/cache/stats"]
    }


@app.get("/health")
def health():
    """Health check for monitoring."""
    return {"status": "ok", "cache_size": cache.size()}


@app.get("/cache/stats")
def cache_stats():
    """Return cache metrics."""
    return {"cached_entries": cache.size()}


@app.post("/compress")
async def compress_text(request: CompressRequest):
    """
    Compress text using adaptive Huffman encoding.
    Returns base64-encoded compressed bytes + padding + metrics.
    """
    text = request.text

    if not text:
        raise HTTPException(status_code=400,
                            detail="Input text cannot be empty")

    # check cache first
    cached = cache.get(text)
    if cached is not None:
        return {"status": "success", "source": "cache", "data": cached}

    try:
        # run compression pipeline
        result = huffman_compress(text)

        # verify lossless round-trip before returning
        recovered = huffman_decompress(result["compressed_bytes"],
                                       result["padding"])
        lossless  = recovered == text

        # base64-encode bytes for JSON transport
        compressed_b64 = base64.b64encode(
                             result["compressed_bytes"]).decode('utf-8')

        response = {
            "lossless_verified":   lossless,
            "original_size_chars": len(text),
            "original_size_bytes": len(text.encode('utf-8')),
            "compressed_size_bytes": len(result["compressed_bytes"]),
            "compressed_bytes_b64": compressed_b64,
            "padding":              result["padding"],
            "metrics": {
                "compression_ratio":   result["metrics"]["compression_ratio"],
                "entropy":             result["metrics"]["entropy"],
                "encoding_efficiency": result["metrics"]["encoding_efficiency"],
                "encoding_efficiency_percent":
                    round(result["metrics"]["encoding_efficiency"] * 100, 2)
            }
        }

        cache.set(text, response)
        return {"status": "success", "source": "computed", "data": response}

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Compression failed: {str(e)}")


@app.post("/decompress")
async def decompress_bytes(request: DecompressRequest):
    """
    Decompress base64-encoded bytes back to the original text.
    """
    try:
        compressed_bytes = base64.b64decode(request.compressed_bytes_b64)
        recovered        = huffman_decompress(compressed_bytes,
                                              request.padding)

        return {
            "status":            "success",
            "recovered_text":    recovered,
            "recovered_length":  len(recovered)
        }

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Decompression failed: {str(e)}")


# allow running directly: python service.py
if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
    
    
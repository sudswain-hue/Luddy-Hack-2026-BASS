# Stage 2 — Adaptive Huffman Compression Microservice

Graduate hackathon submission for Luddy Hack 2026 (Indiana University).
Lossless compression microservice built using the FGK adaptive Huffman algorithm, implemented from scratch with no external compression libraries.

## Requirements

- Python 3.9 or higher
- macOS / Linux / Windows

## Setup

Clone the repository:

```bash
git clone <your-repo-url>
cd Luddy-Hack-2026-BASS
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate      # Mac / Linux
.venv\Scripts\activate         # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the compression microservice

```bash
cd huffman
uvicorn service:app --reload --port 8001
```

Or run directly with a custom port:

```bash
python service.py               # default port 8001
PORT=9000 python service.py     # custom port
```

Open the interactive API docs at: **http://localhost:8001/docs**

## Run the tests

From the `huffman/` folder (with the venv active):

```bash
python test_huffman.py         # full core test suite (33 tests)
python test_large.py           # large text compression (4,700+ chars)
python test_cache_real.py      # cache hit vs miss performance
```

## API Endpoints

### POST /compress

Compress text using adaptive Huffman encoding.

Request:
```json
{ "text": "your text here" }
```

Response:
```json
{
  "status": "success",
  "source": "computed",
  "data": {
    "lossless_verified": true,
    "original_size_chars": 58,
    "original_size_bytes": 58,
    "compressed_size_bytes": 42,
    "compressed_bytes_b64": "...",
    "padding": 2,
    "metrics": {
      "compression_ratio": 1.38,
      "entropy": 4.94,
      "encoding_efficiency": 0.98,
      "encoding_efficiency_percent": 98.53
    }
  }
}
```

### POST /decompress

Recover the original text from compressed bytes.

Request:
```json
{
  "compressed_bytes_b64": "base64 string from /compress",
  "padding": 2
}
```

Response:
```json
{
  "status": "success",
  "recovered_text": "your original text",
  "recovered_length": 58
}
```

### GET /health

Service health check. Returns status and current cache size.

### GET /cache/stats

Returns number of cached entries.

## Project Structure
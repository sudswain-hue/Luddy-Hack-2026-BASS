# huffman/test_step6.py
from pipeline import huffman_compress, huffman_decompress

# ── test 1: basic compress and decompress ─────────────────────────
result = huffman_compress("hello world")
assert "compressed_bytes" in result
assert "padding"          in result
assert "metrics"          in result
print("TEST 1 PASSED — compress returns correct keys")

# ── test 2: lossless round-trip ───────────────────────────────────
texts = [
    "hello",
    "aaabbb",
    "the quick brown fox",
    "Invoice 4821 Total 349",
    "Hello! How are you?",
]
for text in texts:
    result    = huffman_compress(text)
    recovered = huffman_decompress(
                    result["compressed_bytes"],
                    result["padding"])
    assert recovered == text, f"FAILED on {text!r}, got {recovered!r}"
print("TEST 2 PASSED — all round-trips lossless")

# ── test 3: metrics are all present and valid numbers ─────────────
result  = huffman_compress("adaptive huffman test")
metrics = result["metrics"]
assert isinstance(metrics["compression_ratio"],   float)
assert isinstance(metrics["entropy"],             float)
assert isinstance(metrics["encoding_efficiency"], float)
assert metrics["entropy"]             >= 0.0
assert metrics["encoding_efficiency"] >  0.0
print(f"TEST 3 PASSED — metrics valid")
print(f"  compression_ratio   = {metrics['compression_ratio']}")
print(f"  entropy             = {metrics['entropy']}")
print(f"  encoding_efficiency = {metrics['encoding_efficiency']}")

# ── test 4: compressed bytes are actually bytes ───────────────────
result = huffman_compress("test")
assert isinstance(result["compressed_bytes"], bytes)
assert isinstance(result["padding"], int)
print("TEST 4 PASSED — types correct")

# ── test 5: this is what Person D will literally call ─────────────
print("\n--- Person D usage demo ---")
ocr_output = "Total Amount Due 349 99 USD"
result     = huffman_compress(ocr_output)
recovered  = huffman_decompress(result["compressed_bytes"],
                                result["padding"])
print(f"Original  : {ocr_output}")
print(f"Compressed: {len(result['compressed_bytes'])} bytes "
      f"(padding={result['padding']})")
print(f"Recovered : {recovered}")
print(f"Lossless  : {recovered == ocr_output}")
print(f"Metrics   : {result['metrics']}")

print("\nALL STEP 6 TESTS PASSED — pipeline.py ready for Person D")
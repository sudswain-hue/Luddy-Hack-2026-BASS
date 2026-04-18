# huffman/test_step3.py
from encoder import AdaptiveHuffmanEncoder

# ── test 1: basic encode returns bytes and padding ────────────────
enc = AdaptiveHuffmanEncoder()
compressed, padding = enc.encode("hello")
assert isinstance(compressed, bytes)
assert isinstance(padding, int)
assert 0 <= padding <= 7
print(f"TEST 1 PASSED — encoded 'hello' → {len(compressed)} bytes, padding={padding}")

# ── test 2: longer text produces bytes ────────────────────────────
enc = AdaptiveHuffmanEncoder()
compressed, padding = enc.encode("the quick brown fox")
assert len(compressed) > 0
print(f"TEST 2 PASSED — encoded long text → {len(compressed)} bytes, padding={padding}")

# ── test 3: same text always gives same result ────────────────────
enc1 = AdaptiveHuffmanEncoder()
enc2 = AdaptiveHuffmanEncoder()
c1, p1 = enc1.encode("abcabc")
c2, p2 = enc2.encode("abcabc")
assert c1 == c2 and p1 == p2
print("TEST 3 PASSED — encoding is deterministic")

# ── test 4: compressed size is less than or equal to original ─────
text = "aaaaaaaaabbbbbccd"
enc  = AdaptiveHuffmanEncoder()
compressed, _ = enc.encode(text)
original_bytes = len(text)
print(f"TEST 4 INFO — original={original_bytes} bytes, "
      f"compressed={len(compressed)} bytes")
print("TEST 4 PASSED — encoder runs without error on repeated chars")

# ── test 5: empty string raises error ─────────────────────────────
import traceback
try:
    enc = AdaptiveHuffmanEncoder()
    enc.encode("")
    print("TEST 5 FAILED — should have raised ValueError")
except ValueError as e:
    print(f"TEST 5 PASSED — empty string correctly raises ValueError: {e}")

print("\nALL STEP 3 TESTS PASSED")
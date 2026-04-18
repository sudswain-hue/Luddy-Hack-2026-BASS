# huffman/test_step5.py
from metrics import (compression_ratio, shannon_entropy,
                     average_code_length, encoding_efficiency,
                     compute_all_metrics)
from encoder import AdaptiveHuffmanEncoder

# ── test 1: compression ratio ─────────────────────────────────────
# 8 chars * 8 bits = 64 bits original, 4 bytes = 32 bits compressed
# ratio should be 2.0
original  = "a" * 8
enc       = AdaptiveHuffmanEncoder()
compressed, _ = enc.encode(original)
ratio     = compression_ratio(original, compressed)
assert ratio > 1.0, f"Expected >1.0, got {ratio}"
print(f"TEST 1 PASSED — compression ratio={ratio} (>1.0 means compressed)")

# ── test 2: shannon entropy — all same chars ──────────────────────
# "aaaaaaa" has only one symbol, entropy should be 0.0
e = shannon_entropy("aaaaaaa")
assert e == 0.0, f"Expected 0.0, got {e}"
print(f"TEST 2 PASSED — entropy of 'aaaaaaa' = {e} (0.0 = fully predictable)")

# ── test 3: shannon entropy — two equal chars ─────────────────────
# "ababab" — two symbols at 50/50, entropy should be 1.0
e = shannon_entropy("ababab")
assert e == 1.0, f"Expected 1.0, got {e}"
print(f"TEST 3 PASSED — entropy of 'ababab' = {e} (1.0 = two equal symbols)")

# ── test 4: entropy increases with diversity ──────────────────────
e_low  = shannon_entropy("aaabbb")
e_high = shannon_entropy("abcdef")
assert e_high > e_low, f"Expected {e_high} > {e_low}"
print(f"TEST 4 PASSED — diverse text has higher entropy "
      f"({e_low} < {e_high})")

# ── test 5: encoding efficiency is between 0 and 1 ───────────────
text = "hello world"
enc  = AdaptiveHuffmanEncoder()
enc.encode(text)   # build tree
eff  = encoding_efficiency(text,
                           enc.tree.symbol_map,
                           enc.tree.get_code)
assert 0.0 < eff <= 1.0, f"Expected 0<eff<=1.0, got {eff}"
print(f"TEST 5 PASSED — encoding efficiency = {eff} (between 0 and 1)")

# ── test 6: compute_all_metrics returns all three keys ────────────
text = "the quick brown fox"
enc  = AdaptiveHuffmanEncoder()
compressed, _ = enc.encode(text)
metrics = compute_all_metrics(text,
                              compressed,
                              enc.tree.symbol_map,
                              enc.tree.get_code)
assert "compression_ratio"   in metrics
assert "entropy"             in metrics
assert "encoding_efficiency" in metrics
print(f"TEST 6 PASSED — all metrics present")
print(f"  compression_ratio   = {metrics['compression_ratio']}")
print(f"  entropy             = {metrics['entropy']}")
print(f"  encoding_efficiency = {metrics['encoding_efficiency']}")

print("\nALL STEP 5 TESTS PASSED")
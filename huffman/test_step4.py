# huffman/test_step4.py
from encoder import AdaptiveHuffmanEncoder
from decoder import AdaptiveHuffmanDecoder

def roundtrip(text):
    enc = AdaptiveHuffmanEncoder()
    dec = AdaptiveHuffmanDecoder()
    compressed, padding = enc.encode(text)
    recovered = dec.decode(compressed, padding)
    return recovered, compressed, padding

tests = [
    "hello",
    "aaabbb",
    "a",
    "ab",
    "aaaaa",
    "Hi 123",
    "fox jumps",
    "Total 349",
    "the quick brown fox",
]

all_passed = True
for i, text in enumerate(tests, 1):
    recovered, compressed, _ = roundtrip(text)
    ok     = recovered == text
    status = "PASS" if ok else "FAIL"
    print(f"TEST {i} {status} | '{text}' "
          f"| {len(text)}→{len(compressed)} bytes")
    if not ok:
        all_passed = False
        print(f"  EXPECTED: {text!r}")
        print(f"  GOT:      {recovered!r}")

print()
print("ALL STEP 4 TESTS PASSED" if all_passed else "SOME TESTS FAILED")
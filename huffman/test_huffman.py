# huffman/test_huffman.py
"""
Complete test suite for the Adaptive Huffman Compression Pipeline.
Graduate requirement verification included.
Run: python test_huffman.py
"""
from pipeline import huffman_compress, huffman_decompress

PASS = 0
FAIL = 0

def check(label, condition, info=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS — {label}")
    else:
        FAIL += 1
        print(f"  FAIL — {label} {info}")

def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")

# ══════════════════════════════════════════════════════
section("1. LOSSLESS ROUND-TRIP — core correctness")
# ══════════════════════════════════════════════════════

cases = [
    ("single char",        "a"),
    ("two chars",          "ab"),
    ("repeated chars",     "aaabbbccc"),
    ("all same",           "aaaaaaaaaa"),
    ("simple word",        "hello"),
    ("mixed alphanumeric", "Hello 123"),
    ("with punctuation",   "Hi! How are you?"),
    ("short OCR text",     "Invoice 4821 Total 349"),
    ("numbers only",       "0123456789"),
    ("spaces and symbols", "foo bar_baz"),
]

for label, text in cases:
    result    = huffman_compress(text)
    recovered = huffman_decompress(result["compressed_bytes"],
                                   result["padding"])
    check(label, recovered == text,
          f"— expected {text!r} got {recovered!r}")

# ══════════════════════════════════════════════════════
section("2. COMPRESSION RATIO — works on longer text")
# ══════════════════════════════════════════════════════

long_text = (
    "UNIVERSITY HOSPITAL SYSTEM PATIENT DISCHARGE SUMMARY "
    "Patient Name Jonathan Alexander Worthington III "
    "Patient ID PAT-2024-88421-XYZ Date of Birth March 14 1978 "
    "Admission Date January 10 2024 Discharge Date January 17 2024 "
    "Ward Cardiology Room 412B Attending Physician Dr Priya Subramaniam "
    "Blood Pressure 172 98 mmHg Heart Rate 102 bpm Respiratory Rate 22 "
    "Troponin I 4.82 ng/mL CK-MB 28.4 BNP 412 pg/mL HbA1c 8.1 percent "
    "LDL Cholesterol 142 mg/dL HDL Cholesterol 38 mg/dL Creatinine 1.1 "
    "Aspirin 81mg once daily Clopidogrel 75mg once daily Atorvastatin 80mg "
    "Metoprolol Succinate 50mg Lisinopril 10mg Nitroglycerin 0.4mg "
    "Cardiac Rehabilitation Program enroll within 2 weeks of discharge "
    "follow up with Dr Subramaniam in 2 weeks January 31 2024 at 10 AM "
    "aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeee "
    "aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeee "
    "ICD-10 CODES I21.09 STEMI I10 Hypertension E11.9 T2DM E78.5 "
    "DRG CODE 247 Percutaneous Cardiovascular Procedure Drug-Eluting Stent "
    "TOTAL LOS 7 days TOTAL CHARGES 84291.00 END OF DISCHARGE SUMMARY "
)

result = huffman_compress(long_text)
m      = result["metrics"]
recovered = huffman_decompress(result["compressed_bytes"],
                               result["padding"])

check("lossless on long text",       recovered == long_text)
check("compression ratio > 1.0",    m["compression_ratio"] > 1.0,
      f"— got {m['compression_ratio']}")
check("ratio is a float",            isinstance(m["compression_ratio"], float))

print(f"\n  compression_ratio   = {m['compression_ratio']}x")
print(f"  original size       = {len(long_text)} chars")
print(f"  compressed size     = {len(result['compressed_bytes'])} bytes")

# ══════════════════════════════════════════════════════
section("3. SHANNON ENTROPY — graduate requirement")
# ══════════════════════════════════════════════════════

result_rep  = huffman_compress("aaaaaaaaaa")
result_div  = huffman_compress("abcdefghij")

e_rep = result_rep["metrics"]["entropy"]
e_div = result_div["metrics"]["entropy"]

check("entropy of uniform text is 0.0",   e_rep == 0.0,
      f"— got {e_rep}")
check("entropy of diverse text > 0.0",    e_div > 0.0,
      f"— got {e_div}")
check("diverse text has higher entropy",  e_div > e_rep,
      f"— {e_div} > {e_rep}")
check("entropy is a float",               isinstance(e_div, float))

print(f"\n  entropy (uniform) = {e_rep}")
print(f"  entropy (diverse) = {e_div}")

# ══════════════════════════════════════════════════════
section("4. ENCODING EFFICIENCY — graduate requirement")
# ══════════════════════════════════════════════════════

result = huffman_compress(long_text)
eff    = result["metrics"]["encoding_efficiency"]

check("efficiency is a float",          isinstance(eff, float))
check("efficiency is between 0 and 1",  0.0 < eff <= 1.0,
      f"— got {eff}")
check("efficiency > 0.8 on long text",  eff > 0.8,
      f"— got {eff}")

print(f"\n  encoding_efficiency = {eff}")
print(f"  (1.0 = perfect, >0.9 = excellent)")

# ══════════════════════════════════════════════════════
section("5. METRICS STRUCTURE — Person D API contract")
# ══════════════════════════════════════════════════════

result = huffman_compress("test text for api")
check("returns compressed_bytes key",   "compressed_bytes" in result)
check("returns padding key",            "padding" in result)
check("returns metrics key",            "metrics" in result)
check("compressed_bytes is bytes",      isinstance(result["compressed_bytes"], bytes))
check("padding is int",                 isinstance(result["padding"], int))
check("padding between 0 and 7",        0 <= result["padding"] <= 7)
check("metrics has compression_ratio",  "compression_ratio"   in result["metrics"])
check("metrics has entropy",            "entropy"             in result["metrics"])
check("metrics has encoding_efficiency","encoding_efficiency" in result["metrics"])

# ══════════════════════════════════════════════════════
section("6. FULL PIPELINE DEMO — what judges will see")
# ══════════════════════════════════════════════════════

ocr_text = (
    "[OCR OUTPUT] Patient Name: John Smith "
    "Date: 2024-01-15 Diagnosis: Hypertension "
    "Medication: Lisinopril 10mg "
    "Total Bill: 349.99 USD "
    "Insurance ID: INS-4821-XYZ"
)

print(f"\n  Input text ({len(ocr_text)} chars):")
print(f"  {ocr_text[:60]}...")

result    = huffman_compress(ocr_text)
recovered = huffman_decompress(result["compressed_bytes"],
                               result["padding"])
m         = result["metrics"]

print(f"\n  Compressed to : {len(result['compressed_bytes'])} bytes")
print(f"  Padding bits  : {result['padding']}")
print(f"\n  --- Metrics (Graduate Requirements) ---")
print(f"  Compression Ratio   : {m['compression_ratio']}x")
print(f"  Shannon Entropy     : {m['entropy']} bits/symbol")
print(f"  Encoding Efficiency : {round(m['encoding_efficiency']*100, 2)}%")
print(f"\n  Recovered text matches original: {recovered == ocr_text}")

check("full pipeline lossless", recovered == ocr_text)
check("ratio computed",         m["compression_ratio"] > 0)
check("entropy computed",       m["entropy"] > 0)
check("efficiency computed",    m["encoding_efficiency"] > 0)

# ══════════════════════════════════════════════════════
print(f"\n{'='*55}")
print(f"  RESULTS: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("  ALL TESTS PASSED — ready for Person D handoff")
else:
    print("  SOME TESTS FAILED — check above")
print(f"{'='*55}\n")
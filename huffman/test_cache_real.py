# huffman/test_cache_real.py
import time
from pipeline import huffman_compress
from cache_manager import CompressionCache

cache = CompressionCache()

# ── long realistic OCR-style text ──────────────────────────────────
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

print(f"Input text length: {len(long_text)} chars\n")

# ── FIRST CALL: cache miss, must compute ──────────────────────────
print("=== First call (cache miss) ===")
start1        = time.perf_counter()
cached        = cache.get(long_text)
if cached is None:
    result    = huffman_compress(long_text)
    cache.set(long_text, result)
    source    = "computed"
else:
    result    = cached
    source    = "cache"
elapsed1      = (time.perf_counter() - start1) * 1000
print(f"Source         : {source}")
print(f"Time taken     : {elapsed1:.2f} ms")
print(f"Compressed size: {len(result['compressed_bytes'])} bytes")
print(f"Metrics        : {result['metrics']}")

# ── SECOND CALL: cache hit, instant ───────────────────────────────
print("\n=== Second call (cache hit) ===")
start2        = time.perf_counter()
cached        = cache.get(long_text)
if cached is None:
    result    = huffman_compress(long_text)
    cache.set(long_text, result)
    source    = "computed"
else:
    result    = cached
    source    = "cache"
elapsed2      = (time.perf_counter() - start2) * 1000
print(f"Source         : {source}")
print(f"Time taken     : {elapsed2:.2f} ms")
print(f"Compressed size: {len(result['compressed_bytes'])} bytes")

# ── THIRD CALL: different text, cache miss again ──────────────────
different_text = long_text + " EXTRA TEXT ADDED AT THE END"
print("\n=== Third call (different text — cache miss) ===")
start3        = time.perf_counter()
cached        = cache.get(different_text)
if cached is None:
    result    = huffman_compress(different_text)
    cache.set(different_text, result)
    source    = "computed"
else:
    result    = cached
    source    = "cache"
elapsed3      = (time.perf_counter() - start3) * 1000
print(f"Source         : {source}")
print(f"Time taken     : {elapsed3:.2f} ms")

# ── SUMMARY ───────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  CACHE PERFORMANCE SUMMARY")
print("=" * 55)
print(f"  Cache miss (first call) : {elapsed1:.2f} ms")
print(f"  Cache hit  (same text)  : {elapsed2:.2f} ms")
print(f"  Cache miss (new text)   : {elapsed3:.2f} ms")
print(f"  Speedup from caching    : {elapsed1/elapsed2:.0f}x faster")
print(f"  Total cached entries    : {cache.size()}")
print("=" * 55)
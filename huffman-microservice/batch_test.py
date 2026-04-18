import requests
import json

URL = "http://127.0.0.1:8000/compress"

# 1. Large Technical Dataset (Simulating a full document scan)
large_doc = """
Adaptive Huffman coding was first conceived by Faller and Gallager, and later refined by Knuth and Vitter. 
Unlike static Huffman coding, which requires two passes over the data (one to build the frequency table and another to encode), 
adaptive Huffman coding builds the tree dynamically as it reads the characters. 
This is critical for real-time systems like OCR pipelines where the statistics of the data may change over time.
The 'Sibling Property' is the core constraint that keeps the Huffman tree optimal. 
Whenever a weight is incremented, the algorithm checks if the node still holds its correct position in the order. 
If not, a swap occurs. This ensures that the most frequent symbols always have the shortest bit-length.
By the time we reach the end of this paragraph, the algorithm has seen enough lowercase 'e's and 't's 
that it is likely encoding them in just 3 or 4 bits instead of the original 8.
""" * 5  # We multiply by 5 to create a substantial data block

# 2. Large Log File (Simulating a massive system dump)
large_log = "[root@silo-iu ~]# system_diagnostic_report --verbose\n" + (
    "ERROR: Segfault at 0x7ffd5a; memory_leak_detected: True; retry_count: 5\n"
    "DEBUG: free -m; used: 4096; free: 1024; shared: 256; buff/cache: 512\n"
) * 20 

# 3. Massive Repetition (Testing the 1-bit limit)
massive_repetition = ("A" * 500) + ("B" * 500) + ("!" * 100)

batches = {
    "Large Document (Real English)": large_doc,
    "Large System Log (Structured)": large_log,
    "Massive Repetition (The 1-Bit Test)": massive_repetition
}

print("--- STARTING LARGE-SCALE BATCH TEST ---\n")

for name, text in batches.items():
    print(f"Testing {name} ({len(text)} chars)...")
    try:
        response = requests.post(URL, json={"text": text})
        if response.status_code == 200:
            res_data = response.json()["data"]
            m = res_data["metrics"]
            print(f"  [SUCCESS]")
            print(f"  - Ratio: {m['compression_ratio']}x")
            print(f"  - Efficiency: {m['encoding_efficiency_percent']}%")
            print(f"  - Lossless: {res_data['lossless_verification']}")
        else:
            print(f"  [FAILED] {response.status_code}")
    except Exception as e:
        print(f"  [ERROR] {e}")
    print("-" * 50)

print("\n--- TEST COMPLETE ---")
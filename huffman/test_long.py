# huffman/test_long.py
from pipeline import huffman_compress, huffman_decompress

long_text = """
[root@silo-iu ~]# cat /var/log/syslog | grep 'segfault'
Segmentation fault at memory address 0x7ffd5a...
Core dumped. Attempting to debug student C code...
running: gcc -Wall -Werror hw7.c -o hw7
Warning: implicit declaration of function 'malloc' in assignment 8.
Checking memory usage: free -m
Changing file permissions: chmod 777 hw7.c
OCR noise simulated: T#i$ !s a v3ry m3ssy s(r!ng w!th r@nd0m p*nctuat!0n.
Let's test repetitive stream balancing: aaaaaaaaaabbbbbbbbbbcccccccccc
EOF. Terminating process 010101010101.
"""

print("Compressing...")
result    = huffman_compress(long_text.strip())
recovered = huffman_decompress(result["compressed_bytes"],
                               result["padding"])

m = result["metrics"]
print(f"Original Text Length : {len(long_text.strip())} chars")
print(f"Compressed Size      : {len(result['compressed_bytes'])} bytes")
print(f"Lossless Match       : {recovered == long_text.strip()}")
print(f"Compression Ratio    : {m['compression_ratio']}x")
print(f"Text Entropy         : {m['entropy']} bits/symbol")
print(f"Encoding Efficiency  : {round(m['encoding_efficiency']*100, 2)}%")
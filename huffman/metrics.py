# huffman/metrics.py
import math

def compression_ratio(original_text: str, compressed_bytes: bytes) -> float:
    """
    How many times smaller the compressed output is vs original.
    > 1.0 means compression happened. = 1.0 means no gain.
    """
    original_bits   = len(original_text) * 8
    compressed_bits = len(compressed_bytes) * 8
    if compressed_bits == 0:
        return 0.0
    return round(original_bits / compressed_bits, 4)


def shannon_entropy(text: str) -> float:
    """
    Theoretical minimum bits per character for this text.
    Higher = more random/diverse. Lower = more repetitive.
    """
    if not text:
        return 0.0

    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1

    total   = len(text)
    entropy = 0.0
    for count in freq.values():
        p        = count / total
        entropy -= p * math.log2(p)

    return round(entropy, 4)


def average_code_length(text: str, symbol_map: dict, get_code_fn) -> float:
    """
    Weighted average of how many bits each character takes
    in the final Huffman tree state.
    """
    if not text:
        return 0.0

    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1

    total  = len(text)
    avg    = 0.0
    for char, count in freq.items():
        if char in symbol_map:
            code_len = len(get_code_fn(symbol_map[char]))
            avg     += (count / total) * code_len

    return round(avg, 4)


def encoding_efficiency(text: str, symbol_map: dict, get_code_fn) -> float:
    """
    How close we are to the theoretical minimum (Shannon entropy).
    1.0 = perfect. Closer to 1.0 = better encoding.
    """
    entropy  = shannon_entropy(text)
    avg_bits = average_code_length(text, symbol_map, get_code_fn)

    if avg_bits == 0:
        return 0.0
    return round(entropy / avg_bits, 4)


def compute_all_metrics(original_text: str,
                        compressed_bytes: bytes,
                        symbol_map: dict,
                        get_code_fn) -> dict:
    """
    Single call that returns all three graduate-required metrics.
    This is what pipeline.py will call.
    """
    return {
        "compression_ratio":   compression_ratio(original_text, compressed_bytes),
        "entropy":             shannon_entropy(original_text),
        "encoding_efficiency": encoding_efficiency(original_text,
                                                   symbol_map,
                                                   get_code_fn)
    }
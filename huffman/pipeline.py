# huffman/pipeline.py
from encoder import AdaptiveHuffmanEncoder
from decoder import AdaptiveHuffmanDecoder
from metrics import compute_all_metrics


def huffman_compress(text: str) -> dict:
    """
    Compress text using adaptive Huffman encoding.

    Returns a dict with:
      - compressed_bytes : bytes   — the compressed data
      - padding          : int     — trailing bits added to fill last byte
      - metrics          : dict    — compression_ratio, entropy, encoding_efficiency
    
    Person D passes compressed_bytes + padding to huffman_decompress()
    to recover the original text.
    """
    encoder              = AdaptiveHuffmanEncoder()
    compressed, padding  = encoder.encode(text)

    metrics = compute_all_metrics(
        original_text    = text,
        compressed_bytes = compressed,
        symbol_map       = encoder.tree.symbol_map,
        get_code_fn      = encoder.tree.get_code
    )

    return {
        "compressed_bytes": compressed,
        "padding":          padding,
        "metrics":          metrics
    }


def huffman_decompress(compressed_bytes: bytes, padding: int) -> str:
    """
    Recover original text from compressed bytes.
    Lossless — output is identical to the original text passed to huffman_compress().
    """
    decoder = AdaptiveHuffmanDecoder()
    return decoder.decode(compressed_bytes, padding)
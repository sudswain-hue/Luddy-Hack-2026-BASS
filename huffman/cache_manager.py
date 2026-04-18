# huffman/cache_manager.py
import hashlib

class CompressionCache:
    """
    In-memory cache for compression results.
    Uses MD5 hash of input text as the key.
    """

    def __init__(self):
        self._cache = {}

    def _get_hash(self, text: str) -> str:
        """Create a short unique fingerprint for the input text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def get(self, text: str):
        """Return cached response data if present, else None."""
        return self._cache.get(self._get_hash(text))

    def set(self, text: str, response_data: dict):
        """Store a compression response under the text's hash."""
        self._cache[self._get_hash(text)] = response_data

    def clear(self):
        """Clear all cached entries."""
        self._cache.clear()

    def size(self) -> int:
        """Return the number of cached entries."""
        return len(self._cache)
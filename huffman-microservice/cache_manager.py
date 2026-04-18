import hashlib

class CompressionCache:
    def __init__(self):
        self._cache = {}

    def get_hash(self, text):
        # Creates a unique, short fingerprint for the long text string
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text):
        return self._cache.get(self.get_hash(text))

    def set(self, text, response_data):
        self._cache[self.get_hash(text)] = response_data

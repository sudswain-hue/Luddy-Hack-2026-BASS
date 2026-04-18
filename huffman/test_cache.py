# huffman/test_cache.py
from cache_manager import CompressionCache

cache = CompressionCache()

# test 1: empty cache returns None
assert cache.get("hello") is None
print("TEST 1 PASS — empty cache returns None")

# test 2: set then get returns same data
cache.set("hello", {"result": "abc"})
assert cache.get("hello") == {"result": "abc"}
print("TEST 2 PASS — set and get work")

# test 3: different text returns different cache miss
assert cache.get("world") is None
print("TEST 3 PASS — different key returns None")

# test 4: same hash for same text
h1 = cache._get_hash("same text")
h2 = cache._get_hash("same text")
assert h1 == h2
print("TEST 4 PASS — same text produces same hash")

# test 5: different hash for different text
h3 = cache._get_hash("different text")
assert h1 != h3
print("TEST 5 PASS — different text produces different hash")

# test 6: size tracking
assert cache.size() == 1
cache.set("world", {"result": "xyz"})
assert cache.size() == 2
print("TEST 6 PASS — size tracks correctly")

# test 7: clear
cache.clear()
assert cache.size() == 0
assert cache.get("hello") is None
print("TEST 7 PASS — clear empties cache")

print("\nALL CACHE TESTS PASSED")
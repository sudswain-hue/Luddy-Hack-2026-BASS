# huffman/test_step2.py
from node import Node
from tree import AdaptiveHuffmanTree

# ── test 1: initial state ─────────────────────────────────────────
tree = AdaptiveHuffmanTree()
assert tree.root is tree.nyt
assert tree.root.is_nyt()
assert len(tree.nodes) == 1
print("TEST 1 PASSED — initial state correct")

# ── test 2: add first symbol ──────────────────────────────────────
tree = AdaptiveHuffmanTree()
leaf_a = tree.add_new_symbol('a')
tree.update(leaf_a)

# tree should now have 3 nodes: internal root, new_nyt, leaf_a
assert len(tree.nodes) == 4  # original NYT replaced + 3 new nodes added
assert 'a' in tree.symbol_map
assert tree.symbol_map['a'] is leaf_a
assert leaf_a.weight == 1
print("TEST 2 PASSED — first symbol added and updated")

# ── test 3: add second symbol ─────────────────────────────────────
tree = AdaptiveHuffmanTree()
leaf_a = tree.add_new_symbol('a')
tree.update(leaf_a)
leaf_b = tree.add_new_symbol('b')
tree.update(leaf_b)

assert 'a' in tree.symbol_map
assert 'b' in tree.symbol_map
assert tree.symbol_map['a'].weight == 1
assert tree.symbol_map['b'].weight == 1
print("TEST 3 PASSED — two symbols in tree")

# ── test 4: get_code returns a string ─────────────────────────────
tree = AdaptiveHuffmanTree()
leaf_a = tree.add_new_symbol('a')
tree.update(leaf_a)
code = tree.get_code(leaf_a)
assert isinstance(code, str)
assert all(c in '01' for c in code)
print(f"TEST 4 PASSED — code for 'a' after one insertion: '{code}'")

# ── test 5: repeated symbol increases weight ──────────────────────
tree = AdaptiveHuffmanTree()
for _ in range(3):
    if 'a' not in tree.symbol_map:
        leaf = tree.add_new_symbol('a')
    else:
        leaf = tree.symbol_map['a']
    tree.update(leaf)

assert tree.symbol_map['a'].weight == 3
print("TEST 5 PASSED — weight increments correctly for repeated symbol")

print("\nALL STEP 2 TESTS PASSED")
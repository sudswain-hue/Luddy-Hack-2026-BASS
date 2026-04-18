# huffman/debug_test.py
print("start")

from tree import AdaptiveHuffmanTree
print("tree imported")

t = AdaptiveHuffmanTree()
print("tree created")

leaf = t.add_new_symbol('h')
print("symbol added")

t.update(leaf)
print("update done")

code = t.get_code(leaf)
print(f"code: {code!r}")

print("DONE")
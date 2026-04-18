# huffman/test_step1.py
from node import Node

# test 1: basic creation
n = Node(symbol='a', weight=3, order=10)
print(n)
assert n.symbol == 'a'
assert n.weight == 3
assert n.order  == 10
assert n.parent is None
assert n.left   is None
assert n.right  is None

# test 2: is_leaf
assert n.is_leaf() == True
n.left = Node()
assert n.is_leaf() == False

# test 3: is_nyt
nyt = Node(symbol='NYT')
assert nyt.is_nyt() == True
assert n.is_nyt()   == False

print("ALL STEP 1 TESTS PASSED")
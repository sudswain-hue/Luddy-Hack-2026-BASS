# huffman/node.py

class Node:
    def __init__(self, symbol=None, weight=0, order=0):
        self.symbol = symbol    # None = internal, 'NYT' = not yet transmitted
        self.weight = weight    # frequency count
        self.order  = order     # higher order = higher priority in sibling property
        self.parent = None
        self.left   = None
        self.right  = None

    def is_leaf(self):
        return self.left is None and self.right is None

    def is_nyt(self):
        return self.symbol == 'NYT'

    def __repr__(self):
        return (f"Node(symbol={self.symbol!r}, "
                f"weight={self.weight}, order={self.order})")
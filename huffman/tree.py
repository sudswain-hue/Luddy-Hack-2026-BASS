# huffman/tree.py
from node import Node

class AdaptiveHuffmanTree:

    def __init__(self):
        self.nyt        = Node(symbol='NYT', weight=0, order=512)
        self.root       = self.nyt
        self.symbol_map = {}

    def get_code(self, node):
        if node == self.root:
            return ""
        code    = ""
        current = node
        while current.parent is not None:
            if current.parent.left == current:
                code = "0" + code
            else:
                code = "1" + code
            current = current.parent
        return code

    def _find_block_leader(self, weight):
        """Find highest-order node with given weight (excluding root)."""
        leader    = None
        max_order = -1

        def search(node):
            nonlocal leader, max_order
            if node is None:
                return
            if (node.weight == weight
                    and node.order > max_order
                    and node != self.root):
                leader    = node
                max_order = node.order
            search(node.left)
            search(node.right)

        search(self.root)
        return leader

    def _swap(self, n1, n2):
        """Swap two nodes, handling sibling and non-sibling cases."""
        if (n1.parent is None or n2.parent is None
                or n1 == n2
                or n1.parent == n2
                or n2.parent == n1):
            return

        p1, p2 = n1.parent, n2.parent

        if p1 == p2:
            # siblings — swap left/right directly
            p1.left, p1.right = p1.right, p1.left
        else:
            if p1.left == n1:
                p1.left  = n2
            else:
                p1.right = n2
            if p2.left == n2:
                p2.left  = n1
            else:
                p2.right = n1

        n1.parent, n2.parent = p2, p1
        n1.order,  n2.order  = n2.order, n1.order

    def update(self, symbol):
        """Add new symbol or increment existing — rebalance tree after each."""
        current = None

        if symbol not in self.symbol_map:
            # split NYT into: internal -> (new_nyt, new_leaf)
            nyt_order  = self.nyt.order
            internal   = Node(weight=0, order=nyt_order)
            new_leaf   = Node(symbol=symbol, weight=0, order=nyt_order - 1)
            parent_nyt = self.nyt.parent

            if parent_nyt is not None:
                if parent_nyt.left == self.nyt:
                    parent_nyt.left  = internal
                else:
                    parent_nyt.right = internal
                internal.parent = parent_nyt
            else:
                self.root = internal

            new_leaf.parent  = internal
            self.nyt.order   = nyt_order - 2
            self.nyt.parent  = internal
            internal.left    = self.nyt
            internal.right   = new_leaf

            self.symbol_map[symbol] = new_leaf
            current = new_leaf
        else:
            current = self.symbol_map[symbol]

        # walk up and rebalance
        while current is not None:
            leader = self._find_block_leader(current.weight)
            if (leader is not None
                    and leader != current
                    and leader.parent != current
                    and current.parent != leader):
                self._swap(current, leader)
            current.weight += 1
            current = current.parent
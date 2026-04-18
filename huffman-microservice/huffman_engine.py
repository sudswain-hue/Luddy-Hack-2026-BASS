import math
from collections import Counter

class Node:
    def __init__(self, symbol=None, weight=0, order=0):
        self.symbol = symbol      # The character (None for internal nodes)
        self.weight = weight      # Frequency count
        self.order = order        # Node number (used for keeping the tree balanced)
        self.parent = None
        self.left = None
        self.right = None

    def is_leaf(self):
        return self.left is None and self.right is None
    
class AdaptiveHuffman:
    def __init__(self):
        # The NYT (Not Yet Transmitted) node is the starting point
        self.NYT = Node(symbol="NYT", weight=0, order=512) 
        self.root = self.NYT
        self.nodes_map = {} # Maps characters to their specific leaf nodes

    def get_code(self, node):
        """Traverses up the tree from a node to get its binary path."""
        if node == self.root:
            return ""
        code = ""
        curr = node
        while curr.parent is not None:
            if curr.parent.left == curr:
                code = "0" + code
            else:
                code = "1" + code
            curr = curr.parent
        return code

    def find_block_leader(self, weight):
        """Finds the node with the highest order that has the given weight."""
        # In a full FGK implementation, this ensures the 'sibling property' is maintained.
        # For this base setup, we simulate finding the highest ordered node of same weight.
        leader = None
        max_order = -1
        
        def search(node):
            nonlocal leader, max_order
            if node is None: return
            if node.weight == weight and node.order > max_order and node != self.root:
                leader = node
                max_order = node.order
            search(node.left)
            search(node.right)
            
        search(self.root)
        return leader

    def swap_nodes(self, n1, n2):
        """Swaps two nodes in the tree to maintain balance, handling sibling collisions."""
        if n1.parent is None or n2.parent is None or n1 == n2 or n1.parent == n2 or n2.parent == n1:
            return

        p1, p2 = n1.parent, n2.parent
        
        if p1 == p2:
            # They are siblings. Swap the left and right pointers directly.
            p1.left, p1.right = p1.right, p1.left
        else:
            # They have different parents. Safely reassign.
            if p1.left == n1: p1.left = n2
            else: p1.right = n2
            
            if p2.left == n2: p2.left = n1
            else: p2.right = n1
                
        # Swap parent pointers
        n1.parent, n2.parent = p2, p1
        # Swap order numbers
        n1.order, n2.order = n2.order, n1.order

    def update(self, symbol):
        """The core tree balancing engine. Updates weights and swaps nodes."""
        curr = None
        if symbol not in self.nodes_map:
            # 1. Handle New Symbol (Split the NYT node)
            nyt_order = self.NYT.order
            
            # Create new internal node where NYT used to be
            internal_node = Node(weight=0, order=nyt_order) # Initialize weight to 0. It will be incremented later.

            # Save parent of the current NYT
            parent_of_nyt = self.NYT.parent

            # Assign internal_node as child of parent_of_old_nyt
            if parent_of_nyt is not None:
                if parent_of_nyt.left == self.NYT:
                    parent_of_nyt.left = internal_node
                else:
                    parent_of_nyt.right = internal_node
                internal_node.parent = parent_of_nyt
            else:
                self.root = internal_node # If NYT was root, this new internal node becomes root

            # Create new leaf for the character
            new_leaf = Node(symbol=symbol, weight=0, order=nyt_order - 1) # Initialize new leaf weight to 0. It will be incremented later.
            new_leaf.parent = internal_node
            self.nodes_map[symbol] = new_leaf

            # Reposition the old NYT node to be a child of the new internal node
            self.NYT.order = nyt_order - 2 # New NYT has lowest order in this block
            self.NYT.parent = internal_node

            # Connect the new internal node to its children
            internal_node.left = self.NYT # NYT is typically the left child
            internal_node.right = new_leaf # New char leaf is the right child

            # Start updating weights from the new character leaf
            curr = new_leaf 
        else:
            # 2. Handle Existing Symbol
            curr = self.nodes_map[symbol]

        # 3. Traverse up and balance the tree
        # This loop now starts from 'curr' which is either the new leaf node or an existing character's leaf
        while curr is not None:
            # Find the block leader for the current node's weight
            # This simplified find_block_leader needs to be robust,
            # or a proper FGK algorithm should be implemented.
            leader = self.find_block_leader(curr.weight)
            
            # Only swap if a valid leader is found and it's not the same node or its parent
            # and the leader's parent is not the current node
            if leader is not None and leader != curr and leader.parent != curr and curr.parent != leader:
                self.swap_nodes(curr, leader)

            curr.weight += 1
            curr = curr.parent

    def encode(self, text):
        """Compresses the text into a binary string."""
        encoded_data = ""
        for char in text:
            if char not in self.nodes_map:
                # Send NYT path + standard 8-bit ASCII representation
                encoded_data += self.get_code(self.NYT)
                encoded_data += format(ord(char), '08b')
            else:
                # Send path of existing character
                encoded_data += self.get_code(self.nodes_map[char])
            
            # Dynamically update the tree
            self.update(char)
        return encoded_data

    def decode(self, binary_data):
        """Decompresses the binary string back to text (Lossless)."""
        decoded_text = ""
        curr = self.root
        i = 0
        
        while i < len(binary_data):
            # 1. Check if we are at the NYT node before reading a traversal bit
            if curr == self.NYT:
                # Read the next 8 bits for the ASCII character
                ascii_bin = binary_data[i:i+8]
                char = chr(int(ascii_bin, 2))
                decoded_text += char
                self.update(char)
                curr = self.root
                i += 8
                continue

            # 2. Traverse the tree based on the bit
            bit = binary_data[i]
            if bit == '0':
                curr = curr.left
            else:
                curr = curr.right
            
            # Move the index forward after reading the routing bit
            i += 1 

            # 3. Check if we hit a leaf AFTER traversing
            if curr.is_leaf():
                if curr == self.NYT:
                    # We hit the NYT node! Do NOT print it. 
                    # Just let the loop restart so it hits step 1 and reads 8 bits.
                    pass 
                else:
                    # We hit a normal character leaf.
                    char = curr.symbol
                    decoded_text += char
                    self.update(char)
                    curr = self.root
            
        return decoded_text
    
def calculate_metrics(original_text, compressed_binary):
    # 1. Compression Ratio (Assuming 8 bits per original character)
    original_bits = len(original_text) * 8
    compressed_bits = len(compressed_binary)
    ratio = original_bits / compressed_bits if compressed_bits > 0 else 0
    
    # 2. Shannon Entropy
    freqs = Counter(original_text)
    length = len(original_text)
    entropy = -sum((count/length) * math.log2(count/length) for count in freqs.values())
    
    # 3. Encoding Efficiency (Entropy / Average bits per char)
    avg_bits_per_char = compressed_bits / length if length > 0 else 0
    efficiency = (entropy / avg_bits_per_char) * 100 if avg_bits_per_char > 0 else 0
    
    return ratio, entropy, efficiency


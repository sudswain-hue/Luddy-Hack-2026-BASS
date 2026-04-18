# huffman/decoder.py
from tree import AdaptiveHuffmanTree

class AdaptiveHuffmanDecoder:

    def __init__(self):
        self.tree = AdaptiveHuffmanTree()

    def decode(self, compressed_bytes: bytes, padding: int) -> str:
        if not compressed_bytes:
            raise ValueError("Cannot decode empty bytes")

        bit_string = ''.join(format(b, '08b') for b in compressed_bytes)
        if padding > 0:
            bit_string = bit_string[:-padding]

        decoded = ""
        current = self.tree.root
        i       = 0

        while i < len(bit_string):

            if current == self.tree.nyt:
                # read 16 bits as Unicode code point
                if i + 16 > len(bit_string):
                    break
                symbol  = chr(int(bit_string[i:i+16], 2))
                decoded += symbol
                self.tree.update(symbol)
                current  = self.tree.root
                i       += 16
                continue

            bit     = bit_string[i]
            current = current.left if bit == '0' else current.right
            i      += 1

            if current.is_leaf():
                if current == self.tree.nyt:
                    pass
                else:
                    symbol   = current.symbol
                    decoded += symbol
                    self.tree.update(symbol)
                    current  = self.tree.root

        return decoded
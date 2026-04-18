# huffman/encoder.py
from tree import AdaptiveHuffmanTree
from typing import Tuple

class AdaptiveHuffmanEncoder:

    def __init__(self):
        self.tree = AdaptiveHuffmanTree()

    def encode(self, text: str) -> Tuple[bytes, int]:
        if not text:
            raise ValueError("Cannot encode empty string")

        bit_string = ""

        for symbol in text:
            if symbol not in self.tree.symbol_map:
                # emit NYT code + fixed 16-bit Unicode code point
                bit_string += self.tree.get_code(self.tree.nyt)
                bit_string += format(ord(symbol), '016b')
            else:
                bit_string += self.tree.get_code(
                                  self.tree.symbol_map[symbol])
            self.tree.update(symbol)

        padding = (8 - len(bit_string) % 8) % 8
        bit_string += '0' * padding

        byte_array = bytearray()
        for i in range(0, len(bit_string), 8):
            byte_array.append(int(bit_string[i:i+8], 2))

        return bytes(byte_array), padding
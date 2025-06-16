# Arithmetic coder generated with LLM based on pseudocode(https://medium.com/@roselyn.crsstm/arithmetic-coding-854ce46b2e22)
# because there is no stand-alone library in Python, with small manual modifications to work with the code.
class BitOutputStream:
    def __init__(self):
        self.buffer = []
        self.current_byte = 0
        self.num_bits_filled = 0

    def write(self, bit):
        if bit not in (0, 1):
            raise ValueError("Bit must be 0 or 1")
        self.current_byte = (self.current_byte << 1) | bit
        self.num_bits_filled += 1
        if self.num_bits_filled == 8:
            self.flush()

    def flush(self):
        if self.num_bits_filled > 0:
            self.current_byte <<= (8 - self.num_bits_filled)
            self.buffer.append(self.current_byte)
            self.current_byte = 0
            self.num_bits_filled = 0

    def get_bytes(self):
        self.flush()
        return bytes(self.buffer)

    def print_bits(self):
        self.flush()
        bits = []
        for byte in self.buffer:
            bits.append(f'{byte:08b}')
        print(''.join(bits))

class BitInputStream:
    def __init__(self, data):
        self.data = data
        self.position = 0
        self.current_byte = 0
        self.num_bits_remaining = 0

    def read(self):
        if self.num_bits_remaining == 0:
            if self.position >= len(self.data):
                return -1
            self.current_byte = self.data[self.position]
            self.position += 1
            self.num_bits_remaining = 8
        self.num_bits_remaining -= 1
        return (self.current_byte >> self.num_bits_remaining) & 1

class ArithmeticCoderBase:
    def __init__(self, num_bits, cumulative_freq):
        self.num_state_bits = num_bits
        self.full_range = 1 << num_bits
        self.half_range = self.full_range >> 1
        self.quarter_range = self.half_range >> 1
        self.low = 0
        self.high = self.full_range - 1
        self.cumulative_freq = cumulative_freq

class ArithmeticEncoder(ArithmeticCoderBase):
    def __init__(self, num_bits, cumulative_freq):
        super().__init__(num_bits, cumulative_freq)
        self.output = BitOutputStream()
        self.num_underflow = 0

    def write(self, symbol):
        total = self.cumulative_freq[-1]
        low_count = self.cumulative_freq[symbol]
        high_count = self.cumulative_freq[symbol + 1]

        range_ = self.high - self.low + 1
        self.high = self.low + (range_ * high_count // total) - 1
        self.low = self.low + (range_ * low_count // total)

        while True:
            if self.high < self.half_range:
                self._write_bit(0)
            elif self.low >= self.half_range:
                self._write_bit(1)
                self.low -= self.half_range
                self.high -= self.half_range
            elif self.low >= self.quarter_range and self.high < 3 * self.quarter_range:
                self.num_underflow += 1
                self.low -= self.quarter_range
                self.high -= self.quarter_range
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) + 1

    def finish(self):
        self.num_underflow += 1
        self._write_bit(0 if self.low < self.quarter_range else 1)
        self.output.flush()

    def _write_bit(self, bit):
        self.output.write(bit)
        for _ in range(self.num_underflow):
            self.output.write(1 - bit)
        self.num_underflow = 0

    def get_encoded_data(self):
        return self.output.get_bytes()

class ArithmeticDecoder(ArithmeticCoderBase):
    def __init__(self, num_bits, cumulative_freq, input_bytes):
        super().__init__(num_bits, cumulative_freq)
        self.input = BitInputStream(input_bytes)
        self.code = 0
        for _ in range(num_bits):
            self.code = (self.code << 1) | self._read_bit()

    def _read_bit(self):
        bit = self.input.read()
        return 0 if bit == -1 else bit

    def read(self):
        total = self.cumulative_freq[-1]
        range_ = self.high - self.low + 1
        offset = self.code - self.low
        value = ((offset + 1) * total - 1) // range_

        low_idx, high_idx = 0, len(self.cumulative_freq) - 1
        while low_idx + 1 < high_idx:
            mid = (low_idx + high_idx) // 2
            if self.cumulative_freq[mid] > value:
                high_idx = mid
            else:
                low_idx = mid

        symbol = low_idx
        low_count = self.cumulative_freq[symbol]
        high_count = self.cumulative_freq[symbol + 1]

        self.high = self.low + (range_ * high_count // total) - 1
        self.low = self.low + (range_ * low_count // total)

        while True:
            if self.high < self.half_range:
                pass
            elif self.low >= self.half_range:
                self.low -= self.half_range
                self.high -= self.half_range
                self.code -= self.half_range
            elif self.low >= self.quarter_range and self.high < 3 * self.quarter_range:
                self.low -= self.quarter_range
                self.high -= self.quarter_range
                self.code -= self.quarter_range
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) + 1
            self.code = (self.code << 1) | self._read_bit()

        return symbol
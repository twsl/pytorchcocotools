class RLE:
    def __init__(self, h=0, w=0, m=0, cnts=None):
        self.h = h
        self.w = w
        self.m = m
        self.cnts = cnts if cnts is not None else [0] * m


def rle_to_string(R):
    """Convert RLE counts to a string using a similar method to LEB128 but using 6 bits per char."""
    s = []
    for i in range(R.m):
        x = R.cnts[i]
        if i > 2:
            x -= R.cnts[i - 2]
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = (c & 0x10) != 0 if x != -1 else x != 0
            if more:
                c |= 0x20
            c += 48
            s.append(chr(c))
    return "".join(s)


def rle_from_string(R, s, h, w):
    """Convert a string back to RLE counts."""
    m = 0
    p = 0
    cnts = []
    while p < len(s):
        x = 0
        k = 0
        more = True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = (c & 0x20) != 0
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if m > 2:
            x += cnts[m - 2]
        cnts.append(x)
        m += 1
    R.__init__(h, w, m, cnts)


# Test the functions to ensure they return the initial value
# Initialize an RLE object
original_cnts = [1, 2, 3, 4, 5]
rle_original = RLE(h=10, w=15, m=5, cnts=original_cnts)

# Convert RLE to string
rle_string = rle_to_string(rle_original)

# Create a new RLE from string
rle_new = RLE()
rle_from_string(rle_new, rle_string, 10, 15)

# Check if the new RLE counts match the original
rle_new.cnts == original_cnts, rle_new.cnts, original_cnts

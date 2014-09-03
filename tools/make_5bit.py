# script to take an input raw binary audio file, assumed to be 8-bit audio, and
# output two 4-bit bytestreams where each byte sums to a 5-bit approximation of
# the 8-bit input file.

import struct
import sys

def main(argv):
  if len(argv) != 4:
    print 'usage: make_5bit.py input_file.raw out_file_a.bin out_file_b.bin'
    sys.exit()

  out1 = open(argv[2], "wb")
  out2 = open(argv[3], "wb")

  with open(argv[1], "rb") as f:
    byte = f.read(1)
    while byte != b"":
      b = struct.unpack('B', byte)[0] >> 3
      out1.write(struct.pack('B', b >> 1))
      if b % 2:
        out2.write(struct.pack('B', (b >> 1) + 1))
      else:
        out2.write(struct.pack('B', (b >> 1)))
      byte = f.read(1)

if __name__ == "__main__":
  main(sys.argv)

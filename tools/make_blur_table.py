# simple script to output a gaussian blur filter kernel

import math
import sys

def main(argv):
  if len(argv) != 3:
    print 'usage: make_blur_table.py <table dimension> <sigma>'
    sys.exit()

  dim = int(argv[1])
  sigma = float(argv[2])
  sigma_sq = sigma * sigma
  scalar = 1.0 / (2 * math.pi * sigma_sq)
  for y in range(-dim/2 + 1, dim/2 + 1):
    for x in range(-dim/2 + 1, dim/2 + 1):
      x_sq = float(x * x)
      y_sq = float(y * y)
      k = scalar * math.exp(-1.0 * (x_sq + y_sq) / (2.0 * sigma_sq))
      print '  %0.15ff,  // x = %d, y = %d' % (k, x, y)

if __name__ == "__main__":
  main(sys.argv)

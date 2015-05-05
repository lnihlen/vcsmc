import struct
import sys

def main(argv):
  if len(argv) != 2:
    print 'usage: sum_bin.py frame_program.bin'
    sys.exit()

  file_name = argv[1]
  total_clocks = 0
  address = int(0)

  with open(file_name, 'rb') as f:
    byte = f.read(1)
    address += 1
    while byte != b"":
      op = struct.unpack('B', byte)[0]
      line_str = '{:11d}'.format(total_clocks)
      line_str += ' {:0>8x}: '.format(address)
      if op == 0x4c:  # jmp
        total_clocks += 9
        new_address = (int(address / 1024) + 1) * 1024
        f.read(new_address - address)
        address = new_address
        line_str += 'jmp'
      elif op == 0xa9 or op == 0xa2 or op == 0xa0:  # lda, ldx, ldy
        total_clocks += 6
        raw_val = f.read(1)
        val = struct.unpack('B', raw_val)[0]
        address += 1
        if op == 0xa9:
          line_str += 'lda'
        elif op == 0xa2:
          line_str += 'ldx'
        else:
          line_str += 'ldy'
        line_str += ' {:0>2x}'.format(val)
      elif op == 0xea:  # nop
        total_clocks += 6
        line_str += 'nop'
      elif op == 0x85 or op == 0x86 or op == 0x84:  # sta, stx, sty
        raw_addy = f.read(1)
        addy = struct.unpack('B', raw_addy)[0]
        if op == 0x85:
          line_str += 'sta'
        elif op == 0x86:
          line_str += 'stx'
        else:
          line_str += 'sty'
        line_str += ' {:0>2x}'.format(addy)
        if addy == 0x02:
          total_clocks += 228 - (total_clocks % 228)
        else:
          total_clocks += 9
        address += 1
      else:
        print 'unknown opcode %x' % op
        sys.exit()
      print line_str
      byte = f.read(1)
      address += 1

  print '%d total clocks' % total_clocks


if __name__ == "__main__":
  main(sys.argv)

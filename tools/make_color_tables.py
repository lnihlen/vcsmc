# Script to make the .cc tables of the Atari colors in RGB and L*ab color
# spaces. Also generates the test data used in the OpenCL conversion test data.
# relies on the colormath module

import sys

from colormath.color_conversions import convert_color
from colormath.color_objects import LabColor
from colormath.color_objects import sRGBColor

def build_atari_colors_list():
  atari_colors = []
  atari_colors.append(sRGBColor.new_from_rgb_hex('000000'))  # 0
  atari_colors.append(sRGBColor.new_from_rgb_hex('404040'))  # 2
  atari_colors.append(sRGBColor.new_from_rgb_hex('6c6c6c'))  # 4
  atari_colors.append(sRGBColor.new_from_rgb_hex('909090'))  # 6
  atari_colors.append(sRGBColor.new_from_rgb_hex('b0b0b0'))  # 8
  atari_colors.append(sRGBColor.new_from_rgb_hex('c8c8c8'))  # a
  atari_colors.append(sRGBColor.new_from_rgb_hex('dcdcdc'))  # c
  atari_colors.append(sRGBColor.new_from_rgb_hex('ececec'))  # e
  atari_colors.append(sRGBColor.new_from_rgb_hex('444400'))  # 10
  atari_colors.append(sRGBColor.new_from_rgb_hex('646410'))  # 12
  atari_colors.append(sRGBColor.new_from_rgb_hex('848424'))  # 14
  atari_colors.append(sRGBColor.new_from_rgb_hex('a0a034'))  # 16
  atari_colors.append(sRGBColor.new_from_rgb_hex('b8b840'))  # 18
  atari_colors.append(sRGBColor.new_from_rgb_hex('d0d050'))  # 1a
  atari_colors.append(sRGBColor.new_from_rgb_hex('e8e85c'))  # 1c
  atari_colors.append(sRGBColor.new_from_rgb_hex('fcfc68'))  # 1e
  atari_colors.append(sRGBColor.new_from_rgb_hex('702800'))  # 20
  atari_colors.append(sRGBColor.new_from_rgb_hex('844414'))  # 22
  atari_colors.append(sRGBColor.new_from_rgb_hex('985c28'))  # 24
  atari_colors.append(sRGBColor.new_from_rgb_hex('ac783c'))  # 26
  atari_colors.append(sRGBColor.new_from_rgb_hex('bc8c4c'))  # 28
  atari_colors.append(sRGBColor.new_from_rgb_hex('cca05c'))  # 2a
  atari_colors.append(sRGBColor.new_from_rgb_hex('dcb468'))  # 2c
  atari_colors.append(sRGBColor.new_from_rgb_hex('ecc878'))  # 2e
  atari_colors.append(sRGBColor.new_from_rgb_hex('841800'))  # 30
  atari_colors.append(sRGBColor.new_from_rgb_hex('983418'))  # 32
  atari_colors.append(sRGBColor.new_from_rgb_hex('ac5030'))  # 34
  atari_colors.append(sRGBColor.new_from_rgb_hex('c06848'))  # 36
  atari_colors.append(sRGBColor.new_from_rgb_hex('d0805c'))  # 38
  atari_colors.append(sRGBColor.new_from_rgb_hex('e09470'))  # 3a
  atari_colors.append(sRGBColor.new_from_rgb_hex('eca880'))  # 3c
  atari_colors.append(sRGBColor.new_from_rgb_hex('fcbc94'))  # 3e
  atari_colors.append(sRGBColor.new_from_rgb_hex('880000'))  # 40
  atari_colors.append(sRGBColor.new_from_rgb_hex('9c2020'))  # 42
  atari_colors.append(sRGBColor.new_from_rgb_hex('b03c3c'))  # 44
  atari_colors.append(sRGBColor.new_from_rgb_hex('c05858'))  # 46
  atari_colors.append(sRGBColor.new_from_rgb_hex('d07070'))  # 48
  atari_colors.append(sRGBColor.new_from_rgb_hex('e08888'))  # 4a
  atari_colors.append(sRGBColor.new_from_rgb_hex('eca0a0'))  # 4c
  atari_colors.append(sRGBColor.new_from_rgb_hex('fcb4b4'))  # 4e
  atari_colors.append(sRGBColor.new_from_rgb_hex('78005c'))  # 50
  atari_colors.append(sRGBColor.new_from_rgb_hex('8c2074'))  # 52
  atari_colors.append(sRGBColor.new_from_rgb_hex('a03c88'))  # 54
  atari_colors.append(sRGBColor.new_from_rgb_hex('b0589c'))  # 56
  atari_colors.append(sRGBColor.new_from_rgb_hex('c070b0'))  # 58
  atari_colors.append(sRGBColor.new_from_rgb_hex('d084c0'))  # 5a
  atari_colors.append(sRGBColor.new_from_rgb_hex('dc9cd0'))  # 5c
  atari_colors.append(sRGBColor.new_from_rgb_hex('ecb0e0'))  # 5e
  atari_colors.append(sRGBColor.new_from_rgb_hex('480078'))  # 60
  atari_colors.append(sRGBColor.new_from_rgb_hex('602090'))  # 62
  atari_colors.append(sRGBColor.new_from_rgb_hex('783ca4'))  # 64
  atari_colors.append(sRGBColor.new_from_rgb_hex('8c58b8'))  # 66
  atari_colors.append(sRGBColor.new_from_rgb_hex('a070cc'))  # 68
  atari_colors.append(sRGBColor.new_from_rgb_hex('b484dc'))  # 6a
  atari_colors.append(sRGBColor.new_from_rgb_hex('c49cec'))  # 6c
  atari_colors.append(sRGBColor.new_from_rgb_hex('d4b0fc'))  # 6e
  atari_colors.append(sRGBColor.new_from_rgb_hex('140084'))  # 70
  atari_colors.append(sRGBColor.new_from_rgb_hex('302098'))  # 72
  atari_colors.append(sRGBColor.new_from_rgb_hex('4c3cac'))  # 74
  atari_colors.append(sRGBColor.new_from_rgb_hex('6858c0'))  # 76
  atari_colors.append(sRGBColor.new_from_rgb_hex('7c70d0'))  # 78
  atari_colors.append(sRGBColor.new_from_rgb_hex('9488e0'))  # 7a
  atari_colors.append(sRGBColor.new_from_rgb_hex('a8a0ec'))  # 7c
  atari_colors.append(sRGBColor.new_from_rgb_hex('bcb4fc'))  # 7e
  atari_colors.append(sRGBColor.new_from_rgb_hex('000088'))  # 80
  atari_colors.append(sRGBColor.new_from_rgb_hex('1c209c'))  # 82
  atari_colors.append(sRGBColor.new_from_rgb_hex('3840b0'))  # 84
  atari_colors.append(sRGBColor.new_from_rgb_hex('505cc0'))  # 86
  atari_colors.append(sRGBColor.new_from_rgb_hex('6874d0'))  # 88
  atari_colors.append(sRGBColor.new_from_rgb_hex('7c8ce0'))  # 8a
  atari_colors.append(sRGBColor.new_from_rgb_hex('90a4ec'))  # 8c
  atari_colors.append(sRGBColor.new_from_rgb_hex('a4b8fc'))  # 8e
  atari_colors.append(sRGBColor.new_from_rgb_hex('00187c'))  # 90
  atari_colors.append(sRGBColor.new_from_rgb_hex('1c3890'))  # 92
  atari_colors.append(sRGBColor.new_from_rgb_hex('3854a8'))  # 94
  atari_colors.append(sRGBColor.new_from_rgb_hex('5070bc'))  # 96
  atari_colors.append(sRGBColor.new_from_rgb_hex('6888cc'))  # 98
  atari_colors.append(sRGBColor.new_from_rgb_hex('7c9cdc'))  # 9a
  atari_colors.append(sRGBColor.new_from_rgb_hex('90b4ec'))  # 9c
  atari_colors.append(sRGBColor.new_from_rgb_hex('a4c8fc'))  # 9e
  atari_colors.append(sRGBColor.new_from_rgb_hex('002c5c'))  # a0
  atari_colors.append(sRGBColor.new_from_rgb_hex('1c4c78'))  # a2
  atari_colors.append(sRGBColor.new_from_rgb_hex('386890'))  # a4
  atari_colors.append(sRGBColor.new_from_rgb_hex('5084ac'))  # a6
  atari_colors.append(sRGBColor.new_from_rgb_hex('689cc0'))  # a8
  atari_colors.append(sRGBColor.new_from_rgb_hex('7cb4d4'))  # aa
  atari_colors.append(sRGBColor.new_from_rgb_hex('90cce8'))  # ac
  atari_colors.append(sRGBColor.new_from_rgb_hex('a4e0fc'))  # ae
  atari_colors.append(sRGBColor.new_from_rgb_hex('003c2c'))  # b0
  atari_colors.append(sRGBColor.new_from_rgb_hex('1c5c48'))  # b2
  atari_colors.append(sRGBColor.new_from_rgb_hex('387c64'))  # b4
  atari_colors.append(sRGBColor.new_from_rgb_hex('509c80'))  # b6
  atari_colors.append(sRGBColor.new_from_rgb_hex('68b494'))  # b8
  atari_colors.append(sRGBColor.new_from_rgb_hex('7cd0ac'))  # ba
  atari_colors.append(sRGBColor.new_from_rgb_hex('90e4c0'))  # bc
  atari_colors.append(sRGBColor.new_from_rgb_hex('a4fcd4'))  # be
  atari_colors.append(sRGBColor.new_from_rgb_hex('003c00'))  # c0
  atari_colors.append(sRGBColor.new_from_rgb_hex('205c20'))  # c2
  atari_colors.append(sRGBColor.new_from_rgb_hex('407c40'))  # c4
  atari_colors.append(sRGBColor.new_from_rgb_hex('5c9c5c'))  # c6
  atari_colors.append(sRGBColor.new_from_rgb_hex('74b474'))  # c8
  atari_colors.append(sRGBColor.new_from_rgb_hex('8cd08c'))  # ca
  atari_colors.append(sRGBColor.new_from_rgb_hex('a4e4a4'))  # cc
  atari_colors.append(sRGBColor.new_from_rgb_hex('b8fcb8'))  # ce
  atari_colors.append(sRGBColor.new_from_rgb_hex('143800'))  # d0
  atari_colors.append(sRGBColor.new_from_rgb_hex('345c1c'))  # d2
  atari_colors.append(sRGBColor.new_from_rgb_hex('507c38'))  # d4
  atari_colors.append(sRGBColor.new_from_rgb_hex('6c9850'))  # d6
  atari_colors.append(sRGBColor.new_from_rgb_hex('84b468'))  # d8
  atari_colors.append(sRGBColor.new_from_rgb_hex('9ccc7c'))  # da
  atari_colors.append(sRGBColor.new_from_rgb_hex('b4e490'))  # dc
  atari_colors.append(sRGBColor.new_from_rgb_hex('c8fca4'))  # de
  atari_colors.append(sRGBColor.new_from_rgb_hex('2c3000'))  # e0
  atari_colors.append(sRGBColor.new_from_rgb_hex('4c501c'))  # e2
  atari_colors.append(sRGBColor.new_from_rgb_hex('687034'))  # e4
  atari_colors.append(sRGBColor.new_from_rgb_hex('848c4c'))  # e6
  atari_colors.append(sRGBColor.new_from_rgb_hex('9ca864'))  # e8
  atari_colors.append(sRGBColor.new_from_rgb_hex('b4c078'))  # ea
  atari_colors.append(sRGBColor.new_from_rgb_hex('ccd488'))  # ec
  atari_colors.append(sRGBColor.new_from_rgb_hex('e0ec9c'))  # ee
  atari_colors.append(sRGBColor.new_from_rgb_hex('442800'))  # f0
  atari_colors.append(sRGBColor.new_from_rgb_hex('644818'))  # f2
  atari_colors.append(sRGBColor.new_from_rgb_hex('846830'))  # f4
  atari_colors.append(sRGBColor.new_from_rgb_hex('a08444'))  # f6
  atari_colors.append(sRGBColor.new_from_rgb_hex('b89c58'))  # f8
  atari_colors.append(sRGBColor.new_from_rgb_hex('d0b46c'))  # fa
  atari_colors.append(sRGBColor.new_from_rgb_hex('e8cc7c'))  # fc
  atari_colors.append(sRGBColor.new_from_rgb_hex('fce08c'))  # fe
  return atari_colors

def main(argv):
  atari_colors_rgb = build_atari_colors_list()
  atari_colors_lab = []
  for color_rgb in atari_colors_rgb:
    atari_colors_lab.append(convert_color(color_rgb, LabColor))

  output_file = file('../src/color_table.cc', 'w')
  output_file.write(
"""// generated file, do not edit. edit make_color_tables.py instead!
#include "color_table.h"

namespace vcsmc {

// Index is atari_color / 2, 0xAABBGGRR
const uint32 kAtariNTSCABGRColorTable[128] = {
""")

  # build ABGR uint32 hex strings
  rgb_strings = []
  for color_rgb in atari_colors_rgb:
    rgb = color_rgb.get_upscaled_value_tuple()
    rgb_strings.append('  0xff%02x%02x%02x' % (rgb[2], rgb[1], rgb[0]))

  output_file.write(',\n'.join(rgb_strings))
  output_file.write("""
};

const float kAtariNTSCLabColorTable[128 * 4] = {
""")
  lab_strings = []
  for color_lab in atari_colors_lab:
    lab_strings.append('  %f, %f, %f, 1.0' % (color_lab.lab_l, color_lab.lab_a, color_lab.lab_b))

  output_file.write(',\n'.join(lab_strings))
  output_file.write("""
};

}  // namespace vcsmc
""")
  output_file.close()

if __name__ == "__main__":
    main(sys.argv)

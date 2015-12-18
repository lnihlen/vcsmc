# Script to make the .cc tables of the Atari colors in RGB and L*ab color
# spaces. Also generates the test data used in the OpenCL conversion test data.
# relies on the colormath module

import sys

from colormath.color_conversions import convert_color
from colormath.color_objects import LabColor
from colormath.color_objects import sRGBColor

def build_atari_colors_list():
  atari_colors = []
  # new values derived from default Stella color palette, for ease of comparison
  # to their colors.
  atari_colors.append(sRGBColor.new_from_rgb_hex('000000')) # 00
  atari_colors.append(sRGBColor.new_from_rgb_hex('4a4a4a')) # 02
  atari_colors.append(sRGBColor.new_from_rgb_hex('6f6f6f')) # 04
  atari_colors.append(sRGBColor.new_from_rgb_hex('8e8e8e')) # 06
  atari_colors.append(sRGBColor.new_from_rgb_hex('aaaaaa')) # 08
  atari_colors.append(sRGBColor.new_from_rgb_hex('c0c0c0')) # 0a
  atari_colors.append(sRGBColor.new_from_rgb_hex('d6d6d6')) # 0c
  atari_colors.append(sRGBColor.new_from_rgb_hex('ececec')) # 0e
  atari_colors.append(sRGBColor.new_from_rgb_hex('484800')) # 10
  atari_colors.append(sRGBColor.new_from_rgb_hex('69690f')) # 12
  atari_colors.append(sRGBColor.new_from_rgb_hex('86861d')) # 14
  atari_colors.append(sRGBColor.new_from_rgb_hex('a2a22a')) # 16
  atari_colors.append(sRGBColor.new_from_rgb_hex('bbbb35')) # 18
  atari_colors.append(sRGBColor.new_from_rgb_hex('d2d240')) # 1a
  atari_colors.append(sRGBColor.new_from_rgb_hex('e8e84a')) # 1c
  atari_colors.append(sRGBColor.new_from_rgb_hex('fcfc54')) # 1e
  atari_colors.append(sRGBColor.new_from_rgb_hex('7c2c00')) # 20
  atari_colors.append(sRGBColor.new_from_rgb_hex('904811')) # 22
  atari_colors.append(sRGBColor.new_from_rgb_hex('a26221')) # 24
  atari_colors.append(sRGBColor.new_from_rgb_hex('b47a30')) # 26
  atari_colors.append(sRGBColor.new_from_rgb_hex('c3903d')) # 28
  atari_colors.append(sRGBColor.new_from_rgb_hex('d2a44a')) # 2a
  atari_colors.append(sRGBColor.new_from_rgb_hex('dfb755')) # 2c
  atari_colors.append(sRGBColor.new_from_rgb_hex('ecc860')) # 2e
  atari_colors.append(sRGBColor.new_from_rgb_hex('901c00')) # 30
  atari_colors.append(sRGBColor.new_from_rgb_hex('a33915')) # 32
  atari_colors.append(sRGBColor.new_from_rgb_hex('b55328')) # 34
  atari_colors.append(sRGBColor.new_from_rgb_hex('c66c3a')) # 36
  atari_colors.append(sRGBColor.new_from_rgb_hex('d5824a')) # 38
  atari_colors.append(sRGBColor.new_from_rgb_hex('e39759')) # 3a
  atari_colors.append(sRGBColor.new_from_rgb_hex('f0aa67')) # 3c
  atari_colors.append(sRGBColor.new_from_rgb_hex('fcbc74')) # 3e
  atari_colors.append(sRGBColor.new_from_rgb_hex('940000')) # 40
  atari_colors.append(sRGBColor.new_from_rgb_hex('a71a1a')) # 42
  atari_colors.append(sRGBColor.new_from_rgb_hex('b83232')) # 44
  atari_colors.append(sRGBColor.new_from_rgb_hex('c84848')) # 46
  atari_colors.append(sRGBColor.new_from_rgb_hex('d65c5c')) # 48
  atari_colors.append(sRGBColor.new_from_rgb_hex('e46f6f')) # 4a
  atari_colors.append(sRGBColor.new_from_rgb_hex('f08080')) # 4c
  atari_colors.append(sRGBColor.new_from_rgb_hex('fc9090')) # 4e
  atari_colors.append(sRGBColor.new_from_rgb_hex('840064')) # 50
  atari_colors.append(sRGBColor.new_from_rgb_hex('97197a')) # 52
  atari_colors.append(sRGBColor.new_from_rgb_hex('a8308f')) # 54
  atari_colors.append(sRGBColor.new_from_rgb_hex('b846a2')) # 56
  atari_colors.append(sRGBColor.new_from_rgb_hex('c659b3')) # 58
  atari_colors.append(sRGBColor.new_from_rgb_hex('d46cc3')) # 5a
  atari_colors.append(sRGBColor.new_from_rgb_hex('e07cd2')) # 5c
  atari_colors.append(sRGBColor.new_from_rgb_hex('ec8ce0')) # 5e
  atari_colors.append(sRGBColor.new_from_rgb_hex('500084')) # 60
  atari_colors.append(sRGBColor.new_from_rgb_hex('68199a')) # 62
  atari_colors.append(sRGBColor.new_from_rgb_hex('7d30ad')) # 64
  atari_colors.append(sRGBColor.new_from_rgb_hex('9246c0')) # 66
  atari_colors.append(sRGBColor.new_from_rgb_hex('a459d0')) # 68
  atari_colors.append(sRGBColor.new_from_rgb_hex('b56ce0')) # 6a
  atari_colors.append(sRGBColor.new_from_rgb_hex('c57cee')) # 6c
  atari_colors.append(sRGBColor.new_from_rgb_hex('d48cfc')) # 6e
  atari_colors.append(sRGBColor.new_from_rgb_hex('140090')) # 70
  atari_colors.append(sRGBColor.new_from_rgb_hex('331aa3')) # 72
  atari_colors.append(sRGBColor.new_from_rgb_hex('4e32b5')) # 74
  atari_colors.append(sRGBColor.new_from_rgb_hex('6848c6')) # 76
  atari_colors.append(sRGBColor.new_from_rgb_hex('7f5cd5')) # 78
  atari_colors.append(sRGBColor.new_from_rgb_hex('956fe3')) # 7a
  atari_colors.append(sRGBColor.new_from_rgb_hex('a980f0')) # 7c
  atari_colors.append(sRGBColor.new_from_rgb_hex('bc90fc')) # 7e
  atari_colors.append(sRGBColor.new_from_rgb_hex('000094')) # 80
  atari_colors.append(sRGBColor.new_from_rgb_hex('181aa7')) # 82
  atari_colors.append(sRGBColor.new_from_rgb_hex('2d32b8')) # 84
  atari_colors.append(sRGBColor.new_from_rgb_hex('4248c8')) # 86
  atari_colors.append(sRGBColor.new_from_rgb_hex('545cd6')) # 88
  atari_colors.append(sRGBColor.new_from_rgb_hex('656fe4')) # 8a
  atari_colors.append(sRGBColor.new_from_rgb_hex('7580f0')) # 8c
  atari_colors.append(sRGBColor.new_from_rgb_hex('8490fc')) # 8e
  atari_colors.append(sRGBColor.new_from_rgb_hex('001c88')) # 90
  atari_colors.append(sRGBColor.new_from_rgb_hex('183b9d')) # 92
  atari_colors.append(sRGBColor.new_from_rgb_hex('2d57b0')) # 94
  atari_colors.append(sRGBColor.new_from_rgb_hex('4272c2')) # 96
  atari_colors.append(sRGBColor.new_from_rgb_hex('548ad2')) # 98
  atari_colors.append(sRGBColor.new_from_rgb_hex('65a0e1')) # 9a
  atari_colors.append(sRGBColor.new_from_rgb_hex('75b5ef')) # 9c
  atari_colors.append(sRGBColor.new_from_rgb_hex('84c8fc')) # 9e
  atari_colors.append(sRGBColor.new_from_rgb_hex('003064')) # a0
  atari_colors.append(sRGBColor.new_from_rgb_hex('185080')) # a2
  atari_colors.append(sRGBColor.new_from_rgb_hex('2d6d98')) # a4
  atari_colors.append(sRGBColor.new_from_rgb_hex('4288b0')) # a6
  atari_colors.append(sRGBColor.new_from_rgb_hex('54a0c5')) # a8
  atari_colors.append(sRGBColor.new_from_rgb_hex('65b7d9')) # aa
  atari_colors.append(sRGBColor.new_from_rgb_hex('75cceb')) # ac
  atari_colors.append(sRGBColor.new_from_rgb_hex('84e0fc')) # ae
  atari_colors.append(sRGBColor.new_from_rgb_hex('004030')) # b0
  atari_colors.append(sRGBColor.new_from_rgb_hex('18624e')) # b2
  atari_colors.append(sRGBColor.new_from_rgb_hex('2d8169')) # b4
  atari_colors.append(sRGBColor.new_from_rgb_hex('429e82')) # b6
  atari_colors.append(sRGBColor.new_from_rgb_hex('54b899')) # b8
  atari_colors.append(sRGBColor.new_from_rgb_hex('65d1ae')) # ba
  atari_colors.append(sRGBColor.new_from_rgb_hex('75e7c2')) # bc
  atari_colors.append(sRGBColor.new_from_rgb_hex('84fcd4')) # be
  atari_colors.append(sRGBColor.new_from_rgb_hex('004400')) # c0
  atari_colors.append(sRGBColor.new_from_rgb_hex('1a661a')) # c2
  atari_colors.append(sRGBColor.new_from_rgb_hex('328432')) # c4
  atari_colors.append(sRGBColor.new_from_rgb_hex('48a048')) # c6
  atari_colors.append(sRGBColor.new_from_rgb_hex('5cba5c')) # c8
  atari_colors.append(sRGBColor.new_from_rgb_hex('6fd26f')) # ca
  atari_colors.append(sRGBColor.new_from_rgb_hex('80e880')) # cc
  atari_colors.append(sRGBColor.new_from_rgb_hex('90fc90')) # ce
  atari_colors.append(sRGBColor.new_from_rgb_hex('143c00')) # d0
  atari_colors.append(sRGBColor.new_from_rgb_hex('355f18')) # d2
  atari_colors.append(sRGBColor.new_from_rgb_hex('527e2d')) # d4
  atari_colors.append(sRGBColor.new_from_rgb_hex('6e9c42')) # d6
  atari_colors.append(sRGBColor.new_from_rgb_hex('87b754')) # d8
  atari_colors.append(sRGBColor.new_from_rgb_hex('9ed065')) # da
  atari_colors.append(sRGBColor.new_from_rgb_hex('b4e775')) # dc
  atari_colors.append(sRGBColor.new_from_rgb_hex('c8fc84')) # de
  atari_colors.append(sRGBColor.new_from_rgb_hex('303800')) # e0
  atari_colors.append(sRGBColor.new_from_rgb_hex('505916')) # e2
  atari_colors.append(sRGBColor.new_from_rgb_hex('6d762b')) # e4
  atari_colors.append(sRGBColor.new_from_rgb_hex('88923e')) # e6
  atari_colors.append(sRGBColor.new_from_rgb_hex('a0ab4f')) # e8
  atari_colors.append(sRGBColor.new_from_rgb_hex('b7c25f')) # ea
  atari_colors.append(sRGBColor.new_from_rgb_hex('ccd86e')) # ec
  atari_colors.append(sRGBColor.new_from_rgb_hex('e0ec7c')) # ee
  atari_colors.append(sRGBColor.new_from_rgb_hex('482c00')) # f0
  atari_colors.append(sRGBColor.new_from_rgb_hex('694d14')) # f2
  atari_colors.append(sRGBColor.new_from_rgb_hex('866a26')) # f4
  atari_colors.append(sRGBColor.new_from_rgb_hex('a28638')) # f6
  atari_colors.append(sRGBColor.new_from_rgb_hex('bb9f47')) # f8
  atari_colors.append(sRGBColor.new_from_rgb_hex('d2b656')) # fa
  atari_colors.append(sRGBColor.new_from_rgb_hex('e8cc63')) # fc
  atari_colors.append(sRGBColor.new_from_rgb_hex('fce070')) # fe
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

const double kAtariNTSCLabColorTable[128 * 4] = {
""")
  lab_strings = []
  for color_lab in atari_colors_lab:
    lab_strings.append('  %.32g, %.32g, %.32g, 1.0' % (color_lab.lab_l, color_lab.lab_a, color_lab.lab_b))

  output_file.write(',\n'.join(lab_strings))
  output_file.write("""
};

}  // namespace vcsmc
""")
  output_file.close()

if __name__ == "__main__":
    main(sys.argv)

#include "atari_ntsc_rgb_color_table.h"

namespace vcsmc {

const uint8 kAtariNtscRedColorTable[128] = {
  0x00, 0x4a, 0x6f, 0x8e, 0xaa, 0xc0, 0xd6, 0xec,
  0x48, 0x69, 0x86, 0xa2, 0xbb, 0xd2, 0xe8, 0xfc,
  0x7c, 0x90, 0xa2, 0xb4, 0xc3, 0xd2, 0xdf, 0xec,
  0x90, 0xa3, 0xb5, 0xc6, 0xd5, 0xe3, 0xf0, 0xfc,
  0x94, 0xa7, 0xb8, 0xc8, 0xd6, 0xe4, 0xf0, 0xfc,
  0x84, 0x97, 0xa8, 0xb8, 0xc6, 0xd4, 0xe0, 0xec,
  0x50, 0x68, 0x7d, 0x92, 0xa4, 0xb5, 0xc5, 0xd4,
  0x14, 0x33, 0x4e, 0x68, 0x7f, 0x95, 0xa9, 0xbc,
  0x00, 0x18, 0x2d, 0x42, 0x54, 0x65, 0x75, 0x84,
  0x00, 0x18, 0x2d, 0x42, 0x54, 0x65, 0x75, 0x84,
  0x00, 0x18, 0x2d, 0x42, 0x54, 0x65, 0x75, 0x84,
  0x00, 0x18, 0x2d, 0x42, 0x54, 0x65, 0x75, 0x84,
  0x00, 0x1a, 0x32, 0x48, 0x5c, 0x6f, 0x80, 0x90,
  0x14, 0x35, 0x52, 0x6e, 0x87, 0x9e, 0xb4, 0xc8,
  0x30, 0x50, 0x6d, 0x88, 0xa0, 0xb7, 0xcc, 0xe0,
  0x48, 0x69, 0x86, 0xa2, 0xbb, 0xd2, 0xe8, 0xfc
};

const uint8 kAtariNtscGreenColorTable[128] = {
  0x00, 0x4a, 0x6f, 0x8e, 0xaa, 0xc0, 0xd6, 0xec,
  0x48, 0x69, 0x86, 0xa2, 0xbb, 0xd2, 0xe8, 0xfc,
  0x2c, 0x48, 0x62, 0x7a, 0x90, 0xa4, 0xb7, 0xc8,
  0x1c, 0x39, 0x53, 0x6c, 0x82, 0x97, 0xaa, 0xbc,
  0x00, 0x1a, 0x32, 0x48, 0x5c, 0x6f, 0x80, 0x90,
  0x00, 0x19, 0x30, 0x46, 0x59, 0x6c, 0x7c, 0x8c,
  0x00, 0x19, 0x30, 0x46, 0x59, 0x6c, 0x7c, 0x8c,
  0x00, 0x1a, 0x32, 0x48, 0x5c, 0x6f, 0x80, 0x90,
  0x00, 0x1a, 0x32, 0x48, 0x5c, 0x6f, 0x80, 0x90,
  0x1c, 0x3b, 0x57, 0x72, 0x8a, 0xa0, 0xb5, 0xc8,
  0x30, 0x50, 0x6d, 0x88, 0xa0, 0xb7, 0xcc, 0xe0,
  0x40, 0x62, 0x81, 0x9e, 0xb8, 0xd1, 0xe7, 0xfc,
  0x44, 0x66, 0x84, 0xa0, 0xba, 0xd2, 0xe8, 0xfc,
  0x3c, 0x5f, 0x7e, 0x9c, 0xb7, 0xd0, 0xe7, 0xfc,
  0x38, 0x59, 0x76, 0x92, 0xab, 0xc2, 0xd8, 0xec,
  0x2c, 0x4d, 0x6a, 0x86, 0x9f, 0xb6, 0xcc, 0xe0
};

const uint8 kAtariNtscBlueColorTable[128] = {
  0x00, 0x4a, 0x6f, 0x8e, 0xaa, 0xc0, 0xd6, 0xec,
  0x00, 0x0f, 0x1d, 0x2a, 0x35, 0x40, 0x4a, 0x54,
  0x00, 0x11, 0x21, 0x30, 0x3d, 0x4a, 0x55, 0x60,
  0x00, 0x15, 0x28, 0x3a, 0x4a, 0x59, 0x67, 0x74,
  0x00, 0x1a, 0x32, 0x48, 0x5c, 0x6f, 0x80, 0x90,
  0x64, 0x7a, 0x8f, 0xa2, 0xb3, 0xc3, 0xd2, 0xe0,
  0x84, 0x9a, 0xad, 0xc0, 0xd0, 0xe0, 0xee, 0xfc,
  0x90, 0xa3, 0xb5, 0xc6, 0xd5, 0xe3, 0xf0, 0xfc,
  0x94, 0xa7, 0xb8, 0xc8, 0xd6, 0xe4, 0xf0, 0xfc,
  0x88, 0x9d, 0xb0, 0xc2, 0xd2, 0xe1, 0xef, 0xfc,
  0x64, 0x80, 0x98, 0xb0, 0xc5, 0xd9, 0xeb, 0xfc,
  0x30, 0x4e, 0x69, 0x82, 0x99, 0xae, 0xc2, 0xd4,
  0x00, 0x1a, 0x32, 0x48, 0x5c, 0x6f, 0x80, 0x90,
  0x00, 0x18, 0x2d, 0x42, 0x54, 0x65, 0x75, 0x84,
  0x00, 0x16, 0x2b, 0x3e, 0x4f, 0x5f, 0x6e, 0x7c,
  0x00, 0x14, 0x26, 0x38, 0x47, 0x56, 0x63, 0x70
};

}  // namespace vcsmc

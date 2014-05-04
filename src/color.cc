#include "color.h"

#include <unordered_map>

namespace vcsmc {

// Values cribbed from http://www.qotile.net/minidig/docs/tia_color.html
// Index is atari_color / 2, 0xAABBGGRR
static const uint32_t kAtariNTSCColorTable[128] = {
  0xff000000, 0xff404040, 0xff6c6c6c, 0xff909090,
  0xffb0b0b0, 0xffc8c8c8, 0xffdcdcdc, 0xffececec,

  0xff004444, 0xff106464, 0xff248484, 0xff34a0a0,
  0xff40b8b8, 0xff50d0d0, 0xff5ce8e8, 0xff68fcfc,

  0xff002870, 0xff144484, 0xff285c98, 0xff3c78ac,
  0xff4c8cbc, 0xff5ca0cc, 0xff68b4dc, 0xff78c8ec,

  0xff001884, 0xff183498, 0xff3050ac, 0xff4868c0,
  0xff5c80d0, 0xff7094e0, 0xff80a8ec, 0xff94bcfc,

  0xff000088, 0xff20209c, 0xff3c3cb0, 0xff5858c0,
  0xff7070d0, 0xff8888e0, 0xffa0a0ec, 0xffb4b4fc,

  0xff5c0078, 0xff74208c, 0xff883ca0, 0xff9c58b0,
  0xffb070c0, 0xffc084d0, 0xffd09cdc, 0xffe0b0ec,

  0xff780048, 0xff902060, 0xffa43c78, 0xffb8588c,
  0xffcc70a0, 0xffdc84b4, 0xffec9cc4, 0xfffcb0d4,

  0xff840014, 0xff982030, 0xffac3c4c, 0xffc05868,
  0xffd0707c, 0xffe08894, 0xffeca0a8, 0xfffcb4bc,

  0xff880000, 0xff9c201c, 0xffb04038, 0xffc05c50,
  0xffd07468, 0xffe08c7c, 0xffeca490, 0xfffcb8a4,

  0xff7c1700, 0xff90381c, 0xffa85438, 0xffbc7050,
  0xffcc8868, 0xffdc9c7c, 0xffecb490, 0xfffcc8a4,

  0xff6c2c00, 0xff784c1c, 0xff906838, 0xffac8450,
  0xffc09c68, 0xffd4b47c, 0xffe8cc90, 0xfffce0a4,

  0xff2c3c00, 0xff485c1c, 0xff647c38, 0xff809c50,
  0xff94b468, 0xffacd07c, 0xffc0e490, 0xffd4fca4,

  0xff003c00, 0xff205c20, 0xff407c40, 0xff5c9c5c,
  0xff74b474, 0xff8cd08c, 0xffa4e4a4, 0xffb8fcb8,

  0xff003814, 0xff1c5c34, 0xff387c50, 0xff50986c,
  0xff68b484, 0xff7ccc9c, 0xff90e4b4, 0xffa4fcc8,

  0xff00302c, 0xff1c504c, 0xff347068, 0xff4c8c84,
  0xff64a89c, 0xff78c0b4, 0xff88d4cc, 0xff9cece0,

  0xff002844, 0xff184864, 0xff306884, 0xff4484a0,
  0xff589cb8, 0xff6cb4d0, 0xff7ccce8, 0xff8ce0fc
};

static std::unordered_map<uint32, uint8> AtariNTSCColorMap;

// static
uint32 Color::AtariColorToABGR(uint8 atari_color) {
  return kAtariNTSCColorTable[atari_color / 2];
}

// static
uint8 Color::ABGRToAtariColor(uint32 abgr) {
  // Check to see if color map is unititalized
  if (!AtariNTSCColorMap.size()) {
    for (uint8 i = 0; i < 128; ++i) {
      AtariNTSCColorMap.insert(
          std::make_pair(kAtariNTSCColorTable[i], (i * 2)));
    }
  }

  // Set all alpha bits to 0xff as we don't worry about transparent colors.
  uint32 bgr = abgr | 0xff000000;

  // Look for element in our color map
  std::unordered_map<uint32, uint8>::const_iterator it =
      AtariNTSCColorMap.find(bgr);

  if (it != AtariNTSCColorMap.end()) {
    return it->second;
  }

  // No exact color match found, search through table for minimum cartesian distance
  // in ARGB color space.
  uint8_t best_match_index = 0;
  double minimum_distance = CartesianDistanceSquaredABGR(
      bgr, kAtariNTSCColorTable[0]);

  for (uint8_t i = 1; i < 128; ++i) {
    double distance = CartesianDistanceSquaredABGR(
        bgr, kAtariNTSCColorTable[i]);

    if (distance < minimum_distance) {
      best_match_index = i;
      minimum_distance = distance;
    }
  }

  return best_match_index * 2;
}

// static
double Color::CartesianDistanceSquaredABGR(uint32_t a, uint32_t b) {
  double a_b = static_cast<double>((a >> 16) & 0x000000ff);
  double a_g = static_cast<double>((a >> 8) & 0x000000ff);
  double a_r = static_cast<double>(a & 0x000000ff);
  double b_b = static_cast<double>((b >> 16) & 0x000000ff);
  double b_g = static_cast<double>((b >> 8) & 0x000000ff);
  double b_r = static_cast<double>(b & 0x000000ff);

  double d_b = a_b - b_b;
  double d_g = a_g - b_g;
  double d_r = a_r - b_r;

  return (d_b * d_b) + (d_g * d_g) + (d_r * d_r);
}

}  // namespace vcsmc

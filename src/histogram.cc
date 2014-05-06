#include "histogram.h"

#include <algorithm>
#include <unordered_map>

#include "color.h"
#include "constants.h"
#include "pixel_strip.h"

namespace vcsmc {

Histogram::Histogram(PixelStrip* pixel_strip) {
  std::unordered_map<uint32, uint32> color_map;

  for (uint32 i = 0; i < pixel_strip->width(); ++i) {
    uint32 color = pixel_strip->pixel(i);
    std::unordered_map<uint32, uint32>::iterator it = color_map.find(color);
    if (it != color_map.end()) {
      ++it->second;
    } else {
      color_map[color] = 1;
    }
  }

  // Seed vector with indicies and non-zero counts.
  for (std::unordered_map<uint32, uint32>::iterator it = color_map.begin();
       it != color_map.end(); ++it) {
    color_counts_.emplace_back(it->second, it->first);
  }

  // Sorting pairs sorts by first and then second.
  std::sort(color_counts_.begin(), color_counts_.end());
  // Reverse sort order so the highest counts are first.
  std::reverse(color_counts_.begin(), color_counts_.end());
}

double Histogram::DistanceFrom(uint32 color) {
  double accum = 0;
  for (ColorCounts::iterator i = color_counts_.begin();
       i != color_counts_.end(); ++i) {
    accum += static_cast<double>(i->first) *
        Color::CartesianDistanceSquaredABGR(i->second, color);
  }
  return accum;
}

}  // namespace vcsmc

#include "histogram.h"

#include <algorithm>
#include <cstring>
#include <map>

#include "constants.h"

namespace vcsmc {

Histogram::Histogram(ColuStrip* colu_strip) {
  // Build histo from all possible values of 1-byte colu.
  uint32 histo[256];
  std::memset(histo, 0, sizeof(histo));
  for (uint32 i = 0; i < colu_strip->width(); ++i) {
    ++histo[colu_strip->colu(i)];
  }

  // Seed vector with indicies and non-zero counts.
  for (uint32 i = 0; i < 256; ++i) {
    if (histo[i]) {
      colu_counts_.emplace_back(histo[i], static_cast<uint8>(i));
    }
  }

  // Sorting pairs sorts by first and then second.
  std::sort(colu_counts_.begin(), colu_counts_.end());
  // Reverse sort order so the highest counts are first.
  std::reverse(colu_counts_.begin(), colu_counts_.end());
}

}  // namespace vcsmc

#ifndef SRC_HISTOGRAM_H_
#define SRC_HISTOGRAM_H_

#include <vector>

#include "types.h"

namespace vcsmc {

class PixelStrip;

// A Histogram takes either an input ColuStrip or Frame and computes a frequency
// of colu values.
class Histogram {
 public:
  // Compute from a single strip of color.
  Histogram(PixelStrip* pixel_strip);

  const uint32 unique_colors() const { return colu_counts_.size(); }
  // Returns the ith most frequent color, for i in [0, unique_colors())
  const uint32 colu(uint32 i) const { return colu_counts_[i].second; }
  // Returns the count of the ith most frequent color.
  const uint32 count(uint32 i) const { return colu_counts_[i].first; }

 private:
  // Sorted vector of ordered pairs of (count, colu value). Only elements with
  // non-zero counts are retained.
  std::vector<std::pair<uint32, uint32>> colu_counts_;
};

}  // namespace vcsmc

#endif  // SRC_HISTOGRAM_H_

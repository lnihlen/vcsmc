#ifndef SRC_HISTOGRAM_H_
#define SRC_HISTOGRAM_H_

#include <vector>

#include "colu_strip.h"
#include "frame.h"

namespace vcsmc {

// A Histogram takes either an input ColuStrip or Frame and computes a frequency
// of colu values.
class Histogram {
 public:
  // Compute from a single strip of color.
  Histogram(ColuStrip* colu_strip);

  const uint32 unique_colors() const { return colu_counts_.size(); }
  // Returns the ith most frequent color, for i in [0, unique_colors())
  const uint8 colu(uint32 i) const { return colu_counts_[i].second; }
  // Returns the count of the ith most frequent color.
  const uint32 count(uint32 i) const { return colu_counts_[i].first; }

 private:
  // Sorted vector of ordered pairs of (count, colu value). Only elements with
  // non-zero counts are retained.
  std::vector<std::pair<uint32, uint8>> colu_counts_;
};

}  // namespace vcsmc

#endif  // SRC_HISTOGRAM_H_

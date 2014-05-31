#ifndef SRC_RANGE_H_
#define SRC_RANGE_H_

#include "types.h"

namespace vcsmc {

// A Range is a simple value container for a pair of uint32 color_clock times.
// The range is meant to indicate [start_time, end_time). There are assertions
// to prevent negative-duration ranges, that is end_time < start_time.
class Range {
 public:
  // Makes an empty range, no duration and start time of zero.
  Range();
  Range(uint32 start_time, uint32 end_time);
  Range(const Range& range);
  const Range& operator=(const Range& range);

  const bool Contains(uint32 time) const {
    return start_time_ <= time && time < end_time_;
  }
  const uint32 Duration() const { return end_time_ - start_time_; }
  const bool IsEmpty() const { return start_time_ == 0 && end_time_ == 0; }

  // Returns the empty range (0, 0) if the intersection is empty.
  static Range IntersectRanges(const Range& r1, const Range& r2);
  // static Range UnionRanges(const Range& r1, const Range& r2);

  void set_start_time(uint32 start_time) { start_time_ = start_time; }
  void set_end_time(uint32 end_time) { end_time_ = end_time; }
  const uint32 start_time() const { return start_time_; }
  const uint32 end_time() const { return end_time_; }

 private:
  uint32 start_time_;
  uint32 end_time_;
};

}  // namespace vcsmc

#endif  // SRC_RANGE_H_

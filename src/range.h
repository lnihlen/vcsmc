#ifndef SRC_RANGE_H_
#define SRC_RANGE_H_

#include "types.h"

namespace vcsmc {

// A Range is a simple value container for a pair of uint64 color_clock times.
// The range is meant to indicate [start_time, end_time). There are assertions
// to prevent negative-duration ranges, that is end_time < start_time.
class Range {
 public:
  // Makes an empty range, no duration and start time of zero.
  Range();
  Range(uint64 start_time, uint64 end_time);
  Range(const Range& range);
  const Range& operator=(const Range& range);
  const bool operator==(const Range& range) const;
  const bool operator!=(const Range& range) const;

  const bool Contains(uint64 time) const {
    return start_time_ <= time && time < end_time_;
  }
  const uint64 Duration() const { return end_time_ - start_time_; }
  const bool IsEmpty() const { return start_time_ == end_time_; }

  // Returns the empty range (0, 0) if the intersection is empty.
  static Range IntersectRanges(const Range& r1, const Range& r2);
  // static Range UnionRanges(const Range& r1, const Range& r2);

  // Saves the range as two 64-bit numbers (begin, end) in the supplied buffer.
  // Returns the number of bytes saved, or 16 bytes.
  size_t Serialize(uint8* buffer);

  void set_start_time(uint64 start_time);
  void set_end_time(uint64 end_time);
  const uint64 start_time() const { return start_time_; }
  const uint64 end_time() const { return end_time_; }

 private:
  uint64 start_time_;
  uint64 end_time_;
};

}  // namespace vcsmc

#endif  // SRC_RANGE_H_

#include "range.h"

#include <algorithm>
#include <cassert>

namespace vcsmc {

Range::Range()
  : start_time_(0),
    end_time_(0) {
}

Range::Range(uint32 start_time, uint32 end_time)
  : start_time_(start_time),
    end_time_(end_time) {
  assert(end_time_ >= start_time_);
}

Range::Range(const Range& range)
  : start_time_(range.start_time_),
    end_time_(range.end_time_) {
}

const Range& Range::operator=(const Range& range) {
  start_time_ = range.start_time_;
  end_time_ = range.end_time_;
  return *this;
}

const bool Range::operator==(const Range& range) const {
  return (start_time() == range.start_time()) &&
      (end_time() == range.end_time());
}

const bool Range::operator!=(const Range& range) const {
  return (start_time() != range.start_time()) ||
      (end_time() != range.end_time());
}

// static
Range Range::IntersectRanges(const Range& r1, const Range& r2) {
  uint32 start_time = std::max(r1.start_time(), r2.start_time());
  uint32 end_time = std::min(r1.end_time(), r2.end_time());
  if (start_time > end_time)
    return Range();
  return Range(start_time, end_time);
}

size_t Range::Serialize(uint8* buffer) {
  uint32* buffer_long = reinterpret_cast<uint32*>(buffer);
  buffer_long[0] = start_time_;
  buffer_long[1] = end_time_;
  return 2 * sizeof(uint32);
}

// static
Range Range::Deserialize(const uint8* buffer, size_t* bytes_read_out) {
  const uint32* buffer_long = reinterpret_cast<const uint32*>(buffer);
  Range range(buffer_long[0], buffer_long[1]);
  *bytes_read_out = 2 * sizeof(uint32);
  return range;
}

void Range::set_start_time(uint32 start_time) {
  assert(end_time_ >= start_time);
  start_time_ = start_time;
}

void Range::set_end_time(uint32 end_time) {
  assert(end_time >= start_time_);
  end_time_ = end_time;
}

}  // namespace vcsmc

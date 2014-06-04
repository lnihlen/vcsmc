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

// static
Range Range::IntersectRanges(const Range& r1, const Range& r2) {
  uint32 start_time = std::max(r1.start_time(), r2.start_time());
  uint32 end_time = std::min(r1.end_time(), r2.end_time());
  if (start_time > end_time)
    return Range();
  return Range(start_time, end_time);
}

void Range::SetStartTime(uint32 start_time) {
  assert(end_time_ >= start_time);
  start_time_ = start_time;
}

void Range::SetEndTime(uint32 end_time) {
  assert(end_time >= start_time_);
  end_time_ = end_time;
}

}  // namespace vcsmc

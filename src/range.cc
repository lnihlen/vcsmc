#include "range.h"

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

}  // namespace vcsmc

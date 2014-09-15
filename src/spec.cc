#include "spec.h"

namespace vcsmc {

Spec::Spec(TIA tia, uint8 value, const Range& range)
  : tia_(tia),
    value_(value),
    range_(range) {
}

size_t Spec::Serialize(uint8* buffer) {
  size_t range_size = range_.Serialize(buffer);
  uint8* buffer_after_range = buffer + range_size;
  buffer_after_range[0] = static_cast<uint8>(tia_);
  buffer_after_range[1] = value_;
  return range_size + 2;
}

}  // namespace vcsmc

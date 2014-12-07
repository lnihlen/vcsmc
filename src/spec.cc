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

// static
Spec Spec::Deserialize(const uint8* buffer, size_t* bytes_read_out) {
  size_t spec_read = 0;
  Range range = Range::Deserialize(buffer, &spec_read);
  const uint8* buffer_after_range = buffer + spec_read;
  *bytes_read_out = spec_read + 2;
  return Spec(static_cast<TIA>(buffer_after_range[0]), buffer_after_range[1],
      range);
}

}  // namespace vcsmc

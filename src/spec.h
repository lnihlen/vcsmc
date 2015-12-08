#ifndef SRC_SPEC_H_
#define SRC_SPEC_H_

#include <memory>

#include "constants.h"
#include "range.h"
#include "types.h"

namespace vcsmc {

// A Spec defines a set of instructions and a time range that those instructions
// have to be added to any potential Kernel. This ensures that the randomly
// generated Kernels can continue to be valid frame programs, as well as allows
// individual frames to add audio or other interactive programming bits to
// a class of Kernels, perhaps all representing the unvarying requirements of an
// individual frame of video.
class Spec {
 public:
  // Returns a range in CPU cycles that the spec is required to operate for.
  const Range& range() const { return range_; }
  size_t bytes() const { return bytes_; }
  const uint8* bytecode() const { return bytecode_.get(); }

 private:
  Range range_;
  size_t bytes_;
  std::unique_ptr<uint8[]> bytecode_;
};

}  // namespace vcsmc

#endif  // SRC_SPEC_H_

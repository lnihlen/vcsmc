#ifndef SRC_SPEC_H_
#define SRC_SPEC_H_

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
  // Saves the Spec into the provided buffer. Returns the number of bytes stored
  // in the buffer.
  size_t Serialize(uint8* buffer);

  // Loads a spec from a buffer. Returns the spec, plus the number of bytes read
  // is saved at |bytes_read_out|, currently 10 bytes.
  static Spec Deserialize(const uint8* buffer, size_t* bytes_read_out);

  // Returns a range in CPU cycles that the spec is required to operate for.
  const Range& range() const { return range_; }

 private:
  Range range_;
  size_t bytes_;
  std::unique_ptr<uint8[]> bytecode_;
};

}  // namespace vcsmc

#endif  // SRC_SPEC_H_

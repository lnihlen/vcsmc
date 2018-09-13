#ifndef SRC_SPEC_H_
#define SRC_SPEC_H_

#include <array>
#include <string>
#include <vector>

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
  Spec(const Range& range, size_t size, std::unique_ptr<uint8[]> bytecode);
  Spec(const Spec& spec);
  const Spec& operator=(const Spec& spec);

  // Returns a range in CPU cycles that the spec is required to operate for.
  const Range& range() const { return range_; }
  // Size in bytes.
  size_t size() const { return size_; }
  // Pointer to the bytecode.
  const uint8* bytecode() const { return bytecode_.get(); }

 private:
  Range range_;
  size_t size_;
  std::unique_ptr<uint8[]> bytecode_;
};

typedef std::shared_ptr<std::vector<Spec>> SpecList;
typedef std::vector<Spec>::const_iterator SpecIterator;

}  // namespace vcsmc

#endif  // SRC_SPEC_H_

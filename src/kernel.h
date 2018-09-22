#ifndef SRC_KERNEL_H_
#define SRC_KERNEL_H_

#include <array>

#include "constants.h"
#include "types.h"

namespace vcsmc {

// A Kernel contains 6502 bytecode suitable for executing on an Atari 2600
// system for generation of a single frame of image and audio data. Kernels
// are typically generated by Translating a Genome along with a Spec, and can
// then be simulated and scored against the target image.
class Kernel {
 public:
  Kernel() : size_(0) { bytecode_.fill(0); }

  // Append |byte| to bytecode in this Kernel.
  void Append(uint8 byte) { bytecode_[size_] = byte; ++size_; }

  // Read-only access.
  const uint8* bytecode() const { return bytecode_.data(); }
  size_t size() const { return size_; }

 private:
  std::array<uint8, kMaxKernelSize> bytecode_;
  size_t size_;
};

}  // namespace vcsmc


#endif  // SRC_KERNEL_H_

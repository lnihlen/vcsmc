#ifndef SRC_KERNEL_H_
#define SRC_KERNEL_H_

#include "job.h"
#include "types.h"

namespace vcsmc {

class Kernel;

// Given a pointer to a completely empty Kernel this Job will populate it with
// totally random bytecode.
class GenerateRandomKernelJob : public Job {
 public:
  explicit GenerateRandomKernelJob(shared_ptr<Kernel> kernel)
    : kernel_(kernel) {}
  void Execute() override;

 private:
  shared_ptr<Kernel> kernel_;
};

// A Kernel represents a program in 6502 bytecode for the VCS that generates one
// frame of output imagery when run on the Atari.
class Kernel {
 public:
  // Creates an empty kernel.
  Kernel(shared_ptr<std::vector<Spec>> specs);

  uint8* bytecode();

 private:
  shared_ptr<std::vector<Spec>> specs_;
  // List of dynamic program code that exists between specs.
  std::vector<std::vector<uint32>> opcodes_;
  std::unique_ptr<uint8[]> bytecode_;
  size_t bytecode_size_;
};

} // namespace vcsmc

#endif  // SRC_KERNEL_H_

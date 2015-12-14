#ifndef SRC_KERNEL_H_
#define SRC_KERNEL_H_

#include <memory>
#include <random>

#include "job.h"
#include "range.h"
#include "spec.h"
#include "types.h"

namespace vcsmc {

// A Kernel represents a program in 6502 bytecode for the VCS that generates one
// frame of output imagery when run on the Atari.
class Kernel {
 public:
  // Creates an empty kernel with the provided random seed and specs.
  explicit Kernel(std::seed_seq& seed);

  const uint8* bytecode() const { return bytecode_.get(); }
  size_t bytecode_size() const { return bytecode_size_; }
  uint64 fingerprint() const { return fingerprint_; }

  // Given a pointer to a completely empty Kernel this Job will populate it with
  // totally random bytecode.
  class GenerateRandomKernelJob : public Job {
   public:
    explicit GenerateRandomKernelJob(
        std::shared_ptr<Kernel> kernel, SpecList specs)
      : kernel_(kernel), specs_(specs) {}
    void Execute() override;

   private:
    std::shared_ptr<Kernel> kernel_;
    SpecList specs_;
  };

 private:
  uint32 GenerateRandomOpcode(uint32 cycles_remaining);
  uint32 GenerateRandomLoad();
  uint32 GenerateRandomStore();
  void AppendJmpSpec(uint32 current_cycle, size_t current_bank_size);
  // Given valid data in opcodes_ refills bytecode_ with the concatenated data
  // in opcodes_ and specs_, appends jumps and updates fingerprint_.
  void RegenerateBytecode(size_t bytecode_size);

  std::default_random_engine engine_;
  SpecList specs_;
  // List of dynamic program code that exists between specs.
  std::vector<std::unique_ptr<std::vector<uint32>>> opcodes_;
  std::vector<Range> opcode_ranges_;
  std::unique_ptr<uint8[]> bytecode_;
  size_t bytecode_size_;
  uint64 fingerprint_;
};

} // namespace vcsmc

#endif  // SRC_KERNEL_H_

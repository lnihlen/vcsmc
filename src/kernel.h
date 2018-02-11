#ifndef SRC_KERNEL_H_
#define SRC_KERNEL_H_

#include <cassert>
#include <memory>

#include "tbb/tbb.h"

#include "range.h"
#include "spec.h"
#include "types.h"
#include "tls_prng.h"

namespace vcsmc {

class Kernel;
typedef std::shared_ptr<std::vector<std::shared_ptr<Kernel>>> Generation;

// A Kernel represents a program in 6502 bytecode for the VCS that generates one
// frame of output imagery when run on the Atari.
class Kernel {
 public:
  // Creates an empty kernel.
  Kernel();
  // Creates a kernel that takes on ownership of |specs| and builds out internal
  // state from |dynamic_areas| and |packed_opcodes|.
  Kernel(
      SpecList specs,
      const std::vector<Range>& dynamic_areas,
      const std::vector<std::unique_ptr<uint8[]>>& packed_opcodes);

  // Make a value copy of this Kernel and return a pointer to it. Need to call
  // RegenerateBytecode() before use.
  std::shared_ptr<Kernel> Clone();

  void GenerateRandom(const SpecList specs, TlsPrngList::reference tls_prng);
  void ClobberSpec(const SpecList new_specs);

  // Randomly regenerate one opcode per call. After one or more calls to Mutate,
  // please call RegenerateBytecode() in order to finalize the new binary.
  void Mutate(TlsPrngList::reference engine);
  void RegenerateBytecode();

  const uint8* bytecode() const { return bytecode_.get(); }
  const std::vector<Range>& dynamic_areas() const { return dynamic_areas_; }
  size_t bytecode_size() const { return bytecode_size_; }
  uint64 fingerprint() const { return fingerprint_; }
  const SpecList specs() const { return specs_; }
  const std::vector<std::vector<uint32>>& opcodes() const { return opcodes_; }

 private:
  uint32 GenerateRandomOpcode(uint32 cycles_remaining,
                              TlsPrngList::reference engine);
  uint32 GenerateRandomLoad(TlsPrngList::reference engine);
  uint32 GenerateRandomStore(TlsPrngList::reference engine);
  void AppendJmpSpec(uint32 current_cycle, size_t current_bank_size);
  // Given a number within [0, total_dynamic_opcodes_) returns the index of the
  // vector within opcodes_ that contains this value.
  size_t OpcodeFieldIndex(size_t opcode_index);

  SpecList specs_;
  std::vector<Range> opcode_ranges_;
  std::vector<std::vector<uint32>> opcodes_;

  size_t bytecode_size_ = 0;
  std::vector<Range> dynamic_areas_;
  std::unique_ptr<uint8[]> bytecode_;

  size_t total_dynamic_opcodes_;
  std::vector<size_t> opcode_counts_;

  uint64 fingerprint_ = 0;
};


} // namespace vcsmc

#endif  // SRC_KERNEL_H_

#ifndef SRC_KERNEL_H_
#define SRC_KERNEL_H_

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

  // Save the simulated kernel image to provided png filename.
  bool SaveImage(const std::string& file_name) const;

  void ResetVictories() { victories_ = 0; }
  void AddVictory() { ++victories_; }

  void GenerateRandom(const SpecList specs, TlsPrngList::reference tls_prng);
  void ClobberSpec(const SpecList new_specs);

  const uint8* bytecode() const { return bytecode_.get(); }
  const std::vector<Range>& dynamic_areas() const { return dynamic_areas_; }
  const uint8* sim_frame() const { return sim_frame_.get(); }
  size_t bytecode_size() const { return bytecode_size_; }
  uint64 fingerprint() const { return fingerprint_; }
  bool score_valid() const { return score_valid_; }
  double score() const { return score_; }
  uint32 victories() const { return victories_; }
  const SpecList specs() const { return specs_; }
  const std::vector<std::vector<uint32>>& opcodes() const { return opcodes_; }

  // Given a pointer to a completely empty Kernel this Job will populate it with
  // totally random bytecode.
  class GenerateRandomKernelJob {
   public:
    GenerateRandomKernelJob(Generation generation,
                            SpecList specs,
                            TlsPrngList& tls_prng_list)
        : generation_(generation),
          specs_(specs),
          tls_prng_list_(tls_prng_list) {}
    void operator()(const tbb::blocked_range<size_t>& r) const;

   private:
    Generation generation_;
    const SpecList specs_;
    TlsPrngList& tls_prng_list_;
  };

  // Score an unscored kernel by simulating it and then comparing it to the
  // target lab image.
  class ScoreKernelJob {
   public:
    ScoreKernelJob(Generation generation, const uint8* target_colors)
        : generation_(generation), target_colors_(target_colors) {}
    void operator()(const tbb::blocked_range<size_t>& r) const;

   private:
    Generation generation_;
    const uint8* target_colors_;
  };

  // Given a provided reference kernel, generate the target kernel as a copy of
  // the reference with the provided number of random mutations. Should iterate
  // over the first half of |generation|, which will use that as source material
  // and target the latter half of the array for copy and mutation.
  class MutateKernelJob {
   public:
    MutateKernelJob(Generation source_generation,
                    Generation target_generation,
                    size_t target_index_offset,
                    size_t number_of_mutations,
                    TlsPrngList& tls_prng_list)
        : source_generation_(source_generation),
          target_generation_(target_generation),
          target_index_offset_(target_index_offset),
          number_of_mutations_(number_of_mutations),
          tls_prng_list_(tls_prng_list) {}
    void operator()(const tbb::blocked_range<size_t>& r) const;
   private:
    Generation source_generation_;
    Generation target_generation_;
    size_t target_index_offset_;
    const size_t number_of_mutations_;
    TlsPrngList& tls_prng_list_;
  };

  // Given an existing Kernel and a new list of specs (typically audio) this
  // will clobber the existing specs and regenerate the bytecode.
  class ClobberSpecJob {
   public:
    ClobberSpecJob(Generation generation,
                   SpecList specs)
        : generation_(generation),
          specs_(specs) {}
    void operator()(const tbb::blocked_range<size_t>& r) const;
   private:
    Generation generation_;
    const SpecList specs_;
  };

 private:
  uint32 GenerateRandomOpcode(uint32 cycles_remaining,
      TlsPrngList::reference engine);
  uint32 GenerateRandomLoad(TlsPrngList::reference engine);
  uint32 GenerateRandomStore(TlsPrngList::reference engine);
  void AppendJmpSpec(uint32 current_cycle, size_t current_bank_size);
  // Given valid data in opcodes_ refills bytecode_ with the concatenated data
  // in opcodes_ and specs_, appends jumps and updates fingerprint_.
  void RegenerateBytecode(size_t bytecode_size);
  void SimulateAndScore(const uint8* target_colors);
  void Mutate(TlsPrngList::reference engine);
  // Given a number within [0, total_dynamic_opcodes_) returns the index of the
  // vector within opcodes_ that contains this value.
  size_t OpcodeFieldIndex(size_t opcode_index);

  SpecList specs_;
  std::vector<Range> opcode_ranges_;
  std::vector<std::vector<uint32>> opcodes_;

  size_t bytecode_size_;
  std::vector<Range> dynamic_areas_;
  std::unique_ptr<uint8[]> bytecode_;
  std::unique_ptr<uint8[]> sim_frame_;

  size_t total_dynamic_opcodes_;
  std::vector<size_t> opcode_counts_;

  uint64 fingerprint_;
  bool score_valid_;
  double score_;
  uint32 victories_;
};


} // namespace vcsmc

#endif  // SRC_KERNEL_H_

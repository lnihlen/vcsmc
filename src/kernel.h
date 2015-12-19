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

  // Save the simulated kernel image to provided png filename.
  bool SaveImage(const std::string& file_name) const;

  void ResetVictories() { victories_ = 0; }
  void AddVictory() { ++victories_; }

  const uint8* bytecode() const { return bytecode_.get(); }
  const uint8* sim_frame() const { return sim_frame_.get(); }
  size_t bytecode_size() const { return bytecode_size_; }
  uint64 fingerprint() const { return fingerprint_; }
  bool score_valid() const { return score_valid_; }
  double score() const { return score_; }
  uint32 victories() const { return victories_; }
  const SpecList specs() const { return specs_; }

  // Given a pointer to a completely empty Kernel this Job will populate it with
  // totally random bytecode.
  class GenerateRandomKernelJob : public Job {
   public:
    GenerateRandomKernelJob(std::shared_ptr<Kernel> kernel, SpecList specs)
        : kernel_(kernel), specs_(specs) {}
    void Execute() override;

   private:
    std::shared_ptr<Kernel> kernel_;
    SpecList specs_;
  };

  // Score an unscored kernel by simulating it and then comparing it to the
  // target lab image.
  class ScoreKernelJob : public Job {
   public:
    ScoreKernelJob(std::shared_ptr<Kernel> kernel, const double* target_lab)
        : kernel_(kernel), target_lab_(target_lab) {}
    void Execute() override;

   private:
    std::shared_ptr<Kernel> kernel_;
    const double* target_lab_;
  };

  // Given a provided reference kernel, generate the target kernel as a copy of
  // the reference with the provided number of random mutations.
  class MutateKernelJob : public Job {
    public:
     MutateKernelJob(const std::shared_ptr<Kernel> original,
                     std::shared_ptr<Kernel> target,
                     size_t number_of_mutations)
        : original_(original),
          target_(target),
          number_of_mutations_(number_of_mutations) {}
     void Execute() override;
    private:
     const std::shared_ptr<Kernel> original_;
     std::shared_ptr<Kernel> target_;
     size_t number_of_mutations_;
  };

 private:
  uint32 GenerateRandomOpcode(uint32 cycles_remaining);
  uint32 GenerateRandomLoad();
  uint32 GenerateRandomStore();
  void AppendJmpSpec(uint32 current_cycle, size_t current_bank_size);
  // Given valid data in opcodes_ refills bytecode_ with the concatenated data
  // in opcodes_ and specs_, appends jumps and updates fingerprint_.
  void RegenerateBytecode(size_t bytecode_size);
  void SimulateAndScore(const double* target_lab);
  void Mutate();

  std::default_random_engine engine_;

  SpecList specs_;
  std::vector<std::vector<uint32>> opcodes_;
  std::vector<Range> opcode_ranges_;

  size_t bytecode_size_;
  std::unique_ptr<uint8[]> bytecode_;
  std::unique_ptr<uint8[]> sim_frame_;

  uint64 fingerprint_;
  bool score_valid_;
  double score_;
  uint32 victories_;
};

} // namespace vcsmc

#endif  // SRC_KERNEL_H_

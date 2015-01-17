#ifndef SRC_LINE_KERNEL_H_
#define SRC_LINE_KERNEL_H_

#include <memory>
#include <vector>

#include "constants.h"
#include "types.h"

namespace vcsmc {

class Random;
class State;

namespace op {
class OpCode;
}

class LineKernel {
 public:
  LineKernel();
  ~LineKernel();

  std::unique_ptr<LineKernel> Clone();

  // Assumed called on an empty LineKernel, generates a set of load/store pairs
  // sufficient to fill one line of CPU time.
  void Randomize(Random* random);
  void Mutate(Random* random);
  // Always resets the victories count.
  void Simulate(const uint8* half_colus, uint32 scan_line,
      const State* entry_state, uint32 lines_to_score);
  void Compete(LineKernel* lk);
  void Append(std::vector<std::unique_ptr<op::OpCode>>* opcodes,
              std::vector<std::unique_ptr<State>>* states);
  void ResetVictories();

  uint32 total_cycles() const { return total_cycles_; }
  uint32 total_bytes() const { return total_bytes_; }
  float sim_error() const { return sim_error_; }
  uint32 victories() const { return victories_; }
  const State* exit_state() const { return states_.rbegin()->get(); }

 private:
  TIA PickRandomAddress(Random* random);
  std::unique_ptr<op::OpCode> MakeRandomOpCode(Random* random);
  // Change around the order of the opcodes by exchanging two of them.
  void MutateSwapOpCodes(Random* random);
  // Re-roll one of the opcodes at random.
  void MutateChangeOpCode(Random* random);

  bool IsAcceptableLength() const;

  std::vector<std::unique_ptr<op::OpCode>> opcodes_;
  std::vector<std::unique_ptr<State>> states_;
  uint32 total_cycles_;
  uint32 total_bytes_;
  float sim_error_;
  uint32 victories_;
};


}  // namespace vcsmc

#endif  // SRC_LINE_KERNEL_H_

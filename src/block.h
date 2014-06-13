#ifndef _SRC_BLOCK_H_
#define _SRC_BLOCK_H_

#include "range.h"
#include "types.h"

#include <memory>
#include <vector>

namespace vcsmc {

namespace op {
  class OpCode;
}  // namespace op

class Spec;
class State;

class Block {
 public:
  // Creates a Block with default initial state of everything unknown.
  Block();

  // Will create a copy of state and set all register values to unknown. The
  // |range_| of this Block will be empty and it will start at the end time of
  // the range supplied in |state|.
  explicit Block(const State* state);

  // Returns 0 if this |spec| can be scheduled before this Block. Returns the
  // time within the block (currently always the end) if this spec should be
  // appended to this block. Returns kInfinity on error.
  const uint32 EarliestTimeAfter(const Spec& spec) const;

  // Returns the number of clock cycles that would be consumed by appending
  // |spec| to the end of this Block.
  const uint32 ClocksToAppend(const Spec& spec) const;

  void Append(const Spec& spec);
  void AppendBlock(const Block& block);

  const State* final_state() const { return (*states_.rbegin()).get(); };
  const Range& range() const { return range_; }
  uint32 bytes() const { return total_bytes_; }
  uint32 clocks() const { return range_.Duration(); }

 protected:
  Range range_;
  uint32 total_bytes_;
  std::vector<std::unique_ptr<State>> states_;
  std::vector<std::unique_ptr<op::OpCode>> opcodes_;
};

}  // namespace vcsmc

#endif  // SRC_BLOCK_H_

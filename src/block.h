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
  // Will create a copy of state and set all register values to unknown.
  Block(const State* state);
  const uint32 CostToAppend(const Spec& spec) const;
  void Append(const Spec& spec);

  void AppendBlock(const Block& block);

  const State* final_state() const { return (*states_.rbegin()).get(); };
  const Range& range() const { return range_; }

 protected:
  Range range_;
  std::vector<std::unique_ptr<State>> states_;
  std::vector<std::unique_ptr<op::OpCode>> opcodes_;
};

}  // namespace vcsmc

#endif  // SRC_BLOCK_H_

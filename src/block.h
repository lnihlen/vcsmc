#ifndef _SRC_BLOCK_H_
#define _SRC_BLOCK_H_

#include "range.h"
#include "types.h"

namespace vcsmc {

class OpCode;
class Spec;
class State;

class Block {
 public:
  // Will create a copy of state and set all register values to unknown.
  Block(const State* state);
  uint32 CostToAppend(const Spec& spec);
  void Append(const Spec& spec);

  void AppendBlock(const Block& block);

  const State* final_state() const { return *(states_.rbegin()).get() };
  const Range& range() const { return range_; }

 protected:
  Range range_;
  std::list<std::unique_ptr<State>> states_;
  std::list<std::unique_ptr<OpCode>> opcodes_;
};

}  // namespace vcsmc

#endif  // SRC_BLOCK_H_

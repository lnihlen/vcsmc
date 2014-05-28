#ifndef _SRC_BLOCK_H_
#define _SRC_BLOCK_H_

#include "range.h"
#include "types.h"

namespace vcsmc {

class OpCode;
class State;

class Block {
 public:
  // Need at least one OpCode and a starting State to start a Block.
  Block(std::unique_ptr<State> state, std::unique_ptr<OpCode> opcode);
  void AppendOpCode(std::unique_ptr<OpCode> opcode);

  // Possibly, for light copies of Blocks in the schedule for tentative
  // scheduling, or other uses when defragging.
  // void AppendBlock(const Block& block);

  const State* final_state() const { return *(states_.rbegin()).get() };

 protected:
  Range time_span_;
  std::list<std::unique_ptr<State>> states_;
  std::list<std::unique_ptr<OpCode>> opcodes_;
};

}  // namespace vcsmc

#endif  // SRC_BLOCK_H_

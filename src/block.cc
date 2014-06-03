#include "block.h"

#include "constants.h"
#include "opcode.h"
#include "spec.h"
#include "state.h"

namespace vcsmc {

Block::Block(const State* state) {
}

const uint32 Block::CostToAppend(const Spec& spec) const {
  return kInfinity;
}

void Block::Append(const Spec& spec) {
}

void Block::AppendBlock(const Block& block) {
}

}  // namespace vcsmc

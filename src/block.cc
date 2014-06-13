#include "block.h"

#include "constants.h"
#include "opcode.h"
#include "spec.h"
#include "state.h"

namespace vcsmc {

Block::Block() : total_bytes_(0) {
  states_.push_back(std::unique_ptr<State>(new State));
}

Block::Block(const State* state) : total_bytes_(0) {
  std::unique_ptr<State> entry_state = state->MakeEntryState();
  range_ = entry_state->range();
  states_.push_back(std::move(entry_state));
}

const uint32 Block::EarliestTimeAfter(const Spec& spec) const {
  return kInfinity;
}

const uint32 Block::ClocksToAppend(const Spec& spec) const {
  return kInfinity;
}

void Block::Append(const Spec& spec) {
}

void Block::AppendBlock(const Block& block) {
}

}  // namespace vcsmc

#include "block.h"

#include "constants.h"
#include "opcode.h"
#include "spec.h"
#include "state.h"

namespace vcsmc {

Block::Block() : total_bytes_(0) {
  states_.push_back(std::unique_ptr<State>(new State));
  for (uint8 i = 0; i < Register::REGISTER_COUNT; ++i)
    register_usage_times_[i] = kInfinity;
}

Block::Block(const State* state) : total_bytes_(0) {
  std::unique_ptr<State> entry_state = state->MakeEntryState();
  range_ = entry_state->range();
  states_.push_back(std::move(entry_state));
  for (uint8 i = 0; i < Register::REGISTER_COUNT; ++i)
    register_usage_times_[i] = kInfinity;
}

const uint32 Block::EarliestTimeAfter(const Spec& spec) const {
  return kInfinity;
}

const uint32 Block::ClocksToAppend(const Spec& spec) const {
  return kInfinity;
}

void Block::Append(const Spec& spec) {
  // First check is maybe this |spec| is already the way things are, which is
  // great because we get this append for free.
  if (final_state()->tia_known(spec.tia()) &&
      final_state()->tia(spec.tia()) == spec.value())
    return;

}

void Block::AppendBlock(const Block& block) {
}

}  // namespace vcsmc

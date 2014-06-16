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

uint32 Block::EarliestTimeAfter(const Spec& spec) const {
  return kInfinity;
}

uint32 Block::ClocksToAppend(const Spec& spec) const {
  // First check is maybe this |spec| is already the way things are, which is
  // great because we get this append for free.
  if (final_state()->tia_known(spec.tia()) &&
      final_state()->tia(spec.tia()) == spec.value())
    return 0;


  Register reg = Register::REGISTER_COUNT;
  // Search for register possibly already storing value.
  for (uint8 i = 0; i < Register::REGISTER_COUNT; ++i) {
    if (final_state()->register_known(static_cast<Register>(i)) &&
        final_state()->reg(static_cast<Register>(i)) == spec.value()) {
      reg = static_cast<Register>(i);
      break;
    }
  }

  return reg == Register::REGISTER_COUNT ?
      kLoadImmediateColorClocks + kStoreZeroPageColorClocks :
      kStoreZeroPageColorClocks;
}

void Block::Append(const Spec& spec) {
  // First check is maybe this |spec| is already the way things are, which is
  // great because we get this append for free.
  if (final_state()->tia_known(spec.tia()) &&
      final_state()->tia(spec.tia()) == spec.value())
    return;

  // Second check is that one of the registers already holds the required value,
  // allowing us to omit the load.
  Register reg = Register::REGISTER_COUNT;
  for (uint8 i = 0; i < Register::REGISTER_COUNT; ++i) {
    if (final_state()->register_known(static_cast<Register>(i)) &&
        final_state()->reg(static_cast<Register>(i)) == spec.value()) {
      reg = static_cast<Register>(i);
      break;
    }
  }

  // If we didn't find the value in a current register we need to load this
  // value into a new register, pick either an unknown one or the one least
  // recently used.
  if (reg == Register::REGISTER_COUNT) {
    for (uint8 i = 0; i < Register::REGISTER_COUNT; ++i) {
      uint32 oldest_usage_time = kInfinity;
      if (!final_state()->register_known(static_cast<Register>(i))) {
        reg = static_cast<Register>(i);
        break;
      }
      if (register_usage_times_[i] < oldest_usage_time) {
        oldest_usage_time = register_usage_times_[i];
        reg = static_cast<Register>(i);
      }
    }

    // Add a Load OpCode to the |opcodes_| block and make our new final state.
    assert(reg != Register::REGISTER_COUNT);
    std::unique_ptr<op::OpCode> load(new op::LoadImmediate(spec.value(), reg));
    states_.push_back(load->Transform((*states_.rbegin()).get()));
    range_.set_end_time(
        range_.end_time() + (load->cycles() * kColorClocksPerCPUCycle));
    total_bytes_ += load->bytes();
    opcodes_.push_back(std::move(load));
  }

  assert(reg != Register::REGISTER_COUNT);
  assert(final_state()->register_known(reg));
  assert(final_state()->reg(reg) == spec.value());
  register_usage_times_[reg] = range_.end_time();

  std::unique_ptr<op::OpCode> store(new op::StoreZeroPage(spec.tia(), reg));
  states_.push_back(store->Transform((*states_.rbegin()).get()));
  range_.set_end_time(
      range_.end_time() + (store->cycles() * kColorClocksPerCPUCycle));
  total_bytes_ += store->bytes();
  opcodes_.push_back(std::move(store));
}

void Block::AppendBlock(std::unique_ptr<Block> block) {
}

}  // namespace vcsmc

#include "block.h"
#include "gtest/gtest.h"
#include "opcode.h"
#include "spec.h"
#include "state.h"

namespace vcsmc {

TEST(BlockTest, DefaultConstructor) {
  // Default ctor should have made one state with default initial value.
  Block block;
  // All register values should be unknown.
  EXPECT_FALSE(block.final_state()->register_known(Register::A));
  EXPECT_FALSE(block.final_state()->register_known(Register::X));
  EXPECT_FALSE(block.final_state()->register_known(Register::Y));
  // Block's range should be empty and start at zero.
  EXPECT_EQ(Range(0, 0), block.range());
  EXPECT_EQ(0, block.clocks());
  EXPECT_EQ(0, block.bytes());
}

TEST(BlockTest, StateConstructor) {
  std::unique_ptr<State> state(new State);
  state = state->AdvanceTimeAndSetRegister(1, Register::X, 0x00);
  state = state->AdvanceTimeAndSetRegister(1, Register::Y, 0xff);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::X, TIA::PF0);
  std::unique_ptr<State> entry_state =
      state->AdvanceTimeAndCopyRegisterToTIA(1, Register::Y, TIA::COLUBK);
  state = entry_state->AdvanceTime(1);
  Block block(entry_state.get());

  // All register values should be unknown.
  EXPECT_FALSE(block.final_state()->register_known(Register::A));
  EXPECT_FALSE(block.final_state()->register_known(Register::X));
  EXPECT_FALSE(block.final_state()->register_known(Register::Y));

  // Block's range should be empty and start at the |entry_state| end time.
  EXPECT_EQ(
      Range(entry_state->range().end_time(), entry_state->range().end_time()),
      block.range());

  EXPECT_EQ(0, block.clocks());
  EXPECT_EQ(0, block.bytes());
}

TEST(BlockTest, EarliestTimeAfter) {

}



}  // namespace vcsmc

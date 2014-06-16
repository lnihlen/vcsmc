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
  EXPECT_EQ(Range(entry_state->range().end_time(),
      entry_state->range().end_time()), block.range());

  EXPECT_EQ(0, block.clocks());
  EXPECT_EQ(0, block.bytes());
}

TEST(BlockTest, EarliestTimeAfterScheduleBefore) {
  // TODO
}

TEST(BlockTest, EarliestTimeAfterScheduleAppend) {
  // TODO
}

TEST(BlockTest, EarliestTimeAfterScheduleError) {
  // TODO
}

TEST(BlockTest, ClocksToAppendRegsUnknown) {
  Block block;
  Spec pf2_spec(TIA::PF2, 0x00, Range(0, kScanLineWidthCycles));
  EXPECT_EQ(kLoadImmediateColorClocks + kStoreZeroPageColorClocks,
      block.ClocksToAppend(pf2_spec));
  EXPECT_EQ(0, block.bytes());
  EXPECT_EQ(0, block.clocks());
}

TEST(BlockTest, ClocksToAppendRegsKnownButDifferent) {
  Block block;
}

TEST(BlockTest, ClocksToAppendRegsKnownAndSame) {
  // TODO
}

TEST(BlockTest, ClocksToAddNoChangeToTIA) {
  // TODO
}

// Strobes should always cost one StoreZeroPage operation to append, with or
// without register value reuse and even if the current stored value for the
// strobe is the same.
TEST(BlockTest, ClocksToAppendStrobeAlwaysCheaperButNotFree) {
  // TODO
}

TEST(BlockTest, AppendRegsAndTIAUnknown) {
  Block block;
  Spec colupf_spec(TIA::COLUPF, 0xfe, Range(0, kScanLineWidthCycles));
  uint32 colupf_clocks = block.ClocksToAppend(colupf_spec);
  block.Append(colupf_spec);
  EXPECT_TRUE(block.final_state()->tia_known(TIA::COLUPF));
  EXPECT_EQ(Range(0, colupf_clocks), block.range());
  EXPECT_LT(0, block.bytes());
  EXPECT_EQ(colupf_clocks, block.clocks());
  uint32 known_count = 0;
  for (uint8 i = 0; i < Register::REGISTER_COUNT; ++i) {
    if (block.final_state()->register_known(static_cast<Register>(i))) {
      ++known_count;
      EXPECT_EQ(colupf_spec.value(),
          block.final_state()->reg(static_cast<Register>(i)));
    }
  }
  EXPECT_EQ(1, known_count);
}

// A different TIA state change but with the same value should result in a
// smaller increase in size and time due to register re-use.
TEST(BlockTest, AppendDuplicateRegisterValue) {
  Block block;
  Spec colubk_spec(TIA::COLUBK, 0xfe, Range(0, kScanLineWidthCycles));
  block.Append(colubk_spec);
  uint32 colubk_clocks = block.clocks();
  uint32 colubk_bytes = block.bytes();
  Spec ctrlpf_spec(TIA::CTRLPF, 0xfe, Range(0, kScanLineWidthCycles));
  block.Append(ctrlpf_spec);
  uint32 ctrlpf_clocks = block.clocks() - colubk_clocks;
  uint32 ctrlpf_bytes = block.bytes() - colubk_bytes;
  ASSERT_GT(colubk_clocks, ctrlpf_clocks);
  ASSERT_GT(colubk_bytes, ctrlpf_bytes);
}

// The same state change appended multiple times should add zero size or time
// to the Block.
TEST(BlockTest, AppendNoChangeToTIA) {
  Block block;
  Spec p0_spec(TIA::PF0, 0xff, Range(0, kScanLineWidthCycles));
  block.Append(p0_spec);
  uint32 block_bytes = block.bytes();
  uint32 block_clocks = block.clocks();
  block.Append(p0_spec);
  EXPECT_EQ(block_bytes, block.bytes());
  EXPECT_EQ(block_clocks, block.clocks());
}

TEST(BlockTest, AppendStrobeAlwaysCheaperButNotFree) {
  // TODO
}

}  // namespace vcsmc

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
  Block block(entry_state.get(), 1);

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

// A Block with every state within returning 0 for a given Spec should also
// return 0.
TEST(BlockTest, EarliestTimeAfterScheduleBefore) {
  // Build a block within the HBlank and then ask for a CTRLPF Spec, which since
  // the entire block is in HBlank should return 0.
  std::unique_ptr<State> state(new State);
  Block block(state.get(), 16);
  block.Append(Spec(TIA::COLUBK, 0x00, Range(0, kFrameSizeClocks)));
  block.Append(Spec(TIA::COLUPF, 0xfe, Range(0, kFrameSizeClocks)));
  block.Append(Spec(TIA::PF0, 0x00, Range(0, kFrameSizeClocks)));
  block.Append(Spec(TIA::PF1, 0xaa, Range(0, kFrameSizeClocks)));
  block.Append(Spec(TIA::PF2, 0xef, Range(0, kFrameSizeClocks)));
  EXPECT_EQ(0,
      block.EarliestTimeAfter(
          Spec(TIA::CTRLPF, 0x02, Range(0, kFrameSizeClocks))));
}

// A Block with one state returning nonzero scheduling needs to indicate it can
// support an append of the Spec by returning its current end_time() - 1.
TEST(BlockTest, EarliestTimeAfterScheduleAppend) {

}

// Blocks could suggest the creation of a new Block if they return a time
// greater than their range().end_time().
TEST(BlockTest, EarliestTimeAfterNewBlock) {
  std::unique_ptr<State> state(new State);
  state = state->AdvanceTimeAndSetRegister(1, Register::A, 0x00);
  std::unique_ptr<State> entry_state =
      state->AdvanceTimeAndCopyRegisterToTIA(1, Register::A, TIA::CTRLPF);
  // Start block halfway through right side PF1.
  Block block(entry_state.get(),
      kHBlankWidthClocks - 2 + 16 + 32 + 32 + 16 + 16);
  block.Append(Spec(TIA::COLUBK, 0x00, Range(0, kFrameSizeClocks)));
  block.Append(Spec(TIA::COLUPF, 0xfe, Range(0, kFrameSizeClocks)));
  // The block should suggest the creation of a new Block starting at the
  // beginning of the second line, which is well after this block ends but the
  // earliest that the Block final_state should recommend a change to PF2.
  EXPECT_EQ(kScanLineWidthClocks - 1,
      block.EarliestTimeAfter(Spec(TIA::PF2, 0xff,
          Range(0, kScanLineWidthClocks + kHBlankWidthClocks + 16 + 32))));
  // The block should still respect the range of the Spec, and return the
  // range().start_time() of that Spec if it is later than the earliest the
  // |final_state| could schedule.
  EXPECT_EQ(kScanLineWidthClocks + 32,
      block.EarliestTimeAfter(Spec(TIA::PF2, 0x52,
          Range(kScanLineWidthClocks + 32,
              kScanLineWidthClocks + kHBlankWidthClocks + 16 + 32))));
  // If the Spec calls for something sooner and there's still room regular
  // appends should work as expected.
  EXPECT_EQ(block.range().end_time() - 1,
      block.EarliestTimeAfter(Spec(TIA::PF2, 0x2b,
          Range(block.final_state()->range().start_time(),
            kScanLineWidthClocks))));
  // Logically, a Spec that requires a change within a deadline unacceptable
  // to the final state should return an error.
  EXPECT_EQ(kInfinity,
      block.EarliestTimeAfter(Spec(TIA::PF2, 0x2b,
          Range(kScanLineWidthClocks - 16, kScanLineWidthClocks))));
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
  // TODO
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
  std::unique_ptr<State> state(new State);
  std::unique_ptr<State> entry_state = state->AdvanceTimeAndSetRegister(
      162, Register::X, 0x00);

  Block block(entry_state.get(), 158);
  Spec colupf_spec(TIA::COLUPF, 0xfe, Range(0, kScanLineWidthCycles));
  uint32 colupf_clocks = block.ClocksToAppend(colupf_spec);
  block.Append(colupf_spec);
  EXPECT_TRUE(block.final_state()->tia_known(TIA::COLUPF));
  EXPECT_TRUE(block.final_state()->range().IsEmpty());
  EXPECT_EQ(block.range().end_time(),
      block.final_state()->range().start_time());
  EXPECT_EQ(Range(entry_state->range().end_time(),
      entry_state->range().end_time() + colupf_clocks), block.range());
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
  EXPECT_EQ(Range(0, colubk_clocks), block.range());
  EXPECT_EQ(Range(colubk_clocks, colubk_clocks), block.final_state()->range());
  uint32 colubk_bytes = block.bytes();
  Spec ctrlpf_spec(TIA::CTRLPF, 0xfe, Range(0, kScanLineWidthCycles));
  block.Append(ctrlpf_spec);
  EXPECT_EQ(Range(block.range().end_time(), block.range().end_time()),
      block.final_state()->range());
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
  EXPECT_EQ(Range(0, block_clocks), block.range());
  EXPECT_EQ(Range(block_clocks, block_clocks), block.final_state()->range());
  block.Append(p0_spec);
  EXPECT_EQ(block_bytes, block.bytes());
  EXPECT_EQ(block_clocks, block.clocks());
  EXPECT_EQ(Range(0, block_clocks), block.range());
  EXPECT_EQ(Range(block_clocks, block_clocks), block.final_state()->range());
}

TEST(BlockTest, AppendStrobeAlwaysCheaperButNotFree) {
  // TODO
}

}  // namespace vcsmc

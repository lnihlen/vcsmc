#include "colu_strip.h"
#include "gtest/gtest.h"
#include "state.h"

namespace vcsmc {

class StateTest : public ::testing::Test {
 protected:
  void CompareStatesExceptRange(const State* s1, const State* s2) const {
    CompareStatesRegisters(s1, s2);
    CompareStatesTIA(s1, s2);
  }

  void CompareStatesRegisters(const State* s1, const State* s2) const {
    for (uint8 i = 0; i < Register::REGISTER_COUNT; ++i) {
      Register reg = static_cast<Register>(i);
      EXPECT_EQ(s1->register_known(reg), s2->register_known(reg));
      if (s1->register_known(reg) && s2->register_known(reg))
        EXPECT_EQ(s1->reg(reg), s2->reg(reg));
    }
  }

  void CompareStatesTIA(const State* s1, const State* s2) const {
    for (uint8 i = 0; i < TIA::TIA_COUNT; ++i) {
      TIA tia = static_cast<TIA>(i);
      EXPECT_EQ(s1->tia_known(tia), s2->tia_known(tia));
      if (s1->tia_known(tia) && s2->tia_known(tia))
        EXPECT_EQ(s1->tia(tia), s2->tia(tia));
    }
  }
};

TEST_F(StateTest, InitialStateHasFullProgramRange) {
  State state;
  EXPECT_EQ(0, state.range().start_time());
  EXPECT_EQ(kFrameSizeClocks, state.range().end_time());
}

TEST_F(StateTest, InitialStateHasAllRegistersUnknown) {
  State state;
  EXPECT_FALSE(state.register_known(Register::A));
  EXPECT_FALSE(state.register_known(Register::X));
  EXPECT_FALSE(state.register_known(Register::Y));
}

TEST_F(StateTest, InitialStateHasOnlyStrobeTIAValuesKnown) {
  State state;
  EXPECT_FALSE(state.tia_known(TIA::VSYNC));
  EXPECT_FALSE(state.tia_known(TIA::VBLANK));
  EXPECT_TRUE(state.tia_known(TIA::WSYNC));
  EXPECT_TRUE(state.tia_known(TIA::RSYNC));
  EXPECT_FALSE(state.tia_known(TIA::NUSIZ0));
  EXPECT_FALSE(state.tia_known(TIA::NUSIZ1));
  EXPECT_FALSE(state.tia_known(TIA::COLUP0));
  EXPECT_FALSE(state.tia_known(TIA::COLUP1));
  EXPECT_FALSE(state.tia_known(TIA::COLUPF));
  EXPECT_FALSE(state.tia_known(TIA::COLUBK));
  EXPECT_FALSE(state.tia_known(TIA::CTRLPF));
  EXPECT_FALSE(state.tia_known(TIA::REFP0));
  EXPECT_FALSE(state.tia_known(TIA::REFP1));
  EXPECT_FALSE(state.tia_known(TIA::PF0));
  EXPECT_FALSE(state.tia_known(TIA::PF1));
  EXPECT_FALSE(state.tia_known(TIA::PF2));
  EXPECT_TRUE(state.tia_known(TIA::RESP0));
  EXPECT_TRUE(state.tia_known(TIA::RESP1));
  EXPECT_TRUE(state.tia_known(TIA::RESM0));
  EXPECT_TRUE(state.tia_known(TIA::RESM1));
  EXPECT_TRUE(state.tia_known(TIA::RESBL));
  EXPECT_FALSE(state.tia_known(TIA::AUDC0));
  EXPECT_FALSE(state.tia_known(TIA::AUDC1));
  EXPECT_FALSE(state.tia_known(TIA::AUDF0));
  EXPECT_FALSE(state.tia_known(TIA::AUDF1));
  EXPECT_FALSE(state.tia_known(TIA::AUDV0));
  EXPECT_FALSE(state.tia_known(TIA::AUDV1));
  EXPECT_FALSE(state.tia_known(TIA::GRP0));
  EXPECT_FALSE(state.tia_known(TIA::GRP1));
  EXPECT_FALSE(state.tia_known(TIA::ENAM0));
  EXPECT_FALSE(state.tia_known(TIA::ENAM1));
  EXPECT_FALSE(state.tia_known(TIA::ENABL));
  EXPECT_FALSE(state.tia_known(TIA::HMP0));
  EXPECT_FALSE(state.tia_known(TIA::HMP1));
  EXPECT_FALSE(state.tia_known(TIA::HMM0));
  EXPECT_FALSE(state.tia_known(TIA::HMM1));
  EXPECT_FALSE(state.tia_known(TIA::HMBL));
  EXPECT_FALSE(state.tia_known(TIA::VDELP0));
  EXPECT_FALSE(state.tia_known(TIA::VDELP1));
  EXPECT_FALSE(state.tia_known(TIA::VDELBL));
  EXPECT_FALSE(state.tia_known(TIA::RESMP0));
  EXPECT_FALSE(state.tia_known(TIA::RESMP1));
  EXPECT_TRUE(state.tia_known(TIA::HMOVE));
  EXPECT_TRUE(state.tia_known(TIA::HMCLR));
  EXPECT_TRUE(state.tia_known(TIA::CXCLR));
}

TEST_F(StateTest, CloneCopiesInitialState) {
  State state;
  std::unique_ptr<State> clone(state.Clone());
  EXPECT_EQ(state.range(), clone->range());
  CompareStatesExceptRange(&state, clone.get());
}

TEST_F(StateTest, CloneCopiesLaterState) {
  // Set registers A and Y, along with a few TIA values.
  std::unique_ptr<State> state(new State);
  state = state->AdvanceTimeAndSetRegister(2, Register::A, 0x7c);
  state = state->AdvanceTimeAndCopyRegisterToTIA(3, Register::A, TIA::GRP1);
  state = state->AdvanceTimeAndCopyRegisterToTIA(3, Register::A, TIA::PF0);
  state = state->AdvanceTimeAndSetRegister(2, Register::Y, 0xae);
  state = state->AdvanceTimeAndCopyRegisterToTIA(3, Register::Y, TIA::VDELP0);
  state = state->AdvanceTimeAndSetRegister(2, Register::A, 0x44);
  state = state->AdvanceTimeAndCopyRegisterToTIA(3, Register::Y, TIA::COLUBK);
  state = state->AdvanceTimeAndCopyRegisterToTIA(3, Register::A, TIA::NUSIZ1);

  // Make a copy of end state.
  std::unique_ptr<State> clone = state->Clone();
  EXPECT_EQ(state->range(), clone->range());
  CompareStatesExceptRange(state.get(), clone.get());
}

TEST_F(StateTest, AdvanceTimeClonesAndSetsRanges) {
  std::unique_ptr<State> state(new State);
  std::unique_ptr<State> advance = state->AdvanceTime(kScanLineWidthClocks);
  EXPECT_EQ(Range(0, kScanLineWidthClocks), state->range());
  EXPECT_EQ(Range(kScanLineWidthClocks, kFrameSizeClocks), advance->range());
  CompareStatesExceptRange(state.get(), advance.get());
}

TEST_F(StateTest, AdvanceTimeAndSetRegisterClonesAndSetsRegisterValue) {
  std::unique_ptr<State> state(new State);
  std::unique_ptr<State> advance = state->AdvanceTimeAndSetRegister(
      32, Register::X, 0xcc);

  CompareStatesTIA(state.get(), advance.get());

  EXPECT_FALSE(state->register_known(Register::A));
  EXPECT_FALSE(advance->register_known(Register::A));
  EXPECT_FALSE(state->register_known(Register::X));
  EXPECT_TRUE(advance->register_known(Register::X));
  EXPECT_FALSE(state->register_known(Register::Y));
  EXPECT_FALSE(advance->register_known(Register::Y));
  EXPECT_EQ(0xcc, advance->x());

  EXPECT_EQ(Range(0, 32), state->range());
  EXPECT_EQ(Range(32, kFrameSizeClocks), advance->range());
}

TEST_F(StateTest, AdvanceTimeAndCopyRegisterToTIAClonesAndSetsTIAValue) {
  std::unique_ptr<State> state(new State);
  state = state->AdvanceTimeAndSetRegister(64, Register::A, 0xde);
  std::unique_ptr<State> advance = state->AdvanceTimeAndCopyRegisterToTIA(
      64, Register::A, TIA::COLUP0);
  CompareStatesRegisters(state.get(), advance.get());
  EXPECT_FALSE(state->tia_known(TIA::COLUP0));
  EXPECT_TRUE(advance->tia_known(TIA::COLUP0));
  EXPECT_EQ(0xde, advance->tia(TIA::COLUP0));
  EXPECT_EQ(Range(64, 128), state->range());
  EXPECT_EQ(Range(128, kFrameSizeClocks), advance->range());
  for (uint8 i = 0; i < TIA::TIA_COUNT; ++i) {
    TIA tia = static_cast<TIA>(i);
    if (tia == TIA::COLUP0)
      continue;
    EXPECT_EQ(state->tia_known(tia), advance->tia_known(tia));
    if (state->tia_known(tia) && advance->tia_known(tia))
      EXPECT_EQ(state->tia(tia), advance->tia(tia));
  }
}

TEST_F(StateTest, AdvanceTimeAndCopyRegisterToTIAStrobeWSYNC) {
}

TEST_F(StateTest, AdvanceTimeAndCopyRegisterToTIAStrobeRSYNC) {
}

TEST_F(StateTest, AdvanceTimeAndCopyRegisterToTIAStrobeRESP0) {
}

TEST_F(StateTest, AdvanceTimeAndCopyRegisterToTIAStrobeRESP1) {
}

TEST_F(StateTest, AdvanceTimeAndCopyRegisterToTIAStrobeRESM0) {
}

TEST_F(StateTest, AdvanceTimeAndCopyRegisterToTIAStrobeRESM1) {
}

TEST_F(StateTest, AdvanceTimeAndCopyRegisterToTIAStrobeRESBL) {
}

TEST_F(StateTest, AdvanceTimeAndCopyRegisterToTIAStrobeHMOVE) {
}

TEST_F(StateTest, AdvanceTimeAndCopyRegisterToTIAStrobeHMCLR) {
}

TEST_F(StateTest, AdvanceTimeAndCopyRegisterToTIAStrobeCXCLR) {
}

TEST_F(StateTest, PaintIntoBeforeStateRange) {
}

TEST_F(StateTest, PaintIntoAfterStateRange) {
}

TEST_F(StateTest, PaintBackgroundPartial) {
}

TEST_F(StateTest, PaintBackgroundEntire) {
}

TEST_F(StateTest, PaintPlayfieldPartial) {
}

TEST_F(StateTest, PaintPlayerXYZ) {
}

TEST(StateDeathTest, AdvanceTimeZero) {
  std::unique_ptr<State> state(new State);
  EXPECT_DEATH(state = state->AdvanceTime(0), "delta > 0");
}

TEST(StateDeathTest, AdvanceTimeAndCopyRegisterToTIAUnknownRegister) {
}

TEST(StateDeathTest, TIAUnknownAccess) {
}

TEST(StateDeathTest, RegisterUnknownAccess) {
}

TEST(StateDeathTest, RegisterAUnknownAccess) {
}

TEST(StateDeathTest, RegisterXUnknownAccess) {
}

TEST(StateDeathTest, RegisteryYUnknownAccess) {
}

}  // namespace vcsmc

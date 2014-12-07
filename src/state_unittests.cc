#include "gtest/gtest.h"
#include "spec.h"
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

  void ExpectCanScheduleBefore(const State* s, TIA tia) const {
    EXPECT_EQ(0, s->EarliestTimeAfter(Spec(tia, 0xfe,
        Range(0, kFrameSizeClocks))));
    if (s->range().start_time() > 0) {
      EXPECT_EQ(0, s->EarliestTimeAfter(Spec(tia, 0x42,
          Range(s->range().start_time() - 1, s->range().end_time()))));
    } else {
      EXPECT_EQ(0, s->EarliestTimeAfter(Spec(tia, 0x15,
          Range(0, s->range().end_time()))));
    }
    EXPECT_EQ(s->range().start_time() + 1, s->EarliestTimeAfter(Spec(tia, 0x00,
        Range(s->range().start_time() + 1, s->range().end_time()))));
    EXPECT_EQ(kInfinity, s->EarliestTimeAfter(Spec(tia, 0x29,
        Range(s->range().end_time() + 1, s->range().end_time() + 5))));
  }

  void ExpectCannotScheduleBefore(const State* s, TIA tia) const {
    EXPECT_EQ(kInfinity, s->EarliestTimeAfter(
        Spec(tia, 0x01, Range(0, kFrameSizeClocks))));
    EXPECT_EQ(kInfinity, s->EarliestTimeAfter(Spec(tia, 0x23, s->range())));
    EXPECT_EQ(kInfinity, s->EarliestTimeAfter(Spec(tia, 0x10,
        Range(s->range().start_time(), s->range().start_time() + 2))));
  }

  void ExpectCanScheduleWithin(const State* s, TIA tia, uint32 before) const {
    EXPECT_EQ(before, s->EarliestTimeAfter(Spec(tia, 0x42,
        Range(0, kFrameSizeClocks))));
    if (before + 1 < s->range().end_time()) {
      EXPECT_EQ(before + 1, s->EarliestTimeAfter(Spec(tia, 0x12,
          Range(before + 1, s->range().end_time()))));
    }
    if (before - 1 > s->range().start_time()) {
      EXPECT_EQ(kInfinity, s->EarliestTimeAfter(Spec(tia, 0x77,
          Range(s->range().start_time(), before - 1))));
    }
  }
};

TEST_F(StateTest, InitialStateHasFullProgramRange) {
  State state;
  EXPECT_EQ(0, state.range().start_time());
  EXPECT_EQ(kFrameSizeClocks, state.range().end_time());
  uint8 tia_values[TIA::TIA_COUNT];
  State state_value(tia_values);
  EXPECT_EQ(0, state_value.range().start_time());
  EXPECT_EQ(kFrameSizeClocks, state_value.range().end_time());
}

TEST_F(StateTest, InitialStateHasAllRegistersUnknown) {
  State state;
  EXPECT_FALSE(state.register_known(Register::A));
  EXPECT_FALSE(state.register_known(Register::X));
  EXPECT_FALSE(state.register_known(Register::Y));
  uint8 tia_values[TIA::TIA_COUNT];
  std::memset(tia_values, 0, sizeof(tia_values));
  State state_value(tia_values);
  EXPECT_FALSE(state_value.register_known(Register::A));
  EXPECT_FALSE(state_value.register_known(Register::X));
  EXPECT_FALSE(state_value.register_known(Register::Y));
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

TEST_F(StateTest, InitialStateWithValuesHasAllTIAValuesKnown) {
  uint8 tia_values[TIA::TIA_COUNT];
  for (uint8 i = 0; i < TIA::TIA_COUNT; ++i)
    tia_values[i] = i;
  State state_value(tia_values);
  for (uint8 i = 0; i < TIA::TIA_COUNT; ++i) {
    EXPECT_TRUE(state_value.tia_known(static_cast<TIA>(i)));
    EXPECT_EQ(tia_values[i], state_value.tia(static_cast<TIA>(i)));
  }
}

TEST_F(StateTest, CloneCopiesInitialState) {
  State state;
  std::unique_ptr<State> clone(state.Clone());
  EXPECT_EQ(state.range(), clone->range());
  CompareStatesExceptRange(&state, clone.get());
}

TEST_F(StateTest, CloneCopiesLaterState) {
  std::unique_ptr<State> state(new State);
  state = state->AdvanceTimeAndSetRegister(2, Register::A, 0x7c);
  state = state->AdvanceTimeAndCopyRegisterToTIA(3, Register::A, TIA::GRP1);
  state = state->AdvanceTimeAndCopyRegisterToTIA(3, Register::A, TIA::PF0);
  state = state->AdvanceTimeAndSetRegister(2, Register::Y, 0xae);
  state = state->AdvanceTimeAndCopyRegisterToTIA(3, Register::Y, TIA::VDELP0);
  state = state->AdvanceTimeAndSetRegister(2, Register::A, 0x44);
  state = state->AdvanceTimeAndCopyRegisterToTIA(3, Register::Y, TIA::COLUBK);
  state = state->AdvanceTimeAndCopyRegisterToTIA(3, Register::A, TIA::NUSIZ1);

  std::unique_ptr<State> clone = state->Clone();
  EXPECT_EQ(state->range(), clone->range());
  CompareStatesExceptRange(state.get(), clone.get());
}

TEST_F(StateTest, MakeEntryState) {
  std::unique_ptr<State> state(new State);
  state = state->AdvanceTimeAndSetRegister(1, Register::A, 0x42);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::A, TIA::COLUBK);
  state = state->AdvanceTimeAndSetRegister(1, Register::Y, 0xee);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::Y, TIA::PF2);

  std::unique_ptr<State> entry_state = state->MakeEntryState(10);
  CompareStatesTIA(state.get(), entry_state.get());
  EXPECT_FALSE(entry_state->register_known(Register::A));
  EXPECT_FALSE(entry_state->register_known(Register::X));
  EXPECT_FALSE(entry_state->register_known(Register::Y));
  EXPECT_EQ(Range(state->range().end_time(), state->range().end_time()),
      entry_state->range());
}

TEST_F(StateTest, MakeEntryStateZeroDelta) {
  std::unique_ptr<State> state(new State);
  state = state->AdvanceTimeAndSetRegister(19, Register::X, 0xa9);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::X, TIA::CTRLPF);
  state = state->AdvanceTimeAndCopyRegisterToTIA(21, Register::X, TIA::PF1);
  std::unique_ptr<State> entry_state = state->MakeEntryState(0);
  CompareStatesTIA(state.get(), entry_state.get());
  EXPECT_FALSE(entry_state->register_known(Register::A));
  EXPECT_FALSE(entry_state->register_known(Register::X));
  EXPECT_FALSE(entry_state->register_known(Register::Y));
  EXPECT_EQ(Range(state->range().start_time(), state->range().start_time()),
      entry_state->range());
}

// An entry state starts with an empty range, test advancing time on that.
TEST_F(StateTest, AdvanceTimeEntryState) {
  std::unique_ptr<State> state(new State);
  std::unique_ptr<State> ending_state = state->AdvanceTime(1024);

  std::unique_ptr<State> entry_state = ending_state->MakeEntryState(1024);
  EXPECT_EQ(Range(1024, 2048), ending_state->range());

  std::unique_ptr<State> entry_advance = entry_state->AdvanceTime(2048);
  EXPECT_EQ(Range(2048, 4096), entry_state->range());
  EXPECT_EQ(Range(4096, 4096), entry_advance->range());
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
  // TODO
}

TEST_F(StateTest, AdvanceTimeAndCopyRegisterToTIAStrobeRSYNC) {
  // TODO
}

TEST_F(StateTest, AdvanceTimeAndCopyRegisterToTIAStrobeRESP0) {
  // TODO
}

TEST_F(StateTest, AdvanceTimeAndCopyRegisterToTIAStrobeRESP1) {
  // TODO
}

TEST_F(StateTest, AdvanceTimeAndCopyRegisterToTIAStrobeRESM0) {
  // TODO
}

TEST_F(StateTest, AdvanceTimeAndCopyRegisterToTIAStrobeRESM1) {
  // TODO
}

TEST_F(StateTest, AdvanceTimeAndCopyRegisterToTIAStrobeRESBL) {
  // TODO
}

TEST_F(StateTest, AdvanceTimeAndCopyRegisterToTIAStrobeHMOVE) {
  // TODO
}

TEST_F(StateTest, AdvanceTimeAndCopyRegisterToTIAStrobeHMCLR) {
  // TODO
}

TEST_F(StateTest, AdvanceTimeAndCopyRegisterToTIAStrobeCXCLR) {
  // TODO
}

/*
TEST_F(StateTest, PaintIntoBeforeStateRange) {
  std::unique_ptr<State> state(new State);
  state = state->AdvanceTimeAndSetRegister(
      kScanLineWidthClocks, Register::A, 0x00);
  state = state->AdvanceTimeAndCopyRegisterToTIA(
      kScanLineWidthClocks, Register::A, TIA::COLUBK);
  ColuStrip colu_strip(1);
  state->PaintInto(&colu_strip);
  ExpectUnpainted(&colu_strip);
}

TEST_F(StateTest, PaintIntoAfterStateRange) {
  std::unique_ptr<State> original_state(new State);
  original_state->AdvanceTimeAndSetRegister(
      kScanLineWidthClocks, Register::X, 0xfe);
  ColuStrip colu_strip(7);
  original_state->PaintInto(&colu_strip);
  ExpectUnpainted(&colu_strip);
}

TEST_F(StateTest, PaintIntoDuringHBlank) {
  std::unique_ptr<State> state(new State);
  state->AdvanceTime(kHBlankWidthClocks);
  ColuStrip colu_strip(0);
  state->PaintInto(&colu_strip);
  ExpectUnpainted(&colu_strip);
}

TEST_F(StateTest, PaintBackgroundPartial) {
  std::unique_ptr<State> state(new State);
  state = state->AdvanceTimeAndSetRegister(1, Register::Y, 0x00);
  state = state->AdvanceTimeAndSetRegister(1, Register::X, 0x02);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::X, TIA::COLUBK);
  state = state->AdvanceTimeAndSetRegister(1, Register::A, kColuUnpainted);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::A, TIA::COLUPF);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::Y, TIA::PF0);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::Y, TIA::CTRLPF);

  // should still be in HBlank
  EXPECT_GT(kHBlankWidthClocks, state->range().start_time());
  // Advance state to end of HBlank + 5 pixels.
  uint32 hb_plus_five = kHBlankWidthClocks + 5 - state->range().start_time();
  state = state->AdvanceTimeAndCopyRegisterToTIA(
      hb_plus_five, Register::Y, TIA::PF1);
  // Make and discard a next state, to set this state's upper bound at +13 pix.
  state->AdvanceTimeAndCopyRegisterToTIA(13, Register::Y, TIA::PF2);

  ColuStrip colu_strip(0);
  state->PaintInto(&colu_strip);
  for (uint32 i = 0; i < 5; ++i)
    EXPECT_EQ(kColuUnpainted, colu_strip.colu(i));
  for (uint32 i = 5; i < 18; ++i)
    EXPECT_EQ(0x02, colu_strip.colu(i));
  for (uint32 i = 18; i < kFrameWidthPixels; ++i)
    EXPECT_EQ(kColuUnpainted, colu_strip.colu(i));
}

TEST_F(StateTest, PaintBackgroundEntire) {
  std::unique_ptr<State> state(new State);
  state = state->AdvanceTimeAndSetRegister(1, Register::Y, 0x00);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::Y, TIA::COLUBK);
  state = state->AdvanceTimeAndSetRegister(1, Register::A, kColuUnpainted);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::A, TIA::COLUPF);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::Y, TIA::PF0);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::Y, TIA::PF1);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::Y, TIA::PF2);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::Y, TIA::CTRLPF);

  // should still be in HBlank
  EXPECT_GT(kHBlankWidthClocks, state->range().start_time());
  ColuStrip colu_strip(1);
  state->PaintInto(&colu_strip);
  for (uint32 i = 0; i < kFrameWidthPixels; ++i)
    EXPECT_EQ(0x00, colu_strip.colu(i));
}

TEST_F(StateTest, PaintPlayfieldEntireRepeatedNoScore) {
  std::unique_ptr<State> state(new State);
  // pf0 gets 0x87 = 1000 0111
  state = state->AdvanceTimeAndSetRegister(1, Register::A, 0x87);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::A, TIA::PF0);
  // pf1 gets 0x4b = 0100 1011
  state = state->AdvanceTimeAndSetRegister(1, Register::A, 0x4b);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::A, TIA::PF1);
  // pf2 gets 0x2d = 0010 1101
  state = state->AdvanceTimeAndSetRegister(1, Register::A, 0x2d);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::A, TIA::PF2);
  const uint8 colubk = 0x5e;
  state = state->AdvanceTimeAndSetRegister(1, Register::X, colubk);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::X, TIA::COLUBK);
  const uint8 colupf = 0xde;
  state = state->AdvanceTimeAndSetRegister(1, Register::Y, colupf);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::Y, TIA::COLUPF);
  // disable playfield mirroring and score color mode
  state = state->AdvanceTimeAndSetRegister(1, Register::A, 0x00);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::A, TIA::CTRLPF);

  // should still be in HBLANK
  EXPECT_GT(kHBlankWidthClocks, state->range().start_time());
  ColuStrip colu_strip(0);
  state->PaintInto(&colu_strip);
  uint32 col = 0;
  for (uint32 i = 0; i < 2; ++i) {
    // pf0 paints bits 4 - 7 left to right at 4 bits per pixel. First 3 bits
    // are 0, so that's 12 colus of the BG color.
    for (uint32 j = 0; j < 12; ++j)  // 1 000
      EXPECT_EQ(colubk, colu_strip.colu(col++));
    for (uint32 j = 0; j < 4; ++j)  // 1
      EXPECT_EQ(colupf, colu_strip.colu(col++));

    // pf1 paints bits 7 - 0, 01001011
    for (uint32 j = 0; j < 4; ++j)  // 0 1001011
      EXPECT_EQ(colubk, colu_strip.colu(col++));
    for (uint32 j = 0; j < 4; ++j)  // 1 001011
      EXPECT_EQ(colupf, colu_strip.colu(col++));
    for (uint32 j = 0; j < 8; ++j)  // 00 1011
      EXPECT_EQ(colubk, colu_strip.colu(col++));
    for (uint32 j = 0; j < 4; ++j)  // 1 011
      EXPECT_EQ(colupf, colu_strip.colu(col++));
    for (uint32 j = 0; j < 4; ++j)  // 0 11
      EXPECT_EQ(colubk, colu_strip.colu(col++));
    for (uint32 j = 0; j < 8; ++j)  // 11
      EXPECT_EQ(colupf, colu_strip.colu(col++));

    // pf2 paints bits 0 - 7, 00101101
    for (uint32 j = 0; j < 4; ++j)  // 0010110 1
      EXPECT_EQ(colupf, colu_strip.colu(col++));
    for (uint32 j = 0; j < 4; ++j)  // 001011 0
      EXPECT_EQ(colubk, colu_strip.colu(col++));
    for (uint32 j = 0; j < 8; ++j)  // 0010 11
      EXPECT_EQ(colupf, colu_strip.colu(col++));
    for (uint32 j = 0; j < 4; ++j)  // 001 0
      EXPECT_EQ(colubk, colu_strip.colu(col++));
    for (uint32 j = 0; j < 4; ++j)  // 00 1
      EXPECT_EQ(colupf, colu_strip.colu(col++));
    for (uint32 j = 0; j < 8; ++j)  // 00
      EXPECT_EQ(colubk, colu_strip.colu(col++));
  }

  EXPECT_EQ(kFrameWidthPixels, col);
}

TEST_F(StateTest, PaintPlayfieldEntireMirroredNoScore) {
  std::unique_ptr<State> state(new State);
  // pf0 gets 0x18 = 0001 1000
  state = state->AdvanceTimeAndSetRegister(1, Register::A, 0x18);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::A, TIA::PF0);
  // pf1 gets 0x2d = 0010 1101
  state = state->AdvanceTimeAndSetRegister(1, Register::A, 0x2d);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::A, TIA::PF1);
  // pf2 gets 0x4b = 0100 1011
  state = state->AdvanceTimeAndSetRegister(1, Register::A, 0x4b);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::A, TIA::PF2);
  const uint8 colubk = 0xfe;
  state = state->AdvanceTimeAndSetRegister(1, Register::X, colubk);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::X, TIA::COLUBK);
  const uint8 colupf = 0x0e;
  state = state->AdvanceTimeAndSetRegister(1, Register::Y, colupf);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::Y, TIA::COLUPF);
  // enable playfield mirroring and disable score color mode
  state = state->AdvanceTimeAndSetRegister(1, Register::A, 0x01);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::A, TIA::CTRLPF);

  // should still be in HBLANK
  EXPECT_GT(kHBlankWidthClocks, state->range().start_time());
  ColuStrip colu_strip(0);
  state->PaintInto(&colu_strip);
  uint32 col = 0;

  // left side pf0 paints bits 4 - 7, 0001
  for (uint32 i = 0; i < 4; ++i)   // 000 1
    EXPECT_EQ(colupf, colu_strip.colu(col++));
  for (uint32 i = 0; i < 12; ++i)  // 000
    EXPECT_EQ(colubk, colu_strip.colu(col++));

  // left side pf1 paints bits 7 - 0, 00101101
  for (uint32 i = 0; i < 8; ++i)  // 00 101101
    EXPECT_EQ(colubk, colu_strip.colu(col++));
  for (uint32 i = 0; i < 4; ++i)  // 1 01101
    EXPECT_EQ(colupf, colu_strip.colu(col++));
  for (uint32 i = 0; i < 4; ++i)  // 0 1101
    EXPECT_EQ(colubk, colu_strip.colu(col++));
  for (uint32 i = 0; i < 8; ++i)  // 11 01
    EXPECT_EQ(colupf, colu_strip.colu(col++));
  for (uint32 i = 0; i < 4; ++i)  // 0 1
    EXPECT_EQ(colubk, colu_strip.colu(col++));
  for (uint32 i = 0; i < 4; ++i)  // 1
    EXPECT_EQ(colupf, colu_strip.colu(col++));

  // left side pf2 paints bits 0 - 7, 01001011
  for (uint32 i = 0; i < 8; ++i)  // 010010 11
    EXPECT_EQ(colupf, colu_strip.colu(col++));
  for (uint32 i = 0; i < 4; ++i)  // 01001 0
    EXPECT_EQ(colubk, colu_strip.colu(col++));
  for (uint32 i = 0; i < 4; ++i)  // 0100 1
    EXPECT_EQ(colupf, colu_strip.colu(col++));
  for (uint32 i = 0; i < 8; ++i)  // 01 00
    EXPECT_EQ(colubk, colu_strip.colu(col++));
  for (uint32 i = 0; i < 4; ++i)  // 0 1
    EXPECT_EQ(colupf, colu_strip.colu(col++));
  for (uint32 i = 0; i < 4; ++i)  // 0
    EXPECT_EQ(colubk, colu_strip.colu(col++));

  // right side pf2 paints bits 7 - 0, 01001011
  for (uint32 i = 0; i < 4; ++i)  // 0 1001011
    EXPECT_EQ(colubk, colu_strip.colu(col++));
  for (uint32 i = 0; i < 4; ++i)  // 1 001011
    EXPECT_EQ(colupf, colu_strip.colu(col++));
  for (uint32 i = 0; i < 8; ++i)  // 00 1011
    EXPECT_EQ(colubk, colu_strip.colu(col++));
  for (uint32 i = 0; i < 4; ++i)  // 1 011
    EXPECT_EQ(colupf, colu_strip.colu(col++));
  for (uint32 i = 0; i < 4; ++i)  // 0 11
    EXPECT_EQ(colubk, colu_strip.colu(col++));
  for (uint32 i = 0; i < 8; ++i)  // 11
    EXPECT_EQ(colupf, colu_strip.colu(col++));

  // right side pf1 paints bits 0 - 7, 00101101
  for (uint32 i = 0; i < 4; ++i)  // 0010110 1
    EXPECT_EQ(colupf, colu_strip.colu(col++));
  for (uint32 i = 0; i < 4; ++i)  // 001011 0
    EXPECT_EQ(colubk, colu_strip.colu(col++));
  for (uint32 i = 0; i < 8; ++i)  // 0010 11
    EXPECT_EQ(colupf, colu_strip.colu(col++));
  for (uint32 i = 0; i < 4; ++i)  // 001 0
    EXPECT_EQ(colubk, colu_strip.colu(col++));
  for (uint32 i = 0; i < 4; ++i)  // 00 1
    EXPECT_EQ(colupf, colu_strip.colu(col++));
  for (uint32 i = 0; i < 8; ++i)  // 00
    EXPECT_EQ(colubk, colu_strip.colu(col++));

  // right side pf0 paints bits 7 - 4, 0001
  for (uint32 i = 0; i < 12; ++i)  // 000 1
    EXPECT_EQ(colubk, colu_strip.colu(col++));
  for (uint32 i = 0; i < 4; ++i)  // 1
    EXPECT_EQ(colupf, colu_strip.colu(col++));

  EXPECT_EQ(kFrameWidthPixels, col);
}

TEST_F(StateTest, PaintPlayfieldEntireRepeatedWithScore) {
  std::unique_ptr<State> state(new State);
  // pf0 gets 0xb4 = 1011 0100
  state = state->AdvanceTimeAndSetRegister(1, Register::X, 0xb4);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::X, TIA::PF0);
  // pf1 gets 0xd2 = 1101 0010
  state = state->AdvanceTimeAndSetRegister(1, Register::Y, 0xd2);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::Y, TIA::PF1);
  // pf2 gets 0xe1 = 1110 0001
  state = state->AdvanceTimeAndSetRegister(1, Register::A, 0xe1);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::A, TIA::PF2);
  const uint8 colubk = 0xfe;
  state = state->AdvanceTimeAndSetRegister(1, Register::X, colubk);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::X, TIA::COLUBK);
  const uint8 colupf = 0x0e;
  state = state->AdvanceTimeAndSetRegister(1, Register::Y, colupf);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::Y, TIA::COLUPF);
  const uint8 colup0 = 0x60;
  state = state->AdvanceTimeAndSetRegister(1, Register::A, colup0);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::A, TIA::COLUP0);
  const uint8 colup1 = 0x70;
  state = state->AdvanceTimeAndSetRegister(1, Register::A, colup1);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::A, TIA::COLUP1);
  // disable playfield mirroring and enable score color mode
  state = state->AdvanceTimeAndSetRegister(1, Register::A, 0x02);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::A, TIA::CTRLPF);

  // should still be in HBLANK
  EXPECT_GT(kHBlankWidthClocks, state->range().start_time());
  ColuStrip colu_strip(0);
  state->PaintInto(&colu_strip);
  uint32 col = 0;
  for (uint32 i = 0; i < 2; ++i) {
    uint8 colupx = i == 0 ? colup0 : colup1;
    // pf0 paints bits 4 - 7, 1011
    for (uint32 j = 0; j < 8; ++j)  // 10 11
      EXPECT_EQ(colupx, colu_strip.colu(col++));
    for (uint32 j = 0; j < 4; ++j)  // 1 0
      EXPECT_EQ(colubk, colu_strip.colu(col++));
    for (uint32 j = 0; j < 4; ++j)  // 1
      EXPECT_EQ(colupx, colu_strip.colu(col++));

    // pf1 paints bits 7 - 0, 11010010
    for (uint32 j = 0; j < 8; ++j)  // 11 010010
      EXPECT_EQ(colupx, colu_strip.colu(col++));
    for (uint32 j = 0; j < 4; ++j)  // 0 10010
      EXPECT_EQ(colubk, colu_strip.colu(col++));
    for (uint32 j = 0; j < 4; ++j)  // 1 0010
      EXPECT_EQ(colupx, colu_strip.colu(col++));
    for (uint32 j = 0; j < 8; ++j)  // 00 10
      EXPECT_EQ(colubk, colu_strip.colu(col++));
    for (uint32 j = 0; j < 4; ++j)  // 1 0
      EXPECT_EQ(colupx, colu_strip.colu(col++));
    for (uint32 j = 0; j < 4; ++j)  // 0
      EXPECT_EQ(colubk, colu_strip.colu(col++));

    // pf2 paints bits 0 - 7, 11100001
    for (uint32 j = 0; j < 4; ++j)  // 1110000 1
      EXPECT_EQ(colupx, colu_strip.colu(col++));
    for (uint32 j = 0; j < 16; ++j)  // 111 0000
      EXPECT_EQ(colubk, colu_strip.colu(col++));
    for (uint32 j = 0; j < 12; ++j)  // 111
      EXPECT_EQ(colupx, colu_strip.colu(col++));
  }

  EXPECT_EQ(kFrameWidthPixels, col);
}

TEST_F(StateTest, PaintPlayfieldEntireMirroredWithScore) {
  std::unique_ptr<State> state(new State);
  // pf0 gets 0xe7 = 1110 1000
  state = state->AdvanceTimeAndSetRegister(1, Register::A, 0xe7);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::A, TIA::PF0);
  // pf1 gets 0x75 = 0111 0101
  state = state->AdvanceTimeAndSetRegister(1, Register::A, 0x75);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::A, TIA::PF1);
  // pf2 gets 0x87 = 1000 0111
  state = state->AdvanceTimeAndSetRegister(1, Register::A, 0x87);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::A, TIA::PF2);
  const uint8 colubk = 0x40;
  state = state->AdvanceTimeAndSetRegister(1, Register::A, colubk);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::A, TIA::COLUBK);
  const uint8 colupf = 0x20;
  state = state->AdvanceTimeAndSetRegister(1, Register::A, colupf);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::A, TIA::COLUPF);
  const uint8 colup0 = 0x10;
  state = state->AdvanceTimeAndSetRegister(1, Register::A, colup0);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::A, TIA::COLUP0);
  const uint8 colup1 = 0x00;
  state = state->AdvanceTimeAndSetRegister(1, Register::A, colup1);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::A, TIA::COLUP1);
  // enable playfield mirroring and enable score color mode
  state = state->AdvanceTimeAndSetRegister(1, Register::A, 0x03);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::A, TIA::CTRLPF);

  // should still be in HBLANK
  EXPECT_GT(kHBlankWidthClocks, state->range().start_time());
  ColuStrip colu_strip(0);
  state->PaintInto(&colu_strip);
  uint32 col = 0;

  // left side pf0 paints bits 4 - 7, 1110
  for (uint32 i = 0; i < 4; ++i)   // 111 0
    EXPECT_EQ(colubk, colu_strip.colu(col++));
  for (uint32 i = 0; i < 12; ++i)  // 111
    EXPECT_EQ(colup0, colu_strip.colu(col++));

  // left side pf1 paints bits 7 - 0, 01110101
  for (uint32 i = 0; i < 4; ++i)  // 0 1110101
    EXPECT_EQ(colubk, colu_strip.colu(col++));
  for (uint32 i = 0; i < 12; ++i)  // 111 0101
    EXPECT_EQ(colup0, colu_strip.colu(col++));
  for (uint32 i = 0; i < 4; ++i)  // 0 101
    EXPECT_EQ(colubk, colu_strip.colu(col++));
  for (uint32 i = 0; i < 4; ++i)  // 1 01
    EXPECT_EQ(colup0, colu_strip.colu(col++));
  for (uint32 i = 0; i < 4; ++i)  // 0 1
    EXPECT_EQ(colubk, colu_strip.colu(col++));
  for (uint32 i = 0; i < 4; ++i)  // 1
    EXPECT_EQ(colup0, colu_strip.colu(col++));

  // left side pf2 paints bits 0 - 7, 10000111
  for (uint32 i = 0; i < 12; ++i)  // 10000 111
    EXPECT_EQ(colup0, colu_strip.colu(col++));
  for (uint32 i = 0; i < 16; ++i)  // 1 0000
    EXPECT_EQ(colubk, colu_strip.colu(col++));
  for (uint32 i = 0; i < 4; ++i)  // 1
    EXPECT_EQ(colup0, colu_strip.colu(col++));

  // right side pf2 paints bits 7 - 0, 10000111
  for (uint32 i = 0; i < 4; ++i)  // 1 0001111
    EXPECT_EQ(colup1, colu_strip.colu(col++));
  for (uint32 i = 0; i < 16; ++i)  // 0000 111
    EXPECT_EQ(colubk, colu_strip.colu(col++));
  for (uint32 i = 0; i < 12; ++i)  // 111
    EXPECT_EQ(colup1, colu_strip.colu(col++));

  // right side pf1 paints bits 0 - 7, 01110101
  for (uint32 i = 0; i < 4; ++i)  // 0111010 1
    EXPECT_EQ(colup1, colu_strip.colu(col++));
  for (uint32 i = 0; i < 4; ++i)  // 011101 0
    EXPECT_EQ(colubk, colu_strip.colu(col++));
  for (uint32 i = 0; i < 4; ++i)  // 01110 1
    EXPECT_EQ(colup1, colu_strip.colu(col++));
  for (uint32 i = 0; i < 4; ++i)  // 0111 0
    EXPECT_EQ(colubk, colu_strip.colu(col++));
  for (uint32 i = 0; i < 12; ++i)  // 0 111
    EXPECT_EQ(colup1, colu_strip.colu(col++));
  for (uint32 i = 0; i < 4; ++i)  // 0
    EXPECT_EQ(colubk, colu_strip.colu(col++));

  // right side pf0 paints bits 7 - 4, 1110
  for (uint32 i = 0; i < 12; ++i)  // 111 0
    EXPECT_EQ(colup1, colu_strip.colu(col++));
  for (uint32 i = 0; i < 4; ++i)  // 0
    EXPECT_EQ(colubk, colu_strip.colu(col++));

  EXPECT_EQ(kFrameWidthPixels, col);
}
*/
TEST_F(StateTest, EarliestTimeAfterWithSpecBeforeState) {
  std::unique_ptr<State> state(new State);
  state = state->AdvanceTimeAndSetRegister(25, Register::A, 0x00);
  Spec before(TIA::COLUP0, 0x00, Range(0, 10));
  EXPECT_EQ(kInfinity, state->EarliestTimeAfter(before));
}

TEST_F(StateTest, EarliestTimeAfterWithSpecAfterState) {
  std::unique_ptr<State> state(new State);
  state->AdvanceTimeAndSetRegister(3, Register::X, 0xff);
  Spec after(TIA::PF2, 0x77, Range(7, 11));
  EXPECT_EQ(kInfinity, state->EarliestTimeAfter(after));
}

TEST_F(StateTest, EarliestTimeAfterCOLUPF) {
  // COLUPF can be set any time the TIA is not rendering the playfield color,
  // i.e. any time the TIA is not rendering a 1 in the playfield.
  std::unique_ptr<State> state(new State);
  state = state->AdvanceTimeAndSetRegister(1, Register::A, 0x00);
  std::unique_ptr<State> hblank_state = state->AdvanceTimeAndCopyRegisterToTIA(
      1, Register::A, TIA::CTRLPF);
  // A state within the HBLANK should always return 0.
  state = hblank_state->AdvanceTimeAndSetRegister(
      kHBlankWidthClocks - 2, Register::A, 0x00);
  ExpectCanScheduleBefore(hblank_state.get(), TIA::COLUPF);

  // A state that straddles a line break and has pf0 painting on the previous
  // line should return the color clock of the last lit pixel in right side pf0.
  state = state->AdvanceTimeAndCopyRegisterToTIA(
      kFrameWidthPixels, Register::A, TIA::PF1);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::A, TIA::PF2);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::A, TIA::CTRLPF);
  state = state->AdvanceTimeAndSetRegister(1, Register::Y, 0x20);
  std::unique_ptr<State> spanning_state =
      state->AdvanceTimeAndCopyRegisterToTIA(1, Register::Y, TIA::PF0);
  state = spanning_state->AdvanceTimeAndSetRegister(
      kScanLineWidthClocks, Register::X, 0x01);
  // Expected value is second line, right side pf0, second bit, last pixel.
  ExpectCanScheduleWithin(spanning_state.get(), TIA::COLUPF,
      kScanLineWidthClocks + 148 + 4 + 3);
  // Turn on mirroring and try again.
  std::unique_ptr<State> mirrored_span =
      state->AdvanceTimeAndCopyRegisterToTIA(1, Register::X, TIA::CTRLPF);
  state = mirrored_span->AdvanceTimeAndSetRegister(
      kScanLineWidthClocks, Register::X, 0x02);
  // Expected value is third line, mirrored right side pf0, third bit, last pix.
  ExpectCanScheduleWithin(mirrored_span.get(), TIA::COLUPF,
      (2 * kScanLineWidthClocks) + 212 + 8 + 3);
  std::unique_ptr<State> score_span =
      state->AdvanceTimeAndCopyRegisterToTIA(1, Register::X, TIA::CTRLPF);
  state = score_span->AdvanceTimeAndCopyRegisterToTIA(
      2 * kScanLineWidthClocks, Register::A, TIA::CTRLPF);
  ExpectCanScheduleBefore(score_span.get(), TIA::COLUPF);

  // A state in the middle of a line with no 1s in playfield should return 0.
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::A, TIA::PF0);
  state = state->AdvanceTimeAndSetRegister(1, Register::Y, 0x08);
  // Timing is to start this state on the second bit of left side PF2.
  std::unique_ptr<State> middle_state = state->AdvanceTimeAndCopyRegisterToTIA(
      116 + 4 - (state->range().end_time() % kScanLineWidthClocks),
      Register::Y,
      TIA::PF1);
  std::unique_ptr<State> mirrored_middle =
      middle_state->AdvanceTimeAndCopyRegisterToTIA(
          15, Register::X, TIA::CTRLPF);
  ExpectCanScheduleBefore(middle_state.get(), TIA::COLUPF);
  state = mirrored_middle->AdvanceTimeAndCopyRegisterToTIA(
      15, Register::A, TIA::CTRLPF);
  ExpectCanScheduleBefore(mirrored_middle.get(), TIA::COLUPF);

  // A state with 1 playfield pixel at its start time should return that time.
  state = state->AdvanceTimeAndCopyRegisterToTIA(
      kScanLineWidthClocks -
          (state->range().start_time() % kScanLineWidthClocks),
      Register::A,
      TIA::PF1);
  state = state->AdvanceTimeAndSetRegister(1, Register::Y, 0x01);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::Y, TIA::PF2);
  // Start leftmost_state at the last lit playfield pixel of PF2.
  std::unique_ptr<State> leftmost_state = state->AdvanceTimeAndSetRegister(
      116 - 2 + 3, Register::X, 0x01);
  state = leftmost_state->AdvanceTimeAndCopyRegisterToTIA(
      16, Register::X, TIA::CTRLPF);
  ExpectCanScheduleWithin(leftmost_state.get(), TIA::COLUPF,
      leftmost_state->range().start_time());
  std::unique_ptr<State> leftmost_mirror =
      state->AdvanceTimeAndSetRegister(13 + 31, Register::X, 0x02);
  std::unique_ptr<State> leftmost_score =
      leftmost_mirror->AdvanceTimeAndCopyRegisterToTIA(
          16, Register::X, TIA::CTRLPF);
  ExpectCanScheduleWithin(leftmost_mirror.get(), TIA::COLUPF,
      leftmost_mirror->range().start_time());
  state = leftmost_score->AdvanceTimeAndCopyRegisterToTIA(
      2 * kScanLineWidthClocks, Register::A, TIA::CTRLPF);
  ExpectCanScheduleBefore(leftmost_score.get(), TIA::COLUPF);

  // A state with a 1 on its rightmost pixel should not allow scheduling before.
  state = state->AdvanceTimeAndSetRegister(
      kScanLineWidthClocks -
          (state->range().start_time() % kScanLineWidthClocks),
      Register::Y,
      0xff);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::Y, TIA::PF0);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::Y, TIA::PF1);
  std::unique_ptr<State> rightmost_state =
      state->AdvanceTimeAndCopyRegisterToTIA(1, Register::Y, TIA::PF2);
  state = rightmost_state->AdvanceTimeAndSetRegister(
      kHBlankWidthClocks + 16, Register::X, 0x01);
  ExpectCannotScheduleBefore(rightmost_state.get(), TIA::COLUPF);
  std::unique_ptr<State> rightmost_mirror =
      state->AdvanceTimeAndCopyRegisterToTIA(84, Register::X, TIA::CTRLPF);
  state = rightmost_mirror->AdvanceTimeAndSetRegister(16, Register::X, 0x03);
  ExpectCannotScheduleBefore(rightmost_mirror.get(), TIA::COLUPF);
  std::unique_ptr<State> rightmost_score =
      state->AdvanceTimeAndCopyRegisterToTIA(16, Register::X, CTRLPF);
  state = rightmost_score->AdvanceTime(kScanLineWidthCycles);
  ExpectCanScheduleBefore(rightmost_score.get(), TIA::COLUPF);
}

TEST_F(StateTest, EarliestTimeAfterCOLUBK) {
  // COLUBK can be set anytime the background isn't rendering, so any time
  // anything else is rendering.
  std::unique_ptr<State> state(new State);
  std::unique_ptr<State> hblank_state =
      state->AdvanceTimeAndSetRegister(kScanLineWidthClocks, Register::X, 0x00);
  std::unique_ptr<State> no_pf_state =
      hblank_state->AdvanceTimeAndCopyRegisterToTIA(
          kHBlankWidthClocks, Register::X, TIA::PF0);
  // A state within the hblank should always return 0.
  ExpectCanScheduleBefore(hblank_state.get(), TIA::COLUBK);

  // A state containing an empty playfield should not allow scheduling before.
  state = no_pf_state->AdvanceTimeAndSetRegister(12, Register::Y, 0xff);
  ExpectCannotScheduleBefore(no_pf_state.get(), TIA::COLUBK);

  // A state containing a completely full playfield should return 0.
  std::unique_ptr<State> full_pf_state =
      state->AdvanceTimeAndCopyRegisterToTIA(4, Register::Y, TIA::PF1);
  state = full_pf_state->AdvanceTimeAndSetRegister(20, Register::A, 0x0f0);
  ExpectCanScheduleBefore(full_pf_state.get(),TIA::COLUBK);

  // A state with playfield on both ends and background in the middle should
  // return the right edge of the middle.
  std::unique_ptr<State> middle_pf_state =
      state->AdvanceTimeAndCopyRegisterToTIA(4, Register::A, TIA::PF2);
  state = middle_pf_state->AdvanceTimeAndCopyRegisterToTIA(
      32, Register::X, TIA::CTRLPF);
  // Should be 8 pixels of pf1 lit, followed by 16 pixels of pf2 unlit, followed
  // by 8 pixels of pf2 lit. so |start_time| + 23
  ExpectCanScheduleWithin(middle_pf_state.get(), TIA::COLUBK,
      middle_pf_state->range().start_time() + 23);

  // TODO: CTRLPF?
  // TODO: Players?
  // TODO: Missile?
  // TODO: Ball?
}

// Bit 0 is mirroring, can only be changed during hblank or left side of pf.
// Bit 1 is score mode, can be changed any time PF isn't rendering actively.
// The union of these is that CTRLPF can only be changed during HBlank.
// TODO: remaining bits?
TEST_F(StateTest, EarliestTimeAfterCTRLPF) {
  std::unique_ptr<State> hblank_state(new State);
  std::unique_ptr<State> edge_state =
      hblank_state->AdvanceTime(kHBlankWidthClocks - 5);
  ExpectCanScheduleBefore(hblank_state.get(), TIA::CTRLPF);

  std::unique_ptr<State> line_state = edge_state->AdvanceTime(10);
  ExpectCannotScheduleBefore(edge_state.get(), TIA::CTRLPF);

  std::unique_ptr<State> straddle_state =
      line_state->AdvanceTime(kFrameWidthPixels - 15);
  ExpectCannotScheduleBefore(line_state.get(), TIA::CTRLPF);

  straddle_state->AdvanceTime(kHBlankWidthClocks);
  ExpectCanScheduleWithin(straddle_state.get(), TIA::CTRLPF,
      straddle_state->range().start_time() + 10);
}

// PF0 can be set any time it is not being used to render the playfield.
TEST_F(StateTest, EarliestTimeAfterPF0) {
  std::unique_ptr<State> state(new State);
  state = state->AdvanceTimeAndSetRegister(
      kScanLineWidthClocks, Register::A, 0x00);
  std::unique_ptr<State> hblank_state = state->AdvanceTimeAndCopyRegisterToTIA(
      kScanLineWidthClocks, Register::A, TIA::CTRLPF);
  std::unique_ptr<State> pf0_state = hblank_state->AdvanceTime(
      kHBlankWidthClocks - 16);
  // State within the HBlank should return 0.
  ExpectCanScheduleBefore(hblank_state.get(), TIA::PF0);

  // State containing first range of PF0 should report when PF0 ends. We start
  // with 16 clocks of HBlank, followed by the 16 pixels of PF0, ending with
  // 16 pixels of PF1.
  std::unique_ptr<State> middle_state = pf0_state->AdvanceTime(16 + 16 + 16);
  ExpectCanScheduleWithin(pf0_state.get(), TIA::PF0,
      pf0_state->range().start_time() + 16 + 15);

  // State in the middle of the two PF0s should return 0. |middle_state| starts
  // with 16 pixels of PF1, and then consumes 16 pixels of PF2.
  std::unique_ptr<State> ending_state = middle_state->AdvanceTime(16 + 16);
  ExpectCanScheduleBefore(middle_state.get(), TIA::PF0);

  // State ending within right-side PF0 should not allow scheduling before.
  // |ending_state| starts with 16 pixels of PF2 left side and ends with 8
  // pixels of right side PF0.
  std::unique_ptr<State> straddle_state = ending_state->AdvanceTime(16 + 8);
  ExpectCannotScheduleBefore(ending_state.get(), TIA::PF0);

  // State straddling line but starting in right side PF0 should return end of
  // right-side PF0. |straddle_state| starts with 8 pixels of right side pf0 and
  // ends at the end of the next line's HBlank.
  std::unique_ptr<State> line_state =
      straddle_state->AdvanceTime(8 + 32 + 32 + kHBlankWidthClocks);
  ExpectCanScheduleWithin(straddle_state.get(), TIA::PF0,
      straddle_state->range().start_time() + 7);

  // State holding an entire line should return end of right PF0. |line_state|
  // starts on the first bit of pf0 and ends on the next line start.
  state = line_state->AdvanceTimeAndSetRegister(
      kFrameWidthPixels, Register::X, 0x01);
  ExpectCanScheduleWithin(line_state.get(), TIA::PF0,
      line_state->range().start_time() + 16 + 32 + 32 + 15);

  // Turn mirroring on. State within HBlank should still return 0.
  std::unique_ptr<State> hblank_mirror =
      state->AdvanceTimeAndCopyRegisterToTIA(1, Register::X, TIA::CTRLPF);
  std::unique_ptr<State> left_mirror =
      hblank_mirror->AdvanceTime(kHBlankWidthClocks - 1);
  ExpectCanScheduleBefore(hblank_mirror.get(), TIA::PF0);

  // State that holds all of the scan line up to right side mirrored pf0 should
  // return left side pf0 for in-range queries.
  std::unique_ptr<State> right_span_mirror =
      left_mirror->AdvanceTime(16 + 32 + 32 + 32 + 32);
  ExpectCanScheduleWithin(left_mirror.get(), TIA::PF0,
      left_mirror->range().start_time() + 15);

  right_span_mirror->AdvanceTime(16 + kHBlankWidthClocks);
  ExpectCanScheduleWithin(right_span_mirror.get(), TIA::PF0,
      right_span_mirror->range().start_time() + 15);
}

// PF1 can be set any time it is not being used to render the playfield.
TEST_F(StateTest, EarliestTimeAfterPF1) {
  // State within HBlank should always return 0.
  std::unique_ptr<State> state(new State);
  state = state->AdvanceTimeAndSetRegister(1, Register::X, 0x00);
  std::unique_ptr<State> hblank_state =
      state->AdvanceTimeAndCopyRegisterToTIA(1, Register::X, TIA::CTRLPF);
  std::unique_ptr<State> hblank_and_pf0 =
      hblank_state->AdvanceTime(kHBlankWidthClocks - 12);
  ExpectCanScheduleBefore(hblank_state.get(), TIA::PF1);

  // State containing HBlank and PF0 should always return 0. Test range starts
  // 10 pixels in to HBlank and ends 8 pixels in to PF0.
  std::unique_ptr<State> left_pf0_and_pf1 = hblank_and_pf0->AdvanceTime(10 + 8);
  ExpectCanScheduleBefore(hblank_and_pf0.get(), TIA::PF1);

  // State containing PF0 and PF1 should return kInfinity. Test range starts 8
  // pixels within left PF0 and ends 8 pixels within left PF1.
  std::unique_ptr<State> within_left_pf1 = left_pf0_and_pf1->AdvanceTime(8 + 8);
  ExpectCannotScheduleBefore(left_pf0_and_pf1.get(), TIA::PF1);

  // State entirely within PF1 should return kInfinity. Test range covers the
  // middle 16 pixels of left side PF1, leaving 8 on each side.
  std::unique_ptr<State> left_pf1_and_pf2 = within_left_pf1->AdvanceTime(16);
  ExpectCannotScheduleBefore(within_left_pf1.get(), TIA::PF1);

  // State starting within left-side PF1 and ending outside of it should return
  // the last pixel in left-side PF1. Test range covers last 8 pixels of left
  // side pf1 and 17 pixels of left side PF2.
  std::unique_ptr<State> left_pf2_and_right_pf0 =
      left_pf1_and_pf2->AdvanceTime(8 + 17);
  ExpectCanScheduleWithin(left_pf1_and_pf2.get(), TIA::PF1,
      kHBlankWidthClocks + 16 + 31);

  // State between left-side PF1 and right-side PF1 should return 0. Test range
  // covers 15 pixels of left side PF2 and 3 pixels of right-side PF0.
  std::unique_ptr<State> right_pf0_and_pf1 =
      left_pf2_and_right_pf0->AdvanceTime(15 + 3);
  ExpectCanScheduleBefore(left_pf2_and_right_pf0.get(), TIA::PF1);

  // State starting in right-side PF0 and ending in right-side PF1 should return
  // kInfinity. Test range covers 13 pixels of right_side PF0 and 23 pixels of
  // right-side PF1.
  std::unique_ptr<State> right_pf1_to_hblank =
      right_pf0_and_pf1->AdvanceTime(13 + 23);
  ExpectCannotScheduleBefore(right_pf0_and_pf1.get(), TIA::PF1);

  // State starting in right-side PF1 and ending in next line HBlank should
  // return last pixel in right-side PF1. Test range covers 9 pixels of right
  // side PF1, 32 pixels of PF2, and 11 pixels of next line HBlank.
  state = right_pf1_to_hblank->AdvanceTime(9 + 32 + 11);
  ExpectCanScheduleWithin(right_pf1_to_hblank.get(), TIA::PF1,
      right_pf1_to_hblank->range().start_time() + 8);

  std::unique_ptr<State> left_pf1_to_right_pf1 =
      state->AdvanceTime(kHBlankWidthClocks - 11 + 16 + 9);

  // State starting in left-side PF1 and ending in right-side PF1 should return
  // kInfinity. Test range covers 23 pixels of left pf1 through to 19 pixels of
  // right PF1.
  std::unique_ptr<State> right_pf1_to_left_pf1 =
      left_pf1_to_right_pf1->AdvanceTime(23 + 32 + 16 + 19);
  ExpectCannotScheduleBefore(left_pf1_to_right_pf1.get(), TIA::PF1);

  // State starting in right-side PF1 and ending in left-side PF1 on next line
  // should return kInfinity. Test range covers 13 pixels of right PF1 through
  // to 14 pixels of left side PF1 on the next line.
  state = right_pf1_to_left_pf1->AdvanceTimeAndSetRegister(
      13 + 32 + kHBlankWidthClocks + 16 + 14, Register::Y, 0x01);
  ExpectCannotScheduleBefore(right_pf1_to_left_pf1.get(), TIA::PF1);

  // Turn mirroring on. Repeat a few of the above. Start with HBlank only.
  std::unique_ptr<State> hblank_mirror =
      state->AdvanceTimeAndCopyRegisterToTIA(18 + 32 + 32 + 32 + 16,
          Register::Y, TIA::CTRLPF);
  std::unique_ptr<State> hblank_and_pf1_mirror = hblank_mirror->AdvanceTime(19);
  ExpectCanScheduleBefore(hblank_mirror.get(), TIA::PF1);

  // HBlank and left-side PF1. Test range is balance of HBlank to 7 pixels in to
  // left-side PF1.
  std::unique_ptr<State> left_pf1_to_middle_mirror =
      hblank_and_pf1_mirror->AdvanceTime(kHBlankWidthClocks - 19 + 16 + 7);
  ExpectCannotScheduleBefore(hblank_and_pf1_mirror.get(), TIA::PF1);

  // Left-side PF1 to middle where PF1 would be if mirroring were off. Test
  // range is 25 pixels of left-side PF1, through left-side PF2, then 19 pixels
  // in to mirrored PF2.
  std::unique_ptr<State> middle_mirror =
      left_pf1_to_middle_mirror->AdvanceTime(25 + 32 + 19);
  ExpectCanScheduleWithin(left_pf1_to_middle_mirror.get(), TIA::PF1,
       left_pf1_to_middle_mirror->range().start_time() + 24);

  // A range entirely contained within the right-side PF1 area if mirroring were
  // turned off. Test range starts 19 pixels in to right-side mirrored PF2 and
  // covers the next 7 pixels.
  std::unique_ptr<State> middle_to_right_pf1_mirror =
      middle_mirror->AdvanceTime(7);
  ExpectCanScheduleBefore(middle_mirror.get(), TIA::PF1);

  // Middle to right-side PF1. Test range is remaining 6 pixels of right-side
  // mirrored PF2 into 13 pixels of right-side mirrored PF1.
  std::unique_ptr<State> right_pf1_to_hblank_mirror =
      middle_to_right_pf1_mirror->AdvanceTime(6 + 13);
  ExpectCannotScheduleBefore(middle_to_right_pf1_mirror.get(), TIA::PF1);

  // Right-side PF1 to HBlank on next line. Test range is remaining 19 pixels of
  // right-side mirrored PF1, through 16 pixels of mirrored PF0, and 10 pixels
  // of HBlank.
  right_pf1_to_hblank_mirror->AdvanceTime(19 + 16 + 10);
  ExpectCanScheduleWithin(right_pf1_to_hblank_mirror.get(), TIA::PF1,
      right_pf1_to_hblank_mirror->range().start_time() + 18);
}

// PF2 can be set any time it is not being used to render the playfield.
TEST_F(StateTest, EarliestTimeAfterPF2) {
  std::unique_ptr<State> state(new State);
  state = state->AdvanceTimeAndSetRegister(1, Register::Y, 0x00);
  std::unique_ptr<State> hblank_state =
      state->AdvanceTimeAndCopyRegisterToTIA(1, Register::Y, TIA::CTRLPF);

  // HBlank should allow scheduling before. Test range is 2 clocks in to HBlank
  // to all but 13 clocks of HBlank.
  std::unique_ptr<State> hblank_to_pf1 = hblank_state->AdvanceTime(
      kHBlankWidthClocks - 2 - 13);
  ExpectCanScheduleBefore(hblank_state.get(), TIA::PF2);

  // HBlank, PF0, and PF1 should allow scheduling before. Test range is 13
  // clocks of HBlank followed by PF0 and 7 pixels of PF1.
  std::unique_ptr<State> pf1_to_pf2 = hblank_to_pf1->AdvanceTime(13 + 16 + 7);
  ExpectCanScheduleBefore(hblank_to_pf1.get(), TIA::PF2);

  // State ending in left-side PF2 should not allow scheduling before. Test
  // range is 25 pixels of PF1 and 19 pixels of left-side PF2.
  std::unique_ptr<State> within_left_pf2 = pf1_to_pf2->AdvanceTime(25 + 19);
  ExpectCannotScheduleBefore(pf1_to_pf2.get(), TIA::PF2);

  // State entirely within left-side PF2 should not allow scheduling before.
  // Test range is 5 pixels of left-side PF2, starting from pixel 19 and ending
  // on pixel 24.
  std::unique_ptr<State> left_pf2_to_right_pf0 =
      within_left_pf2->AdvanceTime(5);
  ExpectCannotScheduleBefore(within_left_pf2.get(), TIA::PF2);

  // State starting in left-side PF2 and ending in right-side PF0 should allow
  // scheduling any time after PF2 ends. Test range is 8 pixels of left-side PF2
  // and 4 pixels of right-side PF0.
  std::unique_ptr<State> between_left_and_right =
      left_pf2_to_right_pf0->AdvanceTime(8 + 4);
  ExpectCanScheduleWithin(left_pf2_to_right_pf0.get(), TIA::PF2,
      left_pf2_to_right_pf0->range().start_time() + 7);

  // State in between left and right PF2 should allow scheduling before. Test
  // range is 12 pixels of right-side PF0 to 15 pixels of right-side PF1.
  std::unique_ptr<State> contains_right_pf2 =
      between_left_and_right->AdvanceTimeAndSetRegister(
          12 + 15, Register::A, 0x01);
  ExpectCanScheduleBefore(between_left_and_right.get(), TIA::PF2);

  // State entirely containing right-side PF2 should allow scheduling any time
  // after PF2 ends. Test range is 17 pixels of right-side PF1, all 32 pixels
  // of right-side PF2, and 23 pixels of next line HBlank.
  std::unique_ptr<State> mirror_on_hblank =
      contains_right_pf2->AdvanceTimeAndCopyRegisterToTIA(
          17 + 32 + 23, Register::A, TIA::CTRLPF);
  ExpectCanScheduleWithin(contains_right_pf2.get(), TIA::PF2,
      contains_right_pf2->range().start_time() + 17 + 31);

  // State within HBlank should still allow scheduling before regardless of
  // mirroring. Test range is from 23 pixels of HBlank to 48, so covers 25
  // pixels of HBlank.
  std::unique_ptr<State> contains_left_pf2 = mirror_on_hblank->AdvanceTime(25);
  ExpectCanScheduleBefore(mirror_on_hblank.get(), TIA::PF2);

  // State containing left and right-side PF2 should return the last pixel of
  // right-side mirrored PF2. Test range is remaining 20 pixels of HBlank,
  // 16 of PF0, 32 of PF1, 32 of left-side PF2, 32 of right-side PF2, and 12 of
  // right-side mirrored PF1.
  std::unique_ptr<State> line_after_mirrored_pf2 =
      contains_left_pf2->AdvanceTime(20 + 16 + 32 + 32 + 32 + 12);
  ExpectCanScheduleWithin(contains_left_pf2.get(), TIA::PF2,
      contains_left_pf2->range().end_time() - 13);

  // State that occurs after PF2 on a mirrored line should allow scheduling
  // before. Test range covers 20 pixels of mirrored PF1 and 4 pixels of
  // mirrored right-side PF0.
  line_after_mirrored_pf2->AdvanceTime(20 + 4);
  ExpectCanScheduleBefore(line_after_mirrored_pf2.get(), TIA::PF2);
}

TEST_F(StateTest, EarliestTimeAfterWithEndTime) {
  std::unique_ptr<State> state(new State);

  // Default Range for a new State is [0, kFrameSizeClocks). It should therefore
  // not be possible to schedule something dependent on HBlank any time before
  // the last HBlank in the frame.
  EXPECT_EQ(kInfinity, state->EarliestTimeAfter(Spec(TIA::CTRLPF, 0x00,
      Range(0, kHBlankWidthClocks + 25))));

  // If we set the |end_time| to something containing only the first HBlank this
  // should allow scheduling before.
  EXPECT_EQ(0, state->EarliestTimeAfterWithEndTime(
      Spec(TIA::CTRLPF, 0x00, Range(0, kScanLineWidthClocks)),
          kHBlankWidthClocks));
  EXPECT_EQ(0, state->EarliestTimeAfterWithEndTime(
      Spec(TIA::CTRLPF, 0x00, Range(0, kScanLineWidthClocks)), 2));

  // Setting the |end_time| to end outside of the first HBlank should result in
  // an error again.
  EXPECT_EQ(kInfinity, state->EarliestTimeAfterWithEndTime(
      Spec(TIA::CTRLPF, 0x00, Range(0, kHBlankWidthClocks + 25)),
          kHBlankWidthClocks + 1));
  EXPECT_EQ(kInfinity, state->EarliestTimeAfterWithEndTime(
      Spec(TIA::CTRLPF, 0x00, Range(0, kHBlankWidthClocks + 25)),
          kScanLineWidthClocks));

  state = state->AdvanceTimeAndSetRegister(1, Register::X, 0x00);
  state = state->AdvanceTimeAndCopyRegisterToTIA(1, Register::X, TIA::CTRLPF);

  // Default range() for |state| should not let us set PF0 on anything but the
  // last line.
  EXPECT_EQ(kInfinity, state->EarliestTimeAfter(
      Spec(TIA::PF0, 0x00, Range(0, kScanLineWidthClocks))));
  EXPECT_EQ(kFrameSizeClocks - 32 - 32 - 1, state->EarliestTimeAfter(
      Spec(TIA::PF0, 0x00, Range(0, kFrameSizeClocks))));

  // Setting the |end_time| to beyond left-side PF0 should allow us to schedule
  // PF0 changes any time after PF0 finishes rendering on the first scanline.
  EXPECT_EQ(kHBlankWidthClocks + 15, state->EarliestTimeAfterWithEndTime(
      Spec(TIA::PF0, 0x00, Range(0, kScanLineWidthClocks)),
          kHBlankWidthClocks + 16 + 32));

  // Setting the |end_time| back to the HBlank should permit changes to PF0 to
  // be scheduled before.
  EXPECT_EQ(0, state->EarliestTimeAfterWithEndTime(
      Spec(TIA::PF0, 0x00, Range(0, kScanLineWidthClocks)), 25));
}

TEST(StateDeathTest, AdvanceTimeZero) {
  std::unique_ptr<State> state(new State);
  EXPECT_DEATH(state->AdvanceTime(0), "delta > 0");
}

TEST(StateDeathTest, AdvanceTimeAndCopyRegisterToTIAUnknownRegister) {
  std::unique_ptr<State> state(new State);
  EXPECT_DEATH(
      state->AdvanceTimeAndCopyRegisterToTIA(1, Register::X, TIA::PF0),
      "register_known");
}

TEST(StateDeathTest, TIAUnknownAccess) {
  std::unique_ptr<State> state(new State);
  EXPECT_DEATH(state->tia(TIA::HMM0), "tia_known");
}

TEST(StateDeathTest, RegisterUnknownAccess) {
  std::unique_ptr<State> state(new State);
  EXPECT_DEATH(state->reg(Register::Y), "register_known");
}

TEST(StateDeathTest, RegisterAUnknownAccess) {
  std::unique_ptr<State> state(new State);
  EXPECT_DEATH(state->a(), "register_known");
}

TEST(StateDeathTest, RegisterXUnknownAccess) {
  std::unique_ptr<State> state(new State);
  EXPECT_DEATH(state->x(), "register_known");
}

TEST(StateDeathTest, RegisteryYUnknownAccess) {
  std::unique_ptr<State> state(new State);
  EXPECT_DEATH(state->y(), "register_known");
}

}  // namespace vcsmc

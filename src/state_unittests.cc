#include "colu_strip.h"
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

  void ExpectUnpainted(const ColuStrip* colu_strip) const {
    for (uint32 i = 0; i < kFrameWidthPixels; ++i)
      EXPECT_EQ(kColuUnpainted, colu_strip->colu(i));
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

TEST_F(StateTest, EarliestTimeAfterCTRLPF) {
}

TEST_F(StateTest, EarliestTimeAfterCOLUPF) {
  // COLUPF can be set any time the TIA is not rendering the playfield color,
  // i.e. any time the TIA is not rendering a 1 in the playfield.

  // Ideas for test cases:

  // A state within the HBLANK should always return 0.

  // A state that straddles a line break and has pf1 painting on the previous
  // line should return the color clock of the last pixel in that line.

  // A state in the middle of a line with no 1s in playfield should return 0.

  // A state with 1 pixel color followed by 0s should return that time.

  // A state with a 1 on its rightmost pixel should return kInfinity

  // Test on the left and right of the field with mirroring turned on as well.

  std::unique_ptr<State> state(new State);
}

TEST_F(StateTest, EarliestTimeAfterCOLUBK) {
}

TEST_F(StateTest, EarliestTimeAfterPF0) {
}

TEST_F(StateTest, EarliestTimeAfterPF1) {
}

TEST_F(StateTest, EarliestTimeAfterPF2) {
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

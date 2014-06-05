#include "state.h"

#include "gtest/gtest.h"

#include "colu_strip.h"

TEST(StateTest, InitialStateHasFullProgramRange) {
  vcsmc::State state;
  EXPECT_EQ(0, state.range().start_time());
  EXPECT_EQ(vcsmc::kFrameSizeClocks, state.range().end_time());
}

TEST(StateTest, InitialStateHasAllRegistersUnknown) {
  vcsmc::State state;
  EXPECT_FALSE(state.register_known(vcsmc::Register::A));
  EXPECT_FALSE(state.register_known(vcsmc::Register::X));
  EXPECT_FALSE(state.register_known(vcsmc::Register::Y));
}

TEST(StateTest, InitialStateHasOnlyStrobeTIAValuesKnown) {
  vcsmc::State state;
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::VSYNC));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::VBLANK));
  EXPECT_TRUE(state.tia_known(vcsmc::TIA::WSYNC));
  EXPECT_TRUE(state.tia_known(vcsmc::TIA::RSYNC));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::NUSIZ0));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::NUSIZ1));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::COLUP0));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::COLUP1));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::COLUPF));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::COLUBK));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::CTRLPF));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::REFP0));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::REFP1));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::PF0));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::PF1));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::PF2));
  EXPECT_TRUE(state.tia_known(vcsmc::TIA::RESP0));
  EXPECT_TRUE(state.tia_known(vcsmc::TIA::RESP1));
  EXPECT_TRUE(state.tia_known(vcsmc::TIA::RESM0));
  EXPECT_TRUE(state.tia_known(vcsmc::TIA::RESM1));
  EXPECT_TRUE(state.tia_known(vcsmc::TIA::RESBL));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::AUDC0));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::AUDC1));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::AUDF0));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::AUDF1));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::AUDV0));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::AUDV1));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::GRP0));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::GRP1));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::ENAM0));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::ENAM1));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::ENABL));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::HMP0));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::HMP1));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::HMM0));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::HMM1));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::HMBL));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::VDELP0));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::VDELP1));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::VDELBL));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::RESMP0));
  EXPECT_FALSE(state.tia_known(vcsmc::TIA::RESMP1));
  EXPECT_TRUE(state.tia_known(vcsmc::TIA::HMOVE));
  EXPECT_TRUE(state.tia_known(vcsmc::TIA::HMCLR));
  EXPECT_TRUE(state.tia_known(vcsmc::TIA::CXCLR));
}

TEST(StateTest, PaintIntoBeforeStateRange) {
}

TEST(StateTest, PaintIntoAfterStateRange) {
}

TEST(StateTest, PaintBackgroundPartial) {
}

TEST(StateTest, PaintBackgroundEntire) {
}

TEST(StateTest, PaintPlayfieldPartial) {
}

TEST(StateTest, PaintPlayfieldEntire) {
}


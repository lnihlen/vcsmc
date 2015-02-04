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
};

TEST_F(StateTest, InitialStateHasFullProgramRange) {
  State state;
  EXPECT_EQ(0, state.range().start_time());
  EXPECT_EQ(kScreenSizeClocks, state.range().end_time());
  uint8 tia_values[TIA::TIA_COUNT];
  State state_value(tia_values);
  EXPECT_EQ(0, state_value.range().start_time());
  EXPECT_EQ(kScreenSizeClocks, state_value.range().end_time());
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
  EXPECT_EQ(Range(kScanLineWidthClocks, kScreenSizeClocks), advance->range());
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
  EXPECT_EQ(Range(32, kScreenSizeClocks), advance->range());
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
  EXPECT_EQ(Range(128, kScreenSizeClocks), advance->range());
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
  std::unique_ptr<State> state(new State);
  // A strobe to WSYNC on first operation of line should consume entire line.
  std::unique_ptr<State> whole_line = state->AdvanceTimeAndSetRegister(
      0, Register::A, 0);
  state = whole_line->AdvanceTimeAndCopyRegisterToTIA(
      1, Register::A, TIA::WSYNC);
  EXPECT_EQ(0, whole_line->range().start_time());
  EXPECT_EQ(kScanLineWidthClocks, whole_line->range().end_time());
  EXPECT_EQ(kScanLineWidthClocks, state->range().start_time());

  // Test in HBLANK, in scanout, and on last clock of the line.
  std::unique_ptr<State> hblank = state->AdvanceTime(10);
  state = hblank->AdvanceTimeAndCopyRegisterToTIA(
      kInfinity, Register::A, TIA::WSYNC);
  EXPECT_EQ(10 + kScanLineWidthClocks, hblank->range().start_time());
  EXPECT_EQ(2 * kScanLineWidthClocks, hblank->range().end_time());

  std::unique_ptr<State> scanout = state->AdvanceTime(kHBlankWidthClocks + 67);
  state = scanout->AdvanceTimeAndCopyRegisterToTIA(0, Register::A, TIA::WSYNC);
  EXPECT_EQ(3 * kScanLineWidthClocks, scanout->range().end_time());

  std::unique_ptr<State> last = state->AdvanceTime(kScanLineWidthClocks - 1);
  state = last->AdvanceTimeAndCopyRegisterToTIA(2, Register::A, TIA::WSYNC);
  EXPECT_EQ(4 * kScanLineWidthClocks, last->range().end_time());
}

TEST_F(StateTest, AdvanceTimeAndCopyRegisterToTIAStrobeRSYNC) {
  // TODO
}

// This lovely table is copied from Andrew Towers, who wrote some very detailed
// notes about the TIA. I found a copy here:
// http://www.atarihq.com/danb/files/TIA_HW_Notes.txt
//
// CPU  CLK Pixel  Main Close Medium  Far  PF
//
// 0      0  -  1    17    33    65    -
// ...
// 22    66  -  1    17    33    65    -
// 22.6 --------------------------------------------------------
// 23    69     1     6    22    38    70  0.25
// 24    72     4     9    25    41    73  1
// 25    75     7    12    28    44    76  1.75
// 26    78    10    15    31    47    79  2.5
// 27    81    13    18    34    50    82  3.25
// 28    84    16    21    37    53    85  3
// 29    87    19    24    40    56    88
// 30    90    22    27    43    59    91
// 31    93    25    30    46    62    94
// 32    96    28    33    49    65    97
// 33    99    31    36    52    68   100
// 34   102    34    39    55    71   103
// 35   105    37    42    58    74   106
// 36   108    40    45    61    77   109
// 37   111    43    48    64    80   112
// 38   114    46    51    67    83   115
// 39   117    49    54    70    86   118
// 40   120    52    57    73    89   121
// 41   123    55    60    76    92   124
// 42   126    58    63    79    95   127
// 43   129    61    66    82    98   130
// 44   132    64    69    85   101   133
// 45   135    67    72    88   104   136
// 46   138    70    75    91   107   139
// 47   141    73    78    94   110   142
// 48   144    76    81    97   113   145
// 49   147    79    84   100   116   148
// 50   150    82    87   103   119   151
// 51   153    85    90   106   122   154
// 52   156    88    93   109   125   157
// 53   159    91    96   112   128     0
// 54   162    94    99   115   131     3
// 55   165    97   102   118   134     6
// 56   168   100   105   121   137     9
// 57   171   103   108   124   140    12
// 58   174   106   111   127   143    15
// 59   177   109   114   130   146    18
// 60   180   112   117   133   149    21
// 61   183   115   120   136   152    24
// 62   186   118   123   139   155    27
// 63   189   121   126   142   158    30
// 64   192   124   129   145     1    33
// 65   195   127   132   148     4    36
// 66   198   130   135   151     7    39
// 67   201   133   138   154    10    42
// 68   204   136   141   157    13    45
// 69   207   139   144     0    16    48
// 70   210   142   147     3    19    51
// 71   213   145   150     6    22    54
// 72   216   148   153     9    25    57
// 73   219   151   156    12    28    60
// 74   222   154   159    15    31    63
// 75   225   157     2    18    34    66
// 76   228     0     5    21    37    69
// ----------------------------------------------------- Start HBLANK
//

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

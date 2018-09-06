#include "state.h"
#include "gtest/gtest.h"

#include "codon.h"
#include "constants.h"

namespace {

bool IsLoad(uint8 op) {
  if (op == vcsmc::LDA_Immediate ||
      op == vcsmc::LDX_Immediate ||
      op == vcsmc::LDY_Immediate) {
    return true;
  }
  return false;
}

bool IsStore(uint8 op) {
  if (op == vcsmc::STA_ZeroPage ||
      op == vcsmc::STX_ZeroPage ||
      op == vcsmc::STY_ZeroPage) {
    return true;
  }
  return false;
}

bool IsLoadAndStorePair(uint8 op_load, uint8 op_store) {
  if (op_load == vcsmc::LDA_Immediate) {
    return op_store == vcsmc::STA_ZeroPage;
  }
  if (op_load == vcsmc::LDX_Immediate) {
    return op_store == vcsmc::STX_ZeroPage;
  }
  if (op_load == vcsmc::LDY_Immediate) {
    return op_store == vcsmc::STY_ZeroPage;
  }
  return false;
}

}

namespace vcsmc {

TEST(StateTest, SequenceWaitTwo) {
  Codon wait_codon_two = MakeWaitCodon(2);
  State state;
  Snippet snippet = state.Sequence(wait_codon_two);
  ASSERT_EQ(1u, snippet.size);
  EXPECT_EQ(NOP_Implied, snippet.bytecode[0]);
  EXPECT_EQ(2u, snippet.duration);
  EXPECT_FALSE(snippet.should_advance_register_rotation);
}

TEST(StateTest, SequenceWaitThree) {
  Codon wait_codon_three = MakeWaitCodon(3);
  State state;
  Snippet snippet = state.Sequence(wait_codon_three);
  // Don't care about the zero page address BIT tests, but there does need to
  // be an address there, making the length 2 bytes.
  ASSERT_EQ(2u, snippet.size);
  EXPECT_EQ(BIT_ZeroPage, snippet.bytecode[0]);
  EXPECT_EQ(3u, snippet.duration);
  EXPECT_FALSE(snippet.should_advance_register_rotation);
}

TEST(StateTest, SequenceWaitManyEven) {
  Codon wait_codon_many = MakeWaitCodon(98);
  State state;
  Snippet snippet = state.Sequence(wait_codon_many);
  ASSERT_EQ(49u, snippet.size);
  for (auto i = 0; i < 49; ++i) {
    EXPECT_EQ(NOP_Implied, snippet.bytecode[i]);
  }
  EXPECT_EQ(98u, snippet.duration);
  EXPECT_FALSE(snippet.should_advance_register_rotation);
}

TEST(StateTest, SequenceWaitManyOdd) {
  Codon wait_codon_many = MakeWaitCodon(111);
  State state;
  Snippet snippet = state.Sequence(wait_codon_many);
  ASSERT_EQ(56u, snippet.size);
  for (auto i = 0; i < 54; ++i) {
    EXPECT_EQ(NOP_Implied, snippet.bytecode[i]);
  }
  EXPECT_EQ(BIT_ZeroPage, snippet.bytecode[54]);
  EXPECT_EQ(111u, snippet.duration);
  EXPECT_FALSE(snippet.should_advance_register_rotation);
}

TEST(StateTest, SequenceWaitMax) {
  Codon wait_codon_max = MakeWaitCodon(255);
  State state;
  Snippet snippet = state.Sequence(wait_codon_max);
  ASSERT_EQ(128u, snippet.size);
  for (auto i = 0; i < 126; ++i) {
    EXPECT_EQ(NOP_Implied, snippet.bytecode[i]);
  }
  EXPECT_EQ(BIT_ZeroPage, snippet.bytecode[126]);
  EXPECT_EQ(255u, snippet.duration);
  EXPECT_FALSE(snippet.should_advance_register_rotation);
}

TEST(StateTest, SequenceNoChangeNoMask) {
  State state;
  state.tia()[PF2] = 0xa2;
  Codon redundant_codon = MakeTIACodon(kSetPF2, 0xa2);
  Snippet snippet = state.Sequence(redundant_codon);
  EXPECT_EQ(0u, snippet.size);
  EXPECT_EQ(0u, snippet.duration);
  EXPECT_FALSE(snippet.should_advance_register_rotation);
}

TEST(StateTest, SequenceNoChangeWithMask) {
  State state;
  state.tia()[ENAM1] = 0b11111101;
  Codon redundant_codon = MakeTIACodon(kSetENAM1, 0);
  Snippet snippet = state.Sequence(redundant_codon);
  EXPECT_EQ(0u, snippet.size);
  EXPECT_EQ(0u, snippet.duration);
  EXPECT_FALSE(snippet.should_advance_register_rotation);
}

TEST(StateTest, SequenceSharedTIANoChange) {
  State state;
  state.tia()[NUSIZ1] = 0b00101011;
  Codon redundant_shared = MakeTIACodon(kSetNUSIZ1_M1, 0x20);
  Snippet snippet = state.Sequence(redundant_shared);
  EXPECT_EQ(0u, snippet.size);
  EXPECT_EQ(0u, snippet.duration);
  EXPECT_FALSE(snippet.should_advance_register_rotation);
}

TEST(StateTest, SequenceSkipLoadExactMatch) {
  State state;
  state.registers()[X] = 0xa5;
  state.set_current_time(4321);
  Codon reuse_x_codon = MakeTIACodon(kSetGRP0, 0xa5);
  Snippet snippet = state.Sequence(reuse_x_codon);
  ASSERT_EQ(2u, snippet.size);
  EXPECT_EQ(STX_ZeroPage, snippet.bytecode[0]);
  EXPECT_EQ(GRP0, snippet.bytecode[1]);
  EXPECT_EQ(3u, snippet.duration);
  EXPECT_TRUE(snippet.should_advance_register_rotation);
}

TEST(StateTest, SequenceSkipLoadMaskMatch) {
  State state;
  // REFP1 only cares about bit 3, which we set to 1 here.
  state.registers()[A] = 0b10101011;
  state.set_current_time(0xfeedfeed);
  Codon reuse_a_codon = MakeTIACodon(kSetREFP1, 0x08);
  Snippet snippet = state.Sequence(reuse_a_codon);
  ASSERT_EQ(2u, snippet.size);
  EXPECT_EQ(STA_ZeroPage, snippet.bytecode[0]);
  EXPECT_EQ(REFP1, snippet.bytecode[1]);
  EXPECT_EQ(3u, snippet.duration);
  EXPECT_TRUE(snippet.should_advance_register_rotation);
}

TEST(StateTest, SequenceRegisterRotation) {
  State state;
  state.registers()[A] = 0x42;
  state.register_last_used()[A] = 23;
  state.registers()[X] = 0xff;
  state.register_last_used()[X] = 117;
  state.registers()[Y] = 0xed;
  state.register_last_used()[Y] = 4;
  state.set_current_time(140);
  Codon register_rotate = MakeTIACodon(kSetGRP0, 0x02);
  Snippet snippet = state.Sequence(register_rotate);
  ASSERT_EQ(4u, snippet.size);
  EXPECT_EQ(LDY_Immediate, snippet.bytecode[0]);
  EXPECT_EQ(0x02u, snippet.bytecode[1]);
  EXPECT_EQ(STY_ZeroPage, snippet.bytecode[2]);
  EXPECT_EQ(GRP0, snippet.bytecode[3]);
  EXPECT_EQ(5u, snippet.duration);
  EXPECT_TRUE(snippet.should_advance_register_rotation);
}

TEST(StateTest, SequenceStrobeShouldNotAdvanceRotation) {
  State state;
  Codon strobe = MakeTIACodon(kStrobeRESP0, 0xff);
  Snippet snippet = state.Sequence(strobe);
  ASSERT_EQ(2u, snippet.size);
  EXPECT_TRUE(IsStore(snippet.bytecode[0]));
  EXPECT_EQ(RESP0, snippet.bytecode[1]);
  EXPECT_EQ(3u, snippet.duration);
  EXPECT_FALSE(snippet.should_advance_register_rotation);
}

// CTRLPF, NUSIZ0, and NUSIZ1 all pack more than one state value into a shared
// register. We test that values not being modified in the same register are
// preserved.
TEST(StateTest, SequenceSharedTIAPreserveState) {
  State state;
  state.tia()[CTRLPF] = 0b00010101;
  Codon shared = MakeTIACodon(kSetCTRLPF_BALL, 0x20);
  Snippet snippet = state.Sequence(shared);
  ASSERT_EQ(4u, snippet.size);
  EXPECT_TRUE(IsLoad(snippet.bytecode[0]));
  EXPECT_EQ(0b00100101u, snippet.bytecode[1]);
  EXPECT_TRUE(IsStore(snippet.bytecode[2]));
  EXPECT_TRUE(IsLoadAndStorePair(snippet.bytecode[0], snippet.bytecode[2]));
  EXPECT_EQ(CTRLPF, snippet.bytecode[3]);
  EXPECT_EQ(5u, snippet.duration);
  EXPECT_TRUE(snippet.should_advance_register_rotation);
}

TEST(StateTest, SequenceSharedTIAReuseRegister) {
  State state;
  state.tia()[CTRLPF] = 0b10111010;
  state.registers()[X] = 0b01110000;
  Codon reuse_reg = MakeTIACodon(kSetCTRLPF_SCORE, 0);
  Snippet snippet = state.Sequence(reuse_reg);
  ASSERT_EQ(2u, snippet.size);
  EXPECT_EQ(STX_ZeroPage, snippet.bytecode[0]);
  EXPECT_EQ(CTRLPF, snippet.bytecode[1]);
  EXPECT_EQ(3u, snippet.duration);
  EXPECT_TRUE(snippet.should_advance_register_rotation);
}

TEST(StateTest, ApplyEmptySnippetOnEmptyState) {
  State state;
  Snippet snippet;
  state.Apply(snippet);
  for (size_t i = 0; i < TIA_COUNT; ++i) {
    EXPECT_EQ(0u, state.tia()[i]);
  }
  EXPECT_EQ(0u, state.registers()[A]);
  EXPECT_EQ(0u, state.registers()[X]);
  EXPECT_EQ(0u, state.registers()[Y]);
  EXPECT_EQ(0u, state.register_last_used()[A]);
  EXPECT_EQ(0u, state.register_last_used()[X]);
  EXPECT_EQ(0u, state.register_last_used()[Y]);
  EXPECT_EQ(0u, state.current_time());
}

}  // namespace vcsmc

#include "state.h"
#include "gtest/gtest.h"

#include <cstring>

#include "codon.h"
#include "constants.h"
#include "kernel.h"

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

void ExpectEmptyState(vcsmc::State& state) {
  for (size_t i = 0; i < vcsmc::TIA_COUNT; ++i) {
    EXPECT_EQ(0u, state.tia()[i]);
  }
  EXPECT_EQ(0u, state.registers()[vcsmc::A]);
  EXPECT_EQ(0u, state.registers()[vcsmc::X]);
  EXPECT_EQ(0u, state.registers()[vcsmc::Y]);
  EXPECT_EQ(0u, state.register_last_used()[vcsmc::A]);
  EXPECT_EQ(0u, state.register_last_used()[vcsmc::X]);
  EXPECT_EQ(0u, state.register_last_used()[vcsmc::Y]);
}

}

namespace vcsmc {

TEST(StateTest, TranslateWaitTwo) {
  Codon wait_codon_two = MakeWaitCodon(2);
  State state;
  Snippet snippet = state.Translate(wait_codon_two);
  ASSERT_EQ(1u, snippet.size);
  EXPECT_EQ(NOP_Implied, snippet.bytecode[0]);
  EXPECT_EQ(2u, snippet.duration);
  EXPECT_FALSE(snippet.should_advance_register_rotation);
}

TEST(StateTest, TranslateWaitThree) {
  Codon wait_codon_three = MakeWaitCodon(3);
  State state;
  Snippet snippet = state.Translate(wait_codon_three);
  // Don't care about the zero page address BIT tests, but there does need to
  // be an address there, making the length 2 bytes.
  ASSERT_EQ(2u, snippet.size);
  EXPECT_EQ(BIT_ZeroPage, snippet.bytecode[0]);
  EXPECT_EQ(3u, snippet.duration);
  EXPECT_FALSE(snippet.should_advance_register_rotation);
}

TEST(StateTest, TranslateWaitManyEven) {
  Codon wait_codon_many = MakeWaitCodon(98);
  State state;
  Snippet snippet = state.Translate(wait_codon_many);
  ASSERT_EQ(49u, snippet.size);
  for (auto i = 0; i < 49; ++i) {
    EXPECT_EQ(NOP_Implied, snippet.bytecode[i]);
  }
  EXPECT_EQ(98u, snippet.duration);
  EXPECT_FALSE(snippet.should_advance_register_rotation);
}

TEST(StateTest, TranslateWaitManyOdd) {
  Codon wait_codon_many = MakeWaitCodon(111);
  State state;
  Snippet snippet = state.Translate(wait_codon_many);
  ASSERT_EQ(56u, snippet.size);
  for (auto i = 0; i < 54; ++i) {
    EXPECT_EQ(NOP_Implied, snippet.bytecode[i]);
  }
  EXPECT_EQ(BIT_ZeroPage, snippet.bytecode[54]);
  EXPECT_EQ(111u, snippet.duration);
  EXPECT_FALSE(snippet.should_advance_register_rotation);
}

TEST(StateTest, TranslateWaitMax) {
  Codon wait_codon_max = MakeWaitCodon(255);
  State state;
  Snippet snippet = state.Translate(wait_codon_max);
  ASSERT_EQ(128u, snippet.size);
  for (auto i = 0; i < 126; ++i) {
    EXPECT_EQ(NOP_Implied, snippet.bytecode[i]);
  }
  EXPECT_EQ(BIT_ZeroPage, snippet.bytecode[126]);
  EXPECT_EQ(255u, snippet.duration);
  EXPECT_FALSE(snippet.should_advance_register_rotation);
}

TEST(StateTest, TranslateSwitchBanks) {
  for (size_t pad_size = 8; pad_size < kBankPadding * 2; ++pad_size) {
    State state;
    Codon bank_switch = MakeBankSwitchCodon(pad_size);
    Snippet snippet = state.Translate(bank_switch);
    ASSERT_EQ(pad_size, snippet.size);
    EXPECT_EQ(JMP_Absolute, snippet.bytecode[0]);
    EXPECT_EQ(0x00u, snippet.bytecode[1]);
    EXPECT_EQ(0xf0u, snippet.bytecode[2]);
    EXPECT_EQ(0x00u, snippet.bytecode[pad_size - 4]);
    EXPECT_EQ(0xf0u, snippet.bytecode[pad_size - 3]);
    EXPECT_EQ(0x00u, snippet.bytecode[pad_size - 2]);
    EXPECT_EQ(0xf0u, snippet.bytecode[pad_size - 1]);
    EXPECT_EQ(3u, snippet.duration);
    EXPECT_FALSE(snippet.should_advance_register_rotation);
  }
}

TEST(StateTest, TranslateUnknownStateAlwaysLoads) {
  State state;
  state.tia()[VSYNC] = 0;
  state.tia_known()[VSYNC] = false;
  Codon unknown_load = MakeTIACodon(kSetVSYNC, 0);
  Snippet snippet = state.Translate(unknown_load);
  EXPECT_EQ(4u, snippet.size);
  EXPECT_TRUE(IsLoad(snippet.bytecode[0]));
  EXPECT_EQ(0u, snippet.bytecode[1]);
  EXPECT_TRUE(IsStore(snippet.bytecode[2]));
  EXPECT_TRUE(IsLoadAndStorePair(snippet.bytecode[0], snippet.bytecode[2]));
  EXPECT_EQ(VSYNC, snippet.bytecode[3]);
  EXPECT_EQ(5u, snippet.duration);
  EXPECT_TRUE(snippet.should_advance_register_rotation);
}

TEST(StateTest, TranslateNoChangeNoMask) {
  State state;
  state.tia()[PF2] = 0xa2;
  state.tia_known()[PF2] = true;
  Codon redundant_codon = MakeTIACodon(kSetPF2, 0xa2);
  Snippet snippet = state.Translate(redundant_codon);
  EXPECT_EQ(0u, snippet.size);
  EXPECT_EQ(0u, snippet.duration);
  EXPECT_FALSE(snippet.should_advance_register_rotation);
}

TEST(StateTest, TranslateNoChangeWithMask) {
  State state;
  state.tia()[ENAM1] = 0b11111101;
  state.tia_known()[ENAM1] = true;
  Codon redundant_codon = MakeTIACodon(kSetENAM1, 0);
  Snippet snippet = state.Translate(redundant_codon);
  EXPECT_EQ(0u, snippet.size);
  EXPECT_EQ(0u, snippet.duration);
  EXPECT_FALSE(snippet.should_advance_register_rotation);
}

TEST(StateTest, TranslateSharedTIANoChange) {
  State state;
  state.tia()[NUSIZ1] = 0b00101011;
  state.tia_known()[NUSIZ1] = true;
  Codon redundant_shared = MakeTIACodon(kSetNUSIZ1_M1, 0x20);
  Snippet snippet = state.Translate(redundant_shared);
  EXPECT_EQ(0u, snippet.size);
  EXPECT_EQ(0u, snippet.duration);
  EXPECT_FALSE(snippet.should_advance_register_rotation);
}

TEST(StateTest, TranslateSkipLoadExactMatch) {
  State state;
  state.registers()[X] = 0xa5;
  state.register_known()[X] = true;
  state.set_current_time(4321);
  Codon reuse_x_codon = MakeTIACodon(kSetGRP0, 0xa5);
  Snippet snippet = state.Translate(reuse_x_codon);
  ASSERT_EQ(2u, snippet.size);
  EXPECT_EQ(STX_ZeroPage, snippet.bytecode[0]);
  EXPECT_EQ(GRP0, snippet.bytecode[1]);
  EXPECT_EQ(3u, snippet.duration);
  EXPECT_TRUE(snippet.should_advance_register_rotation);
}

TEST(StateTest, TranslateSkipLoadMaskMatch) {
  State state;
  // REFP1 only cares about bit 3, which we set to 1 here.
  state.registers()[A] = 0b10101011;
  state.register_known()[A] = true;
  state.set_current_time(0xfeedfeed);
  Codon reuse_a_codon = MakeTIACodon(kSetREFP1, 0x08);
  Snippet snippet = state.Translate(reuse_a_codon);
  ASSERT_EQ(2u, snippet.size);
  EXPECT_EQ(STA_ZeroPage, snippet.bytecode[0]);
  EXPECT_EQ(REFP1, snippet.bytecode[1]);
  EXPECT_EQ(3u, snippet.duration);
  EXPECT_TRUE(snippet.should_advance_register_rotation);
}

TEST(StateTest, TranslateRegisterRotation) {
  State state;
  state.registers()[A] = 0x42;
  state.register_last_used()[A] = 23;
  state.registers()[X] = 0xff;
  state.register_last_used()[X] = 117;
  state.registers()[Y] = 0xed;
  state.register_last_used()[Y] = 4;
  state.set_current_time(140);
  Codon register_rotate = MakeTIACodon(kSetGRP0, 0x02);
  Snippet snippet = state.Translate(register_rotate);
  ASSERT_EQ(4u, snippet.size);
  EXPECT_EQ(LDY_Immediate, snippet.bytecode[0]);
  EXPECT_EQ(0x02u, snippet.bytecode[1]);
  EXPECT_EQ(STY_ZeroPage, snippet.bytecode[2]);
  EXPECT_EQ(GRP0, snippet.bytecode[3]);
  EXPECT_EQ(5u, snippet.duration);
  EXPECT_TRUE(snippet.should_advance_register_rotation);
}

TEST(StateTest, TranslateStrobeShouldNotAdvanceRotation) {
  State state;
  Codon strobe = MakeTIACodon(kStrobeRESP0, 0xff);
  Snippet snippet = state.Translate(strobe);
  ASSERT_EQ(2u, snippet.size);
  EXPECT_TRUE(IsStore(snippet.bytecode[0]));
  EXPECT_EQ(RESP0, snippet.bytecode[1]);
  EXPECT_EQ(3u, snippet.duration);
  EXPECT_FALSE(snippet.should_advance_register_rotation);
}

// CTRLPF, NUSIZ0, and NUSIZ1 all pack more than one state value into a shared
// register. We test that values not being modified in the same register are
// preserved.
TEST(StateTest, TranslateSharedTIAPreserveState) {
  State state;
  state.tia()[CTRLPF] = 0b00010101;
  Codon shared = MakeTIACodon(kSetCTRLPF_BALL, 0x20);
  Snippet snippet = state.Translate(shared);
  ASSERT_EQ(4u, snippet.size);
  EXPECT_TRUE(IsLoad(snippet.bytecode[0]));
  EXPECT_EQ(0b00100101u, snippet.bytecode[1]);
  EXPECT_TRUE(IsStore(snippet.bytecode[2]));
  EXPECT_TRUE(IsLoadAndStorePair(snippet.bytecode[0], snippet.bytecode[2]));
  EXPECT_EQ(CTRLPF, snippet.bytecode[3]);
  EXPECT_EQ(5u, snippet.duration);
  EXPECT_TRUE(snippet.should_advance_register_rotation);
}

TEST(StateTest, TranslateSharedTIAReuseRegister) {
  State state;
  state.tia()[CTRLPF] = 0b10111010;
  state.tia_known()[CTRLPF] = true;
  state.registers()[X] = 0b01110000;
  state.register_known()[X] = true;
  Codon reuse_reg = MakeTIACodon(kSetCTRLPF_SCORE, 0);
  Snippet snippet = state.Translate(reuse_reg);
  ASSERT_EQ(2u, snippet.size);
  EXPECT_EQ(STX_ZeroPage, snippet.bytecode[0]);
  EXPECT_EQ(CTRLPF, snippet.bytecode[1]);
  EXPECT_EQ(3u, snippet.duration);
  EXPECT_TRUE(snippet.should_advance_register_rotation);
}

TEST(StateTest, ApplyEmptySnippetOnEmptyState) {
  State state;
  Snippet snippet;
  Kernel kernel;
  state.Apply(snippet, kernel);
  ExpectEmptyState(state);
  EXPECT_EQ(0u, state.current_time());
  EXPECT_EQ(0u, kernel.size());
}

TEST(StateTest, ApplyWaits) {
  State state;
  size_t accum = 0;
  for (size_t i = 2; i < 256; ++i) {
    Snippet snippet = state.Translate(MakeWaitCodon(i));
    Kernel kernel;
    state.Apply(snippet, kernel);
    EXPECT_EQ(0, std::memcmp(
        snippet.bytecode.data(), kernel.bytecode(), snippet.size));
    ExpectEmptyState(state);
    accum += i;
    EXPECT_EQ(accum, state.current_time());
  }
}

TEST(StateTest, ApplyJMPs) {
  State state;
  Snippet snippet;
  snippet.Insert(JMP_Absolute);
  snippet.Insert(0x0f);
  snippet.Insert(0x00);
  // Insert some padding.
  snippet.Insert(0x00);
  snippet.Insert(0x00);
  snippet.Insert(0x00);
  // Insert both jump table addresses.
  snippet.Insert(0x0f);
  snippet.Insert(0x00);
  snippet.Insert(0x0f);
  snippet.Insert(0x00);
  snippet.duration = 3;
  Kernel kernel;
  state.Apply(snippet, kernel);
  ExpectEmptyState(state);
  EXPECT_EQ(0, std::memcmp(
      snippet.bytecode.data(), kernel.bytecode(), snippet.size));
  EXPECT_EQ(3u, state.current_time());
}

TEST(StateTest, ApplyThreeLoadsShouldRotateRegisters) {
  State state;
  Snippet grp0_snippet = state.Translate(MakeTIACodon(kSetGRP0, 0xff));
  Kernel kernel;
  state.Apply(grp0_snippet, kernel);
  EXPECT_EQ(5u, state.current_time());
  EXPECT_EQ(5u, state.register_last_used()[A]);
  EXPECT_EQ(0u, state.register_last_used()[X]);
  EXPECT_EQ(0u, state.register_last_used()[Y]);
  EXPECT_EQ(0xffu, state.tia()[GRP0]);
  EXPECT_EQ(0xffu, state.registers()[A]);
  EXPECT_EQ(0u, state.registers()[X]);
  EXPECT_EQ(0u, state.registers()[Y]);
  EXPECT_EQ(0, std::memcmp(
      grp0_snippet.bytecode.data(), kernel.bytecode(), grp0_snippet.size));
  size_t bytecode_size = grp0_snippet.size;

  Snippet pf1_snippet = state.Translate(MakeTIACodon(kSetPF1, 0xaa));
  state.Apply(pf1_snippet, kernel);
  EXPECT_EQ(10u, state.current_time());
  EXPECT_EQ(5u, state.register_last_used()[A]);
  EXPECT_EQ(10u, state.register_last_used()[X]);
  EXPECT_EQ(0u, state.register_last_used()[Y]);
  EXPECT_EQ(0xaau, state.tia()[PF1]);
  EXPECT_EQ(0xffu, state.registers()[A]);
  EXPECT_EQ(0xaau, state.registers()[X]);
  EXPECT_EQ(0u, state.registers()[Y]);
  EXPECT_EQ(0, std::memcmp(pf1_snippet.bytecode.data(),
                           kernel.bytecode() + bytecode_size,
                           pf1_snippet.size));
  bytecode_size += pf1_snippet.size;

  Snippet resp1_snippet = state.Translate(MakeTIACodon(kStrobeRESP1, 0x22));
  state.Apply(resp1_snippet, kernel);
  EXPECT_EQ(13u, state.current_time());
  EXPECT_EQ(5u, state.register_last_used()[A]);
  EXPECT_EQ(10u, state.register_last_used()[X]);
  EXPECT_EQ(0u, state.register_last_used()[Y]);
  EXPECT_EQ(0xaau, state.tia()[PF1]);
  EXPECT_EQ(0xffu, state.registers()[A]);
  EXPECT_EQ(0xaau, state.registers()[X]);
  EXPECT_EQ(0u, state.registers()[Y]);
  EXPECT_EQ(0, std::memcmp(resp1_snippet.bytecode.data(),
                           kernel.bytecode() + bytecode_size,
                           resp1_snippet.size));
  bytecode_size += resp1_snippet.size;

  Snippet colubk_snippet = state.Translate(MakeTIACodon(kSetCOLUBK, 0x77));
  state.Apply(colubk_snippet, kernel);
  EXPECT_EQ(18u, state.current_time());
  EXPECT_EQ(5u, state.register_last_used()[A]);
  EXPECT_EQ(10u, state.register_last_used()[X]);
  EXPECT_EQ(18u, state.register_last_used()[Y]);
  EXPECT_EQ(0x77u, state.tia()[COLUBK]);
  EXPECT_EQ(0xffu, state.registers()[A]);
  EXPECT_EQ(0xaau, state.registers()[X]);
  EXPECT_EQ(0x77u, state.registers()[Y]);
  EXPECT_EQ(0, std::memcmp(colubk_snippet.bytecode.data(),
                           kernel.bytecode() + bytecode_size,
                           colubk_snippet.size));
}

}  // namespace vcsmc

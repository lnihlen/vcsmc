#include "kernel.h"

#include <cstring>
#include <random>

#include "assembler.h"
#include "constants.h"
#include "gtest/gtest.h"

namespace {

void ValidateKernel(vcsmc::SpecList specs,
                    std::shared_ptr<vcsmc::Kernel> kernel) {
  // Bytecode size should be a multiple of the bank size.
  EXPECT_EQ(0U, kernel->bytecode_size() % vcsmc::kBankSize);
  size_t banks = kernel->bytecode_size() / vcsmc::kBankSize;
  // Check each bank for zero padding and a jmp at the end.
  for (size_t i = 0; i < banks; ++i) {
    const uint8* bank_start = kernel->bytecode() + (i * vcsmc::kBankSize);
    // Point at last byte in bank, should be zero for at least kBankPadding
    // bytes, then look for jmp sequence.
    const uint8* bank_byte = bank_start + (vcsmc::kBankSize - 1);
    for (size_t j = 0; j < vcsmc::kBankPadding; ++j) {
      EXPECT_EQ(0x00u, *bank_byte);
      --bank_byte;
    }
    // Now consume remaining padding, which should be zero, until expected jmp
    // command encountered.
    while (*bank_byte == 0x00u) {
      ASSERT_GE(bank_byte, bank_start + 3);
      --bank_byte;
    }
    EXPECT_EQ(0xf0u, *bank_byte);
    --bank_byte;
    EXPECT_EQ(0x00u, *bank_byte);
    --bank_byte;
    EXPECT_EQ(vcsmc::OpCode::JMP_Absolute, *bank_byte);
  }

  // Entire program should take exactly vcsmc::kScreenSizeCycles to execute and
  // should have all of the specs in the speclist at the same place.
  uint32 current_cycle = 0;
  size_t current_spec_index = 0;
  size_t current_byte = 0;
  while (current_byte < kernel->bytecode_size()) {
    uint32 next_spec_start_time = current_spec_index < specs->size() ?
      specs->at(current_spec_index).range().start_time() :
      vcsmc::kScreenSizeCycles;
    if (current_cycle == next_spec_start_time) {
      ASSERT_GT(specs->size(), current_spec_index);
      EXPECT_EQ(0,
          std::memcmp(specs->at(current_spec_index).bytecode(),
                      kernel->bytecode() + current_byte,
                      specs->at(current_spec_index).size()));
      current_byte += specs->at(current_spec_index).size();
      current_cycle = specs->at(current_spec_index).range().end_time();
      ++current_spec_index;
    } else {
      while (current_cycle < next_spec_start_time) {
        vcsmc::OpCode op = static_cast<vcsmc::OpCode>(
            kernel->bytecode()[current_byte]);
        if (op == vcsmc::JMP_Absolute) {
          current_byte =
            ((current_byte / vcsmc::kBankSize) + 1) * vcsmc::kBankSize;
        } else {
          size_t op_size = vcsmc::OpCodeBytes(op);
          EXPECT_NE(0u, op_size);
          current_byte += op_size;
        }
        uint32 op_cycles = vcsmc::OpCodeCycles(op);
        EXPECT_NE(0u, op_cycles);
        current_cycle += op_cycles;
      }
    }
  }
  EXPECT_EQ(kernel->bytecode_size(), current_byte);
  EXPECT_EQ(vcsmc::kScreenSizeCycles, current_cycle);
  EXPECT_EQ(specs->size(), current_spec_index);
}

}

namespace vcsmc {

TEST(GenerateRandomKernelJobTest, GeneratesValidRandomKernel) {
  SpecList specs = ParseSpecListString(
      "- first_cycle: 0\n"
      "  bytecode: |\n"
      "   lda #0\n"
      "   sta VBLANK\n"
      "   ldx #2\n"
      "   stx VSYNC\n"
      "- first_cycle: 228\n"
      "  bytecode: |\n"
      "    lda #0\n"
      "    sta VSYNC\n"
      "- first_cycle: 17632\n"
      "  bytecode: |\n"
      "    lda #$42\n"
      "    sta VBLANK\n");
  ASSERT_NE(nullptr, specs);
  std::string seed_str = "trivial stable testing seed";
  std::seed_seq seed(seed_str.begin(), seed_str.end());
  std::shared_ptr<Kernel> kernel(new Kernel(seed));
  Kernel::GenerateRandomKernelJob job(kernel, specs);
  job.Execute();
  ValidateKernel(specs, kernel);
}

}  // namespace vcsmc

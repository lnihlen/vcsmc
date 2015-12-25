#include "kernel.h"

#include <cstring>
#include <memory>
#include <random>
#include <set>
#include <string>

#include "assembler.h"
#include "color_table.h"
#include "constants.h"
#include "gtest/gtest.h"
#include "job_queue.h"
#include "spec.h"

namespace {

void ValidateKernel(std::shared_ptr<vcsmc::Kernel> kernel) {
  // Bytecode size should be a multiple of the bank size.
  EXPECT_EQ(0U, kernel->bytecode_size() % vcsmc::kBankSize);
  size_t banks = kernel->bytecode_size() / vcsmc::kBankSize;
  // Check each bank for zero padding and a jmp at the end.
  for (size_t i = 0; i < banks; ++i) {
    const uint8* bank_start = kernel->bytecode() + (i * vcsmc::kBankSize);
    // Last 6 bytes in stream should be 0x00 0xf0 0x00 0xf0 0x00 0xf0.
    const uint8* bank_byte = bank_start + (vcsmc::kBankSize - 1);
    for (size_t j = 0; j < 3; ++j) {
      EXPECT_EQ(0xf0, *bank_byte);
      --bank_byte;
      EXPECT_EQ(0x00, *bank_byte);
      --bank_byte;
    }
    // Then we should have all zero bytes for at least the balance of the
    // padding.
    for (size_t j = 0; j < vcsmc::kBankPadding - 6; ++j) {
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
    uint32 next_spec_start_time = current_spec_index < kernel->specs()->size() ?
      kernel->specs()->at(current_spec_index).range().start_time() :
      vcsmc::kScreenSizeCycles;
    if (current_cycle == next_spec_start_time) {
      ASSERT_GT(kernel->specs()->size(), current_spec_index);
      EXPECT_EQ(0,
          std::memcmp(kernel->specs()->at(current_spec_index).bytecode(),
                      kernel->bytecode() + current_byte,
                      kernel->specs()->at(current_spec_index).size()));
      current_byte += kernel->specs()->at(current_spec_index).size();
      current_cycle =
        kernel->specs()->at(current_spec_index).range().end_time();
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
  EXPECT_EQ(kernel->specs()->size(), current_spec_index);
}

class GenerateBackgroundColorKernelJob : public vcsmc::Job {
 public:
  GenerateBackgroundColorKernelJob(
      std::shared_ptr<vcsmc::Kernel> kernel,
      uint8 bg)
      : kernel_(kernel), bg_(bg) {}
  void Execute() override {
    vcsmc::SpecList specs(new std::vector<vcsmc::Spec>());
    // We build a huge no-op kernel out of specs that handle the required
    // vertical blanking and state initialization but also several specs that
    // consist only of nops.
    uint32 spec_cycles = 0;
    size_t spec_size = 0;
    std::unique_ptr<uint8[]> bytecode = vcsmc::AssembleString(
        "lda #0\n"
        "sta VBLANK\n"
        "lda #2\n"
        "sta VSYNC\n", &spec_cycles, &spec_size);
    ASSERT_NE(nullptr, bytecode);
    uint32 total_cycles = spec_cycles;
    size_t total_size = spec_size;
    specs->emplace_back(
        vcsmc::Range(0, spec_cycles), spec_size, std::move(bytecode));

    // Fill space between with nops.
    spec_cycles = 228 - total_cycles;
    // Ensure we can fill the space with only nops, which are two cycles each.
    ASSERT_EQ(0u, spec_cycles % 2);
    spec_size = spec_cycles / 2;
    bytecode.reset(new uint8[spec_size]);
    std::memset(bytecode.get(), vcsmc::NOP_Implied, spec_size);
    specs->emplace_back(
        vcsmc::Range(total_cycles, 228), spec_size, std::move(bytecode));
    total_cycles += spec_cycles;
    total_size += spec_size;

    bytecode = vcsmc::AssembleString(
        "lda #0\n"
        "sta VSYNC\n"
        "sta RESP0\n"
        "sta RESP1\n"
        "sta NUSIZ0\n"
        "sta NUSIZ1\n"
        "sta COLUP0\n"
        "sta COLUP1\n"
        "sta CTRLPF\n"
        "sta REFP0\n"
        "sta REFP1\n"
        "sta PF0\n"
        "sta PF1\n"
        "sta PF2\n"
        "sta AUDC0\n"
        "sta AUDC1\n"
        "sta AUDF0\n"
        "sta AUDF1\n"
        "sta AUDV0\n"
        "sta AUDV1\n"
        "sta GRP0\n"
        "sta GRP1\n"
        "sta ENAM0\n"
        "sta ENAM1\n"
        "sta ENABL\n"
        "sta VDELP0\n"
        "sta VDELP1\n"
        "sta RESMP0\n"
        "sta RESMP1\n"
        "sta HMCLR\n", &spec_cycles, &spec_size);
    ASSERT_NE(nullptr, bytecode);
    specs->emplace_back(
        vcsmc::Range(228, 228 + spec_cycles), spec_size, std::move(bytecode));
    total_cycles += spec_cycles;
    total_size += spec_size;

    bytecode.reset(new uint8[4]);
    bytecode.get()[0] = vcsmc::LDA_Immediate;
    bytecode.get()[1] = bg_;
    bytecode.get()[2] = vcsmc::STA_ZeroPage;
    bytecode.get()[3] = vcsmc::COLUBK;
    specs->emplace_back(
        vcsmc::Range(total_cycles, total_cycles + 5), 4, std::move(bytecode));
    total_cycles += 5;
    total_size += 4;

    // Target is 17632 cycles, also tracking frame size so we can generate bank
    // padding specs as needed.
    while (total_cycles < 17632) {
      spec_cycles = 17632 - total_cycles;
      spec_size = spec_cycles / 2;
      bool need_jmp = false;
      if ((total_size % vcsmc::kBankSize) + spec_size >
          (vcsmc::kBankSize - vcsmc::kBankPadding - 5)) {
        need_jmp = true;
        spec_size = vcsmc::kBankSize - vcsmc::kBankPadding - 5 -
            (total_size % vcsmc::kBankSize);
        spec_cycles = spec_size * 2;
      }
      bytecode.reset(new uint8[spec_size]);
      std::memset(bytecode.get(), vcsmc::NOP_Implied, spec_size);
      specs->emplace_back(
          vcsmc::Range(total_cycles, total_cycles + spec_cycles),
          spec_size, std::move(bytecode));
      total_size += spec_size;
      total_cycles += spec_cycles;
      if (need_jmp) {
        // Let the kernel generation function generate the jump.
        total_cycles += 3;
        total_size += vcsmc::kBankSize - (total_size % vcsmc::kBankSize);
      }
    }

    // More room for a jmp if needed.
    total_cycles += 3;

    // Append instruction to turn on VBLANK, then let kernel random generation
    // fill rest of blank frame with noise.
    bytecode = vcsmc::AssembleString(
      "lda #$42\n"
      "sta VBLANK\n", &spec_cycles, &spec_size);
    specs->emplace_back(vcsmc::Range(total_cycles, total_cycles + spec_cycles),
        spec_size, std::move(bytecode));

    vcsmc::Kernel::GenerateRandomKernelJob gen_job(kernel_, specs);
    gen_job.Execute();
    ValidateKernel(kernel_);
  }

 private:
  std::shared_ptr<vcsmc::Kernel> kernel_;
  uint8 bg_;
};

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
  ValidateKernel(kernel);
}

TEST(ScoreKernelJobTest, SimulatesSimpleFrameKernel) {
  vcsmc::JobQueue job_queue(0);
  std::vector<std::shared_ptr<vcsmc::Kernel>> kernels;

  for (size_t i = 0; i < 128; ++i) {
    char seed_char[64];
    snprintf(seed_char, 64, "trivial stable testing seed 0x%8lx", i);
    std::string seed_str(seed_char);
    std::seed_seq seed(seed_str.begin(), seed_str.end());
    kernels.emplace_back(new Kernel(seed));
    job_queue.Enqueue(std::unique_ptr<vcsmc::Job>(
          new GenerateBackgroundColorKernelJob(
            kernels[i], static_cast<uint8>(i * 2))));
  }

  job_queue.Finish();

  // Simulate and score, should produce a valid empty frame with background
  // color.
  std::vector<Kernel::ScoreKernelJob::ColorDistances> test_distances(128);
  for (size_t i = 0; i < 128; ++i) {
    // Build distance table that treats target background color as zero error,
    // all other colors as nonzero.
    for (size_t j = 0; j < 128; ++j) {
      const double dist = (i == j) ? 0.0 : 1.0;
      test_distances[i].emplace_back(
          kTargetFrameWidthPixels * kFrameHeightPixels, dist);
    }
    job_queue.Enqueue(std::unique_ptr<vcsmc::Job>(
          new vcsmc::Kernel::ScoreKernelJob(kernels[i], test_distances[i])));
  }

  job_queue.Finish();

  // Although differing by only one byte the fingerprints should also be unique
  // for the individual kernels.
  std::set<uint64> fingerprints;

  for (size_t i = 0; i < 128; ++i) {
    ASSERT_EQ(true, kernels[i]->score_valid());
    EXPECT_EQ(0.0, kernels[i]->score());
    fingerprints.insert(kernels[i]->fingerprint());
  }

  EXPECT_EQ(128u, fingerprints.size());
}

TEST(MutateKernelJobTest, MutatesSimpleFrameKernel) {
  std::string seed_str = "trivial stable testing seed for mutate kernel";
  std::seed_seq seed(seed_str.begin(), seed_str.end());
  std::shared_ptr<Kernel> original_kernel(new Kernel(seed));
  GenerateBackgroundColorKernelJob gen_job(original_kernel, 64);
  gen_job.Execute();
  std::string target_seed_str = "target stable testing seed";
  std::seed_seq target_seed(target_seed_str.begin(), target_seed_str.end());
  std::shared_ptr<Kernel> target_kernel(new Kernel(target_seed));
  vcsmc::Kernel::MutateKernelJob mutate_job(
      original_kernel, target_kernel, 128);
  mutate_job.Execute();
  ValidateKernel(target_kernel);
  EXPECT_NE(original_kernel->fingerprint(), target_kernel->fingerprint());
}

}  // namespace vcsmc

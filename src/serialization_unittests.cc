#include "gtest/gtest.h"

#include <cstring>
#include <random>
#include <string>

#include "kernel.h"
#include "serialization.h"
#include "spec.h"

namespace vcsmc {

TEST(SerializationTest, Base64EncodeDecode) {
  uint8 buf[128];
  std::string seed_str = "base64 test seed";
  std::seed_seq seed(seed_str.begin(), seed_str.end());
  std::default_random_engine engine(seed);
  std::uniform_int_distribution<uint8> distro(0, 255);
  for (size_t i = 0; i < 128; ++i) {
    buf[i] = distro(engine);
  }
  for (size_t i = 0; i < 127; ++i) {
    size_t len = 128 - i;
    std::string encode = vcsmc::Base64Encode(buf + i, len, 0);
    std::unique_ptr<uint8[]> decode = vcsmc::Base64Decode(encode, len);
    EXPECT_EQ(0, std::memcmp(decode.get(), buf + i, len));
    encode = vcsmc::Base64Encode(buf, len, 0);
    decode = vcsmc::Base64Decode(encode, len);
    EXPECT_EQ(0, std::memcmp(decode.get(), buf, len));
  }
}

TEST(SerializationTest, ParseGenerationStringSingle) {
  SpecList specs = ParseSpecListString(
      "- first_cycle: 0\n"
      "  assembler: |\n"
      "    lda #0\n"
      "    sta VBLANK\n"
      "    lda #2\n"
      "    sta VSYNC\n"
      "- first_cycle: 228\n"
      "  assembler: |\n"
      "    lda #0\n"
      "    sta VSYNC\n"
      "- first_cycle: 17632\n"
      "  assembler: |\n"
      "    lda #$42\n"
      "    sta VBLANK\n");
  ASSERT_NE(nullptr, specs);
  std::string seed_str = "serialization test seed";
  std::seed_seq seed(seed_str.begin(), seed_str.end());
  std::shared_ptr<Kernel> kernel(new Kernel(seed));
  Kernel::GenerateRandomKernelJob job(kernel, specs);
  job.Execute();
  std::string kernel_str;
  ASSERT_EQ(true, SaveKernelToString(kernel, kernel_str));
  Generation gen = ParseGenerationString(kernel_str);
  ASSERT_NE(nullptr, gen);
  ASSERT_EQ(1u, gen->size());
  EXPECT_EQ(kernel->fingerprint(), gen->at(0)->fingerprint());
}

TEST(SerializationTest, ParseSpecListStringSingle) {
  SpecList sl = ParseSpecListString(
      "first_cycle: 10\n"
      "assembler: sta VBLANK\n");
  ASSERT_NE(sl, nullptr);
  ASSERT_EQ(1u, sl->size());
  EXPECT_EQ(10u, sl->at(0).range().start_time());
  EXPECT_EQ(3u, sl->at(0).range().Duration());
  ASSERT_EQ(2u, sl->at(0).size());
  const uint8* assembler = sl->at(0).bytecode();
  EXPECT_EQ(0x85u, assembler[0]);
  EXPECT_EQ(0x01u, assembler[1]);
}

TEST(SpecTest, ParseSpecListStringMultiple) {
  SpecList sl = ParseSpecListString(
      "- first_cycle: 228  # 3rd scanline, turn off vsync.\n"
      "  assembler: |\n"
      "    ldx #0\n"
      "    sty HMP0\n"
      "- first_cycle: 17632 # 232nd scanline, turn on vblank\n"
      "  assembler: |\n"
      "    ldy #$42\n"
      "    jmp $feed\n"
      "    stx PF2\n");
  ASSERT_NE(sl, nullptr);
  ASSERT_EQ(2u, sl->size());

  EXPECT_EQ(228u, sl->at(0).range().start_time());
  EXPECT_EQ(5u, sl->at(0).range().Duration());
  ASSERT_EQ(4u, sl->at(0).size());
  const uint8* assembler_a = sl->at(0).bytecode();
  EXPECT_EQ(0xa2u, assembler_a[0]);
  EXPECT_EQ(0x00u, assembler_a[1]);
  EXPECT_EQ(0x84u, assembler_a[2]);
  EXPECT_EQ(0x20u, assembler_a[3]);

  EXPECT_EQ(17632u, sl->at(1).range().start_time());
  EXPECT_EQ(8u, sl->at(1).range().Duration());
  ASSERT_EQ(7u, sl->at(1).size());
  const uint8* assembler_b = sl->at(1).bytecode();
  EXPECT_EQ(0xa0u, assembler_b[0]);
  EXPECT_EQ(0x42u, assembler_b[1]);
  EXPECT_EQ(0x4cu, assembler_b[2]);
  EXPECT_EQ(0xedu, assembler_b[3]);
  EXPECT_EQ(0xfeu, assembler_b[4]);
  EXPECT_EQ(0x86u, assembler_b[5]);
  EXPECT_EQ(0x0fu, assembler_b[6]);
}

TEST(SerializationTest, ParseSpecListWithHardDuration) {
  SpecList sl = ParseSpecListString(
      "- first_cycle: 192\n"
      "  hard_duration: 227\n"
      "  assembler: nop\n");
  ASSERT_NE(sl, nullptr);
  ASSERT_EQ(1u, sl->size());

  EXPECT_EQ(192u, sl->at(0).range().start_time());
  EXPECT_EQ(227u, sl->at(0).range().Duration());
  ASSERT_EQ(1u, sl->at(0).size());
  const uint8* assembler = sl->at(0).bytecode();
  EXPECT_EQ(0xeau, assembler[0]);
}

}  // namespace vcsmc

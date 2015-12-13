#include "assembler.h"
#include "gtest/gtest.h"

namespace vcsmc {

// An empty string should assemble successfully to no opcodes.
TEST(AssemblerTest, AssembleEmptyString) {
  PackedOpcodes ops = AssembleStringPacked("");
  ASSERT_NE(ops, nullptr);
  EXPECT_EQ(0U, ops->size());
}

// A string consisting only of a newline should assemble to no opcodes.
TEST(AssemblerTest, AssembleNewLine) {
  PackedOpcodes ops = AssembleStringPacked("\n");
  ASSERT_NE(ops, nullptr);
  EXPECT_EQ(0U, ops->size());
}

// Tabs are valid whitespace as well and should be ignored.
TEST(AssemblerTest, AssembleTab) {
  PackedOpcodes ops = AssembleStringPacked("\t");
  ASSERT_NE(ops, nullptr);
  EXPECT_EQ(0U, ops->size());
}

TEST(AssemblerTest, AssembleCommentOnStartOfLine) {
  PackedOpcodes ops = AssembleStringPacked("; Comment on this line\n");
  ASSERT_NE(ops, nullptr);
  EXPECT_EQ(0U, ops->size());
}

TEST(AssemblerTest, AssembleCommentLaterOnLine) {
  PackedOpcodes ops = AssembleStringPacked("  ; Late-line comment");
  ASSERT_NE(ops, nullptr);
  EXPECT_EQ(0U, ops->size());
}

// A single-line load immediate instruction.
TEST(AssemblerTest, AssembleSingleLine) {
  PackedOpcodes ops = AssembleStringPacked("lda #$f0");
  ASSERT_NE(ops, nullptr);
  ASSERT_EQ(1U, ops->size());
  EXPECT_EQ(0x0000f0a9u, ops->at(0) & 0x0000ffff);
}

// Multi-line assembly with comments.
TEST(AssemblerTest, AssembleMultiLine) {
  PackedOpcodes ops = AssembleStringPacked(
      "; single-line store\n"
      "stx AUDC0  ; update AUDC0\n"
      "nop\n"
      "lda #$ef\n"
      "jmp $f00a");
  ASSERT_NE(ops, nullptr);
  ASSERT_EQ(4U, ops->size());
  EXPECT_EQ(0x00001586u, ops->at(0) & 0x0000ffff);
  EXPECT_EQ(0x000000eau, ops->at(1) & 0x000000ff);
  EXPECT_EQ(0x0000efa9u, ops->at(2) & 0x0000ffff);
  EXPECT_EQ(0x00f00a4cu, ops->at(3) & 0x00ffffff);
}

}  // namespace vcsmc

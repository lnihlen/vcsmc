#include "assembler.h"
#include "gtest/gtest.h"
#include "opcode.h"

namespace vcsmc {

class AssemblerTest : public ::testing::Test {
 protected:
  AssemblerTest() {
    // Ensure Init gets called before any test code executes.
    Assembler::InitAssemblerTables();
  }
};

// An empty string should assemble successfully to no opcodes.
TEST_F(AssemblerTest, AssembleEmptyString) {
  std::vector<std::unique_ptr<op::OpCode>> opcodes;
  EXPECT_TRUE(Assembler::AssembleString("", &opcodes));
  EXPECT_EQ(0U, opcodes.size());
}

// A string consisting only of a newline should assemble to no opcodes.
TEST_F(AssemblerTest, AssembleNewLine) {
  std::vector<std::unique_ptr<op::OpCode>> opcodes;
  EXPECT_TRUE(Assembler::AssembleString("\n", &opcodes));
  EXPECT_EQ(0U, opcodes.size());
}

// Tabs are valid whitespace as well and should be ignored.
TEST_F(AssemblerTest, AssembleTab) {
  std::vector<std::unique_ptr<op::OpCode>> opcodes;
  EXPECT_TRUE(Assembler::AssembleString("\t", &opcodes));
  EXPECT_EQ(0U, opcodes.size());
}

TEST_F(AssemblerTest, AssembleCommentOnStartOfLine) {
  std::vector<std::unique_ptr<op::OpCode>> opcodes;
  EXPECT_TRUE(Assembler::AssembleString("; Comment on this line\n", &opcodes));
  EXPECT_EQ(0U, opcodes.size());
}

TEST_F(AssemblerTest, AssembleCommentLaterOnLine) {
  std::vector<std::unique_ptr<op::OpCode>> opcodes;
  EXPECT_TRUE(Assembler::AssembleString("  ; Late-line comment", &opcodes));
  EXPECT_EQ(0U, opcodes.size());
}

// A single-line load immediate instruction.
TEST_F(AssemblerTest, AssembleSingleLineLDAImmediateHex) {
  std::vector<std::unique_ptr<op::OpCode>> opcodes;
  EXPECT_TRUE(Assembler::AssembleString("  lda #$f0", &opcodes));
  ASSERT_EQ(1U, opcodes.size());
  std::unique_ptr<uint8[]> bytes(new uint8[2]);
  EXPECT_EQ(2, opcodes[0]->bytecode(bytes.get()));
  EXPECT_EQ(0xa9, bytes[0]);
  EXPECT_EQ(0xf0, bytes[1]);
}

TEST_F(AssemblerTest, AssembleSingleLineSTXZeroPageTIA) {
  std::vector<std::unique_ptr<op::OpCode>> opcodes;
  EXPECT_TRUE(Assembler::AssembleString(
      "  ; single-line store\n  stx AUDC0  ; update AUDC0\n", &opcodes));
  ASSERT_EQ(1U, opcodes.size());
  std::unique_ptr<uint8[]> bytes(new uint8[2]);
  EXPECT_EQ(2, opcodes[0]->bytecode(bytes.get()));
  EXPECT_EQ(0x86, bytes[0]);
  EXPECT_EQ(0x15, bytes[1]);
}

}

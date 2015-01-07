#include "opcode.h"

#include "assembler.h"
#include "state.h"

namespace vcsmc {

namespace op {

LoadImmediate::LoadImmediate(uint8 value, Register reg)
    : value_(value),
      register_(reg) { }

std::unique_ptr<State> LoadImmediate::Transform(State* state) const {
  return state->AdvanceTimeAndSetRegister(
      cycles() * kColorClocksPerCPUCycle, register_, value_);
}

std::unique_ptr<OpCode> LoadImmediate::Clone() const {
  return std::unique_ptr<OpCode>(new LoadImmediate(value_, register_));
}

uint32 LoadImmediate::cycles() const { return kLoadImmediateCPUCycles; }

uint32 LoadImmediate::bytes() const { return 2u; }

const std::string LoadImmediate::assembler() const {
  std::string asm_string("ld");

  // Add register string name to instruction.
  asm_string += Assembler::RegisterToString(register_);

  // Pad some space and add the immediate signifier followed by the number.
  asm_string += " #";
  asm_string += Assembler::ByteToHexString(value_);
  return asm_string;
}

uint32 LoadImmediate::bytecode(uint8* output) const {
  switch (register_) {
    case Register::A:
      *output = 0xa9;
      break;

    case Register::X:
      *output = 0xa2;
      break;

    case Register::Y:
      *output = 0xa0;
      break;

    default:
      assert(false);
      break;
  }
  *(output + 1) = value_;
  return bytes();
}

StoreZeroPage::StoreZeroPage(TIA address, Register reg)
    : address_(address),
      register_(reg) { }

std::unique_ptr<State> StoreZeroPage::Transform(State* state) const {
  return state->AdvanceTimeAndCopyRegisterToTIA(
      cycles() * kColorClocksPerCPUCycle, register_, address_);
}

std::unique_ptr<OpCode> StoreZeroPage::Clone() const {
  return std::unique_ptr<OpCode>(new StoreZeroPage(address_, register_));
}

uint32 StoreZeroPage::cycles() const { return kStoreZeroPageCPUCycles; }

uint32 StoreZeroPage::bytes() const { return 2u; }

const std::string StoreZeroPage::assembler() const {
  std::string asm_string("st");
  asm_string += Assembler::RegisterToString(register_);
  asm_string += " ";
  asm_string += Assembler::AddressToString(address_);
  return asm_string;
}

uint32 StoreZeroPage::bytecode(uint8* output) const {
  switch (register_) {
    case Register::A:
      *output = 0x85;
      break;

    case Register::X:
      *output = 0x86;
      break;

    case Register::Y:
      *output = 0x84;
      break;

    default:
      assert(false);
      break;
  }
  *(output + 1) = static_cast<uint8>(address_);
  return bytes();
}

std::unique_ptr<State> NOP::Transform(State* state) const {
  return state->AdvanceTime(cycles() * kColorClocksPerCPUCycle);
}

std::unique_ptr<OpCode> NOP::Clone() const {
  return std::unique_ptr<OpCode>(new NOP());
}

uint32 NOP::cycles() const { return kNoOpCPUCycles; }

uint32 NOP::bytes() const { return 1u; }

const std::string NOP::assembler() const { return std::string("nop"); }

uint32 NOP::bytecode(uint8* output) const {
  *output = 0xea;
  return bytes();
}

}  // namespace op

std::unique_ptr<op::OpCode> makeLDA(uint8 value) {
  return std::unique_ptr<op::OpCode>(new op::LDA(value));
}

std::unique_ptr<op::OpCode> makeLDX(uint8 value) {
  return std::unique_ptr<op::OpCode>(new op::LDX(value));
}

std::unique_ptr<op::OpCode> makeLDY(uint8 value) {
  return std::unique_ptr<op::OpCode>(new op::LDY(value));
}

std::unique_ptr<op::OpCode> makeSTA(TIA address) {
  return std::unique_ptr<op::OpCode>(new op::STA(address));
}

std::unique_ptr<op::OpCode> makeSTX(TIA address) {
  return std::unique_ptr<op::OpCode>(new op::STX(address));
}

std::unique_ptr<op::OpCode> makeSTY(TIA address) {
  return std::unique_ptr<op::OpCode>(new op::STY(address));
}

std::unique_ptr<op::OpCode> makeNOP() {
  return std::unique_ptr<op::OpCode>(new op::NOP);
}

}  // namespace vcsmc

#include "opcode.h"

namespace vcsmc {

namespace op {

LoadImmediate::LoadImmediate(uint8 value, Register reg)
    : value_(value),
      register_(reg) { }

std::unique_ptr<State> LoadImmediate::Transform(State* state) const {
  return state->AdvanceTimeAndSetRegister(
      cycles() * kColorClocksPerCPUCycle, register_, value_);
}

const uint32 LoadImmediate::cycles() const { return kLoadImmediateCPUCycles; }

const uint32 LoadImmediate::bytes() const { return 2u; }

const std::string LoadImmediate::assembler() const {
  std::string asm_string("ld");

  // Add register string name to instruction.
  asm_string += State::RegisterToString(register_);

  // Pad some space and add the immediate signifier followed by the number.
  asm_string += " #";
  asm_string += State::ByteToHexString(value_);
  return asm_string;
}

StoreZeroPage::StoreZeroPage(TIA address, Register reg)
    : address_(address),
      register_(reg) { }

std::unique_ptr<State> StoreZeroPage::Transform(State* state) const {
  return state->AdvanceTimeAndCopyRegisterToTIA(
      cycles() * kColorClocksPerCPUCycle, register_, address_);
}

const uint32 StoreZeroPage::cycles() const { return kStoreZeroPageCPUCycles; }

const uint32 StoreZeroPage::bytes() const { return 2u; }

const std::string StoreZeroPage::assembler() const {
  std::string asm_string("st");
  asm_string += State::RegisterToString(register_);
  asm_string += " ";
  asm_string += State::AddressToString(address_);
  return asm_string;
}

std::unique_ptr<State> NOP::Transform(State* state) const {
  return state->AdvanceTime(cycles() * kColorClocksPerCPUCycle);
}

const uint32 NOP::cycles() const { return kNoOpCPUCycles; }

const uint32 NOP::bytes() const { return 1u; }

const std::string NOP::assembler() const { return std::string("nop"); }

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

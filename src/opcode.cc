#include "opcode.h"

namespace vcsmc {

namespace op {

LoadImmediate::LoadImmediate(uint8 value, State::Register reg)
    : value_(value),
      register_(reg) { }

std::unique_ptr<State> LoadImmediate::Transform(
    const std::unique_ptr<State>& state) const {
  return std::move(state->AdvanceTimeAndSetRegister(
      cycles() * kColorClocksPerCPUCycle, register_, value_));
}

const uint32 LoadImmediate::cycles() const { return 2u; }

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

StoreZeroPage::StoreZeroPage(State::TIA address, State::Register reg)
    : address_(address),
      register_(reg) { }

std::unique_ptr<State> StoreZeroPage::Transform(
    const std::unique_ptr<State>& state) const {
  return std::move(state->AdvanceTimeAndCopyRegisterToTIA(
      cycles() * kColorClocksPerCPUCycle, register_, address_));
}

const uint32 StoreZeroPage::cycles() const { return 3u; }

const uint32 StoreZeroPage::bytes() const { return 2u; }

const std::string StoreZeroPage::assembler() const {
  std::string asm_string("st");
  asm_string += State::RegisterToString(register_);
  asm_string += " ";
  asm_string += State::AddressToString(address_);
  return asm_string;
}

std::unique_ptr<State> NOP::Transform(
    const std::unique_ptr<State>& state) const {
  return std::move(state->AdvanceTime(cycles() * kColorClocksPerCPUCycle));
}

const uint32 NOP::cycles() const { return 2u; }

const uint32 NOP::bytes() const { return 1u; }

const std::string NOP::assembler() const { return std::string("nop"); }

}  // namespace op

}  // namespace vcsmc

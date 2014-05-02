#include "opcode.h"

namespace vcsmc {

namespace op {

LoadImmediate::LoadImmediate(uint8 value, State::Register reg)
    : value_(value),
      register_(reg) { }

std::unique_ptr<State> LoadImmediate::Transform(const State& state) const {
  State new_state(state);
  new_state.SetRegister(register_, value_);
  return new_state;
}

const uint32 LoadImmediate::cycles() const { return 2u; }

const uint32 LoadImmediate::bytes() const { return 2u; }

const std::string assembler() const {
  std::string asm_string("ld");

  // Add register string name to instruction.
  asm_string += State::RegisterToString(register_);

  // Pad some space and add the immediate signifier followed by the number.
  asm_string += " #";
  asm_string += State::ByteToHexString(value_);
  return asm_string;
}

StoreZeroPage::StoreZeroPage(uint8 address, State::Register reg)
    : address_(address),
      register_(reg) { }

const uint32 StoreZeroPage::cycles() const { return 3u; }

const uint32 StoreZeroPage::bytes() const { return 2u; }

const std::string StoreZeroPage::assembler() const {
  std::string asm_string("st");
  asm_string += State::RegisterToString(register_);
  asm_string += " ";
  asm_string += State::AddressToString(address_);
  return asm_string;
}

State NOP::Transform(const State& state) const {
  State new_state(state);
  return new_state;
}

const uint32 NOP::cycles() const { return 2u; }

const uint32 NOP::bytes() const { return 1u; }

const std::string NOP::assembler() const { return std::string("nop"); }

}  // namespace op

}  // namespace vcsmc

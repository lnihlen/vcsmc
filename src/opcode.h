#ifndef SRC_OPCODE_H_
#define SRC_OPCODE_H_

#include <string>

#include "constants.h"
#include "state.h"
#include "types.h"

namespace vcsmc {

// Keep OpCodes in their own sub namespace, to avoid muddying up the main one.
namespace op {

// Abstract base class for simple polymorphism.
class OpCode {
 public:
  // Given input State, apply this OpCode to it and return the resultant output
  // state.
  virtual State Transform(const State& state) const = 0;

  // Returns the number of CPU cycles this opcode takes.
  virtual const uint32 cycles() const = 0;

  // Returns number of bytes this opcode takes as bytecode.
  virtual const uint32 bytes() const = 0;

  // Returns the assember version of this OpCode.
  virtual const std::string assembler() const = 0;
};

class LoadImmediate : public OpCode {
 public:
  virtual State Transform(const State& state) const;
  virtual const uint32 cycles() const override;
  virtual const uint32 bytes() const override;
  virtual const std::string assembler() const override;

 protected:
  LoadImmediate(uint8 value, State::Register reg);
  uint8 value_;
  State::Register register_;

 private:
  LoadImmediate() { /* Do not call me */ }
};

class LDA : public LoadImmediate {
 public:
  LDA(uint8 value) : LoadImmediate(value, State::Register::A) {}
};

class LDX : public LoadImmediate {
 public:
  LDX(uint8 value) : LoadImmediate(value, State::Register::X) {}
};

class LDY : public LoadImmediate {
 public:
  LDY(uint8 value) : LoadImmediate(value, State::Register::Y) {}
};

class StoreZeroPage : public OpCode {
 public:
  virtual State Transform(const State& state) const;
  virtual const uint32 cycles() const override;
  virtual const uint32 bytes() const override;
  virtual const std::string assembler() const override;

 protected:
  StoreZeroPage(uint8 address, State::Register reg);
  uint8 address_;
  State::Register register_;

 private:
  StoreZeroPage() { /* Do not call me */ }
};

class STA : public StoreZeroPage {
 public:
  STA(uint8 address) : StoreZeroPage(address, State::Register::A) {}
};

class STX : public StoreZeroPage {
 public:
  STX(uint8 address) : StoreZeroPage(address, State::Register::X) {}
};

class STY : public StoreZeroPage {
 public:
  STY(uint8 address) : StoreZeroPage(address, State::Register::Y) {}
};

class NOP : public OpCode {
 public:
  NOP() {}
  virtual State Transform(const State& state) const override;
  virtual const uint32 cycles() const override;
  virtual const uint32 bytes() const override;
  virtual const std::string assembler() const override;
};

}  // namespace op

}  // namespace vcsmc

#endif  // SRC_OPCODE_H_

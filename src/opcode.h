#ifndef SRC_OPCODE_H_
#define SRC_OPCODE_H_

#include <memory>
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
  virtual std::unique_ptr<State> Transform(State* state) const = 0;

  // Returns the number of CPU cycles this opcode takes.
  virtual const uint32 cycles() const = 0;

  // Returns number of bytes this opcode takes as bytecode.
  virtual const uint32 bytes() const = 0;

  // Returns the assembler version of this OpCode.
  virtual const std::string assembler() const = 0;
};

class LoadImmediate : public OpCode {
 public:
  LoadImmediate(uint8 value, Register reg);
  virtual std::unique_ptr<State> Transform(State* state) const override;
  virtual const uint32 cycles() const override;
  virtual const uint32 bytes() const override;
  virtual const std::string assembler() const override;

 protected:
  uint8 value_;
  Register register_;

 private:
  LoadImmediate() { /* Do not call me */ }
};

class LDA : public LoadImmediate {
 public:
  LDA(uint8 value) : LoadImmediate(value, Register::A) {}
};

class LDX : public LoadImmediate {
 public:
  LDX(uint8 value) : LoadImmediate(value, Register::X) {}
};

class LDY : public LoadImmediate {
 public:
  LDY(uint8 value) : LoadImmediate(value, Register::Y) {}
};

class StoreZeroPage : public OpCode {
 public:
  StoreZeroPage(TIA address, Register reg);
  virtual std::unique_ptr<State> Transform(State* state) const override;
  virtual const uint32 cycles() const override;
  virtual const uint32 bytes() const override;
  virtual const std::string assembler() const override;

 protected:
  TIA address_;
  Register register_;

 private:
  StoreZeroPage() { /* Do not call me */ }
};

class STA : public StoreZeroPage {
 public:
  STA(TIA address) : StoreZeroPage(address, Register::A) {}
};

class STX : public StoreZeroPage {
 public:
  STX(TIA address) : StoreZeroPage(address, Register::X) {}
};

class STY : public StoreZeroPage {
 public:
  STY(TIA address) : StoreZeroPage(address, Register::Y) {}
};

class NOP : public OpCode {
 public:
  NOP() {}
  virtual std::unique_ptr<State> Transform(State* state) const override;
  virtual const uint32 cycles() const override;
  virtual const uint32 bytes() const override;
  virtual const std::string assembler() const override;
};

}  // namespace op

// Factory methods in vcsmc namespace

std::unique_ptr<op::OpCode> makeLDA(uint8 value);
std::unique_ptr<op::OpCode> makeLDX(uint8 value);
std::unique_ptr<op::OpCode> makeLDY(uint8 value);
std::unique_ptr<op::OpCode> makeSTA(TIA address);
std::unique_ptr<op::OpCode> makeSTX(TIA address);
std::unique_ptr<op::OpCode> makeSTY(TIA address);
std::unique_ptr<op::OpCode> makeNOP();

}  // namespace vcsmc

#endif  // SRC_OPCODE_H_

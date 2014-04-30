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
  virtual const std::string& assembler() const = 0;
};

class LoadImmediate : public OpCode {
 public:
  LoadImmediate(uint8 value);
  virtual const uint32 cycles() const override;
  virtual const uint32 bytes() const override;

 protected:
  uint8 value_;
};

class LDA : public LoadImmediate {
 public:
  LDA(uint8 value);
  virtual State Transform(const State& state) const override;
  virtual const std::string& assembler() const override;
};

class LDX : public LoadImmediate {
 public:
  LDX(uint8 value);
  virtual State Transform(const State& state) const override;
  virtual const std::string& assembler() const override;
};

class LDY : public LoadImmediate {
 public:
  LDY(uint8 value);
  virtual State Transform(const State& state) const override;
  virtual const std::string& assembler() const override;
};

class StoreZeroPage : public OpCode {
 public:
  StoreZeroPage(uint8 address);
  virtual const uint32 cycles() const override;
  virtual const uint32 bytes() const override;
 protected:
  uint8 address_;
};

class STA : public StoreZeroPage {
 public:
  STA(uint8 address);
  virtual State Transform(const State& state) const override;
  virtual const std::string& assembler() const override;
};

class STX : public StoreZeroPage {
 public:
  STX(uint8 address);
  virtual State Transform(const State& state) const override;
  virtual const std::string& assembler() const override;
};

class STY : public StoreZeroPage {
 public:
  STY(uint address);
  virtual State Transform(const State& state) const override;
  virtual const std::string& assembler() const override;
};

class NOP : public OpCode {
 public:
  NOP();
  virtual State Transform(const State& state) const override;
  virtual const uint32 cycles() const override;
  virtual const uint32 bytes() const override;
  virtual const std::string& assembler() const override;
};

}  // namespace op

}  // namespace vcsmc

#endif  // SRC_OPCODE_H_

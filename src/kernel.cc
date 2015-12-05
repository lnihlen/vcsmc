#include "kernel.h"

#include <cassert>

// idea: at end of 4K space the ROM always returns zero for the last 8 bytes or
// so, and we always leave those blank.

// Things we need to be able to do with Kernels:
// a) quickly extract the bytecode for sending to the sim
// b) compute a hash over the bytecode for fingerprinting
// c) enforce validity by setting VBLANK and VSYNC at appropriate times
// d) enforce validity by ensuring that TIA registers have valid initial state.
// e) enforce validity by ensuring that we call JMP back up to the start of
//    the address space before rolling the PC out of the ROM address space.
// f) allow for efficient copy and mutation for subsequent generations.

// vector of uint32s. Each uint32 represents OpCode + arguments. LSB is OpCode,
// next significant byte is ARG1, etc. Can write functions to go from vector
// internel representation to bytecode and back.

// spasm - spec assembler - takes input assembly file with additional SPEC
// data and produces a .spec file, which contains packed uint32 data on where
// in the binary has to be hard coded (by cycle count), with hard-coded
// assembly data.

namespace {

// Given an opcode and two arguments will pack them into a uint32 and return
// the packed value.
inline uint32 PackOpcode(vcsmc::OpCode op, uint8 arg1, uint8 arg2) {
  uint32 packed = static_cast<uint32>(op) |
                  (static_cast<uint32>(arg1) << 8) |
                  (static_cast<uint32>(arg2) << 16);
  return packed;
}

// Given a packed opcode in |packed_op| will unpack the opcode and arguments
// into |target|, then return the number of bytes appended to target.
inline uint32 UnpackOpCode(uint32 packed_op, uint8* target) {
  vcsmc::OpCode op = static_cast<vcsmc::OpCode>(packed_op & 0x000000ff);
  *target = static_cast<uint8>(op);
  ++target;
  uint32 size = OpCodeBytes(op);
  assert(size > 0);
  if (size > 1) {
    uint8 arg1 = static_cast<uint8>((packed_op >> 8) & 0x000000ff);
    *target = arg1;
    ++target;
    if (size > 2) {
      assert(size < 4);
      uint8 arg2 = static_cast<uint8>((packed_op >> 16) & 0x000000ff);
      *target = arg2;
    }
  }
  return size;
}

}

namespace vcsmc {

void GenerateRandomKernelJob::Execute() {

}

}  // namespace vcmsc

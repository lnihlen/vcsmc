#ifndef SRC_ASSEMBLER_H_
#define SRC_ASSEMBLER_H_

#include <memory>
#include <string>
#include <vector>

#include "constants.h"
#include "types.h"

namespace vcsmc {

typedef std::unique_ptr<std::vector<uint32>> PackedOpcodes;

// Our asm input file description:
// a) Empty/whitespace only lines are ignored.
// b) Comments are semi-colons (;), ignore rest of line until end.
// c) TIA addresses are built-in, they are all upcase like VSYNC.
// d) OpCodes are case-insensitive, not all OpCodes are supported. Currently
//    only load immediate, store zero page, jmp, and nop are supported :).
// e) Immediate constants are prefaced with a hash mark (#)
// f) Hex numbers are proceeded by a dollar sign ($)
// g) Hex immediate get the hash first, so #$af not $#af

// Assembles an input string into the reurned PackedOpcodes or nullptr on
// error.
PackedOpcodes AssembleStringPacked(const std::string& input_string);

// Convenience method. Calls AssembleStringPacked, then unpacks the opcodes into
// bytecode, plus updates the values |cycles_out| and |size_out|.
std::unique_ptr<uint8[]> AssembleString(const std::string& input_string,
    uint32* cycles_out, size_t* size_out);

// Given an opcode and two arguments will pack them into a uint32 and return
// the packed value.
uint32 PackOpCode(vcsmc::OpCode op, uint8 arg1, uint8 arg2);

// Given a packed opcode in |packed_op| will unpack the opcode and arguments
// into |target|, then return the number of bytes appended to target.
size_t UnpackOpCode(uint32 packed_op, uint8* target);

size_t OpCodeBytes(OpCode op);
uint32 OpCodeCycles(OpCode op);

size_t OpCodeBytes(uint32 op);
uint32 OpCodeCycles(uint32 op);

}  // namespace vcsmc

#endif  // SRC_ASSEMBLER_H_

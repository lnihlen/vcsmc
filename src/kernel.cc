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

namespace vcsmc {

void GenerateRandomKernelJob::Execute() {

}

}  // namespace vcmsc

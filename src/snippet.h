#ifndef SRC_SNIPPET_H_
#define SRC_SNIPPET_H_

#include <array>

#include "types.h"

namespace vcsmc {

// A Snippet is a small amount of bytecode, up to two opcodes with arguments,
// generated by the State object when sequencing a Codon.
struct Snippet {
 public:
  Snippet() : size(0), duration(0), should_advance_register_rotation(false) {
    bytecode.fill(0);
  }

  void Insert(const uint8 byte) {
    bytecode[size] = byte;
    ++size;
    assert(size <= kSnippetMaxLength);
  }

  // Most Snippets will be short but a kWait snippet can issue up to 128 NOPs
  static const size_t kSnippetMaxLength = 128;
  std::array<uint8, kSnippetMaxLength> bytecode;
  size_t size;
  // Duration is always in units of CPU cycles.
  uint32 duration;
  // When this snippet is applied, if true, the register usage will update the
  // timing values stored to use for register rotation in sequencing.
  bool should_advance_register_rotation;
};

}  // namespace vcsmc

#endif  // SRC_SNIPPET_H_
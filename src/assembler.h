#ifndef SRC_ASSEMBLER_H_
#define SRC_ASSEMBLER_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "constants.h"

namespace vcsmc {

namespace op {
  class OpCode;
}

// Our asm input file description:
// a) Empty/whitespace only lines are ignored.
// b) Comments are semi-colons (;), ignore rest of line until end.
// c) TIA addresses are built-in, they are all upcase like VSYNC.
// d) OpCodes are case-insensitive, not all OpCodes are supported. Currently
//    only load immediate, store zero page, jmp, and nop are supported :).
// e) Immediate constants are prefaced with a hash mark (#)
// f) Hex numbers are proceeded by a dollar sign ($)
// g) Hex immediate get the hash first, so #$af not $#af
class Assembler {
 public:
  // Call me once per program run to initialize static lookup tables.
  static void InitAssemblerTables();

  // Appends |output_codes| with OpCodes assembled from |input_string|.
  // Returns false on error, in which case the contents of |output_codes| are
  // undefined.
  static bool AssembleString(
      const std::string& input_string,
      std::vector<std::unique_ptr<op::OpCode>>* output_codes);

  // Given a value like Register::A returns "a";
  static std::string RegisterToString(const Register reg);
  // Given a ZeroPage address returns either a human-readable name, if within
  // the TIA realm, or a hexadecimal number for the address.
  static std::string AddressToString(const uint8 address);
  // Really only here because the other things are here. Given 0xfe will return
  // the string "$fe".
  static std::string ByteToHexString(const uint8 value);
  static std::string ShortToHexString(const uint16 value);

 private:
  typedef std::unordered_map<std::string, uint8> TiaMap;

  static TiaMap* tia_symbol_map_;

  static bool ProcessLine(
      const std::string& line,
      std::vector<std::unique_ptr<op::OpCode>>* output_codes);

  // Parses a token that represents an immediate value, e.g. #0 or #$23. Returns
  // the value in parse_out, and true on parse success.
  static bool ParseImmediate(const std::string& token, uint8* immed_out);

  static bool ParseZeroPageAddress(
      const std::string& token, uint8* address_out);

  // Parses a value, if starting with $ is hex.
  static bool ParseByte(const std::string& token, uint8* byte_out);
  static bool ParseShort(const std::string& token, uint16* short_out);
};

}  // namespace vcsmc

#endif  // SRC_ASSEMBLER_H_

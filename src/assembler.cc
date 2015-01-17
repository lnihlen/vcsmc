#include "assembler.h"

#include <algorithm>
#include <cctype>
#include <stdlib.h>

#include "opcode.h"

namespace vcsmc {

Assembler::TiaMap* Assembler::tia_symbol_map_ = NULL;

// static
void Assembler::InitAssemblerTables() {
  if (tia_symbol_map_)
    return;
  tia_symbol_map_ = new TiaMap();
  for (uint8 i = 0; i < TIA::TIA_COUNT; ++i)
    tia_symbol_map_->insert(std::make_pair(AddressToString(i), i));
}

// static
bool Assembler::AssembleString(
    const std::string& input_string,
    std::vector<std::unique_ptr<op::OpCode>>* output_codes) {
  // Parse individual lines out of input string.
  size_t line_start = 0;
  while (line_start < input_string.length() &&
         line_start != std::string::npos) {
    size_t line_end = line_start < input_string.length() - 1 ?
        input_string.find_first_of('\n', line_start) :
        std::string::npos;
    const std::string line = line_end == std::string::npos ?
        input_string.substr(line_start) :
        input_string.substr(line_start, line_end - line_start);
    //printf("line_start: %d, line_end: %d, line: %s\n",
    //    line_start, line_end, line.c_str());
    if (!ProcessLine(line, output_codes))
      return false;
    line_start = line_end == std::string::npos ? line_end : line_end + 1;
  }

  return true;
}

// static
std::string Assembler::RegisterToString(const Register reg) {
  if (reg == Register::A) {
    return "a";
  } else if (reg == Register::X) {
    return "x";
  }

  return "y";
}

// static
std::string Assembler::AddressToString(const uint8 address) {
  // First see if within the TIA range or no:
  if (address < TIA::TIA_COUNT) {
    switch (address) {
      case VSYNC:   return "VSYNC";
      case VBLANK:  return "VBLANK";
      case WSYNC:   return "WSYNC";
      case RSYNC:   return "RSYNC";
      case NUSIZ0:  return "NUSIZ0";
      case NUSIZ1:  return "NUSIZ1";
      case COLUP0:  return "COLUP0";
      case COLUP1:  return "COLUP1";
      case COLUPF:  return "COLUPF";
      case COLUBK:  return "COLUBK";
      case CTRLPF:  return "CTRLPF";
      case REFP0:   return "REFP0";
      case REFP1:   return "REFP1";
      case PF0:     return "PF0";
      case PF1:     return "PF1";
      case PF2:     return "PF2";
      case RESP0:   return "RESP0";
      case RESP1:   return "RESP1";
      case RESM0:   return "RESM0";
      case RESM1:   return "RESM1";
      case RESBL:   return "RESBL";
      case AUDC0:   return "AUDC0";
      case AUDC1:   return "AUDC1";
      case AUDF0:   return "AUDF0";
      case AUDF1:   return "AUDF1";
      case AUDV0:   return "AUDV0";
      case AUDV1:   return "AUDV1";
      case GRP0:    return "GRP0";
      case GRP1:    return "GRP1";
      case ENAM0:   return "ENAM0";
      case ENAM1:   return "ENAM1";
      case ENABL:   return "ENABL";
      case HMP0:    return "HMP0";
      case HMP1:    return "HMP1";
      case HMM0:    return "HMM0";
      case HMM1:    return "HMM1";
      case HMBL:    return "HMBL";
      case VDELP0:  return "VDELP0";
      case VDELP1:  return "VDELP1";
      case RESMP0:  return "RESMP0";
      case RESMP1:  return "RESMP1";
      case HMOVE:   return "HMOVE";
      case HMCLR:   return "HMCLR";
      case CXCLR:   return "CXCLR";
      default:      return "???";
    }
  }
  // Must be a memory address, just print as one.
  return ByteToHexString(address);
}

// static
std::string Assembler::ByteToHexString(const uint8 value) {
  // Print the number value in hex to a stack buffer and wrap into string.
  char buf[4];
  std::snprintf(buf, 4, "$%01x", value);
  return std::string(buf);
}

// static
std::string Assembler::ShortToHexString(const uint16 value) {
  char buf[6];
  std::snprintf(buf, 6, "$%03x", value);
  return std::string(buf);
}

// static
bool Assembler::ProcessLine(
    const std::string& line,
    std::vector<std::unique_ptr<op::OpCode>>* output_codes) {
  // Early-out for empty lines.
  if (!line.length()) return true;

  // Tokenize line.
  std::vector<std::string> tokens;
  size_t token_start = line.find_first_not_of(" \t\n");
  while (token_start != std::string::npos) {
    size_t token_end = line.find_first_of(" \t\n", token_start);
    std::string token = token_end == std::string::npos ?
        line.substr(token_start) :
        line.substr(token_start, token_end - token_start);
    //printf("token_start: %d, token_end: %d, token: %s\n",
    //    token_start, token_end, token.c_str());
    // Comments stop the parsing of the rest of the line.
    if (token[0] == ';') break;
    tokens.push_back(token);
    token_start = line.find_first_not_of(" \t\n", token_end);
  }

  // Whitespace-or-comment-only lines are easy, just bail.
  if (!tokens.size()) return true;

  //for (int i = 0; i < tokens.size(); ++i) {
  //  printf("token %d: '%s' ", i, tokens[i].c_str());
  //}
  //printf("\n");

  // First token is now assumed to be an opcode, force it to lower case.
  std::string opcode_str = tokens[0];
  // All opcodes are 3 letters long, bail early if not so.
  if (opcode_str.length() != 3) return false;
  std::transform(opcode_str.begin(), opcode_str.end(), opcode_str.begin(),
      std::tolower);

  std::unique_ptr<op::OpCode> opcode;
  // Homebrew DFA FTW. Probably stupid.
  if (opcode_str[0] == 'l') {
    if (opcode_str[1] == 'd') {  // lda, ldx, ldy
      // Loads require a value argument.
      if (tokens.size() != 2)
        return false;
      // Only immediate loads currently supported.
      uint8 immed = 0;
      if (!ParseImmediate(tokens[1], &immed))
        return false;
      switch(opcode_str[2]) {
        case 'a':
          opcode = makeLDA(immed);
          break;

        case 'x':
          opcode = makeLDX(immed);
          break;

        case 'y':
          opcode = makeLDY(immed);
          break;

        default:
          return false;
      }
    } else {
      return false;
    }
  } else if (opcode_str[0] == 'n') {
    if (opcode_str[1] == 'o') {
      if (opcode_str[2] == 'p') {  // nop
        // A nop requires no argument.
        if (tokens.size() != 1)
          return false;
        opcode = makeNOP();
      } else {
        return false;
      }
    } else {
      return false;
    }
  } else if (opcode_str[0] == 's') {
    if (opcode_str[1] == 't') {  // sta, stx, sty
      // Stores require an address argument.
      if (tokens.size() != 2)
        return false;
      uint8 zero_page_address;
      if (!ParseZeroPageAddress(tokens[1], &zero_page_address))
        return false;
      switch(opcode_str[2]) {
        case 'a':
          opcode = makeSTA(static_cast<TIA>(zero_page_address));
          break;

        case 'x':
          opcode = makeSTX(static_cast<TIA>(zero_page_address));
          break;

        case 'y':
          opcode = makeSTY(static_cast<TIA>(zero_page_address));
          break;

        default:
          return false;
      }
    } else {
      return false;
    }
  } else {
    return false;
  }

  // We should have parsed a valid opcode, save it and bump address by its size.
  assert(opcode);
  output_codes->push_back(std::move(opcode));
  return true;
}

// static
bool Assembler::ParseImmediate(const std::string& token, uint8* immed_out) {
  // First character of the token should be a # to indicate an immediate value,
  // if not we have a problem.
  if (token.length() < 2 || token[0] != '#')
    return false;

  return ParseByte(token.substr(1), immed_out);
}

// static
bool Assembler::ParseZeroPageAddress(
    const std::string& token, uint8* address_out) {
  if (token.length() < 2)
    return false;
  // If value is numeric it will either start with a digit or a $ for hex.
  if (token[0] == '$' || isdigit(token[0])) {
    return ParseByte(token, address_out);
  } else {
    // Should be a TIA symbol.
    assert(tia_symbol_map_);  // Need to call InitAssemblerTables() first.
    TiaMap::iterator it = tia_symbol_map_->find(token);
    if (it == tia_symbol_map_->end())
      return false;
    *address_out = it->second;
  }
  return true;
}

// static
bool Assembler::ParseByte(const std::string& token, uint8* byte_out) {
  // First character could be a dollar sign $ to indicate hex, if not decimal.
  bool is_hex = token[0] == '$';
  if (is_hex) {
    // Rest of token should consist of either 1 or 2 hex digits.
    if (token.length() < 3 || token.length() > 4)
      return false;
    if (token.find_first_not_of("0123456789abcdef", 2) != std::string::npos)
      return false;
    uint32 parsed = strtoul(token.c_str() + 1, NULL, 16);
    *byte_out = static_cast<uint8>(parsed);
  } else {
    // Rest of token should consist of either 1, 2, or 3 decimal digits.
    if (token.length() < 3 || token.length() > 5)
      return false;
    if (token.find_first_not_of("0123456789", 1) != std::string::npos)
      return false;
    uint32 parsed = strtoul(token.c_str(), NULL, 10);
    if (parsed > 255)
      return false;
    *byte_out = static_cast<uint8>(parsed);
  }
  return true;
}

}  // namespace vcsmc

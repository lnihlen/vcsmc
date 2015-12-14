#include "assembler.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cctype>

namespace {

bool ParseByte(const std::string& token, uint8* byte_out) {
  if (!token.length()) return false;
  // First character could be a dollar sign $ to indicate hex, if not decimal.
  bool is_hex = token[0] == '$';
  if (is_hex) {
    // Rest of token should consist of either 1 or 2 hex digits.
    if (token.length() < 2 || token.length() > 3)
      return false;
    if (token.find_first_not_of("0123456789abcdef", 1) != std::string::npos)
      return false;
    uint32 parsed = strtoul(token.c_str() + 1, NULL, 16);
    *byte_out = static_cast<uint8>(parsed);
  } else {
    // Rest of token should consist of either 1, 2, or 3 decimal digits.
    if (token.length() > 3)
      return false;
    if (token.find_first_not_of("0123456789") != std::string::npos)
      return false;
    uint32 parsed = strtoul(token.c_str(), NULL, 10);
    if (parsed > 255)
      return false;
    *byte_out = static_cast<uint8>(parsed);
  }
  return true;
}

bool ParseShort(const std::string& token, uint16* short_out) {
  if (!token.length()) return false;
  bool is_hex = token[0] == '$';
  if (is_hex) {
    // number should consist of 1-4 hex digits
    if (token.length() < 2 || token.length() > 5)
      return false;
    if (token.find_first_not_of("0123456789abcdef", 1) != std::string::npos)
      return false;
    uint32 parsed = strtoul(token.c_str() + 1, NULL, 16);
    *short_out = static_cast<uint16>(parsed);
  } else {
    if (token.length() > 5)
      return false;
    if (token.find_first_not_of("0123456789") != std::string::npos)
      return false;
    uint32 parsed = strtoul(token.c_str(), NULL, 10);
    if (parsed > 65536)
      return false;
    *short_out = static_cast<uint16>(parsed);
  }
  return true;
}

bool ParseImmediate(const std::string& token, uint8* immed_out) {
  // First character of the token should be a # to indicate an immediate value,
  // if not we have a problem.
  if (token.length() < 2 || token[0] != '#')
    return false;

  return ParseByte(token.substr(1), immed_out);
}

bool ParseZeroPageAddress(
    const std::string& token, uint8* address_out) {
  if (token.length() < 2 || token.length() > 8)
    return false;
  // If value is numeric it will either start with a digit or a $ for hex.
  if (token[0] == '$' || isdigit(token[0])) {
    return ParseByte(token, address_out);
  } else {
    // Pack string into a double for use in a switch statement with identity
    // hashes.
    uint64 tia_pack = 0;
    for (size_t i = 0; i < token.length(); ++i)
      tia_pack = (tia_pack << 8) | static_cast<uint64>(token[i]);
    switch (tia_pack) {
      case 0x5653594e43:
        *address_out = vcsmc::TIA::VSYNC;
        return true;

      case 0x56424c414e4b:
        *address_out = vcsmc::TIA::VBLANK;
        return true;

      case 0x5753594e43:
        *address_out = vcsmc::TIA::WSYNC;
        return true;

      case 0x5253594e43:
        *address_out = vcsmc::TIA::RSYNC;
        return true;

      case 0x4e5553495a30:
        *address_out = vcsmc::TIA::NUSIZ0;
        return true;

      case 0x4e5553495a31:
        *address_out = vcsmc::TIA::NUSIZ1;
        return true;

      case 0x434f4c555030:
        *address_out = vcsmc::TIA::COLUP0;
        return true;

      case 0x434f4c555031:
        *address_out = vcsmc::TIA::COLUP1;
        return true;

      case 0x434f4c555046:
        *address_out = vcsmc::TIA::COLUPF;
        return true;

      case 0x434f4c55424b:
        *address_out = vcsmc::TIA::COLUBK;
        return true;

      case 0x4354524c5046:
        *address_out = vcsmc::TIA::CTRLPF;
        return true;

      case 0x5245465030:
        *address_out = vcsmc::TIA::REFP0;
        return true;

      case 0x5245465031:
        *address_out = vcsmc::TIA::REFP1;
        return true;

      case 0x504630:
        *address_out = vcsmc::TIA::PF0;
        return true;

      case 0x504631:
        *address_out = vcsmc::TIA::PF1;
        return true;

      case 0x504632:
        *address_out = vcsmc::TIA::PF2;
        return true;

      case 0x5245535030:
        *address_out = vcsmc::TIA::RESP0;
        return true;

      case 0x5245535031:
        *address_out = vcsmc::TIA::RESP1;
        return true;

      case 0x5245534d30:
        *address_out = vcsmc::TIA::RESM0;
        return true;

      case 0x5245534d31:
        *address_out = vcsmc::TIA::RESM1;
        return true;

      case 0x524553424c:
        *address_out = vcsmc::TIA::RESBL;
        return true;

      case 0x4155444330:
        *address_out = vcsmc::TIA::AUDC0;
        return true;

      case 0x4155444331:
        *address_out = vcsmc::TIA::AUDC1;
        return true;

      case 0x4155444630:
        *address_out = vcsmc::TIA::AUDF0;
        return true;

      case 0x4155444631:
        *address_out = vcsmc::TIA::AUDF1;
        return true;

      case 0x4155445630:
        *address_out = vcsmc::TIA::AUDV0;
        return true;

      case 0x4155445631:
        *address_out = vcsmc::TIA::AUDV1;
        return true;

      case 0x47525030:
        *address_out = vcsmc::TIA::GRP0;
        return true;

      case 0x47525031:
        *address_out = vcsmc::TIA::GRP1;
        return true;

      case 0x454e414d30:
        *address_out = vcsmc::TIA::ENAM0;
        return true;

      case 0x454e414d31:
        *address_out = vcsmc::TIA::ENAM1;
        return true;

      case 0x454e41424c:
        *address_out = vcsmc::TIA::ENABL;
        return true;

      case 0x484d5030:
        *address_out = vcsmc::TIA::HMP0;
        return true;

      case 0x484d5031:
        *address_out = vcsmc::TIA::HMP1;
        return true;

      case 0x484d4d30:
        *address_out = vcsmc::TIA::HMM0;
        return true;

      case 0x484d4d31:
        *address_out = vcsmc::TIA::HMM1;
        return true;

      case 0x484d424c:
        *address_out = vcsmc::TIA::HMBL;
        return true;

      case 0x5644454c5030:
        *address_out = vcsmc::TIA::VDELP0;
        return true;

      case 0x5644454c5031:
        *address_out = vcsmc::TIA::VDELP1;
        return true;

      case 0x5644454c424c:
        *address_out = vcsmc::TIA::VDELBL;
        return true;

      case 0x5245534d5030:
        *address_out = vcsmc::TIA::RESMP0;
        return true;

      case 0x5245534d5031:
        *address_out = vcsmc::TIA::RESMP1;
        return true;

      case 0x484d4f5645:
        *address_out = vcsmc::TIA::HMOVE;
        return true;

      case 0x484d434c52:
        *address_out = vcsmc::TIA::HMCLR;
        return true;

      case 0x4358434c52:
        *address_out = vcsmc::TIA::CXCLR;
        return true;

      default:
        return false;
    }
  }
  return false;
}

bool ProcessLine(const std::string& line, std::vector<uint32>* opcodes) {
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
    // Comments stop the parsing of the rest of the line.
    if (token[0] == ';') break;
    tokens.push_back(token);
    token_start = line.find_first_not_of(" \t\n", token_end);
  }

  // Whitespace-or-comment-only lines are easy, just bail.
  if (!tokens.size()) return true;

  // First token is now assumed to be an opcode, force it to lower case.
  std::string opcode_str = tokens[0];
  // All opcodes are 3 letters long, bail early if not so.
  if (opcode_str.length() != 3) return false;
  std::transform(opcode_str.begin(), opcode_str.end(), opcode_str.begin(),
      std::tolower);

  if (opcode_str == "jmp") {
    // jmp needs a short argument
    if (tokens.size() != 2)
      return false;
    uint16 address = 0;
    if (!ParseShort(tokens[1], &address))
      return false;
    opcodes->push_back(PackOpCode(vcsmc::JMP_Absolute,
        static_cast<uint8>(address & 0x00ff),
        static_cast<uint8>((address >> 8) & 0x00ff)));
    return true;
  } else if (opcode_str[0] == 'l') {
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
          opcodes->push_back(PackOpCode(vcsmc::LDA_Immediate, immed, 0));
          return true;

        case 'x':
          opcodes->push_back(PackOpCode(vcsmc::LDX_Immediate, immed, 0));
          return true;

        case 'y':
          opcodes->push_back(PackOpCode(vcsmc::LDY_Immediate, immed, 0));
          return true;

        default:
          return false;
      }
    }
  } else if (opcode_str[0] == 'n') {
    if (opcode_str[1] == 'o' && opcode_str[2] == 'p') {  // nop
      // A nop requires no argument.
      if (tokens.size() != 1)
        return false;
      opcodes->push_back(PackOpCode(vcsmc::NOP_Implied, 0, 0));
      return true;
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
          opcodes->push_back(PackOpCode(vcsmc::STA_ZeroPage,
                zero_page_address, 0));
          return true;

        case 'x':
          opcodes->push_back(PackOpCode(vcsmc::STX_ZeroPage,
                zero_page_address, 0));
          return true;

        case 'y':
          opcodes->push_back(PackOpCode(vcsmc::STY_ZeroPage,
                zero_page_address, 0));
          return true;

        default:
          return false;
      }
    }
  }
  return false;
}

}  // namespace

namespace vcsmc {

PackedOpcodes AssembleStringPacked(const std::string& input_string) {
  vcsmc::PackedOpcodes ops(new std::vector<uint32>());
  // Parse individual lines out of input string.
  size_t line_start = 0;
  while (line_start < input_string.length() &&
         line_start != std::string::npos) {
    size_t line_end = line_start < input_string.length() - 1 ?
        input_string.find_first_of('\n', line_start) :
        std::string::npos;
    const std::string& line = line_end == std::string::npos ?
        input_string.substr(line_start) :
        input_string.substr(line_start, line_end - line_start);
    if (!ProcessLine(line, ops.get()))
      return nullptr;
    line_start = line_end == std::string::npos ? line_end : line_end + 1;
  }
  return ops;
}

std::unique_ptr<uint8[]> AssembleString(const std::string& input_string,
    uint32* cycles_out, size_t* size_out) {
  PackedOpcodes ops = AssembleStringPacked(input_string);
  if (!ops) return nullptr;
  uint32 cycles = 0;
  size_t size = 0;
  for (size_t i = 0; i < ops->size(); ++i) {
    OpCode op = static_cast<OpCode>(ops->at(i) & 0x000000ff);
    size += OpCodeBytes(op);
    cycles += OpCodeCycles(op);
  }
  std::unique_ptr<uint8[]> bytecode(new uint8[size]);
  uint8* bytes = bytecode.get();
  for (size_t i = 0; i < ops->size(); ++i) {
    size_t opcode_bytes = UnpackOpCode(ops->at(i), bytes);
    bytes += opcode_bytes;
  }
  *cycles_out = cycles;
  *size_out = size;
  return bytecode;
}

// Given an opcode and two arguments will pack them into a uint32 and return
// the packed value.
uint32 PackOpCode(vcsmc::OpCode op, uint8 arg1, uint8 arg2) {
  uint32 packed = static_cast<uint32>(op) |
                  (static_cast<uint32>(arg1) << 8) |
                  (static_cast<uint32>(arg2) << 16);
  return packed;
}

// Given a packed opcode in |packed_op| will unpack the opcode and arguments
// into |target|, then return the number of bytes appended to target.
size_t UnpackOpCode(uint32 packed_op, uint8* target) {
  vcsmc::OpCode op = static_cast<vcsmc::OpCode>(packed_op & 0x000000ff);
  *target = static_cast<uint8>(op);
  ++target;
  size_t size = OpCodeBytes(op);
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

size_t OpCodeBytes(OpCode op) {
  switch (op) {
    case JMP_Absolute:
      return 3;

    case LDA_Immediate:
    case LDX_Immediate:
    case LDY_Immediate:
      return 2;

    case NOP_Implied:
      return 1;

    case STA_ZeroPage:
    case STX_ZeroPage:
    case STY_ZeroPage:
      return 2;
  }
  return 0;
}

uint32 OpCodeCycles(OpCode op) {
  switch (op) {
    case JMP_Absolute:
      return 3;

    case LDA_Immediate:
    case LDX_Immediate:
    case LDY_Immediate:
      return 2;

    case NOP_Implied:
      return 2;

    case STA_ZeroPage:
    case STX_ZeroPage:
    case STY_ZeroPage:
      return 3;
  }
  return 0;
}

size_t OpCodeBytes(uint32 op) {
  return OpCodeBytes(static_cast<OpCode>(op & 0x000000ff));
}

uint32 OpCodeCycles(uint32 op) {
  return OpCodeCycles(static_cast<OpCode>(op & 0x000000ff));
}

}  // namespace vcsmc

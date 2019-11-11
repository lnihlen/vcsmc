#include "serialization.h"

#include <cassert>
#include <cinttypes>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "assembler.h"
#include "constants.h"

namespace {

const std::string kBase64Tokens(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/");

uint32 ValueOfBase64Token(char token) {
  if ('A' <= token && token <= 'Z') {
    return token - 'A';
  } else if ('a' <= token && token <= 'z') {
    return 26 + token - 'a';
  } else if ('0' <= token && token <= '9') {
    return 52 + token - '0';
  } else if (token == '+') {
    return 62;
  } else if (token == '/') {
    return 63;
  }
  printf("undefined b64 character: %c ord: %d\n", token, token);
  assert(false);
  return 64;
}

/*
void RemoveWhitespace(const std::string& input, std::string& output) {
  output = "";
  for (size_t i = 0; i < input.size(); ++i) {
    if (input[i] == ' ' || input[i] == '\n' || input[i] == '\t')
      continue;
    output += input[i];
  }
}
*/

}

namespace vcsmc {

std::string Base64Encode(const uint8* bytes, size_t size, size_t indent) {
  std::string indent_str = indent > 0 ? "\n" : "";
  for (size_t i = 0; i < indent; ++i)
    indent_str += " ";
  std::string result = indent_str;
  size_t line_counter = indent;
  // 3 bytes is 24 bits is 4 base 64 digits.
  for (size_t i = 0; i < size / 3; ++i) {
    uint32 three_bytes = (static_cast<uint32>(bytes[i * 3]) << 24) |
                         (static_cast<uint32>(bytes[(i * 3) + 1]) << 16) |
                         (static_cast<uint32>(bytes[(i * 3) + 2]) << 8);
    char four_b64[5];
    four_b64[0] = kBase64Tokens[three_bytes >> 26];
    three_bytes = three_bytes << 6;
    four_b64[1] = kBase64Tokens[three_bytes >> 26];
    three_bytes = three_bytes << 6;
    four_b64[2] = kBase64Tokens[three_bytes >> 26];
    three_bytes = three_bytes << 6;
    four_b64[3] = kBase64Tokens[three_bytes >> 26];
    four_b64[4] = '\0';
    if (line_counter + 4 > 80) {
      result += indent_str;
      line_counter = indent;
    }
    result += std::string(four_b64);
    line_counter += 4;
  }
  if (size % 3 == 2) {
    // If two bytes remaining (16) bits we need three more characters to encode.
    char three_b64[4];
    uint32 two_bytes = (static_cast<uint32>(bytes[size - 2]) << 24) |
                       (static_cast<uint32>(bytes[size - 1]) << 16);
    three_b64[0] = kBase64Tokens[two_bytes >> 26];
    two_bytes = two_bytes << 6;
    three_b64[1] = kBase64Tokens[two_bytes >> 26];
    two_bytes = two_bytes << 6;
    three_b64[2] = kBase64Tokens[two_bytes >> 26];
    three_b64[3] = '\0';
    if (line_counter + 3 > 80) {
      result += indent_str;
    }
    result += std::string(three_b64);
  } else if (size % 3 == 1) {
    // If one byte remaining (8) bits we need two more characters to encode.
    char two_b64[3];
    uint32 one_byte = static_cast<uint32>(bytes[size - 1] << 24);
    two_b64[0] = kBase64Tokens[one_byte >> 26];
    one_byte = one_byte << 6;
    two_b64[1] = kBase64Tokens[one_byte >> 26];
    two_b64[2] = '\0';
    if (line_counter + 2 > 80) {
      result += indent_str;
    }
    result += std::string(two_b64);
  }
  if (indent > 0) result += "\n";
  return result;
}

// It is assumed string is already concatenated for the decode.
std::unique_ptr<uint8[]> Base64Decode(const std::string& base64, size_t size) {
  std::unique_ptr<uint8[]> bytes_ptr(new uint8[size]);
  uint8* bytes = bytes_ptr.get();
  // Unpack 4 b64 digits into 3 bytes at a time.
  for (size_t i = 0; i < base64.size() / 4; ++i) {
    uint32 accum = (ValueOfBase64Token(base64[i * 4]) << 26) |
                   (ValueOfBase64Token(base64[(i * 4) + 1]) << 20) |
                   (ValueOfBase64Token(base64[(i * 4) + 2]) << 14) |
                   (ValueOfBase64Token(base64[(i * 4) + 3]) << 8);
    bytes[0] = static_cast<uint8>(accum >> 24);
    bytes[1] = static_cast<uint8>((accum >> 16) & 0x000000ff);
    bytes[2] = static_cast<uint8>((accum >> 8) & 0x000000ff);
    bytes += 3;
  }
  if (base64.size() % 4 == 3) {
    // Three characters remaining representing 18 bits or bytes.
    uint32 accum = (ValueOfBase64Token(base64[base64.size() - 3]) << 26) |
                   (ValueOfBase64Token(base64[base64.size() - 2]) << 20) |
                   (ValueOfBase64Token(base64[base64.size() - 1]) << 14);
    bytes[0] = static_cast<uint8>(accum >> 24);
    bytes[1] = static_cast<uint8>((accum >> 16) & 0x000000ff);
    bytes += 2;
  } else if (base64.size() % 4 == 2) {
    // Two characters remaining representing 12 bits or 1 byte.
    uint32 accum = (ValueOfBase64Token(base64[base64.size() - 2]) << 26) |
                   (ValueOfBase64Token(base64[base64.size() - 1]) << 20);
    bytes[0] = static_cast<uint8>(accum >> 24);
    ++bytes;
  } else if (base64.size() % 4 == 1) {
    // Shouldn't happen!
    assert(false);
  }
  assert(size == static_cast<size_t>(bytes - bytes_ptr.get()));
  return bytes_ptr;
}

}  // namespace vcsmc

#include "serialization.h"

#include <cassert>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <yaml.h>

#include "assembler.h"
#include "constants.h"
#include "range.h"

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

void RemoveWhitespace(const std::string& input, std::string& output) {
  output = "";
  for (size_t i = 0; i < input.size(); ++i) {
    if (input[i] == ' ' || input[i] == '\n' || input[i] == '\t')
      continue;
    output += input[i];
  }
}

vcsmc::Generation ParseGenerationInternal(yaml_parser_t parser) {
  yaml_event_t event;
  enum ParserState {
    kIdle,              // Between kernel documents.
    kKernel,            // Parsing a kernel but between mappings.
    kFingerprint,       // Parsing a kernel fingerprint mapping.
    kSeed,              // Parsing a kernel seed mapping.
    kSpecSequenceMap,   // Parsing a specs sequence mapping.
    kSpecSequence,      // Parsing a spec sequence.
    kSpec,              // Parsing a spec but between mappings.
    kSpecFirstCycle,    // Parsing a spec first_cycle mapping.
    kSpecLastCycle,     // Parsing a spec last_cycle mapping.
    kSpecSize,          // Parsing a spec size mapping.
    kSpecBytecode,      // Parsing a spec bytecode mapping.
    kOpcodeRangesMap,   // Parsing an opcode ranges mapping.
    kRangeSequence,     // Parsing opcode ranges sequence but outside a range.
    kRange,             // Parsing an opcode range but between mappings.
    kRangeOffset,       // Parsing a range offset mapping.
    kRangeSize,         // Parsing a range size mapping.
    kRangeBytecode,     // Parsing a range bytecode mapping.
    kDone               // Parser has reached the end of the file.
  };
  ParserState state = kIdle;
  vcsmc::Generation gen(new std::vector<std::shared_ptr<vcsmc::Kernel>>());
  std::string scalar;
  uint64 fingerprint = 0;
  std::string seed_str;
  uint32 first_cycle = vcsmc::kInfinity;
  uint32 last_cycle = vcsmc::kInfinity;
  size_t spec_size = 0;
  std::string bytecode_str;
  vcsmc::SpecList spec_list(new std::vector<vcsmc::Spec>());
  std::vector<vcsmc::Range> dynamic_areas;
  uint32 range_offset = 0;
  uint32 range_size = 0;
  std::vector<std::unique_ptr<uint8[]>> packed_opcodes;
  while (state != kDone) {
    if (!yaml_parser_parse(&parser, &event)) return nullptr;
    switch (event.type) {
      case YAML_STREAM_START_EVENT:
        break;

      // Kernels are individual documents within the larger yaml file.
      case YAML_DOCUMENT_START_EVENT:
        if (state != kIdle)
          return nullptr;
        break;

      case YAML_MAPPING_START_EVENT:
        if (state == kIdle) {
          state = kKernel;
        } else if (state == kSpecSequence) {
          state = kSpec;
        } else if (state == kRangeSequence) {
          state = kRange;
        } else {
          return nullptr;
        }
        break;

      case YAML_SEQUENCE_START_EVENT:
        if (state == kSpecSequenceMap) {
          state = kSpecSequence;
        } else if (state == kOpcodeRangesMap) {
          state = kRangeSequence;
        } else {
          return nullptr;
        }
        break;

      case YAML_SCALAR_EVENT:
        scalar = std::string(reinterpret_cast<char*>(event.data.scalar.value));
        if (state == kKernel) {
          // Top-level mapping of kernel elements.
          if (scalar == "fingerprint") {
            state = kFingerprint;
          } else if (scalar == "seed") {
            state = kSeed;
          } else if (scalar == "specs") {
            state = kSpecSequenceMap;
          } else if (scalar == "opcode_ranges") {
            state = kOpcodeRangesMap;
          } else {
            return nullptr;
          }
        } else if (state == kFingerprint) {
          fingerprint = strtoull(scalar.c_str(), NULL, 16);
          state = kKernel;
        } else if (state == kSeed) {
          seed_str = scalar;
          state = kKernel;
        } else if (state == kSpec) {
          // Spec mapping elements.
          if (scalar == "first_cycle") {
            state = kSpecFirstCycle;
          } else if (scalar == "last_cycle") {
            state = kSpecLastCycle;
          } else if (scalar == "size") {
            state = kSpecSize;
          } else if (scalar == "bytecode") {
            state = kSpecBytecode;
          } else {
            return nullptr;
          }
        } else if (state == kSpecFirstCycle) {
          first_cycle = strtoul(scalar.c_str(), NULL, 10);
          state = kSpec;
        } else if (state == kSpecLastCycle) {
          last_cycle = strtoul(scalar.c_str(), NULL, 10);
          state = kSpec;
        } else if (state == kSpecSize) {
          spec_size = strtoul(scalar.c_str(), NULL, 10);
          state = kSpec;
        } else if (state == kSpecBytecode) {
          RemoveWhitespace(scalar, bytecode_str);
          state = kSpec;
        } else if (state == kRange) {
          // Opcode Range mapping elements.
          if (scalar == "offset") {
            state = kRangeOffset;
          } else if (scalar == "size") {
            state = kRangeSize;
          } else if (scalar == "bytecode") {
            state = kRangeBytecode;
          } else {
            return nullptr;
          }
        } else if (state == kRangeOffset) {
          range_offset = strtoul(scalar.c_str(), NULL, 10);
          state = kRange;
        } else if (state == kRangeSize) {
          range_size = strtoul(scalar.c_str(), NULL, 10);
          state = kRange;
        } else if (state == kRangeBytecode) {
          RemoveWhitespace(scalar, bytecode_str);
          state = kRange;
        } else {
          return nullptr;
        }
        break;

      case YAML_MAPPING_END_EVENT:
        if (state == kSpec) {
          // Should have a complete spec here.
          if (first_cycle == vcsmc::kInfinity ||
              last_cycle == vcsmc::kInfinity  ||
              spec_size == 0                  ||
              bytecode_str.size() == 0) {
            return nullptr;
          }
          spec_list->emplace_back(
              vcsmc::Range(first_cycle, last_cycle),
              spec_size,
              vcsmc::Base64Decode(bytecode_str, spec_size));
          first_cycle = vcsmc::kInfinity;
          last_cycle = vcsmc::kInfinity;
          spec_size = 0;
          bytecode_str = "";
          state = kSpecSequence;
        } else if (state == kRange) {
          if (range_offset == 0 ||
              range_size == 0   ||
              bytecode_str.size() == 0) {
            return nullptr;
          }
          dynamic_areas.emplace_back(
              vcsmc::Range(range_offset, range_offset + range_size));
          packed_opcodes.emplace_back(
              vcsmc::Base64Decode(bytecode_str, range_size));
          range_offset = 0;
          range_size = 0;
          bytecode_str = "";
          state = kRangeSequence;
        } else if (state == kKernel) {
          if (fingerprint == 0           ||
              seed_str == ""             ||
              spec_list->size() == 0     ||
              dynamic_areas.size() == 0  ||
              packed_opcodes.size() == 0 ||
              dynamic_areas.size() != packed_opcodes.size()) {
            return nullptr;
          }
          gen->emplace_back(new vcsmc::Kernel(
                seed_str, spec_list, dynamic_areas, packed_opcodes));
          fingerprint = 0;
          seed_str = "";
          spec_list.reset(new std::vector<vcsmc::Spec>());
          dynamic_areas.clear();
          packed_opcodes.clear();
          state = kIdle;
        } else {
          return nullptr;
        }
        break;

      case YAML_ALIAS_EVENT:
      case YAML_NO_EVENT:
        break;

      case YAML_SEQUENCE_END_EVENT:
        if (state == kSpecSequence) {
          state = kKernel;
        } else if (state == kRangeSequence) {
          state = kKernel;
        } else {
          return nullptr;
        }
        break;

      case YAML_DOCUMENT_END_EVENT:
        if (state != kIdle) return nullptr;
        break;

      case YAML_STREAM_END_EVENT:
        if (state == kIdle) {
          state = kDone;
        } else {
          return nullptr;
        }
        break;
    }
    yaml_event_delete(&event);
  }
  return gen;
}

vcsmc::SpecList ParseSpecListInternal(yaml_parser_t parser) {
  yaml_event_t event;
  enum ParserState {
    kIdle,   // Outside of a Spec, waiting for the next one.
    kReady,  // Inside of a Spec, but not midway through a key-value pair.
    kFirstCycle,  // Next scalar should be the numeric value of first_cycle.
    kHardDuration,  // Optional next scalar should be hard-coded duration.
    kAssembler,   // Next scalar value should be the assembler code.
    kDone,  // Stream is finished, exit parser loop.
  };
  ParserState state = kIdle;
  uint32 first_cycle = vcsmc::kInfinity;
  char* assembly = nullptr;
  size_t assembly_length = 0;
  uint32 assembly_cycles = 0;
  uint32 hard_duration = 0;
  uint32 last_cycle = 0;
  std::unique_ptr<uint8[]> bytecode;
  std::string scalar;
  vcsmc::SpecList spec_list(new std::vector<vcsmc::Spec>());
  while (state != kDone) {
    if (!yaml_parser_parse(&parser, &event)) return nullptr;
    switch (event.type) {
      case YAML_STREAM_START_EVENT:
      case YAML_DOCUMENT_START_EVENT:
      case YAML_SEQUENCE_START_EVENT:
        break;

      // SpecList YAML is considered as a sequence of mappings. Each mapping in
      // the sequence represents an individual Spec entry. Therefore the start
      // of a new mapping should represent the start of a new Spec entry.
      case YAML_MAPPING_START_EVENT:
        first_cycle = vcsmc::kInfinity;
        assembly = nullptr;
        assembly_length = 0;
        assembly_cycles = 0;
        hard_duration = 0;
        last_cycle = 0;
        state = kReady;
        break;

      // Mapping key-value pairs appear as a sequence of scalar events, one each
      // for both key and value.
      case YAML_SCALAR_EVENT:
        scalar = std::string(reinterpret_cast<char*>(event.data.scalar.value));
        if (state == kReady) {
          if (scalar == "first_cycle") {
            state = kFirstCycle;
          } else if (scalar == "assembler") {
            state = kAssembler;
          } else if (scalar == "hard_duration") {
            state = kHardDuration;
          } else {
            return nullptr;
          }
        } else if (state == kFirstCycle) {
          first_cycle = strtoul(scalar.c_str(), NULL, 10);
          state = kReady;
        } else if (state == kAssembler) {
          bytecode =
            vcsmc::AssembleString(scalar, &assembly_cycles, &assembly_length);
          if (!bytecode) return nullptr;
          state = kReady;
        } else if (state == kHardDuration) {
          hard_duration = strtoul(scalar.c_str(), NULL, 10);
          state = kReady;
        } else {
          return nullptr;
        }
        break;

      // End of map, check that we have a complete entry and create a new Spec
      // from it.
      case YAML_MAPPING_END_EVENT:
        last_cycle = first_cycle +
          (hard_duration > 0 ? hard_duration : assembly_cycles);
        spec_list->emplace_back(
            vcsmc::Range(first_cycle, last_cycle),
            assembly_length,
            std::move(bytecode));
        state = kIdle;
        break;

      case YAML_ALIAS_EVENT:
      case YAML_NO_EVENT:
        break;

      case YAML_SEQUENCE_END_EVENT:
      case YAML_DOCUMENT_END_EVENT:
        break;

      case YAML_STREAM_END_EVENT:
        assert(state == kIdle);
        state = kDone;
        break;
    }
    yaml_event_delete(&event);
  }
  return spec_list;
}

}

namespace vcsmc {

Generation ParseGenerationFile(const std::string& file_name) {
  yaml_parser_t parser;
  yaml_parser_initialize(&parser);
  FILE* file = fopen(file_name.c_str(), "rb");
  if (!file) return nullptr;
  yaml_parser_set_input_file(&parser, file);
  Generation gen = ParseGenerationInternal(parser);
  yaml_parser_delete(&parser);
  fclose(file);
  return gen;
}

Generation ParseGenerationString(const std::string& spec_string) {
  yaml_parser_t parser;
  yaml_parser_initialize(&parser);
  yaml_parser_set_input_string(&parser,
      reinterpret_cast<const unsigned char*>(spec_string.c_str()),
      spec_string.size());
  Generation gen = ParseGenerationInternal(parser);
  yaml_parser_delete(&parser);
  return gen;
}

bool SaveGenerationFile(Generation gen, const std::string& file_name) {
  int fd = open(file_name.c_str(),
      O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
  if (fd < 0) return false;
  for (size_t i = 0; i < gen->size(); ++i) {
    std::string kernel_str;
    bool result = SaveKernelToString(gen->at(i), kernel_str);
    if (!result) return false;
    kernel_str = "---\n" + kernel_str;
    size_t bytes_written = write(fd, kernel_str.c_str(), kernel_str.size());
    if (bytes_written != kernel_str.size())
      return false;
  }
  close(fd);
  return true;
}

bool SaveKernelToString(std::shared_ptr<Kernel> kernel,
    std::string& string_out) {
  char buf[128];
  snprintf(buf, 128, "fingerprint: %016llx\n", kernel->fingerprint());
  string_out += std::string(buf);
  string_out += "seed: " + kernel->GetRandomState() + "\n";
  const SpecList specs = kernel->specs();
  string_out += "specs:\n";
  for (size_t j = 0; j < specs->size(); ++j) {
    snprintf(buf, 128, "  - first_cycle: %d\n"
                       "    last_cycle: %d\n"
                       "    size: %lu\n"
                       "    bytecode: |",
        specs->at(j).range().start_time(),
        specs->at(j).range().end_time(),
        specs->at(j).size());
    string_out += std::string(buf);
    string_out +=
        Base64Encode(specs->at(j).bytecode(), specs->at(j).size(), 6);
  }
  string_out += "opcode_ranges:\n";
  for (size_t j = 0; j < kernel->dynamic_areas().size(); ++j) {
    snprintf(buf, 128, "  - offset: %d\n"
                       "    size: %d\n"
                       "    bytecode: |",
      kernel->dynamic_areas()[j].start_time(),
      kernel->dynamic_areas()[j].Duration());
      string_out += std::string(buf);
      string_out += Base64Encode(
          kernel->bytecode() + kernel->dynamic_areas()[j].start_time(),
          kernel->dynamic_areas()[j].Duration(), 6);
  }
  return true;
}

bool SaveKernelToFile(std::shared_ptr<Kernel> kernel,
    const std::string& file_name) {
  std::string kernel_str;
  if (!SaveKernelToString(kernel, kernel_str))
    return false;
  int fd = open(file_name.c_str(),
      O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
  if (fd < 0) return false;
  size_t bytes_written = write(fd, kernel_str.c_str(), kernel_str.size());
  if (bytes_written != kernel_str.size())
    return false;
  close(fd);
  return true;
}

bool AppendKernelBinary(std::shared_ptr<Kernel> kernel,
    const std::string& file_name) {
  int fd = open(file_name.c_str(),
      O_APPEND | O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR);
  if (fd < 0) {
    return false;
  }
  // Write kernel binary.
  size_t bytes_written = write(fd, kernel->bytecode(), kernel->bytecode_size());
  close(fd);
  return bytes_written == kernel->bytecode_size();
}

SpecList ParseSpecListFile(const std::string& file_name) {
  yaml_parser_t parser;
  yaml_parser_initialize(&parser);
  FILE* file = fopen(file_name.c_str(), "rb");
  if (!file) return nullptr;
  yaml_parser_set_input_file(&parser, file);
  SpecList result = ParseSpecListInternal(parser);
  yaml_parser_delete(&parser);
  fclose(file);
  return result;
}

SpecList ParseSpecListString(const std::string& spec_string) {
  yaml_parser_t parser;
  yaml_parser_initialize(&parser);
  yaml_parser_set_input_string(&parser,
      reinterpret_cast<const unsigned char*>(spec_string.c_str()),
      spec_string.size());
  SpecList result = ParseSpecListInternal(parser);
  yaml_parser_delete(&parser);
  return result;
}

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
  return std::move(bytes_ptr);
}

}  // namespace vcsmc

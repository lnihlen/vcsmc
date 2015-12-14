#include "spec.h"

#include <cassert>
#include <stdlib.h>
#include <stdio.h>
#include <yaml.h>

#include "assembler.h"
#include "constants.h"

namespace {

vcsmc::SpecList ParseSpecListInternal(yaml_parser_t parser) {
  yaml_event_t event;
  enum ParserState {
    kIdle,   // Outside of a Spec, waiting for the next one.
    kReady,  // Inside of a Spec, but not midway through a key-value pair.
    kFirstCycle,  // Next scalar should be the numeric value of first_cycle.
    kHardDuration,  // Optional next scalar should be hard-coded duration.
    kBytecode,   // Next scalar value should be the assembler code.
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
          } else if (scalar == "bytecode") {
            state = kBytecode;
          } else if (scalar == "hard_duration") {
            state = kHardDuration;
          } else {
            return nullptr;
          }
        } else if (state == kFirstCycle) {
          first_cycle = strtoul(scalar.c_str(), NULL, 10);
          state = kReady;
        } else if (state == kBytecode) {
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

Spec::Spec(const Range& range, size_t size, std::unique_ptr<uint8[]> bytecode)
  : range_(range),
    size_(size),
    bytecode_(std::move(bytecode)) {}

Spec::Spec(const Spec& spec)
   : range_(spec.range_),
     size_(spec.size_),
     bytecode_(new uint8[spec.size_]) {
  std::memcpy(bytecode_.get(), spec.bytecode_.get(), spec.size_);
}

SpecList ParseSpecListFile(const std::string& file_name) {
  yaml_parser_t parser;
  yaml_parser_initialize(&parser);
  FILE* file = fopen(file_name.c_str(), "rb");
  if (!file) return nullptr;
  yaml_parser_set_input_file(&parser, file);
  SpecList result = ParseSpecListInternal(parser);
  yaml_parser_delete(&parser);
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

}  // namespace vcsmc

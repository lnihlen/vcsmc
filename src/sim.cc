// sim - TIA gate/part-level simulator development tool.

#include <stdio.h>
#include <yaml.h>

#include "parts/and.h"
#include "parts/block.h"
#include "parts/d1.h"
#include "parts/d2.h"
#include "parts/dgnor.h"
#include "parts/dgnot.h"
#include "parts/dir.h"
#include "parts/dl.h"
#include "parts/dxnor.h"
#include "parts/f1.h"
#include "parts/net.h"
#include "parts/not.h"
#include "parts/part.h"
#include "parts/waa.h"
#include "parts/xor.h"

#include <map>
#include <string>
#include <vector>

std::shared_ptr<vcsmc::parts::Part> MakePart(
    const std::map<std::string, std::string>& scalar_pairs,
    const std::map<std::string, std::vector<std::string>>& vector_pairs) {
  std::shared_ptr<vcsmc::parts::Part> part;
  // Need at least a 'name' and a 'type' key in scalar_pairs, and 'in' and 'out'
  // vector keys to build a valid Part.
  if (scalar_pairs.find("name") == scalar_pairs.end() ||
      scalar_pairs.find("type") == scalar_pairs.end()) {
    return part;
  }

  const std::string& name = scalar_pairs.find("name")->second;
  const std::string& type = scalar_pairs.find("type")->second;

  if (type == "AND") {
    part.reset(new vcsmc::parts::And(name));
  } else if (type == "BLOCK_1C61") {
    part.reset(new vcsmc::parts::Block1C61(name));
  } else if (type == "BLOCK_1D41") {
    part.reset(new vcsmc::parts::Block1D41(name));
  } else if (type == "BLOCK_4A81") {
    part.reset(new vcsmc::parts::Block4A81(name));
  } else if (type == "BLOCK_4D61") {
    part.reset(new vcsmc::parts::Block4D61(name));
  } else if (type == "D1") {
    part.reset(new vcsmc::parts::D1(name));
  } else if (type == "D2") {
    part.reset(new vcsmc::parts::D2(name));
  } else if (type == "DGNOR") {
    part.reset(new vcsmc::parts::DGNor(name));
  } else if (type == "DGNOT") {
    part.reset(new vcsmc::parts::DGNot(name));
  } else if (type == "DIR") {
    part.reset(new vcsmc::parts::Dir(name));
  } else if (type == "DL") {
    part.reset(new vcsmc::parts::Dl(name));
  } else if (type == "DXNOR") {
    part.reset(new vcsmc::parts::DXNor(name));
  } else if (type == "F1") {
    part.reset(new vcsmc::parts::F1(name));
  } else if (type == "L") {
    part.reset(new vcsmc::parts::L(name));
  } else if (type == "NOT") {
    part.reset(new vcsmc::parts::Not(name));
  } else if (type == "WAA") {
    part.reset(new vcsmc::parts::WAA(name));
  } else if (type == "XOR") {
    part.reset(new vcsmc::parts::Xor(name));
  } else {
    fprintf(stderr, "unrecognized part type %s for part named %s.\n",
        type.c_str(), name.c_str());
  }

  return part;
}

std::shared_ptr<vcsmc::parts::Net> FindOrMakeNet(
    const std::string& net_name,
    std::map<std::string, std::shared_ptr<vcsmc::parts::Net>>* nets) {
  std::shared_ptr<vcsmc::parts::Net> net;
  if (nets->find(net_name) == nets->end()) {
    net.reset(new vcsmc::parts::Net(net_name));
    nets->insert(std::make_pair(net_name, net));
    return net;
  }
  return nets->find(net_name)->second;
}

int main(int argc, char* argv[]) {
  const char* file_name = "sim/page1.yaml";
  FILE* fh = fopen(file_name, "r");
  if (!fh) {
    fprintf(stderr, "error opening sim file!\n");
    return -1;
  }
  yaml_parser_t parser;
  if (!yaml_parser_initialize(&parser)) {
    fprintf(stderr, "error initializing yaml parser.\n");
    return -1;
  }
  yaml_parser_set_input_file(&parser, fh);

  std::map<std::string, std::shared_ptr<vcsmc::parts::Net>> nets;
  std::map<std::string, std::shared_ptr<vcsmc::parts::Part>> parts;

  yaml_token_t token;
  yaml_parser_scan(&parser, &token);
  std::string block_key;
  std::string scalar;
  std::vector<std::string> sequence_values;
  enum ParserState {
    OUTSIDE_BLOCK,  // Not currently parsing a Part.
    INSIDE_BLOCK,   // In the middle of parsing a part, but not depending on
                    // a particular sequence of tokens.
    AWAITING_KEY_SCALAR,  // Just got a YAML_KEY_TOKEN, so next token should be
                          // a SCALAR with that key's data payload.
    AWAITING_VALUE,  // Just got a YAML_VALUE_TOKEN, so next token should
                     // be either a SCALAR with that value's data payload, or
                     // the start of a sequence of scalars.
    INSIDE_FLOW_SEQUENCE  // Expecting a series of SCALARS to add.
  };
  std::map<std::string, std::string> scalar_pairs;
  std::map<std::string, std::vector<std::string>> vector_pairs;
  ParserState parser_state = OUTSIDE_BLOCK;
  std::shared_ptr<vcsmc::parts::Part> part;
  while (token.type != YAML_STREAM_END_TOKEN) {
    switch (token.type) {
      // Received at the start of a Part description.
      case YAML_DOCUMENT_START_TOKEN:
        if (parser_state != OUTSIDE_BLOCK) {
          fprintf(stderr, "ERROR parsing %s on line %lu: "
              "unexpected start of document.\n", file_name,
              token.start_mark.line);
          return -1;
        }
        break;

      // Also received at the start of a Part description, but could be used
      // to indicate the start of any other block map.
      case YAML_BLOCK_MAPPING_START_TOKEN:
        if (parser_state != OUTSIDE_BLOCK) {
          fprintf(stderr, "ERROR parsing %s on line %lu: nested documents.\n",
              file_name, token.start_mark.line);
          return -1;
        }
        parser_state = INSIDE_BLOCK;
        break;

      // Received for any key in the block mapping, next token should be a
      // SCALAR token containing the key string.
      case YAML_KEY_TOKEN:
        if (parser_state != INSIDE_BLOCK) {
          fprintf(stderr, "ERROR parsing %s on line %lu: key not expected.\n",
              file_name, token.start_mark.line);
          return -1;
        }
        parser_state = AWAITING_KEY_SCALAR;
        break;

      // Received for any value part of a key: value pair in the block mapping.
      // Next token should be a SCALAR token containing the value string.
      case YAML_VALUE_TOKEN:
        if (parser_state != INSIDE_BLOCK) {
          fprintf(stderr, "ERROR parsing %s on line %lu: value not expected.\n",
              file_name, token.start_mark.line);
          return -1;
        }
        parser_state = AWAITING_VALUE;
        break;

      // The actual data payload token.
      case YAML_SCALAR_TOKEN:
        // LibYAML provides UTF8 strings as unsigned char*, we cast so
        // std::string can pick it up.
        scalar = std::string((const char*)token.data.scalar.value);
        if (parser_state == AWAITING_KEY_SCALAR) {
          block_key = scalar;
          parser_state = INSIDE_BLOCK;
        } else if (parser_state == AWAITING_VALUE) {
          if (block_key.empty()) {
            fprintf(stderr, "ERROR parsing %s on line %lu: scalar value %s"
                "defined without key\n", file_name, token.start_mark.line,
                scalar.c_str());
            return -1;
          }
          if (scalar_pairs.find(block_key) != scalar_pairs.end() ||
              vector_pairs.find(block_key) != vector_pairs.end()) {
            fprintf(stderr, "ERROR parsing %s on line %lu: duplicate key %s.\n",
                file_name, token.start_mark.line, block_key.c_str());
            return -1;
          }
          scalar_pairs[block_key] = scalar;
          block_key.clear();
          parser_state = INSIDE_BLOCK;
        } else if (parser_state == INSIDE_FLOW_SEQUENCE) {
          sequence_values.push_back(scalar);
        } else {
          fprintf(stderr, "ERROR parsing %s on line %lu: "
              "unexpected scalar %s.\n", file_name, token.start_mark.line,
              scalar.c_str());
          return -1;
        }
        break;

      // Start of a sequence, which is expected to be a value, probably the
      // list of inputs or outputs.
      case YAML_FLOW_SEQUENCE_START_TOKEN:
        if (parser_state != AWAITING_VALUE) {
          fprintf(stderr, "ERROR parsing %s on line %lu: "
              "unexpected sequence.\n", file_name, token.start_mark.line);
          return -1;
        }
        parser_state = INSIDE_FLOW_SEQUENCE;
        break;

      // End of a sequence, we can save it to the map of lists.
      case YAML_FLOW_SEQUENCE_END_TOKEN:
        if (parser_state != INSIDE_FLOW_SEQUENCE || block_key.empty()) {
          fprintf(stderr, "ERROR parsing %s on line %lu: "
              "unexpected end of sequence.\n", file_name,
              token.start_mark.line);
          return -1;
        }
        if (vector_pairs.find(block_key) != vector_pairs.end() ||
            scalar_pairs.find(block_key) != scalar_pairs.end()) {
          fprintf(stderr, "ERROR parsing %s on line %lu: duplicate key %s.\n",
              file_name, token.start_mark.line, block_key.c_str());
          return -1;
        }
        vector_pairs[block_key] = sequence_values;
        block_key.clear();
        sequence_values.clear();
        parser_state = INSIDE_BLOCK;
        break;

      // Received at the end of the Part description. Should be able to actually
      // assemble the Part from the provided data, or report an error.
      case YAML_BLOCK_END_TOKEN:
        if (parser_state != INSIDE_BLOCK) {
          fprintf(stderr, "ERROR parsing %s on line %lu: "
              "unexpected end of block.\n", file_name, token.start_mark.line);
          return -1;
        }

        /*
        // Dump parse to console.
        for (auto it = scalar_pairs.begin(); it != scalar_pairs.end(); ++it) {
          printf("%s: %s\n", it->first.c_str(), it->second.c_str());
        }
        for (auto it = vector_pairs.begin(); it != vector_pairs.end(); ++it) {
          printf("%s: [ ", it->first.c_str());
          for (auto vt = it->second.begin(); vt != it->second.end(); ++vt) {
            printf("%s ", vt->c_str());
          }
          printf("]\n");
        }
        printf("---\n");
        */

        // Build Part and attach Nets.
        part = MakePart(scalar_pairs, vector_pairs);
        if (!part)
          return -1;
        if (vector_pairs.find("in") == vector_pairs.end() ||
            vector_pairs.find("out") == vector_pairs.end()) {
          fprintf(stderr, "ERROR: part %s missing input or output vector.\n",
              part->name().c_str());
          return -1;
        }
        if (parts.find(part->name()) != parts.end()) {
          fprintf(stderr, "ERROR: duplicate part name %s.\n",
              part->name().c_str());
          return -1;
        }
        {
          const std::vector<std::string>& ins =
              vector_pairs.find("in")->second;
          const std::vector<std::string>& outs =
              vector_pairs.find("out")->second;
          if (part->NumberOfInputs() != ins.size()) {
            fprintf(stderr, "ERROR: part %s has %lu inputs, expecting %d.\n",
                part->name().c_str(), ins.size(), part->NumberOfInputs());
            return -1;
          }
          for (uint32 i = 0; i < ins.size(); ++i) {
            std::shared_ptr<vcsmc::parts::Net> net =
                FindOrMakeNet(ins[i], &nets);
            net->AddOutput(part);
          }
          if (part->NumberOfOutputs() != outs.size()) {
            fprintf(stderr, "ERROR: part %s has %lu outputs, expecting %d.\n",
                part->name().c_str(), outs.size(), part->NumberOfOutputs());
            return -1;
          }
          for (uint32 i = 0; i < outs.size(); ++i) {
            std::shared_ptr<vcsmc::parts::Net> net =
                FindOrMakeNet(outs[i], &nets);
            net->AddInput(part);
          }
        }

        // Clear out parse values for next Part.
        block_key.clear();
        sequence_values.clear();
        scalar_pairs.clear();
        vector_pairs.clear();
        parser_state = OUTSIDE_BLOCK;
        break;

      // One of the other diverse tokens the YAML parser sends, not needed so
      // we simply ignore.
      default:
        break;
    }

    yaml_token_delete(&token);
    yaml_parser_scan(&parser, &token);
  }

  yaml_token_delete(&token);
  yaml_parser_delete(&parser);
  fclose(fh);

  // Check all Nets should have at least one input and one output.
  for (auto it = nets.begin(); it != nets.end(); ++it) {
    if (it->second->NumberOfInputs() == 0)
      printf("warning: net %s has no inputs.\n", it->second->name().c_str());
    if (it->second->NumberOfOutputs() == 0)
      printf("warning: net %s has no outputs.\n", it->second->name().c_str());
  }

  return 0;
}

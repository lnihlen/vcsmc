#include "parts/block.h"

namespace vcsmc {

namespace parts {


// Block1C64 ===================================================================

Block1C61::Block1C61(const std::string& name)
    : Part(name) {
}

bool Block1C61::Sim() {
  return false;
}

uint32 Block1C61::NumberOfInputs() const {
  return 4;
}

uint32 Block1C61::NumberOfOutputs() const {
  return 1;
}


// Block1D41 ===================================================================

Block1D41::Block1D41(const std::string& name)
    : Part(name) {
}

bool Block1D41::Sim() {
  return false;
}

uint32 Block1D41::NumberOfInputs() const {
  return 1;
}

uint32 Block1D41::NumberOfOutputs() const {
  return 2;
}

// Block4D41 ===================================================================

Block4A81::Block4A81(const std::string& name)
    : Part(name) {
}

bool Block4A81::Sim() {
  return false;
}

uint32 Block4A81::NumberOfInputs() const {
  return 3;
}

uint32 Block4A81::NumberOfOutputs() const {
  return 3;
}

// Block4D41 ===================================================================

Block4D41::Block4D41(const std::string& name)
    : Part(name) {
}

bool Block4D41::Sim() {
  return false;
}

uint32 Block4D41::NumberOfInputs() const {
  return 3;
}

uint32 Block4D41::NumberOfOutputs() const {
  return 3;
}

}  // namespace parts

}  // namespace vcsmc

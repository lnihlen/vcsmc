#include "parts/xor.h"

namespace vcsmc {

namespace parts {

Xor::Xor(const std::string& name)
    : Part(name) {
}

bool Xor::Sim() {
  return false;
}

uint32 Xor::NumberOfInputs() const {
  return 2;
}

uint32 Xor::NumberOfOutputs() const {
  return 1;
}

}  // namespace parts

}  // namespace vcsmc

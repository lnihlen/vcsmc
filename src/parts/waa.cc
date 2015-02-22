#include "parts/waa.h"

namespace vcsmc {

namespace parts {

WAA::WAA(const std::string& name)
    : Part(name) {
}

bool WAA::Sim() {
  return false;
}

uint32 WAA::NumberOfInputs() const {
  return 6;
}

uint32 WAA::NumberOfOutputs() const {
  return 8;
}

}  // namespace parts

}  // namespace vcsmc

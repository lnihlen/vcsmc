#include "parts/d2.h"

namespace vcsmc {

namespace parts {

D2::D2(const std::string& name)
    : Part(name) {
}

bool D2::Sim() {
  return false;
}

uint32 D2::NumberOfInputs() const {
  return 4;
}

uint32 D2::NumberOfOutputs() const {
  return 1;
}

}  // namespace parts

}  // namespace vcsmc

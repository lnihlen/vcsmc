#include "parts/d1.h"

namespace vcsmc {

namespace parts {

D1::D1(const std::string& name)
    : Part(name) {
}

bool D1::Sim() {
  return false;
}

uint32 D1::NumberOfInputs() const {
  return 3;
}

uint32 D1::NumberOfOutputs() const {
  return 1;
}

}  // namespace parts

}  // namespace vcsmc

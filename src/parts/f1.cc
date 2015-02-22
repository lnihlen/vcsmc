#include "parts/f1.h"

namespace vcsmc {

namespace parts {

F1::F1(const std::string& name)
    : Part(name) {
}

bool F1::Sim() {
  return false;
}

uint32 F1::NumberOfInputs() const {
  return 4;
}

uint32 F1::NumberOfOutputs() const {
  return 2;
}

}  // namespace parts

}  // namespace vcsmc

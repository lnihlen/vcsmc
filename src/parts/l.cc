#include "parts/l.h"

namespace vcsmc {

namespace parts {

L::L(const std::string& name)
    : Part(name) {
}

bool L::Sim() {
  return false;
}

uint32 L::NumberOfInputs() const {
  return 3;
}

uint32 L::NumberOfOutputs() const {
  return 2;
}

}  // namespace parts

}  // namespace vcsmc

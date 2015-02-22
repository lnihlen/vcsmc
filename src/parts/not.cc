#include "parts/not.h"

namespace vcsmc {

namespace parts {

Not::Not(const std::string& name)
    : Part(name) {
}

bool Not::Sim() {
  return false;
}

uint32 Not::NumberOfInputs() const {
  return 1;
}

uint32 Not::NumberOfOutputs() const {
  return 1;
}

}  // namespace parts

}  // namespace vcsmc

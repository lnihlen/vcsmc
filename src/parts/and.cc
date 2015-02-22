#include "parts/and.h"

namespace vcsmc {

namespace parts {

And::And(const std::string& name)
    : Part(name) {
}

bool And::Sim() {
  return false;
}

uint32 And::NumberOfInputs() const {
  return 2;
}

uint32 And::NumberOfOutputs() const {
  return 1;
}

}  // namespace parts

}  // namespace vcsmc

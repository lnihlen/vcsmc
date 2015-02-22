#include "parts/dl.h"

namespace vcsmc {

namespace parts {

Dl::Dl(const std::string& name)
    : Part(name) {
}

bool Dl::Sim() {
  return false;
}

uint32 Dl::NumberOfInputs() const {
  return 4;
}

uint32 Dl::NumberOfOutputs() const {
  return 1;
}

}  // namespace parts

}  // namespace vcsmc

#include "parts/dgnor.h"

namespace vcsmc {

namespace parts {

DGNor::DGNor(const std::string& name)
    : Part(name) {
}

bool DGNor::Sim() {
  return false;
}

uint32 DGNor::NumberOfInputs() const {
  return 2;
}

uint32 DGNor::NumberOfOutputs() const {
  return 1;
}

}  // namespace parts

}  // namespace vcsmc

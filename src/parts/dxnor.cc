#include "parts/dxnor.h"

namespace vcsmc {

namespace parts {

DXNor::DXNor(const std::string& name)
    : Part(name) {
}

bool DXNor::Sim() {
  return false;
}

uint32 DXNor::NumberOfInputs() const {
  return 2;
}

uint32 DXNor::NumberOfOutputs() const {
  return 1;
}

}  // namespace parts

}  // namespace vcsmc

#include "parts/dgnot.h"

namespace vcsmc {

namespace parts {

DGNot::DGNot(const std::string& name)
    : Part(name) {
}

bool DGNot::Sim() {
  return false;
}

uint32 DGNot::NumberOfInputs() const {
  return 1;
}

uint32 DGNot::NumberOfOutputs() const {
  return 1;
}

}  // namespace parts

}  // namespace vcsmc

#include "parts/dir.h"

namespace vcsmc {

namespace parts {

Dir::Dir(const std::string& name)
    : Part(name) {
}

bool Dir::Sim() {
  return false;
}

uint32 Dir::NumberOfInputs() const {
  return 4;
}

uint32 Dir::NumberOfOutputs() const {
  return 1;
}

}  // namespace parts

}  // namespace vcsmc

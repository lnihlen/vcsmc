#include "state.h"

#include <cstring>

namespace vcsmc {

State::State() {
  std::memset(tia_state_, 0, TIA_CONT);
  std::memset(register_state_, 0, REGISTER_COUNT);
}


}  // namespace vcsmc

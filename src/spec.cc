#include "spec.h"

namespace vcsmc {

Spec::Spec(TIA tia, uint8 value, const Range& range)
  : tia_(tia),
    value_(value),
    range_(range) {
}

}  // namespace vcsmc

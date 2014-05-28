#ifndef SRC_SPEC_H_
#define SRC_SPEC_H_

#include "constants.h"
#include "range.h"
#include "types.h"

namespace vcsmc {

// A Spec defines a single request for a TIA state change and a deadline.
// The deadline is the form of a range of times.
class Spec {
 public:
  Spec(TIA tia, uint8 value, const Range& range);

  const TIA tia() const { return tia_; }
  const uint8 value() const { return value_; }
  const Range& range() const { return range_; }

 private:
  TIA tia_;
  uint8 value_;
  Range range_;
};

}  // namespace vcsmc

#endif  // SRC_SPEC_H_

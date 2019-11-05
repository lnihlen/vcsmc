#ifndef SRC_SPEC_H_
#define SRC_SPEC_H_

#include <array>
#include <memory>
#include <vector>

#include "codon.h"
#include "constants.h"
#include "types.h"

namespace vcsmc {

// A Spec defines a start time and a certain number of Codons (up to some max
// limit kMaxSpecCodons) to be translated at that start time. This ensures that
// the randomly generated Kernels can continue to be valid frame programs, as
// well as allows individual frames to add audio or other interactive
// programming bits to a class of Kernels, perhaps all representing the
// unvarying requirements of an individual frame of video.
class Spec {
 public:
  static const uint32 kMaxSpecCodons = 8;

  // Construct an empty Spec with supplied start time. Codons can be inserted
  // using Insert().
  Spec(const uint32 start_time = 0) : start_time_(start_time), size_(0) { }

  void Insert(const Codon codon) {
    codons_[size_] = codon;
    ++size_;
    assert(size_ <= kMaxSpecCodons);
  }

  // Required start time in cycles.
  uint32 start_time() const { return start_time_; }
  // Number of Codons in this Spec.
  size_t size() const { return size_; }
  // Const access to the Codons contained within this Spec.
  const uint32* codons() const { return codons_.data(); }

 private:
  uint32 start_time_;
  size_t size_;
  std::array<uint32, kMaxSpecCodons> codons_;
};

typedef std::shared_ptr<std::vector<Spec>> SpecList;
typedef std::vector<Spec>::const_iterator SpecIterator;

}  // namespace vcsmc

#endif  // SRC_SPEC_H_

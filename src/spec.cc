#include "spec.h"

#include <cstring>

namespace vcsmc {

Spec::Spec(const Range& range, size_t size, std::unique_ptr<uint8[]> bytecode)
  : range_(range),
    size_(size),
    bytecode_(std::move(bytecode)) {}

Spec::Spec(const Spec& spec)
   : range_(spec.range_),
     size_(spec.size_),
     bytecode_(new uint8[spec.size_]) {
  std::memcpy(bytecode_.get(), spec.bytecode_.get(), spec.size_);
}

const Spec& Spec::operator=(const Spec& spec) {
  range_ = spec.range_;
  size_ = spec.size_;
  bytecode_.reset(new uint8[spec.size_]);
  std::memcpy(bytecode_.get(), spec.bytecode_.get(), spec.size_);
  return *this;
}

}  // namespace vcsmc

#include "log.h"

#include <cassert>

namespace vcsmc {

// static
Log* Log::instance_ = nullptr;

Log::Log() {
  assert(!instance_);
  instance_ = this;
}

Log::~Log() {
  assert(instance_);
  instance_ = nullptr;
}

// static
bool Log::Setup() {
  return true;
}

}  // namespace vcsmc

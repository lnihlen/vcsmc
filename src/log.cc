#include "log.h"

#include <cassert>

namespace vcsmc {

// static
Log* Log::instance_ = NULL;

Log::Log() {
  assert(!instance_);
  instance_ = this;
}

Log::~Log() {
  assert(instance_);
  instance_ = NULL;
}

// static
bool Log::Setup() {
  return true;
}

}  // namespace vcsmc

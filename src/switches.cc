#include "switches.h"

#include <cassert>
#include <getopt.h>

namespace {

static struct option long_options[] = {
  { "input", required_argument, 0, Switches::Options::INPUT_FILE_SPEC },
  { "
};

}

namespace vcsmc {

// static
Switches* Switches::instance_ = NULL;

Switches::Switches() {
  assert(!instance_);
  instance_ = this;
}

Switches::~Switches() {
  assert(instance_);
  instance_ = NULL;
}

// static
bool Switches::Parse(int argc, char* argv[]) {
  // Uses getopt_long to parse the command line. There should be one non-option
  // argument, which is the input file spec.

/*
  // Need at least an input file name.
  if (argc < 2) {
    return false;
  }

  instance_->input_file_ = std::string(argv[1]);

  for (int i = 2; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "-o") {
      // Check for follow-on argument of filename.
    } else {
      // No argument matched, return parsing failure.
      return false;
    }
  }
*/
  return true;
}

// static
const std::string& Switches::input_file() {
  assert(instance_);
  return instance_->input_file_;
}

}  // namespace vcsmc

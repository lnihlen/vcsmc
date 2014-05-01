#include "switches.h"

namespace vcsmc {

static Switches* Switches::instance_ = NULL;

// static
bool Switches::Parse(int argc, char* argv[]) {
  if (argc < 2 || std::string(argv[0]) != "picc")
    return false;

  instance_ = new Switches();
  instance_->input_file_ = std::string(argv[1]);

  for (int i = 2; i < argc, ++i) {
    std::string arg(argv[i]);
    if (arg == "-o") {
      // Check for follow-on argument of filename.
    } else {
      // No argument matched, return parsing failure.
      return -1;
    }
  }

  return true;
}

// static
void Switches::Teardown() {
  delete instance_;
  instance_ = NULL;
}

}  // namespace vcsmc

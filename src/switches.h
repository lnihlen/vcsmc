#ifndef SRC_SWITCHES_H_
#define SRC_SWITCHES_H_

#include <string>

namespace vcsmc {

// Singleton class to report command line switches to the picc program, and to
// support defaults living all in one place.
class Switches {
 public:
  // Initializes the singleton pointer, or asserts if one already exists.
  Switches();
  // NULLs the singleton pointer.
  ~Switches();
  // Parse command-arguments. Returns false on failure to parse. Also
  // initializes the Switches singleton for future queries.
  static bool Parse(int argc, char* argv[]);

  static const std::string& input_file();
  static const std::string& output_file();

 private:
  static Switches* instance_;
  std::string input_file_;
  std::string output_file_;
};

}

#endif  // SRC_SWITCHES_H_

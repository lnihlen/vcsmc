#ifndef SRC_CL_PROGRAM_H_
#define SRC_CL_PROGRAM_H_

#include <string>

namespace vcsmc {

class CLProgram {
 public:
  enum Programs : size_t {
    kCiede2k = 0,
    kDownsampleErrors = 1,
    kFFTRadix2 = 2,
    kInverseFFTNormalize = 3,
    kRGBToLab = 4,
    kSpectralResidual = 5,
    PROGRAM_COUNT = 6
  };

  static std::string GetProgramString(Programs program);
  static std::string GetProgramName(Programs program);
};

}  // namespace vcsmc

#endif  // SRC_CL_PROGRAM_H_

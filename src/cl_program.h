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
    kPackComplexToReal = 4,
    kRGBToLab = 5,
    kSpectralResidual = 6,
    kSquare = 7,
    kSum = 8,
    kUnpackRealToComplex = 9,
    PROGRAM_COUNT = 10
  };

  static std::string GetProgramString(Programs program);
  static std::string GetProgramName(Programs program);
};

}  // namespace vcsmc

#endif  // SRC_CL_PROGRAM_H_

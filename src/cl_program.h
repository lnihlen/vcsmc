#ifndef SRC_CL_PROGRAM_H_
#define SRC_CL_PROGRAM_H_

#include <string>

namespace vcsmc {

class CLProgram {
 public:
  enum Programs : size_t {
    kCiede2k              = 0,
    kConvolve             = 1,
    kDownsampleErrors     = 2,
    kFFTRadix2            = 3,
    kInverseFFTNormalize  = 4,
    kMakeBitmap           = 5,
    kMean                 = 6,
    kPackComplexToReal    = 7,
    kRGBToLab             = 8,
    kSpectralResidual     = 9,
    kSquare               = 10,
    kStandardDeviation    = 11,
    kUnpackRealToComplex  = 12,
    PROGRAM_COUNT         = 13
  };

  static std::string GetProgramString(Programs program);
  static std::string GetProgramName(Programs program);
};

}  // namespace vcsmc

#endif  // SRC_CL_PROGRAM_H_

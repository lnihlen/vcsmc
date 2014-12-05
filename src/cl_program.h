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
    kHistogramClasses     = 4,
    kInverseFFTNormalize  = 5,
    kKMeansClassify       = 6,
    kKMeansColor          = 7,
    kMakeBitmap           = 8,
    kMean                 = 9,
    kPackComplexToReal    = 10,
    kRGBToLab             = 11,
    kSpectralResidual     = 12,
    kSquare               = 13,
    kStandardDeviation    = 14,
    kUnpackRealToComplex  = 15,
    PROGRAM_COUNT         = 16
  };

  static std::string GetProgramString(Programs program);
  static std::string GetProgramName(Programs program);
};

}  // namespace vcsmc

#endif  // SRC_CL_PROGRAM_H_

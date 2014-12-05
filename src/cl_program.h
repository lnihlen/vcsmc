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
    kFitPlayer            = 4,
    kFitPlayfield         = 5,
    kHistogramClasses     = 6,
    kInverseFFTNormalize  = 7,
    kKMeansClassify       = 8,
    kKMeansColor          = 9,
    kMakeBitmap           = 10,
    kMean                 = 11,
    kPackComplexToReal    = 12,
    kRGBToLab             = 13,
    kSpectralResidual     = 14,
    kSquare               = 15,
    kStandardDeviation    = 16,
    kUnpackRealToComplex  = 17,
    PROGRAM_COUNT         = 18
  };

  static std::string GetProgramString(Programs program);
  static std::string GetProgramName(Programs program);
};

}  // namespace vcsmc

#endif  // SRC_CL_PROGRAM_H_

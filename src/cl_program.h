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
    kInverseFFTNormalize  = 6,
    kKMeansClassify       = 7,
    kKMeansColor          = 8,
    kMakeBitmap           = 9,
    kMean                 = 10,
    kPackComplexToReal    = 11,
    kRGBToLab             = 12,
    kSpectralResidual     = 13,
    kSquare               = 14,
    kStandardDeviation    = 15,
    kUnpackRealToComplex  = 16,
    PROGRAM_COUNT         = 17
  };

  static std::string GetProgramString(Programs program);
  static std::string GetProgramName(Programs program);
};

}  // namespace vcsmc

#endif  // SRC_CL_PROGRAM_H_

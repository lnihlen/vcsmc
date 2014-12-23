#ifndef SRC_CL_PROGRAM_H_
#define SRC_CL_PROGRAM_H_

#include <string>

namespace vcsmc {

class CLProgram {
 public:
  enum Programs : size_t {
    kCiede2k              = 0,
    kConvolve             = 1,
    kDownsampleColors     = 2,
    kDownsampleErrors     = 3,
    kFFTRadix2            = 4,
    kFitPlayer            = 5,
    kFitPlayfield         = 6,
    kInverseFFTNormalize  = 7,
    kKMeansClassify       = 8,
    kKMeansColor          = 9,
    kLabToRGB             = 10,
    kMakeBitmap           = 11,
    kMean                 = 12,
    kMeanShift            = 13,
    kMinErrorColor        = 14,
    kPackComplexToReal    = 15,
    kRGBToLab             = 16,
    kSpectralResidual     = 17,
    kSquare               = 18,
    kStandardDeviation    = 19,
    kUnpackRealToComplex  = 20,
    PROGRAM_COUNT         = 21
  };

  static std::string GetProgramString(Programs program);
  static std::string GetProgramName(Programs program);
};

}  // namespace vcsmc

#endif  // SRC_CL_PROGRAM_H_

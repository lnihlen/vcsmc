#ifndef SRC_FRAME_FITTER_H_
#define SRC_FRAME_FITTER_H_

#include <memory>
#include <vector>

#include "constants.h"
#include "types.h"

namespace vcsmc {

class Image;
class LineKernel;
class Random;
class State;

namespace op {
class OpCode;
}  // namespace op

class FrameFitter {
 public:
  FrameFitter();
  ~FrameFitter();

  float Fit(const uint8* half_colus);

  std::unique_ptr<Image> SimulateToImage();
  void SaveBinary(const char* file_name);

 private:
  void AppendFramePrefixOpCodes();
  void AppendFrameSuffixOpCodes(bool bank0);
  std::unique_ptr<LineKernel> FitLine(Random* random, const uint8* half_colus,
      uint32 scan_line, const State* entry_state);
  static bool CompareKernels(const std::unique_ptr<LineKernel>& lk1,
                             const std::unique_ptr<LineKernel>& lk2);

  std::vector<std::unique_ptr<State>> states_;
  std::vector<std::unique_ptr<std::vector<std::unique_ptr<op::OpCode>>>> banks_;
};


}  // namespace vcsmc

#endif  // SRC_FRAME_FITTER_H_

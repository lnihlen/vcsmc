#ifndef SRC_LINE_FITTER_H_
#define SRC_LINE_FITTER_H_

#include "constants.h"

#include <vector>

namespace vcsmc {

class LineKernel;
class Random;
class State;

namespace op {
class OpCode;
}  // namespace op

class LineFitter {
 public:
  LineFitter();
  ~LineFitter();

  // |half_colus| points to the whole frame of half color values, and
  // |scan_line| is within [0, kScreenHeight) but normally starts one or two
  // lines before the frame and runs to last line in the frame.
  float Fit(Random* random, const uint8* half_colus, uint32 scan_line,
      const State* entry_state);

  // After Fit() has been called, this will append the OpCodes from the best
  // fit, along with the resulting states, to the supplied vectors.
  void AppendBestFit(std::vector<std::unique_ptr<op::OpCode>>* opcodes,
                     std::vector<std::unique_ptr<State>>* states);

 private:
  // Simulates each LineKernel in the provided population to calculate error of
  // each. Returns the index of the LineKernel in |population| that has the
  // lowest error.
  uint32 SimulatePopulation(
      const std::vector<std::unique_ptr<LineKernel>>& population,
      const uint8* half_colus,
      uint32 scan_line,
      const State* entry_state);

  static bool CompareKernels(const std::unique_ptr<LineKernel>& lk1,
                             const std::unique_ptr<LineKernel>& lk2);

  std::unique_ptr<LineKernel> best_fit_;
};

}  // namespace vcsmc

#endif  // SRC_LINE_FITTER_H_

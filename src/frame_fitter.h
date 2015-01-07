#ifndef SRC_FRAME_FITTER_H_
#define SRC_FRAME_FITTER_H_

#include <memory>
#include <vector>

#include "constants.h"
#include "types.h"

namespace vcsmc {

struct Action;
class Image;
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

 private:
  void AppendFramePrefixOpCodes();

  // Assumes appends to |opcodes_| will occur at the start of the first scanline
  // after the end of the drawn region.
  void AppendFrameSuffixOpCodes();

  std::unique_ptr<Action> BuildAction(const Action* prior_action,
                                      std::unique_ptr<op::OpCode> opcode,
                                      const uint8* half_colus);
  bool ShouldSimulateTIA(TIA tia);
  std::unique_ptr<uint8[]> SimulateBack(const Action* last_action);

  std::vector<std::unique_ptr<op::OpCode>> opcodes_;
  std::vector<std::unique_ptr<State>> states_;
  std::unique_ptr<uint8[]> sim_;
  float error_;
};

}  // namespace vcsmc

#endif  // SRC_FRAME_FITTER_H_

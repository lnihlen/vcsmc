#ifndef SRC_KERNEL_H_
#define SRC_KERNEL_H_

#include <memory>
#include <vector>

#include "types.h"

namespace vcsmc {

class Frame;
class ScanLine;
class State;

// A Kernel, while somewhat an abuse of the term, represents a program to render
// an entire Frame of imagery. It owns a |target_frame_| from which it extracts
// ColuStrips and attempts various Strategies to fit ScanLines to those strips.
// It is responsible for stitching all of those ScanLines together into a
// coherent whole that can be assembled in to 6502 bytecode for running.
class Kernel {
 public:
  // Takes ownership of target_frame.
  Kernel(std::unique_ptr<Frame> target_frame);

  void Fit();

  // Save output.
  void Save();

 private:
  // utility method
  std::unique_ptr<State> EntryStateForLine(uint32 line);

  std::unique_ptr<Frame> target_frame_;
  std::vector<std::unique_ptr<ScanLine>> scan_lines_;
  std::unique_ptr<Frame> output_frame_;
};

}  // namespace vcsmc

#endif  // SRC_KERNEL_H_

#ifndef SRC_COLOR_H_
#define SRC_COLOR_H_

#include <vector>

#include "types.h"

namespace vcsmc {

class CLBuffer;

// Utilities for converting to/from Atari 2600 Color Codes and ABGR color words
class Color {
 public:
  static uint32 AtariColorToABGR(uint8 atari_color);
  // Returns a pointer to 4 floats L, a, b, 1.0
  static const float* AtariColorToLab(uint8 atari_color);

  // Builds or returns a previously built Atari color buffer of Lab color for
  // the provided color number. The buffer will be float4 * screen width in px.
  static const CLBuffer* AtariLabColorBuffer(uint8 atari_color);

  // Call after OpenCL is initialized, will create a queue, upload all the
  // color lines, and wait for the queue to comoplete before returning.
  static bool Setup();
  static void Teardown();

 private:
  static Color* instance_;
  bool CopyColorBuffers();
  vector<std::unique_ptr<CLBuffer>> atari_color_buffers_;
};

}  // namespace vcsmc

#endif  // SRC_COLOR_H_

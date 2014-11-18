#ifndef SRC_VIDEO_FRAME_DATA_H_
#define SRC_VIDEO_FRAME_DATA_H_

#include "types.h"

namespace vcsmc {

// Defines a container class for useful information about a given video frame.
// Typically constructed by a VideoFrameDataParser.
class VideoFrameData {
 public:
  VideoFrameData(uint64 frame_number, uint64 pts, bool is_keyframe);

  uint64 frame_number() { return frame_number_; }
  // Presentation Time Stamp, in color clocks.
  uint64 pts() { return pts_; }
  bool is_keyframe() { return is_keyframe_; }

 private:
  uint64 frame_number_;
  uint64 pts_;
  bool is_keyframe_;
};

}  // namespace vcsmc

#endif  // SRC_VIDEO_FRAME_DATA_H_

#include "video_frame_data.h"

namespace vcsmc {

VideoFrameData::VideoFrameData(uint64 frame_number, uint64 pts,
    bool is_keyframe)
    : frame_number_(frame_number),
      pts_(pts),
      is_keyframe_(is_keyframe) {
}

}  // namespace vcsmc

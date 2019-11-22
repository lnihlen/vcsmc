#ifndef SRC_VIDEO_DECODER_H_
#define SRC_VIDEO_DECODER_H_

#include "Halide.h"

#include <string>
#include <memory>

#include "types.h"

namespace vcsmc {

struct VideoFrame {
  int frame_number;
  bool is_keyframe;
  Halide::Runtime::Buffer<uint8_t, 3> frame_data;
  int64 frame_time_us;
};

class VideoDecoder {
 public:
  VideoDecoder();
  ~VideoDecoder();

  bool OpenFile(const std::string& file_name);
  std::shared_ptr<VideoFrame> GetNextFrame();
  bool AtEndOfFile() const;
  void CloseFile();


 private:
  // Use pImpl pattern to shield rest of the project from the libav headers.
  class VideoDecoderImpl;
  VideoDecoderImpl* p_;
};

}  // namespace vcsmc

#endif  // SRC_VIDEO_DECODER_H_

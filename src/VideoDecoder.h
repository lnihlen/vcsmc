#ifndef SRC_VIDEO_DECODER_H_
#define SRC_VIDEO_DECODER_H_

#include "flatbuffers/flatbuffers.h"

#include <string>

namespace vcsmc {

class VideoDecoder {
 public:
  VideoDecoder();
  ~VideoDecoder();

  bool OpenFile(const std::string& file_name);
  flatbuffers::DetachedBuffer GetNextFrame();
  bool AtEndOfFile() const;
  void CloseFile();

 private:
  // Use pImpl pattern to shield rest of the project from the libav headers.
  class VideoDecoderImpl;
  VideoDecoderImpl* p_;
};

}  // namespace vcsmc

#endif  // SRC_VIDEO_DECODER_H_

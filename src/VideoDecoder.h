#ifndef SRC_VIDEO_DECODER_H_
#define SRC_VIDEO_DECODER_H_

#include "flatbuffers/flatbuffers.h"

#include <string>

namespace leveldb {
    class DB;
}

namespace vcsmc {

class VideoDecoder {
 public:
  VideoDecoder(leveldb::DB* db);
  ~VideoDecoder();

  bool OpenFile(const std::string& file_name);
  bool DecodeNextFrame();
  bool SaveNextFrame();
  bool AtEndOfFile() const;
  void CloseFile();

 private:
  // Use pImpl pattern to shield rest of the project from the libav headers.
  class VideoDecoderImpl;
  VideoDecoderImpl* p_;
};

}  // namespace vcsmc

#endif  // SRC_VIDEO_DECODER_H_

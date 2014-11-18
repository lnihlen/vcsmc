#ifndef SRC_VIDEO_FRAME_DATA_PARSER
#define SRC_VIDEO_FRAME_DATA_PARSER

#include <memory>
#include <string>
#include <vector>

#include "types.h"

namespace vcsmc {

class VideoFrameData;

// Factory for VideoFrameData argument. Takes a .csv video frame data file as
// produced by ffmpeg and stream-parses it (as they can be several GB in size)
// to produce the individual VideoFrameData frames. Useful for extracting
// groups of frames organized by keyframe.
class VideoFrameDataParser {
 public:
  VideoFrameDataParser();
  ~VideoFrameDataParser();

  // Returns true if the parser was able to open the file. Call before trying
  // to extract any data with GetNextFrameSet().
  bool OpenCSVFile(const std::string& file_path);

  typedef std::vector<std::unique_ptr<VideoFrameData>> Frames;
  // Returns a vector of VideoFrameData objects starting with a keyframe and
  // continuing until the frame before the next keyframe, or nullptr if all
  // frames have been parsed from the file.
  std::unique_ptr<Frames> GetNextFrameSet();

 private:
  const size_t kFileBufferSize = 16384;

  // Populates |next_object_| with the next VideoFrameData object from the
  // filestream, or nullptr if no such object available. Returns true or false
  // based on success or failure of read.
  bool ReadNextObject();

  int input_fd_;
  std::unique_ptr<char[]> file_buffer_;
  uint32 bytes_read_;
  uint32 buffer_offset_;
  uint64 frame_number_;
  std::unique_ptr<VideoFrameData> next_object_;
};

}  // namespace vcsmc

#endif  // SRC_VIDEO_FRAME_DATA_PARSER

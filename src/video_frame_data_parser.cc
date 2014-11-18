#include "video_frame_data_parser.h"

#include <cassert>
#include <fcntl.h>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "constants.h"
#include "video_frame_data.h"

namespace vcsmc {

VideoFrameDataParser::VideoFrameDataParser()
    : input_fd_(-1),
      bytes_read_(0),
      buffer_offset_(0),
      frame_number_(1) {
}

VideoFrameDataParser::~VideoFrameDataParser() {
  if (input_fd_ >= 0)
    close(input_fd_);
}

bool VideoFrameDataParser::OpenCSVFile(const std::string& file_path) {
  input_fd_ = open(file_path.c_str(), O_RDONLY);
  if (input_fd_ < 0)
    return false;
  file_buffer_.reset(new char[kFileBufferSize]);
  bytes_read_ = read(input_fd_, file_buffer_.get(), kFileBufferSize);
  if (bytes_read_ == 0) {
    close(input_fd_);
    input_fd_ = -1;
    return false;
  }
  // Should be able to extract at least one VideoFrameData object from file.
  return ReadNextObject();
}

std::unique_ptr<std::vector<std::unique_ptr<VideoFrameData>>>
    VideoFrameDataParser::GetNextFrameSet() {
  if (!next_object_)
    return nullptr;

  std::unique_ptr<std::vector<std::unique_ptr<VideoFrameData>>> frame_set(
    new std::vector<std::unique_ptr<VideoFrameData>>);
  frame_set->push_back(std::move(next_object_));
  while (ReadNextObject()) {
    if (next_object_->is_keyframe())
      break;
    frame_set->push_back(std::move(next_object_));
  }

  return std::move(frame_set);
}

bool VideoFrameDataParser::ReadNextObject() {
  next_object_ = nullptr;
  if (input_fd_ < 0)
    return false;

  std::string csv_line;
  uint32 line_start = buffer_offset_;
  uint32 line_end = line_start;
  // Iterate through buffer looking for newline character.
  while (file_buffer_[line_end] != '\n') {
    ++line_end;
    // Have we reached the end of our buffer?
    if (line_end >= bytes_read_) {
      // Copy out the fragment of the line we already have.
      csv_line += std::string(file_buffer_.get() + line_start,
          bytes_read_ - line_start);
      bytes_read_ = read(input_fd_, file_buffer_.get(), kFileBufferSize);
      line_start = 0;
      line_end = 0;
      if (bytes_read_ == 0) {
        close(input_fd_);
        input_fd_ = -1;
        file_buffer_ = nullptr;
      }
    }
  }
  if (line_end - line_start > 0) {
    csv_line += std::string(file_buffer_.get() + line_start,
        line_end - line_start);
  }

  // Point buffer offset at the first character in the next line.
  buffer_offset_ = line_end + 1;
  if (buffer_offset_ >= bytes_read_) {
    if (input_fd_ >= 0) {
      bytes_read_ = read(input_fd_, file_buffer_.get(), kFileBufferSize);
      if (bytes_read_ == 0) {
        close(input_fd_);
        input_fd_ = -1;
        file_buffer_ = nullptr;
      }
    }
    buffer_offset_ = 0;
  }

  // OK, csv_line should have a line in it, otherwise we didn't get an object
  // out of the file and should return nullptr.
  if (!csv_line.size())
    return false;

  // ffmpeg CSV format seems to be:
  // "frame","video",is_keyframe,pts_in_framerate_ticks_int,
  // pts_in_seconds_float,...don't care
  // Line should start with "frame,video,"
  if (csv_line.substr(0, 12) != "frame,video,")
    return false;

  bool is_keyframe = csv_line[12] == '1';

  // Parse past the pts_in_framerate_ticks_int to the next comma.
  std::size_t pts_start = csv_line.find_first_of(",", 14);
  if (pts_start == std::string::npos)
    return false;

  // Increment past the starting comma and find next comma.
  ++pts_start;
  double pts_real = strtod(csv_line.c_str() + pts_start, nullptr);
  uint64 pts = static_cast<uint64>(pts_real * kClockRateHz);
  next_object_.reset(new VideoFrameData(frame_number_, pts, is_keyframe));
  ++frame_number_;
  return true;
}

}  // namespace vcsmc

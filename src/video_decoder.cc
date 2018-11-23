#include "video_decoder.h"

#include <deque>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#include "Halide.h"
#pragma clang diagnostic pop

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

#include "constants.h"

namespace vcsmc {

class VideoDecoder::VideoDecoderImpl {
 public:
  VideoDecoderImpl()
      : format_context_(nullptr),
        video_codec_(nullptr),
        video_stream_index_(-1),
        video_codec_context_(nullptr),
        video_frame_time_base_us_(-1),
        at_end_of_file_(false) {}

  ~VideoDecoderImpl() {}

  bool OpenFile(const std::string& file_name) {
    format_context_ = avformat_alloc_context();
    if (avformat_open_input(&format_context_, file_name.c_str(),
          nullptr, nullptr) != 0) {
      fprintf(stderr, "avformat_open_input failed.\n");
      return false;
    }

    if (avformat_find_stream_info(format_context_, NULL) < 0) {
      fprintf(stderr, "avformat_find_stream_info failed.\n");
      return false;
    }

    AVCodecParameters* video_codec_parameters = nullptr;

    // Iterate through all streams in file to find the first video stream.
    for (size_t i = 0; i < format_context_->nb_streams; ++i) {
      // Find the appropriate decoder for this stream, if available.
      AVCodecParameters* stream_parameters =
          format_context_->streams[i]->codecpar;
      AVCodec* codec = avcodec_find_decoder(stream_parameters->codec_id);

      // Skip streams with unsupported codecs for now.
      if (codec == nullptr) {
        fprintf(stderr, "Unsupported codec.\n");
        return false;
      }

      if (stream_parameters->codec_type == AVMEDIA_TYPE_VIDEO) {
        video_stream_index_ = i;
        video_codec_ = codec;
        video_codec_parameters = stream_parameters;
        break;
      }
    }

    // Need at least a video stream we can decode in order to continue.
    if (video_stream_index_ < 0) {
      fprintf(stderr, "Failed to find video stream.\n");
      return false;
    }

    video_codec_context_ = avcodec_alloc_context3(video_codec_);
    if (video_codec_context_ == nullptr) {
      fprintf(stderr, "Failed to allocated video codec context.\n");
      return false;
    }

    if (avcodec_parameters_to_context(video_codec_context_,
                                      video_codec_parameters) < 0) {
      fprintf(stderr, "Failed to copy parameters into video codec context.\n");
      return false;
    }

    if (avcodec_open2(video_codec_context_, video_codec_, nullptr) < 0) {
      fprintf(stderr, "Failed to open codec.\n");
      return false;
    }

    resize_context_ = sws_getContext(video_codec_context_->width,
                                     video_codec_context_->height,
                                     video_codec_context_->pix_fmt,
                                     kTargetFrameWidthPixels,
                                     kFrameHeightPixels,
                                     AV_PIX_FMT_GBRP,
                                     SWS_BICUBIC,
                                     nullptr,
                                     nullptr,
                                     nullptr);

    video_frame_time_base_us_ =
        (format_context_->streams[video_stream_index_]->time_base.num *
            1000 * 1000) /
        format_context_->streams[video_stream_index_]->time_base.den;

    return true;
  }

  std::unique_ptr<VideoFrame> GetNextFrame() {
    std::unique_ptr<VideoFrame> frame_image(nullptr);

    AVPacket packet;
    av_init_packet(&packet);
    AVFrame* frame = av_frame_alloc();

    bool frame_read = false;

    // Read frames until we encounter one from the identified video stream.
    int response = 0;
    while (!frame_read &&
           (response = av_read_frame(format_context_, &packet)) >= 0) {
      if (packet.stream_index == video_stream_index_) {
        int response = avcodec_send_packet(video_codec_context_, &packet);
        if (response < 0) {
          av_packet_unref(&packet);
          return frame_image;
        }

        response = avcodec_receive_frame(video_codec_context_, frame);
        if (response == AVERROR(EAGAIN)) {
          continue;
        } else if (response < 0) {
          fprintf(stderr, "Error decoding packet.\n");
          av_packet_unref(&packet);
          return frame_image;
        }

        Halide::Runtime::Buffer<uint8_t, 3> frame_rgb(
            kTargetFrameWidthPixels, kFrameHeightPixels, 3);
        // Note reverse order of plane pointers, in order to get proper RGB
        // plane alignment.
        uint8* plane_pointers[3] = {
          frame_rgb.begin() + kFrameSizeBytes,
          frame_rgb.begin() + (kFrameSizeBytes * 2),
          frame_rgb.begin()
        };
        int plane_widths[3] = {
          kTargetFrameWidthPixels,
          kTargetFrameWidthPixels,
          kTargetFrameWidthPixels
        };
        if (sws_scale(resize_context_,
                      frame->data,
                      frame->linesize,
                      0,
                      video_codec_context_->height,
                      plane_pointers,
                      plane_widths) <= 0) {
          fprintf(stderr, "Error scaling frame image.\n");
          av_frame_free(&frame);
          av_packet_unref(&packet);
          return frame_image;
        }

        frame_read = true;
        frame_image.reset(new VideoFrame());
        frame_image->frame_data = frame_rgb;
        frame_image->is_keyframe = (frame->key_frame == 1);
        frame_image->frame_number = video_codec_context_->frame_number;
        frame_image->frame_time_us = frame->pts * video_frame_time_base_us_;

        av_frame_free(&frame);
      }

      av_packet_unref(&packet);
    }

    if (response == AVERROR_EOF) {
      at_end_of_file_ = true;
    }

    return frame_image;
  }

  bool AtEndOfFile() const {
    return at_end_of_file_;
  }

  void CloseFile() {
    avformat_close_input(&format_context_);
    avformat_free_context(format_context_);
    format_context_ = nullptr;
    avcodec_free_context(&video_codec_context_);
    video_codec_context_ = nullptr;
    sws_freeContext(resize_context_);
    resize_context_ = nullptr;
  }

 private:
  AVFormatContext* format_context_;

  AVCodec* video_codec_;
  int video_stream_index_;
  AVCodecContext* video_codec_context_;
  SwsContext* resize_context_;
  int64 video_frame_time_base_us_;
  bool at_end_of_file_;
};

// static
void VideoDecoder::InitDecoderLibrary() {
  av_register_all();
}

VideoDecoder::VideoDecoder() : p_(new VideoDecoderImpl()) {}

VideoDecoder::~VideoDecoder() { delete p_; p_ = nullptr; }

bool VideoDecoder::OpenFile(const std::string& file_name) {
  return p_->OpenFile(file_name);
}

bool VideoDecoder::AtEndOfFile() const {
  return p_->AtEndOfFile();
}

void VideoDecoder::CloseFile() {
  return p_->CloseFile();
}

std::unique_ptr<VideoFrame> VideoDecoder::GetNextFrame() {
  return p_->GetNextFrame();
}

}  // namespace vcsmc


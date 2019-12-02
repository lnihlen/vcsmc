#include "VideoDecoder.h"

#include "constants.h"
#include "Logger.h"
#include "SourceFrame_generated.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

#include "xxhash.h"
#include "leveldb/db.h"

#include <queue>

namespace vcsmc {

class VideoDecoder::VideoDecoderImpl {
 public:
  VideoDecoderImpl(leveldb::DB* db)
      : format_context_(nullptr),
        video_codec_(nullptr),
        video_stream_index_(-1),
        video_codec_context_(nullptr),
        video_frame_time_base_us_(-1),
        at_end_of_file_(false),
        m_db(db) {}

  ~VideoDecoderImpl() {}

  bool OpenFile(const std::string& file_name) {
    format_context_ = avformat_alloc_context();
    if (avformat_open_input(&format_context_, file_name.c_str(), nullptr, nullptr) != 0) {
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

    video_frame_time_base_us_ = (format_context_->streams[video_stream_index_]->time_base.num * 1000 * 1000) /
        format_context_->streams[video_stream_index_]->time_base.den;

    return DecodeNextFrame();
  }

  // Decode a frame, save in the queue of decoded frames.
  bool DecodeNextFrame() {
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
          return false;
        }

        response = avcodec_receive_frame(video_codec_context_, frame);
        if (response == AVERROR(EAGAIN)) {
            continue;
        } else if (response < 0) {
            LOG_FATAL("error decoding packet");
            av_frame_free(&frame);
            av_packet_unref(&packet);
            return false;
        }

        std::unique_ptr<uint8_t[]> planes(new uint8_t[kFrameSizeBytes * 3]);

        // Note order of plane pointers for proper RGB plane alignment.
        uint8* plane_pointers[3] = {
          planes.get() + kFrameSizeBytes,
          planes.get() + (kFrameSizeBytes * 2),
          planes.get()
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
            LOG_FATAL("error scaling frame image");
            av_frame_free(&frame);
            av_packet_unref(&packet);
            return false;
        }

        frame_read = true;
        uint64_t frameHash = XXH64(planes.get(), kFrameSizeBytes * 3, 0);
        m_colorBytes.push(std::move(planes));

        flatbuffers::FlatBufferBuilder builder(1024);
        Data::SourceFrameBuilder frameBuilder(builder);
        frameBuilder.add_frameNumber(video_codec_context_->frame_number);
        frameBuilder.add_isKeyFrame(frame->key_frame == 1);
        frameBuilder.add_frameTime(frame->pts * video_frame_time_base_us_);
        frameBuilder.add_sourceHash(frameHash);
        auto frameData = frameBuilder.Finish();
        builder.Finish(frameData);
        m_sourceFrames.push(builder.Release());
        av_frame_free(&frame);
      }

      av_packet_unref(&packet);
    }

    if (response == AVERROR_EOF) {
      at_end_of_file_ = true;
      av_packet_unref(&packet);
    }

    return true;
  }

    bool SaveNextFrame() {
        if (m_colorBytes.size() == 0) {
            return true;
        }

        std::unique_ptr<uint8_t[]> planes = std::move(m_colorBytes.front());
        m_colorBytes.pop();
        flatbuffers::DetachedBuffer flatSource = std::move(m_sourceFrames.front());
        m_sourceFrames.pop();
        const Data::SourceFrame* sourceFrame = Data::GetSourceFrame(flatSource.data());
        leveldb::Status status;

        std::array<char, 32> buf;
        std::unique_ptr<leveldb::Iterator> it(m_db->NewIterator(leveldb::ReadOptions()));
        snprintf(buf.data(), sizeof(buf), "sourceImage:%016" PRIx64, sourceFrame->sourceHash());
        if (!it->Valid() || it->key().ToString() != buf.data()) {
            status = m_db->Put(leveldb::WriteOptions(), buf.data(), leveldb::Slice(reinterpret_cast<char*>(
                planes.get()), kFrameSizeBytes * 3));
            if (!status.ok()) {
                LOG_FATAL("error writing frame bytes into database");
                return false;
            }
        }

        snprintf(buf.data(), sizeof(buf), "sourceFrame:%08x", sourceFrame->frameNumber());
        status = m_db->Put(leveldb::WriteOptions(), buf.data(),
            leveldb::Slice(reinterpret_cast<const char*>(flatSource.data()), flatSource.size()));
        if (!status.ok()) {
            LOG_FATAL("error saving source frame %d into database.", sourceFrame->frameNumber());
            return false;
        }

        return true;
    }

  bool AtEndOfFile() const {
    return at_end_of_file_;
  }

  void CloseFile() {
    sws_freeContext(resize_context_);
    resize_context_ = nullptr;
    avformat_close_input(&format_context_);
    avformat_free_context(format_context_);
    format_context_ = nullptr;
    avcodec_free_context(&video_codec_context_);
    video_codec_context_ = nullptr;
  }

 private:
    AVFormatContext* format_context_;
    AVCodec* video_codec_;
    int video_stream_index_;
    AVCodecContext* video_codec_context_;
    SwsContext* resize_context_;
    int64 video_frame_time_base_us_;
    bool at_end_of_file_;
    leveldb::DB* m_db;
    std::queue<flatbuffers::DetachedBuffer> m_sourceFrames;
    std::queue<std::unique_ptr<uint8_t[]>> m_colorBytes;
};

VideoDecoder::VideoDecoder(leveldb::DB* db) : p_(new VideoDecoderImpl(db)) {}

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

bool VideoDecoder::DecodeNextFrame() {
    return p_->DecodeNextFrame();
}

bool VideoDecoder::SaveNextFrame() {
    return p_->SaveNextFrame();
}

}  // namespace vcsmc


#include "Workflow.h"

#include "atari_ntsc_lab_color_table.h"
#include "ciede_2k.h"
#include "constants.h"
#include "Logger.h"
#include "VideoDecoder.h"
#include "rgb_to_lab.h"

#include "TargetFrame_generated.h"

#include "Halide.h"
#include "leveldb/db.h"

#include <array>

namespace vcsmc {

Workflow::Workflow(leveldb::DB* database)
    : m_db(database),
      m_quit(false) {
}

Workflow::~Workflow() {
}

void Workflow::runThread() {
    m_future = std::async(std::launch::async, &Workflow::run, this);
}

void Workflow::shutdown() {
    m_quit = true;
    m_future.wait();
}

void Workflow::run() {
    enum State {
        kInitial,                // First state, does nothing.
        kOpenMovie,              // Open the video file, skipped if all frames already extracted.
        kDecodeFrame,            // Decode a new frame from the encoded movie file.
        kComputeTargetFrameLab,  // Convert the decoded frame to L*a*b* color, for measuring distances.
        kFitTargetFrameColors,   // Find closest average distance NTSC color numbers for each pair of input pixels in
                                 // a target frame.
        kCloseMovie,             // Close the input movie file, start decoding.
        kFinished,
        kTerminal
    };

    State state = kInitial;
    std::array<char, 32> keyBuf;
    leveldb::Status status;

    // kOpenMovie
    VideoDecoder decoder;
    bool decoderOpen = false;
    std::string moviePath;

    // kDecodeFrame
    std::shared_ptr<VideoFrame> videoFrame;

    // kComputeTargetFrameLab
    Halide::Runtime::Buffer<float, 3> frameLab(kTargetFrameWidthPixels, kFrameHeightPixels, 3);

    // kFitTargetFrameColors
    flatbuffers::FlatBufferBuilder builder((kTargetFrameWidthPixels * kFrameHeightPixels) + 1024);
    uint8_t* minIndices;
    Halide::Runtime::Buffer<float, 3> colorDistances(kTargetFrameWidthPixels, kFrameHeightPixels);
    std::array<float, kTargetFrameWidthPixels * kFrameHeightPixels> minDistances;

    while (!m_quit) {
        switch (state) {
            case kInitial:
                state = kOpenMovie;
                break;

            case kOpenMovie:
                status = m_db->Get(leveldb::ReadOptions(), "FLAGS_movie_path", &moviePath);
                if (!status.ok()) {
                    LOG_FATAL("unable to read movie path from database");
                    m_quit = true;
                    break;
                }

                LOG_INFO("opening movie file %s", moviePath.c_str());
                if (!decoder.OpenFile(moviePath)) {
                    LOG_FATAL("error opening movie at %s", moviePath.c_str());
                    m_quit = true;
                    break;
                }

                decoderOpen = true;
                state = kDecodeFrame;
                break;

            // Decode a video frame from compressed file to scaled output.
            case kDecodeFrame:
                LOG_INFO("extracting frame %d", videoFrame->frame_number);
                if (!decoderOpen) {
                    LOG_FATAL("ExtractFrames called with open decoder.");
                    m_quit = true;
                    break;
                }

                videoFrame = decoder.GetNextFrame();
                if (!videoFrame) {
                    LOG_INFO("eof encountered in media, closing");
                    state = kCloseMovie;
                    break;
                }

                state = kComputeTargetFrameLab;
                break;

            // Convert decoded RGB target frame to L*a*b* color.
            case kComputeTargetFrameLab:
                LOG_INFO("converting frame %d to L*a*b*", videoFrame->frame_number);
                if (!videoFrame) {
                    LOG_FATAL("computing target lab color on empty video frame");
                    m_quit = true;
                    break;
                }
                rgb_to_lab(videoFrame->frame_data, frameLab);
                state = kFitTargetFrameColors;
                break;

            case kFitTargetFrameColors:
                LOG_INFO("finding minimum error distance Atari colors for frame %d", videoFrame->frame_number);
                {
                    builder.Clear();
                    minIndices = nullptr;
                    auto indices = builder.CreateUninitializedVector(kTargetFrameWidthPixels * kFrameHeightPixels,
                        &minIndices);
                    std::memset(minIndices, 0, kTargetFrameWidthPixels * kFrameHeightPixels);
                    // Do zero color first to initialize distances and mins.
                    ciede_2k(frameLab, kAtariNtscLabLColorTable[0], kAtariNtscLabAColorTable[0],
                        kAtariNtscLabBColorTable[0], colorDistances);
                    std::memcpy(minDistances.data(), colorDistances.begin(), sizeof(minDistances));
                    for (auto i = 1; i < 128; ++i) {
                        ciede_2k(frameLab, kAtariNtscLabLColorTable[i], kAtariNtscLabAColorTable[i],
                            kAtariNtscLabBColorTable[i], colorDistances);
                        for (auto j = 0; j < kTargetFrameWidthPixels * kFrameHeightPixels; ++j) {
                            if (colorDistances.begin()[j] < minDistances[j]) {
                                minIndices[j] = i;
                            }
                        }
                    }
                    Data::TargetFrameBuilder frameBuilder(builder);
                    frameBuilder.add_frameNumber(videoFrame->frame_number);
                    frameBuilder.add_isKeyFrame(videoFrame->is_keyframe);
                    frameBuilder.add_frameTime(videoFrame->frame_time_us);
                    frameBuilder.add_image(indices);
                    auto frameData = frameBuilder.Finish();
                    builder.Finish(frameData);
                    snprintf(keyBuf.data(), sizeof(keyBuf), "targetFrame:%08x", videoFrame->frame_number);
                    status = m_db->Put(leveldb::WriteOptions(), keyBuf.data(),
                        leveldb::Slice(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize()));
                    if (!status.ok()) {
                        LOG_FATAL("error writing targetFrame %d to database", videoFrame->frame_number);
                        m_quit = true;
                        break;
                    }
                    state = kDecodeFrame;
                }
                break;

            case kCloseMovie:
                if (decoderOpen) {
                    decoder.CloseFile();
                    decoderOpen = false;
                }
                state = kFinished;
                break;

            case kFinished:
                LOG_INFO("state machine finished");
                m_quit = true;
                break;

            case kTerminal:
                LOG_FATAL("termainl state reached");
                m_quit = true;
                break;
        }
    }

    if (decoderOpen) {
        decoder.CloseFile();
        decoderOpen = false;
    }
}



}  // namespace vcsmc

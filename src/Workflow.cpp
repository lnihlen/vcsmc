#include "Workflow.h"

#include "atari_ntsc_lab_color_table.h"
#include "ciede_2k.h"
#include "constants.h"
#include "Logger.h"
#include "VideoDecoder.h"
#include "rgb_to_lab.h"

#include "SourceFrame_generated.h"
#include "TargetFrame_generated.h"

#include "Halide.h"
#include "leveldb/db.h"
#include "xxhash.h"

#include <array>
#include <inttypes.h>

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
    // More than one source frame can have the same sourceHash, the hash of the RGB bytes that describe the frame image.
    // Because of quantization, more quantized frames are likely to have the same hash. Desired output when we start
    // kernel fitting is frame groups and a list of quantized hash values that are within that frame group.

    // There's a flatbuffer called movieStats which contains the final information about the movie. It includes # of
    // frames, unique source frames, unique quantized frames, number of frame groups, total running time, movie path.
    // Presence of that indicates that the entire thing has been decoded and all decode steps should be skipped.

    // Decoding consists of extracting a frame from ffmpeg, hashing the RGB values, and saving the image bytes in the
    // database under that hash. We also save the frame metadata in the SourceFrame flatbuffer.

    // Quantization is the next phase. We iterate through all source images by hash, create quantized frames from those.
    // The presence of map entry from source frame to quantized frame indicates this step has been completed. Fit frames
    // are also saved in their own table.

    enum State {
        kInitial,                // First state, does nothing.
        kOpenMovie,              // Open the video file, skipped if all frames already extracted.
        kDecodeFrames,           // Decode a new frame from the encoded movie file, hash the resulting RGB values and
                                 // save image bytes in database under hash (if unique).
        kCloseMovie,             // Close the input movie file, as we no longer need it.
        kQuantizeFrames,         // Match input frame to best-fit Atari colors, save results.
        kBuildFrameGroups,       // Save the frame group data structures.
        kFinished,
        kTerminal
    };

    State state = kInitial;
    std::array<char, 32> buf;
    leveldb::Status status;

    // kOpenMovie
    VideoDecoder decoder(m_db);
    bool decoderOpen = false;
    std::string moviePath;

    // kQuantizeFrames
    std::unique_ptr<leveldb::Iterator> it;


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
                state = kDecodeFrames;
                break;

            // Decode a video frame from compressed file to scaled output and save in database.
            case kDecodeFrames:
                LOG_INFO("decoding next frame from movie file.");
                if (!decoderOpen) {
                    LOG_FATAL("ExtractFrames called with closed decoder.");
                    m_quit = true;
                    break;
                }

                if (!decoder.SaveNextFrame()) {
                    LOG_INFO("eof encountered in media, closing");
                    state = kCloseMovie;
                    break;
                }
                break;

            case kCloseMovie:
                if (decoderOpen) {
                    decoder.CloseFile();
                    decoderOpen = false;
                }
                it.reset(m_db->NewIterator(leveldb::ReadOptions()));
                it->Seek("sourceImage:0000000000000000");
                state = kQuantizeFrames;
                break;

            case kQuantizeFrames:
                if (!it->Valid() || it->key().ToString().substr(0, 12) != "sourceImage:") {
                    LOG_INFO("end of source frames for quantization.");
                    state = kBuildFrameGroups;
                } else {
                    // Extract hash of SourceImage from key:
                    std::string sourceHash = it->key().ToString().substr(12);
                    LOG_INFO("starting quantization of frame sourceHash %s", sourceHash.c_str());
                    // Check map for existing quantization entry, meaning we've already quantized this image.
                    std::unique_ptr<leveldb::Iterator> quantIt(m_db->NewIterator(leveldb::ReadOptions()));
                    std::string quantMapKey = "quantizeMap:" + sourceHash;
                    quantIt->Seek(quantMapKey);
                    if (quantIt->Valid() && quantIt->key().ToString() == quantMapKey) {
                        LOG_INFO("found existing quantizeMap entry for sourceHash %s => %s", sourceHash.c_str(),
                            quantIt->value().ToString().c_str());
                    } else {
                        // Load color planes into Halide input buffer for conversion to L*a*b* color.
                        Halide::Runtime::Buffer<uint8_t, 3> frameRGB(kTargetFrameWidthPixels, kFrameHeightPixels, 3);
                        std::memcpy(frameRGB.begin(), it->value().data(), kFrameSizeBytes * 3);
                        Halide::Runtime::Buffer<float, 3> frameLab(kTargetFrameWidthPixels, kFrameHeightPixels, 3);
                        rgb_to_lab(frameRGB, frameLab);
                        // Now compute distances for all pixels from each atari reference color.
                        Halide::Runtime::Buffer<float, 3> colorDistances(kTargetFrameWidthPixels, kFrameHeightPixels);
                        std::array<float, kFrameSizeBytes> minDistances;
                        std::array<uint8_t, kFrameSizeBytes> minIndices;
                        std::memset(minIndices.data(), 0, kFrameSizeBytes);
                        // Do zero color first to initialize distances and mins.
                        ciede_2k(frameLab, kAtariNtscLabLColorTable[0], kAtariNtscLabAColorTable[0],
                            kAtariNtscLabBColorTable[0], colorDistances);
                        std::memcpy(minDistances.data(), colorDistances.begin(), sizeof(minDistances));
                        for (auto i = 1; i < 128; ++i) {
                            ciede_2k(frameLab, kAtariNtscLabLColorTable[i], kAtariNtscLabAColorTable[i],
                                kAtariNtscLabBColorTable[i], colorDistances);
                            for (auto j = 0; j < kFrameSizeBytes; ++j) {
                                if (colorDistances.begin()[j] < minDistances[j]) {
                                    minIndices[j] = i;
                                }
                            }
                        }
                        uint64_t quantHash = XXH64(minIndices.data(), kFrameSizeBytes, 0);
                        // Save relationship between source and quantized hash in the map.
                        snprintf(buf.data(), sizeof(buf), "%016" PRIx64, quantHash);
                        LOG_INFO("computed new quantizeMap entry for sourceHash %s => %s", sourceHash.c_str(),
                            buf.data());
                        status = m_db->Put(leveldb::WriteOptions(), quantMapKey, buf.data());
                        std::string quantKey = std::string("quantImage:") + buf.data();
                        // Check for the quantized data already stored in database.
                        quantIt->Seek(quantKey);
                        if (!quantIt->Valid() || quantIt->key().ToString() != quantKey) {
                            m_db->Put(leveldb::WriteOptions(), quantKey, leveldb::Slice(reinterpret_cast<const char*>(
                                minIndices.data()), kFrameSizeBytes));
                        }
                    }
                    it->Next();
                }
                break;

            case kBuildFrameGroups:
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

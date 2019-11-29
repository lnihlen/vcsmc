#include "Workflow.h"

#include "atari_ntsc_lab_color_table.h"
#include "ciede_2k.h"
#include "constants.h"
#include "Logger.h"
#include "VideoDecoder.h"
#include "rgb_to_lab.h"

#include "FrameGroup_generated.h"
#include "SourceFrame_generated.h"

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
    enum State {
        kInitial,                // First state, does nothing.
        kOpenMovie,              // Open the video file, skipped if all frames already extracted.
        kDecodeFrames,           // Decode a new frame from the encoded movie file, hash the resulting RGB values and
                                 // save image bytes in database under hash (if unique).
        kCloseMovie,             // Close the input movie file, as we no longer need it.
        kQuantizeFrames,         // Match input frame to best-fit Atari colors, save results.
        kGroupFrames,            // Save the frame group data structures.
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

    // kGroupFrames
    std::unordered_set<uint64_t> groupImages;
    int groupStartFrame = 1;
    int groupNumber = 0;
    int lastFrameNumber = 0;

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
                    LOG_INFO("end of source images for quantization.");
                    it->Seek("sourceFrame:00000000");
                    groupImages.clear();
                    groupStartFrame = 0;
                    groupNumber = 0;
                    state = kGroupFrames;
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

            case kGroupFrames:
                if (!it->Valid() || it->key().ToString().substr(0, 12) != "sourceFrame:") {
                    // Save final frame group.
                    ++groupNumber;
                    saveFrameGroup(groupImages, groupStartFrame, lastFrameNumber, groupNumber);
                    LOG_INFO("wrote final group %d to database with %d unique frames", groupNumber, groupImages.size());
                    groupImages.clear();
                    it.reset(nullptr);
                    state = kFinished;
                } else {
                    const Data::SourceFrame* sourceFrame = Data::GetSourceFrame(it->value().data());
                    if (!sourceFrame) {
                        LOG_FATAL("error deserializing sourceFrame from database.");
                        state = kTerminal;
                        break;
                    }
                    if (sourceFrame->isKeyFrame() && lastFrameNumber > 0) {
                        ++groupNumber;
                        saveFrameGroup(groupImages, groupStartFrame, lastFrameNumber, groupNumber);
                        LOG_INFO("wrote group %d to database with %d unique frames.", groupNumber, groupImages.size());
                        // Reset for next frame group.
                        groupImages.clear();
                        groupStartFrame = sourceFrame->frameNumber();
                    }
                    // Look up quantized image based on source image hash.
                    snprintf(buf.data(), sizeof(buf), "quantizeMap:%016" PRIx64, sourceFrame->sourceHash());
                    std::string mapKey(buf.data());
                    std::unique_ptr<leveldb::Iterator> mapIt(m_db->NewIterator(leveldb::ReadOptions()));
                    mapIt->Seek(mapKey);
                    if (!mapIt->Valid() || mapIt->key().ToString() != mapKey) {
                        LOG_FATAL("error looking up quantized image map value for source hash key %s", mapKey.c_str());
                        state = kTerminal;
                        break;
                    }
                    uint64_t quantHash = strtoull(mapIt->value().data(), nullptr, 16);
                    if (quantHash == 0) {
                        LOG_FATAL("error parsing quant key %s to uint64_t", mapIt->value());
                        state = kTerminal;
                        break;
                    }
                    groupImages.insert(quantHash);
                    lastFrameNumber = sourceFrame->frameNumber();
                    it->Next();
                }
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

bool Workflow::saveFrameGroup(const std::unordered_set<uint64_t>& groupImages, int groupStartFrame, int lastFrameNumber,
        int groupNumber) {
    // Save current group in database.
    flatbuffers::FlatBufferBuilder builder(1024);
    uint64_t* buildGroupImages = nullptr;
    auto hashes = builder.CreateUninitializedVector(groupImages.size(), &buildGroupImages);
    for (uint64_t image : groupImages) {
        *buildGroupImages = image;
        ++buildGroupImages;
    }
    Data::FrameGroupBuilder groupBuilder(builder);
    groupBuilder.add_firstFrame(groupStartFrame);
    groupBuilder.add_lastFrame(lastFrameNumber);
    groupBuilder.add_imageHashes(hashes);
    auto group = groupBuilder.Finish();
    builder.Finish(group);
    std::array<char, 32> buf;
    snprintf(buf.data(), sizeof(buf), "frameGroup:%08x", groupNumber);
    std::string groupKey(buf.data());
    leveldb::Status status = m_db->Put(leveldb::WriteOptions(), groupKey, leveldb::Slice(reinterpret_cast<const char*>(
        builder.GetBufferPointer()), builder.GetSize()));
    return status.ok();
}

}  // namespace vcsmc

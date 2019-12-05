#include "Task.h"

#include "Logger.h"
#include "atari_ntsc_lab_color_table.h"
#include "ciede_2k.h"
#include "constants.h"
#include "rgb_to_lab.h"
#include "VideoDecoder.h"

#include "FrameGroup_generated.h"
#include "GroupStats_generated.h"
#include "MovieStats_generated.h"
#include "SourceFrame_generated.h"

#include "Halide.h"
#include "leveldb/db.h"
#include "xxhash.h"

#include <array>
#include <inttypes.h>
#include <random>

namespace vcsmc {

// ===== Task
Task::Task(leveldb::DB* db) :
    m_db(db) {
}

Task::~Task() {
}

// static
std::unique_ptr<Task> Task::taskForType(Task::Type type, leveldb::DB* db) {
    std::unique_ptr<Task> task;
    switch (type) {
        case kInitial:
            break;

        case kDecodeFrames:
            task.reset(new DecodeFrames(db));
            break;

        case kQuantizeFrames:
            task.reset(new QuantizeFrames(db));
            break;

        case kGroupFrames:
            task.reset(new GroupFrames(db));
            break;

        case kGeneratePopulation:
            task.reset(new GeneratePopulation(db));
            break;

        case kFinished:
        case kFatal:
            break;
    }

    return task;
}

// ===== DecodeFrames
DecodeFrames::DecodeFrames(leveldb::DB* db) :
    Task(db) {
}

DecodeFrames::~DecodeFrames() {
}

const char* DecodeFrames::name() {
    return "DecodeFrames";
}

Task::Type DecodeFrames::setup() {
    leveldb::Status status = m_db->Get(leveldb::ReadOptions(), "FLAGS_movie_path", &m_moviePath);
    if (!status.ok()) {
        LOG_FATAL("unable to read movie path from database");
        return Type::kFatal;
    }

    // Check for an existing MovieStats object, which indicates the movie has already been decoded.
    std::unique_ptr<leveldb::Iterator> it(m_db->NewIterator(leveldb::ReadOptions()));
    it->Seek("movieStats");
    if (it->Valid() && it->key().ToString() == "movieStats") {
        // Pull record to compare path.
        const Data::MovieStats* movieStats = Data::GetMovieStats(it->value().data());
        if (!movieStats) {
            LOG_FATAL("error deserializing MovieSstats flatbuffer");
            return Type::kFatal;
        }
        if (flatbuffers::GetString(movieStats->moviePath()) != m_moviePath) {
            LOG_WARN("already  decoded movie path %s differs from stored command-line argument path %s.",
                flatbuffers::GetCstring(movieStats->moviePath()), m_moviePath.c_str());
        }
        LOG_INFO("MovieStats structure found, skipping decode step.");
        return Type::kQuantizeFrames;
    }

    m_decoder.reset(new VideoDecoder(m_db));
    LOG_INFO("opening movie file at %s for decoding.", m_moviePath.data());
    if (!m_decoder->OpenFile(m_moviePath)) {
        LOG_FATAL("error opening movie at %s", m_moviePath.data());
        return Type::kFatal;
    }

    return Type::kDecodeFrames;
}

Task::Type DecodeFrames::load() {
    if (m_decoder->AtEndOfFile()) {
        // Save MovieStats structure to database, so we'll skip the decode step in the future.
        flatbuffers::FlatBufferBuilder builder(1024);
        auto moviePath = builder.CreateString(m_moviePath);
        Data::MovieStatsBuilder statsBuilder(builder);
        statsBuilder.add_moviePath(moviePath);
        statsBuilder.add_totalFrames(0);
        statsBuilder.add_reportedDuration(0);
        auto stats = statsBuilder.Finish();
        builder.Finish(stats);
        leveldb::Status status = m_db->Put(leveldb::WriteOptions(), "movieStats",
            leveldb::Slice(reinterpret_cast<const char*>(builder.GetBufferPointer(), builder.GetSize())));
        if (!status.ok()) {
            LOG_FATAL("error saving movieStats key to database.");
            return Type::kFatal;
        }
        return Type::kQuantizeFrames;
    }
    return Type::kDecodeFrames;
}

bool DecodeFrames::execute() {
    return m_decoder->DecodeNextFrame();
}

bool DecodeFrames::store() {
    return m_decoder->SaveNextFrame();
}

bool DecodeFrames::teardown() {
    m_decoder->CloseFile();
    m_decoder.reset(nullptr);
    return true;
}

// ===== QuantizeFrames
QuantizeFrames::QuantizeFrames(leveldb::DB* db) :
    Task(db),
    m_frameRGB(kTargetFrameWidthPixels, kFrameHeightPixels, 3),
    m_frameLab(kTargetFrameWidthPixels, kFrameHeightPixels, 3),
    m_colorDistances(kTargetFrameWidthPixels, kFrameHeightPixels) {
}

QuantizeFrames::~QuantizeFrames() {
}

const char* QuantizeFrames::name() {
    return "QuantizeFrames";
}

Task::Type QuantizeFrames::setup() {
    m_it.reset(m_db->NewIterator(leveldb::ReadOptions()));
    m_it->Seek("sourceImage:0000000000000000");
    if (!m_it->Valid() || m_it->key().ToString().substr(0, 12) != "sourceImage:") {
        return Type::kFatal;
    }
    return Type::kQuantizeFrames;
}

Task::Type QuantizeFrames::load() {
    bool keepGoing = true;
    // Advance through sourceImage keys until we either find one that doesn't have an already quantized image or we
    // reach the end of the keys.
    while (keepGoing) {
        if (!m_it->Valid() || m_it->key().ToString().substr(0, 12) != "sourceImage:") {
            return Type::kGroupFrames;
        }
        m_sourceHash = m_it->key().ToString().substr(12);
        // Check map for existing quantization entry, meaning we've already quantized this image.
        std::unique_ptr<leveldb::Iterator> quantIt(m_db->NewIterator(leveldb::ReadOptions()));
        std::string quantMapKey = "quantizeMap:" + m_sourceHash;
        quantIt->Seek(quantMapKey);
        if (!quantIt->Valid() || quantIt->key().ToString() != quantMapKey) {
            keepGoing = false;
        } else {
            m_it->Next();
        }
    }

    std::memcpy(m_frameRGB.begin(), m_it->value().data(), kFrameSizeBytes * 3);
    return Type::kQuantizeFrames;
}

bool QuantizeFrames::execute() {
    rgb_to_lab(m_frameRGB, m_frameLab);
    std::memset(m_minIndices.data(), 0, kFrameSizeBytes);
    // Do zero color first to initialize distances and mins.
    ciede_2k(m_frameLab, kAtariNtscLabLColorTable[0], kAtariNtscLabAColorTable[0],
            kAtariNtscLabBColorTable[0], m_colorDistances);
    std::memcpy(m_minDistances.data(), m_colorDistances.begin(), sizeof(m_minDistances));
    for (auto i = 1; i < 128; ++i) {
        ciede_2k(m_frameLab, kAtariNtscLabLColorTable[i], kAtariNtscLabAColorTable[i], kAtariNtscLabBColorTable[i],
            m_colorDistances);
        for (auto j = 0; j < kFrameSizeBytes; ++j) {
            if (m_colorDistances.begin()[j] < m_minDistances[j]) {
                m_minIndices[j] = i;
            }
        }
    }

    return true;
}

bool QuantizeFrames::store() {
    uint64_t quantHash = XXH64(m_minIndices.data(), kFrameSizeBytes, 0);
    std::array<char, 32> buf;
    // Save relationship between source and quantized hash in the map.
    snprintf(buf.data(), sizeof(buf), "%016" PRIx64, quantHash);
    std::string quantMapKey = "quantizeMap:" + m_sourceHash;
    leveldb::Status status = m_db->Put(leveldb::WriteOptions(), quantMapKey, buf.data());
    if (!status.ok()) {
        LOG_FATAL("error saving quantMap key %016" PRIx64 " to database", quantHash);
        return false;
    }

    std::string quantKey = std::string("quantImage:") + buf.data();
    // Check for the quantized data already stored in database.
    std::unique_ptr<leveldb::Iterator> quantIt(m_db->NewIterator(leveldb::ReadOptions()));
    quantIt->Seek(quantKey);
    if (!quantIt->Valid() || quantIt->key().ToString() != quantKey) {
        status = m_db->Put(leveldb::WriteOptions(), quantKey, leveldb::Slice(reinterpret_cast<const char*>(
            m_minIndices.data()), kFrameSizeBytes));
        if (!status.ok()) {
            LOG_FATAL("error saving quant image %s to database", quantKey.data());
            return false;
        }
    }

    m_it->Next();
    return true;
}

bool QuantizeFrames::teardown() {
    return true;
}

void QuantizeFrames::setupBenchmark() {
    Halide::Runtime::Buffer<uint8_t, 3> frameRGB(vcsmc::kTargetFrameWidthPixels, vcsmc::kFrameHeightPixels, 3);
    std::random_device randomDevice;
    std::default_random_engine randomEngine(randomDevice());
    std::uniform_int_distribution<uint8_t> distribution(0, 255);

    for (auto i = 0; i < vcsmc::kFrameSizeBytes * 3; ++i) {
        frameRGB.begin()[i] = distribution(randomEngine);
    }
}

// ===== kGroupFrames
GroupFrames::GroupFrames(leveldb::DB* db) :
    Task(db) {
}

GroupFrames::~GroupFrames() {
}

const char* GroupFrames::name() {
    return "GroupFrames";
}

Task::Type GroupFrames::setup() {
    m_groupStartFrame = 1;
    m_groupNumber = 0;
    m_lastFrameNumber = 0;
    m_it.reset(m_db->NewIterator(leveldb::ReadOptions()));
    m_it->Seek("sourceFrame:00000000");
    if (!m_it->Valid() || m_it->key().ToString().substr(0, 12) != "sourceFrame:") {
        return Type::kFatal;
    }
    return Type::kGroupFrames;
}

Task::Type GroupFrames::load() {
    if (!m_it->Valid() || m_it->key().ToString().substr(0, 12) != "sourceFrame:") {
        bool lastSaved = saveFrameGroup();
        if (!lastSaved) {
            return Type::kFatal;
        }
        return Type::kFinished; // FIXME
    }

    m_sourceFrame = Data::GetSourceFrame(m_it->value().data());
    if (!m_sourceFrame) {
        LOG_FATAL("error deserializing sourceFrame key %s from database.", m_it->key().data());
        return Type::kFatal;
    }

    return Type::kGroupFrames;
}

bool GroupFrames::execute() {
    if (m_sourceFrame->isKeyFrame() && m_lastFrameNumber > 0) {
        saveFrameGroup();
        // Reset for next frame group.
        m_groupImages.clear();
        m_groupStartFrame = m_sourceFrame->frameNumber();
    }
    // Look up quantized image based on source image hash.
    std::array<char, 32> buf;
    snprintf(buf.data(), sizeof(buf), "quantizeMap:%016" PRIx64, m_sourceFrame->sourceHash());
    std::string mapKey(buf.data());
    std::unique_ptr<leveldb::Iterator> mapIt(m_db->NewIterator(leveldb::ReadOptions()));
    mapIt->Seek(mapKey);
    if (!mapIt->Valid() || mapIt->key().ToString() != mapKey) {
        LOG_FATAL("error looking up quantized image map value for source hash key %s", mapKey.c_str());
        return false;
    }
    uint64_t quantHash = strtoull(mapIt->value().data(), nullptr, 16);
    if (quantHash == 0) {
        LOG_FATAL("error parsing quant key %s to uint64_t", mapIt->value());
        return false;
    }
    m_groupImages.insert(quantHash);
    m_lastFrameNumber = m_sourceFrame->frameNumber();
    return true;
}

bool GroupFrames::store() {
    m_it->Next();
    return true;
}

bool GroupFrames::teardown() {
    return true;
}

bool GroupFrames::saveFrameGroup() {
    ++m_groupNumber;

    // Save current group in database.
    flatbuffers::FlatBufferBuilder builder(1024);
    uint64_t* buildGroupImages = nullptr;
    auto hashes = builder.CreateUninitializedVector(m_groupImages.size(), &buildGroupImages);
    for (uint64_t image : m_groupImages) {
        *buildGroupImages = image;
        ++buildGroupImages;
    }
    Data::FrameGroupBuilder groupBuilder(builder);
    groupBuilder.add_firstFrame(m_groupStartFrame);
    groupBuilder.add_lastFrame(m_lastFrameNumber);
    groupBuilder.add_imageHashes(hashes);
    auto group = groupBuilder.Finish();
    builder.Finish(group);
    std::array<char, 32> buf;
    snprintf(buf.data(), sizeof(buf), "frameGroup:%08x", m_groupNumber);
    std::string groupKey(buf.data());
    leveldb::Status status = m_db->Put(leveldb::WriteOptions(), groupKey, leveldb::Slice(reinterpret_cast<const char*>(
        builder.GetBufferPointer()), builder.GetSize()));
    return status.ok();
}


// ===== kGeneratePopulation
GeneratePopulation::GeneratePopulation(leveldb::DB* db) :
    Task(db),
    m_groupNumber(0),
    m_frameGroup(nullptr),
    m_allGroupsFinished(false) {
}

GeneratePopulation::~GeneratePopulation() {
}

const char* GeneratePopulation::name() {
    return "GeneratePopulation";
}

// (++) Identify new frame group to fit against.
// Pick parameters for kernel generation. They are a fixed number of kernels per image (say 10k), but a max cap of 1M
// kernels, so divide down in that case.
// Generate an initial population of kernels for all frames.
// Now the main evaluation loop:
// (**) For any unscored kernels in the generation:
//      sequence (per-frame, could consider per frame group)
//      simulate
//      score (per-frame).
//  Now everyone should have a score.
//  Conduct frame-only tournament. Sort frames by victories.
//  Check if all frames have met score criteria or if we've hit generation cap, if so go to (++)
//  Generating next generation:
//      Sort per-frame for winners and select survivors for next generation.
//  Some kind of cross-frame breeding of winners
//  Fill balance of generation with new codons which are offspring of the winners.
//  Go back to (**)

// Each major step should have a way of determining if it should be skipped. So DecodeFrames (should rename to DecodeVideo)
// should be able to check that the video has already been decoded, and if so advance to the next state. (and not be
// counted in the time durations). Maybe a function Task::Type checkForPreviousCompletion() or something? Instead - 
// setup() can also return a state change.
//
// Then for individual iterations we can also check. There should be a way to ignore these Durations too, probably with
// an additional function to Timer that says discard.

Task::Type GeneratePopulation::setup() {
    // TODO: move much of this to load(), I think
    //
    // Iterate through FrameGroups keys to find first unfinished or absent GroupStats key.
    m_groups.reset(m_db->NewIterator(leveldb::ReadOptions()));
    m_groups->Seek("frameGroup:00000000");
    auto isFrameGroup = [](auto& it) {
        return it->Valid() && it->key().ToString().substr(0, 11) == "frameGroup:";
    };
    if (!isFrameGroup(m_groups)) {
        LOG_FATAL("GeneratePopulation unable to find frameGroup keys.");
        return Type::kFatal;
    }

    std::unique_ptr<leveldb::Iterator> stats(m_db->NewIterator(leveldb::ReadOptions()));

    while (isFrameGroup(m_groups)) {
        std::string statsKey = "stats:" + m_groups->key().ToString().substr(11);
        stats->Seek(statsKey);
        if (stats->Valid() && stats->key().ToString() == statsKey) {
            const Data::GroupStats* groupStats = Data::GetGroupStats(stats->value().data());
            if (!groupStats) {
                LOG_FATAL("error deserializing GroupStas record for key %s", statsKey.data());
                return Type::kFatal;
            }
            if (!groupStats->isFinished()) {
                // This means we should transition to one of the intermediate states? Like GeneratePopulation is only
                // for completely new situations?
                break;
            }
        } else {
            break;
        }
        m_groups->Next();
    }

    if (isFrameGroup(m_groups)) {
        m_groupNumber = strtol(m_groups->key().data() + 11, nullptr, 16);
        m_frameGroup = Data::GetFrameGroup(m_groups->value().data());
    } else {
        LOG_INFO("marking all groups as finished.");
        m_allGroupsFinished = true;
    }

    return Type::kGeneratePopulation;
}


Task::Type GeneratePopulation::load() {
    if (m_allGroupsFinished) {
        return Type::kFinished;
    }

    // We know what frame group we're working with. It is assumed we are generating the first generation only, and
    // creating the table structures to do fitting for a new framegroup. 
    return Type::kGeneratePopulation;
}

bool GeneratePopulation::execute() {
    return false;
}

bool GeneratePopulation::store() {
    return false;
}

bool GeneratePopulation::teardown() {
    return false;
}

}  // namespace vcsmc


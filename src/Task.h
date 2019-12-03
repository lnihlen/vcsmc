#ifndef SRC_TASK_H_
#define SRC_TASK_H_

#include "constants.h"

#include "Halide.h"

#include <array>
#include <memory>
#include <string>
#include <unordered_set>

namespace leveldb {
    class DB;
    class Iterator;
}

namespace vcsmc {

class VideoDecoder;
namespace Data {
    class SourceFrame;
};

// A Task represents a unit of work in the overall fitting state machine. It provides serialization into the database,
// structured logging, and inclusion in benchmarks and time tracking.
class Task {
public:
    // Manually kept in sync with the chart javascript in html/index.html.
    enum Type : int32_t {
        kInitial = 0,
        kDecodeFrames = 1,
        kQuantizeFrames = 2,
        kGroupFrames = 3,
        kFinished = 4,
        kFatal = 5
    };

    // Given an enumerated type, returns the associated Task pointer.
    static std::unique_ptr<Task> taskForType(Task::Type type, leveldb::DB* db);

    Task(leveldb::DB* db);
    virtual ~Task();

    // Return a string with the name of the task.
    virtual const char* name() = 0;

    // Called first, perform any one-time setup functions.
    virtual bool setup() = 0;

    // Called iteratively in order load(), execute(), store(), until a different Task::Type than the one currently
    // running is returned by load(). Sets up whatever internal buffers are needed for processing during the benchmarked
    // function execute().
    virtual Task::Type load() = 0;

    // This is the function that gets timed in benchmarks as well as aggregate runtime timer saving.
    virtual bool execute() = 0;

    // Save whatever work was completed in load() to the database. Called separately to not be included in the
    // benchmarking.
    virtual bool store() = 0;

    // Cleanup function, called after load() returns an exit state.
    virtual bool teardown() = 0;

protected:
    leveldb::DB* m_db;
};


class DecodeFrames : public Task {
public:
    DecodeFrames(leveldb::DB* db);
    virtual ~DecodeFrames();

    const char* name() override;
    bool setup() override;
    Task::Type load() override;
    bool execute() override;
    bool store() override;
    bool teardown() override;

protected:
    std::unique_ptr<VideoDecoder> m_decoder;
};

class QuantizeFrames : public Task {
public:
    QuantizeFrames(leveldb::DB* db);
    virtual ~QuantizeFrames();

    const char* name() override;
    bool setup() override;
    Task::Type load() override;
    bool execute() override;
    bool store() override;
    bool teardown() override;

    // Set up member variables with test data so we can benchmark the execute() step.
    void setupBenchmark();

protected:
    std::unique_ptr<leveldb::Iterator> m_it;
    std::string m_sourceHash;
    Halide::Runtime::Buffer<uint8_t, 3> m_frameRGB;
    Halide::Runtime::Buffer<float, 3> m_frameLab;
    Halide::Runtime::Buffer<float, 2> m_colorDistances;
    std::array<float, kFrameSizeBytes> m_minDistances;
    std::array<uint8_t, kFrameSizeBytes> m_minIndices;
};


class GroupFrames : public Task {
public:
    GroupFrames(leveldb::DB* db);
    virtual ~GroupFrames();

    const char* name() override;
    bool setup() override;
    Task::Type load() override;
    bool execute() override;
    bool store() override;
    bool teardown() override;

protected:
    bool saveFrameGroup();

    std::unique_ptr<leveldb::Iterator> m_it;
    std::unordered_set<uint64_t> m_groupImages;
    int m_groupStartFrame;
    int m_groupNumber;
    int m_lastFrameNumber;
    const Data::SourceFrame* m_sourceFrame;
};

}  // namespace vcsmc

#endif  // SRC_TASK_H_


#include "Timer.h"

#include "Duration_generated.h"

#include "leveldb/db.h"

#include <chrono>
#include <inttypes.h>

namespace vcsmc {

Timer::Timer(leveldb::DB* db, Task::Type type) :
    m_db(db),
    m_type(type),
    m_discard(false) {
    m_startTime = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

Timer::~Timer() {
    if (!m_discard) {
        uint64_t stopTime = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        flatbuffers::FlatBufferBuilder builder(1024);
        Data::DurationBuilder durationBuilder(builder);
        durationBuilder.add_type(m_type);
        durationBuilder.add_startTime(m_startTime);
        durationBuilder.add_duration(stopTime - m_startTime);
        auto duration = durationBuilder.Finish();
        builder.Finish(duration);
        std::array<char, 32> buf;
        snprintf(buf.data(), sizeof(buf), "duration:%016" PRIx64, m_startTime);
        m_db->Put(leveldb::WriteOptions(), std::string(buf.data()), leveldb::Slice(
            reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize()));
    }
}

}  // namespace vcsmc


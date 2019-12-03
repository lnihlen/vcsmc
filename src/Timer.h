#ifndef SRC_TIMER_H_
#define SRC_TIMER_H_

#include "Task.h"

namespace leveldb {
    class DB;
}

namespace vcsmc {

// A Timer instance tracks a Task (based on the enumerated type) and then, on destruction, logs the time taken into the
// database, for aggregate logging later.
class Timer {
public:
    Timer(leveldb::DB* db, Task::Type type);
    ~Timer();

private:
    leveldb::DB* m_db;
    Task::Type m_type;
    uint64_t m_startTime;
};

}  // namespace vcsmc

#endif  // SRC_TIMER_H_


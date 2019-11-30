#ifndef SRC_WORKFLOW_H_
#define SRC_WORKFLOW_H_

#include <future>
#include <unordered_set>

namespace leveldb {
class DB;
}

namespace vcsmc {

// Class to represent the resumable state of the vcsmc main program control flow. Builds a series of table entries that
// represent top-level actions to take on the way to generating a complete fit program at the end.
class Workflow {
public:
    Workflow(leveldb::DB* database);
    ~Workflow();

    void runThread();
    void shutdown();

private:
    void run();

    bool saveFrameGroup(const std::unordered_set<uint64_t>& groupImages, int groupStartFrame, int lastFrameNumber,
        int groupNumber);

    leveldb::DB* m_db;
    bool m_quit;
    std::future<void> m_future;
};

}  // namespace vcsmc

#endif  // SRC_WORKFLOW_H_

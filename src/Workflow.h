#ifndef SRC_WORKFLOW_H_
#define SRC_WORKFLOW_H_

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

    void setup();
    void runThread();

private:
    class Task;

    leveldb::DB* m_db;
};

}  // namespace vcsmc

#endif  // SRC_WORKFLOW_H_

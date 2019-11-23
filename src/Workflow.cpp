#include "Workflow.h"

namespace vcsmc {

class Workflow::Task {
public:
protected:
};

Workflow::Workflow(leveldb::DB* database)
    : m_db(database) {
}

Workflow::~Workflow() {
}

void Workflow::runThread() {
}

}  // namespace vcsmc

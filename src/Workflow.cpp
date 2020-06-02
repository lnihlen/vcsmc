#include "Workflow.h"

#include "Logger.h"
#include "Task.h"
#include "Timer.h"

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
    Task::Type type = Task::Type::kDecodeFrames;
    std::unique_ptr<Task> task = Task::taskForType(type, m_db);

    while (!m_quit) {
        LOG_INFO("starting on task %s", task->name());

        bool skipTeardown = false;
        Task::Type nextType = task->setup();
        if (nextType != type) {
            skipTeardown = true;
        }

        while (!m_quit && nextType == type) {
            Timer timer(m_db, type);

            nextType = task->load();
            if (nextType != type) {
                timer.setDiscard(true);
                break;
            }

            if (!task->execute()) {
                LOG_FATAL("error on execution of task %s", task->name());
                m_quit = true;
                timer.setDiscard(true);
                break;
            }

            if (!task->store()) {
                LOG_FATAL("error on storage of task %s", task->name());
                timer.setDiscard(true);
                m_quit = true;
                break;
            }
        }

        if (!skipTeardown) {
            if (!task->teardown()) {
                LOG_FATAL("error tearing down task %s", task->name());
                m_quit = true;
                break;
            }
        }

        type = nextType;
        task = Task::taskForType(type, m_db);
        if (!task) {
            m_quit = true;
            break;
        }
    }

    LOG_WARN("workflow runloop terminating.");
}

}  // namespace vcsmc

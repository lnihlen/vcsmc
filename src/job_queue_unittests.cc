#include "job_queue.h"

#include <atomic>
#include <memory>

#include "gtest/gtest.h"
#include "job.h"

namespace vcsmc {

class SleepJob: public Job {
 public:
  explicit SleepJob(size_t sleep_ms,
                    std::shared_ptr<std::atomic_size_t> active_workers)
      : sleep_ms_(sleep_ms), active_workers_(active_workers) {}
  virtual ~SleepJob() {}
  void Execute() override {
    std::this_thread::sleep_for(
        std::chrono::duration<size_t, std::milli>(sleep_ms_));
    --(*active_workers_);
  }
 private:
  size_t sleep_ms_;
  std::shared_ptr<std::atomic_size_t> active_workers_;
};

TEST(JobQueueTest, EmptyQueueLifeCycle) {
  JobQueue jq(12);
  EXPECT_EQ(12U, jq.number_of_threads());
}

TEST(JobQueueTest, EmptyQueueFinishLifeCycle) {
  JobQueue jq(16);
  jq.Finish();
}

TEST(JobQueueTest, ZeroThreadsPicksSaneDefault) {
  JobQueue jq(0);
  EXPECT_LT(0U, jq.number_of_threads());
}

TEST(JobQueueTest, QueueJobsOfDifferentLengths) {
  JobQueue jq(16);
  std::shared_ptr<std::atomic_size_t> workers(new std::atomic_size_t);
  workers->store(50u);
  jq.LockQueue();
  for (size_t i = 0; i < 50u; ++i) {
    jq.EnqueueLocked(std::unique_ptr<SleepJob>(new SleepJob(i + 1, workers)));
  }
  jq.UnlockQueue();
  jq.Finish();
  size_t workers_now = workers->load();
  EXPECT_EQ(0u, workers_now);
}

TEST(JobQueueTest, QueueJobsSingleWorker) {
  JobQueue jq(1);
  std::shared_ptr<std::atomic_size_t> workers(new std::atomic_size_t);
  workers->store(10u);
  jq.LockQueue();
  for (size_t i = 0; i < 10u; ++i) {
    jq.EnqueueLocked(std::unique_ptr<SleepJob>(new SleepJob(i + 1, workers)));
  }
  jq.UnlockQueue();
  jq.Finish();
  size_t workers_now = workers->load();
  EXPECT_EQ(0u, workers_now);
}


}  // namespace vcsmc

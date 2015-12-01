#include "job_queue.h"
#include "gtest/gtest.h"

namespace vcsmc {

class SleepJob: public Job {
 public:
  explicit SleepJob(size_t sleep_ms) : sleep_ms_(sleep_ms) {}
  virtual ~SleepJob() {}
  void Execute() override {
    std::this_thread::sleep_for(
        std::chrono::duration<size_t, std::milli>(sleep_ms_));
  }
 private:
  size_t sleep_ms_;
};

TEST(JobQueueTest, EmptyQueueLifeCycle) {
  JobQueue jq(12);
  EXPECT_EQ(12U, jq.number_of_threads());
}

TEST(JobQueueTest, ZeroThreadsPicksSaneDefault) {
  JobQueue jq(0);
  EXPECT_LT(0U, jq.number_of_threads());
}

TEST(JobQueueTest, QueueJobsOfDifferentLengths) {
  JobQueue jq(16);
  for (size_t i = 0; i < 100U; ++i) {
    jq.Enqueue(std::unique_ptr<SleepJob>(new SleepJob(i + 1)));
  }
  jq.Finish();
}

}  // namespace vcsmc

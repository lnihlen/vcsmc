#ifndef SRC_JOB_QUEUE_H_
#define SRC_JOB_QUEUE_H_

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace vcsmc {

class Job;

// A JobQueue accepts a series of Job-derived classes, assigning them in
// parallel to each of the worker threads that it creates.
class JobQueue {
 public:
  // Constructs a JobQueue with the supplied number of worker threads. If
  // |number_of_threads| is 0 JobQueue will try to guess a reasonable value
  // based on number of hardware threads.
  explicit JobQueue(size_t number_of_threads);
  ~JobQueue();

  // For shorter tasks, with only one thread queuing the work, it's quite
  // possible for throughput to suffer as the single queuing thread cannot
  // create tasks fast enough for all of the worker threads to stay busy.
  // Therefore we provide methods to lock the queue, enqueue all of the pending
  // jobs with EnqueueLocked(), and not wake up any sleeping workers until
  // UnlockQueue() is called.
  void LockQueue();
  void EnqueueLocked(std::unique_ptr<Job> job);
  void UnlockQueue();

  // Blocks calling thread until work queue is empty.
  void Finish();

  size_t number_of_threads() const { return threads_.size(); }

 private:
  void ThreadWorkLoop();

  std::vector<std::thread> threads_;

  std::mutex queue_mutex_;
  std::unique_lock<std::mutex> queue_lock_;
  std::condition_variable queue_cv_;
  std::condition_variable queue_empty_cv_;
  bool exit_;
  std::queue<std::unique_ptr<Job>> queue_;

  std::mutex active_mutex_;
  std::condition_variable active_cv_;
  size_t active_workers_;
};

}  // namespace vcsmc

#endif  // SRC_JOB_QUEUE_H_

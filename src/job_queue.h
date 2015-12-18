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

  void Enqueue(std::unique_ptr<Job> job);
  // Blocks calling thread until work queue is empty.
  void Finish();

  size_t number_of_threads() const { return threads_.size(); }

 private:
  void ThreadWorkLoop();

  std::vector<std::thread> threads_;

  std::mutex mutex_;
  std::condition_variable cv_;
  bool exit_;
  std::queue<std::unique_ptr<Job>> queue_;
  std::condition_variable queue_size_cv_;

  std::mutex active_mutex_;
  std::condition_variable active_cv_;
  size_t active_workers_;
};

}  // namespace vcsmc

#endif  // SRC_JOB_QUEUE_H_

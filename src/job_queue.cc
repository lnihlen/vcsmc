#include "job_queue.h"
#include "job.h"

namespace vcsmc {

JobQueue::JobQueue(size_t number_of_threads)
  : exit_(false),
    active_workers_(0) {
  if (number_of_threads == 0) {
    number_of_threads = std::thread::hardware_concurrency() ?
      std::thread::hardware_concurrency() : 4;
  }
  threads_.reserve(number_of_threads);
  for (size_t i = 0; i < number_of_threads; ++i) {
    threads_.emplace_back(std::thread([this] { ThreadWorkLoop(); }));
  }
}

JobQueue::~JobQueue() {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    exit_ = true;
  }
  cv_.notify_all();
  for (size_t i = 0; i < threads_.size(); ++i) {
    threads_[i].join();
  }
}

void JobQueue::Enqueue(std::unique_ptr<Job> job) {
  std::unique_lock<std::mutex> lock(mutex_);
  queue_.emplace(std::move(job));
  cv_.notify_one();
}

void JobQueue::Finish() {
  {
    // First wait for the queue to empty out.
    std::unique_lock<std::mutex> lock(mutex_);
    queue_size_cv_.wait(lock, [this] { return queue_.size() == 0; });
  }
  {
    // Now wait for the number of active threads to be zero.
    std::unique_lock<std::mutex> lock(active_mutex_);
    active_cv_.wait(lock, [this] { return active_workers_ == 0; });
  }
}

void JobQueue::ThreadWorkLoop() {
  std::unique_ptr<Job> job;
  while (true) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (exit_) return;
      cv_.wait(lock, [this] { return (queue_.size() > 0 || exit_); });
      if (exit_) return;
      job = std::move(queue_.front());
      queue_.pop();
    }

    {
      std::unique_lock<std::mutex> lock(active_mutex_);
      active_workers_++;
    }

    queue_size_cv_.notify_one();

    job->Execute();

    {
      std::unique_lock<std::mutex> lock(active_mutex_);
      active_workers_--;
    }

    active_cv_.notify_one();
  }
}

}  // namespace vcsmc

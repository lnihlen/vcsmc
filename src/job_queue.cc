#include "job_queue.h"
#include "job.h"

#include <cassert>

namespace vcsmc {

JobQueue::JobQueue(size_t number_of_threads)
  : queue_mutex_(),
    queue_lock_(queue_mutex_, std::defer_lock),
    exit_(false),
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
  queue_lock_.lock();
  exit_ = true;
  queue_lock_.unlock();

  queue_cv_.notify_all();
  queue_empty_cv_.notify_all();

  for (size_t i = 0; i < threads_.size(); ++i) {
    threads_[i].join();
  }
}

void JobQueue::LockQueue() {
  assert(!queue_lock_.owns_lock());
  queue_lock_.lock();
}

void JobQueue::EnqueueLocked(std::unique_ptr<Job> job) {
  assert(queue_lock_.owns_lock());
  queue_.push(std::move(job));
}

void JobQueue::UnlockQueue() {
  assert(queue_lock_.owns_lock());
  queue_lock_.unlock();
  queue_cv_.notify_all();
}

void JobQueue::Finish() {
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_empty_cv_.wait(lock, [this] {
        return (queue_.size() == 0 || exit_);
      });
    if (exit_) return;
  }
  {
    // Wait for the number of active threads to be zero. Threads won't stop
    // working until the queue is empty.
    std::unique_lock<std::mutex> lock(active_mutex_);
    active_cv_.wait(lock, [this] { return active_workers_ == 0; });
  }
}

void JobQueue::ThreadWorkLoop() {
  std::unique_ptr<Job> job;
  while (true) {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      if (exit_) return;
      queue_cv_.wait(lock, [this] { return (queue_.size() > 0 || exit_); });
      if (exit_) return;
      job = std::move(queue_.front());
      queue_.pop();
    }

    {
      std::lock_guard<std::mutex> lock(active_mutex_);
      active_workers_++;
    }

    while (job) {
      job->Execute();
      {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (exit_) return;
        if (queue_.size() > 0) {
          job = std::move(queue_.front());
          queue_.pop();
        } else {
          job.reset(nullptr);
        }
      }
    }

    queue_empty_cv_.notify_one();

    {
      std::lock_guard<std::mutex> lock(active_mutex_);
      active_workers_--;
      if (active_workers_ == 0) {
        active_cv_.notify_one();
      }
    }
  }
}

}  // namespace vcsmc

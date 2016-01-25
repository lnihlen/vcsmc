#ifndef SRC_SOUND_H_
#define SRC_SOUND_H_

#include <memory>

#include "types.h"

namespace vcsmc {

// Signed mono uint32 little-endian audio file at kAudioSampleRate.
class Sound {
 public:
   Sound(std::unique_ptr<uint32[]> samples, size_t number_of_samples)
      : samples_(std::move(samples)),
        number_of_samples_(number_of_samples) {}

   const uint32* samples() const { return samples_.get(); }
   uint32* samples_writeable() { return samples_.get(); }
   size_t number_of_samples() const { return number_of_samples_; }

 private:
  std::unique_ptr<uint32[]> samples_;
  size_t number_of_samples_;
};


}  // namespace vcsmc

#endif  // SRC_SOUND_H_

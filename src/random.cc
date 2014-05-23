#include "random.h"

#include <fcntl.h>
#include <cassert>
#include <cstring>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace vcsmc {

Random::Random(const uint32* seed)
    : index_(0) {
  memcpy(state_, seed, sizeof(state_));
}

Random::Random() : index_(0) {
  int urand_fd = open("/dev/urandom", O_RDONLY);
  assert(urand_fd >= 0);
  int read_size = read(urand_fd, state_, sizeof(state_));
  assert(read_size == sizeof(state_));
  close(urand_fd);
}

// See http://www.lomont.org/Math/Papers/2008/Lomont_PRNG_2008.pdf
//
uint32 Random::next() {
  uint32 a, b, c, d;
  a = state_[index_];
  c = state_[(index_ + 13) & 15];
  b = a ^ c ^ (a << 16) ^ (c << 15);
  c = state_[(index_ + 9) & 15];
  c ^= (c >> 11);
  a = state_[index_] = b ^ c;
  d = a ^ ((a << 5) & 0xda442d24);
  index_ = (index_ + 15) & 15;
  a = state_[index_];
  state_[index_] = a ^ b ^ d ^ (a << 2) ^ (b << 18) ^ (c << 28);
  return state_[index_];
}

}  // namespace vcsmc

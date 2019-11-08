#include "tls_prng.h"

#include <chrono>

namespace vcsmc {

TlsPrng::TlsPrng() {
  seed();
}

TlsPrng::~TlsPrng() {
}

void TlsPrng::seed(uint32 val) {
  (void)val;
  std::random_device urandom;
  uint32 hash = urandom();
  engine_.seed(hash);
}

}  // namespace vcsmc

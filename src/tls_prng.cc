#include "tls_prng.h"

#include <chrono>

#include "tbb/tbb_thread.h"

namespace vcsmc {

TlsPrng::TlsPrng() {
  tbb::tbb_hash<tbb::tbb_thread::id> hasher;
  uint32 hash = hasher(tbb::this_tbb_thread::get_id());
  hash += std::chrono::system_clock::now().time_since_epoch().count();
  engine_.seed(hash);
}

TlsPrng::~TlsPrng() {
}

}  // namespace vcsmc

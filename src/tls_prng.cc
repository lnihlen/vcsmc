#include "tls_prng.h"

#include "tbb/tbb_thread.h"

namespace vcsmc {

TlsPrng::TlsPrng() {
  tbb::tbb_hash<tbb::tbb_thread::id> hasher;
  uint32 hash = hasher(tbb::this_tbb_thread::get_id());
  engine_.seed(hash);
}

TlsPrng::~TlsPrng() {
}

}  // namespace vcsmc

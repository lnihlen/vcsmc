#include "tls_prng.h"

#include <cassert>
#include <chrono>

#include "tbb/tbb_thread.h"

extern "C" {
#include "dc.h"
}

namespace vcsmc {

struct TlsPrng::PImpl {
  mt_struct* mts;
};

TlsPrng::TlsPrng() {
  pimpl_ = new PImpl();
  tbb::tbb_hash<tbb::tbb_thread::id> hasher;
  // Hash must be < 65536.
  uint32 hash = hasher(tbb::this_tbb_thread::get_id()) & 0x0000ffff;
  pimpl_->mts = get_mt_parameter_id_st(32, 521, hash, 0xdeadbeef);
  assert(pimpl_->mts);
  uint32 seed_val = std::chrono::system_clock::now().time_since_epoch().count();
  seed(seed_val);
}

TlsPrng::~TlsPrng() {
  free_mt_struct(pimpl_->mts);
  delete pimpl_;
}

void TlsPrng::seed(uint32 val) {
  sgenrand_mt(val, pimpl_->mts);
}

uint32 TlsPrng::operator()() {
  return genrand_mt(pimpl_->mts);
}

}  // namespace vcsmc

#ifndef SRC_TLS_PRNG_H_
#define SRC_TLS_PRNG_H_

#include "tbb/tbb.h"

#include "types.h"

namespace vcsmc {

// Uses the Dynamic Creator library to lazily create thread-specific independent
// Mersenne Twisters for use as nonblocking PRNG.
//
// Question - can this be used as a std::default_random_engine sort of API?
class TlsPrng {
 public:
  TlsPrng();
  ~TlsPrng();

  typedef uint32 result_type;
  static constexpr uint32 min() { return 0; }
  static constexpr uint32 max() { return std::numeric_limits<uint32>::max(); }
  void seed(uint32 val = 0);
  uint32 operator()();
  void discard() { operator()(); }

 private:
  struct PImpl;
  PImpl* pimpl_;
};

typedef tbb::enumerable_thread_specific<TlsPrng> TlsPrngList;

}  // namespace vcsmc

#endif  // SRC_TLS_PRNG_H_

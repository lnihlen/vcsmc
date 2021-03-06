#ifndef SRC_TLS_PRNG_H_
#define SRC_TLS_PRNG_H_

#include <random>
#include <vector>

#include "types.h"

namespace vcsmc {

class TlsPrng {
 public:
  TlsPrng();
  ~TlsPrng();
  TlsPrng(const TlsPrng&) = delete;

  typedef std::default_random_engine::result_type result_type;
  static constexpr uint32 min() { return std::default_random_engine::min(); }
  static constexpr uint32 max() { return std::default_random_engine::max(); }
  void seed(uint32 val = 0);
  uint32 operator()() { return engine_(); }
  void discard(unsigned long long z) { (void)z; }

 private:
  std::default_random_engine engine_;
};

// TODO: swap with OpenMP TLS threadprivate directive.
typedef std::vector<TlsPrng> TlsPrngList;

}  // namespace vcsmc

#endif  // SRC_TLS_PRNG_H_

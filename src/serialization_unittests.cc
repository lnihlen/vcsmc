#include "gtest/gtest.h"

#include <cstring>
#include <random>
#include <string>

#include "serialization.h"
#include "spec.h"
#include "tls_prng.h"

namespace vcsmc {

TEST(SerializationTest, Base64EncodeDecode) {
  uint8 buf[128];
  std::string seed_str = "base64 test seed";
  std::seed_seq seed(seed_str.begin(), seed_str.end());
  std::default_random_engine engine(seed);
  std::uniform_int_distribution<uint8> distro(0, 255);
  for (size_t i = 0; i < 128; ++i) {
    buf[i] = distro(engine);
  }
  for (size_t i = 0; i < 127; ++i) {
    size_t len = 128 - i;
    std::string encode = vcsmc::Base64Encode(buf + i, len, 0);
    std::unique_ptr<uint8[]> decode = vcsmc::Base64Decode(encode, len);
    EXPECT_EQ(0, std::memcmp(decode.get(), buf + i, len));
    encode = vcsmc::Base64Encode(buf, len, 0);
    decode = vcsmc::Base64Decode(encode, len);
    EXPECT_EQ(0, std::memcmp(decode.get(), buf, len));
  }
}

}  // namespace vcsmc

#include "bit_map.h"
#include "gtest/gtest.h"

namespace vcsmc {

TEST(BitMapTests, SetBitNoExtraByte) {
  BitMap bitmap(128, 16);
  EXPECT_EQ(16U, bitmap.bytes_per_row());

  for (uint32 i = 0; i < 16; ++i) {
    for (uint32 j = 0; j < 16; ++j) {
      uint8 byte_value = (i * 16) + j;
      for (uint32 k = 0; k < 8; ++k) {
        bitmap.SetBit((j * 8) + k, i, byte_value & (1 << k));
      }
    }
  }

  for (uint32 i = 0; i < 256; ++i) {
    EXPECT_EQ(i, *(bitmap.packed_bytes() + i));
  }
}

TEST(BitMapTests, SetBitExtraByte) {
  BitMap bitmap(11, 13);
  EXPECT_EQ(2U, bitmap.bytes_per_row());

  for (uint32 i = 0; i < 13; ++i) {
    bitmap.SetBit(8, i, i & 1);
    bitmap.SetBit(9, i, i & 2);
    bitmap.SetBit(10, i, i & 4);
  }

  for (uint32 i = 0; i < 13; ++i) {
    EXPECT_EQ(i & 7, *(bitmap.packed_bytes() + (2U * i) + 1U) & 7U);
  }
}

TEST(BitMapTests, BitNoExtraByte) {
  std::unique_ptr<uint8[]> packed(new uint8[256]);
  for (uint32 i = 0; i < 256; ++i)
    packed[i] = i;
  BitMap bitmap(128, 16, std::move(packed), 16);

  for (uint32 i = 0; i < 16; ++i) {
    for (uint32 j = 0; j < 16; ++j) {
      uint8 byte_value = (i * 16) + j;
      for (uint32 k = 0; k < 8; ++k) {
        EXPECT_EQ((byte_value & (1 << k)) != 0, bitmap.bit((j * 8) + k, i));
      }
    }
  }
}

}  // namespace vcsmc

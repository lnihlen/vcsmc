#include "bit_map.h"

#include <cassert>

namespace vcsmc {

BitMap::BitMap(uint32 width, uint32 height)
    : ValueMap(width, height),
      bytes_per_row_(width_ / 8) {
  if (width_ % 8)
    ++bytes_per_row_;
  packed_bytes_.reset(new uint8[bytes_per_row_ * height_]);
}

BitMap::BitMap(uint32 width, uint32 height, std::unique_ptr<uint8[]> bytes,
    uint32 bytes_per_row) : ValueMap(width, height),
                            bytes_per_row_(bytes_per_row),
                            packed_bytes_(std::move(bytes)) {
}

BitMap::~BitMap() {
}

// static
std::unique_ptr<BitMap> BitMap::Load(const std::string& file_path) {
  // Load png image bytes, they could be packed single bits or 8-bit unpacked.
  uint32 width, height, bit_depth, bytes_per_row;
  std::unique_ptr<uint8[]> image_bytes(ValueMap::LoadFromFile(file_path,
      width, height, bit_depth, bytes_per_row));
  if (!image_bytes || bit_depth > 8)
    return nullptr;
  // If we loaded bytes pack them into bits before returning, otherwise we can
  // construct the BitMap directly from the image bytes.
  if (bit_depth == 8) {
    std::unique_ptr<BitMap> bitmap(new BitMap(width, height));
    bitmap->Pack(image_bytes.get(), bytes_per_row);
    return std::move(bitmap);
  } else {
    assert(bit_depth == 1);
    return std::unique_ptr<BitMap>(new BitMap(width, height,
        std::move(image_bytes), bytes_per_row));
  }
  return nullptr;
}

void BitMap::Save(const std::string& file_path) {
  std::unique_ptr<uint8[]> unpacked(new uint8[width_ * height_]);
  std::memset(unpacked.get(), 0, width_ * height_);
  uint8* byte_ptr = unpacked.get();
  for (uint32 i = 0; i < height_; ++i) {
    for(uint32 j = 0; j < width_; ++j) {
      if (bit(j, i))
        *byte_ptr = 0xff;
      ++byte_ptr;
    }
  }

  ValueMap::SaveFromBytes(file_path, unpacked.get(), width_, height_, 8,
      width_);
}

void BitMap::Pack(const uint8* bytes, uint32 bytes_per_row_unpacked) {
  assert(bytes_per_row_unpacked >= width_);
  // It is assumed that most bits are zero when packing, as that is what most
  // spectral maps bits are.
  std::memset(packed_bytes_.get(), 0, bytes_per_row_ * height_);
  for (uint32 i = 0; i < height_; ++i) {
    const uint8* row_ptr = bytes + (i * bytes_per_row_unpacked);
    for (uint32 j = 0; j < width_; ++j) {
      if (*row_ptr)
        SetBit(j, i, true);
      ++row_ptr;
    }
  }
}

void BitMap::SetBit(uint32 x, uint32 y, bool value) {
  uint8* byte_ptr = packed_bytes_.get() + (y * bytes_per_row_) + (x / 8);
  uint32 bit_offset = x % 8;
  if (value)
    *byte_ptr = (*byte_ptr) | (1 << bit_offset);
  else
    *byte_ptr = (*byte_ptr) & (~(1 << bit_offset));
}

bool BitMap::bit(uint32 x, uint32 y) const {
  uint8* byte_ptr = packed_bytes_.get() + (y * bytes_per_row_) + (x / 8);
  uint32 bit_offset = x % 8;
  return (*byte_ptr) & (1 << bit_offset);
}

}  // namespace vcsmc

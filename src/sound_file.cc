#include "sound_file.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "constants.h"
#include "sound.h"
#include "types.h"

namespace {

uint32 PackUInt32(const uint8* bytes) {
  return (static_cast<uint32>(*bytes))             |
         (static_cast<uint32>(*(bytes + 1)) << 8)  |
         (static_cast<uint32>(*(bytes + 2)) << 16) |
         (static_cast<uint32>(*(bytes + 3)) << 24);
}

void UnpackUInt32(uint32 word, uint8* bytes) {
  *(bytes) = static_cast<uint8>(word & 0x000000ff);
  *(bytes + 1) = static_cast<uint8>((word >> 8) & 0x000000ff);
  *(bytes + 2) = static_cast<uint8>((word >> 16) & 0x000000ff);
  *(bytes + 3) = static_cast<uint8>((word >> 24) & 0x000000ff);
}

uint16 PackUInt16(const uint8* bytes) {
  return (static_cast<uint16>(*bytes)) |
         (static_cast<uint16>(*(bytes + 1)) << 8);
}

void UnpackUInt16(uint16 word, uint8* bytes) {
  *(bytes) = static_cast<uint8>(word & 0x00ff);
  *(bytes + 1) = static_cast<uint8>((word >> 8) & 0x00ff);
}

std::unique_ptr<vcsmc::Sound> LoadWAV(const std::string& file_name) {
  int fd = open(file_name.c_str(), O_RDONLY);
  if (fd < 0) return nullptr;

  // Read and verify RIFF header.
  uint8 riff_header[48];
  size_t bytes_read = read(fd, &riff_header, 12);
  // First word should read "RIFF", last four should read 'WAVE'.
  if (bytes_read != 12) {
    close(fd);
    return  nullptr;
  }
  uint32 chunk_id = PackUInt32(riff_header);
  uint32 chunk_size = PackUInt32(riff_header + 4);
  uint32 riff_id = PackUInt32(riff_header + 8);
  if (chunk_id != 0x46464952 /* RIFF */ ||
      chunk_size < 4                    ||
      riff_id != 0x45564157 /* WAVE */) {
    close(fd);
    return nullptr;
  }

  // Read and verify "fmt " header.
  bytes_read = read(fd, &riff_header, 24);
  if (bytes_read != 24) {
    close(fd);
    return nullptr;
  }
  chunk_id = PackUInt32(riff_header);
  chunk_size = PackUInt32(riff_header + 4);
  uint16 format_code = PackUInt16(riff_header + 8);
  if (chunk_id != 0x20746d66 ||
      (chunk_size != 16 && chunk_size != 18 && chunk_size != 40) ||
      format_code != 0xfffe) {
    close(fd);
    return nullptr;
  }
  uint16 number_of_channels = PackUInt16(riff_header + 10);
  uint32 sample_rate = PackUInt16(riff_header + 12);
  uint16 bits_per_sample = PackUInt16(riff_header + 22);
  // Mono 31440 Hz signed pcm 32-bit data only please.
  if (number_of_channels != 1                ||
      sample_rate != vcsmc::kAudioSampleRate ||
      bits_per_sample != 32) {
    close(fd);
    return nullptr;
  }
  // Advance file ptr to next chunk.
  if (chunk_size == 18) {
    read(fd, &riff_header, 2);
  } else if (chunk_size == 40) {
    read(fd, &riff_header, 24);
  }

  // There may be a "fact" chunk, get chunk ID and size.
  bytes_read = read(fd, &riff_header, 8);
  if (bytes_read != 8) {
    close(fd);
    return nullptr;
  }
  chunk_id = PackUInt32(riff_header);
  chunk_size = PackUInt32(riff_header + 4);
  if (chunk_id == 0x74636166) {
    if (chunk_size > 40) {
      close(fd);
      return nullptr;
    }
    // Read remaining chunk and chunk header for next chunk.
    bytes_read = read(fd, &riff_header, chunk_size + 8);
    if (bytes_read != chunk_size + 8) {
      close(fd);
      return nullptr;
    }
    chunk_id = PackUInt32(riff_header + chunk_size);
    chunk_size = PackUInt32(riff_header + chunk_size + 4);
  }

  // Handle "data" chunk.
  if (chunk_id != 0x61746164) {
    close(fd);
    return nullptr;
  }
  // The file pointer should now point just past "data" chunk header.
  uint32 sample_count = chunk_size / 4;
  std::unique_ptr<uint32[]> sample_data(new uint32[sample_count]);
  bytes_read = read(fd, sample_data.get(), chunk_size);
  if (bytes_read != chunk_size) {
    close(fd);
    return nullptr;
  }

  close(fd);
  return std::unique_ptr<vcsmc::Sound>(
      new vcsmc::Sound(std::move(sample_data), sample_count));
}

bool SaveWAV(const vcsmc::Sound* sound, const std::string& file_name) {
  int fd = open(file_name.c_str(),
      O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
  if (fd < 0) return false;
  uint8 riff_header[80];
  riff_header[0] = 'R';  // "RIFF".
  riff_header[1] = 'I';
  riff_header[2] = 'F';
  riff_header[3] = 'F';
  uint32 chunk_size = 72 + (sound->number_of_samples() * 4);
  UnpackUInt32(chunk_size, riff_header + 4);  // "RIFF" chunk size.
  riff_header[ 8] = 'W';  // "WAVE".
  riff_header[ 9] = 'A';
  riff_header[10] = 'V';
  riff_header[11] = 'E';
  riff_header[12] = 'f';  // "fmt ".
  riff_header[13] = 'm';
  riff_header[14] = 't';
  riff_header[15] = ' ';
  UnpackUInt32(40, riff_header + 16);  // "fmt " chunk size.
  UnpackUInt16(0xfffe, riff_header + 20);  // PCM extensible wave format.
  UnpackUInt16(1, riff_header + 22);  // 1 channel.
  UnpackUInt32(vcsmc::kAudioSampleRate, riff_header + 24);  // Sample rate.
  UnpackUInt32(vcsmc::kAudioSampleRate * 4, riff_header + 28);  // Data rate.
  UnpackUInt16(4, riff_header + 32);  // Block size in bytes.
  UnpackUInt16(32, riff_header + 34);  // 32 bits per sample.
  UnpackUInt16(22, riff_header + 36);  // 22 bytes in the extension.
  UnpackUInt16(32, riff_header + 38);  // 32 valid bits per sample.
  UnpackUInt32(0x00000004, riff_header + 40);  // Speaker position mask.
  UnpackUInt32(0x00000001, riff_header + 44);  // GUID word 1.
  UnpackUInt32(0x00100000, riff_header + 48);  // GUID word 2.
  UnpackUInt32(0xaa000080, riff_header + 52);  // GUID word 3.
  UnpackUInt32(0x719b3800, riff_header + 56);  // GUID word 4.
  riff_header[60] = 'f';  // "fact"
  riff_header[61] = 'a';
  riff_header[62] = 'c';
  riff_header[63] = 't';
  UnpackUInt32(4, riff_header + 64);  // "fact" chunk size.
  UnpackUInt32(sound->number_of_samples(), riff_header + 68);  // Sample count.
  riff_header[72] = 'd';  // "data"
  riff_header[73] = 'a';
  riff_header[74] = 't';
  riff_header[75] = 'a';
  UnpackUInt32(sound->number_of_samples() * 4, riff_header + 76);  // Data size.
  // Write header.
  write(fd, riff_header, 80);
  // Write data.
  write(fd, sound->samples(), sound->number_of_samples() * 4);
  close(fd);
  return true;
}

}

namespace vcsmc {

std::unique_ptr<Sound> LoadSound(const std::string& file_name) {
  size_t ext_pos = file_name.find_last_of(".");
  if (ext_pos == std::string::npos)
    return nullptr;

  std::string ext = file_name.substr(ext_pos);
  if (ext == ".wav")
    return LoadWAV(file_name);

  return nullptr;
}

bool SaveSound(const Sound* sound, const std::string& file_name) {
  size_t ext_pos = file_name.find_last_of(".");
  if (ext_pos == std::string::npos)
    return false;

  std::string ext = file_name.substr(ext_pos);
  if (ext == ".wav")
    return SaveWAV(sound, file_name);

  return false;
}

}  // namespace vcsmc

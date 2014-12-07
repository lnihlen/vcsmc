// sim - VCS output simulator.

#include <cassert>
#include <cstdio>
#include <fcntl.h>
#include <memory>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#include "cl_image.h"
#include "constants.h"
#include "image.h"
#include "image_file.h"
#include "range.h"
#include "spec.h"
#include "state.h"

bool CompareSpecsStartTime(const vcsmc::Spec& s1, const vcsmc::Spec& s2) {
  return (s1.range().start_time() < s2.range().start_time());
}

bool SimFileIdeal(const std::string& input_file_spec,
                  const std::string& output_file_spec,
                  const uint64& frame_number) {
  // Attempt to open input file.
  const uint32 kMaxFilenameLength = 2048;
  std::unique_ptr<char[]> file_name_buffer(new char[kMaxFilenameLength]);
  snprintf(file_name_buffer.get(), kMaxFilenameLength,
      input_file_spec.c_str(), frame_number);
  int input_fd = open(file_name_buffer.get(), O_RDONLY);
  if (input_fd < 0) {
    fprintf(stderr, "error opening spec input file %s\n",
        file_name_buffer.get());
    return false;
  }

  // Deserialize specs into a vector for sorting and generation of state.
  const uint32 kSpecBufferSize = 1024 * 10;
  std::unique_ptr<uint8[]> spec_buffer(new uint8[kSpecBufferSize]);
  std::vector<vcsmc::Spec> specs;
  size_t bytes_read = read(input_fd, spec_buffer.get(), kSpecBufferSize);
  while (bytes_read > 0) {
    uint8* spec_ptr = spec_buffer.get();
    size_t bytes_deserialized = 0;
    size_t spec_size = 0;
    while (bytes_deserialized < bytes_read) {
      specs.push_back(vcsmc::Spec::Deserialize(spec_ptr, &spec_size));
      spec_ptr += spec_size;
      bytes_deserialized += spec_size;
    }
    bytes_read = read(input_fd, spec_buffer.get(), kSpecBufferSize);
  }

  spec_buffer.reset();
  close(input_fd);

  // An empty spec vector after deserialization indicates a problem, bail.
  if (!specs.size()) {
    fprintf(stderr, "error deserializing specs from %s\n",
        file_name_buffer.get());
    return false;
  }

  // Sort entire spec vector by Spec Range start time.
  std::sort(specs.begin(), specs.end(), CompareSpecsStartTime);

  // Build vector of State objects from Specs. We also check Spec end times,
  // looking for contradictory Specs, about which we can issue a warning.
  std::vector<std::unique_ptr<vcsmc::State>> states;
  states.reserve(specs.size() + 1);
  states.push_back(std::unique_ptr<vcsmc::State>(new vcsmc::State()));
  std::unique_ptr<uint32[]> end_times(new uint32[vcsmc::TIA::TIA_COUNT]);
  std::memset(end_times.get(), 0, sizeof(uint32) * vcsmc::TIA::TIA_COUNT);
  for (uint32 i = 0; i < specs.size(); ++i)
    states.push_back((*states.rbegin())->MakeIdealState(specs[i]));

  vcsmc::Image image(vcsmc::kFrameWidthPixels, vcsmc::kFrameHeightPixels);
  // Simulate through all |states| to generate simulated output image.
  for (uint32 i = 0; i < states.size(); ++i)
    states[i]->PaintInto(&image);

  // Save simulated output image to output file.
  snprintf(file_name_buffer.get(), kMaxFilenameLength,
      output_file_spec.c_str(), frame_number);
  return vcsmc::ImageFile::Save(&image, file_name_buffer.get());
}

int main(int argc, char* argv[]) {
  // Parse command line.
  if (argc != 3) {
    fprintf(stderr,
        "sim ideal mode usage:\n"
        "  sim <input_file_spec.spec> <output_file_spec.png>\n"
        "sim real mode usage: \n"
        "  sim <input_file_spec.bin> <output_file_spec.png>\n"
        "sim example:\n"
        "  sim specs/frame-%%05d.spec sim/frame-%%05d.png\n");
    return -1;
  }

  std::string input_file_spec(argv[1]);
  std::string output_file_spec(argv[2]);

  // We always assume ideal mode for now, real mode support is TODO.
  uint64 frame_number = 1;
  while (SimFileIdeal(input_file_spec, output_file_spec, frame_number))
    ++frame_number;

  return 0;
}
